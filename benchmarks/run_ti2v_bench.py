# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Real end-to-end TI2V-5B benchmark.

Drives the same WanTI2V pipeline generate.py drives, with the same defaults, and
records a machine-readable breakdown of where the time and memory actually go.
Phase timings are collected by wrapping the T5 encoder and the VAE on the
constructed pipeline, so nothing here changes production code paths.

Example
-------
python benchmarks/run_ti2v_bench.py \
    --ckpt_dir K:/wan/Wan2.2-TI2V-5B \
    --label bf16_baseline \
    --dit_quant none
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wan  # noqa: E402
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

GB = 1024**3


def nvidia_smi(query):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=' + query,
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
        return [float(x) for x in out.split(',')]
    except Exception:
        return None


class GpuSampler(threading.Thread):
    """
    Polls nvidia-smi so we can tell a busy GPU from a stalled one.

    NOTE the attribute is `_stop_evt`, not `_stop`: threading.Thread already
    defines a private `_stop()` method that join() calls, and shadowing it with
    an Event makes join() raise "'Event' object is not callable".
    """

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.util = []
        self.mem_used = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            vals = nvidia_smi('utilization.gpu,memory.used')
            if vals:
                self.util.append(vals[0])
                self.mem_used.append(vals[1])
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=10)

    def summary(self):
        if not self.util:
            return {}
        s = sorted(self.util)
        return {
            'gpu_util_mean_pct': round(sum(self.util) / len(self.util), 1),
            'gpu_util_median_pct': s[len(s) // 2],
            'gpu_util_max_pct': max(self.util),
            'gpu_mem_used_max_mib': max(self.mem_used) if self.mem_used else None,
            'samples': len(self.util),
        }


def host_ram_gb():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            'ram_total_gb': round(vm.total / GB, 2),
            'ram_used_gb': round(vm.used / GB, 2),
        }
    except Exception:
        return {}


def proc_ram_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / GB, 2)
    except Exception:
        return None


class Timed:
    """Wraps a callable and accumulates its wall time."""

    def __init__(self, fn):
        self.fn = fn
        self.total = 0.0
        self.calls = 0

    def __call__(self, *a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.fn(*a, **k)
        torch.cuda.synchronize()
        self.total += time.perf_counter() - t0
        self.calls += 1
        return out


def _generate_once(pipe, args, cfg, size, offload):
    return pipe.generate(
        input_prompt=args.prompt,
        size=size,
        max_area=size[0] * size[1],
        frame_num=args.frame_num,
        shift=cfg.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=args.base_seed,
        offload_model=offload,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--label', required=True,
                    help='name for this variant, used in the result filename')
    ap.add_argument('--prompt', default='a cat walking on a beach')
    ap.add_argument('--size', default='832*480')
    ap.add_argument('--frame_num', type=int, default=5)
    ap.add_argument('--sample_steps', type=int, default=4)
    ap.add_argument('--base_seed', type=int, default=42)
    ap.add_argument('--sample_solver', default='unipc')
    ap.add_argument('--dit_quant', default='none')
    ap.add_argument('--offload_model', default='True')
    ap.add_argument('--convert_model_dtype', action='store_true', default=True)
    ap.add_argument('--t5_cpu', action='store_true', default=True)
    ap.add_argument('--t5_fp32_compute', action='store_true', default=False)
    ap.add_argument('--t5_batch', action='store_true', default=False)
    ap.add_argument('--t5_cache', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=1,
                    help='generate N times; exposes cache warm-up effects')
    ap.add_argument('--vae_tile', action='store_true', default=False)
    ap.add_argument('--vae_tile_size', type=int, default=16)   # latent units
    ap.add_argument('--vae_tile_overlap', type=int, default=8)  # latent units
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo, 'benchmarks', 'results')
    os.makedirs(out_dir, exist_ok=True)

    cfg = WAN_CONFIGS['ti2v-5B']
    size = SIZE_CONFIGS[args.size]
    offload = args.offload_model.lower() in ('1', 'true', 'yes')

    result = {
        'label': args.label,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'task': 'ti2v-5B',
            'size': args.size,
            'frame_num': args.frame_num,
            'sample_steps': args.sample_steps,
            'sample_solver': args.sample_solver,
            'sample_shift': cfg.sample_shift,
            'guide_scale': cfg.sample_guide_scale,
            'base_seed': args.base_seed,
            'prompt': args.prompt,
            'dit_quant': args.dit_quant,
            'offload_model': offload,
            'convert_model_dtype': bool(args.convert_model_dtype),
            't5_cpu': bool(args.t5_cpu),
            't5_fp32_compute': bool(args.t5_fp32_compute),
            't5_batch': bool(args.t5_batch),
            't5_cache': args.t5_cache,
            'repeat': args.repeat,
            'vae_tile': bool(args.vae_tile),
            'vae_tile_size': args.vae_tile_size,
            'vae_tile_overlap': args.vae_tile_overlap,
        },
        'env': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'sm': '%d.%d' % torch.cuda.get_device_capability(),
            'vram_total_gb': round(
                torch.cuda.get_device_properties(0).total_memory / GB, 2),
            'platform': platform.platform(),
        },
    }

    free0, total0 = torch.cuda.mem_get_info()
    result['vram_free_before_load_gb'] = round(free0 / GB, 2)
    result.update(host_ram_gb())

    sampler = GpuSampler()
    t_wall0 = time.perf_counter()

    # ---- model load -------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=bool(args.t5_cpu),
        convert_model_dtype=bool(args.convert_model_dtype),
        dit_quant=args.dit_quant,
        t5_fp32_compute=args.t5_fp32_compute,
        t5_batch=args.t5_batch,
        t5_cache=args.t5_cache,
    )
    torch.cuda.synchronize()
    result['t_model_load_s'] = round(time.perf_counter() - t0, 2)

    if args.vae_tile:
        pipe.vae.enable_tiling(
            tile_size=args.vae_tile_size, overlap=args.vae_tile_overlap)

    if getattr(pipe, 'quant_stats', None):
        qs = dict(pipe.quant_stats)
        for k in ('bytes_before', 'bytes_after'):
            if k in qs:
                qs[k.replace('bytes', 'gb')] = round(qs.pop(k) / GB, 3)
        result['quant'] = qs

    dit_bytes = sum(p.numel() * p.element_size()
                    for p in pipe.model.parameters())
    dit_bytes += sum(b.numel() * b.element_size()
                     for b in pipe.model.buffers())
    result['dit_weight_gb'] = round(dit_bytes / GB, 3)

    free1, _ = torch.cuda.mem_get_info()
    result['vram_free_after_load_gb'] = round(free1 / GB, 2)
    result['vram_alloc_after_load_gb'] = round(
        torch.cuda.memory_allocated() / GB, 2)
    smi = nvidia_smi('memory.used')
    if smi:
        result['nvsmi_mem_used_after_load_gb'] = round(smi[0] / 1024, 2)
        # torch thinks it allocated X; the driver reports Y resident. A large
        # positive gap is weight data the driver pushed out to system RAM.
        result['estimated_spill_gb'] = round(
            max(0.0, torch.cuda.memory_allocated() / GB - smi[0] / 1024), 2)

    # ---- instrument phases ------------------------------------------------
    pipe.text_encoder = Timedish(pipe.text_encoder)
    vae_decode = Timed(pipe.vae.decode)
    pipe.vae.decode = vae_decode
    vae_encode = Timed(pipe.vae.encode)
    pipe.vae.encode = vae_encode

    # Time the DiT forwards themselves. Subtracting phases from the total would
    # otherwise fold the offload_model weight transfers (9.3 GB each way at
    # bf16) into "denoise" and badly overstate seconds/step.
    dit_fwd = Timed(pipe.model.forward)
    pipe.model.forward = dit_fwd

    # Sample VRAM at the point the DiT is actually resident and working.
    peak_probe = {'nvsmi_mib': 0}

    def probe():
        v = nvidia_smi('memory.used')
        if v:
            peak_probe['nvsmi_mib'] = max(peak_probe['nvsmi_mib'], v[0])

    orig_fwd = dit_fwd.fn

    def fwd_with_probe(*a, **k):
        out = orig_fwd(*a, **k)
        if dit_fwd.calls == 0:
            probe()
        return out

    dit_fwd.fn = fwd_with_probe

    # ---- generate ---------------------------------------------------------
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    per_run = []
    video = None
    for run_i in range(max(1, args.repeat)):
        t_run = time.perf_counter()
        t5_before = pipe.text_encoder.total
        dec_before = vae_decode.total
        dit_before = dit_fwd.total
        video = _generate_once(pipe, args, cfg, size, offload)
        torch.cuda.synchronize()
        per_run.append({
            'run': run_i,
            't_total_s': round(time.perf_counter() - t_run, 2),
            't_t5_s': round(pipe.text_encoder.total - t5_before, 2),
            't_vae_decode_s': round(vae_decode.total - dec_before, 2),
            't_dit_forward_s': round(dit_fwd.total - dit_before, 2),
        })
    result['per_run'] = per_run
    # Headline numbers describe the FIRST generation (a cold cache), which is
    # what a one-shot CLI run actually costs. per_run carries the rest.
    t_generate = per_run[0]['t_total_s']
    sampler.stop()

    t_t5 = per_run[0]['t_t5_s']
    t_vae_dec = per_run[0]['t_vae_decode_s']
    t_vae_enc = vae_encode.total / max(len(per_run), 1)
    t_dit = per_run[0]['t_dit_forward_s']
    # whatever is left is weight movement, scheduler steps and bookkeeping
    t_other = t_generate - t_t5 - t_vae_dec - t_vae_enc - t_dit

    result.update({
        't_generate_s': round(t_generate, 2),
        't_t5_encode_s': round(t_t5, 2),
        't_vae_encode_s': round(t_vae_enc, 2),
        't_vae_decode_s': round(t_vae_dec, 2),
        't_dit_forward_s': round(t_dit, 2),
        'dit_forward_calls': dit_fwd.calls // max(len(per_run), 1),
        't_transfer_and_other_s': round(t_other, 2),
        'sec_per_step': round(t_dit / args.sample_steps, 3),
        'sec_per_dit_forward': round(t_dit / max(dit_fwd.calls, 1), 3),
        'nvsmi_mem_used_during_denoise_gb': round(
            peak_probe['nvsmi_mib'] / 1024, 2) if peak_probe['nvsmi_mib'] else None,
        'peak_vram_alloc_gb': round(torch.cuda.max_memory_allocated() / GB, 2),
        'peak_vram_reserved_gb': round(
            torch.cuda.max_memory_reserved() / GB, 2),
        'proc_ram_gb': proc_ram_gb(),
    })
    result.update(sampler.summary())

    # ---- save -------------------------------------------------------------
    # Keep the raw tensor too. Quality comparisons between variants have to be
    # made on the pipeline output, not on an H.264 re-encode of it, or the
    # codec's own loss shows up in the numbers.
    npy = os.path.join(out_dir, 'ti2v_%s.npy' % args.label)
    import numpy as np
    np.save(npy, video.detach().float().cpu().numpy().astype('float16'))
    result['tensor_path'] = npy

    mp4 = os.path.join(out_dir, 'ti2v_%s.mp4' % args.label)
    t0 = time.perf_counter()
    # same call generate.py makes: save_video wants [B, C, F, H, W] and
    # unbinds dim 2 as the frame axis -- passing the bare [C, F, H, W]
    # tensor splits on height and writes a malformed file.
    save_video(
        tensor=video[None],
        save_file=mp4,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))
    result['t_video_write_s'] = round(time.perf_counter() - t0, 2)
    result['video_path'] = mp4
    result['video_shape'] = list(video.shape)
    result['t_total_wall_s'] = round(time.perf_counter() - t_wall0, 2)

    path = os.path.join(out_dir, 'ti2v_%s.json' % args.label)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print('\nwrote', path)


class Timedish:
    """Timed wrapper that keeps attribute access working (T5 has .model)."""

    def __init__(self, inner):
        self._inner = inner
        self.total = 0.0
        self.calls = 0

    def __call__(self, *a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self._inner(*a, **k)
        torch.cuda.synchronize()
        self.total += time.perf_counter() - t0
        self.calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


if __name__ == '__main__':
    main()
