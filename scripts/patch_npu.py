"""
NPU 910B support patch for Wan2.2

Applies runtime patches to enable Wan2.2 inference on Huawei Ascend NPU 910B
(32GB HBM + CPU offload). Tested with Wan2.2-S2V-14B on AtomGit NPU 910B 64GB.

Usage:
    python scripts/patch_npu.py [--repo-root /path/to/Wan2.2] [--max-memory 15GB]

Requirements:
    - torch_npu (CANN 8.5+, PyTorch 2.9 + torch_npu 2.9)
    - accelerate (for device_map='auto' staged loading)

Patches applied:
    1.  import torch_npu in generate.py and wan/__init__.py
    2.  torch.cuda.* -> torch.npu.* across all .py files
    3.  nccl -> hccl distributed backend
    4.  Staged loading: release T5 encoder before loading diffusion model
    5.  _configure_model(None) guard for lazy loading
    6.  Skip .to(device)/.cpu() for accelerate-managed models (_hf_hook check)
    7.  VAE dtype fix (encode input .float() cast)
    8.  dtype assert -> .float() cast in model_s2v.py
    9.  flash_attention -> SDPA fallback (NPU has no flash_attn)
    10. max_memory tuning + empty_cache before VAE decode

Author: 0xPabloLI
Tested: Wan2.2-S2V-14B, NPU 910B 64GB, 480x832, 5 steps, 550s inference
"""

import os
import re
import sys
import logging
import argparse


def patch_file(path, replacements, log_label=None):
    with open(path) as f:
        code = f.read()
    orig = code
    for old, new in replacements:
        code = code.replace(old, new)
    if code != orig:
        with open(path, "w") as f:
            f.write(code)
        if log_label:
            logging.info(f"  patched {log_label}")
        return True
    return False


def patch_cuda_to_npu(root):
    logging.info("=== Patch cuda -> npu ===")
    gen_path = os.path.join(root, "generate.py")
    with open(gen_path) as f:
        code = f.read()
    if "import torch_npu" not in code:
        code = code.replace("import torch\n", "import torch\nimport torch_npu\n")
    code = code.replace("torch.cuda.set_device", "torch.npu.set_device")
    code = code.replace("torch.cuda.synchronize()", "torch.npu.synchronize()")
    code = code.replace('backend="nccl"', 'backend="hccl"')
    code = code.replace("backend='nccl'", "backend='hccl'")
    with open(gen_path, "w") as f:
        f.write(code)

    for dirpath, dirs, files in os.walk(os.path.join(root, "wan")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            patch_file(path, [
                ("torch.cuda.", "torch.npu."),
                ('torch.amp.autocast("cuda"', 'torch.amp.autocast("npu"'),
                ("torch.amp.autocast('cuda'", "torch.amp.autocast('npu'"),
                ('device.type == "cuda"', 'device.type == "npu"'),
                ("device.type == 'cuda'", "device.type == 'npu'"),
                ('device="cuda"', 'device="npu"'),
                ("device='cuda'", "device='npu'"),
                ('f"cuda:', 'f"npu:'),
                ("f'cuda:", "f'npu:"),
                ('torch.device("cuda', 'torch.device("npu'),
                ("torch.device('cuda", "torch.device('npu"),
                ("import torch.cuda.amp as amp", "import torch.amp as amp"),
            ])

    init_path = os.path.join(root, "wan", "__init__.py")
    with open(init_path) as f:
        wcode = f.read()
    if "import torch_npu" not in wcode:
        with open(init_path, "w") as f:
            f.write("import torch_npu\n" + wcode)

    t5_path = os.path.join(root, "wan", "modules", "t5.py")
    if os.path.exists(t5_path):
        patch_file(t5_path, [("map_location='cpu'", "map_location='npu'")])
    logging.info("  cuda -> npu done")


def patch_staged_loading(root):
    logging.info("=== Patch staged loading ===")
    s2v_path = os.path.join(root, "wan", "speech2video.py")
    with open(s2v_path) as f:
        s2v = f.read()
    if "Staged loading" in s2v:
        logging.info("  already patched")
        return

    old_init = (
        'if not dit_fsdp:\n'
        '            self.noise_model = WanModel_S2V.from_pretrained(\n'
        '                checkpoint_dir,\n'
        '                torch_dtype=self.param_dtype,\n'
        '                device_map=self.device)\n'
        '        else:\n'
        '            self.noise_model = WanModel_S2V.from_pretrained(\n'
        '                checkpoint_dir, torch_dtype=self.param_dtype)'
    )
    new_init = (
        'self.noise_model = None\n'
        '        self._ckpt_dir = checkpoint_dir\n'
        '        self._dit_fsdp = dit_fsdp'
    )
    if old_init in s2v:
        s2v = s2v.replace(old_init, new_init)
    else:
        pattern = r'(if not dit_fsdp:\s+self\.noise_model = WanModel_S2V\.from_pretrained.*?else:\s+self\.noise_model = WanModel_S2V\.from_pretrained\([^)]+\))'
        match = re.search(pattern, s2v, re.DOTALL)
        if match:
            s2v = s2v[:match.start()] + new_init + s2v[match.end():]

    lazy = '''        # === Staged loading: release T5, load diffusion model ===
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            logging.info(f"Before T5 release: NPU allocated={torch.npu.memory_allocated()/1e9:.2f} GB")
            del self.text_encoder
            torch.npu.empty_cache()
            gc.collect()
            logging.info(f"After T5 release: NPU allocated={torch.npu.memory_allocated()/1e9:.2f} GB")
        if self.noise_model is None:
            logging.info("Loading diffusion model with offload...")
            from functools import partial as _partial
            from .modules.s2v.model_s2v import WanModel_S2V as _WanModel
            from .distributed.fsdp import shard_model as _shard
            if not self._dit_fsdp:
                self.noise_model = _WanModel.from_pretrained(
                    self._ckpt_dir, torch_dtype=self.param_dtype, device_map='auto')
            else:
                self.noise_model = _WanModel.from_pretrained(
                    self._ckpt_dir, torch_dtype=self.param_dtype)
            self.noise_model = self._configure_model(
                model=self.noise_model, use_sp=self.sp_size > 1,
                dit_fsdp=self._dit_fsdp, shard_fn=_partial(_shard, device_id=0),
                convert_model_dtype=False)
            logging.info("Diffusion model loaded")
        # === End staged loading ===

        out = []'''
    if "        out = []" in s2v:
        s2v = s2v.replace("        out = []", lazy, 1)
    with open(s2v_path, "w") as f:
        f.write(s2v)
    logging.info("  staged loading patched")


def patch_configure_model_guard(root):
    logging.info("=== Patch _configure_model guard ===")
    s2v_path = os.path.join(root, "wan", "speech2video.py")
    with open(s2v_path) as f:
        s2v = f.read()
    if "if model is None" in s2v:
        logging.info("  already patched")
        return
    old = "def _configure_model(self, model, use_sp=False, dit_fsdp=False, shard_fn=None, convert_model_dtype=True):"
    new = old + "\n        if model is None:\n            return None"
    if old in s2v:
        s2v = s2v.replace(old, new)
        with open(s2v_path, "w") as f:
            f.write(s2v)
        logging.info("  guard added")


def patch_skip_to_device_for_accelerate(root):
    logging.info("=== Patch skip .to(device)/.cpu() for accelerate models ===")
    s2v_path = os.path.join(root, "wan", "speech2video.py")
    with open(s2v_path) as f:
        s2v = f.read()
    if "_hf_hook" in s2v:
        logging.info("  already patched")
        return
    s2v = s2v.replace(
        "                if offload_model or self.init_on_cpu:\n                    self.noise_model.to(self.device)\n                    torch.npu.empty_cache()",
        "                if offload_model or self.init_on_cpu:\n                    if not hasattr(self.noise_model, '_hf_hook'):\n                        self.noise_model.to(self.device)\n                    torch.npu.empty_cache()"
    )
    s2v = s2v.replace(
        "                if offload_model:\n                    self.noise_model.cpu()\n                    torch.npu.synchronize()\n                    torch.npu.empty_cache()",
        "                if offload_model:\n                    if not hasattr(self.noise_model, '_hf_hook'):\n                        self.noise_model.cpu()\n                    torch.npu.synchronize()\n                    torch.npu.empty_cache()"
    )
    with open(s2v_path, "w") as f:
        f.write(s2v)
    logging.info("  patched")


def patch_vae_dtype(root):
    logging.info("=== Patch VAE dtype ===")
    vae_path = os.path.join(root, "wan", "modules", "vae2_1.py")
    if not os.path.exists(vae_path):
        logging.info("  vae2_1.py not found")
        return
    with open(vae_path) as f:
        vae = f.read()
    if ".float()" in vae and "encode" in vae:
        logging.info("  already patched")
        return
    old = "def encode(self, videos):"
    if old in vae:
        vae = vae.replace(old, "def encode(self, videos):\n        videos = videos.float()", 1)
        with open(vae_path, "w") as f:
            f.write(vae)
        logging.info("  patched")


def patch_dtype_assert(root):
    logging.info("=== Patch dtype assert -> .float() ===")
    path = os.path.join(root, "wan", "modules", "s2v", "model_s2v.py")
    with open(path) as f:
        ms = f.read()
    asserts = [
        "            assert e.dtype == torch.float32 and e0.dtype == torch.float32",
        "        assert e.dtype == torch.float32",
        "        assert e[0].dtype == torch.float32",
    ]
    replacements = [
        "            e = e.float()\n            e0 = e0.float()",
        "        e = e.float()",
        "        e = [t.float() for t in e]",
    ]
    count = 0
    for old, new in zip(asserts, replacements):
        if old in ms:
            ms = ms.replace(old, new)
            count += 1
    with open(path, "w") as f:
        f.write(ms)
    logging.info(f"  patched {count} assertions")


def patch_flash_attention_to_sdpa(root):
    logging.info("=== Patch flash_attention -> SDPA ===")
    path = os.path.join(root, "wan", "modules", "attention.py")
    if not os.path.exists(path):
        logging.info("  attention.py not found")
        return
    with open(path) as f:
        attn = f.read()
    attn = attn.replace(
        "if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:",
        "if False and FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:")
    with open(path, "w") as f:
        f.write(attn)
    logging.info("  flash_attention disabled -> SDPA fallback")


def patch_max_memory(root, max_memory="15GB"):
    logging.info(f"=== Patch max_memory={max_memory} + empty_cache ===")
    s2v_path = os.path.join(root, "wan", "speech2video.py")
    with open(s2v_path) as f:
        s2v = f.read()
    s2v = s2v.replace(
        "max_memory={0: '20GB', 'cpu': '60GB'}",
        f"max_memory={{0: '{max_memory}', 'cpu': '60GB'}}")
    if "empty_cache before VAE" not in s2v:
        s2v = s2v.replace(
            "                image = torch.stack(self.vae.decode(decode_latents))",
            "                torch.npu.empty_cache()\n                # empty_cache before VAE decode\n                image = torch.stack(self.vae.decode(decode_latents))")
    with open(s2v_path, "w") as f:
        f.write(s2v)
    logging.info("  patched")


def main():
    parser = argparse.ArgumentParser(description="Patch Wan2.2 for NPU 910B support")
    parser.add_argument("--repo-root", default=".", help="Path to Wan2.2 repo root")
    parser.add_argument("--max-memory", default="15GB", help="Max NPU HBM memory (default: 15GB)")
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.path.exists(os.path.join(root, "generate.py")):
        print(f"Error: {root} does not contain generate.py. Is this a Wan2.2 repo?")
        sys.exit(1)

    patch_cuda_to_npu(root)
    patch_staged_loading(root)
    patch_configure_model_guard(root)
    patch_skip_to_device_for_accelerate(root)
    patch_vae_dtype(root)
    patch_dtype_assert(root)
    patch_flash_attention_to_sdpa(root)
    patch_max_memory(root, args.max_memory)

    print("\nAll NPU patches applied successfully!")
    print(f"   Repo: {root}")
    print(f"   Max memory: {args.max_memory}")


if __name__ == "__main__":
    main()