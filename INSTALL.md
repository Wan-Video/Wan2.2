# Installation Guide

## Requirements

- Python 3.10 - 3.12
- An NVIDIA GPU with a recent driver (CUDA 12.x builds of PyTorch are used below)

## Install

Create a virtual environment, install PyTorch for your CUDA version first, then
the rest of the requirements:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Or install the package itself:

```bash
pip install .
pip install .[dev]       # linting, formatting and test tooling
```

### Optional pipelines

`WanS2V` (speech-to-video) and `WanAnimate` (character animation) pull in
additional dependencies and are imported lazily, so `import wan` works without
them. Install them only if you need those tasks:

```bash
pip install -r requirements_s2v.txt        # or: pip install .[s2v]
pip install -r requirements_animate.txt    # or: pip install .[animate]
```

### flash-attn is optional

`flash_attn` is **not** required. When it is absent, `wan/modules/attention.py`
falls back to PyTorch's `scaled_dot_product_attention`, which is slower and uses
more memory but produces equivalent results. You will see a one-time warning
saying so.

This matters on Windows, where flash-attn has no official wheel and the source
build is long and frequently fails.

To install it anyway:

```bash
pip install flash-attn --no-build-isolation
```

If that fails with a PEP 517 build error, make sure `pip`, `setuptools` and
`wheel` are up to date first, or install from git:

```bash
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

## Running the model

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage."
```

See the main [README](README.md) for per-task commands and VRAM guidance.

## Optional performance flags (TI2V-5B)

Measured on an RTX 4060 Laptop (8 GB, sm_89) at 832x480, 5 frames, 4 steps.
Full numbers in `benchmarks/results/`.

### `--vae_tile`

Decodes the VAE in overlapping spatial tiles instead of whole frames.

    --vae_tile --vae_tile_size 16 --vae_tile_overlap 8   # latent units

The decoder holds its peak activation for an entire HxW plane, which on an 8 GB
card overflows into system RAM. Tiling bounds that peak by the tile:

| | untiled | tiled |
|---|---:|---:|
| VAE decode | 129.2 s | **10.6 s** |
| peak VRAM allocated | 12.25 GB | 7.68 GB |

Quality cost against the untiled decode is **PSNR 48.3 dB, SSIM 0.9996**
(0.2% mean pixel difference) -- not bit-exact, because overlaps are blended,
but visually indistinguishable. Off by default pending verification at more
resolutions; recommended.

`--vae_tile_overlap` must cover the decoder's receptive field. Too small an
overlap does not produce visible seams so much as diffuse error across each
tile: at overlap 2 the mean error is 0.080, at overlap 6 it is 0.044.

### `--dit_quant fp8`

Quantizes the 300 transformer-block Linear weights to FP8 (per-tensor scale,
`torch._scaled_mm`). Requires compute capability >= 8.9 (Ada/Hopper).

| | bf16 | fp8 |
|---|---:|---:|
| DiT weights | 9.31 GB | **4.74 GB** |
| seconds per DiT forward | 17.8 | **11.9** |
| weight-transfer overhead | 17.1 s | **2.4 s** |

This **changes the generated video**: PSNR 21.7 dB, SSIM 0.85, ~3.9% mean pixel
difference against bf16. Detail statistics are unchanged (sharpness 2.52 vs
2.66, contrast 36.4 vs 37.5), so the result is a different sample rather than a
degraded one -- diffusion amplifies any numerical perturbation. Off by default.

`--dit_quant fp8_wo` stores FP8 weights but computes in bf16. It is more
accurate (SSIM 0.864) but **slower than not quantizing at all** (24.3 s per
forward vs 17.8 baseline) because it dequantizes the weight on every call. It
exists for accuracy comparisons, not for speed.

Note: per-row scaling is unsupported on sm_89 (`Per-row scaling is not
supported for this platform!`), so only per-tensor scaling is available here.

### T5 flags (`--t5_batch`, `--t5_fp32_compute`, `--t5_cache N`)

T5 runs on CPU (`--t5_cpu`) because UMT5-XXL is 10.6 GB. On a CPU without
AVX512-BF16/AMX its bfloat16 matmul is emulated, which made it the largest
phase of generation once the DiT was quantized.

| | default | + batch + fp32 | + cache (repeat run) |
|---|---:|---:|---:|
| T5 encode | 143.9 s | 97.8 s | 0.0 s |

- `--t5_cache 2` is **bit-identical** and costs nothing: the negative prompt is
  the same on every run, so the second generation in one process skips T5
  entirely. Recommended.
- `--t5_batch` and `--t5_fp32_compute` are ~1.47x together but shift the sample
  (SSIM 0.833), so they are off by default.
- `--t5_fp32_compute` keeps the weights in bf16 and casts per call, so it does
  not change the 10.6 GB footprint. It only helps with `--t5_cpu`.

> Note: the benchmark tables use 4 sampling steps so the matrix could run in
> reasonable time. That is 1/12th of the model default (50) and does **not**
> converge -- see benchmarks/README.md. Timing comparisons are like-for-like;
> the 4-step outputs are not usable video.

## Benchmarking

    python benchmarks/run_ti2v_bench.py --ckpt_dir <dir> --label mine --dit_quant fp8
    python benchmarks/compare_videos.py results/ti2v_A.npy results/ti2v_mine.npy

## Optional numerical flag

`WAN_ROPE_FP32=1` applies the rotary embedding in float32 instead of float64.
FP64 runs at 1/64 of FP32 on GeForce cards, so this makes `rope_apply` ~8.8x
faster in isolation.

It is **off by default** because it is not bit-exact: per call the two agree to
~5e-7, but diffusion amplifies that into a visibly different (though equally
valid) sample -- measured 2.6% mean pixel difference on a 4-step run.

It is also only worth enabling on a setup that is actually compute-bound. If
the card has to stream weights over PCIe every step, rope is not the
bottleneck and this changes almost nothing.

## Tests

```bash
pytest tests
```

These run without model checkpoints and cover the attention fallback, the
block modulation arithmetic and the input-validation and save paths. The
GPU-dependent tests skip automatically when no CUDA device is present.

`tests/test.sh` is a separate end-to-end smoke script and does require real
checkpoints and multiple GPUs.

## Formatting

The checked-in code is formatted with `yapf` plus `isort`:

```bash
make format
```
