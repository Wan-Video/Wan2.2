# TI2V-5B benchmarks

Real end-to-end runs: prompt → T5 → DiT denoising → VAE decode → MP4. Not
microbenchmarks.

```bash
python benchmarks/run_ti2v_bench.py --ckpt_dir <ckpt> --label mine --dit_quant fp8
python benchmarks/compare_videos.py \
    benchmarks/results/ti2v_A_bf16_baseline.npy benchmarks/results/ti2v_mine.npy
```

Each run writes `ti2v_<label>.json` (metrics), `.mp4` (viewable) and `.npy`
(the raw pipeline tensor). Quality is compared on the `.npy`, because comparing
two H.264 encodes measures the codec as much as the model.

## Hardware

RTX 4060 Laptop, 8 GB, sm_89 · Windows 11 · 32 GB RAM · torch 2.6.0+cu124 ·
no flash-attn (SDPA fallback)

## Results

832×480, 5 frames, 4 steps, unipc, CFG 5.0, seed 42, `--offload_model True
--convert_model_dtype --t5_cpu`. Identical settings across variants.

| Variant | DiT weights | Peak VRAM alloc | s/DiT fwd | VAE decode | Total wall | vs A |
|---|---:|---:|---:|---:|---:|---:|
| A — bf16 baseline | 9.31 GB | 12.25 GB | 17.82 | 129.2 s | 523.5 s | 1.00× |
| B — fp8 | 4.74 GB | 10.31 GB | 11.91 | 118.8 s | 462.1 s | 1.13× |
| C — fp8_wo | 4.75 GB | 10.31 GB | 24.29 | 101.6 s | 538.3 s | 0.97× |
| E — bf16 + tiled VAE | 9.31 GB | 12.25 GB | 18.52 | **10.6 s** | 414.6 s | 1.26× |
| **D — fp8 + tiled VAE** | **4.74 GB** | **7.68 GB** | **9.46** | **13.7 s** | **330.4 s** | **1.58×** |

Generation phase only (excludes model load): 462.9 s → 244.9 s, **1.89×**.

## Quality, measured against A

| Variant | mean diff | PSNR | SSIM |
|---|---:|---:|---:|
| E (tiling only) | 0.21% | 48.33 dB | 0.9996 |
| B (quant only) | 3.92% | 21.72 dB | 0.8486 |
| D (quant + tiling) | 3.93% | 21.71 dB | 0.8487 |

Tiling is effectively lossless. All of D's deviation comes from quantization,
none from tiling (B vs D: PSNR 47.9 dB, SSIM 0.9995).

FP8 output is *different*, not *degraded* — detail statistics are preserved:

| | A | B | D |
|---|---:|---:|---:|
| sharpness | 2.521 | 2.660 | 2.676 |
| contrast (std) | 36.39 | 37.48 | 37.52 |
| high-frequency energy | 23.09 | 23.40 | 23.43 |

Diffusion amplifies any perturbation, so a changed sample is expected; what
matters is that it is not blurrier or noisier. It is not.

## Where the time goes now (variant D)

| phase | time | share |
|---|---:|---:|
| **T5 encode (CPU)** | **153.4 s** | **63%** |
| DiT forwards | 75.7 s | 31% |
| VAE decode | 13.7 s | 6% |
| transfers & other | 2.2 s | <1% |

T5 is now the dominant cost. It runs on CPU because `--t5_cpu` is required:
UMT5-XXL is 10.6 GB and does not fit alongside anything else on 8 GB.

## Larger generation

| | 5 frames | 49 frames |
|---|---:|---:|
| s per DiT forward | 9.46 | 194.1 |
| peak VRAM allocated | 7.68 GB | 8.77 GB |
| total wall | 330 s | 1885 s |

Tokens grow 6.5× (780 → 5070) but per-forward time grows 20×, and peak
allocation goes back over the 8 GB card — the DiT is spilling again at this
length. 49 frames is feasible (31 min) but no longer resident.

## Caveats

- Laptop GPU clocks vary. Variant A's DiT forward measured 213 s and 143 s on
  two runs of identical settings; treat differences under ~1.2× as noise.
- Benchmarks must run with nothing else on the GPU. An early run was polluted
  by a concurrent test suite and read 46.9 s/step instead of ~32.
- One resolution and frame count. Nothing here is verified at 720p.


## T5 phase

T5 became the dominant cost once the DiT was quantized. Three changes, all
opt-in (`--t5_batch`, `--t5_fp32_compute`, `--t5_cache N`).

832x480, 5 frames, 4 steps, on top of `--dit_quant fp8 --vae_tile`:

| Variant | T5 | Denoise | VAE | Generate | Wall | Quality vs T0 |
|---|---:|---:|---:|---:|---:|---|
| T0 fp8 + tiled VAE | 143.9 s | 93.8 s | 13.8 s | 253.7 s | 339.9 s | reference |
| T1 + T5 batch | 138.8 s | 75.6 s | 13.7 s | 230.0 s | 307.1 s | SSIM 0.840 |
| T2 + batch + fp32 | **97.8 s** | 81.3 s | 13.7 s | **194.5 s** | **267.3 s** | SSIM 0.833 |
| T3 + cache, run 2 | **0.0 s** | 110.7 s | 9.7 s | **122.2 s** | — | identical to T2 |

The cache is **bit-identical** (PSNR inf, SSIM 1.0): a cached embedding is the
same tensor, so the sample cannot move. Batching and fp32 compute each shift
the sample ~4.2%, so they stay opt-in.

### Why fp32 compute, and why it under-delivered

bfloat16 matmul is emulated on CPUs without AVX512-BF16/AMX. An isolated GEMM
benchmark at UMT5-XXL shapes showed bf16 at 94-232 ms against fp32 at 24-64 ms,
suggesting ~3.3x. On the real encoder it was **1.13x**.

The wrapper only converts `nn.Linear` (168 of them). T5 computes attention
scores with `torch.einsum` and allocates a `[1, 64, 512, 512]` `attn_bias` per
layer, none of which the wrapper touches. The microbenchmark measured the part
that was easy to change, not the part that dominates.

Batching alone is 1.11x. It cannot be more: the tokenizer pads every prompt to
exactly 512, so two batch-1 forwards and one batch-2 forward do identical
arithmetic. Together they reach 1.47x.

## Step count: the 4-step benchmarks are not usable video

Every table above uses 4 steps so the matrix could run in reasonable time. That
is 1/12th of the model default (50), and it does not converge:

| run | frames | steps | luma std | luma mean |
|---|---:|---:|---:|---:|
| T2 | 5 | 4 | 0.256 | 0.806 |
| S50 | 5 | **50** | **0.587** | **0.570** |
| L49 | 49 | 4 | 0.034 | 0.948 |
| L121 | 121 | 4 | 0.009 | 0.959 |

At 4 steps the output washes toward white, and it gets worse with length -- at
121 frames the frames are essentially blank. **The timing comparisons are valid
(like-for-like), but no 4-step output here is a usable video.**

## Long generations

| | 5f / 4 steps | 49f / 4 steps | 121f / 4 steps | 5f / 50 steps |
|---|---:|---:|---:|---:|
| T5 | 97.8 s | 96.4 s | 83.0 s | 82.4 s |
| DiT total | 81.3 s | 933.6 s | 4446.0 s | 1086.4 s |
| s per DiT forward | 10.2 | 116.7 | 555.7 | 10.9 |
| VAE decode | 13.7 s | 97.9 s | 257.1 s | 14.4 s |
| peak VRAM alloc | 7.68 GB | 8.77 GB | 10.69 GB | 7.68 GB |
| **total wall** | **267 s** | **20.2 min** | **81.3 min** | **21.3 min** |

DiT time scales superlinearly: 780 -> 5,070 -> 12,090 tokens costs
10.2 -> 116.7 -> 555.7 s per forward. Attention is quadratic, and peak
allocation goes back over the 8 GB card at 49 frames and beyond, so the spill
returns at length.

### How long is ~5 seconds of video, really?

121 frames at 24 fps is 5.04 s. At 4 steps it takes 81 minutes and the result is
blank. At the default 50 steps, the DiT alone is 555.7 s x 100 forwards =
**~15.4 hours**. That is the honest answer: short clips at real quality are
practical on this hardware (5 frames / 50 steps = 21 min), a 5-second clip is
not.
