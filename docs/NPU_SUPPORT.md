# NPU 910B Support

Wan2.2 can run on Huawei Ascend NPU 910B (32GB HBM) with CPU offload, enabling
free inference via platforms like [AtomGit](https://atomgit.com) which provide
NPU 910B 64GB instances at no cost.

## Tested Configuration

| Component | Version |
|-----------|---------|
| NPU | Ascend 910B (32GB HBM + 64GB CPU) |
| CANN | 8.5 |
| PyTorch | 2.9 |
| torch_npu | 2.9 |
| Model | Wan2.2-S2V-14B |
| Resolution | 480x832 |
| Steps | 5 |
| Frames | 40 |
| Inference time | ~550s (9.2 min) |
| Core-hours | ~2.47 |

## Quick Start

```bash
# 1. Clone Wan2.2
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# 2. Apply NPU patches
python scripts/patch_npu.py --repo-root . --max-memory 15GB

# 3. Download model
python -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.2-S2V-14B', local_dir='./Wan2.2-S2V-14B')"

# 4. Run inference
python generate.py \
    --task s2v-14B \
    --size 480*832 \
    --ckpt_dir ./Wan2.2-S2V-14B \
    --offload_model True \
    --sample_steps 5 \
    --num_clip 1 \
    --frame_num 40 \
    --prompt "A person is talking to camera" \
    --image portrait.jpg \
    --audio audio.wav \
    --save_file output.mp4
```

## Patches Overview

The patch script (`scripts/patch_npu.py`) applies 10 modifications:

### 1. CUDA to NPU mapping
Replaces all `torch.cuda.*` references with `torch.npu.*` across the codebase,
adds `import torch_npu`, and switches distributed backend from NCCL to HCCL.

### 2. Staged loading (memory optimization)
Releases the T5 text encoder from HBM before loading the diffusion model,
then loads the diffusion model with `device_map='auto'` for automatic
HBM/CPU offload. This is critical because the 14B model + T5 encoder
exceeds 32GB HBM when loaded simultaneously.

### 3. Accelerate compatibility
Skips `.to(device)` and `.cpu()` calls for models managed by `accelerate`
(detected via `_hf_hook` attribute), since accelerate handles device
placement internally with `device_map='auto'`.

### 4. flash_attention to SDPA
NPU 910B does not support `flash_attn` package. The patch disables the
flash_attention path, falling back to PyTorch's native
`scaled_dot_product_attention` (SDPA).

### 5. dtype fixes
- VAE encode: cast input to `float32` before encoding
- model_s2v: replace `assert e.dtype == torch.float32` with `e = e.float()`
  (graceful cast instead of hard assertion failure)

### 6. Memory tuning
- `max_memory={0: '15GB', 'cpu': '60GB'}` — limits HBM usage to 15GB,
  offloading the rest to CPU
- `torch.npu.empty_cache()` before VAE decode to free intermediate tensors

## Environment Setup

```bash
# CANN 8.5+ must be installed on the host
# PyTorch + torch_npu
pip install torch==2.9.0 torch_npu==2.9.0

# Wan2.2 dependencies
pip install modelscope peft omegaconf ftfy safetensors tqdm einops \
    scipy pillow easydict inflect wetext hydra-core rich \
    opencv-python-headless imageio lightning conformer \
    HyperPyYAML loguru regex requests packaging GitPython \
    diffusers dashscope transformers "huggingface-hub<1.0" imageio-ffmpeg
```

## Known Limitations

- **Resolution**: 480x832 tested; higher resolutions (e.g., 704x960) may
  require more HBM or longer inference time
- **flash_attention**: SDPA fallback is slower than native flash_attention
  on CUDA; this is a trade-off for NPU compatibility
- **Multi-NPU**: Only single-NPU tested; multi-NPU would require HCCL
  distributed setup
- **T2V/I2V**: Only S2V (speech-to-video) tested; T2V and I2V tasks
  should work with the same patches but are untested

## Results

Output: 480x832, 40 frames, H264, ~1.8MB MP4
Quality: AI-rated 8.5/10 (natural lip sync, good facial detail)
Cost: Free on AtomGit NPU 910B 64GB tier

## Credit

Tested and patched by [@0xPabloLI](https://github.com/0xPabloLI).
Original Wan2.2 by Wan-AI team.