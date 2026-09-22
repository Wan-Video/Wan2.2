# Tests

## Unit tests (no checkpoints required)

```bash
pytest tests
```

These run against the code alone — no model weights, no downloads. Tests that
need a GPU skip themselves automatically when no CUDA device is present.

| File | Covers |
|---|---|
| `test_attention.py` | The attention dispatcher and its SDPA fallback: correct shapes, `k_lens` masking, and that `flash_attention` degrades gracefully instead of asserting when flash-attn is not installed. |
| `test_model_modulation.py` | That indexing into the block modulation parameter is arithmetically identical to the previous full-broadcast formulation, and allocates less. |
| `test_rope.py` | The rotary-embedding frequency cache (bit-exact) and the float64 -> float32 rotation (deviation pinned far below bf16 precision). |
| `test_failure_modes.py` | That failures are loud: `save_video` raises and leaves no partial file, the VAE rejects bad input, and CLI arguments are validated up front. |
| `test_checkpoint_loading.py` | That every `torch.load` call passes `weights_only` explicitly -- torch 2.6 flipped the default and the Wan .pth checkpoints cannot be read under the new one. |
| `test_pipeline_e2e.py` | A full run of the real `WanTI2V` sampler loop, DiT forward, VAE decode and save path, using tiny random weights instead of the 23 GB checkpoint. |

## End-to-end smoke script (checkpoints + multiple GPUs required)

Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a
folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```

Note that this script only checks that the commands run; it makes no assertions
about the output.
