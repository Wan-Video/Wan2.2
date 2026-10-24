
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```


## Checkpoint-free RoPE validation

```bash
python -m pytest tests/test_rope.py -q
python tests/benchmark_rope.py --attention flash --output /tmp/rope_benchmark.json
```

The benchmark compares the original complex implementation with the uniform-batch
real implementation, both eager and `torch.compile(dynamic=False)`. It warms each
variant, then shuffles execution order in 12 rounds of two calls per sample. JSON
contains every sample, compilation graph counts, peak allocated CUDA memory and
maximum absolute errors. Compilation and correctness checks are outside timed
samples. Peak memory includes live inputs, both blocks and compiler allocations;
it is not the isolated temporary workspace size.

The block uses deterministic random weights and the upstream architecture: 1.3B
block dimensions for Wan2.1 and 5B block dimensions for Wan2.2. Inputs represent a
3-frame latent chunk (3 x 30 x 52 tokens), not an entire generated video.
`--attention sdpa` provides an explicitly labelled benchmark-only substitute when
FlashAttention is unavailable. It does not change the model source. No weights are
downloaded and no end-to-end video speedup is inferred from these measurements.

The original numerical baseline in this benchmark uses the same disabled
host-metadata read as the patched version. This avoids a PyTorch 2.8
`aten._local_scalar_dense` backend failure from `grid_sizes.tolist()` and isolates
the complex-vs-real tensor computation. No other baseline arithmetic changes.
