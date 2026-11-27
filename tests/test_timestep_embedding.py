"""Check the timestep-embedding change in WanModel.forward.

    python tests/test_timestep_embedding.py

For a 1-D t the time embedding is now computed once per sample and
broadcast, instead of once per token. This is not bit-identical: a GEMM
over 1 row and over seq_len rows rounds differently. The test compares the
difference with the run-to-run noise the original code already has (the
same sample alone vs. in a batch of 2) and checks that batch size 2 with a
1-D t works. Random weights, no checkpoints; needs a CUDA GPU and the
packages in requirements.txt.
"""
import importlib
import sys
import time
import types
from pathlib import Path

import torch


def load_model_module():
    # Import wan/modules/model.py without running wan/__init__.py.
    name = "_wan_modules"
    package = types.ModuleType(name)
    package.__path__ = [
        str(Path(__file__).resolve().parents[1] / "wan" / "modules")
    ]
    sys.modules[name] = package
    return importlib.import_module(name + ".model")


def rel_l2(a, b):
    return ((a - b).norm() / a.norm()).item()


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    M = load_model_module()
    f, h, w = 21, 30, 52  # latent grid of a 480p, 81-frame video
    seq_len = f * h * w
    torch.manual_seed(0)
    model = M.WanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=16,
        dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
        num_heads=12, num_layers=4)  # 1.3B block size, 4 layers
    with torch.no_grad():
        for p in model.parameters():  # default init zeroes the head
            p.normal_(0, 0.02)
    model = model.cuda().eval().requires_grad_(False)
    x = torch.randn(16, f, h * 2, w * 2, device="cuda")
    context = torch.randn(77, 4096, device="cuda") * 0.1

    def forward(xs, t):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), \
                torch.no_grad():
            return model(xs, t=t, context=[context] * len(xs),
                         seq_len=seq_len)

    def time_and_memory(t):
        for _ in range(2):
            forward([x], t)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(3):
            forward([x], t)
        torch.cuda.synchronize()
        return ((time.perf_counter() - start) / 3 * 1000,
                torch.cuda.max_memory_allocated() / 1e9)

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    ok = True
    for tv in (999.0, 500.0, 37.0):
        t = torch.tensor([tv], device="cuda")
        # A per-token t of shape [1, seq_len] runs exactly the computation
        # the original code ran for a 1-D t after t.expand(1, seq_len).
        old = forward([x], t.expand(1, seq_len))[0]
        new = forward([x], t)[0]
        # Noise reference: the original computation, same sample, batch of 2.
        ref = forward([x, x], t.expand(2, seq_len))[0]
        diff, noise = rel_l2(old, new), rel_l2(old, ref)
        ok &= diff <= 2 * noise
        print(f"  t={tv:5.0f}: new vs original rel L2 {diff:.2e}, "
              f"original alone vs in batch of 2 {noise:.2e}")

    out = forward([x, x], torch.tensor([500.0, 500.0], device="cuda"))
    print(f"  batch 2 with 1-D t: works, output {tuple(out[0].shape)} x 2")

    t = torch.tensor([500.0], device="cuda")
    t_old, m_old = time_and_memory(t.expand(1, seq_len))
    t_new, m_new = time_and_memory(t)
    print(f"  forward (4 layers): original {t_old:.1f} ms, {m_old:.1f} GB peak"
          f" | new {t_new:.1f} ms, {m_new:.1f} GB peak")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
