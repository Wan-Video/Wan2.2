"""Check that rope_apply in wan/modules/model.py matches the original
implementation bit for bit (forward and backward) and report the speedup.

    python tests/test_rope_apply.py

Needs only torch and a CUDA GPU; no checkpoints are loaded.
"""
import ast
from pathlib import Path

import torch

MODEL_PY = Path(__file__).resolve().parents[1] / "wan" / "modules" / "model.py"


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_original(x, grid_sizes, freqs):
    # Original implementation, kept here as the reference.
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def load_from_model_py(name):
    # Load one top-level function from model.py without importing the
    # wan package (which would pull in the whole pipeline).
    tree = ast.parse(MODEL_PY.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), str(MODEL_PY), "exec"), ns)
    return ns[name]


def make_freqs(head_dim):
    rope_params = load_from_model_py("rope_params")
    d = head_dim
    return torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6))
    ],
                     dim=1).cuda()


def check_identical(rope_apply, freqs, num_heads, head_dim):
    f, h, w = 21, 30, 52  # 480p, 81 frames
    seq_len = f * h * w
    cases = [
        ("batch 1, fp32", torch.float32, [[f, h, w]]),
        ("batch 1, bf16", torch.bfloat16, [[f, h, w]]),
        ("batch 3, mixed sizes + padding, fp32", torch.float32,
         [[f, h, w], [f, h, w - 4], [f - 5, h, w]]),
        ("batch 3, mixed sizes + padding, bf16", torch.bfloat16,
         [[f, h, w], [f, h, w - 4], [f - 5, h, w]]),
    ]
    ok = True
    for name, dtype, grids in cases:
        x = torch.randn(len(grids), seq_len, num_heads, head_dim,
                        device="cuda").to(dtype)
        grid_sizes = torch.tensor(grids)
        xa = x.clone().requires_grad_()
        xb = x.clone().requires_grad_()
        a = rope_apply_original(xa, grid_sizes, freqs)
        b = rope_apply(xb, grid_sizes, freqs)
        fwd = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        grad = torch.randn_like(a)
        a.backward(grad)
        b.backward(grad)
        bwd = torch.equal(xa.grad, xb.grad)
        ok &= fwd and bwd
        print(f"  {name:40s} forward {'identical' if fwd else 'DIFFERENT'}, "
              f"backward {'identical' if bwd else 'DIFFERENT'}")
    return ok


def time_pair_ms(fn_a, fn_b, *args, warmup=5, rounds=50):
    # Alternate the two functions so neither gets a warm-up advantage.
    def once(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        fn_a(*args)
        fn_b(*args)
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(once(fn_a))
        tb.append(once(fn_b))
    return sorted(ta)[rounds // 2], sorted(tb)[rounds // 2]


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    rope_apply = load_from_model_py("rope_apply")
    head_dim = 128
    freqs = make_freqs(head_dim)
    torch.manual_seed(0)

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    print("Bit-identical check against the original rope_apply:")
    ok = check_identical(rope_apply, freqs, num_heads=12, head_dim=head_dim)

    print("Speed (median of 50 alternating calls, 480p / 81 frames, one call):")
    grid_sizes = torch.tensor([[21, 30, 52]])
    with torch.no_grad():
        for num_heads, label in ((12, "1.3B, 12 heads"), (40, "14B, 40 heads")):
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(1, 21 * 30 * 52, num_heads, head_dim,
                                device="cuda").to(dtype)
                t0, t1 = time_pair_ms(rope_apply_original, rope_apply, x,
                                      grid_sizes, freqs)
                print(f"  {label:15s} {str(dtype):15s} original {t0:6.2f} ms, "
                      f"new {t1:6.2f} ms, {t0 / t1:.2f}x faster")
                del x
                torch.cuda.empty_cache()

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
