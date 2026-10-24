"""Dump actual Inductor wrapper/Triton source for original and patched RoPE.

Run after installing torch and the test dependencies. Uses a PyTorch internal
inspection helper; not imported by the model or the benchmark.
"""

import argparse
import json
from pathlib import Path

import torch
from test_rope import load_model, reference_rope_apply
from torch._inductor.utils import run_and_get_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model()
    torch.manual_seed(123)
    x = torch.randn(1, 21 * 30 * 52, 12, 128, device="cuda")
    grid = torch.tensor([[21, 30, 52]])
    freqs = torch.cat([model.rope_params(64, n) for n in (44, 42, 42)], dim=1).cuda()
    results = {}
    metadata = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        grid=[21, 30, 52],
        dtype=str(x.dtype),
        files={},
    )
    with torch.no_grad(), torch._inductor.config.patch(force_disable_caches=True):
        for name, function in (("original", reference_rope_apply), ("patched", model.rope_apply)):
            result, codes = run_and_get_code(torch.compile(function, dynamic=False), x, grid, freqs)
            torch.cuda.synchronize()
            if not codes:
                raise RuntimeError("No generated code was captured")
            metadata["files"][name] = []
            for i, code in enumerate(codes):
                path = args.output / f"{name}_{i}.py"
                path.write_text(code, encoding="utf-8")
                metadata["files"][name].append(path.name)
            results[name] = result
        torch.testing.assert_close(results["original"], results["patched"], rtol=1e-6, atol=1e-6)
        metadata["max_abs_error"] = float((results["original"] - results["patched"]).abs().max())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
