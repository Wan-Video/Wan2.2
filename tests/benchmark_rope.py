"""Benchmark isolated RoPE and the actual upstream block without checkpoints.

Runs from a Wan checkout when copied to tests/benchmark_rope.py. The optional
SDPA adapter is benchmark-only and is applied equally to both implementations.
"""

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--attention", choices=("flash", "sdpa"), default="flash")
    parser.add_argument("--only", choices=("rope", "block", "all"), default="all")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    tests = load_module(args.repo / "tests/test_rope.py", "_rope_bench_tests")
    tests.load_model()  # establishes the package path without pipeline imports
    model_file = args.repo / "wan/modules/model.py"
    baseline = load_module(model_file, "_wan_rope_modules.baseline")
    candidate = load_module(model_file, "_wan_rope_modules.candidate")
    baseline.rope_apply = tests.reference_rope_apply
    torch.manual_seed(123)
    report = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        compile_mode="default; dynamic=False",
        attention_version=importlib.metadata.version("flash-attn")
        if args.attention == "flash"
        else None,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        platform=platform.platform(),
        base_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        seed=123,
        attention=args.attention,
        checkpoint="none; deterministic random weights",
        rows=[],
        errors=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def benchmark(label, functions, expected):
        funcs = dict(functions)
        for name, function in list(functions.items()):
            funcs[name + "_compiled"] = torch.compile(function, dynamic=False)
        active, samples, peaks, outputs = {}, {}, {}, {}
        for name, function in funcs.items():
            print("warmup", label, name, flush=True)
            started = time.perf_counter()
            before = dict(torch._dynamo.utils.counters["stats"])
            try:
                for _ in range(3):
                    actual = function()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-6 if label.startswith("rope") else 2e-3,
                    atol=1e-6 if label.startswith("rope") else 2e-3,
                )
                outputs[name] = actual.detach().cpu()
                difference = (actual.float() - expected.float()).abs()
                error = float(difference.max())
                del actual, difference
                active[name], samples[name] = function, []
                torch.cuda.reset_peak_memory_stats()
                function()
                torch.cuda.synchronize()
                peaks[name] = torch.cuda.max_memory_allocated() / 2**30
                stats = torch._dynamo.utils.counters["stats"]
                report["rows"].append(
                    dict(
                        case=label,
                        variant=name,
                        max_abs_error=error,
                        warmup_wall_s=time.perf_counter() - started,
                        peak_allocated_gib=peaks[name],
                        compiled_graphs=stats.get("unique_graphs", 0)
                        - before.get("unique_graphs", 0),
                    )
                )
            except Exception as error:
                report["errors"].append(
                    dict(
                        case=label,
                        variant=name,
                        error=type(error).__name__ + ": " + str(error)[:3000],
                    )
                )
                print("unsupported", label, name, str(error)[:180], flush=True)
            save()
        if "baseline_compiled" in outputs and "candidate_compiled" in outputs:
            delta = (outputs["candidate_compiled"] - outputs["baseline_compiled"]).abs()
            torch.testing.assert_close(
                outputs["candidate_compiled"],
                outputs["baseline_compiled"],
                rtol=2e-3 if label.startswith("upstream") else 1e-6,
                atol=2e-3 if label.startswith("upstream") else 1e-6,
            )
            report.setdefault("compiled_pair_error", {})[label] = float(delta.max())
        del outputs
        rng = random.Random(27)
        for round_index in range(args.rounds):
            order = list(active)
            rng.shuffle(order)
            for name in order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.repeat):
                    active[name]()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) / args.repeat)
        for row in report["rows"]:
            if row["case"] == label:
                values = samples[row["variant"]]
                row.update(
                    median_ms=statistics.median(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    samples_ms=values,
                    rounds=args.rounds,
                    repeats_per_sample=args.repeat,
                )
        save()
        print("finished", label, flush=True)

    with torch.no_grad():
        if args.only in ("rope", "all"):
            for grid in ((3, 30, 52), (21, 30, 52)):
                for dtype in (torch.float16, torch.bfloat16, torch.float32):
                    length = grid[0] * grid[1] * grid[2]
                    x = torch.randn(1, length, 12, 128, device="cuda", dtype=dtype)
                    grids = torch.tensor([grid])
                    freqs = tests.frequencies(candidate, 128, "cuda")
                    # frequency table has 32 positions, sufficient for all tested axes
                    if grid[2] > 32:
                        freqs = torch.cat(
                            [candidate.rope_params(64, n) for n in (44, 42, 42)], dim=1
                        ).cuda()
                    functions = dict(
                        baseline=lambda: baseline.rope_apply(x, grids, freqs),
                        candidate=lambda: candidate.rope_apply(x, grids, freqs),
                    )
                    benchmark(
                        "rope_" + str(grid) + "_" + str(dtype), functions, functions["baseline"]()
                    )
                    del x, freqs, functions
                    gc.collect()
                    torch.cuda.empty_cache()
        if args.only in ("block", "all"):
            if args.attention == "sdpa":

                def attention(q, k, v, **kwargs):
                    # Tested uniform, unpadded, noncausal inputs only.
                    dtype = torch.bfloat16
                    result = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2).to(dtype),
                        k.transpose(1, 2).to(dtype),
                        v.transpose(1, 2).to(dtype),
                    )
                    return result.transpose(1, 2).to(q.dtype)

                baseline.flash_attention = candidate.flash_attention = attention
            is_v21 = (
                "cross_attn_type"
                in __import__("inspect").signature(candidate.WanAttentionBlock).parameters
            )
            dim, heads, ffn = (1536, 12, 8960) if is_v21 else (3072, 24, 14336)
            kwargs = dict(dim=dim, ffn_dim=ffn, num_heads=heads, cross_attn_norm=True)
            if is_v21:
                kwargs["cross_attn_type"] = "t2v_cross_attn"
            block0 = baseline.WanAttentionBlock(**kwargs).eval().to("cuda", torch.bfloat16)
            block1 = candidate.WanAttentionBlock(**kwargs).eval().to("cuda", torch.bfloat16)
            block1.load_state_dict(block0.state_dict())
            grid = (3, 30, 52)
            length = grid[0] * grid[1] * grid[2]
            x = torch.randn(1, length, dim, device="cuda")
            e = torch.randn((1, 6, dim) if is_v21 else (1, length, 6, dim), device="cuda") * 0.1
            context = torch.randn(1, 512, dim, device="cuda")
            freqs = torch.cat([candidate.rope_params(64, n) for n in (44, 42, 42)], dim=1).cuda()
            inputs = (x, e, torch.tensor([length]), torch.tensor([grid]), freqs, context, None)

            def call(block):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return block(*inputs)

            functions = dict(baseline=lambda: call(block0), candidate=lambda: call(block1))
            report["block"] = dict(dim=dim, heads=heads, ffn_dim=ffn, grid=grid)
            benchmark("upstream_block_" + str(grid), functions, functions["baseline"]())
    save()


if __name__ == "__main__":
    main()
