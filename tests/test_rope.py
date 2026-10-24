"""Checkpoint-free tests of the upstream RoPE API."""

import importlib
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest
import torch


@lru_cache(None)
def load_model():
    # Import the real module without running wan/__init__.py and loading pipelines.
    name = "_wan_rope_modules"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "wan" / "modules")]
    sys.modules[name] = package
    return importlib.import_module(name + ".model")


@torch.compiler.disable
def reference_grid_sizes(grid_sizes):
    # The same host-metadata boundary as the patch, so the numerical baseline
    # can compile on PyTorch 2.8 instead of failing on aten._local_scalar_dense.
    return grid_sizes.tolist()


@torch.amp.autocast("cuda", enabled=False)
def reference_rope_apply(x, grid_sizes, freqs):
    # Original implementation, retained as a numerical and benchmark oracle.
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(reference_grid_sizes(grid_sizes)):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).float()


def frequencies(model, dim, device):
    sizes = [dim - 4 * (dim // 6), 2 * (dim // 6), 2 * (dim // 6)]
    return torch.cat([model.rope_params(32, size) for size in sizes], dim=1).to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "grids,length",
    [
        ([[2, 2, 3]], 12),
        ([[2, 2, 3], [2, 2, 3]], 12),
        ([[2, 2, 3], [2, 2, 3]], 15),
        ([[2, 2, 3], [1, 3, 2]], 15),
    ],
)
def test_rope_values_gradients_and_padding(device, dtype, grids, length):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = load_model()
    torch.manual_seed(123)
    # 128 includes the nonuniform temporal/spatial channel split used by Wan.
    x = torch.randn(len(grids), length, 2, 128, device=device, dtype=dtype)
    x.requires_grad_()
    freqs = frequencies(model, 128, device)
    grid = torch.tensor(grids)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        expected = reference_rope_apply(x, grid, freqs)
        actual = model.rope_apply(x, grid, freqs)
    if all(row == grids[0] for row in grids):
        c = x.shape[-1] // 2
        parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        fast = model._rope_apply_uniform(x, grids[0], parts)
        torch.testing.assert_close(fast, expected, rtol=1e-6, atol=1e-6)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, probe)[0]
    ref_grad = torch.autograd.grad(expected, x, probe)[0]
    torch.testing.assert_close(grad, ref_grad)
    for i, (f, h, w) in enumerate(grids):
        torch.testing.assert_close(
            actual[i, f * h * w :], x[i, f * h * w :].float(), rtol=0, atol=0
        )


def test_noncontiguous_rope_input():
    model = load_model()
    x = torch.randn(2, 2, 12, 32).transpose(1, 2)
    grid = torch.tensor([[2, 2, 3], [2, 2, 3]])
    freqs = frequencies(model, 32, "cpu")
    torch.testing.assert_close(
        model.rope_apply(x, grid, freqs), reference_rope_apply(x, grid, freqs)
    )
    parts = freqs.split([6, 5, 5], dim=1)
    torch.testing.assert_close(
        model._rope_apply_uniform(x, [2, 2, 3], parts),
        reference_rope_apply(x, grid, freqs),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_uniform_forward_and_backward(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = load_model()
    torch.manual_seed(45)
    x = torch.randn(2, 15, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
    freqs = frequencies(model, 128, "cuda")
    grid = torch.tensor([[2, 2, 3], [2, 2, 3]])
    compiled = torch.compile(model.rope_apply, dynamic=False)
    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    actual = compiled(x, grid, freqs)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > before
    expected = reference_rope_apply(x, grid, freqs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    probe = torch.randn_like(actual)
    gradient = torch.autograd.grad(actual, x, probe)[0]
    expected_gradient = torch.autograd.grad(expected, x, probe)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    changed_grid = torch.tensor([[1, 3, 2], [1, 3, 2]])
    torch.testing.assert_close(
        compiled(x, changed_grid, freqs),
        reference_rope_apply(x, changed_grid, freqs),
        rtol=1e-6, atol=1e-6,
    )


def test_compiled_heterogeneous_batch_keeps_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = load_model()
    x = torch.randn(2, 15, 2, 128, device="cuda", requires_grad=True)
    freqs = frequencies(model, 128, "cuda")
    grid = torch.tensor([[2, 2, 3], [1, 3, 2]])
    actual = torch.compile(model.rope_apply, dynamic=False)(x, grid, freqs)
    expected = reference_rope_apply(x, grid, freqs)
    torch.testing.assert_close(actual, expected)
