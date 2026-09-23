# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for the rotary-embedding frequency cache and the rotation dtype.

Two separate things are covered:

1. `freqs_i` depends only on the grid size and the constant `freqs` buffer, but
   it used to be rebuilt inside every rope_apply call -- twice per attention
   (q and k), per block, per CFG pass, per step. Caching it must be bit-exact.

2. `USE_FP32_ROPE` narrows the complex rotation from float64 to float32. That
   is ~8.8x faster on GeForce parts but is *not* bit-exact, so it is off by
   default. These tests pin both: the default reproduces the original exactly,
   and the opt-in stays within a tight per-call bound.

No model weights needed.
"""
import pytest
import torch

import wan.modules.model as model_mod
from wan.modules.model import _ROPE_FREQS_CACHE, rope_apply, rope_params

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')

# Bounds a single call, not an end-to-end sample: diffusion amplifies this.
ROPE_FP32_TOL = 1e-5


def _old_rope_apply(x, grid_sizes, freqs):
    """Verbatim copy of the pre-change implementation (float64, no cache)."""
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


def _make_freqs(head_dim, device):
    # Built exactly the way WanModel.__init__ builds self.freqs, so the widths
    # line up with the split inside rope_apply.
    d = head_dim
    return torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6))
    ],
                     dim=1).to(device)


@pytest.fixture
def fp32_rope(monkeypatch):
    monkeypatch.setattr(model_mod, 'USE_FP32_ROPE', True)
    _ROPE_FREQS_CACHE.clear()
    yield
    _ROPE_FREQS_CACHE.clear()


def test_fp32_rope_is_off_by_default():
    """Output must match upstream bit-for-bit unless explicitly opted in."""
    assert model_mod.USE_FP32_ROPE is False


@cuda_only
@pytest.mark.parametrize('f,h,w', [(2, 4, 4), (3, 5, 7)])
def test_default_rope_is_bit_identical(f, h, w):
    torch.manual_seed(0)
    _ROPE_FREQS_CACHE.clear()

    n, head_dim = 4, 32
    x = torch.randn(1, f * h * w, n, head_dim, device='cuda')
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _make_freqs(head_dim, 'cuda')

    expected = _old_rope_apply(x, grid_sizes, freqs)
    got = rope_apply(x, grid_sizes, freqs)

    assert torch.equal(got, expected), 'caching must not change any value'
    # and again, now served from the cache
    assert torch.equal(rope_apply(x, grid_sizes, freqs), expected)


@cuda_only
def test_fp32_rope_stays_within_tolerance(fp32_rope):
    torch.manual_seed(0)
    f, h, w = 3, 5, 7
    n, head_dim = 4, 32
    x = torch.randn(1, f * h * w, n, head_dim, device='cuda')
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _make_freqs(head_dim, 'cuda')

    expected = _old_rope_apply(x, grid_sizes, freqs)
    got = rope_apply(x, grid_sizes, freqs)

    assert not torch.equal(got, expected), 'fp32 path should differ slightly'
    err = (got - expected).abs().max().item()
    assert err < ROPE_FP32_TOL, 'deviation {} exceeds {}'.format(
        err, ROPE_FP32_TOL)


@cuda_only
def test_cache_is_keyed_on_the_rotation_dtype(monkeypatch):
    """Flipping the flag must not serve a cached table of the wrong dtype."""
    torch.manual_seed(0)
    _ROPE_FREQS_CACHE.clear()

    f, h, w = 2, 4, 4
    n, head_dim = 4, 32
    x = torch.randn(1, f * h * w, n, head_dim, device='cuda')
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _make_freqs(head_dim, 'cuda')

    fp64_out = rope_apply(x, grid_sizes, freqs)
    monkeypatch.setattr(model_mod, 'USE_FP32_ROPE', True)
    fp32_out = rope_apply(x, grid_sizes, freqs)

    assert not torch.equal(fp64_out, fp32_out)
    assert (fp64_out - fp32_out).abs().max().item() < ROPE_FP32_TOL
    _ROPE_FREQS_CACHE.clear()


@cuda_only
def test_rope_handles_padding():
    """A shorter token count must leave the padded tail untouched."""
    torch.manual_seed(0)
    _ROPE_FREQS_CACHE.clear()

    f, h, w = 2, 3, 3
    n, head_dim = 4, 32
    seq_len, padded = f * h * w, f * h * w + 11
    x = torch.randn(1, padded, n, head_dim, device='cuda')
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _make_freqs(head_dim, 'cuda')

    expected = _old_rope_apply(x, grid_sizes, freqs)
    got = rope_apply(x, grid_sizes, freqs)

    assert got.shape == (1, padded, n, head_dim)
    assert torch.equal(got, expected)
    assert torch.equal(got[0, seq_len:], x[0, seq_len:].float())


@cuda_only
def test_cache_is_keyed_on_the_source_table():
    """A different freqs tensor must not be served a stale cached entry."""
    torch.manual_seed(0)
    _ROPE_FREQS_CACHE.clear()

    f, h, w = 2, 4, 4
    n, head_dim = 4, 32
    x = torch.randn(1, f * h * w, n, head_dim, device='cuda')
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)

    freqs_a = _make_freqs(head_dim, 'cuda')
    out_a = rope_apply(x, grid_sizes, freqs_a)

    # a structurally identical but distinct table with different values
    freqs_b = (freqs_a * torch.polar(
        torch.ones_like(freqs_a.real), torch.full_like(freqs_a.real, 0.5)))
    out_b = rope_apply(x, grid_sizes, freqs_b)

    assert not torch.equal(out_a, out_b)
    assert torch.equal(out_b, _old_rope_apply(x, grid_sizes, freqs_b))
