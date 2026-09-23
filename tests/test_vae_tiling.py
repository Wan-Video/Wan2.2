# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for optional spatially tiled VAE decoding.

On the real 8 GB benchmark the VAE decode was the single largest phase --
roughly 4.5 minutes of a 7 minute run -- because the decoder holds its peak
activation for a whole HxW plane at once. Tiling bounds that peak by the tile
size instead of the frame size.

Tiling blends overlapping regions, so it is not bit-exact and is off by
default. These tests pin the geometry, the seam behaviour and the opt-in.

No model weights needed.
"""
import pytest
import torch

from wan.modules.vae2_2 import (
    Wan2_2_VAE,
    WanVAE_,
    _feather_1d,
    _tile_spans,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')

Z_DIM = 16


def test_tile_spans_cover_the_axis():
    for total, tile, overlap in [(30, 16, 4), (52, 16, 4), (7, 16, 4),
                                 (64, 16, 0), (33, 8, 3)]:
        spans = _tile_spans(total, tile, overlap)
        assert spans[0][0] == 0
        assert spans[-1][1] == total
        for a, b in spans:
            assert 0 <= a < b <= total
            assert b - a <= tile or tile >= total
        # consecutive spans must touch or overlap, never leave a gap
        for (_, prev_end), (nxt_start, _) in zip(spans, spans[1:]):
            assert nxt_start <= prev_end


def test_tile_spans_degenerate_cases():
    assert _tile_spans(10, 0, 0) == [(0, 10)]
    assert _tile_spans(10, 99, 4) == [(0, 10)]


def test_feather_ramp_shape():
    like = torch.zeros(1)
    w = _feather_1d(10, 3, 3, like)
    assert w.shape == (10,)
    assert (w > 0).all() and (w <= 1).all()
    assert w[0] < w[1] < w[2]          # rising edge
    assert w[-1] < w[-2] < w[-3]       # falling edge
    assert w[5] == pytest.approx(1.0)  # untouched middle

    flat = _feather_1d(10, 0, 0, like)
    assert torch.equal(flat, torch.ones(10))


def _tiny_vae(device='cuda'):
    torch.manual_seed(0)
    model = WanVAE_(
        dim=32,
        dec_dim=32,
        z_dim=Z_DIM,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0).eval().requires_grad_(False).to(device)

    vae = Wan2_2_VAE.__new__(Wan2_2_VAE)
    vae.dtype = torch.float32
    vae.device = device
    vae.scale = [
        torch.zeros(Z_DIM, device=device),
        1.0 / torch.ones(Z_DIM, device=device)
    ]
    vae.model = model
    return vae


def test_enable_tiling_rejects_overlap_at_least_tile_size():
    vae = _tiny_vae('cpu')
    with pytest.raises(ValueError, match='overlap'):
        vae.enable_tiling(tile_size=8, overlap=8)
    with pytest.raises(ValueError, match='overlap'):
        vae.enable_tiling(tile_size=8, overlap=12)


def test_tiling_is_off_by_default():
    vae = _tiny_vae('cpu')
    assert getattr(vae.model, 'tile_size', 0) == 0


def test_enable_then_disable_round_trips():
    vae = _tiny_vae('cpu')
    vae.enable_tiling(tile_size=8, overlap=2)
    assert vae.model.tile_size == 8 and vae.model.tile_overlap == 2
    vae.disable_tiling()
    assert vae.model.tile_size == 0


@cuda_only
def test_tiled_decode_matches_whole_decode():
    """
    The real correctness bar: tiled output must be close enough to untiled that
    no seam is visible, even though blending makes it non-bit-exact.
    """
    vae = _tiny_vae()
    torch.manual_seed(1)
    z = [torch.randn(Z_DIM, 2, 16, 24, device='cuda')]

    whole = vae.decode(z)[0]
    vae.enable_tiling(tile_size=16, overlap=8)
    tiled = vae.decode(z)[0]

    assert tiled.shape == whole.shape
    assert torch.isfinite(tiled).all()

    # Tiling is approximate: each tile's convolutions see zero padding where a
    # neighbour's content would be, so the overlap has to cover the decoder's
    # receptive field. Outputs are clamped to [-1, 1], so this is absolute.
    err = (tiled - whole).abs()
    assert err.mean() < 0.02, 'mean deviation {:.4f} too large'.format(
        err.mean())
    assert err.max() < 0.30, 'max deviation {:.4f} too large'.format(err.max())


@cuda_only
def test_more_overlap_reduces_error():
    """
    The invariant that makes the knob meaningful: widening the overlap must
    move the tiled result towards the untiled one. If this ever inverts, the
    blending or the span maths is wrong.
    """
    vae = _tiny_vae()
    torch.manual_seed(1)
    z = [torch.randn(Z_DIM, 2, 16, 24, device='cuda')]
    ref = vae.decode(z)[0]

    errs = []
    for overlap in (2, 4, 6):
        vae.enable_tiling(tile_size=8, overlap=overlap)
        errs.append((vae.decode(z)[0] - ref).abs().mean().item())

    assert errs == sorted(errs, reverse=True), (
        'error should fall as overlap grows, got {}'.format(errs))


@cuda_only
def test_tiled_decode_has_no_seam_discontinuity():
    """
    A blending bug shows up as a column/row of large error at the tile joins,
    not as uniform error. Compare the worst column against the median column.
    """
    vae = _tiny_vae()
    torch.manual_seed(2)
    z = [torch.randn(Z_DIM, 2, 16, 24, device='cuda')]

    whole = vae.decode(z)[0]
    vae.enable_tiling(tile_size=8, overlap=3)
    tiled = vae.decode(z)[0]

    col_err = (tiled - whole).abs().mean(dim=(0, 1, 2))  # per output column
    median = col_err.median()
    worst = col_err.max()
    assert worst < median + 0.1, (
        'column error spikes at a tile boundary: worst {:.4f} vs median '
        '{:.4f}'.format(worst, median))


@cuda_only
def test_tile_larger_than_input_falls_back_to_whole_decode():
    vae = _tiny_vae()
    torch.manual_seed(3)
    z = [torch.randn(Z_DIM, 2, 8, 8, device='cuda')]

    whole = vae.decode(z)[0]
    vae.enable_tiling(tile_size=64, overlap=4)
    tiled = vae.decode(z)[0]

    assert torch.equal(whole, tiled), 'a single tile must take the exact path'
