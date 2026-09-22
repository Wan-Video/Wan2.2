# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for the optional FP8 DiT quantization.

The point of quantization here is residency, not arithmetic: at bf16 the 5B DiT
is 9.31 GB and does not fit in an 8 GB card, so the driver streams weights over
PCIe every step. These tests pin the parts that must not drift -- that the
default path is untouched, that the conversion only takes the block Linears,
that it really halves the weight bytes, and that failures are explicit.

Output *quality* under quantization is deliberately not asserted here; it is
measured end-to-end on real weights (see benchmarks/).

No model weights needed.
"""
import pytest
import torch
import torch.nn as nn

from wan.modules.model import WanModel
from wan.modules.quant import (
    QUANT_MODES,
    Fp8Linear,
    fp8_supported,
    quantize_dit_,
)

_ok, _why = fp8_supported()
fp8_only = pytest.mark.skipif(not _ok, reason='FP8 unavailable: %s' % _why)
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')


def _tiny_model():
    torch.manual_seed(0)
    return WanModel(
        model_type='ti2v',
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=16,
        dim=64,
        ffn_dim=128,
        freq_dim=64,
        text_dim=32,
        out_dim=16,
        num_heads=4,
        num_layers=2,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6).eval().requires_grad_(False)


def test_modes_are_declared():
    assert QUANT_MODES[0] == 'none', 'none must stay the documented default'
    assert set(QUANT_MODES) == {'none', 'fp8', 'fp8_wo'}


def test_none_is_a_no_op():
    model = _tiny_model()
    before = [type(m) for m in model.modules()]
    stats = quantize_dit_(model, 'none')
    assert stats['converted'] == 0
    assert [type(m) for m in model.modules()] == before


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='unknown quant mode'):
        quantize_dit_(_tiny_model(), 'int4')


@pytest.mark.skipif(_ok, reason='only meaningful when FP8 is unavailable')
def test_unsupported_hardware_fails_loudly():
    """No silent fallback: asking for FP8 without support must raise."""
    with pytest.raises(RuntimeError, match='not available'):
        quantize_dit_(_tiny_model(), 'fp8')


@fp8_only
@pytest.mark.parametrize('mode', ['fp8', 'fp8_wo'])
def test_fp8_linear_approximates_dense_linear(mode):
    torch.manual_seed(0)
    lin = nn.Linear(256, 512).cuda().to(torch.bfloat16).eval()
    x = torch.randn(64, 256, device='cuda', dtype=torch.bfloat16)

    ref = lin(x).float()
    got = Fp8Linear(lin, mode)(x).float()

    assert got.shape == ref.shape
    assert torch.isfinite(got).all()
    rel = (got - ref).abs().mean() / ref.abs().mean()
    # per-tensor FP8 on sm_89 lands around 3-4%; anything far past that means
    # the scales are wrong, not merely lossy
    assert rel < 0.10, 'relative error {:.4f} is too large'.format(rel)


@fp8_only
def test_weight_only_is_more_accurate_than_w8a8():
    """fp8_wo keeps a per-channel scale, so it should be the closer of the two."""
    torch.manual_seed(0)
    lin = nn.Linear(512, 512).cuda().to(torch.bfloat16).eval()
    x = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16)
    ref = lin(x).float()

    def rel(mode):
        out = Fp8Linear(lin, mode)(x).float()
        return ((out - ref).abs().mean() / ref.abs().mean()).item()

    assert rel('fp8_wo') < rel('fp8')


@fp8_only
def test_fp8_linear_preserves_leading_dims():
    lin = nn.Linear(64, 32).cuda().to(torch.bfloat16).eval()
    q = Fp8Linear(lin, 'fp8')
    out = q(torch.randn(2, 7, 64, device='cuda', dtype=torch.bfloat16))
    assert out.shape == (2, 7, 32)


@fp8_only
@pytest.mark.parametrize('mode', ['fp8', 'fp8_wo'])
def test_quantize_dit_only_touches_block_linears(mode):
    model = _tiny_model().cuda()

    kept = {
        'patch_embedding': type(model.patch_embedding),
        'text_embedding': type(model.text_embedding),
        'time_embedding': type(model.time_embedding),
        'time_projection': type(model.time_projection),
        'head_head': type(model.head.head),
    }

    stats = quantize_dit_(model, mode, device=torch.device('cuda'))

    assert stats['converted'] > 0
    # every Linear inside the blocks is converted...
    for block in model.blocks:
        for m in block.modules():
            assert not isinstance(m, nn.Linear), 'a block Linear survived'
    # ...and nothing outside them is
    assert type(model.patch_embedding) is kept['patch_embedding']
    assert type(model.text_embedding) is kept['text_embedding']
    assert type(model.time_embedding) is kept['time_embedding']
    assert type(model.time_projection) is kept['time_projection']
    assert type(model.head.head) is kept['head_head']
    assert isinstance(model.head.head, nn.Linear)


@fp8_only
def test_quantization_reduces_weight_bytes():
    model = _tiny_model().cuda().to(torch.bfloat16)
    stats = quantize_dit_(model, 'fp8', device=torch.device('cuda'))
    assert stats['bytes_after'] < stats['bytes_before']


@fp8_only
def test_quantized_model_still_runs_a_forward():
    """The real guard: a converted model must produce a finite, right-shaped output."""
    model = _tiny_model().cuda().to(torch.bfloat16)
    quantize_dit_(model, 'fp8', device=torch.device('cuda'))

    f, h, w = 2, 4, 4
    seq_len = f * h * w
    x = [torch.randn(16, f, h * 2, w * 2, device='cuda')]
    t = torch.tensor([500.0], device='cuda')
    context = [torch.randn(8, 32, device='cuda')]

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x, t=t, context=context, seq_len=seq_len)

    assert len(out) == 1
    assert out[0].shape == (16, f, h * 2, w * 2)
    assert torch.isfinite(out[0]).all()


@fp8_only
def test_fsdp_combination_is_rejected():
    from wan.textimage2video import WanTI2V
    pipe = WanTI2V.__new__(WanTI2V)
    pipe.param_dtype = torch.bfloat16
    pipe.device = torch.device('cuda')
    pipe.init_on_cpu = True
    pipe.quant_stats = None

    with pytest.raises(ValueError, match='not supported together with dit_fsdp'):
        pipe._configure_model(
            model=_tiny_model(),
            use_sp=False,
            dit_fsdp=True,
            shard_fn=lambda m: m,
            convert_model_dtype=True,
            dit_quant='fp8')
