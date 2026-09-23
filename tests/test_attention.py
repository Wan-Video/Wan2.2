# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for the attention dispatcher.

These matter because every model in the repo calls `flash_attention` directly.
Before the SDPA fallback was wired into it, that function ended in a bare
`assert FLASH_ATTN_2_AVAILABLE`, so on any machine without a flash-attn build
(notably Windows, which has no official wheel) inference died with an
AssertionError carrying no message -- while a perfectly good SDPA fallback sat
unreachable a few lines below.

None of this needs model weights.
"""
import math

import pytest
import torch

from wan.modules.attention import (
    FLASH_ATTN_AVAILABLE,
    attention,
    flash_attention,
    sdpa_attention,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')


def _reference_attention(q, k, v, k_lens=None):
    """Plain fp32 softmax attention, used as ground truth."""
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if k_lens is not None:
        lk = k.size(2)
        idx = torch.arange(lk, device=q.device)
        keep = (idx.unsqueeze(0) < k_lens.to(q.device).unsqueeze(1))
        scores = scores.masked_fill(~keep[:, None, None, :], float('-inf'))

    return (scores.softmax(dim=-1) @ v).transpose(1, 2)


@cuda_only
def test_sdpa_matches_reference_without_padding():
    torch.manual_seed(0)
    b, lq, lk, n, c = 2, 24, 24, 4, 32
    q = torch.randn(b, lq, n, c, device='cuda')
    k = torch.randn(b, lk, n, c, device='cuda')
    v = torch.randn(b, lk, n, c, device='cuda')

    out = sdpa_attention(q, k, v)
    ref = _reference_attention(q, k, v)

    assert out.shape == (b, lq, n, c)
    # bf16 compute inside sdpa_attention, so the tolerance is bf16-sized
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_sdpa_honours_k_lens():
    """
    The previous fallback dropped k_lens with only a warning, so padded key
    positions were attended to. flash-attn's varlen kernel excludes them, so the
    two paths disagreed whenever the batch was actually padded.
    """
    torch.manual_seed(0)
    b, lq, lk, n, c = 2, 16, 20, 4, 32
    q = torch.randn(b, lq, n, c, device='cuda')
    k = torch.randn(b, lk, n, c, device='cuda')
    v = torch.randn(b, lk, n, c, device='cuda')
    k_lens = torch.tensor([20, 11])

    out = sdpa_attention(q, k, v, k_lens=k_lens)
    ref = _reference_attention(q, k, v, k_lens=k_lens)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # sample 1 must genuinely ignore the padding: changing the padded keys must
    # not change its output.
    k2 = k.clone()
    v2 = v.clone()
    k2[1, 11:] = torch.randn_like(k2[1, 11:])
    v2[1, 11:] = torch.randn_like(v2[1, 11:])
    out2 = sdpa_attention(q, k2, v2, k_lens=k_lens)
    torch.testing.assert_close(out[1].float(), out2[1].float(),
                               atol=2e-2, rtol=2e-2)


@cuda_only
def test_unpadded_k_lens_take_the_fast_path():
    """
    When nothing is padded we must pass attn_mask=None, otherwise SDPA cannot
    dispatch to its fused kernel. Equivalent results, very different speed.
    """
    from wan.modules.attention import _key_padding_mask

    assert _key_padding_mask(16, None, 'cuda') is None
    assert _key_padding_mask(16, torch.tensor([16, 16]), 'cuda') is None
    assert _key_padding_mask(16, torch.tensor([16, 9]), 'cuda') is not None


@cuda_only
def test_flash_attention_falls_back_instead_of_asserting():
    """
    The regression guard: flash_attention must return a usable tensor whether or
    not flash-attn is installed, because model.py calls it unconditionally.
    """
    torch.manual_seed(0)
    b, lq, n, c = 1, 32, 4, 32
    q = torch.randn(b, lq, n, c, device='cuda')
    k = torch.randn(b, lq, n, c, device='cuda')
    v = torch.randn(b, lq, n, c, device='cuda')

    out = flash_attention(q, k, v, k_lens=torch.tensor([lq]))
    assert out.shape == (b, lq, n, c)
    assert torch.isfinite(out).all()

    # `attention` is just the documented alias and must agree with it
    out2 = attention(q, k, v, k_lens=torch.tensor([lq]))
    torch.testing.assert_close(out.float(), out2.float(),
                               atol=2e-2, rtol=2e-2)


@cuda_only
@pytest.mark.skipif(FLASH_ATTN_AVAILABLE,
                    reason='only meaningful when flash-attn is absent')
def test_grouped_query_heads_supported_in_fallback():
    torch.manual_seed(0)
    b, l, c = 1, 16, 32
    q = torch.randn(b, l, 8, c, device='cuda')
    k = torch.randn(b, l, 2, c, device='cuda')
    v = torch.randn(b, l, 2, c, device='cuda')

    out = sdpa_attention(q, k, v)
    assert out.shape == (b, l, 8, c)
    assert torch.isfinite(out).all()
