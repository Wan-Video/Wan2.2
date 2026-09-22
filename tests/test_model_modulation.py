# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for the modulation change in WanAttentionBlock / Head.

The blocks used to build `(self.modulation.unsqueeze(0) + e)` in full before
chunking it. `e` is [B, L, 6, C] in float32, so that allocated an entire extra
copy of it on every block -- for TI2V-5B at 704x1280 with 121 frames,
L = 27,280 and the copy is 27280 * 6 * 3072 * 4 B = 2.0 GB, once per block,
30 blocks deep. Indexing into the modulation parameter instead keeps only the
[B, L, C] operand actually in use.

These tests pin down that the rewrite is arithmetically identical and that it
really does allocate less. No model weights needed.
"""
import types

import pytest
import torch

from wan.modules.model import Head, WanModel

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')


def test_modulation_indexing_identity():
    """
    The exact algebraic identity the rewrite relies on, on CPU:
        (modulation.unsqueeze(0) + e).chunk(6, dim=2)[i].squeeze(2)
        == e[:, :, i] + modulation[:, i]
    """
    torch.manual_seed(0)
    b, l, c = 2, 7, 16
    e = torch.randn(b, l, 6, c)
    modulation = torch.randn(1, 6, c)

    old = (modulation.unsqueeze(0) + e).chunk(6, dim=2)
    for i in range(6):
        torch.testing.assert_close(old[i].squeeze(2), e[:, :, i] + modulation[:, i])


def test_head_modulation_identity():
    """Same identity for Head, where e is [B, L, C] and modulation is [1, 2, C]."""
    torch.manual_seed(0)
    b, l, c = 2, 7, 16
    e = torch.randn(b, l, c)
    modulation = torch.randn(1, 2, c)

    old = (modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
    for i in range(2):
        torch.testing.assert_close(old[i].squeeze(2), e + modulation[:, i])


def _old_block_forward(self, x, e, seq_lens, grid_sizes, freqs, context,
                       context_lens):
    """Verbatim copy of the pre-change WanAttentionBlock.forward body."""
    assert e.dtype == torch.float32
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        seq_lens, grid_sizes, freqs)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x

    return cross_attn_ffn(x, context, context_lens, e)


def _tiny_model():
    """A structurally real but small TI2V-shaped WanModel."""
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
        eps=1e-6,
    ).eval().requires_grad_(False)


def _block_inputs(model, device, f=2, h=4, w=4):
    torch.manual_seed(1234)
    dim = model.dim
    seq_len = f * h * w
    x = torch.randn(1, seq_len, dim, device=device)
    e = torch.randn(1, seq_len, 6, dim, device=device, dtype=torch.float32)
    seq_lens = torch.tensor([seq_len])
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    context = torch.randn(1, 8, dim, device=device)
    return dict(
        x=x,
        e=e,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=model.freqs.to(device),
        context=context,
        context_lens=None)


@cuda_only
def test_block_output_unchanged():
    """The rewritten block must produce the same numbers as the old one."""
    model = _tiny_model().to('cuda')
    block = model.blocks[0]
    inputs = _block_inputs(model, 'cuda')

    with torch.no_grad():
        new_out = block(**inputs)

        old_block = model.blocks[0]
        old_block.forward = types.MethodType(_old_block_forward, old_block)
        old_out = old_block(**inputs)

    torch.testing.assert_close(new_out, old_out, atol=1e-5, rtol=1e-5)


@cuda_only
def test_block_allocates_less_than_the_old_formulation():
    """
    Directly measure what the change was for: peak allocation during one block
    forward. Sized so the modulation tensor dominates.
    """
    model = _tiny_model().to('cuda')
    inputs = _block_inputs(model, 'cuda', f=8, h=16, w=16)

    def peak(block):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            block(**inputs)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    new_peak = peak(model.blocks[0])

    old_block = model.blocks[1]
    old_block.load_state_dict(model.blocks[0].state_dict())
    old_block.forward = types.MethodType(_old_block_forward, old_block)
    old_peak = peak(old_block)

    assert new_peak < old_peak, (
        'expected the indexed modulation to allocate less; '
        'new={} old={}'.format(new_peak, old_peak))


@cuda_only
def test_head_output_unchanged():
    torch.manual_seed(0)
    dim, out_dim, patch_size = 64, 16, (1, 2, 2)
    head = Head(dim, out_dim, patch_size).to('cuda').eval().requires_grad_(False)

    x = torch.randn(1, 40, dim, device='cuda')
    e = torch.randn(1, 40, dim, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        new_out = head(x, e)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            chunks = (head.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            old_out = head.head(
                head.norm(x) * (1 + chunks[1].squeeze(2)) + chunks[0].squeeze(2))

    torch.testing.assert_close(new_out, old_out, atol=1e-5, rtol=1e-5)
