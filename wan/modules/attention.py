# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

FLASH_ATTN_AVAILABLE = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE

__all__ = [
    'flash_attention',
    'attention',
    'sdpa_attention',
    'FLASH_ATTN_AVAILABLE',
]

# Emit the "no flash-attn, using SDPA" notice once per process instead of on
# every one of the thousands of attention calls in a single video.
_SDPA_NOTICE_EMITTED = False


def _notify_sdpa_fallback():
    global _SDPA_NOTICE_EMITTED
    if not _SDPA_NOTICE_EMITTED:
        _SDPA_NOTICE_EMITTED = True
        warnings.warn(
            'flash-attn is not installed; falling back to PyTorch '
            'scaled_dot_product_attention. This is slower and uses more memory '
            'than flash-attn, but produces equivalent results.')


def _key_padding_mask(lk, k_lens, device):
    """
    Build a [B, 1, 1, Lk] boolean mask from per-sample key lengths, or None when
    nothing is actually padded.

    Returning None matters for speed: SDPA only dispatches to its fused flash
    kernel when attn_mask is None, so we avoid materializing a mask whenever the
    batch is not padded -- which is the single-GPU case in this repo, where
    seq_lens.max() == seq_len.
    """
    if k_lens is None:
        return None
    # k_lens is a CPU tensor everywhere in this repo, so min() is free. Guard
    # anyway so a device tensor costs at most one sync.
    if int(k_lens.min()) >= lk:
        return None
    idx = torch.arange(lk, device=device)
    return (idx.unsqueeze(0) < k_lens.to(device).unsqueeze(1))[:, None, None, :]


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """
    Same contract as `flash_attention`, implemented on PyTorch's
    scaled_dot_product_attention.

    Unlike the previous fallback this one honours `k_lens` by building an
    explicit key-padding mask, so padded positions are excluded exactly as
    flash-attn's varlen kernel excludes them.

    `q_lens` is not used for masking: rows past a sample's query length are
    computed and left for the caller to discard, which is what the flash path
    does as well.
    """
    lq, nq = q.size(1), q.size(2)
    lk, nk = k.size(1), k.size(2)
    out_dtype = q.dtype

    if window_size != (-1, -1):
        warnings.warn(
            'Sliding-window attention is not supported by the SDPA fallback; '
            'ignoring window_size={}.'.format(window_size))

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # [B, L, N, C] -> [B, N, L, C]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # grouped-query attention: replicate kv heads to match the query heads
    if nq != nk:
        assert nq % nk == 0, \
            'num_heads {} must be divisible by kv heads {}'.format(nq, nk)
        k = k.repeat_interleave(nq // nk, dim=1)
        v = v.repeat_interleave(nq // nk, dim=1)

    attn_mask = _key_padding_mask(lk, k_lens, q.device)

    if causal and attn_mask is not None:
        # is_causal and attn_mask are mutually exclusive, so fold the causal
        # triangle into the padding mask.
        causal_mask = torch.ones(
            lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)
        attn_mask = attn_mask & causal_mask
        causal = False

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale)

    # [B, N, Lq, C] -> [B, Lq, N, C]
    return out.transpose(1, 2).contiguous().type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.

    Falls back to PyTorch's scaled_dot_product_attention when flash-attn is not
    installed, so this is safe to call unconditionally.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    if not FLASH_ATTN_AVAILABLE:
        _notify_sdpa_fallback()
        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
