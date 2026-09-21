# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

import warnings

__all__ = [
    'flash_attention',
    'attention',
]

def vanilla_attn_varlen_replacement(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False # Note: This param is for FlashAttention's numerics
) -> torch.Tensor:
    """
    A vanilla PyTorch replacement for flash_attn.flash_attn_varlen_func.

    This function is designed for correctness and clarity, not performance.
    It will be significantly slower and use more memory than FlashAttention.

    Args:
        q, k, v: Input tensors in packed format, e.g., (total_tokens, num_heads, head_dim).
        cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths for Q and K.
        max_seqlen_q, max_seqlen_k: Maximum sequence lengths for Q and K.
        ... other attention parameters.

    Returns:
        A tensor in packed format (total_tokens_q, num_heads, head_dim),
        similar to the output of flash_attn_varlen_func.
    """
    # 1. Deconstruct Packed Tensors and `cu_seqlens`
    # cu_seqlens are cumulative, so get individual lengths by taking the difference.
    # e.g., [0, 10, 25, 30] -> [10, 15, 5]
    q_lens = torch.diff(cu_seqlens_q).cpu() # Move to CPU for list conversion
    k_lens = torch.diff(cu_seqlens_k).cpu()
    batch_size = len(q_lens)

    # Unpack Q, K, V from (total_tokens, ...) to (batch, max_len, ...)
    # This process is memory-intensive and is what FlashAttention avoids.
    unpacked_q = q.new_zeros(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
    unpacked_k = k.new_zeros(batch_size, max_seqlen_k, k.shape[1], k.shape[2])
    unpacked_v = v.new_zeros(batch_size, max_seqlen_k, v.shape[1], v.shape[2])

    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]

        unpacked_q[i, :q_lens[i], :, :] = q[start_q:end_q]
        unpacked_k[i, :k_lens[i], :, :] = k[start_k:end_k]
        unpacked_v[i, :k_lens[i], :, :] = v[start_k:end_k]

    # Transpose to the format expected by F.scaled_dot_product_attention:
    # (batch, num_heads, seq_len, head_dim)
    unpacked_q = unpacked_q.transpose(1, 2)
    unpacked_k = unpacked_k.transpose(1, 2)
    unpacked_v = unpacked_v.transpose(1, 2)

    # 2. Create the Attention Mask
    # This prevents attention to padding tokens.
    attn_mask = torch.ones(max_seqlen_q, max_seqlen_k, device=q.device, dtype=torch.bool)
    if causal:
        attn_mask = torch.tril(attn_mask)

    # Expand mask to handle the batch and head dimensions.
    # Final shape will be broadcastable to (batch_size, num_heads, max_seqlen_q, max_seqlen_k)
    attn_mask = attn_mask[None, None, :, :]

    # Create padding mask for keys (shape: [batch_size, 1, 1, max_seqlen_k])
    # This ensures queries don't attend to padded keys.
    key_padding_mask = torch.arange(max_seqlen_k, device=q.device)[None, :] < k_lens[:, None].to(q.device)
    attn_mask = attn_mask & key_padding_mask[:, None, None, :]


    # 3. Perform Scaled Dot-Product Attention
    # Use PyTorch 2.0's built-in function, which is the modern "vanilla" way.
    # It's optimized but still materializes the attention matrix if a backend
    # like FlashAttention isn't available.

    # Note: The built-in SDPA handles the key padding mask for us.
    # We only need to provide the causal flag.
    padded_output = F.scaled_dot_product_attention(
        query=unpacked_q,
        key=unpacked_k,
        value=unpacked_v,
        attn_mask=None, # The function is more efficient if it builds its own causal mask
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale
    )

    # Re-transpose to (batch, seq_len, num_heads, head_dim)
    padded_output = padded_output.transpose(1, 2)

    # 4. Re-pack the Output
    # The original function returns a packed tensor, so we must do the same.
    # We slice out the valid (non-padded) parts of the output and concatenate them.
    output_parts = [
        padded_output[i, :q_lens[i], :, :] for i in range(batch_size)
    ]
    packed_output = torch.cat(output_parts, dim=0)

    return packed_output


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
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

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
    elif FLASH_ATTN_2_AVAILABLE:
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
    else:
        x = vanilla_attn_varlen_replacement(
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
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
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
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
