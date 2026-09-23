# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Optional FP8 weight quantization for the Wan DiT.

Why this exists
---------------
TI2V-5B is ~5.0B parameters. At bf16 that is 9.31 GB of weights, which does not
fit in an 8 GB card. On Windows the driver silently spills the remainder to
system RAM instead of failing, so generation "works" but every denoising step
streams gigabytes over PCIe -- measured at ~1% GPU utilization and 32.6 s/step
on an RTX 4060 Laptop. The bottleneck is weight residency, not arithmetic.

Quantizing the transformer-block Linear weights to FP8 halves them, bringing the
model to roughly 4.7 GB so it stays resident, and on Ada (sm_89) FP8 matmul
through `torch._scaled_mm` additionally measures ~2x faster than bf16 at these
shapes.

Accuracy
--------
This is NOT free. sm_89 only supports per-tensor ("TensorWise") scaling in
`_scaled_mm` -- per-row scaling raises
`Per-row scaling is not supported for this platform!` -- and per-tensor dynamic
W8A8 measures ~3.75% relative error on a single 3072x3072 matmul, against ~2.65%
for weight-only with a per-output-channel scale.

Diffusion amplifies small perturbations, so both modes visibly change the final
video. They are opt-in for that reason, and `--dit_quant none` remains the
default. Always measure the generated frames, not just tensor error.

Modes
-----
fp8     W8A8. Weights and activations both quantized per tensor, matmul runs in
        FP8 on the tensor cores. Fastest, least accurate.
fp8_wo  W8A16. Only the weights are stored in FP8, with a per-output-channel
        scale; they are dequantized to bf16 per call and the matmul runs in
        bf16. Same memory saving, better accuracy, no matmul speedup.
"""
import logging

import torch
import torch.nn as nn

__all__ = [
    'QUANT_MODES',
    'Fp8Linear',
    'fp8_supported',
    'quantize_dit_',
]

QUANT_MODES = ('none', 'fp8', 'fp8_wo')

# e4m3 saturates at 448
_E4M3_MAX = 448.0


def fp8_supported():
    """(ok, reason) for FP8 on the current device."""
    if not torch.cuda.is_available():
        return False, 'no CUDA device'
    if not hasattr(torch, 'float8_e4m3fn'):
        return False, 'this torch build has no float8_e4m3fn'
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        return False, ('FP8 needs compute capability >= 8.9 (Ada/Hopper), '
                       'this device is sm_{}{}'.format(major, minor))
    return True, ''


def _quantize_per_tensor(w):
    """FP8 tensor + scalar scale such that w ~= q.float() * scale."""
    amax = w.detach().float().abs().amax().clamp(min=1e-12)
    scale = (amax / _E4M3_MAX).float()
    q = (w.detach().float() / scale).clamp(-_E4M3_MAX,
                                           _E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(())


def _quantize_per_channel(w):
    """FP8 tensor + [out, 1] scale. More accurate, but bf16 matmul only."""
    amax = w.detach().float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / _E4M3_MAX).float()
    q = (w.detach().float() / scale).clamp(-_E4M3_MAX,
                                           _E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


class Fp8Linear(nn.Module):
    """Drop-in replacement for nn.Linear holding an FP8 weight."""

    def __init__(self, linear, mode, compute_dtype=torch.bfloat16):
        super().__init__()
        if mode not in ('fp8', 'fp8_wo'):
            raise ValueError('unsupported quant mode: {}'.format(mode))
        self.mode = mode
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype

        w = linear.weight
        if mode == 'fp8':
            q, scale = _quantize_per_tensor(w)
        else:
            q, scale = _quantize_per_channel(w)

        # buffers, not parameters: these are inference-only and must not be
        # picked up by optimizers or FSDP flattening
        self.register_buffer('weight_fp8', q, persistent=False)
        self.register_buffer('weight_scale', scale, persistent=False)
        if linear.bias is not None:
            self.register_buffer(
                'bias', linear.bias.detach().to(compute_dtype), persistent=False)
        else:
            self.bias = None

    def extra_repr(self):
        return 'in_features={}, out_features={}, mode={}'.format(
            self.in_features, self.out_features, self.mode)

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).to(self.compute_dtype)

        if self.mode == 'fp8_wo':
            # dequantize the weight and use an ordinary bf16 matmul
            w = (self.weight_fp8.to(torch.float32) *
                 self.weight_scale).to(self.compute_dtype)
            out = torch.nn.functional.linear(x2d, w, self.bias)
        else:
            # dynamic per-tensor activation scale; kept on device so this does
            # not introduce a host sync
            amax = x2d.detach().float().abs().amax().clamp(min=1e-12)
            x_scale = (amax / _E4M3_MAX).float().reshape(())
            xq = (x2d.float() / x_scale).clamp(
                -_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                xq,
                self.weight_fp8.t(),
                scale_a=x_scale,
                scale_b=self.weight_scale,
                bias=self.bias,
                out_dtype=self.compute_dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


def quantize_dit_(model, mode, device=None, compute_dtype=torch.bfloat16):
    """
    Replace the Linear layers inside the transformer blocks with Fp8Linear,
    in place. Returns a dict of statistics.

    Only `model.blocks[*]` Linears are touched. The patch embedding, text and
    time embeddings, the final head, every norm and the modulation parameters
    stay in their original dtype -- they are a small share of the parameters
    and the numerically sensitive part of the network.

    Conversion happens one layer at a time, moving each weight to `device`
    before quantizing, so peak memory never holds a second full copy of the
    model.
    """
    if mode == 'none':
        return {'mode': 'none', 'converted': 0}
    if mode not in QUANT_MODES:
        raise ValueError('unknown quant mode {!r}, expected one of {}'.format(
            mode, ', '.join(QUANT_MODES)))

    ok, reason = fp8_supported()
    if not ok:
        raise RuntimeError(
            'FP8 quantization requested (--dit_quant {}) but it is not '
            'available: {}'.format(mode, reason))

    if not hasattr(model, 'blocks'):
        raise AttributeError(
            'quantize_dit_ expects a model with a .blocks list')

    device = device or torch.device('cuda')
    converted = 0
    before = sum(p.numel() * p.element_size() for p in model.parameters())

    for block in model.blocks:
        for parent_name, parent in list(block.named_modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                child = child.to(device)
                setattr(parent, child_name,
                        Fp8Linear(child, mode, compute_dtype))
                del child
                converted += 1

    torch.cuda.empty_cache()
    after = sum(p.numel() * p.element_size() for p in model.parameters())
    after += sum(b.numel() * b.element_size() for b in model.buffers())

    stats = {
        'mode': mode,
        'converted': converted,
        'bytes_before': before,
        'bytes_after': after,
    }
    logging.info(
        'FP8 quantization (%s): converted %d Linear layers, '
        'weights %.2f GB -> %.2f GB', mode, converted, before / 1024**3,
        after / 1024**3)
    return stats
