# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Compare two generated videos frame by frame.

Used to decide whether an approximate mode (quantization, tiled VAE decode) is
acceptable. A speedup is not a result on its own -- the output has to be
measured against the reference it claims to replace.
"""
import argparse
import json

import imageio.v3 as iio
import numpy as np


def _gaussian_window(size=11, sigma=1.5):
    g = np.exp(-((np.arange(size) - size // 2)**2) / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def _filter2(img, win):
    """Valid-mode 2-D correlation, no scipy dependency."""
    kh, kw = win.shape
    h, w = img.shape
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((0, 0))
    strided = np.lib.stride_tricks.as_strided(
        img,
        shape=(oh, ow, kh, kw),
        strides=img.strides * 2,
        writeable=False)
    return np.einsum('ijkl,kl->ij', strided, win)


def ssim(a, b, data_range=255.0):
    """Mean SSIM over the luma channel, standard 11x11 gaussian window."""
    win = _gaussian_window()
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2

    vals = []
    for fa, fb in zip(a, b):
        # Rec.601 luma
        la = (0.299 * fa[..., 0] + 0.587 * fa[..., 1] +
              0.114 * fa[..., 2]).astype(np.float64)
        lb = (0.299 * fb[..., 0] + 0.587 * fb[..., 1] +
              0.114 * fb[..., 2]).astype(np.float64)

        mu_a = _filter2(la, win)
        mu_b = _filter2(lb, win)
        maa, mbb, mab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sa = _filter2(la * la, win) - maa
        sb = _filter2(lb * lb, win) - mbb
        sab = _filter2(la * lb, win) - mab

        num = (2 * mab + c1) * (2 * sab + c2)
        den = (maa + mbb + c1) * (sa + sb + c2)
        vals.append(float(np.mean(num / den)))
    return float(np.mean(vals))


def temporal_stats(v):
    """Mean absolute frame-to-frame delta -- a proxy for motion/flicker."""
    if len(v) < 2:
        return 0.0
    d = np.abs(v[1:].astype(np.float64) - v[:-1].astype(np.float64))
    return float(d.mean())


def _load(path):
    """
    Load a .npy pipeline tensor [C, F, H, W] in [-1, 1], or an .mp4.

    Prefer the .npy: comparing two H.264 encodes measures the codec as much as
    the model.
    """
    if path.endswith('.npy'):
        v = np.load(path).astype(np.float64)          # [C, F, H, W], [-1, 1]
        v = np.transpose(v, (1, 2, 3, 0))             # [F, H, W, C]
        return np.clip((v + 1) / 2 * 255.0, 0, 255)
    return iio.imread(path).astype(np.float64)


def compare(ref_path, test_path):
    a = _load(ref_path)
    b = _load(test_path)
    if a.shape != b.shape:
        raise ValueError('shape mismatch: {} vs {}'.format(a.shape, b.shape))

    diff = np.abs(a - b)
    mse = float(((a - b)**2).mean())
    psnr = float('inf') if mse == 0 else 10 * np.log10(255.0**2 / mse)

    return {
        'reference': ref_path,
        'test': test_path,
        'frames': int(a.shape[0]),
        'identical': bool(np.array_equal(a, b)),
        'mean_abs_diff_255': round(float(diff.mean()), 4),
        'mean_abs_diff_pct': round(float(diff.mean()) / 255 * 100, 3),
        'max_abs_diff_255': round(float(diff.max()), 1),
        'pct_pixels_gt_2_255': round(float((diff > 2).mean()) * 100, 2),
        'psnr_db': round(psnr, 2),
        'ssim': round(ssim(a, b), 5),
        'temporal_delta_ref': round(temporal_stats(a), 3),
        'temporal_delta_test': round(temporal_stats(b), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference')
    ap.add_argument('test', nargs='+')
    ap.add_argument('--json_out', default=None)
    args = ap.parse_args()

    out = [compare(args.reference, t) for t in args.test]
    for r in out:
        print(json.dumps(r, indent=2))
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print('wrote', args.json_out)


if __name__ == '__main__':
    main()
