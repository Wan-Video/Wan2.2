# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Tests for failures that used to be silent.

The worst one: save_video wrapped its whole body in `except Exception` and
logged the failure at INFO, returning None. After a full generation run the CLI
would log "Saving generated video to ..." and then exit 0 having written
nothing -- indistinguishable from success in both the exit code and the log
level.

No model weights needed.
"""
import argparse

import pytest
import torch

from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


def _video_tensor(frames=4, h=16, w=16):
    # save_video expects [C, F, H, W] in [-1, 1]
    return torch.rand(3, frames, h, w) * 2 - 1


def test_save_video_writes_and_returns_path(tmp_path):
    target = tmp_path / 'ok.mp4'
    returned = save_video(_video_tensor(), save_file=str(target), fps=8)

    assert returned == str(target)
    assert target.exists() and target.stat().st_size > 0


def test_save_video_frame_axis_is_dim2(tmp_path):
    """
    save_video unbinds dim 2 as the frame axis, so callers must pass
    [B, C, F, H, W]. Passing the bare [C, F, H, W] the pipelines return splits
    on height and silently writes a video with the wrong geometry.
    """
    import imageio.v3 as iio
    frames, h, w = 4, 32, 48
    video = torch.rand(3, frames, h, w) * 2 - 1

    target = tmp_path / 'ok.mp4'
    save_video(tensor=video[None], save_file=str(target), fps=8, nrow=1,
               normalize=True, value_range=(-1, 1))
    got = iio.imread(str(target))
    assert got.shape[0] == frames, (
        'expected {} frames, got shape {}'.format(frames, got.shape))
    assert got.shape[1] == h and got.shape[2] == w


def test_save_video_raises_instead_of_returning_none(tmp_path):
    """A write into a non-existent directory must fail loudly."""
    target = tmp_path / 'no_such_dir' / 'out.mp4'

    with pytest.raises(Exception):
        save_video(_video_tensor(), save_file=str(target), fps=8)


def test_save_video_leaves_no_partial_file(tmp_path):
    """
    A failed write must not leave a truncated file sitting there looking like a
    finished render.
    """
    target = tmp_path / 'broken.mp4'

    class Exploding:
        """Passes the clamp/stack preprocessing, then fails during encode."""

        def __init__(self, inner):
            self._inner = inner

        def clamp(self, *a, **k):
            # stay wrapped, otherwise save_video rebinds to a real tensor and
            # never reaches unbind()
            return Exploding(self._inner.clamp(*a, **k))

        def unbind(self, *a, **k):
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        save_video(Exploding(_video_tensor()), save_file=str(target), fps=8)

    assert not target.exists()


def test_vae_decode_rejects_bad_input_loudly():
    """
    Wan2_2_VAE.encode/decode used to catch TypeError, log it at INFO and return
    None, so a bad call surfaced much later as a confusing 'NoneType' error.
    """
    from wan.modules.vae2_2 import Wan2_2_VAE

    vae = Wan2_2_VAE.__new__(Wan2_2_VAE)  # no weights, no device needed
    with pytest.raises(TypeError):
        Wan2_2_VAE.decode(vae, torch.zeros(1))
    with pytest.raises(TypeError):
        Wan2_2_VAE.encode(vae, torch.zeros(1))


def _args(**overrides):
    base = dict(
        task='ti2v-5B',
        ckpt_dir=None,
        prompt='a cat',
        image=None,
        audio=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        frame_num=None,
        base_seed=0,
        size='1280*704',
        dit_quant='none',
        vae_tile=False,
        t5_cpu=True,
        t5_fp32_compute=False,
        t5_batch=False,
        t5_cache=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _validate(args):
    from generate import _validate_args
    return _validate_args(args)


def test_missing_ckpt_dir_is_rejected():
    with pytest.raises(ValueError, match='ckpt_dir'):
        _validate(_args(ckpt_dir=None))


def test_nonexistent_ckpt_dir_is_rejected_up_front(tmp_path):
    """
    Previously only `is not None` was checked, so a typo'd path surfaced as a
    deep from_pretrained traceback after T5 and the VAE had already loaded.
    """
    with pytest.raises(ValueError, match='does not exist'):
        _validate(_args(ckpt_dir=str(tmp_path / 'typo')))


def test_frame_num_must_be_4n_plus_1(tmp_path):
    with pytest.raises(ValueError, match='4n\\+1'):
        _validate(_args(ckpt_dir=str(tmp_path), frame_num=80))


@pytest.mark.parametrize('frame_num', [1, 5, 49, 81, 121])
def test_valid_frame_nums_accepted(tmp_path, frame_num):
    _validate(_args(ckpt_dir=str(tmp_path), frame_num=frame_num))


def test_unsupported_size_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsupported size'):
        _validate(_args(ckpt_dir=str(tmp_path), size='64*64'))


def test_ti2v_accepts_480p(tmp_path):
    """
    TI2V-5B was whitelisted for 720p only, leaving smaller cards no lever at
    all. 480*832 and 832*480 both divide the 32x total compression.
    """
    for size in ('480*832', '832*480'):
        args = _args(ckpt_dir=str(tmp_path), size=size, frame_num=49)
        _validate(args)


def test_dit_quant_rejected_for_unsupported_tasks(tmp_path):
    """
    The other task-specific flags are silently ignored when they do not apply.
    The quantization flags refuse instead, so nobody thinks they quantized a
    model that was never touched.
    """
    with pytest.raises(ValueError, match='ti2v-5B only'):
        _validate(_args(ckpt_dir=str(tmp_path), task='t2v-A14B',
                        size='1280*720', dit_quant='fp8'))


def test_vae_tile_rejected_for_unsupported_tasks(tmp_path):
    with pytest.raises(ValueError, match='ti2v-5B only'):
        _validate(_args(ckpt_dir=str(tmp_path), task='t2v-A14B',
                        size='1280*720', vae_tile=True))


def test_quant_flags_accepted_for_ti2v(tmp_path):
    for mode in ('none', 'fp8', 'fp8_wo'):
        _validate(_args(ckpt_dir=str(tmp_path), frame_num=49, dit_quant=mode))
    _validate(_args(ckpt_dir=str(tmp_path), frame_num=49, vae_tile=True))


def test_t5_fp32_compute_requires_t5_cpu(tmp_path):
    """
    fp32 compute only helps the CPU path -- bf16 matmul is emulated there, not
    on the GPU. Asking for it without --t5_cpu is a mistake worth naming.
    """
    with pytest.raises(ValueError, match='t5_cpu'):
        _validate(_args(ckpt_dir=str(tmp_path), frame_num=49,
                        t5_fp32_compute=True, t5_cpu=False))


def test_t5_flags_rejected_for_unsupported_tasks(tmp_path):
    for kw in ({'t5_batch': True}, {'t5_cache': 2},
               {'t5_fp32_compute': True}):
        with pytest.raises(ValueError, match='ti2v-5B only'):
            _validate(_args(ckpt_dir=str(tmp_path), task='t2v-A14B',
                            size='1280*720', **kw))


def test_ti2v_5b_config_unchanged():
    """Guard the numbers the memory arithmetic in the tests above depends on."""
    cfg = WAN_CONFIGS['ti2v-5B']
    assert cfg.dim == 3072
    assert cfg.num_layers == 30
    assert cfg.vae_stride == (4, 16, 16)
    assert cfg.patch_size == (1, 2, 2)
