# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
End-to-end run of the real WanTI2V pipeline with tiny random weights.

This drives the actual production call sequence -- the sampler loop in
WanTI2V.t2v/i2v, WanModel.forward (block modulation and rope), the
Wan2_2_VAE decode, and save_video -- rather than re-implementing any of it in
the test. It needs no checkpoint, so it catches plumbing regressions that the
narrower unit tests cannot.

The real TI2V-5B weights are ~23 GB, so a genuine generation cannot run in CI.
"""
import os

import pytest
import torch

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE, WanVAE_
from wan.textimage2video import WanTI2V
from wan.utils.utils import save_video

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device')

Z_DIM = 16
VAE_DIM = 32
DIM = 64
SIZE = (64, 64)  # (W, H); both must divide vae_stride(16) * patch_size(2)
FRAME_NUM = 5  # 4n+1


class _StubT5:
    """Stands in for the 11 GB UMT5 encoder."""

    class _Model:

        def to(self, *a, **k):
            return self

        def cpu(self):
            return self

    model = _Model()

    def __call__(self, texts, device):
        g = torch.Generator(device='cpu').manual_seed(7)
        return [torch.randn(24, 32, generator=g).to(device) for _ in texts]


def _build_pipeline():
    cfg = WAN_CONFIGS['ti2v-5B']
    torch.manual_seed(0)

    vae_model = WanVAE_(
        dim=VAE_DIM,
        dec_dim=VAE_DIM,
        z_dim=Z_DIM,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0).eval().requires_grad_(False).cuda()

    vae = Wan2_2_VAE.__new__(Wan2_2_VAE)
    vae.dtype = torch.float32
    vae.device = 'cuda'
    vae.scale = [
        torch.zeros(Z_DIM, device='cuda'),
        1.0 / torch.ones(Z_DIM, device='cuda')
    ]
    vae.model = vae_model

    model = WanModel(
        model_type='ti2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=Z_DIM,
        dim=DIM,
        ffn_dim=DIM * 2,
        freq_dim=64,
        text_dim=32,
        out_dim=Z_DIM,
        num_heads=4,
        num_layers=2,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6).eval().requires_grad_(False).cuda()

    pipe = WanTI2V.__new__(WanTI2V)
    pipe.device = torch.device('cuda')
    pipe.config = cfg
    pipe.rank = 0
    pipe.t5_cpu = False
    pipe.t5_batch = False
    pipe.init_on_cpu = False
    pipe.num_train_timesteps = cfg.num_train_timesteps
    pipe.param_dtype = cfg.param_dtype
    pipe.text_encoder = _StubT5()
    pipe.vae = vae
    pipe.vae_stride = cfg.vae_stride
    pipe.patch_size = cfg.patch_size
    pipe.model = model
    pipe.sp_size = 1
    pipe.sample_neg_prompt = cfg.sample_neg_prompt
    return pipe


@pytest.fixture(scope='module')
def pipeline():
    return _build_pipeline()


def _generate(pipe, solver='unipc', seed=42):
    return pipe.t2v(
        input_prompt='a cat',
        size=SIZE,
        frame_num=FRAME_NUM,
        sample_solver=solver,
        sampling_steps=3,
        guide_scale=5.0,
        seed=seed,
        offload_model=False)


@cuda_only
def test_t2v_runs_end_to_end(pipeline):
    video = _generate(pipeline)

    assert video.dim() == 4 and video.size(0) == 3, video.shape
    assert video.shape[2:] == (SIZE[1], SIZE[0]), video.shape
    assert torch.isfinite(video).all(), 'pipeline produced non-finite pixels'
    # the VAE decode clamps to [-1, 1]
    assert video.min() >= -1.0 and video.max() <= 1.0


@cuda_only
def test_t2v_is_deterministic_for_a_seed(pipeline):
    """
    Guards the sampler loop: the hoisted mask terms and the reused timestep
    tensor must not perturb the RNG stream or carry state between runs.
    """
    a = _generate(pipeline, seed=42)
    b = _generate(pipeline, seed=42)
    assert torch.equal(a, b)

    c = _generate(pipeline, seed=43)
    assert not torch.equal(a, c), 'different seeds produced identical output'


@cuda_only
@pytest.mark.parametrize('solver', ['unipc', 'dpm++'])
def test_both_solvers_run(pipeline, solver):
    video = _generate(pipeline, solver=solver)
    assert torch.isfinite(video).all()


@cuda_only
def test_unknown_solver_is_rejected(pipeline):
    with pytest.raises(NotImplementedError):
        _generate(pipeline, solver='nope')


@cuda_only
def test_generated_video_is_written_to_disk(pipeline, tmp_path):
    """
    The full path a user actually takes: generate, then save.

    save_video expects [B, C, F, H, W] and unbinds dim 2 as the frame axis, so
    the [C, F, H, W] the pipeline returns has to be unsqueezed first -- exactly
    what generate.py does.
    """
    video = _generate(pipeline)
    target = tmp_path / 'out.mp4'

    written = save_video(
        tensor=video[None], save_file=str(target), fps=8, nrow=1,
        normalize=True, value_range=(-1, 1))

    assert written == str(target)
    assert os.path.getsize(written) > 0
