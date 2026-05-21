# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Base class for all Wan video generation pipelines.

Extracts the common initialization, model configuration, text encoding,
and sampling utilities shared across T2V, I2V, TI2V, S2V, and Animate.
"""
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.t5 import T5EncoderModel
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanPipelineBase:
    """Base class for Wan video generation pipelines.

    Provides shared logic for:
    - T5 text encoder loading
    - VAE loading (subclass selects version)
    - DiT model configuration (FSDP, sequence parallel, dtype conversion)
    - Text encoding (prompt + negative prompt)
    - Sampling scheduler creation (UniPC / DPM++)
    - Model offloading utilities
    - Distributed barrier / cleanup
    """

    # Subclasses should set these or override __init__ completely.
    _use_dual_expert = False  # Set True for T2V-A14B / I2V-A14B

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.boundary = getattr(config, 'boundary', None)

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)

        # Text encoder (shared across all pipelines)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # VAE – subclass is responsible for setting self.vae before or after
        # calling super().__init__(), or override _init_vae().
        self._init_vae(config, checkpoint_dir)

        # DiT model(s) – subclass is responsible for setting self.model
        # (single-expert) or self.low_noise_model / self.high_noise_model
        # (dual-expert).  Call _init_dit() in subclass after super().__init__().
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.sp_size = get_world_size() if use_sp else 1
        self.sample_neg_prompt = config.sample_neg_prompt

    def _init_vae(self, config, checkpoint_dir):
        """Override in subclass to select VAE version (2.1 or 2.2)."""
        raise NotImplementedError("Subclass must implement _init_vae()")

    # ------------------------------------------------------------------
    # Model configuration helpers
    # ------------------------------------------------------------------

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """Configure a DiT model: eval mode, SP hooks, FSDP, dtype, device."""
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    # ------------------------------------------------------------------
    # Dual-expert model switching (T2V-A14B / I2V-A14B)
    # ------------------------------------------------------------------

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        """Return the active model for the current timestep (dual-expert)."""
        if t.item() >= boundary:
            required, offload = 'high_noise_model', 'low_noise_model'
        else:
            required, offload = 'low_noise_model', 'high_noise_model'

        if offload_model or self.init_on_cpu:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            off_model = getattr(self, offload)
            req_model = getattr(self, required)
            is_fsdp = isinstance(off_model, FSDP)
            if not is_fsdp:
                if next(off_model.parameters()).device.type == 'cuda':
                    off_model.to('cpu')
                if next(req_model.parameters()).device.type == 'cpu':
                    req_model.to(self.device)

        return getattr(self, required)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_text(self, prompt, neg_prompt, offload_model=True):
        """Encode prompt and negative prompt with T5.

        Returns:
            (context, context_null) – both are lists of tensors on self.device.
        """
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([neg_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([prompt], torch.device('cpu'))
            context_null = self.text_encoder([neg_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        return context, context_null

    # ------------------------------------------------------------------
    # Sampling scheduler
    # ------------------------------------------------------------------

    def _create_scheduler(self, sample_solver, sampling_steps, shift):
        """Create a flow-matching sampling scheduler and timesteps.

        Returns:
            (sample_scheduler, timesteps)
        """
        if sample_solver == 'unipc':
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == 'dpm++':
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")
        return scheduler, timesteps

    # ------------------------------------------------------------------
    # Seed handling
    # ------------------------------------------------------------------

    @staticmethod
    def _make_seed(seed):
        """Return a deterministic or random seed + generator."""
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        gen = torch.Generator()
        gen.manual_seed(seed)
        return seed, gen

    @staticmethod
    def _make_seed_on_device(seed, device):
        """Return a deterministic or random seed + CUDA generator."""
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return seed, gen

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    def _broadcast_seed(self, seed):
        """Broadcast seed from rank 0 to all ranks."""
        if dist.is_initialized():
            seed_list = [seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(seed_list, src=0)
            return seed_list[0]
        return seed

    def _distributed_barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def _distributed_cleanup(self, offload_model=True):
        """Final cleanup: garbage collect, sync, destroy process group."""
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Context manager for no-sync (FSDP gradient sync suppression)
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def _noop_no_sync():
        yield

    def _get_no_sync(self, model):
        return getattr(model, 'no_sync', self._noop_no_sync)

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_guide_scale(guide_scale):
        """Ensure guide_scale is a 2-tuple (low_noise, high_noise)."""
        if isinstance(guide_scale, (int, float)):
            return (float(guide_scale), float(guide_scale))
        return tuple(float(x) for x in guide_scale)

    def _compute_seq_len(self, target_shape, patch_size, sp_size):
        """Compute padded sequence length for the DiT."""
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (patch_size[1] * patch_size[2])
            * target_shape[1]
            / sp_size
        ) * sp_size
        return seq_len
