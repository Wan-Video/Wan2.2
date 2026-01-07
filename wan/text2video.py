# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# We distinguish between scene-level semantic anchors, injected via cross-attention, and local action prompts that control motion within a scene. Self-attention is selectively routed to preserve temporal continuity only across segments with compatible scene semantics, while transitions that change the ontological status of the scene explicitly limit long-range self-attention to prevent semantic leakage.
#A scene prompt must cover a contiguous span of local prompts where the world ontology is unchanged.
#If a prompt exists across more than one segment, it is treated as a scene prompt.
import gc
import logging
import math
import os
import random
import sys
import types
import json
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

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
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
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

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)


    def _prepare_prompts(self, global_prompt, local_prompts, dependency, scene_prompts, mapping, frame_num, size):
        
        tokenizer = self.text_encoder.tokenizer

        latent_frames = (frame_num - 1) // self.vae_stride[0] + 1
        width, height = size
        h_lat = int(height) // self.vae_stride[1]
        w_lat = int(width) // self.vae_stride[2]
        h_patches = h_lat // self.patch_size[1]
        w_patches = w_lat // self.patch_size[2]
        tokens_per_frame = int(h_patches) * int(w_patches)

        concat_prompt = ""
        for i, scene_prompt in enumerate(scene_prompts):
            concat_prompt += scene_prompt
            mapped_indices = sorted(mapping[str(i+1)])
            for idx in mapped_indices:
                concat_prompt += local_prompts[idx-1]
            

        full_prompt = global_prompt + concat_prompt
        print(f'full prompt: {full_prompt}')
        full_ids = tokenizer(full_prompt, add_special_tokens=True, padding=False, return_mask=False)[0].tolist()
        print(f'full ids: {full_ids}')
        def sentence_to_token_indices(subsentences):
            #returns token indices per local prompt
            def find_subsequence(haystack, needle):
                #Return the start and end index
                for start in range(len(haystack) - len(needle) + 1):
                    if haystack[start: start + len(needle)] == needle: 
                        return (start, start + len(needle))
                
            token_indices = {}
            for subsentence in subsentences:
                sub_ids = tokenizer(subsentence, padding=False, add_special_tokens=False)[0].tolist()
                print(f'sub_ids: {sub_ids}')
                match = find_subsequence(full_ids, sub_ids)
                if match is None:
                    raise ValueError(f"Subsentence not found in full prompt: {subsentence}")
                
                token_indices[subsentence] = match
            return token_indices
        
        def frame_interval_to_q_indices(frame_start, frame_end, Hp, Wp):
            # Returns the q indices for the given frame interval
            q_idx = [] 
            for f in range(frame_start, frame_end):
                base = f * Hp * Wp 
                for i in range(Hp * Wp):
                    q_idx.append(base + i)
            return q_idx

        def build_q_token_idx(low_frame_intervals, high_frame_intervals, low_token_spans, high_token_spans, dependency, tokens_per_frame):        
            q_token_idx = []
            self_attention_map = []
            epsilon = 1e-3

            if len(low_frame_intervals)!=0:

                for frame_start, frame_end, subsentences in low_frame_intervals: 

                    # q_indices = frame_interval_to_q_indices(frame_start, frame_end, h_patches, w_patches)

                    low_spans = []
                    for subsentence in subsentences:
                        start, end = low_token_spans[subsentence]
                        low_spans.extend(range(start, end))

                    sigma =  (0.5 * (frame_start + frame_end))/math.sqrt(2*math.log(1/epsilon))
                   
                    q_token_idx.append({
                        "window": (frame_end - frame_start)//3,
                        "sigma": torch.tensor(sigma, dtype=torch.float16),
                        "midpoint": (frame_start + frame_end) // 2,
                        "tokens_per_frame": tokens_per_frame,
                        "local_token_idx": torch.tensor(low_spans, dtype=torch.long),
                    })

                # "q_idx": torch.tensor(q_indices, dtype=torch.long)

            
            if len(high_frame_intervals)!=0:
                for frame_start, frame_end, subsentences in high_frame_intervals: 
                    # q_indices = frame_interval_to_q_indices(frame_start, frame_end, h_patches, w_patches)

                    high_spans = []
                    for subsentence in subsentences:
                        start, end = high_token_spans[subsentence]
                        high_spans.extend(range(start, end))

                    sigma =  (0.5 * (frame_start + frame_end))/math.sqrt(2*math.log(1/epsilon))

                    q_token_idx.append({
                        "window": (frame_end - frame_start)//3,
                        "sigma": torch.tensor(sigma, dtype=torch.float16),
                        "midpoint": (frame_start + frame_end) // 2,
                        "tokens_per_frame": tokens_per_frame,
                        "local_token_idx": torch.tensor(high_spans, dtype=torch.long),
                    })
                # "q_idx": torch.tensor(q_indices, dtype=torch.long)

            # Build a simple windowed self-attention map: each segment attends to itself and neighboring segments.
            for key, values in dependency.items():
                key = int(key)
                neighbors = [q_token_idx[key-1]['q_idx']]
                for v in values:
                    neighbors.append(q_token_idx[v-1]['q_idx'])
                k_idx = torch.cat(neighbors, dim=0)
                k_idx = torch.unique(k_idx, sorted=True)
                # Ensure deterministic order and no overlaps when concatenating outputs
                q_idx = q_token_idx[key-1]['q_idx']
                self_attention_map.append({
                    "q_start": int(q_idx[0].item()) if len(q_idx) > 0 else 0,
                    "q_idx": q_idx,
                    "k_idx": k_idx,
                })

            # Sort the map by query start to guarantee correct output order
            self_attention_map = sorted(self_attention_map, key=lambda m: m["q_start"])

                
            # print(f'len(q_token_idx): {len(q_token_idx)}')
            return q_token_idx, self_attention_map


        low_spans = sentence_to_token_indices(local_prompts)
        high_spans = sentence_to_token_indices(scene_prompts)
        
        if len(local_prompts) != 0:
            step = math.ceil(latent_frames / len(local_prompts)) 
            low_frame_intervals = [(step*i, 
                                min(step*(i+1), latent_frames), 
                                [local_prompts[i]]) for i in range(len(local_prompts))]

            high_frame_intervals = []
            for i, scene_prompt in enumerate(scene_prompts):
                mapped_indices = sorted(mapping[str(i+1)])
                frame_start = low_frame_intervals[mapped_indices[0]-1][0]
                frame_end = low_frame_intervals[mapped_indices[-1]-1][1]
                high_frame_intervals.append((frame_start, frame_end, [scene_prompt]))

            q_token_idx, self_attention_map = build_q_token_idx(low_frame_intervals = low_frame_intervals,
                                                                high_frame_intervals = high_frame_intervals,
                                                                low_token_spans = low_spans,
                                                                high_token_spans = high_spans,
                                                                dependency = dependency,
                                                                tokens_per_frame = tokens_per_frame,
                                                                )
        else:
            q_token_idx = None
            self_attention_map = None
        return q_token_idx, self_attention_map, full_prompt

       
    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 prompt_filepath=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            cross_attn_q_token_idx (`list`, *optional*, defaults to None):
                Optional cross-attention routing config. Each entry is `(q_start, q_end, token_idx_list)` and
                restricts which text tokens (keys/values) the query slice `[q_start:q_end)` can attend to.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """

        if prompt_filepath is not None:
            with open(prompt_filepath, 'r') as f:
                prompts = json.load(f)
        
                global_prompt = prompts.get("global_prompt", "")
                local_prompts = prompts.get("local_prompts", [])
                dependency = prompts.get("dependency", {})
                scene_prompts = prompts.get("scene_prompts", [])
                mapping = prompts.get("mapping", {})
                local_prompts = [" " + lp for lp in local_prompts]



                cross_attn_q_token_idx, self_attention_map, input_prompt = self._prepare_prompts(global_prompt, local_prompts, dependency, scene_prompts, mapping, frame_num, size)
        else:
            self_attention_map = None


        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {
                'context': context,
                'seq_len': seq_len,
                'cross_attn_q_token_idx': None if prompt_filepath is None else cross_attn_q_token_idx,
                'self_attention_map': self_attention_map,
            }
            arg_null = {'context': context_null, 'seq_len': seq_len, 'self_attention_map': self_attention_map}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
