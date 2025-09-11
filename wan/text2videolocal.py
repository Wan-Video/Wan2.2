# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import hashlib
from contextlib import contextmanager
from functools import partial
import gc

from torch.cuda.amp import autocast

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel, DynamicSwapInstaller
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import psutil


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__

        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def print_stats():
        global gpu_tensor_cache_linear
        # clear cache
        gpu_tensor_cache_linear = LRUCache(maxsize=35)


    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicSwapInstaller._install_module(m, **kwargs)
        print("installed modules: " + str(count_modules) + "\n")
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def get_free_ram(human_readable=False):
    """
    Get available free RAM in bytes or human-readable format.

    Args:
        human_readable (bool): If True, returns formatted string (e.g., "8.5 GB").
                               If False, returns raw bytes (default).

    Returns:
        int or str: Available RAM in bytes or formatted string
    """
    # Get virtual memory statistics
    mem = psutil.virtual_memory()

    # Available memory includes reclaimable cache/buffers
    available_bytes = mem.available

    if human_readable:
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if available_bytes < 1024:
                return f"{available_bytes:.2f} {unit}"
            available_bytes /= 1024
        return f"{available_bytes:.2f} PB"
    else:
        return available_bytes


class WanT2VLocal:

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
        self.param_dtype = torch.bfloat16 #config.param_dtype
        self.t5_fsdp = t5_fsdp

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        self.shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = None
        self.checkpoint_dir = checkpoint_dir

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = None
        #self.vae = Wan2_1_VAE(
        #    vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        #    device="cpu")

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = None
        self.use_sp = use_sp
        self.dit_fsdp = dit_fsdp
        self.convert_model_dtype = True #convert_model_dtype

        self.high_noise_model = None

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
                print(f"convetr to {self.param_dtype}")
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                print(f"load to {self.device}")
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
                 offload_model=True):
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

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale

        # 13 frames 5GB 57.57s/it
        # 17 frames
        # 21 frames
        # 25 frames
        frame_num = 17
        sampling_steps = 15

        F = frame_num

        print(f"Generating {frame_num} frames")

        target_shape = (16,  #self.vae.model.z_dim,
                        (F - 1) // self.vae_stride[0] + 1,
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


        print(f"Loading model T5EncoderModel")
        latent_file_path = "./final_latents.pt"
        if not os.path.exists(latent_file_path):
            print("--- STAGE 1: Generating latents from text prompt ---")
            # Define the cache directory for prompt tensors
            cache_dir = "./prompts_cache_video"
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique filename based on the prompts
            prompt_text = f"{input_prompt}-{n_prompt}"
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            cache_path = os.path.join(cache_dir, f"{prompt_hash}.pt")

            # Check if cached tensors exist
            if os.path.exists(cache_path):
                print(f"Loading cached prompt tensors from: {cache_path}")
                cached_tensors = torch.load(cache_path, map_location=self.device)
                context = cached_tensors['context']
                context_null = cached_tensors['context_null']
            else:
                print(f"No cache found. Encoding prompts and caching to: {cache_path}")
                print(f"Loading model T5EncoderModel to device: {self.device}")
                # Load the T5 model only when we need to encode new prompts
                self.text_encoder = T5EncoderModel(
                    text_len=self.config.text_len,
                    dtype=self.config.t5_dtype,
                    device=torch.device('cpu'),
                    checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
                    tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
                    shard_fn=self.shard_fn if self.t5_fsdp else None)


                DynamicSwapInstaller.install_model(self.text_encoder.model, device="cuda")
                context = self.text_encoder([input_prompt], torch.device('cuda'))
                context_null = self.text_encoder([n_prompt], torch.device('cuda'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
                self.text_encoder = None
                gc.collect()
                torch.cuda.empty_cache()

                # Cache the newly computed prompt tensors for reuse
                torch.save({
                    'context': context,
                    'context_null': context_null
                }, cache_path)

            print(f"Creating noise")

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

            print(f"Creating sample_scheduler")
            # evaluation mode
            with (
                    torch.amp.autocast('cuda', dtype=self.param_dtype),
                    torch.no_grad(),
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

                print(f"sample videos")
                # sample videos
                latents = noise

                arg_c = {'context': context, 'seq_len': seq_len}
                arg_null = {'context': context_null, 'seq_len': seq_len}
                blocks_in_ram = int(get_free_ram() / 830000000)
                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    if t.item() < boundary:
                        print("low noise model")

                    model = None
                    if t.item() >= boundary:
                        # high
                        if not self.high_noise_model:
                            print(f"Loading high_noise_checkpoint")
                            with autocast(True, dtype=torch.bfloat16):
                                self.high_noise_model = WanModel.from_pretrained(
                                    self.checkpoint_dir, subfolder=self.config.high_noise_checkpoint,
                                    model_type='t2v_h', blocks_in_ram=blocks_in_ram
                                )
                                self.high_noise_model = self._configure_model(
                                    model=self.high_noise_model,
                                    use_sp=self.use_sp,
                                    dit_fsdp=self.dit_fsdp,
                                    shard_fn=self.shard_fn,
                                    convert_model_dtype=self.convert_model_dtype)
                                self.high_noise_model.to('cuda')
                                self.high_noise_model.eval()
                                #DynamicSwapInstaller.install_model(self.high_noise_model, device="cuda")

                        model = self.high_noise_model
                    else:
                        # low
                        if not self.low_noise_model:
                            self.high_noise_model.clear_mem()
                            del self.high_noise_model, model
                            self.high_noise_model = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(f"Loading low_noise_checkpoint")
                            with autocast(True, dtype=torch.bfloat16):
                                self.low_noise_model = WanModel.from_pretrained(
                                    self.checkpoint_dir, subfolder=self.config.low_noise_checkpoint,
                                    model_type='t2v_l', blocks_in_ram=blocks_in_ram
                                )
                                self.low_noise_model = self._configure_model(
                                    model=self.low_noise_model,
                                    use_sp=self.use_sp,
                                    dit_fsdp=self.dit_fsdp,
                                    shard_fn=self.shard_fn,
                                    convert_model_dtype=self.convert_model_dtype)
                                self.low_noise_model.to('cuda')
                                self.low_noise_model.eval()
                                #DynamicSwapInstaller.install_model(self.low_noise_model, device="cuda")

                        model = self.low_noise_model

                    sample_guide_scale = guide_scale[1] if t.item(
                    ) >= boundary else guide_scale[0]

                    batche_proc = True
                    if batche_proc:
                        noise_pred_cond, noise_pred_uncond = model(
                            x=latent_model_input,
                            t=timestep,
                            context=context,
                            context_null=context_null,
                            seq_len=seq_len
                        )

                        # 4. Apply guidance
                        noise_pred = noise_pred_uncond[0] + sample_guide_scale * (noise_pred_cond[0] - noise_pred_uncond[0])

                    else:
                        noise_pred_cond = model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, **arg_null)[0]

                        noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                    if noise_pred_cond:
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                print("Denoising complete. Offloading main models...")
                self.low_noise_model.clear_mem()
                del self.low_noise_model, model
                self.low_noise_model = None
                gc.collect()
                torch.cuda.empty_cache()

                x0 = latents

                print(f"Saving final latents to: {latent_file_path}")
                torch.save(x0, latent_file_path)

        else:
            if os.path.exists(latent_file_path):
                print("--- STAGE 2: Found pre-computed latents. Decoding with VAE. ---")

                print(f"Loading latents from: {latent_file_path}")
                x0 = torch.load(latent_file_path)

        if self.rank == 0:
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                device=self.device,
                dtype=torch.bfloat16)
            print("Initializing Lazy State Dict Loader for VAE...")

            print("Loading VAE")

            DynamicSwapInstaller.install_model(self.vae.model, device="cuda")
            videos = self.vae.decode(x0)

            self.vae = None
            gc.collect()
            torch.cuda.empty_cache()

            return videos[0] if self.rank == 0 else None

# test prompt
# kernprof -l -v generate_local.py --task t2v-A14B --size "1280*720" --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
# kernprof -l -v generate_local.py --task t2v-A14B --size "1280*720" --ckpt_dir ./Wan2.2-T2V-A14B --prompt "A breathtaking high-speed chase through a neon-lit cyberpunk city at night. A sleek, futuristic motorcycle with glowing blue accents weaves through flying vehicles and holographic advertisements. The rider, clad in reflective black gear, leans into sharp turns while evading pursuit. Dynamic camera angles follow from above, below, and alongside, capturing the intense motion blur of city lights. Rain-slicked streets create colorful reflections as the motorcycle jumps between elevated highways. Explosions detonate in the background, illuminating the scene with orange bursts. Slow-motion shots capture water droplets and debris frozen in air before returning to blistering speed. The scene culminates in a daring leap across a massive gap between skyscrapers, with the cityscape sprawling below. Cinematic lighting, dramatic shadows, and particle effects enhance the adrenaline-fueled sequence."
# kernprof -l -v generate_local.py --task t2v-A14B --size "1280*720" --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Cinematic cyberpunk rainforest at twilight - Bioluminescent neon vines wrapping around ancient trees, holographic wildlife flickering between digital raindrops, slow-motion drone shot ascending through misty canopy layers. Style - Hyperrealistic Unreal Engine 5 render, neon-noir color palette, electric cyan deep magenta, volumetric lighting, ethereal synthwave soundtrack."

