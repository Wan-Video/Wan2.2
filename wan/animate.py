# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import cv2
import types
from copy import deepcopy
from functools import partial
from einops import rearrange
import numpy as np
import torch
from loguru import logger
import torch.distributed as dist
from peft import set_peft_model_state_dict
from decord import VideoReader
from tqdm import tqdm
import torch.nn.functional as F
from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size

from .modules.animate import WanAnimateModel
from .modules.animate import CLIPModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .modules.animate.animate_utils import TensorList, get_loraconfig
from .utils.utils import load_and_merge_lora_weight_from_safetensors
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler



class WanAnimate:

    def __init__(
        self,
        config,
        checkpoint_dir,
        lora_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_relighting_lora=False
    ):
        r"""
        Initializes the generation model components.

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
            use_relighting_lora (`bool`, *optional*, defaults to False):
               Whether to use relighting lora for character replacement. 
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.boundary = getattr(config, 'boundary', 0.9)
        self._has_lightning_lora = lora_dir is not None

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.clip = CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanAnimate from {checkpoint_dir}")

        # If a Lightning-style LoRA dir is provided, build two models and merge LoRA weights.
        if self._has_lightning_lora:
            assert use_relighting_lora is False, "Do not use --use_relighting_lora together with --lora_dir."
            if not dit_fsdp:
                self.low_noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=self.param_dtype,
                    device_map=self.device)
                self.high_noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=self.param_dtype,
                    device_map=self.device)
            else:
                self.low_noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir, torch_dtype=self.param_dtype)
                self.high_noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir, torch_dtype=self.param_dtype)

            # merge LoRA weights
            low_path = os.path.join(lora_dir, config.low_noise_lora_checkpoint)
            high_path = os.path.join(lora_dir, config.high_noise_lora_checkpoint)

            # Fallbacks: allow single-file LoRA or keyword-based matching
            if not os.path.exists(low_path) or not os.path.exists(high_path):
                try:
                    lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
                except Exception:
                    lora_files = []

                if len(lora_files) == 1:
                    # single LoRA file provided; apply to both models
                    only_file = os.path.join(lora_dir, lora_files[0])
                    logging.warning(f"Only one LoRA file found in {lora_dir}: {lora_files[0]}. Applying it to both low/high noise models.")
                    low_path = only_file
                    high_path = only_file
                else:
                    # try to pick by keywords
                    low_candidate = next((os.path.join(lora_dir, f) for f in lora_files if 'low' in f.lower()), None)
                    high_candidate = next((os.path.join(lora_dir, f) for f in lora_files if 'high' in f.lower()), None)
                    if low_candidate and not os.path.exists(low_path):
                        low_path = low_candidate
                    if high_candidate and not os.path.exists(high_path):
                        high_path = high_candidate

            if not os.path.exists(low_path):
                raise FileNotFoundError(f"LoRA file for low-noise model not found: {low_path}")
            if not os.path.exists(high_path):
                raise FileNotFoundError(f"LoRA file for high-noise model not found: {high_path}")

            logging.info(f"Merging low-noise LoRA from: {low_path}")
            self.low_noise_model = load_and_merge_lora_weight_from_safetensors(self.low_noise_model, low_path)
            logging.info(f"Merging high-noise LoRA from: {high_path}")
            self.high_noise_model = load_and_merge_lora_weight_from_safetensors(self.high_noise_model, high_path)

            # configure
            self.low_noise_model = self._configure_model(
                model=self.low_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
                use_lora=False,
                checkpoint_dir=checkpoint_dir,
                config=config
            )
            self.high_noise_model = self._configure_model(
                model=self.high_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
                use_lora=False,
                checkpoint_dir=checkpoint_dir,
                config=config
            )
        else:
            if not dit_fsdp:
                self.noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=self.param_dtype,
                    device_map=self.device)
            else:
                self.noise_model = WanAnimateModel.from_pretrained(
                    checkpoint_dir, torch_dtype=self.param_dtype)

            self.noise_model = self._configure_model(
                model=self.noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
                use_lora=use_relighting_lora,
                checkpoint_dir=checkpoint_dir,
                config=config
                )

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.sample_prompt = config.prompt


    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, use_lora, checkpoint_dir, config):
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

            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if use_lora:
            logging.info("Loading Relighting Lora. ")
            lora_config = get_loraconfig(
                transformer=model,
                rank=128,
                alpha=128
            )
            model.add_adapter(lora_config)
            lora_path = os.path.join(checkpoint_dir, config.lora_checkpoint)
            peft_state_dict = torch.load(lora_path)["state_dict"]
            set_peft_model_state_dict(model, peft_state_dict)

        if dit_fsdp:
            model = shard_fn(model, use_lora=use_lora)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        """Select and prepare the correct model for the given timestep.

        If lightning LoRA is present, switch between low/high models by boundary.
        Also perform simple CPU/GPU offload if requested.
        """
        if not self._has_lightning_lora:
            return self.noise_model

        if t.item() >= boundary:
            required = 'high_noise_model'
            offload = 'low_noise_model'
        else:
            required = 'low_noise_model'
            offload = 'high_noise_model'

        if offload_model or self.init_on_cpu:
            if next(getattr(self, offload).parameters()).device.type == 'cuda':
                getattr(self, offload).to('cpu')
            if next(getattr(self, required).parameters()).device.type == 'cpu':
                getattr(self, required).to(self.device)
        return getattr(self, required)

    def inputs_padding(self, array, target_len):
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len


    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def padding_resize(self, img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[:, padding: padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[padding: padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path):
        pose_video_reader = VideoReader(src_pose_path)
        pose_len = len(pose_video_reader)
        pose_idxs = list(range(pose_len))
        cond_images = pose_video_reader.get_batch(pose_idxs).asnumpy()

        face_video_reader = VideoReader(src_face_path)
        face_len = len(face_video_reader)
        face_idxs = list(range(face_len))
        face_images = face_video_reader.get_batch(face_idxs).asnumpy()
        height, width = cond_images[0].shape[:2]
        refer_images = cv2.imread(src_ref_path)[..., ::-1]
        refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images
    
    def prepare_source_for_replace(self, src_bg_path, src_mask_path):
        bg_video_reader = VideoReader(src_bg_path)
        bg_len = len(bg_video_reader)
        bg_idxs = list(range(bg_len))
        bg_images = bg_video_reader.get_batch(bg_idxs).asnumpy()

        mask_video_reader = VideoReader(src_mask_path)
        mask_len = len(mask_video_reader)
        mask_idxs = list(range(mask_len))
        mask_images = mask_video_reader.get_batch(mask_idxs).asnumpy()
        mask_images = mask_images[:, :, :, 0] / 255
        return bg_images, mask_images

    def generate(
        self,
        src_root_path,
        replace_flag=False,
        clip_len=77,
        refert_num=1,
        shift=5.0,
        sample_solver='dpm++',
        sampling_steps=20,
        guide_scale=1,
        input_prompt="",
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from input image using diffusion process.

        Args:
            src_root_path ('str'):
                Process output path
            replace_flag (`bool`, *optional*, defaults to False):
                Whether to use character replace.
            clip_len (`int`, *optional*, defaults to 77):
                How many frames to generate per clips. The number should be 4n+1
            refert_num (`int`, *optional*, defaults to 1):
                How many frames used for temporal guidance. Recommended to be 1 or 5.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. 
            sample_solver (`str`, *optional*, defaults to 'dpm++'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 20):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 1.0):
                Classifier-free guidance scale. We only use it for expression control. 
                In most cases, it's not necessary and faster generation can be achieved without it. 
                When expression adjustments are needed, you may consider using this feature.
            input_prompt (`str`):
                Text prompt for content generation. We don't recommend custom prompts (although they work)
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height 
                - W: Frame width 
        """
        assert refert_num == 1 or refert_num == 5, "refert_num should be 1 or 5."

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if input_prompt == "":
            input_prompt = self.sample_prompt

        src_pose_path = os.path.join(src_root_path, "src_pose.mp4")
        src_face_path = os.path.join(src_root_path, "src_face.mp4")
        src_ref_path = os.path.join(src_root_path, "src_ref.png")

        cond_images, face_images, refer_images = self.prepare_source(src_pose_path=src_pose_path, src_face_path=src_face_path, src_ref_path=src_ref_path)
        
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

        real_frame_len = len(cond_images)
        target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        logging.info('real frames: {} target frames: {}'.format(real_frame_len, target_len))
        cond_images = self.inputs_padding(cond_images, target_len)
        face_images = self.inputs_padding(face_images, target_len)
        
        if replace_flag:
            src_bg_path = os.path.join(src_root_path, "src_bg.mp4")
            src_mask_path = os.path.join(src_root_path, "src_mask.mp4")
            bg_images, mask_images = self.prepare_source_for_replace(src_bg_path, src_mask_path)
            bg_images = self.inputs_padding(bg_images, target_len)
            mask_images = self.inputs_padding(mask_images, target_len)

        height, width = refer_images.shape[:2]
        start = 0
        end = clip_len
        all_out_frames = []
        while True:
            if start + refert_num >= len(cond_images):
                break

            if start == 0:
                mask_reft_len = 0
            else:
                mask_reft_len = refert_num

            batch = {
                        "conditioning_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "bg_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "mask_pixel_values": torch.zeros(1, 1, clip_len, height, width),
                        "face_pixel_values": torch.zeros(1, 3, clip_len, 512, 512),
                        "refer_pixel_values": torch.zeros(1, 3, height, width),
                        "refer_t_pixel_values": torch.zeros(refert_num, 3, height, width)
                    }   

            batch["conditioning_pixel_values"] = rearrange(
                torch.tensor(np.stack(cond_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )
            batch["face_pixel_values"] = rearrange(
                torch.tensor(np.stack(face_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )

            batch["refer_pixel_values"] = rearrange(
                torch.tensor(refer_images / 127.5 - 1), "h w c -> 1 c h w"
            )

            if start > 0:
                batch["refer_t_pixel_values"] = rearrange(
                    out_frames[0, :, -refert_num:].clone().detach(),
                    "c t h w -> t c h w",
                )

            batch["refer_t_pixel_values"] = rearrange(batch["refer_t_pixel_values"],
                                            "t c h w -> 1 c t h w",
                                            )

            if replace_flag:
                batch["bg_pixel_values"] = rearrange(
                    torch.tensor(np.stack(bg_images[start:end]) / 127.5 - 1),
                    "t h w c -> 1 c t h w",
                )

                batch["mask_pixel_values"] = rearrange(
                    torch.tensor(np.stack(mask_images[start:end])[:, :, :, None]),
                    "t h w c -> 1 t c h w",
                )
                

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=self.device, dtype=torch.bfloat16)

            ref_pixel_values = batch["refer_pixel_values"]
            refer_t_pixel_values = batch["refer_t_pixel_values"]
            conditioning_pixel_values = batch["conditioning_pixel_values"]
            face_pixel_values = batch["face_pixel_values"]

            B, _, H, W = ref_pixel_values.shape
            T = clip_len
            lat_h = H // 8
            lat_w = W // 8
            lat_t = T // 4 + 1
            target_shape = [lat_t + 1, lat_h, lat_w]
            noise = [
                torch.randn(
                    16,
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            ]
        
            max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / self.sp_size)) * self.sp_size
            if max_seq_len % self.sp_size != 0:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by sp_size {self.sp_size}")

            with (
                torch.autocast(device_type=str(self.device), dtype=torch.bfloat16, enabled=True),
                torch.no_grad()
            ):
                logger.info(f"using {sample_solver}, time_step:{self.num_train_timesteps}, shift:{shift}")
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

                latents = noise

                pose_latents_no_ref =  self.vae.encode(conditioning_pixel_values.to(torch.bfloat16))
                pose_latents_no_ref = torch.stack(pose_latents_no_ref)
                pose_latents = torch.cat([pose_latents_no_ref], dim=2)

                ref_pixel_values = rearrange(ref_pixel_values, "t c h w -> 1 c t h w")
                ref_latents =  self.vae.encode(ref_pixel_values.to(torch.bfloat16))
                ref_latents = torch.stack(ref_latents)

                mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
                y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=self.device)

                img = ref_pixel_values[0, :, 0]
                clip_context = self.clip.visual([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=self.device)

                if mask_reft_len > 0:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        y_reft = self.vae.encode(
                            [
                                torch.concat([refer_t_pixel_values[0, :, :mask_reft_len], bg_pixel_values[0, :, mask_reft_len:]], dim=1).to(self.device)
                            ]
                        )[0]
                        mask_pixel_values = 1 - batch["mask_pixel_values"]
                        mask_pixel_values = rearrange(mask_pixel_values, "b t c h w -> (b t) c h w")
                        mask_pixel_values = F.interpolate(mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.nn.functional.interpolate(refer_t_pixel_values[0, :, :mask_reft_len].cpu(),
                                                                        size=(H, W), mode="bicubic"),
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)
                else:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        mask_pixel_values = 1 - batch["mask_pixel_values"]
                        mask_pixel_values = rearrange(mask_pixel_values, "b t c h w -> (b t) c h w")
                        mask_pixel_values = F.interpolate(mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        bg_pixel_values[0],
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)

                y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=self.device)
                y = torch.concat([y_ref, y_reft], dim=1)

                arg_c = {
                    "context": context, 
                    "seq_len": max_seq_len,
                    "clip_fea": clip_context.to(dtype=torch.bfloat16, device=self.device),
                    "y": [y],
                    "pose_latents": pose_latents,
                    "face_pixel_values": face_pixel_values,
                }

                if guide_scale > 1:
                    face_pixel_values_uncond = face_pixel_values * 0 - 1
                    arg_null = {
                        "context": context_null,
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context.to(dtype=torch.bfloat16, device=self.device),
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values_uncond,
                    }

                boundary = int(self.boundary * self.num_train_timesteps)
                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    # pick model (single or low/high by boundary)
                    model = self._prepare_model_for_timestep(t, boundary, offload_model)
                    noise_pred_cond = TensorList(
                         model(TensorList(latent_model_input), t=timestep, **arg_c)
                    )

                    if guide_scale > 1:
                        noise_pred_uncond = TensorList(
                             model(
                                TensorList(latent_model_input), t=timestep, **arg_null
                            )
                        )
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents[0] = temp_x0.squeeze(0)

                    x0 = latents

                x0 = [x.to(dtype=torch.float32) for x in x0]
                out_frames = torch.stack(self.vae.decode([x0[0][:, 1:]]))
                
                if start != 0:
                    out_frames = out_frames[:, :, refert_num:]

                all_out_frames.append(out_frames.cpu())

                start += clip_len - refert_num
                end += clip_len - refert_num

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]
        return videos[0] if self.rank == 0 else None
