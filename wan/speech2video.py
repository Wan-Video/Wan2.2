# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.s2v.audio_encoder import AudioEncoder
from .modules.s2v.model_s2v import WanModel_S2V, sp_attn_forward_s2v
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.frame_sampling import sample_video_frame_indices


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class WanS2V:

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
        Initializes the image-to-video generation model components.

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
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if not dit_fsdp:
            self.noise_model = WanModel_S2V.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = WanModel_S2V.from_pretrained(
                checkpoint_dir, torch_dtype=self.param_dtype)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.audio_encoder = AudioEncoder(
            model_id=os.path.join(checkpoint_dir,
                                  "wav2vec2-large-xlsr-53-english"))

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0

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
                    sp_attn_forward_s2v, block.self_attn)
            model.use_context_parallel = True

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

    def get_size_less_than_area(self,
                                height,
                                width,
                                target_area=1024 * 704,
                                divisor=64):
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
                2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area /
                                  (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int(
                (target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int(
                (target_area / aspect_ratio)**0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def prepare_default_cond_input(self,
                                   map_shape=[3, 12, 64, 64],
                                   motion_frames=5,
                                   lat_motion_frames=2,
                                   enable_mano=False,
                                   enable_kp=False,
                                   enable_pose=False):
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = torch.ones(
                    map_shape, dtype=self.param_dtype, device=self.device) * d
                cond_lat = torch.cat([
                    map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1),
                    map_value
                ],
                                     dim=2)
                cond_lat = torch.stack(
                    self.vae.encode(cond_lat.to(
                        self.param_dtype)))[:, :, lat_motion_frames:].to(
                            self.param_dtype)

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = torch.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path, infer_frames):
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self,
                           video_path,
                           n_frames,
                           target_fps=16,
                           reverse=False):
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order. 
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        sampled_indices = sample_video_frame_indices(
            original_fps=original_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            num_frames=n_frames,
            from_start=reverse)

        return vr.get_batch(sampled_indices.tolist()).asnumpy()

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, size):
        HEIGHT, WIDTH = size
        if not pose_video is None:
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True)

            resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
            crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
            tensor_trans = transforms.ToTensor()

            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(
                1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat([
                cond_tensor,
                - torch.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])
            ],
                                    dim=2)

            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond],
                             dim=2)
            cond_lat = torch.stack(
                self.vae.encode(
                    cond.to(dtype=self.param_dtype,
                            device=self.device)))[:, :,
                                                  1:].cpu()  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        if not size is None:
            HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(
                    pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(
            HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    def generate(
        self,
        input_prompt,
        ref_image_path,
        audio_path,
        enable_tts,
        tts_prompt_audio,
        tts_prompt_text,
        tts_text,
        num_repeat=1,
        pose_video=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3

        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        ref_image = None
        motion_latents = None

        if ref_image is None:
            ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        if motion_latents is None:
            motion_latents = torch.zeros(
                [1, channel, self.motion_frames, HEIGHT, WIDTH],
                dtype=self.param_dtype,
                device=self.device)

        # extract audio emb
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
            0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device)
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))

        # encode the motion latents
        videos_last_frames = motion_latents.detach()
        drop_first_motion = self.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            motion_latents[:, :, -6:] = ref_pixel_values
        motion_latents = torch.stack(self.vae.encode(motion_latents))

        # get pose cond input if need
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
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

        out = []
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            for r in range(num_repeat):
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + r)

                lat_target_frames = (infer_frames + 3 + self.motion_frames
                                    ) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                noise = [
                    torch.randn(
                        16,
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        dtype=self.param_dtype,
                        device=self.device,
                        generator=seed_g)
                ]
                max_seq_len = np.prod(target_shape) // 4

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

                latents = deepcopy(noise)
                with torch.no_grad():
                    left_idx = r * infer_frames
                    right_idx = r * infer_frames + infer_frames
                    cond_latents = COND[r] if pose_video else COND[0] * 0
                    cond_latents = cond_latents.to(
                        dtype=self.param_dtype, device=self.device)
                    audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    'context': context[0:1],
                    'seq_len': max_seq_len,
                    'cond_states': cond_latents,
                    "motion_latents": input_motion_latents,
                    'ref_latents': ref_latents,
                    "audio_input": audio_input,
                    "motion_frames": [self.motion_frames, lat_motion_frames],
                    "drop_motion_frames": drop_first_motion and r == 0,
                }
                if guide_scale > 1:
                    arg_null = {
                        'context': context_null[0:1],
                        'seq_len': max_seq_len,
                        'cond_states': cond_latents,
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": 0.0 * audio_input,
                        "motion_frames": [
                            self.motion_frames, lat_motion_frames
                        ],
                        "drop_motion_frames": drop_first_motion and r == 0,
                    }
                if offload_model or self.init_on_cpu:
                    self.noise_model.to(self.device)
                    torch.cuda.empty_cache()

                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents[0:1]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    noise_pred_cond = self.noise_model(
                        latent_model_input, t=timestep, **arg_c)

                    if guide_scale > 1:
                        noise_pred_uncond = self.noise_model(
                            latent_model_input, t=timestep, **arg_null)
                        noise_pred = [
                            u + guide_scale * (c - u)
                            for c, u in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents[0] = temp_x0.squeeze(0)

                if offload_model:
                    self.noise_model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                latents = torch.stack(latents)
                if not (drop_first_motion and r == 0):
                    decode_latents = torch.cat([motion_latents, latents], dim=2)
                else:
                    decode_latents = torch.cat([ref_latents, latents], dim=2)
                image = torch.stack(self.vae.decode(decode_latents))
                image = image[:, :, -(infer_frames):]
                if (drop_first_motion and r == 0):
                    image = image[:, :, 3:]

                overlap_frames_num = min(self.motion_frames, image.shape[2])
                videos_last_frames = torch.cat([
                    videos_last_frames[:, :, overlap_frames_num:],
                    image[:, :, -overlap_frames_num:]
                ],
                                               dim=2)
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents.dtype, device=motion_latents.device)
                motion_latents = torch.stack(
                    self.vae.encode(videos_last_frames))
                out.append(image.cpu())

        videos = torch.cat(out, dim=2)
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def tts(self, tts_prompt_audio, tts_prompt_text, tts_text):
        if not hasattr(self, 'cosyvoice'):
            self.load_tts()
        speech_list = []
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        prompt_speech_16k = load_wav(tts_prompt_audio, 16000)
        if tts_prompt_text is not None:
            for i in self.cosyvoice.inference_zero_shot(tts_text, tts_prompt_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        else:
            for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        torchaudio.save('tts.wav', torch.concat(speech_list, dim=1), self.cosyvoice.sample_rate)
        return 'tts.wav'

    def load_tts(self):
        if not os.path.exists('CosyVoice'):
            from wan.utils.utils import download_cosyvoice_repo
            download_cosyvoice_repo('CosyVoice')
        if not os.path.exists('CosyVoice2-0.5B'):
            from wan.utils.utils import download_cosyvoice_model
            download_cosyvoice_model('CosyVoice2-0.5B', 'CosyVoice2-0.5B')
        sys.path.append('CosyVoice')
        sys.path.append('CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosyvoice = CosyVoice2('CosyVoice2-0.5B')
