# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import logging
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

import dashscope
import torch
from PIL import Image

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_VER = 2
except ModuleNotFoundError:
    flash_attn_varlen_func = None  # in compatible with CPU machines
    FLASH_VER = None

from .system_prompt import *

DEFAULT_SYS_PROMPTS = {
    "t2v-A14B": {
        "zh": T2V_A14B_ZH_SYS_PROMPT,
        "en": T2V_A14B_EN_SYS_PROMPT,
    },
    "i2v-A14B": {
        "zh": I2V_A14B_ZH_SYS_PROMPT,
        "en": I2V_A14B_EN_SYS_PROMPT,
        "empty": {
            "zh": I2V_A14B_EMPTY_ZH_SYS_PROMPT,
            "en": I2V_A14B_EMPTY_EN_SYS_PROMPT,
        }
    },
    "ti2v-5B": {
        "t2v": {
            "zh": T2V_A14B_ZH_SYS_PROMPT,
            "en": T2V_A14B_EN_SYS_PROMPT,
        },
        "i2v": {
            "zh": I2V_A14B_ZH_SYS_PROMPT,
            "en": I2V_A14B_EN_SYS_PROMPT,
        }
    },
}


@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


class PromptExpander:

    def __init__(self, model_name, task, is_vl=False, device=0, **kwargs):
        self.model_name = model_name
        self.task = task
        self.is_vl = is_vl
        self.device = device

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image=None,
                        seed=-1,
                        *args,
                        **kwargs):
        pass

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        pass

    def decide_system_prompt(self, tar_lang="zh", prompt=None):
        assert self.task is not None
        if "ti2v" in self.task:
            if self.is_vl:
                return DEFAULT_SYS_PROMPTS[self.task]["i2v"][tar_lang]
            else:
                return DEFAULT_SYS_PROMPTS[self.task]["t2v"][tar_lang]
        if "i2v" in self.task and len(prompt) == 0:
            return DEFAULT_SYS_PROMPTS[self.task]["empty"][tar_lang]
        return DEFAULT_SYS_PROMPTS[self.task][tar_lang]

    def __call__(self,
                 prompt,
                 system_prompt=None,
                 tar_lang="zh",
                 image=None,
                 seed=-1,
                 *args,
                 **kwargs):
        if system_prompt is None:
            system_prompt = self.decide_system_prompt(
                tar_lang=tar_lang, prompt=prompt)
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        if image is not None and self.is_vl:
            return self.extend_with_img(
                prompt, system_prompt, image=image, seed=seed, *args, **kwargs)
        elif not self.is_vl:
            return self.extend(prompt, system_prompt, seed, *args, **kwargs)
        else:
            raise NotImplementedError


class DashScopePromptExpander(PromptExpander):

    def __init__(self,
                 api_key=None,
                 model_name=None,
                 task=None,
                 max_image_size=512 * 512,
                 retry_times=4,
                 is_vl=False,
                 **kwargs):
        '''
        Args:
            api_key: The API key for Dash Scope authentication and access to related services.
            model_name: Model name, 'qwen-plus' for extending prompts, 'qwen-vl-max' for extending prompt-images.
            task: Task name. This is required to determine the default system prompt.
            max_image_size: The maximum size of the image; unit unspecified (e.g., pixels, KB). Please specify the unit based on actual usage.
            retry_times: Number of retry attempts in case of request failure.
            is_vl: A flag indicating whether the task involves visual-language processing.
            **kwargs: Additional keyword arguments that can be passed to the function or method.
        '''
        if model_name is None:
            model_name = 'qwen-plus' if not is_vl else 'qwen-vl-max'
        super().__init__(model_name, task, is_vl, **kwargs)
        if api_key is not None:
            dashscope.api_key = api_key
        elif 'DASH_API_KEY' in os.environ and os.environ[
                'DASH_API_KEY'] is not None:
            dashscope.api_key = os.environ['DASH_API_KEY']
        else:
            raise ValueError("DASH_API_KEY is not set")
        if 'DASH_API_URL' in os.environ and os.environ[
                'DASH_API_URL'] is not None:
            dashscope.base_http_api_url = os.environ['DASH_API_URL']
        else:
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        self.api_key = api_key

        self.max_image_size = max_image_size
        self.model = model_name
        self.retry_times = retry_times

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': prompt
        }]

        exception = None
        for _ in range(self.retry_times):
            try:
                response = dashscope.Generation.call(
                    self.model,
                    messages=messages,
                    seed=seed,
                    result_format='message',  # set the result to be "message" format.
                )
                assert response.status_code == HTTPStatus.OK, response
                expanded_prompt = response['output']['choices'][0]['message'][
                    'content']
                return PromptOutput(
                    status=True,
                    prompt=expanded_prompt,
                    seed=seed,
                    system_prompt=system_prompt,
                    message=json.dumps(response, ensure_ascii=False))
            except Exception as e:
                exception = e
        return PromptOutput(
            status=False,
            prompt=prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception))

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image: Union[Image.Image, str] = None,
                        seed=-1,
                        *args,
                        **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        w = image.width
        h = image.height
        area = min(w * h, self.max_image_size)
        aspect_ratio = h / w
        resized_h = round(math.sqrt(area * aspect_ratio))
        resized_w = round(math.sqrt(area / aspect_ratio))
        image = image.resize((resized_w, resized_h))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image.save(f.name)
            fname = f.name
            image_path = f"file://{f.name}"
        prompt = f"{prompt}"
        messages = [
            {
                'role': 'system',
                'content': [{
                    "text": system_prompt
                }]
            },
            {
                'role': 'user',
                'content': [{
                    "text": prompt
                }, {
                    "image": image_path
                }]
            },
        ]
        response = None
        result_prompt = prompt
        exception = None
        status = False
        for _ in range(self.retry_times):
            try:
                response = dashscope.MultiModalConversation.call(
                    self.model,
                    messages=messages,
                    seed=seed,
                    result_format='message',  # set the result to be "message" format.
                )
                assert response.status_code == HTTPStatus.OK, response
                result_prompt = response['output']['choices'][0]['message'][
                    'content'][0]['text'].replace('\n', '\\n')
                status = True
                break
            except Exception as e:
                exception = e
        result_prompt = result_prompt.replace('\n', '\\n')
        os.remove(fname)

        return PromptOutput(
            status=status,
            prompt=result_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception) if not status else json.dumps(
                response, ensure_ascii=False))


class QwenPromptExpander(PromptExpander):
    model_dict = {
        "QwenVL2.5_3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "QwenVL2.5_7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5_3B": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5_7B": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5_14B": "Qwen/Qwen2.5-14B-Instruct",
    }

    def __init__(self,
                 model_name=None,
                 task=None,
                 device=0,
                 is_vl=False,
                 **kwargs):
        '''
        Args:
            model_name: Use predefined model names such as 'QwenVL2.5_7B' and 'Qwen2.5_14B',
                which are specific versions of the Qwen model. Alternatively, you can use the
                local path to a downloaded model or the model name from Hugging Face."
              Detailed Breakdown:
                Predefined Model Names:
                * 'QwenVL2.5_7B' and 'Qwen2.5_14B' are specific versions of the Qwen model.
                Local Path:
                * You can provide the path to a model that you have downloaded locally.
                Hugging Face Model Name:
                * You can also specify the model name from Hugging Face's model hub.
            task: Task name. This is required to determine the default system prompt.
            is_vl: A flag indicating whether the task involves visual-language processing.
            **kwargs: Additional keyword arguments that can be passed to the function or method.
        '''
        if model_name is None:
            model_name = 'Qwen2.5_14B' if not is_vl else 'QwenVL2.5_7B'
        super().__init__(model_name, task, is_vl, device, **kwargs)
        if (not os.path.exists(self.model_name)) and (self.model_name
                                                      in self.model_dict):
            self.model_name = self.model_dict[self.model_name]

        if self.is_vl:
            # default: Load the model on the available device(s)
            from transformers import (
                AutoProcessor,
                AutoTokenizer,
                Qwen2_5_VLForConditionalGeneration,
            )
            try:
                from .qwen_vl_utils import process_vision_info
            except ImportError:
                from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                use_fast=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if FLASH_VER == 2 else
                torch.float16 if "AWQ" in self.model_name else "auto",
                attn_implementation="flash_attention_2"
                if FLASH_VER == 2 else None,
                device_map="cpu")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
                if "AWQ" in self.model_name else "auto",
                attn_implementation="flash_attention_2"
                if FLASH_VER == 2 else None,
                device_map="cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        self.model = self.model.to(self.device)
        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": prompt
        }]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text],
                                      return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)
        ]

        expanded_prompt = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        self.model = self.model.to("cpu")
        return PromptOutput(
            status=True,
            prompt=expanded_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=json.dumps({"content": expanded_prompt},
                               ensure_ascii=False))

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image: Union[Image.Image, str] = None,
                        seed=-1,
                        *args,
                        **kwargs):
        self.model = self.model.to(self.device)
        messages = [{
            'role': 'system',
            'content': [{
                "type": "text",
                "text": system_prompt
            }]
        }, {
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        expanded_prompt = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        self.model = self.model.to("cpu")
        return PromptOutput(
            status=True,
            prompt=expanded_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=json.dumps({"content": expanded_prompt},
                               ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    seed = 100
    prompt = "夏日海滩度假风格，一只戴着墨镜的白色猫咪坐在冲浪板上。猫咪毛发蓬松，表情悠闲，直视镜头。背景是模糊的海滩景色，海水清澈，远处有绿色的山丘和蓝天白云。猫咪的姿态自然放松，仿佛在享受海风和阳光。近景特写，强调猫咪的细节和海滩的清新氛围。"
    en_prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image = "./examples/i2v_input.JPG"

    def test(method,
             prompt,
             model_name,
             task,
             image=None,
             en_prompt=None,
             seed=None):
        prompt_expander = method(
            model_name=model_name, task=task, is_vl=image is not None)
        result = prompt_expander(prompt, image=image, tar_lang="zh")
        logging.info(f"zh prompt -> zh: {result.prompt}")
        result = prompt_expander(prompt, image=image, tar_lang="en")
        logging.info(f"zh prompt -> en: {result.prompt}")
        if en_prompt is not None:
            result = prompt_expander(en_prompt, image=image, tar_lang="zh")
            logging.info(f"en prompt -> zh: {result.prompt}")
            result = prompt_expander(en_prompt, image=image, tar_lang="en")
            logging.info(f"en prompt -> en: {result.prompt}")

    ds_model_name = None
    ds_vl_model_name = None
    qwen_model_name = None
    qwen_vl_model_name = None

    for task in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        # test prompt extend
        if "t2v" in task or "ti2v" in task:
            # test dashscope api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} dashscope prompt extend")
            test(
                DashScopePromptExpander,
                prompt,
                ds_model_name,
                task,
                image=None,
                en_prompt=en_prompt,
                seed=seed)

            # test qwen api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} qwen prompt extend")
            test(
                QwenPromptExpander,
                prompt,
                qwen_model_name,
                task,
                image=None,
                en_prompt=en_prompt,
                seed=seed)

        # test prompt-image extend
        if "i2v" in task:
            # test dashscope api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} dashscope vl prompt extend")
            test(
                DashScopePromptExpander,
                prompt,
                ds_vl_model_name,
                task,
                image=image,
                en_prompt=en_prompt,
                seed=seed)

            # test qwen api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} qwen vl prompt extend")
            test(
                QwenPromptExpander,
                prompt,
                qwen_vl_model_name,
                task,
                image=image,
                en_prompt=en_prompt,
                seed=seed)

        # test empty prompt extend
        if "i2v-A14B" in task:
            # test dashscope api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} dashscope vl empty prompt extend")
            test(
                DashScopePromptExpander,
                "",
                ds_vl_model_name,
                task,
                image=image,
                en_prompt=None,
                seed=seed)

            # test qwen api
            logging.info(f"-" * 40)
            logging.info(f"Testing {task} qwen vl empty prompt extend")
            test(
                QwenPromptExpander,
                "",
                qwen_vl_model_name,
                task,
                image=image,
                en_prompt=None,
                seed=seed)
