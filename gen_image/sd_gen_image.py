# coding: utf-8
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import os
import torch
import logging


class SDGenImage:
    def __init__(self, model_path, lora_path=None, lora_scale=1.0):
        # 这里先只考虑有1个lora scale:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        # model_path="/root/data/model/stable-diffusion-v1-5/"
        # self.pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipe.to("cuda")
        # lora_path = "/root/data/TheGodOfCookery/lora/meishi.safetensors"
        # if lora_path is not None:
        #     self.pipe.load_lora_weights(lora_path)
        #     self.pipe.fuse_lora(lora_scale = lora_scale)
        # if using torch < 2.0
        # pipe.enable_xformers_memory_efficient_attention()

        # prompt = "An astronaut riding a green horse"

        # images = pipe(prompt=prompt).images[0]

    def create_img(self, prompt):
        try:
            logging.info("[Stable Diffusion] image_query={}".format(prompt))
            img = self.pipe(prompt, guidance_scale=7.5).images[0]
            return True, img
        except Exception as e:
            logging.exception(e)
            return False, "画图出现问题，请休息一下再问我吧"
