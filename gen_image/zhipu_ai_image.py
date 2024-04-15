# coding: utf-8
import logging


# ZhipuAI提供的画图接口

class ZhipuAIImage(object):
    def __init__(self, api_key, model='cogview-3', image_create_size='256x256'):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.image_create_size = image_create_size

    def create_img(self, prompt, retry_count=0, api_key=None, api_base=None):
        try:
            logging.info("[ZHIPU_AI] image_query={}".format(prompt))
            response = self.client.images.generations(
                prompt=prompt,
                n=1,  # 每次生成图片的数量
                model=self.model,
                size=self.image_create_size,  # 图片大小,可选有 256x256, 512x512, 1024x1024
                quality="standard",
            )
            image_url = response.data[0].url
            logging.info("[ZHIPU_AI] image_url={}".format(image_url))
            return True, image_url
        except Exception as e:
            logging.exception(e)
            return False, "画图出现问题，请休息一下再问我吧"
