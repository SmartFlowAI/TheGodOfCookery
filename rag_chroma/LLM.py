#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   llm.py
@Time    :   2023/10/16 18:53:26
@Author  :   Logan Zou
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于InternLM模型自定义 LLM 类
"""

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
# from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


class CookMasterLLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = tokenizer
        self.model = model
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = "您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。"
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM2"



# if __name__ == "__main__":
#     # 测试代码
#     llm = InternLM_LLM(model_path="/root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b")
#     print(llm.predict("你是谁"))
