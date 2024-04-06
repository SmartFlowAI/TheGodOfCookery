from llama_index.core import PromptTemplate
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

llm = None
def completion_to_prompt(completion):
    # 需要严格对应模型的对话模板 这里适配书生浦语系列
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system\n"):
        prompt = "<|im_start|>system\n<|im_end|>\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt


def load_model():
    global llm
    if llm is None:
        llm = HuggingFaceLLM(
        model_name="/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b",
        tokenizer_name="/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b",
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.85,
            "repetition_penalty": 1.005,
        },
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="auto",
        model_kwargs=dict(
            trust_remote_code=True, torch_dtype=torch.bfloat16, do_sample=True
        ),  # 只能这样设置，会传回原来的huggingface接口
        tokenizer_kwargs=dict(trust_remote_code=True),
        stopping_ids=[92542, 2],
    )
        return llm
    else:
        return llm

