"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""

from dataclasses import asdict

import streamlit as st
import torch
from audiorecorder import audiorecorder
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from tools.transformers.interface import (
    GenerationConfig, generate_interactive)
from whisper_app import run_whisper

logger = logging.get_logger(__name__)

# global variables
enable_rag = None
streaming = None
user_avatar = "images/user.png"
robot_avatar = "images/robot.png"
user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"
# speech
audio_save_path = "/tmp/audio.wav"
whisper_model_scale = "medium"


def on_btn_click():
    """
    点击按钮时执行的函数，用于删除session_state中存储的消息。

    Args:
        无

    Returns:
        无
    """
    del st.session_state.messages


@st.cache_resource
def load_model():
    """
    加载预训练模型和分词器。

    Args:
        无。

    Returns:
        model (Transformers模型): 预训练模型。
        tokenizer (Transformers分词器): 分词器。
    """
    model = (
        AutoModelForCausalLM.from_pretrained(
            "zhanghuiATchina/zhangxiaobai_shishen2_full", trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "zhanghuiATchina/zhangxiaobai_shishen2_full", trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    """
    准备生成配置。

    Args:
        无

    Returns:
        Tuple[GenerationConfig, Optional[str]]: 包含生成配置和语音字符串的元组。
            - GenerationConfig: 生成配置。
            - Optional[str]: 语音字符串，如果没有录制语音则为None。
    """
    with st.sidebar:
        # 1. Max length of the generated text
        max_length = st.slider("Max Length", min_value=32,
                               max_value=2048, value=2048)

        # 2. Clear history.
        st.button("Clear Chat History", on_click=on_btn_click)

        # 3. Speech input
        audio = audiorecorder("Record", "Stop record")
        speech_string = None
        if len(audio) > 0:
            audio.export(audio_save_path, format="wav")
            speech_string = run_whisper(
                whisper_model_scale, "cuda",
                audio_save_path)

    generation_config = GenerationConfig(
        max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)

    return generation_config, speech_string


def combine_history(prompt):
    """
    根据用户输入的提示信息，组合出一段完整的对话历史，用于机器人进行对话。

    Args:
        prompt (str): 用户输入的提示信息。

    Returns:
        str: 组合好的对话历史。
    """
    messages = st.session_state.messages
    total_prompt = "您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。"
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.replace("{user}", prompt)
    return total_prompt


def process_user_input(prompt,
                       model,
                       tokenizer,
                       generation_config):
    """
    处理用户输入，根据用户输入内容调用相应的模型生成回复。

    Args:
        prompt (str): 用户输入的内容。
        model (str): 使用的模型名称。
        tokenizer (object): 分词器对象。
        generation_config (dict): 生成配置参数。

    """
    # Check if the user input contains certain keywords
    keywords = ["怎么做", "做法", "菜谱"]
    contains_keywords = any(keyword in prompt for keyword in keywords)

    # Display user message in chat message container
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    real_prompt = combine_history(prompt)

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": user_avatar})

    # If keywords are not present, display a prompt message immediately
    if not contains_keywords:
        with st.chat_message("robot", avatar=robot_avatar):
            st.markdown(
                "我是食神周星星的唯一传人张小白，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？，我会告诉你具体的做法。")
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": "我是食神周星星的唯一传人张小白，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？，我会告诉你具体的做法。", "avatar": robot_avatar})
    else:
        # Generate robot response
        with st.chat_message("robot", avatar=robot_avatar):
            message_placeholder = st.empty()
            generator = generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                # additional_eos_token_id=103028,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            )
            for cur_response in generator:
                cur_response = cur_response.replace('\\n', '\n')
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": cur_response, "avatar": robot_avatar})
        torch.cuda.empty_cache()


def main():
    st.title("食神2——菜谱小助手 by 张小白")
    model, tokenizer = load_model()
    generation_config, speech_prompt = prepare_generation_config()

    # 1.Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2.Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # 3.Process text input
    if text_prompt := st.chat_input("What is up?"):
        process_user_input(text_prompt, model, tokenizer, generation_config)

    # 4. Process speech input
    if speech_prompt is not None:
        process_user_input(speech_prompt, model, tokenizer, generation_config)


if __name__ == "__main__":
    main()
