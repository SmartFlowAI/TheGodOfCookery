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

from rag_chroma.interface import (GenerationConfig,
                                  generate_interactive,
                                  generate_interactive_rag_stream,
                                  generate_interactive_rag)
from whisper_app import run_whisper
from gen_image import image_models
from config import load_config
import os
from datetime import datetime
from PIL import Image
from parse_cur_response import return_final_md
import opencc
from convert_t2s import convert_t2s
logger = logging.get_logger(__name__)

xlab_deploy = load_config('global','xlab_deploy')

if xlab_deploy:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3']= sys.modules.pop('pysqlite3')

# global variables
enable_rag = load_config('global', 'enable_rag')
# streaming = load_config('global', 'streaming')
enable_image = load_config('global', 'enable_image')
enable_markdown = load_config('global', 'enable_markdown')
user_avatar = load_config('global', 'user_avatar')
robot_avatar = load_config('global', 'robot_avatar')
user_prompt = load_config('global', 'user_prompt')
robot_prompt = load_config('global', 'robot_prompt')
cur_query_prompt = load_config('global', 'cur_query_prompt')
error_response = load_config('global', 'error_response')

# speech
audio_save_path = load_config('speech', 'audio_save_path')
whisper_model_scale = load_config('speech', 'whisper_model_scale')

# llm
llm_model_path = load_config('llm', 'llm_model_path')


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
        AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
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

        # 3. Enable RAG
        global enable_rag
        enable_rag = st.checkbox("Enable RAG")

        # 4. Streaming
        # global streaming
        # streaming = st.checkbox("Streaming")

        # 6. Output markdown
        global enable_markdown
        enable_markdown = st.checkbox("Markdown output")

        # 7. Image
        global enable_image
        enable_image = st.checkbox("Show Image")

        # 5. Speech input
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
    print("Origin Prompt:")
    print(prompt)
    prompt = convert_t2s(prompt)
    print("Converted Prompt:")
    print(prompt)

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
            st.markdown(error_response)
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": error_response, "avatar": robot_avatar})
    else:
        # Generate robot response
        with st.chat_message("robot", avatar=robot_avatar):
            message_placeholder = st.empty()
            if enable_rag:
                cur_response = generate_interactive_rag(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    history=real_prompt
                )
                cur_response = cur_response.replace('\\n', '\n')

                print(cur_response)

                if enable_markdown:
                    cur_response = return_final_md(cur_response)
                    print('afer markdown')
                    print(cur_response)

                message_placeholder.markdown(cur_response)
            else:
                generator = generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    # additional_eos_token_id=103028,  #InternLM-7b-chat
                    additional_eos_token_id=92542,  # InternLM2-7b-chat
                    **asdict(generation_config),
                )
                for cur_response in generator:
                    cur_response = cur_response.replace('\\n', '\n')
                    message_placeholder.markdown(cur_response + "▌")

                print(cur_response)
                if enable_markdown:
                    cur_response = return_final_md(cur_response)
                    print('after markdown')
                    print(cur_response)
                message_placeholder.markdown(cur_response)

            if enable_image and prompt:
                food_image_path = text_to_image(prompt, image_model)
                # add food image
                # img = Image.open(food_image_path)
                st.image(food_image_path, width=230)
            # for cur_response in generator:
            #     cur_response = cur_response.replace('\\n', '\n')
            #     message_placeholder.markdown(cur_response + "▌")
            # message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        response_message = {"role": "robot", "content": cur_response, "avatar": robot_avatar}

        if enable_image and prompt:
            response_message.update({'food_image_path': food_image_path})

        st.session_state.messages.append(response_message)
        torch.cuda.empty_cache()


@st.cache_resource
def init_image_model():
    image_model_type = load_config('image', 'image_model_type')
    image_model_config = load_config('image', 'image_model_config').get(image_model_type)
    image_model = image_models[image_model_type](**image_model_config)
    return image_model


# @st.cache_resource
def text_to_image(prompt, image_model):
    file_dir = os.path.dirname(__file__)
    # generate image
    ok, ret = image_model.create_img(prompt)
    if ok:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_name = f"food_{current_datetime}.jpg"
        food_image_path = os.path.join(file_dir, "images/", new_file_name)
        print("Image file name")
        print(food_image_path)
        ret.save(food_image_path)
    else:
        food_image_path = os.path.join(file_dir, f"images/error.jpg")

    return food_image_path


def main():
    print("Torch support GPU: ")
    print(torch.cuda.is_available())

    st.title("食神2 by 其实你也可以是个厨师队")
    model, tokenizer = load_model()
    global image_model
    image_model = init_image_model()
    generation_config, speech_prompt = prepare_generation_config()

    # 1.Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2.Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            if 'food_image_path' in message:
                st.image(message['food_image_path'], width=230)

    # 3.Process text input
    if text_prompt := st.chat_input("What is up?"):
        process_user_input(text_prompt, model, tokenizer, generation_config)

    # 4. Process speech input
    if speech_prompt is not None:
        process_user_input(speech_prompt, model, tokenizer, generation_config)


if __name__ == "__main__":
    main()
