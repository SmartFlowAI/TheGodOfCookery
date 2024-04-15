import os
import sys
import base64
from dataclasses import asdict
from datetime import datetime
import streamlit as st
import torch
from audiorecorder import audiorecorder
from funasr import AutoModel
from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging
from config import load_config
from gen_image import image_models
from convert_t2s import convert_t2s
from parse_cur_response import return_final_md
from rag.CookMasterLLM import CookMasterLLM
from rag.interface import (GenerationConfig,
                           generate_interactive,
                           generate_interactive_rag)
from speech import get_local_model

logger = logging.get_logger(__name__)

# solve: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
xlab_deploy = load_config('global', 'xlab_deploy')
if xlab_deploy:
    print("load sqllite3 module...")
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# llm
load_4bit = load_config('llm', 'load_4bit')
llm_model_path = load_config('llm', 'llm_model_path')
base_model_type = load_config('llm', 'base_model_type')
llm = None
print(f"base model type:{base_model_type}")

# rag
rag_model_type = load_config('rag', 'rag_model_type')
verbose = load_config('rag', 'verbose')
print(f"RAG model type:{rag_model_type}")

# speech
audio_save_path = load_config('speech', 'audio_save_path')
speech_model_type = load_config('speech', 'speech_model_type')
speech_model_path = load_config('speech', 'speech_model_path')
print(f"speech model type:{speech_model_type}")


@st.cache_resource
def load_model(generation_config):
    """
    加载预训练模型和分词器。

    Args:
        generation_config：模型配置参数。

    Returns:
        model (Transformers模型): 预训练模型。
        tokenizer (Transformers分词器): 分词器。
        llm (CookMasterLLM): langchain封装的大模型。
    """

    if load_4bit == False:

        model = (
            AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True)
            .to(torch.bfloat16)
            .cuda()
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)

    else:
        # int4 量化加载
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("正在从本地加载模型...")
        model = AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                                     device_map="auto",
                                                     quantization_config=quantization_config).eval()
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)

    llm = CookMasterLLM(model, tokenizer)
    print("完成本地模型的加载")
    model.generation_config.max_length = generation_config.max_length
    model.generation_config.top_p = generation_config.top_p
    model.generation_config.temperature = generation_config.temperature
    model.generation_config.repetition_penalty = generation_config.repetition_penalty
    # print(model.generation_config)
    return model, tokenizer, llm


def combine_history(prompt):
    """
    根据用户输入的提示信息，组合出一段完整的对话历史，用于机器人进行对话。

    Args:
        prompt (str): 用户输入的提示信息。

    Returns:
        str: 组合好的对话历史。
    """
    messages = st.session_state.messages
    meta_instruction = "您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。"
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
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


@st.cache_resource
def init_image_model():
    image_model_type = load_config('image', 'image_model_type')
    image_model_config = load_config('image', 'image_model_config').get(image_model_type)
    image_model = image_models[image_model_type](**image_model_config)
    return image_model


def text_to_image(prompt, image_model):
    file_dir = os.path.dirname(__file__)
    ok, ret = image_model.create_img(prompt)
    if ok:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_name = f"food_{current_datetime}.jpg"
        food_image_path = os.path.join(file_dir, "images/", new_file_name)
        print(f"Image file name:{food_image_path}")
        ret.save(food_image_path)
    else:
        food_image_path = os.path.join(file_dir, f"images/error.jpg")

    return food_image_path


@st.cache_resource
def load_speech_model():
    model_dict = get_local_model(speech_model_path)
    model = AutoModel(**model_dict)
    return model


def speech_rec(speech_model):
    audio = audiorecorder("开始语音输入", "停止语音输入")
    audio_b64 = base64.b64encode(audio.raw_data)
    speech_string = None
    if len(audio) > 0 and (
            'last_audio_b64' not in st.session_state or st.session_state['last_audio_b64'] != audio_b64):
        st.session_state['last_audio_b64'] = audio_b64
        try:
            audio.export(audio_save_path, format="wav")
            speech_string = speech_model.generate(input=audio_save_path)[0]['text']
            # 语言识别模型的返回结果可能有繁体字，需要转换。
            # print(f"Origin speech_string:{speech_string}")
            speech_string = convert_t2s(speech_string)
            # print(f"Converted speech_string:{speech_string}")
            # 语音识别模型的返回结果可能有多余的空格，需要去掉。
            speech_string = speech_string.replace(' ', '')
        except Exception as e:
            logger.warning('speech rec warning, exception is', e)
    return speech_string


def on_clear_btn_click():
    """
    点击按钮时执行的函数，用于删除session_state中存储的消息。
    """
    del st.session_state.messages


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
                               max_value=32768, value=32768)
        # 2. Clear history.
        st.button("Clear Chat History", on_click=on_clear_btn_click)

        # 3. Enable RAG
        global enable_rag
        enable_rag = st.checkbox("Enable RAG", value=True)

        # 4. Streaming
        # global streaming
        # streaming = st.checkbox("Streaming")

        # 5. Output markdown
        global enable_markdown
        enable_markdown = st.checkbox("Markdown output")

        # 6. Image
        global enable_image
        enable_image = st.checkbox("Show Image")

        # 7. Speech input
        speech_prompt = speech_rec(speech_model)
        st.session_state['speech_prompt'] = speech_prompt

    if base_model_type == 'internlm-chat-7b':
        generation_config = GenerationConfig(max_length=max_length)  # InternLM1
    elif base_model_type == 'internlm2-chat-1.8b':
        generation_config = GenerationConfig(
            max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.17)  # InternLM2 1.8b need 惩罚参数
    else:
        generation_config = GenerationConfig(
            max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.005)  # InternLM2 2 need 惩罚参数

    return generation_config


def process_user_input(prompt,
                       model,
                       tokenizer,
                       llm,
                       generation_config):
    """
    处理用户输入，根据用户输入内容调用相应的模型生成回复。

    Args:
        prompt (str): 用户输入的内容。
        model (str): 使用的模型名称。
        tokenizer (object): 分词器对象。
        llm: langchain包装的大模型
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
            st.markdown(error_response)
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": error_response, "avatar": robot_avatar})
    else:
        # Generate robot response
        with st.chat_message("robot", avatar=robot_avatar):
            message_placeholder = st.empty()
            # print("prompt:", prompt)
            # print("real_prompt:", real_prompt)
            if enable_rag:
                cur_response = generate_interactive_rag(
                    llm=llm,
                    question=prompt,
                    verbose=verbose,
                )

                cur_response = cur_response.replace('\\n', '\n')

                # print(cur_response)
                if enable_markdown:
                    cur_response = return_final_md(cur_response)
                    # print('after markdown：', cur_response)
                message_placeholder.markdown(cur_response + "▌")
            else:
                if base_model_type == 'internlm-chat-7b':
                    additional_eos_token_id = 103028  # InternLM-7b-chat
                elif base_model_type == 'internlm2-chat-1.8b':
                    additional_eos_token_id = 92542  # InternLM2-1.8b-chat
                else:
                    additional_eos_token_id = 92542  # InternLM2-7b-chat

                # print(f"additional_eos_token_id:{additional_eos_token_id}")

                generator = generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=additional_eos_token_id,  # InternLM or InternLM2
                    **asdict(generation_config),
                )

                for cur_response in generator:
                    cur_response = cur_response.replace('\\n', '\n')
                    message_placeholder.markdown(cur_response + "▌")

                # print(cur_response)
                if enable_markdown:
                    cur_response = return_final_md(cur_response)
                    # print('after markdown：', cur_response)
                message_placeholder.markdown(cur_response)

            if enable_image and prompt:
                food_image_path = text_to_image(prompt, image_model)
                # add food image
                st.image(food_image_path, width=230)

        # Add robot response to chat history
        response_message = {"role": "robot", "content": cur_response, "avatar": robot_avatar}

        if enable_image and prompt:
            response_message.update({'food_image_path': food_image_path})

        st.session_state.messages.append(response_message)
        torch.cuda.empty_cache()


def main():
    print(f"Torch support GPU: {torch.cuda.is_available()}")

    global speech_model
    speech_model = load_speech_model()
    generation_config = prepare_generation_config()

    st.title("食神2 by 其实你也可以是个厨师队")
    model, tokenizer, llm = load_model(generation_config)

    global image_model
    image_model = init_image_model()

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
    if text_prompt := st.chat_input("请在这里输入"):
        process_user_input(text_prompt, model, tokenizer, llm, generation_config)

    # 4. Process speech input
    if speech_prompt := st.session_state['speech_prompt']:
        process_user_input(speech_prompt, model, tokenizer, llm, generation_config)


if __name__ == "__main__":
    main()
