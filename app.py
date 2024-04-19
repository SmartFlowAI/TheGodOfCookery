import os
import sys
import copy
import base64
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, List, Optional
import streamlit as st
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging
from audiorecorder import audiorecorder
from funasr import AutoModel
from gen_image import image_models
from convert_t2s import convert_t2s
from speech import get_local_model
from parse_cur_response import return_final_md
from config import load_config
# from config_test import load_config

logger = logging.get_logger(__name__)

# solve: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
xlab_deploy = load_config('global', 'xlab_deploy')
if xlab_deploy:
    print("load sqllite3 module...")
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# global variables
enable_rag = load_config('global', 'enable_rag')
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
print(f"base model type:{base_model_type}")

# rag
rag_framework = load_config('global', 'rag_framework')
if rag_framework == 'langchain':
    from rag_langchain.interface import RagPipeline
    rag_model_type = load_config('rag_langchain', 'rag_model_type')
    verbose = load_config('rag_langchain', 'verbose')
else:
    from rag_llama.interface import RagPipeline
    rag_model_type = load_config('rag_llama', 'rag_model_type')
    verbose = load_config('rag_llama', 'verbose')
print(f"RAG framework:{rag_framework}")
print(f"RAG model type:{rag_model_type}")


# speech
audio_save_path = load_config('speech', 'audio_save_path')
speech_model_type = load_config('speech', 'speech_model_type')
speech_model_path = load_config('speech', 'speech_model_path')
print(f"speech model type:{speech_model_type}")

@dataclass
class GenerationConfig:
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0


# 书生浦语大模型实战营提供的生成函数模版
# 搭配streamlit使用，实现markdown结构化输出与流式输出
@torch.inference_mode()
def generate_interactive(
        model,
        tokenizer,
        prompt,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        additional_eos_token_id: Optional[int] = None,
        **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]  # noqa: F841
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id  # noqa: F841
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


@st.cache_resource
def load_model():
    """
    加载预训练模型和分词器。

    Args:
        generation_config：模型配置参数。

    Returns:
        model (Transformers模型): 预训练模型。
        tokenizer (Transformers分词器): 分词器。
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
        model = (
            AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True,
                                                 quantization_config=quantization_config, device_map="auto")
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)

    print("完成本地模型的加载")
    return model, tokenizer


@st.cache_resource
def load_rag_pipeline():
    """
    加载RAG模型。

    Returns:
        pipeline (RagPipeline): RAG模型。
    """
    rag_pipeline = RagPipeline()
    return rag_pipeline


def combine_history(prompt, retrieval_content=None):
    """
    根据用户输入的提示信息，组合出一段完整的对话历史，用于机器人进行对话。

    Args:
        prompt (str): 用户输入的提示信息。
        retrieval_content (str): 从RAG模块中检索到的内容。

    Returns:
        str: 组合好的对话历史。
    """
    messages = st.session_state.messages
    # RAG对话模板
    rag_template = """先对上下文进行内容总结,再使上下文来回答用户的问题。总是使用中文回答。
可参考的上下文：
···
{context}
···
问题: {question}
如果给定的上下文无法让你做出回答，请根据你自己所掌握的知识进行回答。
有用的回答:"""
    meta_instruction = "您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。"
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    # 最后一条message是当前用户输入的内容，要单独处理
    for i in range(len(messages) - 1):
        cur_content = messages[i]["content"]
        if messages[i]["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif messages[i]["role"] == "robot":
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    if retrieval_content:
        prompt = rag_template.replace("{context}", retrieval_content).replace("{question}", prompt)
    total_prompt += cur_query_prompt.replace("{user}", prompt)
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
        # 1. Set generation parameters
        max_length = st.slider("Max Length", min_value=8, max_value=32768, value=32768, step=1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

        # 2. Clear history.
        st.button("Clear Chat History", on_click=on_clear_btn_click)

        # 3. Enable RAG
        global enable_rag
        enable_rag = st.checkbox("Enable RAG", value=True)

        # 4. Output markdown
        global enable_markdown
        enable_markdown = st.checkbox("Markdown output")

        # 5. Image
        global enable_image
        enable_image = st.checkbox("Show Image")

        # 6. Speech input
        speech_prompt = speech_rec(speech_model)
        st.session_state['speech_prompt'] = speech_prompt

    if base_model_type == 'internlm-chat-7b':
        generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)  # InternLM1
    elif base_model_type == 'internlm2-chat-1.8b':
        generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature,
                                             repetition_penalty=1.17)  # InternLM2 1.8b need 惩罚参数
    else:
        generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature,
                                             repetition_penalty=1.005)  # InternLM2 2 need 惩罚参数

    return generation_config


def process_user_input(prompt,
                       model,
                       tokenizer,
                       rag_pipeline,
                       generation_config):
    """
    处理用户输入，根据用户输入内容调用相应的模型生成回复。

    Args:
        prompt (str): 用户输入的内容。
        model (str): 使用的模型名称。
        tokenizer (object): 分词器对象。
        rag_pipeline (RagPipeline): RAG模块。
        generation_config (dict): 生成配置参数。

    """

    # Check if the user input contains certain keywords
    keywords = ["怎么做", "做法", "菜谱"]
    contains_keywords = any(keyword in prompt for keyword in keywords)

    # Display user message in chat message container
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avatar})

    # If keywords are not present, display a prompt message immediately
    if not contains_keywords:
        with st.chat_message("robot", avatar=robot_avatar):
            st.markdown(error_response)
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": error_response, "avatar": robot_avatar})
    else:
        # Retrieve content
        retrieval_content = None
        if enable_rag and rag_pipeline:
            retrieval_content = rag_pipeline.get_retrieval_content(prompt)
        real_prompt = combine_history(prompt, retrieval_content)
        # Generate robot response
        with st.chat_message("robot", avatar=robot_avatar):
            message_placeholder = st.empty()


            if base_model_type == 'internlm-chat-7b':
                additional_eos_token_id = 103028  # InternLM-7b-chat
            elif base_model_type == 'internlm2-chat-1.8b':
                additional_eos_token_id = 92542  # InternLM2-1.8b-chat
            else:
                additional_eos_token_id = 92542  # InternLM2-7b-chat

            if verbose:
                print("prompt———————————————————————————————————————————————————————————————————————————————\n", prompt)
                print("Retrieval content—————————————————————————————————————————————————————————\n", retrieval_content)
                print("real_prompt—————————————————————————————————————————————————————————————————————\n", real_prompt)
                print(f"additional_eos_token_id——————————————————————————————————————————————{additional_eos_token_id}")

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
            message_placeholder.markdown(cur_response + "▌")

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
    model, tokenizer = load_model()
    rag_pipeline = None
    if enable_rag:
        rag_pipeline = load_rag_pipeline()

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
        process_user_input(text_prompt, model, tokenizer, rag_pipeline, generation_config)

    # 4. Process speech input
    if speech_prompt := st.session_state['speech_prompt']:
        process_user_input(speech_prompt, model, tokenizer, rag_pipeline, generation_config)


if __name__ == "__main__":
    main()
