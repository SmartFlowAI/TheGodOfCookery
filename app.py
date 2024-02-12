"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""

from dataclasses import asdict

from modelscope import AutoModelForCausalLM, AutoTokenizer
#from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st
import torch
import re  
import sys

from transformers.utils import logging

from tools.transformers.interface import GenerationConfig, generate_interactive
#from modelscope import GenerationConfig

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from rag.LLM import CookMasterLLM

embed_model_dir = "/home/zhanghui/models/m3e-base"
llm_model_dir = "/home/zhanghui/models/zhanghuiATchina/zhangxiaobai_shishen2_full"


logger = logging.get_logger(__name__)

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (
        AutoModelForCausalLM.from_pretrained("zhanghuiATchina/zhangxiaobai_shishen2_full", trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained("zhanghuiATchina/zhangxiaobai_shishen2_full", trust_remote_code=True)
    return model, tokenizer

@st.cache_resource
def load_chain():
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_dir)

    # 向量数据库持久化路径
    persist_directory = './rag/database'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    llm = CookMasterLLM(model_path = llm_model_dir)

    # 我们所构造的 Prompt 模板
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""

    # 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)


    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    return qa_chain, llm



enable_rag = None

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        #top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        #temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)
        enable_rag = st.checkbox("Enable RAG")

    #generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)
    generation_config = GenerationConfig(max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)
    #generation_config = GenerationConfig(max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.05)

    return generation_config


user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


def combine_history(prompt):
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


def main():
    # torch.cuda.empty_cache()
    print("load chain begin.")
    #model, tokenizer = load_model()
    qa_chain, llm  = load_chain()
    print("load chain end.")

    user_avator = "images/user.png"
    robot_avator = "images/robot.png"

    st.title("食神2——菜谱小助手 by 张小白")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Check if the user input contains certain keywords
        keywords = ["怎么做", "做法", "菜谱"]
        contains_keywords = any(keyword in prompt for keyword in keywords)

        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        # If keywords are not present, display a prompt message immediately
        if not contains_keywords:
            with st.chat_message("robot", avatar=robot_avator):
                st.markdown("我是食神周星星的唯一传人张小白，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？，我会告诉你具体的做法。")
            # Add robot response to chat history
            st.session_state.messages.append({"role": "robot", "content": "我是食神周星星的唯一传人张小白，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？，我会告诉你具体的做法。", "avatar": robot_avator})
        else:
            # Generate robot response
            with st.chat_message("robot", avatar=robot_avator):
                message_placeholder = st.empty()

                if enable_rag:
                    print("Question"+ prompt)
                    cur_response = qa_chain({"query": prompt})
                    print("检索问答链回答 question 的结果：")
                    print(cur_response)
                else:
                    print("Question"+ prompt)
                    cur_response = llm(prompt)
                    print("大模型回答 question 的结果：")
                    print(cur_response)


                #for cur_response in generate_interactive(
                #    model=model,
                #    tokenizer=tokenizer,
                #    prompt=real_prompt,
                #    # additional_eos_token_id=103028,
                #    additional_eos_token_id=92542,
                #    **asdict(generation_config),
                #):
                    # Display robot response in chat message container
                    #cur_response = cur_response.replace('\\n', '\n')
                    #message_placeholder.markdown(cur_response + "▌")
                message_placeholder.markdown(cur_response)
            # Add robot response to chat history
            st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
