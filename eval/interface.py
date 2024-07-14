from dataclasses import dataclass
from typing import Optional
import pickle
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from BCEmbedding.tools.langchain import BCERerank
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.append('..')
from rag_langchain.HyQEContextualCompressionRetriever import HyQEContextualCompressionRetriever
from rag_langchain.CookMasterLLM import CookMasterLLM
from config import load_config


@dataclass
class GenerationConfig:
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0


def load_vector_db():
    # 加载编码模型
    bce_emb_config = load_config('rag_langchain', 'bce_emb_config')
    embeddings = HuggingFaceEmbeddings(**bce_emb_config)
    # 加载本地索引，创建向量检索器
    # 除非指定使用chroma，否则默认使用faiss
    rag_model_type = load_config('rag_langchain', 'rag_model_type')
    if rag_model_type == "chroma":
        vector_db_path = "../rag_langchain/chroma_db"
        vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        vector_db_path = "../rag_langchain/faiss_index"
        vectordb = FAISS.load_local(folder_path=vector_db_path, embeddings=embeddings)
    return vectordb


def load_retriever():
    # 加载本地索引，创建向量检索器
    vectordb = load_vector_db()
    rag_model_type = load_config('rag_langchain', 'rag_model_type')
    if rag_model_type == "chroma":
        db_retriever_config = load_config('rag_langchain', 'chroma_config')
    else:
        db_retriever_config = load_config('rag_langchain', 'faiss_config')
    db_retriever = vectordb.as_retriever(**db_retriever_config)

    # 加载BM25检索器
    bm25_config = load_config('rag_langchain', 'bm25_config')
    bm25_load_path = "../rag_langchain/retriever/bm25retriever.pkl"
    bm25retriever = pickle.load(open(bm25_load_path, 'rb'))
    bm25retriever.k = bm25_config['search_kwargs']['k']

    # 向量检索器与BM25检索器组合为集成检索器
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, db_retriever], weights=[0.5, 0.5])

    # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
    bce_reranker_config = load_config('rag_langchain', 'bce_reranker_config')
    reranker = BCERerank(**bce_reranker_config)
    # 可以替换假设问题为原始菜谱的Retriever
    compression_retriever = HyQEContextualCompressionRetriever(base_compressor=reranker,
                                                               base_retriever=ensemble_retriever)
    return compression_retriever


def load_chain(llm, verbose=False):
    # 加载检索器
    retriever = load_retriever()

    # RAG对话模板
    qa_template = """先对上下文进行内容总结,再使上下文来回答用户的问题。总是使用中文回答。
可参考的上下文：
···
{context}
···
问题: {question}
如果给定的上下文无法让你做出回答，请根据你自己所掌握的知识进行回答。
有用的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=qa_template)

    # 单轮对话RAG问答链
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": verbose},
                                           retriever=retriever)

    return qa_chain


def load_model():
    llm_model_path = load_config('llm', 'llm_model_path')
    base_model_type = load_config('llm', 'base_model_type')
    print(f"base model type:{base_model_type}")
    max_length = 32768
    if base_model_type == 'internlm-chat-7b':
        generation_config = GenerationConfig(
            max_length=max_length)  # InternLM1
    elif base_model_type == 'internlm2-chat-1.8b':
        generation_config = GenerationConfig(
            max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.17)  # InternLM2 1.8b need 惩罚参数
    else:
        generation_config = GenerationConfig(
            max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)  # InternLM2 2 need 惩罚参数
    # int4 量化加载
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("正在从本地加载模型...")
    print(llm_model_path)
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                                 device_map="auto",
                                                 quantization_config=quantization_config).eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
    llm = CookMasterLLM(model, tokenizer)
    model.generation_config.max_length = generation_config.max_length
    model.generation_config.top_p = generation_config.top_p
    model.generation_config.temperature = generation_config.temperature
    model.generation_config.repetition_penalty = generation_config.repetition_penalty
    print(model.generation_config)
    print("完成本地模型的加载")
    return model, tokenizer, llm
