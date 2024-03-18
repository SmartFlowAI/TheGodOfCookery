from collections import Counter
import re
import json

import torch
from tqdm import tqdm

import pickle
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from BCEmbedding.tools.langchain import BCERerank
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from config import load_config
# from config_test.config_test import load_config
from rag.HyQEContextualCompressionRetriever import HyQEContextualCompressionRetriever
from rag.interface import GenerationConfig
from rag.CookMasterLLM import CookMasterLLM
from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_vector_db():
    # 加载编码模型
    bce_emb_config = load_config('rag', 'bce_emb_config')
    embeddings = HuggingFaceEmbeddings(**bce_emb_config)
    # 加载本地索引，创建向量检索器
    # 除非指定使用chroma，否则默认使用faiss
    rag_model_type = load_config('rag', 'rag_model_type')
    if rag_model_type == "chroma":
        vector_db_path = "../rag/chroma_db"
        vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        vector_db_path = "../rag/faiss_index"
        vectordb = FAISS.load_local(folder_path=vector_db_path, embeddings=embeddings)
    return vectordb


def load_retriever():
    # 加载本地索引，创建向量检索器
    vectordb = load_vector_db()
    rag_model_type = load_config('rag', 'rag_model_type')
    if rag_model_type == "chroma":
        db_retriever_config = load_config('rag', 'chroma_config')
    else:
        db_retriever_config = load_config('rag', 'faiss_config')
    db_retriever = vectordb.as_retriever(**db_retriever_config)

    # 加载BM25检索器
    bm25_config = load_config('rag', 'bm25_config')
    pickle_path = "../rag/retriever/bm25retriever.pkl"
    bm25retriever = pickle.load(open(pickle_path, 'rb'))
    bm25retriever.k = bm25_config['search_kwargs']['k']

    # 向量检索器与BM25检索器组合为集成检索器
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, db_retriever], weights=[0.5, 0.5])

    # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
    bce_reranker_config = load_config('rag', 'bce_reranker_config')
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


def de_punct(output: str):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    output = rule.sub('', output)
    return output


def f1_score(output, gt):
    output = de_punct(output)
    gt = de_punct(gt)
    common = Counter(output) & Counter(gt)

    # Same words
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    # precision
    precision = 1.0 * num_same / len(output)

    # recall
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_retriever():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    f1_sum = 0
    retriever = load_retriever()
    for d in tqdm(data):
        query = d["conversation"][0]['input']
        docs = retriever.get_relevant_documents(query)
        if len(docs) == 0:
            output = ""
        else:
            output = docs[0].page_content
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        # print("--------------------输入：", query)
        # print("--------------------输出：", output)
        # print("--------------------答案：", gt)
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')


def evaluate_model():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        # 评测模型耗时较长，可以只评测部分数据
        data = json.load(f)[:10]
    f1_sum = 0
    model, tokenizer, llm = load_model()
    qa_chain = load_chain(llm)
    for d in tqdm(data):
        query = d["conversation"][0]['input']
        output = qa_chain({"query": query})['result']
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        print("--------------------输入：", query)
        print("--------------------输出：", output)
        print("--------------------答案：", gt)
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')


if __name__ == '__main__':
    evaluate_retriever()
    # evaluate_model()
