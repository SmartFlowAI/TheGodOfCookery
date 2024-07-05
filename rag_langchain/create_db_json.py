# 首先导入所需第三方库
import json
import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
import sys
sys.path.append('..')
from config import load_config

dataset_config = load_config('rag_langchain', 'dataset_config')
data_path = dataset_config['data_path']
test_count = dataset_config['test_count']

print("开始加载数据集")
# 加载数据集
with open(data_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 取前test_count个元素, 用于测试
# test_count小于等于0时, 取全部元素
if test_count > 0:
    json_data = json_data[:test_count]
print("数据集加载完成")

emb_strategy = load_config('rag_langchain', 'emb_strategy')
assert emb_strategy['source_caipu'] or emb_strategy['HyQE'], "source_caipu and HyQE cannot be both False"

print("开始构建待编码文档集")
# 创建待编码文档集
split_docs = []
for i in range(len(json_data)):
    question = json_data[i]['conversation'][0]['input']
    # 如果input只有菜名，则加上“的做法”
    if "做" not in question:
        question += "的做法"
    answer = json_data[i]['conversation'][0]['output']
    # 加入原始菜谱
    if emb_strategy['source_caipu']:
        split_docs.append(Document(page_content=question + "\n" + answer))
    # 假设问题为“菜谱名+怎么做”
    # 加入假设问题，原始菜谱存放入metadata
    if emb_strategy['HyQE']:
        split_docs.append(Document(page_content=question, metadata={"caipu": question + "\n" + answer}))
print("待编码文档集构建完成")

print("开始加载编码模型")
# 加载编码模型
bce_emb_config = load_config('rag_langchain', 'bce_emb_config')
# 编码向量数据库时显示进度条
bce_emb_config["encode_kwargs"]["show_progress_bar"] = True
embeddings = HuggingFaceEmbeddings(**bce_emb_config)
print("编码模型加载完成")

print("开始构建BM25检索器")
# 构建BM25检索器
bm25_config = load_config('rag_langchain', 'bm25_config')
bm25retriever = BM25Retriever.from_documents(documents=split_docs)
bm25retriever.k = bm25_config['search_kwargs']['k']

# BM25Retriever序列化到磁盘
if not os.path.exists(bm25_config['dir_path']):
    os.mkdir(bm25_config['dir_path'])
pickle.dump(bm25retriever, open(bm25_config['save_path'], 'wb'))
print("BM25检索器构建完成")

print("开始编码向量数据库")
# 构建向量数据库
rag_model_type = load_config('rag_langchain', 'rag_model_type')
if rag_model_type == "chroma":
    vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings,
                                     persist_directory=load_config('rag_langchain', 'chroma_config')['save_path'])
    # 持久化到磁盘
    vectordb.persist()
else:
    faiss_index = FAISS.from_documents(documents=split_docs, embedding=embeddings,
                                       distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    # 保存索引到磁盘
    faiss_index.save_local(load_config('rag_langchain', 'faiss_config')['save_path'])
print("向量数据库编码完成")
