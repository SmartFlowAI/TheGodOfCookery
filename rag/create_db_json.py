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

# 正式在完整数据集上构建向量数据库时
# 1. data_path为你的数据集路径
# 2. 修改embedding_model_name为你的embedding模型路径
data_path = "./data/tran_dataset_1000.json"
embedding_model_name = 'F:/OneDrive/Pythoncode/BCE_model/bce-embedding-base_v1'

split_docs = []
with open(data_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

for i in range(len(json_data)):
    question = json_data[i]['conversation'][0]['input']
    if "做" not in question:
        question += "的做法"
    answer = json_data[i]['conversation'][0]['output']
    # 加入原始菜谱
    split_docs.append(Document(page_content=question+"\n"+answer))
    # 假设问题为“菜谱名+怎么做”
    # 加入假设问题，原始菜谱存放入metadata
    split_docs.append(Document(page_content=question, metadata={"caipu": question+"\n"+answer}))


# 构建向量数据库

embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': True}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

bm25retriever = BM25Retriever.from_documents(documents=split_docs)
bm25retriever.k = 5

faiss_index = FAISS.from_documents(documents=split_docs, embedding=embeddings,
                                   distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory='./chroma_db')


# BM25Retriever序列化到磁盘
if not os.path.exists("./retriever"):
    os.mkdir("./retriever")
pickle.dump(bm25retriever, open('./retriever/bm25retriever.pkl', 'wb'))

# 保存索引到磁盘
faiss_index.save_local('./faiss_index')

# 将加载的向量数据库持久化到磁盘上
vectordb.persist()

