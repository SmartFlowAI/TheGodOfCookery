import json
import os
import pickle
import faiss
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
import sys
sys.path.append('..')
from config import load_config
from rag_llama.tokenize_chinese import tokenize_chinese  # 这个路径要写完整，不然pickle load的时候会找不到tokenize_chinese

dataset_config = load_config('rag_llama', 'dataset_config')
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
print("共有{}条数据".format(len(json_data)))
print("数据集加载完成")

emb_strategy = load_config('rag_llama', 'emb_strategy')
assert emb_strategy['source_caipu'] or emb_strategy['HyQE'], "source_caipu and HyQE cannot be both False"

print("开始构建待编码文档集")
# 创建待编码文本节点集
nodes = []
for i in range(len(json_data)):
    question = json_data[i]['conversation'][0]['input']
    # 如果input只有菜名，则加上“的做法”
    if "做" not in question:
        question += "的做法"
    answer = json_data[i]['conversation'][0]['output']
    # 加入原始菜谱
    if emb_strategy['source_caipu']:
        nodes.append(TextNode(text=question + "\n" + answer))
    # 假设问题为“菜谱名+怎么做”
    # 加入假设问题，原始菜谱存放入metadata
    if emb_strategy['HyQE']:
        HyQE_textNode = TextNode(text=question, metadata={"caipu": question + "\n" + answer})
        # 设置metadata中的answer不参与LLM的读取
        HyQE_textNode.excluded_llm_metadata_keys = ["answer"]
        # 设置metadata中的answer不参与编码和检索
        HyQE_textNode.excluded_embed_metadata_keys = ["answer"]
        nodes.append(HyQE_textNode)
print("待编码文档集构建完成")

print("开始加载编码模型")
# 加载编码模型
bce_emb_config = load_config('rag_llama', 'bce_emb_config')
embeddings = HuggingFaceEmbedding(**bce_emb_config)
# 设置编码模型
Settings.embed_model = embeddings
print("编码模型加载完成")

print("开始构建BM25检索器")
# 构建BM25检索器
bm25_config = load_config('rag_llama', 'bm25_config')
bm25_top_k = bm25_config['search_kwargs']['k']
bm25retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=bm25_top_k, tokenizer=tokenize_chinese)

# BM25Retriever序列化到磁盘
if not os.path.exists(bm25_config['dir_path']):
    os.mkdir(bm25_config['dir_path'])
pickle.dump(bm25retriever, open(bm25_config['save_path'], 'wb'))
print("BM25检索器构建完成")

print("开始编码向量数据库")
# 构建向量数据库
faiss_index = faiss.index_factory(768, "HNSW64", faiss.METRIC_L2)  # embedding的维度，这里用的bce_embedding的维度
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
save_path = load_config('rag_llama', 'faiss_config')['save_path']
index.storage_context.persist(save_path)
print("向量数据库编码完成")
