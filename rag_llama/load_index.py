from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import faiss
import os
from tqdm import tqdm
from config import load_config

"""
TODO:对一般格式的文本进行适配，因为这里没有采用切chunk，而直接把一个问答当一个node
"""


def load_embedding_model():
    print("正在读取Embedding模型")
    # 设置全局embedding模型
    Settings.embed_model = HuggingFaceEmbedding(model_name=load_config("rag", "hf_emb_config")["model_name"],device="cuda:0")
    print("Done!")


def traindata2nodes():
    # 确保同目录下的dataset文件夹中有这个文件！
    if not os.path.exists("dataset/train_dataset_single.json"):
        print("当前dataset文件夹中没有原训练集json文件,使用1000个训练集的例子为替代")
        data_path = "dataset/example1000.json"
    else:
        data_path = "dataset/train_dataset_single.json"

    nodes = []
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for i in tqdm(range(len(json_data)), desc="Formating"):
        question = json_data[i]["conversation"][0]["input"]
        if "做" not in question:
            question += "的做法"
        answer = json_data[i]["conversation"][0]["output"]
        nodes.append(TextNode(text=question + "\n" + answer))
    return nodes


def init_index():
    load_embedding_model()
    if os.path.exists("./storage"):
        print("正在从本地读取索引")
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)
        print("Done!")
        return index
    else:
        nodes = traindata2nodes()
        faiss_index = faiss.index_factory(
            768, # embedding的维度，这里用的bce
            "HNSW64", # 检索算法，使用HNSW64: https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37
            faiss.METRIC_L2 # 相似度指标，使用L2距离
        ) 
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes, # 数据节点
            storage_context=storage_context, # 存储容器
            show_progress=True, # 是否显示进度
            insert_batch_size=10240, # 存储批量大小
        )
        index.storage_context.persist("./storage")
        print("索引存储完毕！")
        return index


if __name__ == "__main__":
    init_index()
