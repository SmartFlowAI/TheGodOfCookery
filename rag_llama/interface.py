import pickle
from llama_index.core import Settings
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore
from rag_llama.HyQEFusionRerankRetriever import HyQEFusionRerankRetriever
from config import load_config


def load_embedding_model():
    # 加载编码模型
    bce_emb_config = load_config('rag_llama', 'bce_emb_config')
    embeddings = HuggingFaceEmbedding(**bce_emb_config)
    # 设置编码模型
    Settings.embed_model = embeddings


def load_vector_db():
    # 加载向量数据库
    save_path = load_config('rag_llama', 'faiss_config')['save_path']
    vector_store = FaissVectorStore.from_persist_dir(save_path)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=save_path)
    index = load_index_from_storage(storage_context=storage_context)
    return index


def load_retriever():
    # 加载向量检索器
    index = load_vector_db()
    faiss_top_k = load_config('rag_llama', 'faiss_config')['search_kwargs']['k']
    db_retriever = VectorIndexRetriever(index=index, similarity_top_k=faiss_top_k)
    # 加载BM25检索器
    bm25_config = load_config('rag_llama', 'bm25_config')
    bm25_save_path = bm25_config['save_path']
    bm25retriever = pickle.load(open(bm25_save_path, 'rb'))
    # 加载Reranker模型Config
    bce_reranker_config = load_config('rag_llama', 'bce_reranker_config')
    # 搭建HyQEFusionRerankRetriever
    return HyQEFusionRerankRetriever(retrievers=[db_retriever, bm25retriever], bce_reranker_config=bce_reranker_config)


class RagPipeline:
    def __init__(self):
        load_embedding_model()
        self.retriever = load_retriever()

    def get_retrieval_content(self, prompt) -> str:
        docs = self.retriever.retrieve(prompt)
        content = ""
        for doc in docs:
            content += doc.node.get_content() + "\n"
        return content.replace("\n\n", "\n")
