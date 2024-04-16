import pickle

# from llm_for_rag import load_model
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import PromptTemplate
from warnings import simplefilter
import sys

sys.path.append('..')
from config import load_config
# from config_test import load_config
from HyQEFusionRerankRetriever import HyQEFusionRerankRetriever
simplefilter("ignore")

qa_template = """你是一个经验丰富的大厨，善于根据用户需求给出食谱和做法。
可参考的菜品食谱：
---
{context_str}
---
如果参考中没有有效的信息，请根据你自己所掌握的知识进行回答。
若用户提供的非现实存在的菜谱，请你发挥自己的想象力回答
问题: {query_str}
回答: 
"""


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
    # TODO:要一个友好传入参数的方式，比如滑动Streamlit条设置top-
    # 加载向量检索器
    index = load_vector_db()
    faiss_top_k = load_config('rag_llama', 'faiss_config')['search_kwargs']['k']
    db_retriever = VectorIndexRetriever(index=index, similarity_top_k=faiss_top_k)
    # 加载BM25检索器
    bm25_config = load_config('rag_llama', 'bm25_config')
    bm25_load_path = bm25_config['save_path']
    bm25retriever = pickle.load(open(bm25_load_path, 'rb'))
    # 加载Reranker模型Config
    bce_reranker_config = load_config('rag_llama', 'bce_reranker_config')
    # 搭建HyQEFusionRerankRetriever
    return HyQEFusionRerankRetriever(retrievers=[db_retriever, bm25retriever], bce_reranker_config=bce_reranker_config)


def load_query_engine(retriever=None):
    if retriever is None:
        retriever = load_retriever()
    # 见https://www.bluelabellabs.com/blog/llamaindex-response-modes-explained/#:~:text=LlamaIndex%20has%205%20built%2Din,tree_summarize%2C%20accumulation%2C%20and%20simple_summarize.
    # 这里会损失部分信息，如超出上下文，可能就保留每个文档的n%，n为能容纳所有文档并不超过上下文的最大值，除非自定义retriever过程
    qa_prompt_tmpl = PromptTemplate(qa_template)
    Settings.llm = load_model()
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=qa_prompt_tmpl,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]
    )
    return query_engine


if __name__ == "__main__":
    load_embedding_model()
    retriever = load_retriever()
    d1 = retriever.retrieve("红烧滩羊肉的做法")
    print(d1)
