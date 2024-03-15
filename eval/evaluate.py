from collections import Counter
import re
import json
from tqdm import tqdm
import pickle
from transformers.utils import logging
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from BCEmbedding.tools.langchain import BCERerank
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
# from config import load_config
from config_test.config_test import load_config
from rag.HyQEContextualCompressionRetriever import HyQEContextualCompressionRetriever

logger = logging.get_logger(__name__)


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


def test_retriever(retriever, query) -> str:
    docs = retriever.get_relevant_documents(query)
    if len(docs) == 0:
        return ""
    return docs[0].page_content


def evaluate(retriever):
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    f1_sum = 0
    for d in tqdm(data):
        output = test_retriever(retriever, d["conversation"][0]['input'])
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')


if __name__ == '__main__':
    retriever = load_retriever()
    evaluate(retriever)
