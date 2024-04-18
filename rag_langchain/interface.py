import pickle
from transformers.utils import logging
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from BCEmbedding.tools.langchain import BCERerank
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from rag_langchain.HyQEContextualCompressionRetriever import HyQEContextualCompressionRetriever
from config import load_config
# from config_test import load_config

logger = logging.get_logger(__name__)
chain_instance = None


def load_vector_db():
    # 加载编码模型
    bce_emb_config = load_config('rag_langchain', 'bce_emb_config')
    embeddings = HuggingFaceEmbeddings(**bce_emb_config)
    # 加载本地索引，创建向量检索器
    # 除非指定使用chroma，否则默认使用faiss
    rag_model_type = load_config('rag_langchain', 'rag_model_type')
    if rag_model_type == "chroma":
        vector_db_path = load_config('rag_langchain', 'chroma_config')['load_path']
        vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        vector_db_path = load_config('rag_langchain', 'faiss_config')['load_path']
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
    bm25_load_path = bm25_config['load_path']
    bm25retriever = pickle.load(open(bm25_load_path, 'rb'))
    bm25retriever.k = bm25_config['search_kwargs']['k']

    # 向量检索器与BM25检索器组合为集成检索器
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, db_retriever], weights=[0.5, 0.5])

    # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
    bce_reranker_config = load_config('rag_langchain', 'bce_reranker_config')
    reranker = BCERerank(**bce_reranker_config)
    # 依次调用ensemble_retriever与reranker，并且可以将替换假设问题为原始菜谱的Retriever
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


def load_chain_with_memory(llm, verbose=False):
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
    # RAG问答链
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=verbose, prompt=QA_CHAIN_PROMPT)

    # 基于大模型的问题生成器
    # 将多轮对话和问题整合为一个新的问题
    question_template = """以下是一段对话和一个后续问题，请将上下文和后续问题整合为一个新的问题。
对话内容:
···
{chat_history}
···
后续问题: {question}
新的问题:"""
    QUESTION_PROMPT = PromptTemplate(input_variables=["chat_history", "question"],
                                     template=question_template)
    question_generator = LLMChain(llm=llm, prompt=QUESTION_PROMPT, verbose=verbose)

    # 记录对话历史
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )

    # 多轮对话RAG问答链
    qa_chain = ConversationalRetrievalChain(
        question_generator=question_generator,
        retriever=retriever,
        combine_docs_chain=doc_chain,
        memory=memory,
        verbose=verbose
    )
    return qa_chain


class RagPipeline:
    def __init__(self):
        self.retriever = load_retriever()

    def get_retrieval_content(self, prompt) -> str:
        docs = self.retriever.get_relevant_documents(prompt)
        content = ""
        for doc in docs:
            content += doc.page_content + "\n"
        return content.replace("\n\n", "\n")
