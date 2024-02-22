# 导入必要的库
import pickle

import gradio as gr
from BCEmbedding.tools.langchain import BCERerank
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import BooleanOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.llms.tongyi import Tongyi

TONGYI_API_KEY = "sk-c36e7b51417b44bc9f084c936c982815"

# 加载通义千问大语言模型
llm = Tongyi(dashscope_api_key=TONGYI_API_KEY, temperature=0, model_name="qwen-turbo")


# 加载internLM2模型
# llm = InternLM_LLM(model_path="internlm2-chat-7b")

def load_vector_db(db_name="faiss"):
    # 加载编码模型
    embedding_model_name = './model/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载本地索引，创建向量检索器
    if db_name == "chroma":
        vectordb = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
        db_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    else:
        vectordb = FAISS.load_local(folder_path='./faiss_index', embeddings=embeddings)
        db_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return db_retriever


def load_retriever(llm, verbose=False):
    # 加载本地索引，创建向量检索器
    db_retriever = load_vector_db("faiss")

    # 创建BM25检索器
    bm25retriever = pickle.load(open('./retriever/bm25retriever.pkl', 'rb'))
    bm25retriever.k = 5

    # 向量检索器与BM25检索器组合为集成检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25retriever, db_retriever], weights=[0.5, 0.5],
    )

    #     # 创建带大模型过滤器的检索器，对集成检索器的结果进行过滤
    #     filter_prompt_template = """以下是一段可参考的上下文和一个问题, 如果可参看上下文和问题相关请输出 YES , 否则输出 NO .
    # 可参考的上下文：
    # ···
    # {context}
    # ···
    # 问题: {question}
    # 相关性 (YES / NO):"""
    #
    #     FILTER_PROMPT_TEMPLATE = PromptTemplate(
    #         template=filter_prompt_template,
    #         input_variables=["question", "context"],
    #         output_parser=BooleanOutputParser(),
    #     )
    #     llm_filter = LLMChainFilter.from_llm(llm, prompt=FILTER_PROMPT_TEMPLATE, verbose=verbose)
    #     filter_retriever = ContextualCompressionRetriever(
    #         base_compressor=llm_filter, base_retriever=ensemble_retriever
    #     )

    # 创建带reranker的检索器，对大模型过滤器的结果进行再排序
    reranker_args = {'model': './model/bce-reranker-base_v1', 'top_n': 2, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
    return compression_retriever


def load_chain(llm, retriever, verbose=False):
    qa_template = """使用以下可参考的上下文来回答用户的问题。
可参考的上下文：
···
{context}
···
问题: {question}
如果你不知道答案，就说你不知道。如果给定的上下文无法让你做出回答，请回答无法从上下文找到相关内容。
有用的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=qa_template)

    question_template = """以下是一段对话和一个后续问题，请将上下文和后续问题整合为一个新的问题。
对话内容:
···
{chat_history}
···
后续问题: {question}
新的问题:"""
    QUESTION_PROMPT = PromptTemplate(input_variables=["chat_history", "question"],
                                     template=question_template)

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )

    question_generator = LLMChain(llm=llm, prompt=QUESTION_PROMPT, verbose=verbose)
    compression_retriever = retriever
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=verbose, prompt=QA_CHAIN_PROMPT)

    qa_chain = ConversationalRetrievalChain(
        question_generator=question_generator,
        retriever=compression_retriever,
        combine_docs_chain=doc_chain,
        memory=memory,
        verbose=True
    )
    return qa_chain


class Model_center():
    """
    存储问答 Chain 的对象
    """

    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history


def run_gradio():
    model_center = Model_center()

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>InternLM</center></h1>
                    <center>你的专属助手</center>
                    """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")

                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("Chat")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear console")

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                msg, chatbot], outputs=[msg, chatbot])

        gr.Markdown("""提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)
    gr.close_all()
    # 直接启动
    demo.launch()


def run_terminal():
    qa_chain = load_chain(llm, load_retriever(llm, verbose=True), verbose=True)
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        predict = qa_chain({"question": question})
        print(predict["answer"])


if __name__ == "__main__":
    # run_gradio()
    run_terminal()
