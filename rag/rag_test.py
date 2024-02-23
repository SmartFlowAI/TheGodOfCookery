# 导入必要的库
from langchain_community.llms.tongyi import Tongyi
from interface import load_vector_db, load_retriever, load_chain


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:  " + d.page_content for i, d in enumerate(docs)]))


def test_retriever(llm, verbose=True):
    # 加载检索器
    retriever = load_retriever(llm, verbose=verbose)

    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        # docs = vectordb.similarity_search_with_score(question)
        # print(docs)
        docs = retriever.get_relevant_documents(question)
        pretty_print_docs(docs)


def test_vector_db(vector_db_name="faiss"):
    # 加载检索器
    vectordb = load_vector_db(vector_db_name=vector_db_name)
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        docs = vectordb.similarity_search_with_score(question)
        print(docs)


def run_terminal(llm):
    qa_chain = load_chain(llm, vector_db_name="faiss", verbose=True)
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        predict = qa_chain({"question": question})
        print(predict["answer"])


if __name__ == '__main__':
    TONGYI_API_KEY = open("TONGYI_API_KEY.txt", "r").read().strip()
    verbose = True

    # 加载通义千问大语言模型
    llm = Tongyi(dashscope_api_key=TONGYI_API_KEY, temperature=0, model_name="qwen-turbo")
    # test_vector_db("faiss")
    # test_retriever(llm, verbose=verbose)
    # run_terminal(llm)
