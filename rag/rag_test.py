# 导入必要的库
from langchain_community.llms.tongyi import Tongyi
from run_local import load_retriever,load_chain


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:  " + d.page_content for i, d in enumerate(docs)]))


TONGYI_API_KEY = "sk-c36e7b51417b44bc9f084c936c982815"
verbose = True

# 加载通义千问大语言模型
llm = Tongyi(dashscope_api_key=TONGYI_API_KEY, temperature=0, model_name="qwen-turbo")

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
