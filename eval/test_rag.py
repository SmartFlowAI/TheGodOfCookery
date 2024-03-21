from interface import load_vector_db, load_retriever


def test_vector_db():
    # 加载检索器
    vectordb = load_vector_db()
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        docs = vectordb.similarity_search_with_score(question, top_k=3)
        for doc in docs:
            print("page_content: ", doc[0].page_content)
            if "caipu" in doc[0].metadata:
                print("caipu in metadata: ", doc[0].metadata["caipu"])
            print("similarity score: ", doc[1])


def test_retriever():
    # 加载检索器
    retriever = load_retriever()

    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        docs = retriever.get_relevant_documents(question)
        for doc in docs:
            print("page_content: ", doc.page_content)


if __name__ == '__main__':
    # test_vector_db()
    test_retriever()
