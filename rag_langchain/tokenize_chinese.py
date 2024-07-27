import jieba


def tokenize_chinese(text):
    # 我也不知道为什么，直接把jieba.lcut作为BM25Retriever.from_defaults的参数，会导致无法pickle序列化
    # 这里重新封装一下，就可以了
    return jieba.lcut(text)
