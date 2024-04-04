基于 llama-index 实现的 RAG

# 目前问题

1. 路径还没统一
2. 百万级数据用的方法读取很快，但存储占用过大，读取速度也慢（或许要进一步优化数据）
3. 似乎对话模板还有一定问题？
4. 待发现

在 rag_llama 目录中 运行`python load_rag_engine.py`可以测试

faiss-gpu==1.7.2
llama-index-embeddings-huggingface==0.2.0
llama-index-vector-stores-faiss==0.1.2
llama-index==0.10.26