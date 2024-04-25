# 基于LangChain的RAG

本文档简要介绍本项目基于LangChain的RAG系统

[TOC]

## 技术路线

这个RAG系统各部分的技术路线分别是：

- Embedding 模型：`BCE-Embedding`

- LLM基座：`InternLM2-Chat-1.8B` `InternLM2-Chat-7B`

- RAG框架：`LangChain`, 提供了检索和与大模型交互等接口，我们基于此框架实现了
  
  - 由团队成员 [**@乐正萌**](https://github.com/YueZhengMeng) 提出的 `HyQE` 检索技术，该检索技术的详细介绍见[此处](#使用基于HyQE的检索器)
  
  - 融合LangChain自带检索器和`BM25`检索器
  
  - 使用BCE-Reranker做检索后重排

- 向量数据库：
  
  - ~~`Chroma`~~: 轻量级的向量数据库，容易上手，但检索速度较慢。在项目二期开始时由团队成员 **[@Charles](https://github.com/SchweitzerGAO)** 使用此向量数据库建立RAG系统的最初版本，现已弃用。
  
  - `FAISS`：相较Chroma的主要优点是支持GPU加速，项目二期由团队成员 @乐正萌 完成 Chroma 到 FAISS 的迁移工作

## 环境搭建

```bash
# 克隆仓库
git clone https://github.com/SmartFlowAI/TheGodOfCookery

# 安装依赖
cd rag_langchain
pip install -r requirements.txt
```

## 使用指南

### 创建数据集

用于构建向量数据库的数据集中，每条数据需具有如下格式。保存时，请将所有数据保存为一个`json`文件：

```json
 {
    "conversation": [
      {
        "system": "",
        "input": "", // 此处填写input内容，即问题
        "output": ""  // 此处填写output内容，即答案
      }
    ]
  },
```

### 构建向量数据库

1. 修改配置文件

构建向量数据库之前，需要先修改`config/config.py`中`Config['rag_langchain']`中的内容，具体如下所示,可按自己的需求修改：

```python
# RAG with LangChain
Config['rag_langchain'] = {
    # 'rag_model_type': "chroma", # 使用chroma数据库
    'rag_model_type': "faiss",  # 使用faiss数据库
    'verbose': True,  # 是否打印详细的模型输入内容信息
    'dataset_config': {
        'data_path': "./data/tran_dataset_1000.json",  # 这里更换为完整的数据集路径
        'test_count': 1000  # 测试数据量，填入-1表示使用全部数据
    },
    'emb_strategy': {
        "source_caipu": False,  # 是否编码原始菜谱
        "HyQE": True,  # 是否使用HyQE
    },
    # streamlit加载使用的相对路径格式和直接运行python文件使用的相对路径格式不同
    'faiss_config': {
        'save_path': './faiss_index',  # 保存faiss索引的路径
        'load_path': './rag_langchain/faiss_index',  # streamlit加载faiss索引的路径
        'search_type': "similarity_score_threshold", # 搜索方式，指定相似度指标
        'search_kwargs': {"k": 3, "score_threshold": 0.6} # k: 保留top-k, score_threshold为相似度阈值
    },
    # chroma_config 键已弃用
    # 'chroma_config': {
    #     'save_path': './chroma_db',  # 保存chroma索引的路径
    #     'load_path': './rag_langchain/chroma_db',  # streamlit加载chroma索引的路径
    #     'search_type': "similarity", # 使用点积相似度标准搜索
    #     'search_kwargs': {"k": 3} # 保留top-k
    # },
    'bm25_config': {
        'dir_path': './retriever',  # 保存bm25检索器的文件夹的路径
        'save_path': './retriever/bm25retriever.pkl',  # 保存bm25检索器的路径
        'load_path': './rag_langchain/retriever/bm25retriever.pkl',  # streamlit加载bm25检索器的路径
        'search_kwargs': {"k": 3} # 保留top-k
    },
    'bce_emb_config': {
        'model_name': os.environ.get('HOME') + "/models/bce-embedding-base_v1", # 模型路径
        'model_kwargs': {'device': 'cuda:0'}, # 加载模型的其他可选参数，这里设置了加载到GPU
        'encode_kwargs': {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False} # 进行encode时的超参设置
    },
    'bce_reranker_config': {
        'model': os.environ.get('HOME') + "/models/bce-reranker-base_v1", # 模型路径
        'top_n': 1, # 保留top-n
        'device': 'cuda:0', # 加载设备
        'use_fp16': True # 模型参数精度是否使用fp16
    }
}

```

2. 运行建库脚本

```bash
python create_db_json.py
```

建库脚本主要执行以下步骤：

- 加载数据集

- 将数据集构建为LangChain文档集，由于`json`文件已经是高度结构化的文档，故无需切分，每一条`json`数据就可以作为一个文档加入文档集中

- 加载 `BCE-Embedding` 模型

- 根据文档集构建`BM25`检索器

- 根据文档集构建`FAISS`索引（即向量数据库）并持久化

完整代码实现见[此处](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/rag_langchain/create_db_json.py)

### 基于HyQE的检索器

如果你阅读了构建向量数据库的代码，你可能会注意到以下几行：

```python
# 加入原始菜谱
if emb_strategy['source_caipu']:
    split_docs.append(Document(page_content=question + "\n" + answer))
# 假设问题为“菜谱名+怎么做”
# 加入假设问题，原始菜谱存放入metadata
if emb_strategy['HyQE']:
    split_docs.append(Document(page_content=question, metadata={"caipu": question + "\n" + answer}))
```

这就是`HyQE`策略在建库时的体现。具体来说， `HyQE`（假设问题编码）策略的这个名字借鉴了`HyDE`(假设文档编码), 但思路有很大不同。一般的RAG系统在建立向量数据库时会将参考文档全部编码，这不仅增加了数据库的存储消耗，还可能降低检索准确度。**在本项目中**，考虑到用户的问题形式比较单一，我们可以只编码问题（“xxx怎么做”）而将答案作为问题的`metadata`，不进行编码。

在检索时，只需检索与用户输入最相似的问题，而后直接取检索结果的`metadata`作为参考文档即可，检索代码如下：

```python
def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        # 调用基础检索器(vector retriever与bm25 retriever)获取相关文档
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child(), **kwargs
        )
        if docs:
            # reranker重排序
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            # 上一步返回的compressed_docs本身就是list，langchain源码里又手动转换了一次
            # 可能是为了保证在top_n=1时返回值的类型同样是list
            compressed_docs = list(compressed_docs)
            # 遍历compressed_docs，将metadata中的caipu字段取出，放入page_content中
            for i in range(len(compressed_docs)):
                if "caipu" in compressed_docs[i].metadata:
                    # 取出metadata中的caipu字段，放入page_content中
                    compressed_docs[i].page_content = compressed_docs[i].metadata["caipu"]
                    compressed_docs[i].metadata = {}
            return compressed_docs
        else:
            return []
```

完整的检索器实现在[此处](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/rag_langchain/HyQEContextualCompressionRetriever.py)

如要使用这个检索器，可参考以下代码：

```python
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
```

## 后续计划

我们已经完成基于`LangChain`的基础功能并成功部署，后续我们将对RAG建库数据集和测试数据集进行清洗（包括去重和筛选等）并使用传统指标（`F1`，`BLEU`等）和评测框架（`RAGAs`等）对基于`LangChain`的RAG系统进行全面评估

## 相关链接

- BCE Embedding & BCE Reranker
  
  - [Embedding 模型](https://huggingface.co/maidalun1020/bce-embedding-base_v1)
  
  - [Reranker 模型](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
  
  - [Github 仓库](https://github.com/netease-youdao/BCEmbedding)

- InternLM2
  
  - [Chat-1.8B模型](https://huggingface.co/internlm/internlm2-chat-1_8b)
  
  - [Chat-7B模型](https://huggingface.co/internlm/internlm2-chat-7b)
  
  - [技术报告](https://arxiv.org/html/2403.17297v1)

- LangChain 
  
  - [文档](https://python.langchain.com/docs/get_started/introduction/)
  
  - [Github 仓库](https://github.com/langchain-ai/langchain)

- BM25
  
  - [简介]((https://en.wikipedia.org/wiki/Okapi_BM25)

- FAISS
  
  - [Github 仓库](https://github.com/facebookresearch/faiss)
  
  - [文档](https://faiss.ai/)
