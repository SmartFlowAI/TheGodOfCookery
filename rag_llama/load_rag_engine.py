from load_index import init_index
from llm_for_rag import load_model
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import PromptTemplate
from warnings import simplefilter

simplefilter("ignore")

qa_template = """你是一个经验丰富的大厨，善于根据用户需求给出食谱和做法。
可参考的菜品食谱：
---
{context_str}
---
如果参考中没有有效的信息，请根据你自己所掌握的知识进行回答。
问题: {query_str}
回答: \
"""


def load_retriever():
    index = init_index()
    # TODO:要一个友好传入参数的方式，比如滑动Streamlit条设置top-k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )
    return retriever


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
    retriever = load_retriever()
    print(retriever.retrieve("烤牛肉怎么做"))
    query_engine = load_query_engine(retriever)
    print(query_engine.query("烤牛肉怎么做"))
