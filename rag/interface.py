import copy
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional
import pickle
import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
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
from langchain.output_parsers import BooleanOutputParser
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from rag.HyQEContextualCompressionRetriever import HyQEContextualCompressionRetriever
from config import load_config
# from config_test.config_test import load_config

logger = logging.get_logger(__name__)
chain_instance = None


def load_vector_db():
    # 加载编码模型
    bce_emb_config = load_config('rag', 'bce_emb_config')
    embeddings = HuggingFaceEmbeddings(**bce_emb_config)
    # 加载本地索引，创建向量检索器
    # 除非指定使用chroma，否则默认使用faiss
    rag_model_type = load_config('rag', 'rag_model_type')
    if rag_model_type == "chroma":
        vector_db_path = load_config('rag', 'chroma_config')['load_path']
        vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        vector_db_path = load_config('rag', 'faiss_config')['load_path']
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
    pickle_path = bm25_config['pickle_path']
    bm25retriever = pickle.load(open(pickle_path, 'rb'))
    bm25retriever.k = bm25_config['search_kwargs']['k']

    # 向量检索器与BM25检索器组合为集成检索器
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, db_retriever], weights=[0.5, 0.5])

    #     # 创建带大模型过滤器的检索器，对集成检索器的结果进行过滤
    #     # TongYi api拒绝该请求，可能是禁止将大模型用于数据标注任务
    #     # 该检索器效率太低，已废弃
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
    bce_reranker_config = load_config('rag', 'bce_reranker_config')
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


@dataclass
class GenerationConfig:
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0


# 书生浦语大模型实战营提供的生成函数模版
# 搭配streamlit使用，实现markdown结构化输出与流式输出
@torch.inference_mode()
def generate_interactive(
        model,
        tokenizer,
        prompt,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        additional_eos_token_id: Optional[int] = None,
        **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]  # noqa: F841
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id  # noqa: F841
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


# 用于实现流式输出，实现难度太大，已放弃
@torch.inference_mode()
def generate_interactive_rag_stream(
        llm,
        question,
        verbose=False
):
    global chain_instance
    if chain_instance is None:
        chain_instance = load_chain(llm, verbose=verbose)
    # chain = chain | _get_answer
    for cur_response in chain_instance.stream({"query": question}):
        yield cur_response.get('result', '')


@torch.inference_mode()
def generate_interactive_rag(
        llm,
        question,
        verbose=False
):
    global chain_instance
    if chain_instance is None:
        chain_instance = load_chain(llm, verbose=verbose)
    return chain_instance({"query": question})['result']
