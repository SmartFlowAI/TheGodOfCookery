import sys
sys.path.append('..')

from collections import Counter
import re
import json
from tqdm import tqdm
from interface import load_retriever as load_retriever_lc
from interface import load_model, load_chain
from rag_llama.load_rag_engine import load_retriever as load_retriever_llama
from rag_llama.load_rag_engine import load_query_engine

import time
import random

def de_punct(output: str):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    output = rule.sub('', output)
    return output


def f1_score(output, gt):
    output = de_punct(output)
    gt = de_punct(gt)
    common = Counter(output) & Counter(gt)

    # Same words
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    # precision
    precision = 1.0 * num_same / len(output)

    # recall
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_retriever_lc():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    f1_sum = 0
    retriever = load_retriever_lc()
    for d in tqdm(data):
        query = d["conversation"][0]['input']
        docs = retriever.get_relevant_documents(query)
        if len(docs) == 0:
            output = ""
        else:
            output = docs[0].page_content
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        # print("--------------------输入：", query)
        # print("--------------------输出：", output)
        # print("--------------------答案：", gt)
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')


def evaluate_model_lc():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        # 评测模型耗时较长，可以只评测部分数据
        data = json.load(f)[:10]
    f1_sum = 0
    model, tokenizer, llm = load_model()
    qa_chain = load_chain(llm)
    for d in tqdm(data):
        query = d["conversation"][0]['input']
        output = qa_chain({"query": query})['result']
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        print("--------------------输入：", query)
        print("--------------------输出：", output)
        print("--------------------答案：", gt)
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')

def evaluate_retriever_llama():
    retriever = load_retriever_llama()
    with open('./eval_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    idx = random.randint(0, len(data) - 1)
    start = time.time()
    f1_sum = 0
    p_sum = 0
    r_sum = 0
    for i, d in enumerate(data):
        query = d["conversation"][0]['input']
        docs = retriever.retrieve(query)
        output = ""
        if len(docs) > 0:
            output = docs[0].metadata['answer']
        gt = d["conversation"][0]['output']
        if i == idx:
            print("输入：", query)
            print("输出：", output)
            print("答案：", gt)
        p, r, f1 = f1_score(str(output), gt)
        f1_sum += f1
        p_sum += p
        r_sum += r
    end = time.time()
    print(f'The number of data: {len(data)}')
    print(f'Precision Average: {p_sum / len(data)}')
    print(f'Recall Average: {r_sum / len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')
    print(f'Retrieval Time Average: {(end-start) / len(data)} s')

def evaluate_model_llama():
    query_engine = load_query_engine()
    with open('./eval_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    idx = random.randint(0, len(data) - 1)
    start = time.time()
    f1_sum = 0
    p_sum = 0
    r_sum = 0
    for i, d in enumerate(data):
        query = d["conversation"][0]['input']
        output = query_engine.query(query)
        gt = d["conversation"][0]['output']
        if i == idx:
            print("输入：", query)
            print("输出：", output)
            print("答案：", gt)
        p, r, f1 = f1_score(output, gt)
        f1_sum += f1
        p_sum += p
        r_sum += r
    end = time.time()
    print(f'The number of data: {len(data)}')
    print(f'Precision Average: {p_sum / len(data)}')
    print(f'Recall Average: {r_sum / len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')
    print(f'Retrieval Time Average: {(end-start) / len(data)} s')


    

if __name__ == '__main__':
    evaluate_retriever_llama()
    evaluate_model_llama()
