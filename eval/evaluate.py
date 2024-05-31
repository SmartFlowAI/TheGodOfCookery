from collections import Counter
import re
import json
from tqdm import tqdm
from interface import load_retriever, load_model, load_chain
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

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
    return f1
def hit_rate(retrived_docs, gt):
    # 遍历召回的所有doc，如果有一个doc的内容和gt一样，返回1，否则返回0
    for i in retrived_docs:
        if i.page_content == gt:
            return 1
    return 0
def MRR(retrived_doc, gt):
    # 遍历召回的所有doc，如果有一个doc的内容和gt一样，返回1/(index+1)，否则返回0
    for index, i in enumerate(retrived_doc):
        if i.page_content == gt:
            return 1/(index+1)
    return 0

def rouge_score(output, reference):
    rouge = Rouge()
    scores = rouge.get_scores(output, reference)
    return scores[0]['rouge-l']['f']

def bleu_score(output, reference):
    return sentence_bleu([reference.split()], output.split())

def evaluate_retriever():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    hit_rate_sum = 0
    MRR_sum = 0
    retriever = load_retriever()
    for d in tqdm(data):
        query = d["conversation"][0]['input'] 
        gt = d["conversation"][0]['output']
        retrived_docs = retriever.get_relevant_documents(query)
        print("--------------------输入：", query)
        print("--------------------输出：", gt)
        print("--------------------答案：", retrived_docs)
        hit_rate_sum += hit_rate(retrived_docs,gt)
        MRR_sum += MRR(retrived_docs,gt)

    print(f'hit rate sum: {hit_rate_sum}')
    print(f'MRR sum: {MRR_sum}')
    print(f'The number of data: {len(data)}')
    print(f'hit rate average: {hit_rate_sum / len(data)}')
    print(f'MRR average: {MRR_sum / len(data)}')


# def evaluate_retriever():
#     # 正式评估请手动替换下一行的路径
#     with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     f1_sum = 0
#     retriever = load_retriever()
#     for d in tqdm(data):
#         query = d["conversation"][0]['input']
#         docs = retriever.get_relevant_documents(query)
#         if len(docs) == 0:
#             output = ""
#         else:
#             output = docs[0].page_content
#         gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
#         # print("--------------------输入：", query)
#         # print("--------------------输出：", output)
#         # print("--------------------答案：", gt)
#         f1_sum += f1_score(output, gt)
#     print(f'F1 score sum: {f1_sum}')
#     print(f'The number of data: {len(data)}')
#     print(f'F1 average: {f1_sum / len(data)}')


# def evaluate_model():
#     # 正式评估请手动替换下一行的路径
#     with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
#         # 评测模型耗时较长，可以只评测部分数据
#         data = json.load(f)[:10]
#     f1_sum = 0
#     model, tokenizer, llm = load_model()
#     qa_chain = load_chain(llm)
#     for d in tqdm(data):
#         query = d["conversation"][0]['input']
#         output = qa_chain({"query": query})['result']
#         gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
#         print("--------------------输入：", query)
#         print("--------------------输出：", output)
#         print("--------------------答案：", gt)
#         f1_sum += f1_score(output, gt)
#     print(f'F1 score sum: {f1_sum}')
#     print(f'The number of data: {len(data)}')
#     print(f'F1 average: {f1_sum / len(data)}')

def evaluate_model():
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)[:10]
    rouge_sum = 0
    bleu_sum = 0
    model, tokenizer, llm = load_model()
    qa_chain = load_chain(llm)
    for d in tqdm(data):
        query = d["conversation"][0]['input']
        output = qa_chain({"query": query})['result']
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        rouge_sum += rouge_score(output, gt)
        bleu_sum += bleu_score(output, gt)
    print(f'ROUGE-L F1 average: {rouge_sum / len(data)}')
    print(f'BLEU score average: {bleu_sum / len(data)}')
if __name__ == '__main__':
    evaluate_retriever()
    evaluate_model()
