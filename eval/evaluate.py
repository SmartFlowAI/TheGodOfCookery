from collections import Counter
import re
import json
from tqdm import tqdm
from interface import load_retriever, load_model, load_chain


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


def evaluate_retriever():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    f1_sum = 0
    retriever = load_retriever()
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


def evaluate_model():
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


if __name__ == '__main__':
    evaluate_retriever()
    # evaluate_model()
