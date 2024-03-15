from collections import Counter
import re
import json

def de_punct(output: str):
    rule = re.compile(ur"[^a-zA-Z0-9\u4e00-\u9fa5]")
    output = rule.sub('',output)
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

def get_answer(model, query) -> str:
    # TODO: implement this to get the answer of the LLM
    pass

def evaluate(model):
    with open('./eval_dataset.json','r',encoding='') as f:
        data = json.load(f)
    f1_sum = 0
    for d in data:
        output = get_answer(model, d['query'])
        gt = d['gt']
        f1_sum += f1_score(output, gt)
    print(f'F1 score sum: {f1_sum}')
    print(f'The number of data: {len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')


model = None # replace with an LLM
evaluate(model)