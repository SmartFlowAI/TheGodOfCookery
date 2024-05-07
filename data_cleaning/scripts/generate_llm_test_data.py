import json
import random

n = 50
test_data = []
with open('../data/dataset_large_dedup.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    random.shuffle(lines)
    for l in lines:
        if n <= 0:
            break
        data = json.loads(l)
        if data['dish'] == 'Unknown':
            test_data.append(data)
            n -= 1
with open('../exp7/test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
