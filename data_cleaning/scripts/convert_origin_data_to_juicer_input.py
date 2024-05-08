import os
import re
import json
from emoji import replace_emoji
from tqdm import tqdm
from functools import cmp_to_key


def compare(x, y):
    # 先按照dish(菜品)排序
    # 相同菜品按照name(菜名)长度排序
    # 方便后续查看去重效果
    if x['dish'] != y['dish']:
        if x['dish'] > y['dish']:
            return 1
        else:
            return -1
    else:
        return len(x['name']) - len(y['name'])


print("开始加载数据集")
# 原始数据集的后缀虽然是json，但实际上是jsonl格式，文件每行为一个json对象
# 所以这里需要逐行读取
data_path = os.environ.get('HOME') + "/cook-data/recipe_corpus_full.json"
f1 = open(data_path, 'r', encoding='utf-8')
json_data = []
for line in tqdm(f1.readlines()):
    # 去掉emoji
    line = replace_emoji(line)
    # 多个空格替换为一个空格
    line = re.sub(r"\s+", " ", line)
    json_data.append(json.loads(line))
f1.close()
print("加载数据集结束")

print("开始排序数据集")
json_data.sort(key=cmp_to_key(compare))
print("数据集排序结束")

print("开始保存数据集")
# data juicer的输入与输出文件格式都是jsonl
# 所以这里需要逐行写入
data_path_juicer = os.environ.get('HOME') + "/cook-data/recipe_corpus_juicer_input_name.jsonl"
f2 = open(data_path_juicer, 'w', encoding='utf-8')
for recipe in tqdm(json_data):
    f2.write(json.dumps(recipe, ensure_ascii=False) + '\n')
f2.close()
print("保存数据集结束")
