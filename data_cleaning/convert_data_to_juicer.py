import os
import json
from tqdm import tqdm
from functools import cmp_to_key

print("开始加载数据集")
data_path = os.environ.get('HOME') + "/cook-data/recipe_corpus_full.json"
f = open(data_path, 'r', encoding='utf-8')
json_data = []
for line in f.readlines():
    json_data.append(json.loads(line))
print("加载数据集结束")

print("开始排序数据集")


def compare(x, y):
    # 先按菜品排序
    # 同一道菜品按名字长度排序
    if x['dish'] != y['dish']:
        if x['dish'] > y['dish']:
            return 1
        else:
            return -1
    else:
        return len(x['name']) - len(y['name'])


json_data.sort(key=cmp_to_key(compare))
print("数据集排序结束")

result = []
print("开始转换数据集")
# 原计划在这里去掉多余字段
# 现放弃，转换为最终数据集时再处理
for recipe in tqdm(json_data):
    # result.append({'name': recipe['name'], 'recipeIngredient': recipe['recipeIngredient'], 'recipeInstructions': recipe['recipeInstructions']})
    result.append(recipe)
print("转换数据集结束")

print("开始保存数据集")
data_path_juicer = os.environ.get('HOME') + "/cook-data/recipe_corpus_juicer.json"
with open(data_path_juicer, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("保存数据集结束")