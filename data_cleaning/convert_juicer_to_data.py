import os
import json
from emoji import replace_emoji
from tqdm import tqdm

print("开始加载数据集")
data_path_juicer = os.environ.get('HOME') + "/cook-data/recipe_corpus_dedup.json"
f = open(data_path_juicer, 'r', encoding='utf-8')
json_data = []
# data juicer在保存时，以1000条数据为一组，每组数据单独保存在一行
# 所有这里需要两重循环
for line in f.readlines():
    for recipe in json.loads(line):
        json_data.append(recipe)
print("加载数据集结束")

result = []
print("开始转换数据集")
for recipe in tqdm(json_data):
    # 转化为dataset前对input和output字段去掉emoji
    result.append({"conversation": [{
        "system": "你是一个专业的厨师，你会做很多菜。用户报上自己所需的菜名后，你可以把做菜所需要的原料，以及做菜的方法告诉用户",
        "input": recipe['name'] + "的做法",
        "output": "您需要准备以下食材:\n" + replace_emoji(str(recipe['recipeIngredient'])) + "\n按以下方法制作:\n"
                  + replace_emoji(str(recipe['recipeInstructions'])) + "\n"}]
    })
print("转换数据集结束")

print("开始保存数据集")
data_path_cleaned = os.environ.get('HOME') + "/cook-data/dataset_large.json"
with open(data_path_cleaned, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("保存数据集结束")
