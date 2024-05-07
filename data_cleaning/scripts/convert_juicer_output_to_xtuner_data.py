import os
import json
from tqdm import tqdm

print("开始加载数据集")
# data juicer保存的输出文件是jsonl格式，这里需要逐行读取
data_path_juicer = os.environ.get('HOME') + "/cook-data/recipe_corpus_dedup.jsonl"
f1 = open(data_path_juicer, 'r', encoding='utf-8')
json_data = []
for line in tqdm(f1.readlines()):
    json_data.append(json.loads(line))
f1.close()
print("加载数据集结束")

result = []
print("开始转换数据集")
for recipe in tqdm(json_data):
    result.append({"conversation": [{
        "system": "你是一个专业的厨师，你会做很多菜。用户报上自己所需的菜名后，你可以把做菜所需要的原料，以及做菜的方法告诉用户",
        "input": recipe['name'] + "的做法",
        "output": "您需要准备以下食材:\n" + str(recipe['recipeIngredient']) + "\n按以下方法制作:\n"
                  + str(recipe['recipeInstructions']) + "\n"}]
    })
print("转换数据集结束")

print("开始保存数据集")
# xtuner数据集格式是json
# 详细见：https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_format.md
data_path_cleaned = os.environ.get('HOME') + "/cook-data/dataset_large.json"
with open(data_path_cleaned, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("保存数据集结束")
