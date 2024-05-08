import os
import json
from tqdm import tqdm

print("开始加载数据集")
# data juicer保存的输出文件是jsonl格式，这里需要逐行读取
data_path_name = os.environ.get('HOME') + "/cook-data/recipe_corpus_dedup_name.jsonl"
f1 = open(data_path_name, 'r', encoding='utf-8')
json_data = []
for line in tqdm(f1.readlines()):
    json_data.append(json.loads(line))
f1.close()
print("加载数据集结束")

result = []
print("开始转换数据集")
for recipe in tqdm(json_data):
    result.append({
        "input": recipe['name'] + "的做法",
        "output": ("您需要准备以下食材:\n" + str(recipe['recipeIngredient']) + "\n按以下方法制作:\n"
                  + str(recipe['recipeInstructions']) + "\n").replace('\n\n', '\n').replace('\\n', '\n')})
print("转换数据集结束")

print("开始保存数据集")
# data juicer的输入与输出文件格式都是jsonl
# 所以这里需要逐行写入
data_path_juicer = os.environ.get('HOME') + "/cook-data/recipe_corpus_juicer_input_conversation.jsonl"
f2 = open(data_path_juicer, 'w', encoding='utf-8')
for recipe in tqdm(result):
    f2.write(json.dumps(recipe, ensure_ascii=False) + '\n')
f2.close()
print("保存数据集结束")
