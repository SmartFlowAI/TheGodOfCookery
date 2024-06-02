import json
from openai import OpenAI
import requests
import time
import json
import jsonlines
import random
import csv





prompt = """你是一个菜品名称提取器，熟悉各种菜品名称，现在给你一些含有冗余描述的菜品名称
你需要按照“序号 提取结果”输出你的提取结果，例如：
---
输入：
10寸戚风蛋糕，云朵般柔软
超简单家常版奶茶（龟苓膏奶茶）
香酥鱼排//日式七味虾-大宇空气煎炸杯食谱
金针菇这么做太好吃了，蒜香浓郁又爽口
红烧牛肉面 台湾经典 康师傅 牛腱肉 鲜香扑鼻 酥烂又有嚼劲
---
输出：
1 10寸戚风蛋糕
2 龟苓膏奶茶
3 香酥鱼排和日式七味虾
4 蒜香金针菇
5 红烧牛肉面
---
现在，请你提取以下输入的菜品名称,注意，严格按照以上格式输出，不要输出多余字符：
{input}"""

data_path = '../data/recipe_corpus_dedup_conversation.jsonl' # 修改成你的路径
batch_size = 10 # 每次送入大模型处理的数据量，可视情况修改

'''
Stage 0: preprocess the data
'''
# with open(data_path, 'r',encoding='utf-8') as f:
#     lines = f.readlines()
#     data = [json.loads(l) for l in lines]
#     new_data = []
#     for d in data:
#         d['input'] = d['input'].replace('\n','').replace('\r','')
#         new_data.append(d)
#     random.shuffle(new_data)
# with jsonlines.open(data_path,'w') as f:
#     for d in new_data:
#         f.write(d)





'''
Stage 1: Read the data
'''
def read_data():
    with open(data_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        names = [json.loads(l)['input'][:-3] for l in lines]
    return names

# names = read_data()

# with open('./original_name.txt','w',encoding='utf-8') as f:
#     names_with_newline = list(map(lambda s: s + '\n',names))
#     f.writelines(names_with_newline)





'''
Stage 2: Load the data to the prompt
'''
def generate_prompt():
    global names
    i = 0
    while i < len(names[:100]):
        batch = '\n'.join(names[i:i+batch_size])
        i += batch_size
        yield batch


'''
Stage 3: Feed into the LLMs
'''

def deepseek():
    global names

    with open('./deepseek_key.txt','r',encoding='utf-8') as f:
        DEEPSEEK_KEY = f.readline()
    extracted_names = []
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/")
    i = 0
    for batch in generate_prompt():
        final_prompt = prompt.replace('{input}',batch)
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                 {"role": "system", "content": ""},
                 {"role": "user", "content": final_prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
            stream=False
        )
        names_with_idx = response.choices[0].message.content.split('\n') # 得到 “序号 菜品名称”
        if len(names_with_idx) != batch_size:
            print(names_with_idx)
            print('数据长度有误，停止请求')
            break
        for name in names_with_idx:
            splitted_name_idx = name.strip().split(' ') # 分出 “菜品名称” 这一项
            if len(splitted_name_idx) < 2: # 回答格式有误，加入原名称
                print(name + '格式不正确, 加入原名称')
                extracted_names.append(names[i])
            else:
                extracted_names.append(' '.join(splitted_name_idx[1:])) # splitted_name_idx[1:] 防止菜品名称本身带有空格
            i += 1
        if i % 1000 == 999:
            print(i + 1)
        time.sleep(1)
    return extracted_names
    
def internlm2_20b():
    global names
    extracted_names = []
    # Based on the service deployed on intern-studio by LMdeploy
    client = OpenAI(api_key="KEY", base_url="http://localhost:23333/v1/")
    i = 0
    
    for batch in generate_prompt():
        final_prompt = prompt.replace('{input}','\n'.join(batch))
        response = client.chat.completions.create(
            model='internlm2',
            messages=[
                 {"role": "user", "content": final_prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
            stream=False
        )
        name_with_idx = response.choices[0].message.content.split('\n') # 模型的回答
        for name in name_with_idx:
            splitted_name_idx = name.split(' ') # 分出 “菜品名称” 这一项
            if len(splitted_name_idx) != 2: # 回答格式有误，加入原名称
                print(name + '格式不正确, 加入原名称')
                extracted_names.append(names[i])
            else:
                extracted_names.append(splitted_name_idx[1])
            i += 1
        time.sleep(1)
    return extracted_names

'''
Stage 4: Store the extracted names by deepseek to compare
'''
def store_extracted(api: str = 'deepseek'):
    if api == 'deepseek':
        extracted_names = deepseek()
    elif api == 'internlm2':
         extracted_names = internlm2_20b()
    else:
        raise NotImplementedError

    with open(f'./{api}_test_extracted_name.txt','w',encoding='utf-8') as f:
        extracted_names = list(map(lambda s: s + '\n',extracted_names))
        f.writelines(extracted_names)

'''
Stage 5: Store the original and extracted to compare
'''
def store_original_and_compare(api: str = 'deepseek'):
    with open('./original_name.txt','r',encoding='utf-8') as f:
        original_names = f.readlines()[:200]
    with open(f'./{api}_test_extracted_name.txt','r',encoding='utf-8') as f:
        extracted_names = f.readlines()

    with open(f'{api}_compare.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original','Deepseek'])
        for o, e in zip(original_names, extracted_names):
            writer.writerow([o, e])


names = read_data()
store_extracted(api='deepseek')
store_original_and_compare(api = 'deepseek')

