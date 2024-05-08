import json
from openai import OpenAI
import requests

prompt = """你是一个菜品名称提取器，熟悉各种菜品名称，现在给你一些含有冗余词语的菜品名称，冗余的词语可能是emoji或不必要的修饰语。
你需要按照“序号 提取结果”输出你的提取结果，例如：
输入：
10寸戚风蛋糕，云朵般柔软
10寸披萨🍕一个的做法
输出：
1 10寸戚风蛋糕
2 10寸披萨
现在，请你提取以下输入的菜品名称,注意，严格按照以上格式输出，不要输出多余字符：
{input}
"""

data_path = '../data/recipe_corpus_dedup.jsonl' # 修改成你的路径
batch_size = 20 # 每次送入大模型处理的数据量，可视情况修改

'''
Stage 1: Read the data
'''
def read_data():
    with open(data_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        names = [json.loads(l)['name'] for l in lines]
    return names

names = read_data()

'''
Stage 2: Load the data to the prompt
'''
def generate_prompt():
    global names
    i = 0
    while i < len(names[:200]):
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
            ]
        )
        names_with_idx = response.choices[0].message.content.split('\n') # 得到 “序号 菜品名称”
        for name in names_with_idx:
            splitted_name_idx = name.split(' ') # 分出 “菜品名称” 这一项
            if len(splitted_name_idx) != 2: # 回答格式有误，加入原名称
                print(name + '格式不正确, 加入原名称')
                extracted_names.append(names[i])
            else:
                extracted_names.append(splitted_name_idx[1])
        i += 1
    return extracted_names
    
def internlm2_20b():
    extracted_names = []
    i = 0
    for batch in generate_prompt():
        final_prompt = prompt.replace('{input}','\n'.join(batch))
        """
        TODO: 调用internlm2-chat-20b接口
        """
        name_with_idx = [] # 模型的回答
        for name in name_with_idx:
            splitted_name_idx = name.split(' ') # 分出 “菜品名称” 这一项
            if len(splitted_name_idx) != 2: # 回答格式有误，加入原名称
                print(name + '格式不正确, 加入原名称')
                extracted_names.append(names[i])
            else:
                extracted_names.append(splitted_name_idx[1])
        i += 1
    return extracted_names

'''
TODO: Stage 4: Store the extracted names to compare
'''


for batch in generate_prompt():
    print(prompt.replace('{input}',batch))