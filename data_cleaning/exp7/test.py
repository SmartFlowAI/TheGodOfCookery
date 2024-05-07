import json

test_data_path = './test.json'

prompt = """你是一个菜品名称提取器，熟悉各种菜品名称，现在给你一些含有冗余词语的菜品名称，冗余的词语可能是emoji或不必要的修饰语。
你需要按照“序号 提取结果”输出你的提取结果，例如：
输入：
10寸戚风蛋糕，云朵般柔软
10寸披萨🍕一个的做法
输出：
1 戚风蛋糕
2 披萨
现在，请你提取以下输入的菜品名称,注意，严格按照以上格式输出，不要输出多余字符：
{input}
"""

with open(test_data_path,'r',encoding='utf-8') as f:
    data = json.load(f)
names = '\n'.join([d['name'] for d in data])
prompt = prompt.replace('{input}',names)


# 调用API