test_txt = open('source_data/text.txt', 'r', encoding='utf-8')
lines = test_txt.readlines()
caipu_list = []
temp_string = ''
# 逐行读取文本内容
# text.txt最后留超过一行的空行
# 避免最后一个菜谱无法加入到caipu_list中
for i in range(len(lines) - 1):
    if lines[i + 1].find('原料') != -1:
        caipu_list.append(temp_string)
        temp_string = ''
    temp_string += lines[i]

# 去掉第一个空行
caipu_list.pop(0)

test_txt.close()


# 去除多余字符
for i in range(len(caipu_list)):
    caipu_list[i] = caipu_list[i].replace('\n', '')
    caipu_list[i] = caipu_list[i].replace('\t', '')
    caipu_list[i] = caipu_list[i].replace(' ', '')
    caipu_list[i] = caipu_list[i].replace('  ', '')
    caipu_list[i] = caipu_list[i].replace('"', '')
    caipu_list[i] = caipu_list[i].replace('…', '')
    caipu_list[i] = caipu_list[i].replace('：', ':')

# 按内容去重并且保持原有顺序
new_caipu_list = list(set(caipu_list))
new_caipu_list.sort(key=caipu_list.index)

caipu_dict = {}
# 按菜名去重，同名菜谱保留最长的内容
for i in range(len(new_caipu_list)):
    caipu_split = new_caipu_list[i].split('原料:')
    if len(caipu_split) <= 1:
        continue
    caipu_name = caipu_split[0]
    caipu_content = "原料:" + caipu_split[1]
    if caipu_name not in caipu_dict:
        caipu_dict[caipu_name] = caipu_content
    elif len(caipu_dict[caipu_name]) < len(caipu_content):
        caipu_dict[caipu_name] = caipu_content

result_caipu_list = []

for key in caipu_dict:
    # 恢复换行格式
    caipu_dict[key] = caipu_dict[key].replace('原料:', ' 原料:')
    caipu_dict[key] = caipu_dict[key].replace('制作方法:', ' 制作方法:')
    caipu_dict[key] = caipu_dict[key].replace('特点:', ' 特点:')
    caipu_dict[key] = caipu_dict[key].replace('所属菜系:', ' 所属菜系:')
    result_caipu_list.append(key + " " + caipu_dict[key] + "\n\n")
    """
    # 单独保存各个菜谱的文件，过短的菜谱为异常格式，不保存
    if len(key) < 1 or len(caipu_dict[key]) < 5:
        continue
    else:
        file_name = key.replace('::', '').replace(':', '').replace('-', '')
        caipu_txt = open('data/'+file_name+'.txt', 'w', encoding='utf-8')
        caipu_txt.write(key + caipu_dict[key] + '\n')
        caipu_txt.close()
    """


# 将去重后的菜谱列表写入新的文件
caipu_txt = open('data/caipu.txt', 'w', encoding='utf-8')
for i in range(len(result_caipu_list)):
    caipu_txt.write(result_caipu_list[i])
caipu_txt.close()
