import os
import re

import jieba.posseg as pseg
# 特殊情况处理，添加用户自定义词典
# jieba.suggest_freq('番茄', True)

def split_ingredients(ingredient):
    ingredient = ingredient.replace('kg', '千克').replace('g', '克').replace('蒜', '蒜头')

    words = pseg.cut(ingredient)  # 使用posseg.cut进行分词和词性标注

    quantity = ""
    name = ""
    for word, flag in words:
        # 方法一：特殊量词处理
        regex_pattern = r"(\d+|[零一二两三四五六七八九十百千万亿半整适些少零壹貳贰叁參肆伍陆陸柒捌玖拾佰仟萬億兩]+)(勺|盒|箱|个|瓣|滴|克|块|量|许|根)"
        special_flag = bool(re.search(regex_pattern, word))
        # 方法二：词性判断
        if flag in ["m", "q", "x"] or special_flag:
            # "m"表示数词,"q"表示量词, "x"非中文词汇
            quantity += word
        elif "根" in word and len(word) > 1:
            name = word.replace("根", "")
            quantity += "根"
        else:
            name += word
    return quantity, name

def return_final_md(cur_response):
    try:
        # Extracting the ingredients and steps from the given response
        try:
            _, ingredients_raw, steps_raw = cur_response.split(":")
            ingredients_raw, _ = ingredients_raw.split("按")
        except Exception as e:
            print("error message is ...", e)
            return cur_response
        # Evaluating the strings to lists
        try:
            ingredients_list = eval(ingredients_raw.strip())
            steps_list = eval(steps_raw.strip())
        except Exception as e:
            print("error message is ...", e)
            return cur_response

        file_dir = os.path.dirname(os.path.abspath(__file__))
        # Generate markdown for ingredients table
        ingredients_md = "| 序号 | 数量 | 食材 ||\n| --- | --- | --- |---|\n"
        for i, item in enumerate(ingredients_list):
            try:
                quantity, name = split_ingredients(item)
                image_path = os.path.join(file_dir, f"src/{name}.png")
                if os.path.exists(image_path):
                    image_path_and_style = f"<img src={image_path} width = '50' height = '50' align=center />"
                else:
                    image_path_and_style = ""
                line = f"| {i + 1} | {quantity} | {name} |{image_path_and_style}|"
                ingredients_md += line + "\n"
            except Exception as e:
                print("error message is ...", e)

        # Generate markdown for steps table
        steps_md = "| 步骤 | 做法 |\n| --- | --- |\n"
        try:
            steps_md += "\n".join([f"| {i + 1} | {step} |" for i, step in enumerate(steps_list) if step != '好吃'])
        except Exception as e:
            print("error message is ...", e)

        # Combine both markdowns into one final markdown
        # Define the recipe name for the markdown output
        recipe_name = f'<span style="color: red;">您的菜谱</span>'
        # add ingredients and steps
        ingredients_title = f'<span style="color: green;">食材</span>'
        step_title = f'<span style="color: blue;">制作步骤</span>'
        final_md = f"# {recipe_name}\n\n## {ingredients_title}\n{ingredients_md}\n\n## {step_title}\n{steps_md}"
    except Exception as e:
        print("error message is ...", e)
        final_md = cur_response
    return final_md


# Execute the function and print the output
if __name__ == '__main__':

    cur_response = """
                    您需要准备以下食材:
                    ['1块鸡胸肉', '适量花生米', '1勺淀粉', '1勺料酒', '1勺生抽', '1勺老抽', '2勺醋', '1勺白糖', '1勺豆瓣酱', '1勺蚝油', '1勺食用油', '5片姜', '5瓣蒜', '1根葱','1勺盐', '5勺水']
                    按以下方法制作:
                    ['鸡胸肉切丁，放入一勺料酒，一勺生抽，一勺淀粉，一勺食用油腌制10分钟', '葱切段，姜切丝，蒜切沫', '花生米放微波炉高火2分钟，或者炒熟', '调碗汁：一勺生抽，一勺老抽，两勺醋，一勺蚝油，一勺白糖，一勺淀粉，5勺水搅拌均匀', '起锅烧油，放入腌制好的鸡肉，炒至变白', '加入一勺豆瓣酱，炒至出红油', '加入葱姜蒜炒香', '加入碗汁和花生米，翻炒均匀', '加入一勺盐', '翻炒均匀', '出锅', '好吃']
                    """

    final_md = return_final_md(cur_response)
    print(final_md)

    # import streamlit as st
    # final_md = return_final_md(cur_response)
    # st.markdown(final_md, unsafe_allow_html=True)
