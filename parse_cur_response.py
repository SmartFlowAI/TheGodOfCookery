import re

def split_ingredients(ingredient):            
    # 特殊处理，如"适量"等非数字开头的情况
    if ingredient.startswith('适量'):
        quantity = '适量'
        name = ingredient[2:].strip()
    elif ingredient.startswith('些许'):
        quantity = '些许'
        name = ingredient[2:].strip()
    else:
        quantity = ingredient[:2].strip()
        name = ingredient[2:].strip()
    # 对于没有明确分隔符的特殊情况，进一步处理
    if not name:  # 如果没有名称，说明没有分隔符，整个字符串可能是名称
        name = ingredient
    if not quantity:  # 如果没有提取到数量，可能是适量等特殊情况
        quantity = '适量'
        name = ingredient
    return quantity, name



def return_final_md(cur_response): 
    try:
        # Extracting the ingredients and steps from the given response
        _, ingredients_raw, steps_raw = cur_response.split(":")
        ingredients_raw, _ = ingredients_raw.split("按")
    
        # Evaluating the strings to lists
        ingredients_list = eval(ingredients_raw.strip())
        steps_list = eval(steps_raw.strip())
    
        # Define the recipe name for the markdown output
        recipe_name = "您的菜谱"
    
        # Generate markdown for ingredients table
        ingredients_md = "| 序号 | 数量 | 食材 ||\n| --- | --- | --- |---|\n"
        ingredients_md += "\n".join([f"| {i+1} | {split_ingredients(item)[0]} | {split_ingredients(item)[1]} |<img src='./src/{split_ingredients(item)[1]}.png' width = '50' height = '50' align=center />|" 
                                    for i, item in enumerate(ingredients_list)])
    
        # Generate markdown for steps table
        steps_md = "| 步骤 | 做法 |\n| --- | --- |\n"
        steps_md += "\n".join([f"| {i+1} | {step} |" for i, step in enumerate(steps_list) if step != '好吃'])
    
        # Combine both markdowns into one final markdown
        final_md = f"# {recipe_name}\n\n## 食材\n{ingredients_md}\n\n## 制作步骤\n{steps_md}"
    except Exception as e:
        print("error message is ...", e)
        final_md = cur_response
    return final_md

# Execute the function and print the output
if __name__ == '__main__':
    cur_response = """
                    您需要准备以下食材:
                    ['1块鸡胸肉', '适量花生米', '1勺淀粉', '1勺料酒', '1勺生抽', '1勺老抽', '2勺醋', '1勺白糖', '1勺豆瓣酱', '1勺蚝油', '1勺食用油', '5片姜', '5瓣蒜', '1根葱', '1勺盐', '5勺水']
                    按以下方法制作:
                    ['鸡胸肉切丁，放入一勺料酒，一勺生抽，一勺淀粉，一勺食用油腌制10分钟', '葱切段，姜切丝，蒜切沫', '花生米放微波炉高火2分钟，或者炒熟', '调碗汁：一勺生抽，一勺老抽，两勺醋，一勺蚝油，一勺白糖，一勺淀粉，5勺水搅拌均匀', '起锅烧油，放入腌制好的鸡肉，炒至变白', '加入一勺豆瓣酱，炒至出红油', '加入葱姜蒜炒香', '加入碗汁和花生米，翻炒均匀', '加入一勺盐', '翻炒均匀', '出锅', '好吃']
                    """

    # final_md = return_final_md(cur_response)
    # print(final_md)

    # import streamlit as st
    # final_md = return_final_md(cur_response)
    # st.markdown(final_md, unsafe_allow_html=True)
