import json
from typing import List


def number_F2H(text: str):
    """
    Full-width number to half-width number
    """
    text = list(text)
    for i in range(len(text)):
        if 65296 <= ord(text[i]) <= 65305:
            text[i] = chr(ord(text[i]) - 65248)
    return ''.join(text)


def format_data(texts: List[str]):
    formatted = []
    for text in texts:
        formatted.append(number_F2H(text))
    return formatted


def filter(texts: List[str]):
    filtered = []
    for text in texts:
        text = text.strip()
        if text != '' and not '百类蔬菜营养食谱' in text:
            filtered.append(text)
    return filtered


def read_data():
    with open('./data_source.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def build_dataset():
    lines = read_data()
    filtered = filter(lines)
    formatted = format_data(filtered)
    print("len formatted: ", len(formatted))
    final_dataset = []
    for raw in formatted:
        try:
            raw_split = raw.split('  ')
            dish = raw_split[0]
            intro = raw_split[1]
            query = f'{dish}的做法'
            intro_split = intro.split(' ')
            if len(intro_split) > 1:
                recipe = ''.join(intro_split[:2])
                datum = {
                    "conversation": [
                        {
                            "system": "",
                            "input": query,
                            "output": recipe
                        }
                    ]
                }
                final_dataset.append(datum)
        except Exception:
            pass
    print("len final_dataset: ", len(final_dataset))
    with open('./eval_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    build_dataset()
