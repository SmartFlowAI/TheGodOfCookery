import json
from utils import *
def read_data():
    with open('./data_source.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def build_dataset():
    lines = read_data()
    filtered = filter(lines)
    formatted = format_data(filtered)
    print(len(formatted))
    final_dataset = []
    for raw in formatted:
        try:
            raw_split = raw.split('  ')
            dish = raw_split[0]
            intro = raw_split[1]
            query = f'{dish}怎么做'
            intro_split = intro.split(' ')
            if len(intro_split) > 1:
                recipe = ''.join(intro_split[:2])
                datum = {'query': query, 'gt':recipe}
                final_dataset.append(datum)
        except Exception:
            pass
    print(len(final_dataset))
    with open('./eval_dataset.json','w',encoding='utf-8') as f:
        json.dump(final_dataset,f,ensure_ascii=False, indent=4)

build_dataset()
