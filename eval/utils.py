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
