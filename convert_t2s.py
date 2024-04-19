#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 14:13
# @File    : convert_t2s.py
# requirements.txt OpenCC==1.1.6
import opencc


def convert_t2s(prompt):
    try:
        # 创建一个繁体到简体的转换器
        converter = opencc.OpenCC('t2s')
        # 将繁体字转换为简体字
        prompt_simplified = converter.convert(prompt)
    except Exception as e:
        print("convert prompt error ...", e)
        prompt_simplified = prompt
    return prompt_simplified


if __name__ == '__main__':
    prompt = "這是一段繁體字"
    prompt_simplified = convert_t2s(prompt)
    print(prompt_simplified)