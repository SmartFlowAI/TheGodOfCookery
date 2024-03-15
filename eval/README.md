# RAG retriever评估模块使用说明
## 0. 正式评估前请先生成向量数据库，详见rag/README.md
## 1. 指定评估数据集，修改evaluater.py第98行
```python
def evaluate(retriever):
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
```

## 2. 运行evaluater.py


# 注意：
## 当前向量数据库中只有1000条测试数据，如果需要评估完整数据集，请先生成完整向量数据库
## 当前评估数据集为eval_dataset_test.json，内容与向量数据库中的数据相同
## 正式评估数据集为eval_dataset.json，正式评估时手动替换evaluater.py中的路径即可使用
  
### 当前的测试评估结果：
F1 score sum: 993.7684651584043  
The number of data: 1000  
F1 average: 0.9937684651584043  
