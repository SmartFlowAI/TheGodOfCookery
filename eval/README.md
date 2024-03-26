# RAG retriever评估模块使用说明
## 0. 正式评估前请先生成向量数据库，详见rag/README.md
## 1. 正式评估
## 1.1. 评估retriever
### 1.1.1. 指定评估数据集，修改evaluater.py第157行
```python
def evaluate_retriever():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
```
### 1.1.2. 启动在evaluate.py中的evaluate_retriever()函数
```python
if __name__ == '__main__':
    evaluate_retriever()
    # evaluate_model()
```
### 1.1.3. 运行evaluate.py




## 1.2. 评估模型回答
### 1.2.1. 指定评估数据集，修改evaluater.py第180行
```python
def evaluate_model():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
    # 评测模型耗时较长，可以只评测部分数据
        data = json.load(f)[:10]
```

### 1.2.2. 启动在evaluate.py中的evaluate_model()函数
```python
if __name__ == '__main__':
    # evaluate_retriever()
    evaluate_model()
```
### 1.2.3. 运行evaluate.py

# 注意：
## 当前向量数据库中只有1000条测试数据，如果需要评估完整数据集，请先生成完整向量数据库
## 当前评估数据集为eval_dataset_test.json，内容与向量数据库中的数据相同
## 正式评估数据集为eval_dataset.json，正式评估时手动替换evaluater.py中的路径即可使用
## 评测模型耗时较长，可以只评测部分数据，当前通过对数据集进行切片的方式实现只评测前10条数据
  
### 当前的测试评估结果：
评估retriever：  
F1 score sum: 993.7684651584043  
The number of data: 1000  
F1 average: 0.9937684651584043  
评估模型回答：  
F1 score sum: 5.221176107197454  
The number of data: 10  
F1 average: 0.5221176107197454 

# RAG测试模块使用说明
## 1. 测试vector db
启动test_rag.py中的test_vector_db()函数
```python
if __name__ == '__main__':
    test_vector_db()
    # test_retriever()
```

## 2. 测试retriever
启动test_rag.py中的test_retriever()函数
```python
if __name__ == '__main__':
    # test_vector_db()
    test_retriever()
```

