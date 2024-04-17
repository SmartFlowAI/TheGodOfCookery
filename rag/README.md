# 创建向量数据库使用说明

## 1. 修改config/config.py
### 1.1 修改向量数据库类型
在config/config.py中修改`Config['rag']['rag_model_type']`的值，可选值为`"chroma"`和`"faiss"`，分别表示使用`chroma`数据库和`faiss`数据库。
```python
Config['rag'] = {
    # 'rag_model_type': "chroma", # 使用chroma数据库
    'rag_model_type': "faiss",  # 使用faiss数据库
}
```

### 1.2 修改数据集地址
在config/config.py中修改`Config['rag']['data_path']`的值，为完整数据集的地址。
test_count表示测试数据量，填入-1表示使用全部数据。
```python
Config['rag'] = {
    'dataset_config': {
        'data_path': "./data/tran_dataset_1000.json",  # 这里更换为完整的数据集路径
        'test_count': 1000  # 测试数据量，填入-1表示使用全部数据
    }
}
```

### 1.3 设置编码策略
在config/config.py中修改`Config['rag']['emb_strategy']`的值，可设定选项为`"source_caipu"`和`"HyQE"`
- `"source_caipu"`表示是否编码原始菜谱
- `"HyQE"`表示是否使用HyQE,编码假设问题
- 两者都为True表示同时使用两种编码策略
- 两者不能同时为False
```python
Config['rag'] = {
    'emb_strategy': {
        "source_caipu": False,  # 是否编码原始菜谱
        "HyQE": True,  # 是否使用HyQE
    }
}
```

## 2. 运行rag/create_db_json.py
一次运行只生成`Config['rag']['rag_model_type']`指定的数据库类型的数据库文件。  
如果需要切换数据库类型，请修改`Config['rag']['rag_model_type']`文件后，重新运行一次。

# 注意：
## 向量数据库只需要生成一次，之后可以直接使用。  
## 如果需要重新生成数据库，请首先删除`/rag/faiss_index`目录或`/rag/chroma_db`目录下的所有文件。  
## 然后重新运行`rag/create_db_json.py`。  
## 不先删除再生成，可能会触发向量数据库的合并，导致生成的数据库文件过大。
