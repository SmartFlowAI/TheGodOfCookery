import os
from collections import defaultdict


def load_config(domain, key):
    return Config.get(domain).get(key, None)


# 总的config
Config = defaultdict(dict)

# global variables
Config['global'] = {
    'enable_rag': True,
    'streaming': None,
    'enable_markdown': None,
    'enable_image': None,
    'user_avatar': "images/user.png",
    'robot_avatar': "images/robot.png",
    'user_prompt': "<|User|>:{user}\n",
    'robot_prompt': "<|Bot|>:{robot}<eoa>\n",
    'cur_query_prompt': "<|User|>:{user}<eoh>\n<|Bot|>:",
    'error_response': "我是食神周星星的唯一传人，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？我会告诉你具体的做法。如果您遇到一些异常，请刷新页面重新提问。",
    'xlab_deploy': False
}

# llm
Config['llm'] = {
    'finetuned': True,
    'base_model_type': "internlm2-chat-1.8b",
    'llm_model_path': "F:/OneDrive/Pythoncode/BCE_model/internlm2-chat-1_8b"
}

# rag
Config['rag'] = {
    # 'rag_model_type': "chroma",
    'rag_model_type': "faiss",
    'dataset_config': {
        'data_path': "./data/tran_dataset_1000.json",  # 这里更换为完整的数据集路径
        'test_count': 1000  # 测试数据量，填入-1表示使用全部数据
    },
    'emb_strategy': {
        "source_caipu": False,  # 是否编码原始菜谱
        "HyQE": True,  # 是否使用HyQE
    },
    'faiss_config': {
        'save_path': './faiss_index',
        'load_path': './rag/faiss_index',
        'search_type': "similarity_score_threshold",
        'search_kwargs': {"k": 3, "score_threshold": 0.6}
    },
    'chroma_config': {
        'save_path': './chroma_db',
        'load_path': './rag/chroma_db',
        'search_type': "similarity",
        'search_kwargs': {"k": 3}
    },
    'bm25_config': {
        'dir_path': './retriever',
        'save_path': './retriever/bm25retriever.pkl',
        'pickle_path': './rag/retriever/bm25retriever.pkl',
        'search_kwargs': {"k": 3}
    },
    'bce_emb_config': {
        'model_name': "F:/OneDrive/Pythoncode/BCE_model/bce-embedding-base_v1",
        'model_kwargs': {'device': 'cuda:0'},
        'encode_kwargs': {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': True}
    },
    'bce_reranker_config': {
        'model': 'F:/OneDrive/Pythoncode/BCE_model/bce-reranker-base_v1',
        'top_n': 1,
        'device': 'cuda:0',
        'use_fp16': True
    }
}
