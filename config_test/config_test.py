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
    # 'base_model_type': "internlm2-chat-7b",
    # 'llm_model_path': os.environ.get('HOME') + "/models/zhanghuiATchina/zhangxiaobai_shishen2_full"
    # 'llm_model_path': os.environ.get('HOME') + "/models/Shanghai_AI_Laboratory/internlm-chat-7b"
    # 'llm_model_path': os.environ.get('HOME') + "/models/zhanghuiATchina/zhangxiaobai_shishen_full"
    # 'llm_model_path': "/mnt/d//models/zhanghuiATchina/zhangxiaobai_shishen_full",
}

# rag
Config['rag'] = {
    # 'rag_model_type': "chroma",
    'rag_model_type': "faiss",
    'vector_db': {
        'name': "faiss",
        'path': './rag/faiss_index'
    },
    'hf_emb_config': {
        'model_name': "F:/OneDrive/Pythoncode/BCE_model/bce-embedding-base_v1",
        'model_kwargs': {'device': 'cuda:0'},
        'encode_kwargs': {
            'batch_size': 32,
            'normalize_embeddings': True,
            'show_progress_bar': True,
            }
    },
    'retriever': {
        'db': {
            'search_type': "similarity",
            'search_kwargs': {"k": 5}
        },
        'bm25': {
            'pickle_path': './rag/retriever/bm25retriever.pkl',
            'search_kwargs': {"k": 5}
        }
    },
    'reranker': {
        'bce': {
            'model': 'F:/OneDrive/Pythoncode/BCE_model/bce-reranker-base_v1',
            'top_n': 3,
            'device': 'cuda:0',
            'use_fp16': True
        }
    }
}
