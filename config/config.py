import os
from collections import defaultdict

# 总的config
Config = defaultdict(dict)

# global variables
Config['global'] = {
    'enable_rag': None,
    'streaming': None,
    'enable_markdown': None,
    'enable_image': None,
    'user_avatar': "images/user.png",
    'robot_avatar': "images/robot.png",
    'user_prompt': "<|User|>:{user}\n",
    'robot_prompt': "<|Bot|>:{robot}<eoa>\n",
    'cur_query_prompt': "<|User|>:{user}<eoh>\n<|Bot|>:",
    'error_response': "我是食神周星星的唯一传人，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？我会告诉你具体的做法。如果您遇到一些异常，请刷新页面重新提问。",
    'xlab_deploy': True
}

# llm
Config['llm'] = {
    'finetuned': True,
    # 'load_4bit': True,
    'load_4bit': True,

    # 1.8b 二代
    # 'base_model_type': "internlm2-chat-1.8b",
    # finetuned = True
    # 'llm_model_path': os.environ.get('HOME') + "/models/zhanghuiATchina/zhangxiaobai_shishen2_1_8b",
    # finetuned = False
    # 'llm_model_path': os.environ.get('HOME') + "/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",

    # 7b 二代
    'base_model_type': "internlm2-chat-7b",
    # finetuned = True
    'llm_model_path': os.environ.get('HOME') + "/models/zhanghuiATchina/zhangxiaobai_shishen2_full",
    # finetuned = False
    # 'llm_model_path': os.environ.get('HOME') + "/models/Shanghai_AI_Laboratory/internlm2-chat-7b",
    # 'llm_model_path': "/mnt/d/models/internlm/internlm2-chat-7b",

    # 7b 一代
    # 'base_model_type': "internlm-chat-7b",
    # finetuned = True
    # 'llm_model_path': os.environ.get('HOME') + "/models/“ + ”zhanghuiATchina/zhangxiaobai_shishen_full"
    # 'llm_model_path': "/mnt/d//models/zhanghuiATchina/zhangxiaobai_shishen_full",
    # finetuned = False
    # 'llm_model_path': os.environ.get('HOME') + "/models/Shanghai_AI_Laboratory/internlm-chat-7b"
}

# speech
Config['speech'] = {
    # 'speech_model_type':"whisper",
    'speech_model_type': "paraformer",
    'audio_save_path': "/tmp/audio.wav",
    'whisper_model_scale': "medium",
    # 'whisper_model_path': "",
    'speech_model_path': os.environ.get(
        'HOME') + "/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
}

# rag
Config['rag'] = {
    'rag_model_type': "faiss",
    # 'rag_model_type': "chroma",
    'verbose': True,
    'vector_db': {
        'name': "faiss",
        'path': './rag/faiss_index'
    },
    'hf_emb_config': {
        'model_name': os.environ.get('HOME') + "/models/bce-embedding-base_v1",
        'model_kwargs': {'device': 'cuda:0'},
        'encode_kwargs': {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': True}
    },
    'retriever': {
        'db': {
            'search_type': "similarity_score_threshold",
            'search_kwargs': {"k": 3, "score_threshold": 0.6}
        },
        'bm25': {
            'pickle_path': './rag/retriever/bm25retriever.pkl',
            'search_kwargs': {"k": 3}
        }
    },
    'reranker': {
        'bce': {
            'model': os.environ.get('HOME') + '/models/bce-reranker-base_v1',
            'top_n': 1,
            'device': 'cuda:0',
            'use_fp16': True
        }
    }
}

# 文生图部分config
Config['image'] = {
    'image_model_type': 'stable-diffusion',  # stable-diffusion or glm-4
    'image_model_config': {
        'stable-diffusion': {
            "model_path": os.environ.get('HOME') + "/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
            "lora_path": 'gen_image/lora_weights/meishi.safetensors',
            "lora_scale": 0.75  # range[0.,1.]
        },
        'glm-4': {
            "api_key": "*****"
        }
    }
}
