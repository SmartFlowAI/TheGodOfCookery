import os
from collections import defaultdict

# 总的config
Config = defaultdict(dict)

# global variables
Config['global']= {
    'enable_rag': None,
    'streaming': None,
    'enable_markdown': None,
    'enable_image': None,
    'user_avatar': "images/user.png",
    'robot_avatar': "images/robot.png",
    'user_prompt': "<|User|>:{user}\n",
    'robot_prompt': "<|Bot|>:{robot}<eoa>\n",
    'cur_query_prompt': "<|User|>:{user}<eoh>\n<|Bot|>:",
    'error_response': "我是食神周星星的唯一传人，我什么菜都会做，包括黑暗料理，您可以问我什么菜怎么做———比如酸菜鱼怎么做？我会告诉你具体的做法。如果您遇到一些异常，请刷新页面重新提问。"

}

# llm
Config['llm'] = {
    'llm_model_path': "zhanghuiATchina/zhangxiaobai_shishen2_full"
}

# speech
Config['speech'] = {
    'audio_save_path': "/tmp/audio.wav", 
    'whisper_model_scale': "medium"   
}

# rag
Config['rag'] = {
    'project_path': "/root/TheGodOfCookery/",    
    'model_dir_path': "/root/models/",  
    'bm25retriever_file': "/root/TheGodOfCookery/rag/retriever/bm25retriever.pkl",  
    'embedding_model_path': os.environ.get('HOME') + "/models/bce-embedding-base_v1", 
    'reranker_model_path': os.environ.get('HOME') + "/models/bce-reranker-base_v1", 
    'vectordb_chroma_path': "/root/TheGodOfCookery/rag/chroma_db", 
    'vectordb_faiss_path': "/root/TheGodOfCookery/rag/faiss_index"  
}

# 文生图部分config
Config['image'] = {
    'image_model_type': 'stable-diffusion',  # stable-diffusion or glm-4
    'image_model_config': {
        'stable-diffusion': {
            "model_path": os.environ.get('HOME') +"/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
            "lora_path": 'gen_image/lora_weights/meishi.safetensors',
            "lora_scale": 0.75  #range[0.,1.]
        },
        'glm-4': {
            "api_key": "*****"
        }
    }
}

