import os
from modelscope import snapshot_download
from config import load_config

# download shishen LLM model
base_model_type = load_config('llm', 'base_model_type')
finetuned = load_config('llm', 'finetuned')

if base_model_type == 'internlm2-chat-7b':

    if finetuned:
        if not os.path.exists(os.environ.get('HOME') + "/zhanghuiATchina/zhangxiaobai_shishen2_full"):
            model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full',
                                          cache_dir=os.environ.get('HOME') + '/models')
    else:
        if not os.path.exists(os.environ.get('HOME') + "/Shanghai_AI_Laboratory/internlm2-chat-7b"):
            model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b',
                                          cache_dir=os.environ.get('HOME') + '/models')

elif base_model_type == 'internlm2-chat-1.8b':
    if finetuned:
        if not os.path.exists(os.environ.get('HOME') + "/zhanghuiATchina/zhangxiaobai_shishen2_1_8b"):
            model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_1_8b',
                                          cache_dir=os.environ.get('HOME') + '/models')
    else:
        if not os.path.exists(os.environ.get('HOME') + "/Shanghai_AI_Laboratory/internlm2-chat-1_8b"):
            model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-1_8b',
                                          cache_dir=os.environ.get('HOME') + '/models', revision='v1.1.0')

else:
    if finetuned:
        if not os.path.exists(os.environ.get('HOME') + "/zhanghuiATchina/zhangxiaobai_shishen_full"):
            model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen_full',
                                          cache_dir=os.environ.get('HOME') + '/models')
    else:
        if not os.path.exists(os.environ.get('HOME') + "/Shanghai_AI_Laboratory/internlm-chat-7b"):
            model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b',
                                          cache_dir=os.environ.get('HOME') + '/models')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# download RAG model
print("download rag model")
print("Download bce embedding base model")
if not os.path.exists(os.environ.get('HOME') + '/models/bce-embedding-base_v1'):
    command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-embedding-base_v1 --local-dir-use-symlinks False --local-dir ' + os.environ.get(
        'HOME') + '/models/bce-embedding-base_v1'
    os.system(command_str)

print("Download bce reranker base model")
if not os.path.exists(os.environ.get('HOME') + '/models/bce-reranker-base_v1'):
    command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-reranker-base_v1 --local-dir-use-symlinks False --local-dir ' + os.environ.get(
        'HOME') + '/models/bce-reranker-base_v1'
    os.system(command_str)

# download SD model
if not os.path.exists(os.environ.get('HOME') + '/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'):
    command_str = 'huggingface-cli download --resume-download IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 --local-dir-use-symlinks False --local-dir ' + os.environ.get(
        'HOME') + '/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
    os.system(command_str)

# download voice recognize model
# os.system('python download_paraformer.py')
print("Download voice model:paraformer")
if not os.path.exists(os.environ.get(
        'HOME') + 'models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'):
    model_dir = snapshot_download(
        "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        revision="v2.0.4",
        cache_dir=os.environ.get('HOME') + '/models')
