
import os
from modelscope import snapshot_download
import whisper
from config import load_config

# download shishen LLM model
finetuned = True
if finetuned:
   if not os.path.exists(os.environ.get('HOME') + "/zhanghuiATchina/zhangxiaobai_shishen2_full"):
       model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full', cache_dir=os.environ.get('HOME')+'/models')
       
else:
    if not os.path.exists(os.environ.get('HOME') + "/Shanghai_AI_Laboratory/internlm2-chat-7b"):
        model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir=os.environ.get('HOME')+'/models')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# download RAG model
rag_model_type = load_config('rag', 'rag_model_type')
if rag_model_type == "chroma":
    #os.system('python download_rag_chroma.py')
    print("download rag model for chroma with sqllite")
    # download m3e model
    if not os.path.exists(os.environ.get('HOME') + '/models/m3e-base'):
        command_str = 'huggingface-cli download --resume-download moka-ai/m3e-base --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/m3e-base'
        os.system(command_str)
else :
    #os.system('python download_rag_faiss.py')
    print("download rag model for faiss-gpu")
    print("Download bce embudding base model")
    if not os.path.exists(os.environ.get('HOME') + '/models/bce-embedding-base_v1'):
        command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-embedding-base_v1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/bce-embedding-base_v1'
        os.system(command_str)

    print("Download bce reranker base model")
    if not os.path.exists(os.environ.get('HOME') + '/models/bce-reranker-base_v1'):
        command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-reranker-base_v1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/bce-reranker-base_v1'
        os.system(command_str)

# download SD model
if not os.path.exists(os.environ.get('HOME') +  '/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'):
    command_str = 'huggingface-cli download --resume-download IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
    os.system(command_str)

# download voice recognize model
speech_model_type = load_config('speech', 'speech_model_type')
if speech_model_type == "whisper":
    #os.system('python download_whisper.py')
    print("Download voice model:whisper")
    # download whisper models
    scales = ["tiny", "base", "small", "medium", "large"]
    for scale in scales:
        whisper.load_model(scale)
else :
    #os.system('python download_paraformer.py')
    print("Download voice model:paraformer")
    if not os.path.exists(os.environ.get('HOME') + 'models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'):
        model_dir = snapshot_download(
            "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
            revision="v2.0.4",
            cache_dir=os.environ.get('HOME')+'/models')