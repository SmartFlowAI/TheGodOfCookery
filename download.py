
import os
from modelscope import snapshot_download
import whisper
# download  shishen model
# model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # download m3e model
os.system('huggingface-cli download --resume-download moka-ai/m3e-base ')

# download whisper models
scales = ["tiny", "base", "small", "medium", "large"]
for scale in scales:
    whisper.load_model(scale)