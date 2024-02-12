
import os
from modelscope import snapshot_download
model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full', cache_dir='/home/xlab-app-center/models')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download moka-ai/m3e-base '
          '--local-dir /home/xlab-app-center/models/m3e-base')