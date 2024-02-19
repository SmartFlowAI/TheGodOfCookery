
import os
from modelscope import snapshot_download
import whisper
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_root_dir = os.environ.get('HOME') + '/models/'

# download  shishen model
_ = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full',
                              cache_dir=model_root_dir + 'shishen2')

# download whisper models
scales = ["tiny", "base", "small", "medium", "large"]
for scale in scales:
    whisper.load_model(scale,download_root=model_root_dir+'whisper')