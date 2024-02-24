
import os
from modelscope import snapshot_download
import whisper

# download  shishen model
model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# download m3e model
command_str = 'huggingface-cli download --resume-download moka-ai/m3e-base --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/m3e-base'
os.system(command_str)

# download whisper models
scales = ["tiny", "base", "small", "medium", "large"]
for scale in scales:
    whisper.load_model(scale)

# download taiyi model
command_str = 'huggingface-cli download --resume-download IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
os.system(command_str)
