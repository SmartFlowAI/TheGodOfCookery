import os
from modelscope import snapshot_download
import whisper

# download shishen model
# model_dir = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full')

# download m3e model
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
command_str = 'huggingface-cli download --resume-download moka-ai/m3e-base --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/m3e-base'
#command_str = 'huggingface-cli download --resume-download moka-ai/m3e-base --local-dir-use-symlinks False --local-dir /home/xlab-app-center/models/m3e-base'
os.system(command_str)

# download whisper models
scales = ["tiny", "base", "small", "medium", "large"]
for scale in scales:
    whisper.load_model(scale)
