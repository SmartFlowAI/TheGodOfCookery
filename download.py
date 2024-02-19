
import os
from funasr import AutoModel
from modelscope import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_root_dir = os.environ.get('HOME') + '/models/'

# download  shishen model
# _ = snapshot_download('zhanghuiATchina/zhangxiaobai_shishen2_full',
#                               cache_dir=model_root_dir + 'shishen2')

# download paraformer models
model = AutoModel(
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 
    model_revision="v2.0.4")