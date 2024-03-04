import os
from modelscope import snapshot_download

# download paraformer model
if not os.path.exists(os.environ.get('HOME') + 'models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'):
    model_dir = snapshot_download(
        "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
        revision="v2.0.4",
        cache_dir=os.environ.get('HOME')+'/models')