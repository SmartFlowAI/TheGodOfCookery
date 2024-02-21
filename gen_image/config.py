import os
from collections import defaultdict
image_model_type = 'stable-diffusion' # stable-diffusion or glm-4

image_model_config = defaultdict(dict)


image_model_config['stable-diffusion'] = {
    "model_path": os.environ.get('HOME') +"/models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
    "lora_path": 'gen_image/lora_weights/meishi.safetensors',
    "lora_scale": 0.75  #range[0.,1.]

}

image_model_config['glm-4'] = {
    "api_key": "*****"
}

