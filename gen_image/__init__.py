from .zhipu_ai_image import *
from .sd_gen_image import *

image_models = {
    'stable-diffusion': SDGenImage,
    'glm-4': ZhipuAIImage
}
