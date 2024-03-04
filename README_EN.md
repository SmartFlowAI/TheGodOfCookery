<div align="center">
  <img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/cooker.png" width="1092"/>
  <br /><br />

![license](https://img.shields.io/github/license/SmartFlowAI/TheGodOfCookery.svg)  [![issue resolution](https://img.shields.io/github/issues-closed-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)   [![open issues](https://img.shields.io/github/issues-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)

[中文](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README.md)|[English](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README_EN.md)

🔍 Explore our models：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

</div>
</p>

## Introduction

The basic idea of this APP is to generate a recipe dialogue model based on InternLM chat model, which is fine-tuned on XiaChuFang Recipe Corpus with 1,520,327 Chinese recipes. The model is stored on modelscope and is deployed on openxlab. We would like to thank the modelscope  community for providing free model storage space, and OpenXLab for providing the application deployment environment and GPU resources.

The answers provided by this APP are for reference only, and should  not be used as the actual steps of official recipes. Due to the "hallucinatory" nature of the large model, it is likely that some recipes may have adverse psychological or physical effects on the user, so do not take them to the extreme.

### Recent Updates

- [2024.1.30] First-generation model and APP fine-tuned on 1.5 million recipes released . (Fine-tuned using InternStudio + A100 1/4X2 40G video memory, 1.25 15:46-1.30 12:25, fine-tuning lasted 4 days, 20 hours, 39 minutes)
- [2024.1.28] Second-generation model and app fine-tuned on 1.5 million recipes released . (Fine-tuned using WSL+Ubuntu22.04+RTX4090 24G video memory, 1.26 18:40-1.28 13:46 in 1 day 19 hours 6 minutes).
- [2024.2.22] Based on team member @fangshengliang's text2image module and @solo fish's whisper voice input module, we integrate the text2image branch, and release the first openxlab A100-based application of the second stage . [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.2.24] Based on team member @Charles' RAG module (Chroma), we integrate the text2image branch, and release the second openxlab A100-based application of the second stage [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.3.3] Based on team member @solo fish's paraformer voice input module, we integrate the text2image branch, and release the third openxlab A100-based application of the second stage [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.3.5] Add English README

## Stage 1

### Installation

1. Prepare a Python Virtual Environment：

   ```bash
   conda create -n xtunernew python=3.10 -y
   conda activate xtunernew
   ```

2. Clone the Repo：

   ```shell
   git clone https://github.com/zhanghui-china/intro_myself.git
   cd ./intro_myself
   ```

3. Install Pytorch and Dependency Libraries：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```



### Training

​		The first generation model of stage 1 was trained with xtuner0.1.9 and fine-tuned based on internlm-chat-7b，[model link](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen_full/summary)

​		The second generation of model of stage 1 was trained with xtuner0.1.13 and fine-tuned on internlm2-chat-7b，[mdoel link](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

1. The fine-tuning method is as follows

   ```shell
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
   ```

   - `--deepspeed` for useing  [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 optimize the training process. XTuner has several built-in strategies, including ZeRO-1, ZeRO-2, ZeRO-3 and so on. If you want to disable this feature, please remove this parameter directly.

2. Converts the saved `.pth` model (which will be a folder if DeepSpeed is used) to a LoRA model:

   ```shell
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ${YOUR_CONFIG} ${PTH} ${LoRA_PATH}
   ```

3. Merge the LoRA model into the HuggingFace model:

```shell
xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${SAVE_PATH}
```



### Chat

```shell
xtuner chat ${SAVE_PATH} [optional arguments]
```

configs：

- `--prompt-template`: The first generation model uses internlm_chat and the second generation uses internlm2_chat.
- `--system`: specifies the system fields for the dialog。
- `--bits {4,8,None}`: specifies the number of bits in the LLM. The default is fp16。
- `--no-streamer`: whether to remove the streamer。
- `--top`: for the second generation model, 0.8 is recommended。
- `--temperature`: for the second generation model, 0.8 is recommended。
- `--repetition-penalty`: for second generation model, 1.002 is recommended, for first-generation model, it can be ignored.
- For more information, run `xtuner chat -h`

### Demo

Demo link：https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen

<div align="center">
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer001.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer002.png" width="600"/>
  <br />
</div>



### Model

[openxlab first generation model ](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen_full)    <br />
[openxlab second generation model ](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen2_full)    <br />

```shell
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tools.transformers.interface import GenerationConfig, generate_interactive

model_name_or_path = "zhanghuiATchina/zhangxiaobai_shishen_full" #对于二代模型改为 zhangxiaobai_shishen2_full

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

messages = []
generation_config = GenerationConfig(max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "酸菜鱼怎么做", history=history)
print(response)
```




### Practice document

[The practice of the first generation model](https://zhuanlan.zhihu.com/p/678019309)  <br />
[The practice of the second generation model](https://zhuanlan.zhihu.com/p/678376843)  <br />

[video](https://www.bilibili.com/video/BV1Ut421W7Qg)  <br />


## Stage 2 planning

Stage 2 we plans to complete and refine the following features:

1.Realization of RAG+LLM combination

2.Realize voice input, voice output

3.Embedded recipe images (images can be generated by the text2image models)

4.Optimize prompt and dialog

### Installation

1. Prepare a Python Virtual Environment：

   ```bash
   conda create -n cook python=3.10 -y
   conda activate cook
   ```

2. Clone the Repo：

   ```shell
   git clone https://github.com/zhanghui-china/TheGodOfCookery.git
   cd ./TheGodOfCookery
   ```

3. Install Pytorch and Dependency Libraries：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
The cuda version is determined based on the user's own cuda version. It is usually 11.8 or 12.1

### Dataset

Waiting for adding

### Training

Waiting for adding

### Chat

Waiting for adding

### Models

Waiting for adding

### Demo

Demo link：https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3

<div align="center">
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer101.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer102.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer103.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer104.png" width="600"/>
  <br />
</div>


## Project Code Structure

   ```shell
项目目录
|---config   # Configuration file directory (main contributor @FangYuLiang)
|     |---__init__.py                                      #Initialization Script
|     |---config.py                                        #Configuration script
|
|---gen_image    # Text-to-Picture directory (Main contributor @Fang Yuliang)
|     |---__init__.py                                      #Initialization Script
|     |---sd_gen_image.py                                  #Text-to-Picture Module with Stabble Disffion
|     |---zhipu_ai_image.py                                #Text-to-Picture Module with zhipu ai
|
|---images  # Image directory, the generated images are temporarily placed here, and will be considered for migration to other directories in the future.
|     |---robot.png                                        #Dialog Bot Icon 
|     |---user.png                                         #Dialog user icon
|     |---shishen.png                                      #Project Icons 
|
|---rag   # Second Generation RAG Code Catalog (Major Contributor @Le Zhengmeng)
|     |---source_data                                      #Directory of original datasets
|     |     |- text.txt                                    #Original recipe dataset
|     |---data                                             #Directory of processed data sets
|     |     |- caipu.txt                                   #Processed recipe dataset
|     |---chroma_db                                        #chroma database directory
|     |     |- chroma.sqlite3                              #chroma library files
|     |---faiss_index                                      #FAISS database directory
|     |     |- index.faiss   
|     |     |- index.pkl
|     |---retrieve                                         #retrieve directory
|     |     |- bm25retriever.pkl
|     |---CookMasterLLM.py
|     |---convert_txt.py
|     |---create_db.py
|     |---interface.py
|     |---rag_test.py
|     |---run_local.py
|
|---rag_chroma   # Second Generation RAG Code Catalog (major contributor @Charles)
|     |---database                                         #chroma database directory
|     |     |- chroma.sqlite3                              #chroma library files
|     |---LLM.py
|     |---create_db.py
|     |---interface.py
|
|---src    # Ingredients Icons Catalog
|     |---*.png                                            #Various ingredient icons
|
|---tools    # Tool files directory
|
|---whisper_app    # Speech recognition directory (main contributor @solo fish)
|     |---__init__.py                                      #Initialization Script
|     |---whisper.py                                       #Speech Recognition Processing Script
|
|---speech    # paraformer speech recognition catalog (main contributor @solo fish)
|     |---__init__.py                                      #Initialization Script
|     |---utils.py                                         #Speech Recognition Processing Script
|
|---requirements.txt                                       #System dependency packages (please use pip install -r requirements.txt to install)
|---convert_t2s.py                                         #Traditional Chinese to Simplified Chinese Conversion Tool (Main Contributor @彬彬)
|---parse_cur_response.py                                  #Output Formatting Processing Tool (main contributor @BinBin)
|---README.md
|---README_EN.md
|---cli_demo.py                                            #Model Test Scripts
|---download.py                                            #Model Download Script
|---download_whisper.py                                    #Download the whisper model
|---download_paraformer.py                                 #Download paraformer model
|---download_rag2_model.py                                 #Download scripts for models required for second-generation RAGs only
|---start.py                                               #Web Demo Launch Script
|---start2.py                                              #Web Demo startup script (RAG2 support)
|---start_rag_chroma.py                                    #Web Demo startup script (RAG1 support)
|---start_rag2.py                                          #Web Demo startup script (RAG2 support only)
|---app.py                                                 #Web Demo main Script (RAG1+whisper+image+markdown)
|---app_paraformer.py                                      #Web Demo main script (RAG1+paraformer+image+markdown)
|---app2.py                                                #Web Demo main script (RAG2+whisper+image+markdown)
|---app-enhanced-rag.py                                    #main script only the RAG2 supported
|---app-rag-with-chroma.py                                 #main script only the RAG1 supported
   ```

### Project participants (in no particular order)

1.Zhang Xiaobai，project planning, testing and miscellaneous work. @Nanjing University [知乎](https://www.zhihu.com/people/zhanghui_china)

2.sole fish：speech input  [github](https://github.com/YanxingLiu)  

3.Charles：first generation RAG（Based on Chroma）@Tongji University [github](https://github.com/SchweitzerGAO)

4.Le Zhengmeng：second generation RAG (Based on faiss&Chroma) [github](https://github.com/YueZhengMeng)

5.Bin Bin：format the output [github](https://github.com/Everfighting) [知乎](https://www.zhihu.com/people/everfighting)

6.Fang Yuliang：text-generated image [github](https://github.com/leonfrank)   

7.Liu Guanglei：iconic design, front-end optimization [github](https://github.com/Mrguanglei)

8.Miao Miaomi：datasets @Beijing University of Aeronautics and Astronautics [github](https://github.com/miyc1996)

9.Wang Weilong：datasets, fine-tuning

10.Xuan Yuan：document preparation, datasets, fine-tuning @Nanjing University [github](https://github.com/zzd2001)

11.Puguese Assistant: provide full support for InternIM model, tool chain, training environment, talent training and so on. [github](https://github.com/InternLM/InternLM)


## License

The project follows [Apache License 2.0](LICENSE.txt)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SmartFlowAI/TheGodOfCookery&type=Date)](https://star-history.com/#SmartFlowAI/TheGodOfCookery&Date)
