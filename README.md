<div align="center">
  <img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/cooker.png" width="1092"/>
  <br /><br />

![license](https://img.shields.io/github/license/SmartFlowAI/TheGodOfCookery.svg)  [![issue resolution](https://img.shields.io/github/issues-closed-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)   [![open issues](https://img.shields.io/github/issues-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)

[中文](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README.md)|[English](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README_EN.md)

🔍 探索我们的模型：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

</div>
</p>

## 介绍

本APP用于参加 【书生·浦语大模型实战营】的项目实战。用于实现咨询菜谱的对话。

本APP的基本思想，是基于InternLM的对话模型，采用 XiaChuFang Recipe Corpus 提供的1,520,327种中国食谱进行微调，生成食谱模型。 模型存放在modelscope上，应用部署在openxlab上。为此感谢魔搭社区提供免费的模型存放空间，感谢OpenXLab提供应用部署环境及GPU资源。

本APP提供的回答仅供参考，不作为正式菜谱的真实制作步骤。由于大模型的“幻觉”特性，很可能有些食谱会给用户带来心理或生理上的不利影响，切勿上纲上线。



## 更新说明

- [2024.1.30] 基于二代150万菜谱微调的模型和APP发布。（使用InternStudio+A100 1/4X2 40G显存微调，1.25 15:46-1.30 12:25，微调历时4天20小时39分钟）
- [2024.1.28] 基于一代150万菜谱微调的模型和APP发布。（使用WSL+Ubuntu22.04+RTX4090 24G显存微调，1.26 18:40-1.28 13:46历时1天19小时6分钟）。
- [2024.2.22] 基于团队成员 @房生亮 的文生图模块 以及 @solo fish 的 whisper语音输入模块，整合 text2image分支，发布二阶段第1个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.2.24] 基于团队成员 @Charles 的RAG模块(Chroma)，整合 text2image分支，发布二阶段第2个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.3.3] 基于团队成员 @solo fish 的 paraformer语音输入模块，整合 text2image分支，发布二阶段第3个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.3.4] 增加英文readme。

## 一阶段

### 一阶段安装

1. 准备 Python 虚拟环境：

   ```bash
   conda create -n xtunernew python=3.10 -y
   conda activate xtunernew
   ```

2. 克隆该仓库：

   ```shell
   git clone https://github.com/zhanghui-china/intro_myself.git
   cd ./intro_myself
   ```

3. 安装Pytorch和依赖库：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```



### 一阶段训练

​		一阶段一代模型 使用 xtuner0.1.9 训练，在 internlm-chat-7b 上进行微调，[模型地址](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen_full/summary)

​		一阶段二代模型 使用 xtuner0.1.13 训练，在 internlm2-chat-7b 上进行微调，[模型地址](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

1. 微调方法如下

   ```shell
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
   ```

   - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 LoRA 模型：

   ```shell
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ${YOUR_CONFIG} ${PTH} ${LoRA_PATH}
   ```

   3.将LoRA模型合并入 HuggingFace 模型：

```shell
xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${SAVE_PATH}
```



### 一阶段对话

```shell
xtuner chat ${SAVE_PATH} [optional arguments]
```

参数：

- `--prompt-template`: 一代模型使用 internlm_chat，二代使用  internlm2_chat。
- `--system`: 指定对话的系统字段。
- `--bits {4,8,None}`: 指定 LLM 的比特数。默认为 fp16。
- `--no-streamer`: 是否移除 streamer。
- `--top`: 对于二代模型，建议为0.8。
- `--temperature`: 对于二代模型，建议为0.8。
- `--repetition-penalty`: 对于二代模型，建议为1.002，对于一代模型可不填。
- 更多信息，请执行 `xtuner chat -h` 查看。

### 一阶段演示

Demo 访问地址：https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen

<div align="center">
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer001.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/TheGodOfCookery/blob/main/images/answer002.png" width="600"/>
  <br />
</div>



### 一阶段模型

[openxlab一代模型](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen_full)    <br />
[openxlab二代模型](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen2_full)    <br />

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



### 一阶段已知问题

一阶段在书生.浦语实战营期间由张小白独立完成。

1.目前基于二代模型微调后的食谱模型，可能还会不定期出现死循环吐字的问题。即便加了repetition_penalty=1.002参数也不能完全阻止这一不可控的行为。而且，当repetition_penalty=1.05时，答案会出现不符合预期输出的格式。说明本模型还是太年轻。需要不断完善调教的方法（说不定也需要基准模型不断地提高相关的能力）<br />

2.目前对提问采用简单的过滤方式，如果用户提问的关键词中没有“怎么做”、"做法"、“食谱”等字样，就要求用户提供相关的指令，否则一直会提示错误。今后可考虑采用多轮对话来获取明确的菜名信息（如先问想吃什么菜——比如川菜或者东北菜，再问什么口味——比如偏甜还是偏辣等等），以便提供精确的菜谱信息。 <br />  

3.今后会考虑对接文生图的应用，在生成菜谱的制作过程之后，同时生成一副该菜的照片，文图并茂展示信息。  <br />

4.看看能不能将提示符工程应用到项目里面去。这次虽然写了prompt，但是感觉相关的交互结果并没有严格按照prompt走。 <br />


### 一阶段实践文档

[一代实践](https://zhuanlan.zhihu.com/p/678019309)  <br />
[二代实践](https://zhuanlan.zhihu.com/p/678376843)  <br />

[实践视频](https://www.bilibili.com/video/BV1Ut421W7Qg)  <br />


## 二阶段规划

二阶段引入了各位大佬（参见 项目参与人员），计划完成和完善以下功能：

1.实现RAG+LLM结合

2.实现语音输入、语音输出

3.嵌入食谱图片（图片能通过文生图模型生成）

4.优化prompt和对话

### 二阶段安装

1. 准备 Python 虚拟环境：

   ```bash
   conda create -n cook python=3.10 -y
   conda activate cook
   ```

2. 克隆该仓库：

   ```shell
   git clone https://github.com/zhanghui-china/TheGodOfCookery.git
   cd ./TheGodOfCookery
   ```

3. 安装Pytorch和依赖库：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
这里cuda的版本根据用户自己的cuda版本确定。一般为 11.8或12.1

### 二阶段数据集

未完成。

### 二阶段训练

未进行。

### 二阶段对话

   ```bash
先在 config/config.py文件中设置好 speech_model_type(缺省为paraformer)和 rag_model_type(缺省为chroma）

执行 python start.py

浏览器打开 http://127.0.0.1:7860 即可。

   ```

### 二阶段模型

与一阶段相同

### 二阶段演示

Demo 访问地址：https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024

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

### 二阶段已知问题

1.语音识别 部分：有时候会识别出同音字，期待后续AI能够自动识别同音字。

2.格式化输出 部分：食材图片似乎因文件名数据集问题无法显示。解析食材会出现部分偏差。

### 二阶段实践文档

[项目介绍视频](https://www.bilibili.com/video/BV1kr421W7iA)  <br />

## 二阶段项目代码结构

   ```shell
项目目录
|---config   # 配置文件目录（主要贡献者 @房宇亮）
|     |---__init__.py                                      #初始化脚本
|     |---config.py                                        #配置脚本
|
|---gen_image    # 文生图目录（主要贡献者 @房宇亮）
|     |---__init__.py                                      #初始化脚本
|     |---sd_gen_image.py                                  #使用Stabble Disffion的文生图模块
|     |---zhipu_ai_image.py                                #使用智谱AI的文生图模块
|
|---images  # 的图片目录，生成的图片临时也放在这里，今后会考虑迁移到其他目录
|     |---robot.png                                        #对话机器人图标 
|     |---user.png                                         #对话用户图标 
|     |---shishen.png                                      #项目图标 
|
|---rag   # 二代RAG代码目录（主要贡献者 @乐正萌）
|     |---source_data                                      #原始数据集目录
|     |     |- text.txt                                    #原始菜谱数据集
|     |---data                                             #处理后的数据集目录
|     |     |- caipu.txt                                   #处理后的菜谱数据集
|     |---chroma_db                                        #chroma数据库目录
|     |     |- chroma.sqlite3                              #chroma库文件
|     |---faiss_index                                      #FAISS数据库目录
|     |     |- index.faiss   
|     |     |- index.pkl
|     |---retrieve                                         #retrieve目录
|     |     |- bm25retriever.pkl
|     |---CookMasterLLM.py
|     |---convert_txt.py
|     |---create_db.py
|     |---interface.py
|     |---rag_test.py
|     |---run_local.py
|
|---rag_chroma   # 二代RAG代码目录（主要贡献者 @Charles）
|     |---database                                         #chroma数据库目录
|     |     |- chroma.sqlite3                              #chroma库文件
|     |---LLM.py
|     |---create_db.py
|     |---interface.py
|
|---src    # 食材图标目录
|     |---*.png                                            #各类食材图标
|
|---tools    # 工具文件目录
|
|---whisper_app    # 语音识别目录（主要贡献者 @solo fish）
|     |---__init__.py                                      #初始化脚本
|     |---whisper.py                                       #语音识别处理脚本
|
|---speech    # paraformer语音识别目录（主要贡献者 @solo fish）
|     |---__init__.py                                      #初始化脚本
|     |---utils.py                                         #语音识别处理脚本
|
|---requirements.txt                                       #系统依赖包（请使用pip install -r requirements.txt安装）
|---convert_t2s.py                                         #繁体字转简体字工具（主要贡献者 @彬彬）
|---parse_cur_response.py                                  #输出格式化处理工具 （主要贡献者 @彬彬）
|---README.md                                              #本文档
|---cli_demo.py                                            #模型测试脚本
|---download.py                                            #模型下载脚本
|---download_whisper.py                                    #下载whisper模型
|---download_paraformer.py                                 #下载paraformer模型
|---download_rag2_model.py                                 #仅二代RAG所需模型的下载脚本
|---start.py                                               #Web Demo启动脚本
|---start2.py                                              #Web Demo启动脚本（支持RAG2）
|---start_rag_chroma.py                                    #Web Demo启动脚本（支持RAG1）
|---start_rag2.py                                          #Web Demo启动脚本（仅支持RAG2）
|---app.py                                                 #Web Demo主脚本（RAG1+whisper+image+markdown）
|---app_paraformer.py                                      #Web Demo主脚本（RAG1+paraformer+image+markdown）
|---app2.py                                                #Web Demo主脚本（RAG2+whisper+image+markdown）
|---app-enhanced-rag.py                                    #仅支持RAG2的主脚本
|---app-rag-with-chroma.py                                 #支持RAG1的主脚本
   ```

## 项目参与人员（排名不分先后）

1.张小白，项目策划、测试和打杂。现为某IT公司数据工程师，华为云HCDE（原华为云MVP），2020年华为云社区十佳博主，2022年昇腾社区优秀开发者，2022年华为云社区年度优秀版主，MindSpore布道师，DataWhale优秀学习者， [知乎](https://www.zhihu.com/people/zhanghui_china)

2.sole fish：语音输入  [github](https://github.com/YanxingLiu)  

3.Charles：一代RAG（基于Chroma）现为同济大学本科毕业生，考研中。 [github](https://github.com/SchweitzerGAO)

4.乐正萌：二代RAG（基于faiss&Chroma）[github](https://github.com/YueZhengMeng)

5.彬彬：格式化输出 [github](https://github.com/Everfighting) [知乎](https://www.zhihu.com/people/everfighting)

6.房宇亮：文生图、配置工具 [github](https://github.com/leonfrank)   

7.刘光磊：图标设计，前端优化 [github](https://github.com/Mrguanglei)

8.喵喵咪：数据集准备，后续本地小模型部署测试，北京航空航天大学硕士，现为上海某国企工程师。 [github](https://github.com/miyc1996)

9.王巍龙：数据集，微调

10.轩辕：文档准备，数据集，微调 现为南京大学在读硕士。[github](https://github.com/zzd2001)

11.浦语小助手：提供书生浦语大模型、工具链、训练环境、人才培养等全方面支持 [github](https://github.com/InternLM/InternLM)


## 开源许可证

该项目采用 [Apache License 2.0 开源许可证](LICENSE.txt)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SmartFlowAI/TheGodOfCookery&type=Date)](https://star-history.com/#SmartFlowAI/TheGodOfCookery&Date)
