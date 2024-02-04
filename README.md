<div align="center">
  <img src="https://github.com/zhanghui-china/intro_myself/blob/main/images/shishen.jpg" width="600"/>
  <br /><br />

![license](https://img.shields.io/github/license/zhanghui-china/intro_myself.svg)  [![issue resolution](https://img.shields.io/github/issues-closed-raw/zhanghui-china/intro_myself)](https://github.com/zhanghui-china/intro_myself/issues)   [![open issues](https://img.shields.io/github/issues-raw/zhanghui-china/intro_myself)](https://github.com/zhanghui-china/intro_myself/issues)

🔍 探索我们的模型：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

</div>
</p>

## 介绍

本APP用于参加 【书生·浦语大模型实战营】的项目实战。用于实现咨询菜谱的对话。

本APP的基本思想，是基于InternLM的对话模型，采用 XiaChuFang Recipe Corpus 提供的1,520,327种中国食谱进行微调，生成食谱模型。 模型存放在modelscope上，应用部署在openxlab上。为此感谢魔搭社区提供免费的模型存放空间，感谢OpenXLab提供应用部署环境及GPU资源。

本APP借用了偶像周星星《食神》的电影截图，不用于商业目的。仅仅由此表达对周星星的尊敬之情。

本APP提供的回答仅供参考，不作为正式菜谱的真实制作步骤。由于大模型的“幻觉”特性，很可能有些食谱会给用户带来心理或生理上的不利影响，希望用户仅认为这是周星星远程教育学院学员张小白跟大家开的一次无厘头的玩笑，切勿上纲上线。



## 更新说明

- [2024.1.30] 基于二代150万菜谱微调的模型和APP发布。（使用InternStudio+A100 1/4X2 40G显存微调，1.25 15:46-1.30 12:25，微调历时4天20小时39分钟）
- [2024.1.28] 基于一代150万菜谱微调的模型和APP发布。（使用WSL+Ubuntu22.04+RTX4090 24G显存微调，1.26 18:40-1.28 13:46历时1天19小时6分钟）。



## 快速上手



### 安装

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



### 训练

​		本项目一代模型 使用 xtuner0.1.9 训练，在 internlm-chat-7b 上进行微调，[模型地址](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen_full/summary)

​		二代模型 使用 xtuner0.1.13 训练，在 internlm2-chat-7b 上进行微调，[模型地址](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

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



### 对话

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

## 演示

Demo 访问地址：https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen

<div align="center">
  <img src="https://github.com/zhanghui-china/intro_myself/blob/main/images/answer001.png" width="600"/>
  <br />
  <img src="https://github.com/zhanghui-china/intro_myself/blob/main/images/answer002.png" width="600"/>
  <br />
</div>



## 模型

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



## 已知问题和未来规划

1.目前基于二代模型微调后的食谱模型，可能还会不定期出现死循环吐字的问题。即便加了repetition_penalty=1.002参数也不能完全阻止这一不可控的行为。而且，当repetition_penalty=1.05时，答案会出现不符合预期输出的格式。说明本模型还是太年轻。需要不断完善调教的方法（说不定也需要基准模型不断地提高相关的能力）<br />

2.目前对提问采用简单的过滤方式，如果用户提问的关键词中没有“怎么做”、"做法"、“食谱”等字样，就要求用户提供相关的指令，否则一直会提示错误。今后可考虑采用多轮对话来获取明确的菜名信息（如先问想吃什么菜——比如川菜或者东北菜，再问什么口味——比如偏甜还是偏辣等等），以便提供精确的菜谱信息。 <br />  

3.今后会考虑对接文生图的应用，在生成菜谱的制作过程之后，同时生成一副该菜的照片，文图并茂展示信息。  <br />



4.看看能不能将提示符工程应用到项目里面去。这次虽然写了prompt，但是感觉相关的交互结果并没有严格按照prompt走。 <br />






## 实践文档

[一代实践](https://zhuanlan.zhihu.com/p/678019309)  <br />
[二代实践](https://zhuanlan.zhihu.com/p/678376843)  <br />

[实践视频](https://www.bilibili.com/video/BV1Ut421W7Qg)  <br />



## 关于作者

项目作者：张小白，字大白，优质回复家。斜杠青年，精力充沛。简称钢筋。 [博客地址](https://www.zhihu.com/people/zhanghui_china)




## 开源许可证

该项目采用 [Apache License 2.0 开源许可证](LICENSE.txt)。
