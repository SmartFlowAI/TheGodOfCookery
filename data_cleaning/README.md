# 原始数据集去重测试实验记录
## 本文件夹说明
convert_data_to_juicer.py : 将原始数据集转换为data juicer输入格式的脚本  
config.yaml data : data juicer配置文件  
convert_juicer_to_data.py : 将data juicer输出转换为xtuner格式数据集的脚本  
log : data juicer去重记录日志  
trace : data juicer重复数据样本  
## 实验记录
### 下载原始数据集
https://counterfactual-recipe-generation.github.io/dataset_en.html  
选择Download Full Recipe Dataset  
下载后解压到/root/cook-data/目录下  
### 安装data juicer
我们的任务只需要安装data juicer最小依赖项和simhash算子即可，安装方法如下：  
建议先新建一个python 3.10的conda虚拟环境，然后再安装data juicer
```shell
# conda create -n data-juicer python=3.10.13 -y
# conda activate data-juicer
git clone https://github.com/modelscope/data-juicer.git
cd <path_to_data_juicer>
pip install -v -e .
pip install simhash-pybind
```
### 原始数据集转换
该脚本会在转换的同时，将数据安装菜品排序，相同菜品按照菜名长度排序，方便后续查看去重效果
```shell
cd <path_to_script>
python convert_data_to_juicer.py
```
### data juicer去重
```shell
cd <path_to_data_juicer>
python tools/process_data.py --config <path_to_config.yaml>
```
### data juicer输出转换
```shell
cd <path_to_script>
python convert_juicer_to_data.py
```
## 第一次实验
使用之前Charles佬编写的config.yaml  
清洗后数据剩余869264条  
高字符相似度数据去重效果较好，但是低字符相似度数据去重效果较差  
实验日志与去重样本见exp1下log和trace文件夹  

## 第二次实验
修改实验一的config.yaml  
增加了文档级MD5 hash去重  
扩大window_size为data juicer默认值  
其他参数中也全部使用data juicer默认值  
清洗后数据剩余726717条  
仍有低字符相似度数据未去重问题  
实验日志与去重样本见exp2下log和trace文件夹  

## 第三次实验
在实验二的基础上，修改tokenization分割方法为space  
尝试进行sub sentence level的去重  
清洗后数据剩余725条  
显然有些用力过猛  
实验日志与去重样本见exp3下log和trace文件夹

## 第四次实验
在实验二的基础上，修改tokenization分割方法为punctuation  
尝试进行sub sentence level的去重  
清洗后数据剩余3187条  
还是有些用力过猛  
实验日志与去重样本见exp4下log和trace文件夹

## 第五次实验
在实验二的基础上，修改tokenization分割方法为sentencepiece  
sentencepiece模型为 internlm2-chat-1_8b/tokenizer.model  
清洗后数据剩余559546条   
实验日志与去重样本见exp5下log和trace文件夹  

## 后续改进方向
1. 在实验五的基础上，继续增加window_size
