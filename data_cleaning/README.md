## 本文件夹说明
convert_data_to_juicer.py : 将原始数据集转换为data juicer输入格式的脚本  
dedup.yaml data : data juicer配置文件  
convert_juicer_to_data.py : 将data juicer输出转换为xtuner格式数据集的脚本
## 实验记录
### 下载原始数据集
https://counterfactual-recipe-generation.github.io/dataset_en.html  
选择Download Full Recipe Dataset  
下载后解压到/root/cook-data/目录下  
### 安装data juicer
我们的任务只需要安装data juicer最小依赖项和部分算子即可，安装方法如下：  
建议先新建一个python 3.10的conda虚拟环境，然后再安装data juicer
```shell
# conda create -n data-juicer python=3.10.13 -y
# conda activate data-juicer
git clone https://github.com/modelscope/data-juicer.git
cd <path_to_data_juicer>
pip install -v -e .
pip install simhash-pybind fasttext-wheel kenlm sentencepiece ftfy transformers==4.37
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
