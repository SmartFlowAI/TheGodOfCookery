# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


# 获取文件路径函数
# def get_files(dir_path):
#     # args：dir_path，目标文件夹路径
#     file_list = []
#     for filepath, dirnames, filenames in os.walk(dir_path):
#         # os.walk 函数将递归遍历指定文件夹
#         for filename in filenames:
#             # 通过后缀名判断文件类型是否满足要求
#             if filename.endswith(".md"):
#                 # 如果满足要求，将其绝对路径加入到结果列表
#                 file_list.append(os.path.join(filepath, filename))
#             elif filename.endswith(".txt"):
#                 file_list.append(os.path.join(filepath, filename))
#     return file_list


# 加载文件函数
def get_text(file_path):
    # args：dir_path，目标文件夹路径
    # # 首先调用上文定义的函数得到目标文件路径列表
    # file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_path):
        loader = UnstructuredFileLoader(one_file)
        docs.extend(loader.load())
    return docs


# 目标文件夹
# tar_dir = [
#     "/root/data/InternLM",
#     "/root/data/InternLM-XComposer",
#     "/root/data/lagent",
#     "/root/data/lmdeploy",
#     "/root/data/opencompass",
#     "/root/data/xtuner"
# ]
files = [
    './data/text.txt'
]

# 加载目标文件
docs = []
docs.extend(get_text(files))

# 对文本进行分块
text_splitter = TokenTextSplitter(
    chunk_size=700, chunk_overlap=200, allowed_special={'<|endoftext|>'})  # PDF文件中的特殊词元<|endoftext|>
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="./m3e-base")

# 构建向量数据库
# 定义持久化路径
persist_directory = './database'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
