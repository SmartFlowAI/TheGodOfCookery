# 首先导入所需第三方库
import os
import pickle

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from tqdm import tqdm


# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list


# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in file_lst:
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs


# 目标文件夹
tar_dir = [
    "./data"
]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 对文本进行分块
# text_splitter = CharacterTextSplitter(separator="\n\n")
# split_docs = text_splitter.split_documents(docs)

split_docs = []
caipu_txt = open('data/caipu.txt', 'r', encoding='utf-8')
lines = caipu_txt.readlines()
for line in lines:
    if len(line) > 2:
        # 加入原始菜谱
        # split_docs.append(Document(page_content=line))
        # 假设问题为“菜谱名+怎么做”
        # 加入假设问题，原始菜谱存放入metadata
        caipu_name = line.split('  ')[0]
        split_docs.append(Document(page_content=caipu_name+"怎么做", metadata={"caipu": line}))

# 构建向量数据库
embedding_model_name = 'F:/OneDrive/Pythoncode/BCE_model/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': True}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

bm25retriever = BM25Retriever.from_documents(documents=split_docs)
bm25retriever.k = 5

faiss_index = FAISS.from_documents(documents=split_docs, embedding=embeddings,
                                   distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory='./chroma_db')


# BM25Retriever序列化到磁盘
if not os.path.exists("./retriever"):
    os.mkdir("./retriever")
pickle.dump(bm25retriever, open('./retriever/bm25retriever.pkl', 'wb'))

# 保存索引到磁盘
faiss_index.save_local('./faiss_index')

# 将加载的向量数据库持久化到磁盘上
vectordb.persist()

