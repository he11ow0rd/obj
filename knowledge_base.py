"""
知识库
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str:str):
    #检查传入的md5字符串是否处理过
    if not os.path.exists(config.md5_path):
        #文件不存在
        open(config.md5_path,"w",encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path,"r",encoding="utf-8").readlines():
            line = line.strip()
            if line == md5_str:
                return True #已处理过

        return False



def save_md5(md5_str:str):
    #将传入的md5字符串，记录到文件内保存
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str:str, encoding="utf-8"):
    #将传入的字符串转换为md5字符串
    str_bytes = input_str.encode(encoding)
    md5_hex = hashlib.md5(str_bytes).hexdigest()
    return md5_hex


class KnowledgeBaseService:
    def __init__(self):
        #如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name, #数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key='sk-1be70003a1a64abe9db254139f0f0abe'),
            persist_directory=config.persist_directory, #数据库本地存储文件夹
        )  #向量储存的实力Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self,data:str,filename):
        #将传入的字符串，进行向量化。存入向量库中
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex): #如果有数据
            return "[跳过]内容已经存在知识库"

        if len(data) > config.max_split_char_number:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else: #如果没有数据
            knowledge_chunks = [data]
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小胡"
        }
        self.chroma.add_texts( #内容加载到向量库中
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)

        return "[成功]数据已加载向量库"



if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦","testfile")
    print(r)