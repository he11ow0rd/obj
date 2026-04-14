import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

#添加网页标题
st.title("你好")

uploader_file = st.file_uploader(
    "请上传一个TXT文档",
    type=["txt"],
    accept_multiple_files=False,#仅接收一个文件的上传，不接受多文件
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名:{file_name}") #稍微大一点的字体
    st.write(f"文件格式:{file_type}  大小:{file_size:.2f}KB")

    #get_value
    text = uploader_file.getvalue().decode("utf-8")

    st.write(text)
    with st.spinner("载入知识库中。。。"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)