from langchain_community.embeddings import DashScopeEmbeddings

md5_path = "./md5.text"

#Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",",",".","？","，","。","?"," ",""]

max_split_char_number = 1000

similarity_threshold = 5

embeddings_model_name="text-embedding-v4"
dashscope_api_key = "sk-1be70003a1a64abe9db254139f0f0abe"

chat_model_name = "deepseek-chat"
api_key = "sk-3155cdf820e64531b9aac730badcd23b"
base_url = "https://api.deepseek.com"

session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }

