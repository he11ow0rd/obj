import config_data as config
from langchain_chroma import Chroma

class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,

        )

    def get_retriever(self):
        """返回向量检索器,方便加入链"""
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    """返回的检索对象"""
    retriever = VectorStoreService(DashScopeEmbeddings(
        model=config.embeddings_model_name,
        dashscope_api_key=config.dashscope_api_key
    )).get_retriever()

    res = retriever.invoke("我体重116斤，帮我推荐尺码")
    print(res)