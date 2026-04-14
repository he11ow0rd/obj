from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document
import vector_stores
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from file_histroy_store import get_history

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):
    def __init__(self):
        self.vector_service = vector_stores.VectorStoreService(embedding=DashScopeEmbeddings(
        model=config.embeddings_model_name,
        dashscope_api_key=config.dashscope_api_key
    ))

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """你是一个专业的客服助手，必须严格遵守以下规则：

        1. **严格基于参考资料**：只能使用下方【参考资料】中的信息来回答问题
        2. **简洁回答**：回答控制在50字以内，使用要点式（bullet points）
        3. **拒绝猜测**：如果参考资料中没有相关信息，直接说"抱歉，没有找到相关信息"
        4. **不重复对话历史**：除非必要，否则不重复用户说过的话

        【参考资料】
        {context}

        【对话历史仅供参考】
        """),
                MessagesPlaceholder("history"),
                ("user", "【用户问题】\n{input}")
            ]
        )

        self.chat_model = ChatDeepSeek(
            api_key=config.api_key,
            model=config.chat_model_name,
            base_url=config.base_url,
        )

        self.chain = self.get_chain()

    def get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()
        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content},文档元数据:{doc.metadata}\n\n"
            return formatted_str


        def format_for_retriever(value):
            return value["input"]

        def format_for_prompt_template(value):
            # input,content,history
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["context"]=value["context"]
            new_value["history"]=value["input"]["history"]
            return new_value
            # print("-------------",value)
            # return value

        chain = (
            {"input":RunnablePassthrough(),"context":RunnableLambda(format_for_retriever) | retriever | format_document}
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",

        )
        return conversation_chain


if __name__ == '__main__':
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    res = RagService().chain.invoke({"input":"体重116，给我尺码推荐"},session_config)
    print(res)
