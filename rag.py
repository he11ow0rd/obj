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

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是威星智能仪表的专业客服助手，负责解答关于燃气表、水表、流量计等产品的问题。

        【产品资料】
        {context}

        【回答规则】
        1. 严格基于上述【产品资料】回答，资料没有的信息不要编造
        2. 如果资料中没有相关信息，回复："抱歉，根据现有产品资料，我无法回答这个问题。建议拨打客服热线4006-157-808咨询。"
        3. 回答要专业、准确，体现工业仪表产品的特点
        4. 使用"- "列出产品功能或技术要点
        5. 如果用户询问产品选型，要根据需求推荐合适的型号
        6. 回答要简洁明了，不要啰嗦

        【用户问题】
        {input}

        请开始回答："""),
            MessagesPlaceholder("history"),  # 这个用于消息历史占位
            ("user", "{input}")  # 用户当前输入
        ])

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

test_queries = [
    "你们公司生产哪些类型的燃气表？",
    "民用智能燃气表和超声波燃气表有什么区别？",
    "智慧水务平台有哪些功能？",
    "你们提供哪些解决方案？",
    "售后服务怎么联系？"
]
if __name__ == '__main__':
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    res = RagService().chain.invoke({"input":test_queries},session_config)
    print(res)
