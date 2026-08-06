import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
BASE = "https://open.bigmodel.cn/api/paas/v4"

embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=API_KEY)
llm = ChatOpenAI(model="glm-4-flash", api_key=API_KEY, base_url=BASE, temperature=0.3)

PROMPT = ChatPromptTemplate.from_template(
    """你是"AI 面试题库"问答助手。仅根据【上下文】回答用户问题；
    若上下文没有相关信息，请明确回答"抱歉，知识库中未找到相关内容"，不要编造。
    回答末尾用 [1][2] 标注引用来源编号。

【上下文】
{context}

【问题】{question}
【回答】"""
)

def format_docs(docs):
    out = []
    for i, d in enumerate(docs, 1):
        out.append(f"[{i}] 来源：{d.metadata.get('source','未知')}\n{d.page_content}")
    return "\n\n".join(out)

def build_chain():
    if os.path.exists("./chroma_db"):
        vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"})
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        ).split_documents(loader.load())
        vs = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )

if __name__ == "__main__":
    chain = build_chain()
    while True:
        q = input("\n你：")
        if q in ("exit", "quit"): break
        print("助手：", chain.invoke(q))