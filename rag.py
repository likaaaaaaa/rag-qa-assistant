import os
import requests
import jieba
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
BASE = "https://open.bigmodel.cn/api/paas/v4"


class BatchedZhipuEmbeddings(ZhipuAIEmbeddings):
    """智谱 embedding API 对单次请求的总文本量有限制（长文本约 30-60 条就超限），
    按 20 条分批，保证任何知识库规模都能安全入库"""

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 20):
            out.extend(super().embed_documents(texts[i:i + 20]))
        return out


embeddings = BatchedZhipuEmbeddings(model="embedding-2", api_key=API_KEY)
llm = ChatOpenAI(model="glm-4-flash", api_key=API_KEY, base_url=BASE, temperature=0.3, timeout=60)

PROMPT = ChatPromptTemplate.from_template(
    """你是"AI 面试题库"问答助手，服务对象是备战 AI 应用岗面试的求职者。
回答规范：
1. 仅根据【上下文】回答，不凭记忆发挥、不得引入上下文之外的概念；
2. 拒答规则：若上下文没有相关信息，或检索内容与问题无关，必须明确回答"抱歉，知识库中未找到相关内容"，绝不编造，不得根据上下文中的比喻、例子推断出答案；
3. 回答结构：先一句话直接给结论，再分要点展开，语言通俗、层次清晰；
4. 引用规则：回答末尾必须用 [1][2] 标注引用来源编号并注明来源文件名（格式如 [1](kb_05-prompt工程.txt)），只要回答内容来自上下文就必须附引用清单，不得省略；
5. 讲解原则：涉及原理/对比的问题，用大白话解释"为什么"，帮用户真正理解而非背诵。

【上下文】
{context}

【问题】{question}
【回答】"""
)


def format_docs(docs):
    out = []
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get('source', '未知'))
        out.append(f"[{i}] 来源：{src}\n{d.page_content}")
    return "\n\n".join(out)


# ============ 阶段 1 新增：三模式可插拔检索 ============
# mode = "vector"        阶段 0 基线：纯向量检索 top-3（默认，与线上完全一致）
# mode = "hybrid"        1A：向量 k=10 + BM25 k=10 → RRF 融合 → top-3
# mode = "hybrid_rerank" 1B：混合检索候选 → 智谱 Rerank 精排 → top-3
# -------------------------------------------------------

_vs = None      # 向量库单例
_docs = None    # 文档列表单例
_ensemble = None  # 混合检索器单例


def _tokenize(text):
    """jieba 中文分词：BM25 按词匹配，中文必须分词（昨天实测：不分词会把"过拟合"挤出 top-3）"""
    return list(jieba.cut(text))


def _get_vs():
    """懒加载向量库：存在 chroma_db 就加载，否则重建"""
    global _vs
    if _vs is None:
        if os.path.exists("./chroma_db"):
            _vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        else:
            _vs = Chroma.from_documents(_get_docs(), embeddings, persist_directory="./chroma_db")
    return _vs


def _get_docs():
    """懒加载文档：与建库时完全相同的切分规则（保证 BM25 与向量库对齐）"""
    global _docs
    if _docs is None:
        loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"})
        _docs = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        ).split_documents(loader.load())
    return _docs


def _get_ensemble():
    """混合检索器：向量(语义) + BM25(关键词)，RRF 倒数排名融合，返回全部候选"""
    global _ensemble
    if _ensemble is None:
        vector_r = _get_vs().as_retriever(search_kwargs={"k": 10})
        bm25_r = BM25Retriever.from_documents(_get_docs(), k=10, preprocess_func=_tokenize)
        # id_key=None：用 page_content 作为融合去重键（Chroma/BM25 的 metadata 无 doc_id）
        _ensemble = EnsembleRetriever(
            retrievers=[vector_r, bm25_r], weights=[0.5, 0.5], id_key=None
        )
    return _ensemble


def zhipu_rerank(query, docs, top_n=3):
    """调用智谱 Rerank API 对候选文档精排（与现有 key 同平台，改动最小）"""
    resp = requests.post(
        f"{BASE}/rerank",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "rerank",
            "query": query,
            "documents": [d.page_content for d in docs],
            "top_n": top_n,
            "return_documents": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # results 已按 relevance_score 降序，index 指向原 documents 位置
    return [docs[r["index"]] for r in data["results"]]


def retrieve_docs(query, mode="vector"):
    """统一的文档检索入口：返回按相关度排序的 list[Document]"""
    if mode == "vector":
        return _get_vs().as_retriever(search_kwargs={"k": 3}).invoke(query)
    candidates = _get_ensemble().invoke(query)  # 混合检索全部候选
    if mode == "hybrid":
        return candidates[:3]                    # RRF 融合后取 top-3
    if mode == "hybrid_rerank":
        return zhipu_rerank(query, candidates, top_n=3)
    raise ValueError(f"未知检索模式: {mode}")


def build_chain(mode="vector"):
    """构建问答链。默认 vector：与原版行为完全一致（线上部署零影响）。
    非 vector 模式：上下文改用对应模式的检索结果。"""
    if mode == "vector":
        retriever = _get_vs().as_retriever(search_kwargs={"k": 3})
        ctx_provider = retriever | format_docs
    else:
        def ctx_provider(q):
            return format_docs(retrieve_docs(q, mode=mode))

    return (
        {"context": ctx_provider, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "vector"
    print(f"检索模式：{mode}")
    chain = build_chain(mode=mode)
    while True:
        q = input("\n你：")
        if q in ("exit", "quit"): break
        print("助手：", chain.invoke(q))
