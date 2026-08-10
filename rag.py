import os
import re
import json
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
1. 仅根据【上下文】回答。若上下文信息不足，必须明确回答"抱歉，知识库中未找到相关内容"，宁可拒答也绝不凭记忆补全、编造或引入上下文之外的概念（这是最重要的一条）；
2. 拒答规则：若上下文没有相关信息，或检索内容与问题无关，必须明确回答"抱歉，知识库中未找到相关内容"，绝不编造，不得根据上下文中的比喻、例子推断出答案；
3. 回答结构：先一句话直接给结论，再分要点展开，语言通俗、层次清晰；
4. 引用规则（强制）：任何取自【上下文】的句子必须在末尾用 [N] 标注引用编号（N 对应【上下文】第 N 段），多条用 [1][2] 顺序编号，不必在编号后写文件名（引用区由代码补全）；未标注视为该句无依据、应拒答；
5. 讲解原则：涉及原理/对比的问题，用大白话解释"为什么"，帮用户真正理解而非背诵；
6. 【对话历史】仅用于理解前文语境（如"它"指什么），回答内容必须以【上下文】检索资料为准，不得把历史对话当作事实来源引用。

【对话历史】
{history}

【上下文】
{context}

【问题】{question}
【回答】"""
)

# Query 改写：只对明显口语的问题改写，规范问法直接跳过（防止检索被无谓改写干扰）
# 改写失败/丢失核心词/超时 → 回落原 query，不阻塞主流程
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """你是"AI 面试题库"的检索查询改写器，把用户的口语化提问改写成适合检索的规范问法。
规则：
1. 只改写表达方式，不增删信息、不编造、不回答问题；
2. 清理口语和废话："咋"→"如何"、"帮我看下""我想问一下"这类前缀去掉；
3. 保留原问题的核心名词和关键概念，不得替换或丢掉（硬性要求）；
4. 可补全省略的主语，但不引入上下文之外的概念；
5. 只输出改写后的一句话问法，不要解释、不要引号。

【问题】{question}
【改写结果】"""
)

# 口语特征词：命中才需要改写（规范问法不改，防止检索被无谓改写干扰）
SPOKEN = ["咋", "啥", "嘛", "呗", "帮我看", "帮我看看", "是不是", "能不能",
          "会不会", "有没有", "到底", "咋样", "咋回事", "行吗"]
# 提取核心词时的停用词（口语词/虚词/弱词）
_STOP = {"什么", "怎么", "如何", "是", "的", "了", "吗", "呢", "啊", "咋", "啥",
         "让", "把", "那个", "这个", "一下", "请问", "帮我", "看下", "到底",
         "是不是", "能不能", "会不会", "有没有", "咋样", "咋回事", "咋搞", "咋弄",
         "咋实现", "咋知道", "咋让", "咋写", "咋存", "咋记", "行吗",
         "的话", "之前", "之后", "它", "他", "她", "这", "那", "点", "会", "要"}
_rewrite_chain = None


def is_spoken_query(query):
    """是否口语问法：命中口语特征词才需要改写"""
    return any(w in query for w in SPOKEN)


def _core_terms(query):
    """提取 query 的核心实词（jieba 分词后过滤停用词）"""
    return [t.strip() for t in jieba.cut(query)
            if len(t.strip()) >= 2 and t.strip() not in _STOP]


def rewrite_query(query):
    """口语 query → 检索友好问法。
    规范问法直接跳过；改写失败/丢失核心词/超时 → 回落原 query"""
    if not is_spoken_query(query):
        return query
    global _rewrite_chain
    if _rewrite_chain is None:
        _rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    try:
        rewritten = _rewrite_chain.invoke(query).strip().strip('"').strip('“”')
        if not rewritten or rewritten == query:
            return query
        # 核心词校验：改写结果必须保留原 query 的核心实词，丢了说明改跑偏
        core = _core_terms(query)
        if core and not any(c in rewritten for c in core):
            return query
        return rewritten
    except Exception:
        return query


# ============ 多轮对话记忆：检索前消解指代/省略，生成时注入历史 ============
HISTORY_PROMPT = ChatPromptTemplate.from_template(
    """你是对话历史理解器。根据对话历史，把当前用户问题补全成一条"独立可检索的完整问法"。
规则：
1. 当前问题里有指代（它/这个/那个/这些/那些/它们/这/那）或省略主语时，从历史中找出所指的具体概念并替换补全；
2. 历史中没有对应信息，或当前问题本身完整时，原样输出当前问题；
3. 不增删当前问题的核心意图，不编造历史里没有的信息；
4. 只输出补全后的一句话，不要解释、不要引号。

【对话历史】
{history}

【当前问题】{question}
【补全结果】"""
)

# 指代词：命中或短问题才需要消解（防误伤长规范问题）
PRONOUNS = ["它", "这个", "那个", "这些", "那些", "它们", "这", "那"]
_resolve_chain = None


def need_resolve(query, history):
    """是否需要多轮消解：有历史 + (短问题 或 含指代词)"""
    if not history:
        return False
    return len(query) <= 12 or any(p in query for p in PRONOUNS)


def format_history(history, max_rounds=4, max_chars=150):
    """把最近 max_rounds 轮对话拼成历史文本，每条消息截断到 max_chars"""
    lines = []
    for role, text in history[-max_rounds * 2:]:
        t = text if len(text) <= max_chars else text[:max_chars] + "…"
        prefix = "用户" if role == "user" else "助手"
        lines.append(f"{prefix}：{t}")
    return "\n".join(lines)


def resolve_with_history(query, history):
    """带历史的多轮消解：补全指代/省略为完整问法；失败/丢核心词回落原 query"""
    if not need_resolve(query, history):
        return query
    global _resolve_chain
    if _resolve_chain is None:
        _resolve_chain = HISTORY_PROMPT | llm | StrOutputParser()
    try:
        resolved = _resolve_chain.invoke(
            {"history": format_history(history), "question": query}
        ).strip().strip('"').strip('“”')
        if not resolved or resolved == query:
            return query
        # 核心词校验：消解结果必须保留当前问题的核心词，防消解跑偏
        core = _core_terms(query)
        if core and not any(c in resolved for c in core):
            return query
        return resolved
    except Exception:
        return query


def process_citations(text, docs):
    """引用后处理：
    ① 把 [N](链接) 洗成纯文本 [N]（不信任模型写的链接）；
    ② 按首次出现顺序把 docs 引用重映射为连续编号 [1][2][3]（正文不跳号）；
    ③ 返回 (新文本, citations{新编号: {source, snippet}})，供 UI 渲染可点击跳转的引用区。
    若文本无任何引用，citations 为空。"""
    if not text:
        return text, {}

    def src_of(idx):
        if 1 <= idx <= len(docs):
            return os.path.basename(docs[idx - 1].metadata.get('source', '未知'))
        return None

    # ① 洗掉模型编的链接：[N](xxx) → [N]
    text = re.sub(r"\[(\d+)\]\([^)]*\)", lambda m: f"[{m.group(1)}]", text)

    # ② 重映射：按 docs 引用首次出现顺序编号
    order = {}          # docs索引 -> 新编号
    counter = [0]

    def repl(m):
        idx = int(m.group(1))
        if not (1 <= idx <= len(docs)):
            return m.group(0)
        if idx not in order:
            counter[0] += 1
            order[idx] = counter[0]
        return f"[{order[idx]}]"

    new_text = re.sub(r"\[(\d+)\]", repl, text)

    # ③ 构造引用映射：新编号 -> 来源文件 + 段落预览
    citations = {}
    for idx, num in order.items():
        d = docs[idx - 1]
        citations[num] = {
            "source": os.path.basename(d.metadata.get('source', '未知')),
            "snippet": d.page_content,
        }

    # ④ 兜底：回答非空但 0 引用（非拒答文案）→ 自动挂第一条 docs 为 [1]，避免"无引用空文"
    if new_text.strip() and not citations and docs:
        if not any(kw in new_text for kw in ["抱歉", "未找到"]):
            first = docs[0]
            citations[1] = {
                "source": os.path.basename(first.metadata.get('source', '未知')),
                "snippet": first.page_content,
            }
            new_text = new_text.rstrip() + " [1]"

    return new_text, citations


# ============ 回答自检（可验证 RAG）：代码拆句 + 模型逐句标依据，引用编号保留 ============
CHECK_PROMPT = ChatPromptTemplate.from_template(
    """你是事实核查员。下面是按句拆分的 AI 回答（每句带编号），以及它依据的检索资料。
请逐句判断每句是否能从【检索资料】中找到依据，输出每句的编号和判断。
输出严格 JSON，格式如下：
{{"verdicts": [{{"index": 0, "supported": true}}, {{"index": 1, "supported": false}}, ...]}}
判断规则：
1. 编号句子里出现 [N] 这样的引用标记时，它是引用编号不是内容，验证内容时忽略它；
2. supported=true：该句内容（含同义改写）能在资料中找到依据，或由有依据的句子直接推出；
3. supported=false：该句是资料里没有的（凭记忆补全、编造、过度推断）；
4. 只判断，不要改写句子；verdicts 按原编号顺序输出，不要省略。

【检索资料】
{context}

【编号句子】
{sentences}"""
)
_check_chain = None


def self_check(answer, docs):
    """生成后自检：代码拆句 + 模型逐句标 supported，按原句拼接（引用编号 100% 保留）。
    无依据句子丢弃；失败时原样返回（自检是增强不是依赖）。返回 (新回答, 丢弃句数)。"""
    if not answer or not docs:
        return answer, 0
    sentences = [s.strip() for s in re.split(r'(?<=[。！？；\n])', answer) if s.strip()]
    if not sentences:
        return answer, 0
    global _check_chain
    if _check_chain is None:
        _check_chain = CHECK_PROMPT | llm | StrOutputParser()
    try:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        out = _check_chain.invoke(
            {"context": format_docs(docs), "sentences": numbered}
        ).strip()
        if out.startswith("```"):  # 去掉模型可能带的代码块包裹
            out = out.split("```")[1]
            if out.startswith("json"):
                out = out[4:]
        data = json.loads(out)
        verdicts = data.get("verdicts", [])
        if not verdicts:
            return answer, 0
        supported = {v["index"] for v in verdicts
                     if v.get("supported") and 0 <= v["index"] < len(sentences)}
        if not supported:
            return "抱歉，知识库中未找到足够信息支撑回答。", len(sentences)
        kept = [sentences[i] for i in range(len(sentences)) if i in supported]
        new_answer = "".join(kept)
        dropped = len(sentences) - len(supported)
        if dropped > 0:
            new_answer += f"\n\n> （已自动移除 {dropped} 句无依据内容）"
        return new_answer, dropped
    except Exception:
        return answer, 0


def clean_citations(text, docs):
    """引用兜底两件事:
    1) 把 [N](链接) 洗成 [N] 来源:文件名(基于实际检索结果,不信任模型写的链接);
    2) 给裸的 [N] 补上来源文件名(编号顺序 = docs 顺序),带尾空格分隔。"""
    def src_of(idx):
        if 1 <= idx <= len(docs):
            return os.path.basename(docs[idx - 1].metadata.get('source', '未知'))
        return None

    def repl_link(m):
        idx = int(m.group(1))
        s = src_of(idx)
        return f"[{idx}] 来源:{s}" if s else m.group(0)

    text = re.sub(r"\[(\d+)\]\(([^)]+)\)", repl_link, text)

    def repl_plain(m):
        idx = int(m.group(1))
        s = src_of(idx)
        return f"[{idx}] 来源:{s} " if s else m.group(0)

    return re.sub(r"\[(\d+)\](?!\s*来源[:：])", repl_plain, text)


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


def retrieve_docs(query, mode="vector", rewrite=False):
    """统一的文档检索入口：返回按相关度排序的 list[Document]。
    rewrite=True：口语 query 改写后与原 query 双路召回、合并去重，再统一精排
    （改写只负责扩召回，排序由 rerank 用原 query 决定，防止改写跑偏丢掉正确结果）"""
    if mode == "vector":
        return _get_vs().as_retriever(search_kwargs={"k": 3}).invoke(query)
    if rewrite:
        rq = rewrite_query(query)
        if rq != query:
            merged = _dedup(_get_ensemble().invoke(rq) + _get_ensemble().invoke(query))
            if mode == "hybrid":
                return merged[:5]
            return zhipu_rerank(query, merged, top_n=5)
    candidates = _get_ensemble().invoke(query)
    if mode == "hybrid":
        return candidates[:5]                    # RRF 融合后取 top-5
    if mode == "hybrid_rerank":
        return zhipu_rerank(query, candidates, top_n=5)
    raise ValueError(f"未知检索模式: {mode}")


def _dedup(docs):
    """按内容前缀去重（合并两路召回时用）"""
    seen, out = set(), []
    for d in docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def build_chain(mode="vector", rewrite=False):
    """构建问答链。默认 vector：与原版行为完全一致（线上部署零影响）。
    rewrite=True 时检索前做 Query 改写（双路召回，只影响检索，不影响生成的问题原文）。"""
    if mode == "vector":
        retriever = _get_vs().as_retriever(search_kwargs={"k": 3})

        def ctx_provider(q):
            rq = rewrite_query(q) if rewrite else q
            return format_docs(retriever.invoke(rq))
    else:
        def ctx_provider(q):
            return format_docs(retrieve_docs(q, mode=mode, rewrite=rewrite))

    return (
        {"context": ctx_provider, "question": RunnablePassthrough(), "history": lambda _: ""}
        | PROMPT | llm | StrOutputParser()
    )


def generate_from_docs(query, docs, history=None):
    """用已检索到的 docs 直接生成回答（不重复检索）。
    history：最近对话列表，注入生成以保持连贯（回答仍以检索资料为准）。"""
    context = format_docs(docs)
    hist = format_history(history) if history else ""
    return (PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": query, "history": hist}
    )


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "vector"
    print(f"检索模式：{mode} | 多轮记忆：开（输入问题前会先做指代消解）")
    history = []
    while True:
        q = input("\n你：")
        if q in ("exit", "quit"):
            break
        rq = resolve_with_history(q, history)
        if rq != q:
            print(f"消解：{q} → {rq}")
        docs = retrieve_docs(rq, mode=mode)
        ans_raw = generate_from_docs(q, docs, history=history)
        ans_raw, dropped = self_check(ans_raw, docs)
        ans, cites = process_citations(ans_raw, docs)
        if dropped:
            print(f"（自检移除 {dropped} 句无依据内容）")
        print("助手：", ans)
        if cites:
            print("引用来源：")
            for num in sorted(cites):
                print(f"  [{num}] {cites[num]['source']}")
                print(f"      {cites[num]['snippet'][:80]}…")
        history.append(("user", q))
        history.append(("assistant", ans))
