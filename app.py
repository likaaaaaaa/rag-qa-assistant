import os
import time
import re
import streamlit as st
from rag import retrieve_docs, generate_from_docs


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


st.set_page_config(page_title="AI 面试题库 · RAG 智能问答", page_icon="🤖")
st.title("AI 面试题库 · RAG 智能问答")

# ============ 检索模式开关(侧边栏,一键切换) ============
MODE_INFO = {
    "vector": ("纯向量检索", "仅按语义相似度检索 top-3,最基础的基线模式。"),
    "hybrid": ("混合检索", "向量 + BM25 关键词双路检索,RRF 融合,兼顾语义与精确匹配。"),
    "hybrid_rerank": ("混合检索 + Rerank", "混合检索候选经智谱 Rerank 精排取 top-3,效果最准。"),
}
mode = st.sidebar.radio(
    "检索模式",
    list(MODE_INFO.keys()),
    format_func=lambda m: MODE_INFO[m][0],
    help="切换后立即生效,首次加载问答链约需几秒。",
)
st.sidebar.caption(f"当前:{MODE_INFO[mode][1]}")

# ============ 消息历史(单一真理源:所有渲染都从 msgs 来) ============
if "msgs" not in st.session_state:
    st.session_state.msgs = []

for role, text in st.session_state.msgs:
    st.chat_message(role).write(text)

# ============ 新问题处理 ============
if q := st.chat_input("输入你的问题…"):
    # 生成期间:占位容器显示"用户气泡 + 思考中",立即反馈不干等
    live = st.empty()
    with live.container():
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                t0 = time.time()
                docs = retrieve_docs(q, mode=mode)
                t_retrieve = time.time() - t0
                t1 = time.time()
                ans = generate_from_docs(q, docs)
                t_generate = time.time() - t1

    ans = clean_citations(ans, docs)
    head = f"⚡ 检索 {t_retrieve:.1f}s · 生成 {t_generate:.1f}s"
    full = f"{head}\n\n{ans}"

    # 写入历史 → 清空占位容器 → rerun 统一重建(消除灰影)
    st.session_state.msgs.append(("user", q))
    st.session_state.msgs.append(("assistant", full))
    live.empty()
    st.rerun()
