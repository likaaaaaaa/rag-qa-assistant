import os
import time
import streamlit as st
from rag import retrieve_docs, generate_from_docs, clean_citations, resolve_with_history


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
def recent_history(msgs, rounds=4):
    """取最近 rounds 轮对话做历史（助手消息去掉耗时头，供消解与生成注入）"""
    out = []
    for role, text in msgs[-(rounds * 2):]:
        if role == "assistant" and text.startswith("⚡"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        out.append((role, text))
    return out


if q := st.chat_input("输入你的问题…"):
    # 生成期间:占位容器显示"用户气泡 + 思考中",立即反馈不干等
    live = st.empty()
    with live.container():
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                history = recent_history(st.session_state.msgs)
                t0 = time.time()
                rq = resolve_with_history(q, history)
                t_resolve = time.time() - t0
                t1 = time.time()
                docs = retrieve_docs(rq, mode=mode)
                t_retrieve = time.time() - t1
                t2 = time.time()
                ans = generate_from_docs(q, docs, history=history)
                t_generate = time.time() - t2

    ans = clean_citations(ans, docs)
    head = f"⚡ 消解 {t_resolve:.1f}s · 检索 {t_retrieve:.1f}s · 生成 {t_generate:.1f}s"
    if rq != q:
        head += f"  (消解: {rq})"
    full = f"{head}\n\n{ans}"

    # 写入历史 → 清空占位容器 → rerun 统一重建(消除灰影)
    st.session_state.msgs.append(("user", q))
    st.session_state.msgs.append(("assistant", full))
    live.empty()
    st.rerun()
