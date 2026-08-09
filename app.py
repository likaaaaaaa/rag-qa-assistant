import streamlit as st
from rag import build_chain

st.set_page_config(page_title="AI 面试题库 · RAG 智能问答", page_icon="🤖")
st.title("AI 面试题库 · RAG 智能问答")

# ============ 检索模式开关(侧边栏,一键切换) ============
# vector        纯向量检索(阶段0 基线)
# hybrid        向量 + BM25 混合检索(阶段1A)
# hybrid_rerank 混合检索 + 智谱 Rerank 精排(阶段1B,最准)
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
st.sidebar.caption(f"当前：{MODE_INFO[mode][1]}")

# 按模式缓存问答链:切回已用过的模式时秒开,不用重建
@st.cache_resource
def init(mode):
    return build_chain(mode=mode)

chain = init(mode)

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for role, text in st.session_state.msgs:
    st.chat_message(role).write(text)

if q := st.chat_input("输入你的问题…"):
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        ans = chain.invoke(q)
        st.write(ans)
    st.session_state.msgs.append(("user", q))
    st.session_state.msgs.append(("assistant", ans))
