import streamlit as st
from rag import build_chain

st.set_page_config(page_title="RAG 智能问答助手", page_icon="🤖")
st.title("🤖 RAG 智能问答助手")

@st.cache_resource
def init():
    return build_chain()

chain = init()
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
