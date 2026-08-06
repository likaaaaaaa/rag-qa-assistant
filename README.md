# RAG 智能问答助手
基于本地知识库的大模型检索增强生成（RAG）应用。

## 架构
用户提问 → 向量检索(Top-3) → 拼接上下文 → 智谱 glm-4-flash 生成 → 返回答案+来源
文档加载 → 切分(500/50) → embedding-2 向量化 → Chroma 持久化

## 技术栈
LangChain · Chroma · 智谱 AI · Streamlit

## 本地运行
pip install -r requirements.txt
echo "ZHIPU_API_KEY=xxx" > .env
streamlit run app.py
