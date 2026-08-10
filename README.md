# AI 领域知识库 RAG 问答助手

基于本地知识库的大模型检索增强生成（RAG）应用：**211 段 AI 领域知识库 · 混合检索 + Rerank 优化 · RAGAS 量化评测 · Streamlit 已部署上线**。

> 🔗 **在线体验**:https://rag-app-assistant-lp6fulldtxdpde5ba4f5eg.streamlit.app

## 功能亮点

- **知识库**：11 大主题（机器学习 / 深度学习 / 大模型 / RAG / Agent / Prompt / 微调 / 向量 / 部署 / 评估 / 多模态），211 段内容（课程知识 + 开源资料整理）；
- **检索优化（三模式可插拔）**：
  - `vector`：纯向量检索（基线）
  - `hybrid`：向量 + BM25（jieba 中文分词）→ RRF 融合
  - `hybrid_rerank`：混合检索候选 → 智谱 Rerank 精排 → top-3
- **量化评测**：自建测试集 + RAGAS 四指标（忠实度/相关度/召回/精确度）；
- **优化成果**：
  - 14 题旧库：检索精确度 0.75 → **0.89（+19%）**，忠实度保持不降；
  - 30 题新库复测：混合检索精确度 0.861 → **0.894**、召回 0.856 → **0.933**，剔除拒答口径精确度 0.923 → **0.958**；
- 拒答约束 + 引用标注（带来源文件名），答案可溯源、不编造；
- 聊天 UI：检索/生成耗时显示、检索模式三档切换（侧边栏）、引用来源代码补全。

## 架构

```
用户提问 → 混合检索(向量+BM25, Top-10) → RRF 融合 → Rerank 精排 → Top-3 上下文
文档加载 → 切分(500/50) → embedding-2 分批向量化 → Chroma 持久化 → 检索 → 智谱 glm-4-flash 生成
```

## 技术栈

LangChain · Chroma · 智谱 AI（glm-4-flash / embedding-2 / Rerank）· BM25 · jieba · RAGAS · Streamlit

## 本地运行

```bash
pip install -r requirements.txt
echo "ZHIPU_API_KEY=xxx" > .env
streamlit run app.py        # 或命令行体验：python rag.py
```

## 评测

```bash
python eval_ragas.py --mode vector           # 基线
python eval_ragas.py --mode hybrid           # 混合检索
python eval_ragas.py --mode hybrid_rerank    # 混合 + Rerank
```
