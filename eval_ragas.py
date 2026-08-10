import os
import asyncio
import argparse
import pandas as pd
import sniffio
from dotenv import load_dotenv
load_dotenv()

# 修复 sniffio 检测：显式声明当前运行环境是 asyncio，避免 AsyncLibraryNotFoundError
try:
    sniffio._impl.thread_local.name = "asyncio"
except Exception:
    pass

# 验证 ragas 的同步补丁是否生效（防止 __pycache__ 缓存了旧代码）
import inspect
from ragas.llms.base import LangchainLLMWrapper
_src = inspect.getsource(LangchainLLMWrapper.agenerate_text)
print("✓ ragas 同步补丁已生效" if "兼容补丁" in _src else "✗ ragas 同步补丁未生效（缓存问题）")

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from rag import build_chain, retrieve_docs
from test_set import TEST_SET

# ① 命令行参数：检索模式 + 输出文件名
parser = argparse.ArgumentParser(description="RAGAS 评测（支持三档检索模式对比）")
parser.add_argument("--mode", default="vector",
                    choices=["vector", "hybrid", "hybrid_rerank"],
                    help="vector=纯向量基线 / hybrid=混合检索 / hybrid_rerank=混合检索+Rerank")
parser.add_argument("--output", default=None, help="输出 CSV 文件名（默认按模式命名）")
parser.add_argument("--rewrite", action="store_true", help="开启 Query 改写（用于对比改写前后效果）")
parser.add_argument("--subset", default="all", choices=["all", "e"],
                    help="all=全部39题 / e=仅E类口语化9题（省 API 用量）")
args = parser.parse_args()
MODE = args.mode
REWRITE = args.rewrite
if args.subset == "e":
    TEST_SET = TEST_SET[-9:]
    print(f"⚠ 子集模式：仅 E 类口语化题 {len(TEST_SET)} 题（省 API 用量）")
OUTPUT = args.output or {
    "vector": "评测基线_纯向量.csv",
    "hybrid": "评测基线_混合检索.csv",
    "hybrid_rerank": "评测基线_混合检索+Rerank.csv",
}[MODE]
print(f"检索模式：{MODE} | Query 改写：{'开' if REWRITE else '关'} → 输出：{OUTPUT}")

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE = "https://open.bigmodel.cn/api/paas/v4"

# ② 裁判模型（智谱当评委）——注意 temperature=0，打分要稳定
judge_llm = ChatOpenAI(model="glm-4-flash", api_key=API_KEY, base_url=BASE, temperature=0, timeout=60)
llm_wrapper = LangchainLLMWrapper(judge_llm)
emb_wrapper = LangchainEmbeddingsWrapper(
    ZhipuAIEmbeddings(model="embedding-2", api_key=API_KEY)
)

# ③ 把"裁判"接给每个指标（0.2.x 写法：给指标对象赋 llm / embeddings）
faithfulness.llm = llm_wrapper
answer_relevancy.llm = llm_wrapper
answer_relevancy.embeddings = emb_wrapper
context_recall.llm = llm_wrapper

metrics_list = [faithfulness, answer_relevancy, context_recall]

# ④ 第 4 个指标：检索精确度（0.2.x 中它是"类"，要实例化；版本没有会自动跳过）
try:
    from ragas.metrics import LLMContextPrecisionWithoutReference
    context_precision_metric = LLMContextPrecisionWithoutReference()
    context_precision_metric.llm = llm_wrapper
    metrics_list.append(context_precision_metric)
    print("✓ 已启用第 4 个指标：检索精确度")
except ImportError:
    print("ℹ 本版本无检索精确度指标，先跑 3 个核心指标")

# ⑤ 加载系统：链与检索器都用同一个 mode（保证 answer 基于同一批上下文）
chain = build_chain(mode=MODE, rewrite=REWRITE)

# ⑥ 跑系统：对每个问题拿到 检索上下文 + 系统回答
samples = []
for item in TEST_SET:
    q = item["question"]
    try:
        docs = retrieve_docs(q, mode=MODE, rewrite=REWRITE)
        contexts = [d.page_content for d in docs]
        answer = chain.invoke(q)
    except Exception as e:
        print(f"✗ 问答失败({q[:20]}): {e!r}", flush=True)
        docs, contexts, answer = [], [], ""
    samples.append(SingleTurnSample(
        user_input=q,
        retrieved_contexts=contexts,
        response=answer,
        reference=item["reference"],
    ))
    print(f"✓ {q[:20]}...", flush=True)

# ⑦ 手写打分循环：逐样本、逐指标调用（绕开 ragas 的异步 Executor，
#    规避 Python 3.14 的 asyncio 兼容问题）。串行执行，56 次调用，耐心等几分钟。
async def run_scores(metrics_list, samples):
    import traceback
    all_rows = []
    for idx, s in enumerate(samples):
        row = {"question": s.user_input}
        for m in metrics_list:
            try:
                row[m.name] = await m.single_turn_ascore(sample=s, callbacks=[])
            except Exception as e:
                print(f"--- {m.name} 打分失败，完整堆栈：")
                traceback.print_exc()
                print(f"--- 错误详情: {e!r}")
                row[m.name] = f"ERROR: {type(e).__name__}"
        all_rows.append(row)
        print(f"✓ 第 {idx+1}/{len(samples)} 题打分完成")
    return all_rows

print("开始评测（串行打分，请耐心等待）…")
rows = asyncio.run(run_scores(metrics_list, samples))

df = pd.DataFrame(rows)
# 列名翻译成中文（可读性更好，面试展示也直观）
COL_ZH = {
    "question": "问题",
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关度",
    "context_recall": "检索召回",
    "llm_context_precision_without_reference": "检索精确度",
}
df = df.rename(columns=COL_ZH)
print("\n======== 评测结果 ========")
print(df.to_string())
df.to_csv(OUTPUT, index=False)
print(f"\n已保存到 {OUTPUT}")
