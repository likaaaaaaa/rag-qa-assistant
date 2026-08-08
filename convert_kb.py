# -*- coding: utf-8 -*-
"""把 awesome-ai-knowledge 的 md 转成知识库 txt（清洗 markdown 语法）"""
import os
import re

SRC = r"C:\Users\19602\Downloads\awesome-ai-knowledge\docs"
DST = r"C:\Users\19602\WorkBuddy\2026-08-06-14-58-52\rag-qa-assistant\data"
PREFIX = "kb_"  # 新知识库文件加前缀，与 sample.txt 区分


def clean_md(text: str) -> list[tuple[str, str]]:
    """返回 [(标题, 内容), ...]"""
    lines = text.split("\n")
    blocks = []          # 当前问题块
    sections = []        # 所有 (标题, 内容)
    cur_title = None
    cur_content = []

    for raw in lines:
        line = raw.strip()
        # 主题大标题（# xxx）→ 作为前缀信息，跳过（标题会包含文件名）
        if line.startswith("# "):
            continue
        # 返回首页/引用/分割线 → 跳过
        if line.startswith("[←") or line.startswith("> 题目") or line.startswith("> 给"):
            continue
        if line == "---" or line == "":
            continue
        # 问题标题（## N. xxx）
        if line.startswith("## "):
            if cur_title and cur_content:
                sections.append((cur_title, "\n".join(cur_content)))
            cur_title = line[3:].strip()
            cur_content = []
            continue
        # 四段结构标签（**💡 一句话先懂** 等）→ 转成小标题行
        m = re.match(r"\*\*[💡🌰🔑🚀]?\s*(.+?)\*\*", line)
        if m and cur_title is not None:
            if cur_content:
                cur_content.append("")
            cur_content.append(m.group(1).strip())
            continue
        # 普通内容：去 markdown 符号
        if cur_title is not None:
            clean = line
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)      # **bold**
            clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)   # [text](url)
            clean = re.sub(r"^[-\*]\s+", "· ", clean)           # 列表项
            clean = re.sub(r"^#{1,6}\s*", "", clean)            # 残余标题符
            cur_content.append(clean)

    if cur_title and cur_content:
        sections.append((cur_title, "\n".join(cur_content)))
    return sections


def main():
    os.makedirs(DST, exist_ok=True)
    total = 0
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(SRC, fname), encoding="utf-8") as f:
            text = f.read()
        sections = clean_md(text)
        if not sections:
            continue
        out_name = PREFIX + os.path.splitext(fname)[0] + ".txt"
        out_path = os.path.join(DST, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (title, content) in enumerate(sections, 1):
                f.write(f"{title}\n\n{content}\n\n")
        total += len(sections)
        print(f"✓ {out_name}: {len(sections)} 个知识点（{os.path.getsize(out_path)} 字节）")
    print(f"\n共转换 {total} 个知识点 → data/ 目录")


if __name__ == "__main__":
    main()
