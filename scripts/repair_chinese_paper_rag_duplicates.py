"""Repair duplicate question text in the generated Chinese RAG annotations."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_chinese_paper_rag_annotations as builder  # noqa: E402

from taskforge.config import Settings  # noqa: E402


def _repair_prompt(title: str, abstract: str, chunks: list[str], forbidden: list[str]) -> str:
    numbered = "\n\n".join(
        f"[CHUNK_{index:03d}]\n{chunk}" for index, chunk in enumerate(chunks)
    )
    blocked = "；".join(forbidden)
    return f"""你是中文科研论文 RAG 数据集标注员。只能依据给出的论文文本块工作。

论文标题：{title}
论文摘要：{abstract}

请生成恰好 3 条中文问题，类型必须分别为 method、result、contribution。
本次重点修复 contribution 问题：它必须包含论文中的具体方法名、系统名、数据集名或量化实验结论；禁止使用以下已经重复的问题：{blocked}
每个答案必须被给出的文本块直接支持；每条 evidence 至少一个 Chunk；chunk_index 必须有效；quote 必须来自对应 Chunk 的连续原文（可保留论文中的空格和标点）。
问题不能只询问作者、标题、DOI 或参考文献，不能回答文本中没有的信息。
只输出合法 JSON，不要 Markdown：
{{"items":[{{"question_type":"method|result|contribution","question":"...","answer":"...","evidence":[{{"chunk_index":0,"quote":"..."}}],"difficulty":"easy|medium|hard"}}]}}

论文文本块：
{numbered}
"""


def main() -> None:
    output = PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v1"
    annotations_path = output / "annotations.jsonl"
    source = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2" / "corpus.jsonl.gz"
    rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_rows = {str(row["paper_id"]): row for row in builder._read_jsonl_gz(source)}
    by_question: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for item_index, item in enumerate(row["items"]):
            by_question[str(item["question"]).strip()].append((row_index, item_index))
    duplicate_groups = [
        values
        for question, values in by_question.items()
        if len(values) > 1 and question.strip()
    ]
    targets = [location for group in duplicate_groups for location in group[1:]]
    if not targets:
        print("duplicate_questions=0")
        return
    settings = Settings()
    if settings.deepseek_api_key is None or settings.deepseek_model is None:
        raise RuntimeError("DeepSeek credentials are not configured")
    client = httpx.Client(timeout=httpx.Timeout(150.0))
    try:
        for row_index, item_index in targets:
            row = rows[row_index]
            paper_id = str(row["paper_id"])
            paper = source_rows[paper_id]
            chunks = builder._chunk_text(str(paper["text"]))
            forbidden = [
                str(rows[group_row]["items"][group_item]["question"])
                for group in duplicate_groups
                for group_row, group_item in group
                if group_row == row_index or rows[group_row]["items"][group_item]["question"] == row["items"][item_index]["question"]
            ]
            current = rows[row_index]["items"][item_index]["question"]
            replacement = None
            error = ""
            for _ in range(3):
                raw = builder._call_deepseek(
                    client,
                    url=settings.deepseek_base_url.rstrip("/") + "/chat/completions",
                    api_key=settings.deepseek_api_key.get_secret_value(),
                    model=settings.deepseek_model,
                    prompt=_repair_prompt(
                        str(paper.get("title", "")),
                        str(paper.get("abstract", "")),
                        chunks,
                        forbidden,
                    ),
                )
                candidates, error = builder._validate_items(raw, chunks)
                if candidates:
                    candidate = next(
                        item
                        for item in candidates
                        if item["question_type"] == rows[row_index]["items"][item_index]["question_type"]
                    )
                    if candidate["question"] != current and candidate["question"] not in by_question:
                        replacement = candidate
                        break
                forbidden.append(current)
            if replacement is None:
                raise RuntimeError(f"could not repair duplicate question for {paper_id}: {error}")
            rows[row_index]["items"][item_index] = replacement
            print(f"repaired {paper_id}")
    finally:
        client.close()
    builder._write_jsonl(output / "annotations.partial.jsonl", rows)
    # Rebuild derived query/qrels/chunk files without issuing more requests.
    builder.build_dataset(
        source=source,
        output=output,
        seed=20260827,
        limit=30,
        confirm_external_calls=True,
    )


if __name__ == "__main__":
    main()
