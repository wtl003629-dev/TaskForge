"""Freeze the 100-query open-literature benchmark with source attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://huggingface.co/datasets/shenhao/ScholarGym/resolve/main/"
    "scholargym_bench.jsonl"
)
SOURCE_SHA256 = "869f507eacb7f554b8f4e6dc65ea97b01f95140ba5aa86a5119b330c41b9d551"

# Real user-style bilingual research needs authored for TaskForge. Targets are
# public arXiv works and are labels, never query expansions at runtime.
CURATED: tuple[tuple[str, str, str, str, int], ...] = (
    ("curated-01", "检索增强生成最早如何把参数记忆与外部非参数记忆结合？", "2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020),
    ("curated-02", "开放域问答中，哪项工作用双编码器做稠密段落检索？", "2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering", 2020),
    ("curated-03", "有哪些论文让语言模型交替执行推理轨迹和外部动作？", "2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", 2022),
    ("curated-04", "查找让模型自主判断检索并通过反思 token 批判生成结果的工作。", "2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", 2023),
    ("curated-05", "Which foundational model introduced deep bidirectional pre-training with masked language modeling?", "1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018),
    ("curated-06", "Which paper established few-shot in-context learning at the 175B-parameter scale?", "2005.14165", "Language Models are Few-Shot Learners", 2020),
    ("curated-07", "找出首次提出完全基于注意力机制、去掉循环和卷积的序列模型论文。", "1706.03762", "Attention Is All You Need", 2017),
    ("curated-08", "参数高效微调中，哪篇论文通过低秩矩阵注入来适配大模型？", "2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", 2021),
    ("curated-09", "What work showed that chain-of-thought demonstrations elicit multi-step reasoning in large language models?", "2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2022),
    ("curated-10", "Find the paper that teaches a language model to decide when and how to call external tools using self-supervised API examples.", "2302.04761", "Toolformer: Language Models Can Teach Themselves to Use Tools", 2023),
    ("curated-11", "哪项检索工作通过 token 级 late interaction 在效果与效率之间折中？", "2004.12832", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", 2020),
    ("curated-12", "长上下文模型为何会忽略输入中间信息？请找对应的系统评测论文。", "2307.03172", "Lost in the Middle: How Language Models Use Long Contexts", 2023),
    ("curated-13", "Which system introduced billion-scale similarity search with GPU-accelerated vector indices?", "1702.08734", "Billion-scale similarity search with GPUs", 2017),
    ("curated-14", "检索从自然语言监督学习可迁移视觉表示、联合训练图像与文本编码器的论文。", "2103.00020", "Learning Transferable Visual Models From Natural Language Supervision", 2021),
    ("curated-15", "Which paper demonstrated that a pure transformer can classify images from sequences of patches?", "2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2020),
    ("curated-16", "深层视觉网络退化问题是通过哪篇残差学习论文解决的？", "1512.03385", "Deep Residual Learning for Image Recognition", 2015),
    ("curated-17", "Find the neural scene representation work that renders novel views using a continuous volumetric radiance field.", "2003.08934", "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", 2020),
    ("curated-18", "哪篇工作建立了去噪扩散概率模型的经典生成建模框架？", "2006.11239", "Denoising Diffusion Probabilistic Models", 2020),
    ("curated-19", "Which reinforcement-learning paper introduced the clipped surrogate objective used by PPO?", "1707.06347", "Proximal Policy Optimization Algorithms", 2017),
    ("curated-20", "查找不依赖人类棋谱、通过自我博弈掌握国际象棋将棋和围棋的通用算法论文。", "1712.01815", "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", 2017),
)


def _load_source(cache: Path) -> list[dict[str, object]]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310
            cache.write_bytes(response.read())
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"ScholarGym source hash mismatch: {digest}")
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]


def _case(row: dict[str, object]) -> dict[str, object]:
    papers = row.get("cited_paper")
    papers = papers if isinstance(papers, list) else []
    targets = [
        {
            "arxiv_id": str(paper["arxiv_id"]),
            "title": str(paper.get("title") or ""),
            "year": int(paper["year"]) if isinstance(paper.get("year"), int) else None,
            "relevance": int(label),
        }
        for paper, label in zip(papers, row.get("gt_label", []), strict=True)
        if isinstance(paper, dict) and paper.get("arxiv_id") and int(label) > 0
    ]
    years = [int(item["year"]) for item in targets if isinstance(item.get("year"), int)]
    return {
        "id": str(row["qid"]),
        "query": str(row["query"]),
        "source_group": str(row["source"]),
        "query_type": "real_scholar" if row["source"] == "PASA_RealScholar" else "litsearch",
        "target_papers": targets,
        "must_select_arxiv_ids": [item["arxiv_id"] for item in targets],
        "acceptable_arxiv_ids": [item["arxiv_id"] for item in targets],
        "year_from": max(1000, min(years) - 2) if years else None,
        "year_to": min(3000, max(years) + 1) if years else None,
        "venue_constraints": [],
    }


def build(rows: list[dict[str, object]]) -> dict[str, object]:
    real = [_case(row) for row in rows if row.get("source") == "PASA_RealScholar"]
    litsearch = [_case(row) for row in rows if row.get("source") == "LitSearch"][:30]
    curated = [
        {
            "id": case_id,
            "query": query,
            "source_group": "TaskForge_bilingual_curated",
            "query_type": "curated_actual_need",
            "target_papers": [
                {"arxiv_id": arxiv_id, "title": title, "year": year, "relevance": 1}
            ],
            "must_select_arxiv_ids": [arxiv_id],
            "acceptable_arxiv_ids": [arxiv_id],
            "year_from": year - 1,
            "year_to": year + 1,
            "venue_constraints": [],
        }
        for case_id, query, arxiv_id, title, year in CURATED
    ]
    cases = [*real, *litsearch, *curated]
    if len(real) != 50 or len(litsearch) != 30 or len(curated) != 20 or len(cases) != 100:
        raise RuntimeError("benchmark composition must be exactly 50 + 30 + 20")
    return {
        "schema_version": "1.0",
        "name": "taskforge-open-literature-discovery-100",
        "license": "Apache-2.0 for ScholarGym-derived rows; TaskForge curated rows",
        "source": {
            "name": "ScholarGym",
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "paper": "arXiv:2601.21654",
        },
        "composition": {
            "PASA_RealScholar": 50,
            "LitSearch": 30,
            "TaskForge_bilingual_curated": 20,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "scholargym_bench.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "literature-discovery-benchmark-100.json",
    )
    args = parser.parse_args()
    benchmark = build(_load_source(args.cache))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(benchmark["composition"], ensure_ascii=False))


if __name__ == "__main__":
    main()
