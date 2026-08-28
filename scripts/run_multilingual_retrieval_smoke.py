"""Run a tiny real-model Chinese/cross-lingual retrieval smoke test.

This is a routing and local-model sanity check, not a QASPER quality claim.
It never calls an external LLM or provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskforge.hybrid_retrieval import FastEmbedEmbedder
from taskforge.knowledge import AccessContext, InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.research_reranking import build_research_reranker
from taskforge.research_retrieval import ResearchQuery, ResearchRetrievalService


def _corpus() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="zh-relevant",
            tenant_id="smoke",
            text=(
                "本文研究了图神经网络在医学文本分类任务中的性能，"
                "实验报告了准确率和召回率。"
            ),
            source_uri="paper://zh",
            document_id="zh-paper",
        ),
        KnowledgeChunk(
            chunk_id="zh-weather",
            tenant_id="smoke",
            text="本文讨论天气预报和气候变化趋势。",
            source_uri="paper://zh",
            document_id="zh-paper",
        ),
        KnowledgeChunk(
            chunk_id="en-relevant",
            tenant_id="smoke",
            text="This paper studies machine translation and reports BLEU scores.",
            source_uri="paper://en",
            document_id="en-paper",
        ),
    ]


def run(output: Path, cache_root: Path) -> None:
    embedder = FastEmbedEmbedder(
        "intfloat/multilingual-e5-large",
        model_cache_dir=cache_root,
    )
    reranker = build_research_reranker(
        "fastembed",
        "jinaai/jina-reranker-v2-base-multilingual",
        fastembed_cache_dir=cache_root,
    )
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(_corpus()),
        dense_embedder=embedder,
        reranker=reranker,
        multilingual_dense_embedder=embedder,
        multilingual_reranker=reranker,
        graph_enabled=False,
    )
    principal = AccessContext(tenant_id="smoke", user_id="smoke-user")
    cases = [
        (
            "zh_query_zh_corpus",
            "医学文本分类模型的准确率和召回率是多少？",
            "zh-relevant",
        ),
        (
            "en_query_cross_lingual_corpus",
            "Which study reports BLEU scores?",
            "en-relevant",
        ),
    ]
    rows: list[dict[str, object]] = []
    for case_id, query, expected_id in cases:
        result = service.search(
            ResearchQuery(query=query, top_k=3, candidate_k=10),
            principal,
        )
        top_ids = [item.chunk_id for item in result.evidence]
        rows.append(
            {
                "id": case_id,
                "query": query,
                "route": result.retrieval_route,
                "top_ids": top_ids,
                "top_scores": [item.score for item in result.evidence],
                "expected_top_id": expected_id,
                "passed": bool(top_ids and top_ids[0] == expected_id),
                "api_calls": 0,
            }
        )
    report = {
        "schema_version": "2.0",
        "status": "local_real_model_smoke_passed"
        if all(bool(row["passed"]) for row in rows)
        else "local_real_model_smoke_failed",
        "scope": "Chinese and cross-lingual local retrieval route",
        "api_calls": 0,
        "models": {
            "embedding": "intfloat/multilingual-e5-large",
            "reranker": "jinaai/jina-reranker-v2-base-multilingual",
            "cache_root": str(cache_root),
        },
        "cases": rows,
        "limitations": [
            "This is a two-case local smoke test, not a Chinese benchmark.",
            "It does not change the English QASPER headline metrics.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/multilingual-retrieval-smoke-v2.json"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".taskforge/model-cache/fastembed"),
    )
    args = parser.parse_args()
    run(args.output, args.cache_root)


if __name__ == "__main__":
    main()
