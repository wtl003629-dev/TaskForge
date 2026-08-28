"""Run a tiny real-API Chinese and cross-lingual Bailian retrieval smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.semantic_providers import BailianDenseEmbedder  # noqa: E402

DOCUMENTS = {
    "zh-medical": "本文研究图神经网络在医学文本分类中的表现，并报告准确率和召回率。",
    "en-translation": "This study evaluates machine translation and reports BLEU scores.",
    "zh-weather": "本文分析天气预报和长期气候变化趋势。",
    "en-finance": "This report discusses quarterly revenue and operating expenses.",
}

CASES = [
    ("zh_to_zh", "哪项研究报告医学文本分类的准确率和召回率？", "zh-medical"),
    ("en_to_en", "Which study reports BLEU scores?", "en-translation"),
    ("zh_to_en", "哪项研究报告了机器翻译的 BLEU 分数？", "en-translation"),
    (
        "en_to_zh",
        "Which study evaluates graph neural networks for medical text classification?",
        "zh-medical",
    ),
]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/bailian-multilingual-smoke-v1.json"),
    )
    parser.add_argument("--confirm-external-calls", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_calls:
        raise SystemExit(
            "refusing Bailian calls without --confirm-external-calls"
        )
    settings = Settings()
    if settings.bailian_api_key is None:
        raise SystemExit("TASKFORGE_BAILIAN_API_KEY is not configured")
    with BailianDenseEmbedder(
        api_key=settings.bailian_api_key.get_secret_value(),
        base_url=settings.bailian_base_url,
        model_name=settings.bailian_model,
        dimension=settings.bailian_embedding_dimension,
        batch_size=settings.bailian_batch_size,
        timeout_seconds=settings.bailian_timeout_seconds,
        max_retries=settings.bailian_max_retries,
        cache_path=settings.bailian_cache_path,
        index_name=settings.bailian_index_name,
    ) as embedder:
        document_ids = list(DOCUMENTS)
        document_vectors = embedder.embed_documents(list(DOCUMENTS.values()))
        rows: list[dict[str, object]] = []
        for case_id, query, expected in CASES:
            query_vector = embedder.embed_query(query)
            ranked = sorted(
                (
                    (document_id, _cosine(query_vector, vector))
                    for document_id, vector in zip(
                        document_ids,
                        document_vectors,
                        strict=True,
                    )
                ),
                key=lambda item: (-item[1], item[0]),
            )
            rows.append(
                {
                    "id": case_id,
                    "expected_top_id": expected,
                    "top_ids": [item[0] for item in ranked],
                    "top_scores": [item[1] for item in ranked],
                    "passed": ranked[0][0] == expected,
                }
            )
    report = {
        "schema_version": "1.0",
        "status": (
            "bailian_multilingual_smoke_passed"
            if all(bool(row["passed"]) for row in rows)
            else "bailian_multilingual_smoke_failed"
        ),
        "model": settings.bailian_model,
        "dimension": settings.bailian_embedding_dimension,
        "cases": rows,
        "limitations": [
            "This four-case smoke verifies routing sanity, not benchmark quality.",
            "Promotion still depends on a frozen representative multilingual set.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}))


if __name__ == "__main__":
    main()
