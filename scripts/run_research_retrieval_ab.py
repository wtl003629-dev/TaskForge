"""Run a deterministic A/B smoke test for the paper-research retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.research_retrieval import (  # noqa: E402
    ResearchQuery,
    ResearchRetrievalService,
)
from taskforge.routed_knowledge import RoutedKnowledgeStore  # noqa: E402


def _fixture_store() -> InMemoryKnowledgeStore:
    return InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="paper-a-method",
                tenant_id="tenant-a",
                text="Paper A uses retrieval augmented generation and reports recall 0.91.",
                source_uri="paper://a",
                document_id="paper-a",
                metadata={"evidence_id": "paper-a:method", "title": "Paper A", "section": "Method"},
            ),
            KnowledgeChunk(
                chunk_id="paper-a-table",
                tenant_id="tenant-a",
                text="Table 2 reports recall 0.91 on the test set.",
                source_uri="paper://a",
                document_id="paper-a",
                metadata={"evidence_id": "paper-a:table", "title": "Paper A", "kind": "table", "table_rows": ["recall 0.91"], "page": 7},
            ),
            KnowledgeChunk(
                chunk_id="paper-b-result",
                tenant_id="tenant-a",
                text="Paper B reports recall 0.88 with a different retriever.",
                source_uri="paper://b",
                document_id="paper-b",
                metadata={"evidence_id": "paper-b:result", "title": "Paper B", "section": "Results"},
            ),
        ]
    )


def _score(ids: list[str], gold: set[str]) -> dict[str, float | int]:
    found = set(ids)
    hits = len(found & gold)
    return {"hit_count": hits, "gold_count": len(gold), "recall_at_10": hits / len(gold) if gold else 1.0}


def run(cases_path: Path) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("cases must be a JSON array")
    store = _fixture_store()
    principal = AccessContext(tenant_id="tenant-a", user_id="research-eval")
    routed = RoutedKnowledgeStore(store, general_text_backend="bm25")
    unified = ResearchRetrievalService(store, operator_budget_standard=0, operator_budget_rigorous=0)
    adaptive = ResearchRetrievalService(store, operator_budget_standard=1, operator_budget_rigorous=2)
    rows: list[dict[str, object]] = []
    for case in cases:
        query = str(case["query"])
        gold = {str(value) for value in case["gold_evidence_ids"]}
        started = time.perf_counter()
        routed_hits = routed.search(query, principal, top_k=10)
        routed_result = {
            "evidence_ids": [
                str(hit.chunk.metadata.get("evidence_id") or hit.chunk.chunk_id)
                for hit in routed_hits
            ],
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        unified_result = unified.search(ResearchQuery(query=query, mode="standard"), principal)
        adaptive_result = adaptive.search(ResearchQuery(query=query, mode="rigorous"), principal)
        rows.append(
            {
                "id": case["id"],
                "query": query,
                "gold_evidence_ids": sorted(gold),
                "routed": {**routed_result, **_score(routed_result["evidence_ids"], gold)},
                "unified": {
                    "evidence_ids": [item.evidence_id for item in unified_result.evidence],
                    **_score([item.evidence_id for item in unified_result.evidence], gold),
                    "activated_operators": list(unified_result.activated_operators),
                    "elapsed_ms": unified_result.elapsed_ms,
                },
                "adaptive": {
                    "evidence_ids": [item.evidence_id for item in adaptive_result.evidence],
                    **_score([item.evidence_id for item in adaptive_result.evidence], gold),
                    "activated_operators": list(adaptive_result.activated_operators),
                    "elapsed_ms": adaptive_result.elapsed_ms,
                },
            }
        )
    summary: dict[str, dict[str, float]] = {}
    for name in ("routed", "unified", "adaptive"):
        values = [float(row[name]["recall_at_10"]) for row in rows]
        summary[name] = {"mean_recall_at_10": sum(values) / len(values) if values else 0.0}
    dataset_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    return {"schema_version": "1.0", "dataset": str(cases_path), "dataset_sha256": dataset_hash, "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "eval" / "research-retrieval-cases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval" / "research_retrieval_ab.json")
    args = parser.parse_args()
    report = run(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
