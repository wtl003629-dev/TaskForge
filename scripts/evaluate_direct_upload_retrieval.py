"""Evaluate PDF upload -> parse -> index -> bounded retrieval recall."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.synthetic_pdf_eval import (  # noqa: E402
    generate_synthetic_pdfs,
    load_synthetic_suite,
)


def _pages(value: object) -> set[int]:
    return {int(item) for item in re.findall(r"\d+", str(value or ""))}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _settings(state: Path) -> Settings:
    return Settings(
        _env_file=None,
        sqlite_path=state / "taskforge.sqlite3",
        context_sqlite_path=state / "context.sqlite3",
        operations_sqlite_path=state / "operations.sqlite3",
        orchestration_sqlite_path=state / "orchestration.sqlite3",
        review_case_sqlite_path=state / "review.sqlite3",
        verification_sqlite_path=state / "verification.sqlite3",
        literature_sqlite_path=state / "literature.sqlite3",
        literature_cache_path=state / "literature-cache.sqlite3",
        workspace_root=PROJECT_ROOT,
        artifact_root=state / "artifacts",
        retrieval_routing="lexical",
        general_text_backend="bm25",
        provider="demo",
    )


def run(suite_path: Path, output: Path) -> dict[str, object]:
    suite = load_synthetic_suite(suite_path)
    started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="taskforge-direct-upload-eval-",
        ignore_cleanup_errors=True,
    ) as raw:
        state = Path(raw)
        manifest = generate_synthetic_pdfs(suite_path, state / "pdfs")
        app = create_app(_settings(state))
        auth = {
            "X-TaskForge-Tenant": "upload-eval",
            "X-TaskForge-User": "evaluator",
        }
        scopes: dict[str, dict[str, object]] = {}
        ingestion: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        generated = {item.document_id: item for item in manifest.documents}
        with TestClient(app) as client:
            for document in suite.documents:
                pdf = generated[document.document_id]
                payload = Path(pdf.path).read_bytes()
                upload = client.post(
                    "/api/research/uploads",
                    headers={
                        **auth,
                        "Content-Type": "application/pdf",
                        "X-Filename": document.filename,
                    },
                    params={
                        "conversation_id": f"upload-eval-{document.document_id}",
                        "user_intent": "Answer questions using only this uploaded PDF.",
                        "title": document.pages[0].title,
                    },
                    content=payload,
                )
                upload.raise_for_status()
                uploaded = upload.json()
                scope = uploaded["scope"]
                indexed = client.post(
                    f"/api/research/scopes/{scope['scope_id']}/ingest",
                    headers=auth,
                )
                indexed.raise_for_status()
                statuses = indexed.json()
                if [item["status"] for item in statuses] != ["indexed"]:
                    raise RuntimeError(f"PDF ingestion failed: {statuses}")
                scopes[document.document_id] = scope
                ingestion.append(
                    {
                        "document_id": document.document_id,
                        "filename": document.filename,
                        "bytes": len(payload),
                        "evidence_count": statuses[0]["evidence_count"],
                        "scope_status": client.get(
                            f"/api/research/scopes/{scope['scope_id']}",
                            headers=auth,
                        ).json()["status"],
                    }
                )

            for case in suite.cases:
                if len(case.evidence) != 1:
                    raise ValueError("direct-upload suite cases must reference one PDF")
                gold = case.evidence[0]
                scope = scopes[gold.document_id]
                intent = "numeric_table" if case.category == "table" else "general_fact"
                query_started = perf_counter()
                response = client.post(
                    "/api/research/evidence/search",
                    headers=auth,
                    json={
                        "scope_id": scope["scope_id"],
                        "scope_version": scope["scope_version"],
                        "query": case.question,
                        "intent": intent,
                        "top_k": 50,
                        "candidate_k": 50,
                        "mode": "rigorous",
                    },
                )
                latency_ms = (perf_counter() - query_started) * 1_000
                response.raise_for_status()
                result = response.json()
                evidence = result["evidence"]
                gold_pages = set(gold.pages)
                recalls: dict[str, float] = {}
                for k in (1, 5, 10, 50):
                    retrieved = set().union(
                        *(_pages(item.get("page")) for item in evidence[:k])
                    )
                    recalls[str(k)] = len(gold_pages & retrieved) / len(gold_pages)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "document_id": gold.document_id,
                        "category": case.category,
                        "question": case.question,
                        "gold_pages": sorted(gold_pages),
                        "retrieved_pages": [item.get("page") for item in evidence],
                        "recall_at_k": recalls,
                        "latency_ms": latency_ms,
                        "retrieval_rounds": result["retrieval_rounds"],
                    }
                )

    by_category: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    for category, items in sorted(grouped.items()):
        by_category[category] = {
            "cases": len(items),
            **{
                f"recall_at_{k}": statistics.fmean(
                    float(item["recall_at_k"][str(k)])  # type: ignore[index]
                    for item in items
                )
                for k in (1, 5, 10, 50)
            },
        }
    latencies = [float(row["latency_ms"]) for row in rows]
    recall = {
        str(k): statistics.fmean(
            float(row["recall_at_k"][str(k)])  # type: ignore[index]
            for row in rows
        )
        for k in (1, 5, 10, 50)
    }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_type": "direct_pdf_upload_retrieval_recall",
        "created_at": datetime.now(UTC).isoformat(),
        "suite_id": suite.suite_id,
        "license": suite.license,
        "documents": len(suite.documents),
        "cases": len(rows),
        "pipeline": ["direct_upload", "pdf_parse", "chunk", "index", "bounded_search"],
        "passed": recall["10"] >= 0.80,
        "thresholds": {"recall_at_10": 0.80},
        "metrics": {
            "recall_at_1": recall["1"],
            "recall_at_5": recall["5"],
            "recall_at_10": recall["10"],
            "recall_at_50": recall["50"],
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
        },
        "by_category": by_category,
        "ingestion": ingestion,
        "rows": rows,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            "The suite contains 12 self-authored cases over 3 real PDF binaries.",
            "It is a pipeline regression gate, not a production-scale public benchmark.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "eval" / "synthetic_pdf_suite.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "direct-upload-retrieval.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.suite, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
