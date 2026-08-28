"""Summarize completed real-paper Flat vs Parent-Child evaluation reports."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "eval" / "reports"
SPLIT_ROOT = PROJECT_ROOT / "eval" / "splits"
OUTPUT = REPORT_ROOT / "real-paper-chunking-matrix-v1.json"

RUNS = {
    "chinese_same_language": (
        "chinese-paper-fulltext-15-v1-bailian-current-flat-v1",
        "chinese-paper-fulltext-15-v1-bailian-current-parent_child-v1",
    ),
    "english_query_to_chinese_papers": (
        "chinese-paper-fulltext-en-query-15-v1-bailian-current-flat-v1",
        "chinese-paper-fulltext-en-query-15-v1-bailian-current-parent_child-v1",
    ),
    "bilingual_mixed_corpus": (
        "bilingual-paper-mixed-15-v1-bailian-current-flat-v1",
        "bilingual-paper-mixed-15-v1-bailian-current-parent_child-v1",
    ),
}


def _read(name: str) -> dict[str, object]:
    path = REPORT_ROOT / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"report is not complete: {path}")
    split = Path(str(report["split"]))
    actual = hashlib.sha256(split.read_bytes()).hexdigest()
    if actual != report.get("split_sha256"):
        raise ValueError(f"split hash changed after run: {path}")
    return report


def _row(
    report: dict[str, object], *, chunking: str, scenario: str
) -> dict[str, object]:
    metrics = report["metrics"]
    alignment = report["alignment_diagnostics"]
    retrieval = report["retrieval"]
    ingestion = report.get("ingestion") or []
    parser_statuses = [
        str(item.get("parse_quality", {}).get("status", "unknown"))
        for item in ingestion
        if isinstance(item, dict)
    ]
    languages: dict[str, int] = {}
    split = json.loads(Path(str(report["split"])).read_text(encoding="utf-8"))
    for item in split.get("selected_rows", []):
        if isinstance(item, dict):
            language_value = item.get("query_language")
            if language_value:
                language = str(language_value)
            else:
                # The original Chinese split predates the query_language field.
                # Infer its language from the frozen question text without
                # changing the split (and therefore without invalidating runs).
                query = str(item.get("query") or item.get("question") or "")
                language = (
                    "zh"
                    if any("\u3400" <= char <= "\u9fff" for char in query)
                    else "en"
                )
            languages[language] = languages.get(language, 0) + 1
    return {
        "run": Path(str(report["source_dataset"])).stem + ":" + chunking,
        "chunking": chunking,
        "status": report["status"],
        "papers": report["papers"],
        "cases": report["cases"],
        "query_languages": languages,
        "embedding_model": retrieval.get("semantic_model"),
        "reranker_backend": retrieval.get("reranker_backend"),
        "reranker_model": retrieval.get("reranker_model"),
        "retrieval_route_counts": retrieval.get("retrieval_route_counts"),
        "metrics": {
            key: metrics[key]
            for key in (
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_50",
                "mrr",
                "ndcg_at_8",
                "p50_ms",
                "p95_ms",
            )
        },
        "alignment": {
            key: alignment[key]
            for key in (
                "alignment_coverage",
                "alignment_eligible_case_ratio",
                "ambiguous_units",
                "unaligned_units",
            )
        },
        "parser_quality_statuses": parser_statuses,
        "indexed_chunks": sum(
            int(item.get("indexed_chunks", 0))
            for item in ingestion
            if isinstance(item, dict)
        ),
        "split_sha256": report["split_sha256"],
        "pdf_manifest_sha256": (report.get("pdf_manifest") or {}).get("sha256")
        if isinstance(report.get("pdf_manifest"), dict)
        else None,
    }


def build() -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for scenario, (flat_name, parent_name) in RUNS.items():
        flat = _read(flat_name)
        parent = _read(parent_name)
        flat_row = _row(flat, chunking="flat", scenario=scenario)
        parent_row = _row(parent, chunking="parent_child", scenario=scenario)
        flat_metrics = flat_row["metrics"]
        parent_metrics = parent_row["metrics"]
        delta = {
            key: float(parent_metrics[key]) - float(flat_metrics[key])
            for key in flat_metrics
        }
        scenarios[scenario] = {
            "flat": flat_row,
            "parent_child": parent_row,
            "delta_parent_child_minus_flat": delta,
        }
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "real full-text scholarly PDF RAG",
        "scenarios": scenarios,
        "excluded_runs": [
            {
                "run": "all multilingual-reranker-profile variants",
                "reason": "jinaai/jina-reranker-v2-base-multilingual ONNX file is absent from the local fastembed cache; no multilingual reranker score is included.",
            }
        ],
        "interpretation": {
            "retrieval": "Recall/MRR/nDCG use parser-native gold evidence paragraphs; p50/p95 are end-to-end query latency.",
            "chinese_labels": "15 manually authored, frozen, auditable evidence questions over five real Chinese JOS PDFs.",
            "bilingual_labels": "Mixed scenario combines the five Chinese PDFs with five checksum-pinned real QASPER PDFs; QASPER queries are translated into Chinese and Chinese-paper queries into English.",
            "parser": "Scored runs use the native pypdf+pdfplumber parser. Visual-pending is retained as a parser-quality diagnostic, not silently treated as a chunking win.",
        },
        "decision": {
            "production_default": "keep_current_flat",
            "parent_child_status": "experimental_only",
            "rollback": "No live chain switch was made; the current Flat chain remains the rollback/default path.",
            "reason": "Parent-Child is not uniformly better: it helps deeper Top-10 ranking on same-language Chinese queries but regresses English-to-Chinese and bilingual mixed-corpus head ranking.",
        },
    }


def main() -> None:
    report = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "scenarios": len(report["scenarios"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
