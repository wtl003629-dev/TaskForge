"""Re-project a frozen QASPER report onto the production Top-8 Agent head.

Older schema-2.2 reports serialized a larger returned evidence prefix.  Their
retrieval traces already contain the complete Candidate/Reranked lists, so we
can deterministically derive the schema-2.3 Top-8 view without parsing PDFs or
running a model again.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.qasper_alignment import GoldAlignment, aligned_recall_at_k
from taskforge.rag_evaluation import load_qasper_dataset

RECALL_KS = (1, 5, 10, 50)
VISIBLE_K = 8


def _ids(traces: list[dict[str, Any]], field: str) -> list[str]:
    for trace in reversed(traces):
        values = trace.get(field, [])
        if not isinstance(values, list):
            continue
        result = [
            str(item["chunk_id"])
            for item in values
            if isinstance(item, dict) and item.get("chunk_id")
        ]
        if result:
            return list(dict.fromkeys(result))
    return []


def reproject(input_path: Path, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    dataset = load_qasper_dataset(dataset_path)
    cases = {case.case_id: case for case in dataset.cases}
    rows = report.get("rows", [])
    for row in rows:
        labels = cases[str(row["case_id"])].qasper_gold
        alignments = {
            unit_id: GoldAlignment.model_validate(value)
            for unit_id, value in (row.get("gold_alignments") or {}).items()
        }
        traces = [
            trace for trace in row.get("retrieval_traces", []) if isinstance(trace, dict)
        ]
        reranked_ids = _ids(traces, "reranked_hits")
        candidate_ids = list(dict.fromkeys(
            str(hit["chunk_id"])
            for trace in traces
            for hit in trace.get("candidate_hits", [])
            if isinstance(hit, dict) and hit.get("chunk_id")
        ))
        visible_ids = [str(value) for value in row.get("retrieved_child_ids", [])[:VISIBLE_K]]
        if labels is None:
            raise ValueError(f"case lacks QASPER gold labels: {row['case_id']}")
        ranked = {
            str(k): aligned_recall_at_k(labels, alignments, reranked_ids, k)
            for k in RECALL_KS
        }
        candidate = {
            str(k): aligned_recall_at_k(labels, alignments, candidate_ids, k)
            for k in RECALL_KS
        }
        visible = aligned_recall_at_k(labels, alignments, visible_ids, VISIBLE_K)
        reranked_visible = aligned_recall_at_k(
            labels,
            alignments,
            reranked_ids,
            VISIBLE_K,
        )
        row["recall_at_k"] = {key: value.recall for key, value in ranked.items()}
        row["candidate_child_recall_at_k"] = {
            key: value.recall for key, value in candidate.items()
        }
        row["selected_annotation_at_k"] = {
            key: value.selected_annotation_id for key, value in ranked.items()
        }
        row["candidate_child_selected_annotation_at_k"] = {
            key: value.selected_annotation_id for key, value in candidate.items()
        }
        row["agent_visible_recall_at_8"] = visible.recall
        row["reranked_visible_recall_at_8"] = reranked_visible.recall
        row["retrieved_child_ids"] = visible_ids
        row["retrieved_count"] = len(visible_ids)
        row["retrieved_evidence"] = list(row.get("retrieved_evidence", []))[:VISIBLE_K]
        candidate_pool = aligned_recall_at_k(
            labels,
            alignments,
            candidate_ids,
            max(1, len(candidate_ids)),
        ).recall
        row["stage_recall"] = {
            "candidate_pool": candidate_pool,
            "reranked_top_10": ranked["10"].recall,
            "reranked_top_8": reranked_visible.recall,
            "agent_visible_top_8": visible.recall,
        }
        if row["stage_recall"]["candidate_pool"] < 1.0 - 1e-12:
            row["failure_stage"] = "candidate_missing"
        elif row["stage_recall"]["reranked_top_10"] < row["stage_recall"]["candidate_pool"] - 1e-12:
            row["failure_stage"] = "rerank_top10_missing"
        elif visible.recall < reranked_visible.recall - 1e-12:
            row["failure_stage"] = "presentation_window_missing"
        else:
            row["failure_stage"] = "retrieval_success"

    def mean_nested(parent: str, key: int) -> float:
        return statistics.fmean(float(row[parent][str(key)]) for row in rows) if rows else 0.0

    report["schema_version"] = "2.3"
    report["created_at"] = datetime.now(UTC).isoformat()
    report["retrieval"] = {**report["retrieval"], "agent_visible_k": VISIBLE_K}
    report["metrics"] = {
        **{f"recall_at_{k}": mean_nested("recall_at_k", k) for k in RECALL_KS},
        "p50_ms": report["metrics"]["p50_ms"],
        "p95_ms": report["metrics"]["p95_ms"],
    }
    report["candidate_child_metrics"] = {
        **{f"recall_at_{k}": mean_nested("candidate_child_recall_at_k", k) for k in RECALL_KS},
        "definition": "Gold paragraph coverage inside complete retrieved Child chunks; diagnostic only, not Agent-visible Recall.",
    }
    report["agent_visible_metrics"] = {
        "recall_at_8": statistics.fmean(float(row["agent_visible_recall_at_8"]) for row in rows)
        if rows else 0.0
    }
    report["reranked_visible_metrics"] = {
        "recall_at_8": statistics.fmean(float(row["reranked_visible_recall_at_8"]) for row in rows)
        if rows else 0.0
    }
    report["parser_diagnostics"] = {
        **report.get("parser_diagnostics", {}),
        "papers": report.get("papers", report.get("parser_diagnostics", {}).get("papers", 0)),
    }
    report["limitations"] = [
        item for item in report.get("limitations", [])
        if "schema 2.2" not in str(item).casefold()
    ] + [
        "This schema-2.3 Top-8 projection reuses frozen schema-2.2 PDF parses and retrieval traces; no model or parser call was made during reprojection."
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproject(args.input, args.dataset, args.output)
    print(json.dumps({"output": str(args.output), "cases": result["cases"], "metrics": result["metrics"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
