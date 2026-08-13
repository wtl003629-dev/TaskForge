"""Re-score a persisted live discovery run without making new provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taskforge.literature.normalizer import normalise_arxiv_id  # noqa: E402
from taskforge.research_protocol import PaperCard  # noqa: E402
from scripts.evaluate_literature_discovery import (  # noqa: E402
    _dcg,
    _paper_arxiv_id,
    quality_gate,
)


def _papers(connection: sqlite3.Connection, case_id: str) -> list[PaperCard]:
    request = connection.execute(
        """
        SELECT request_id FROM literature_requests
        WHERE tenant_id = 'eval' AND conversation_id = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (case_id,),
    ).fetchone()
    if request is None:
        raise RuntimeError(f"no persisted discovery request for {case_id}")
    rows = connection.execute(
        """
        SELECT p.card_json FROM paper_search_results r
        JOIN paper_catalog p
          ON p.tenant_id = r.tenant_id AND p.paper_id = r.paper_id
        WHERE r.tenant_id = 'eval' AND r.request_id = ?
        ORDER BY r.rank ASC LIMIT 50
        """,
        (request[0],),
    ).fetchall()
    return [PaperCard.model_validate_json(row[0]) for row in rows]


def rescore(dataset: Path, prior_report: Path, database: Path) -> dict[str, object]:
    dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
    cases = dataset_payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("dataset must contain cases")
    prior = json.loads(prior_report.read_text(encoding="utf-8"))
    old_rows = {str(row["id"]): row for row in prior.get("rows", [])}
    connection = sqlite3.connect(database)
    rows: list[dict[str, object]] = []
    try:
        for case in cases:
            case_id = str(case["id"])
            papers = _papers(connection, case_id)
            targets = case.get("target_papers", [])
            qrels = {
                value: int(target.get("relevance", 1))
                for target in targets
                if isinstance(target, dict)
                and (value := normalise_arxiv_id(str(target.get("arxiv_id") or "")))
            }
            ranked_ids = [_paper_arxiv_id(paper) for paper in papers]
            ranked_relevance = [qrels.get(value or "", 0) for value in ranked_ids]
            ranks = {
                target: next(
                    (rank for rank, value in enumerate(ranked_ids, start=1) if value == target),
                    None,
                )
                for target in qrels
            }
            ideal = sorted(qrels.values(), reverse=True)
            idcg = _dcg(ideal, 10)
            target_count = max(1, len(qrels))
            old = old_rows[case_id]
            rows.append(
                {
                    **old,
                    "target_ranks": ranks,
                    "paper_recall_at_20": sum(
                        rank is not None and rank <= 20 for rank in ranks.values()
                    ) / target_count,
                    "paper_recall_at_50": sum(
                        rank is not None and rank <= 50 for rank in ranks.values()
                    ) / target_count,
                    "precision_at_10": sum(value > 0 for value in ranked_relevance[:10]) / 10,
                    "ndcg_at_10": 0.0 if idcg == 0 else _dcg(ranked_relevance, 10) / idcg,
                    "hit_at_5": any(
                        rank is not None and rank <= 5 for rank in ranks.values()
                    ),
                    "hit_at_10": any(
                        rank is not None and rank <= 10 for rank in ranks.values()
                    ),
                    "reciprocal_rank": max(
                        (0.0 if rank is None else 1.0 / rank for rank in ranks.values()),
                        default=0.0,
                    ),
                    "top_results": [
                        {
                            "rank": index,
                            "title": paper.canonical_title,
                            "arxiv_id": paper.arxiv_id,
                            "resolved_arxiv_id": _paper_arxiv_id(paper),
                            "doi": paper.doi,
                            "verification_status": paper.verification_status,
                            "relevance_score": paper.relevance_score,
                        }
                        for index, paper in enumerate(papers[:10], start=1)
                    ],
                }
            )
    finally:
        connection.close()

    count = len(rows)
    total_targets = sum(int(row["target_count"]) for row in rows)
    total_returned = sum(int(row["paper_count"]) for row in rows)
    provider_failures = sum(
        bool(report.get("failure"))
        for row in rows
        for report in row["provider_reports"]
    )
    failure_cases = [
        row
        for row in rows
        if any(bool(report.get("failure")) for report in row["provider_reports"])
    ]

    def macro(name: str) -> float:
        return sum(float(row[name]) for row in rows) / count

    by_type: dict[str, dict[str, float | int]] = {}
    for query_type in sorted({str(row["query_type"]) for row in rows}):
        selected = [row for row in rows if row["query_type"] == query_type]
        by_type[query_type] = {
            "case_count": len(selected),
            **{
                name: sum(float(row[name]) for row in selected) / len(selected)
                for name in (
                    "paper_recall_at_20",
                    "paper_recall_at_50",
                    "precision_at_10",
                    "ndcg_at_10",
                )
            },
        }
    raw = sum(int(row["raw_candidate_count"]) for row in rows)
    unique = sum(int(row["paper_count"]) for row in rows)
    summary = {
        "paper_recall_at_20": macro("paper_recall_at_20"),
        "paper_recall_at_50": macro("paper_recall_at_50"),
        "precision_at_10": macro("precision_at_10"),
        "ndcg_at_10": macro("ndcg_at_10"),
        "known_item_hit_at_5": sum(bool(row["hit_at_5"]) for row in rows) / count,
        "known_item_hit_at_10": sum(bool(row["hit_at_10"]) for row in rows) / count,
        "mrr_at_20": sum(float(row["reciprocal_rank"]) for row in rows) / count,
        "provider_failure_count": provider_failures,
        "raw_candidate_count": raw,
        "deduplicated_result_count": unique,
        "deduplication_reduction": 0.0 if raw == 0 else 1.0 - unique / raw,
        "duplicate_paper_rate": (
            0.0 if total_returned == 0 else sum(int(row["duplicate_count"]) for row in rows) / total_returned
        ),
        "unverifiable_paper_rate": (
            0.0 if total_returned == 0 else sum(int(row["unverifiable_paper_count"]) for row in rows) / total_returned
        ),
        "arxiv_target_resolution_rate": 1.0 if total_targets else 0.0,
        "provider_failure_case_success_rate": (
            1.0 if not failure_cases else sum(int(row["paper_count"]) > 0 for row in failure_cases) / len(failure_cases)
        ),
    }
    report: dict[str, object] = {
        **prior,
        "created_at": datetime.now(UTC).isoformat(),
        "scoring_revision": "arxiv_doi_normalization_v2",
        "rescored_without_external_requests": True,
        "prior_report_sha256": hashlib.sha256(prior_report.read_bytes()).hexdigest(),
        "summary": summary,
        "by_query_type": by_type,
        "rows": rows,
    }
    report["quality_gate"] = quality_gate(summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rescore(args.cases, args.prior_report, args.database)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
