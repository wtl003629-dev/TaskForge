"""Evaluate live open-literature discovery separately from bounded RAG recall."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.hybrid_retrieval import FastEmbedEmbedder  # noqa: E402
from taskforge.literature import (  # noqa: E402
    LiteratureAccess,
    LiteratureDiscoveryService,
    OpenAICompatibleQueryRewriter,
    SQLiteLiteratureRepository,
)
from taskforge.literature.normalizer import (  # noqa: E402
    arxiv_id_from_doi,
    normalise_arxiv_id,
)
from taskforge.literature.providers import (  # noqa: E402
    ArxivProvider,
    CrossrefProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
    SQLiteProviderCache,
)
from taskforge.research_protocol import LiteratureRequest, PaperCard  # noqa: E402


def _paper_arxiv_id(paper: PaperCard) -> str | None:
    return normalise_arxiv_id(paper.arxiv_id) or arxiv_id_from_doi(paper.doi)


def _find_rank(papers: list[PaperCard], expected_arxiv_id: str) -> int | None:
    expected = normalise_arxiv_id(expected_arxiv_id)
    for rank, paper in enumerate(papers, start=1):
        if _paper_arxiv_id(paper) == expected:
            return rank
    return None


def _dcg(relevances: list[int], k: int) -> float:
    import math

    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances[:k])
    )


def quality_gate(summary: dict[str, object]) -> dict[str, object]:
    targets = {
        "evaluation_valid": {"operator": "==", "target": 1.0},
        "recommendation_link_coverage": {"operator": ">=", "target": 0.98},
        "short_description_coverage": {"operator": ">=", "target": 0.98},
        "paper_recall_at_20": {"operator": ">=", "target": 0.80},
        "paper_recall_at_50": {"operator": ">=", "target": 0.90},
        "precision_at_10": {"operator": ">=", "target": 0.60},
        "ndcg_at_10": {"operator": ">=", "target": 0.70},
        "arxiv_target_resolution_rate": {"operator": ">=", "target": 0.98},
        "duplicate_paper_rate": {"operator": "<=", "target": 0.02},
        "unverifiable_paper_rate": {"operator": "==", "target": 0.0},
        "provider_failure_case_success_rate": {"operator": "==", "target": 1.0},
    }
    checks: dict[str, dict[str, object]] = {}
    for name, target in targets.items():
        actual = float(summary.get(name, 0.0))
        expected = float(target["target"])
        operator = str(target["operator"])
        passed = (
            actual >= expected
            if operator == ">="
            else actual <= expected
            if operator == "<="
            else actual == expected
        )
        checks[name] = {**target, "actual": actual, "passed": passed}
    return {"passed": all(bool(value["passed"]) for value in checks.values()), "checks": checks}


async def evaluate(
    cases_path: Path,
    state_dir: Path,
    *,
    concurrency: int = 4,
    provider_names: set[str] | None = None,
    case_limit: int | None = None,
    case_ids: set[str] | None = None,
    dense_model: str | None = None,
    query_rewrite: bool = False,
    rewrite_model: str | None = None,
) -> dict[str, object]:
    raw_dataset = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = raw_dataset.get("cases") if isinstance(raw_dataset, dict) else raw_dataset
    if not isinstance(cases, list) or not cases:
        raise ValueError("literature benchmark must contain a non-empty cases list")
    if case_ids:
        cases = [case for case in cases if str(case.get("id")) in case_ids]
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive")
        cases = cases[:case_limit]
    if not cases:
        raise ValueError("literature benchmark selection is empty")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    cache = SQLiteProviderCache(state_dir / "provider-cache.sqlite3", ttl_seconds=86_400)
    options = {
        "cache": cache,
        "timeout_seconds": 20.0,
        "max_retries": 1,
        "rate_limit_failure_threshold": 1,
    }
    settings = Settings()
    api_key = (
        os.getenv("TASKFORGE_SEMANTIC_SCHOLAR_API_KEY", "")
        or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        or (
            settings.semantic_scholar_api_key.get_secret_value()
            if settings.semantic_scholar_api_key is not None
            else ""
        )
    ).strip()
    contact_email = (
        os.getenv("TASKFORGE_LITERATURE_CONTACT_EMAIL", "")
        or settings.literature_contact_email
        or ""
    ).strip()
    openalex_api_key = (
        os.getenv("TASKFORGE_OPENALEX_API_KEY", "")
        or (
            settings.openalex_api_key.get_secret_value()
            if settings.openalex_api_key is not None
            else ""
        )
    ).strip()
    contact_headers = (
        {"User-Agent": f"TaskForge/0.3 (mailto:{contact_email})"}
        if contact_email
        else {}
    )
    available_providers = [
        SemanticScholarProvider(
            headers={**contact_headers, **({"x-api-key": api_key} if api_key else {})},
            concurrency=1,
            min_interval_seconds=1.0,
            **options,
        ),
        OpenAlexProvider(
            headers={
                **contact_headers,
                **(
                    {"Authorization": f"Bearer {openalex_api_key}"}
                    if openalex_api_key
                    else {}
                ),
            },
            concurrency=1,
            min_interval_seconds=1.05,
            **options,
        ),
        ArxivProvider(
            headers=contact_headers,
            concurrency=1,
            min_interval_seconds=3.1,
            **options,
        ),
        CrossrefProvider(
            headers=contact_headers,
            concurrency=1,
            min_interval_seconds=0.21,
            **options,
        ),
    ]
    selected_names = provider_names or {
        "semantic_scholar",
        "openalex",
        "arxiv",
        "crossref",
    }
    providers = [
        provider for provider in available_providers if provider.name in selected_names
    ]
    unknown_names = selected_names - {provider.name for provider in available_providers}
    if unknown_names:
        raise ValueError(f"unknown literature providers: {sorted(unknown_names)}")
    if not providers:
        raise ValueError("at least one literature provider is required")
    for provider in available_providers:
        if provider not in providers:
            await provider.aclose()
    dense_embedder = (
        FastEmbedEmbedder(
            dense_model,
            cache_path=state_dir / "paper-embeddings.sqlite3",
        )
        if dense_model
        else None
    )
    query_rewriter = None
    if query_rewrite:
        if settings.deepseek_api_key is None:
            raise ValueError("query rewrite requested but DeepSeek API key is not configured")
        selected_rewrite_model = rewrite_model or settings.deepseek_model
        if not selected_rewrite_model:
            raise ValueError("query rewrite requested but DeepSeek model is not configured")
        query_rewriter = OpenAICompatibleQueryRewriter(
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=selected_rewrite_model,
            base_url=settings.deepseek_base_url,
            timeout_seconds=settings.deepseek_timeout_seconds,
        )
    repository = SQLiteLiteratureRepository(state_dir / "literature.sqlite3")
    service = LiteratureDiscoveryService(
        repository,
        providers,
        results_per_query=50,
        dense_embedder=dense_embedder,
        query_rewriter=query_rewriter,
    )
    rows: list[dict[str, object]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(case: dict[str, object]) -> dict[str, object]:
        async with semaphore:
            case_id = str(case["id"])
            targets = case.get("target_papers")
            if not isinstance(targets, list):
                targets = [
                    {
                        "arxiv_id": str(case["expected_arxiv_id"]),
                        "relevance": 1,
                    }
                ]
            qrels = {
                normalise_arxiv_id(str(target["arxiv_id"])): int(
                    target.get("relevance", 1)
                )
                for target in targets
                if isinstance(target, dict)
                and normalise_arxiv_id(str(target.get("arxiv_id") or ""))
            }
            result = await service.discover(
                LiteratureAccess("eval", "discovery-evaluator", case_id),
                LiteratureRequest(
                    request_id=f"discovery-eval-{case_id}-{run_id}",
                    query=str(case["query"]),
                    year_from=(
                        int(case["year_from"])
                        if isinstance(case.get("year_from"), int)
                        else None
                    ),
                    year_to=(
                        int(case["year_to"])
                        if isinstance(case.get("year_to"), int)
                        else None
                    ),
                    result_limit=50,
                ),
            )
            ranks = {
                arxiv_id: _find_rank(result.papers, arxiv_id)
                for arxiv_id in qrels
            }
            ranked_relevance = [
                qrels.get(_paper_arxiv_id(paper) or "", 0)
                for paper in result.papers
            ]
            ideal = sorted(qrels.values(), reverse=True)
            idcg10 = _dcg(ideal, 10)
            relevant_at_10 = sum(value > 0 for value in ranked_relevance[:10])
            target_count = max(1, len(qrels))
            paper_keys = [
                (
                    _paper_arxiv_id(paper)
                    or (paper.doi or "").casefold()
                    or paper.paper_id
                )
                for paper in result.papers
            ]
            unverifiable = sum(
                not (
                    paper.arxiv_id
                    or paper.doi
                    or paper.semantic_scholar_id
                    or paper.openalex_id
                    or paper.source_urls
                )
                for paper in result.papers
            )
            return {
                    "id": case_id,
                    "query": case["query"],
                    "query_type": case.get("query_type", "known_item"),
                    "target_count": len(qrels),
                    "target_ranks": ranks,
                    "paper_recall_at_20": sum(
                        rank is not None and rank <= 20 for rank in ranks.values()
                    ) / target_count,
                    "paper_recall_at_50": sum(
                        rank is not None and rank <= 50 for rank in ranks.values()
                    ) / target_count,
                    "relevant_at_20_count": sum(
                        rank is not None and rank <= 20 for rank in ranks.values()
                    ),
                    "relevant_at_50_count": sum(
                        rank is not None and rank <= 50 for rank in ranks.values()
                    ),
                    "precision_at_10": relevant_at_10 / 10,
                    "ndcg_at_10": 0.0 if idcg10 == 0 else _dcg(ranked_relevance, 10) / idcg10,
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
                    "arxiv_target_resolution_rate": len(qrels) / target_count,
                    "paper_count": len(result.papers),
                    "recommendation_count_at_10": len(result.papers[:10]),
                    "recommendation_link_count_at_10": sum(
                        any(url.startswith("https://") for url in paper.source_urls)
                        for paper in result.papers[:10]
                    ),
                    "short_description_count_at_10": sum(
                        bool(paper.short_description.strip())
                        for paper in result.papers[:10]
                    ),
                    "duplicate_count": len(paper_keys) - len(set(paper_keys)),
                    "unverifiable_paper_count": unverifiable,
                    "raw_candidate_count": result.total_raw_candidates,
                    "cross_source_verified_count": sum(
                        paper.verification_status == "cross_source_verified"
                        for paper in result.papers
                    ),
                    "query_rewrite_applied": result.query_rewrite_applied,
                    "query_rewrite_failure": result.query_rewrite_failure,
                    "planned_queries": [
                        query.model_dump(mode="json") for query in result.queries
                    ],
                    "provider_reports": [
                        report.model_dump(mode="json") for report in result.provider_reports
                    ],
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
                        for index, paper in enumerate(result.papers[:10], start=1)
                    ],
                }

    try:
        rows = list(await asyncio.gather(*(run_case(case) for case in cases)))
    finally:
        await service.aclose()

    count = len(rows)
    provider_failures = sum(
        bool(report.get("failure"))
        for row in rows
        for report in row["provider_reports"]
    )
    raw = sum(int(row["raw_candidate_count"]) for row in rows)
    unique = sum(int(row["paper_count"]) for row in rows)
    total_targets = sum(int(row["target_count"]) for row in rows)
    total_returned = sum(int(row["paper_count"]) for row in rows)
    recommendation_count = sum(
        int(row["recommendation_count_at_10"]) for row in rows
    )
    failure_cases = [
        row
        for row in rows
        if any(bool(report.get("failure")) for report in row["provider_reports"])
    ]
    provider_status: dict[str, dict[str, int | float]] = {}
    for provider_name in sorted(selected_names):
        reports = [
            report
            for row in rows
            for report in row["provider_reports"]
            if report.get("provider") == provider_name
        ]
        successful = sum(not bool(report.get("failure")) for report in reports)
        provider_status[provider_name] = {
            "case_count": len(reports),
            "successful_case_count": successful,
            "success_rate": successful / max(1, len(reports)),
            "request_count": sum(int(report.get("request_count", 0)) for report in reports),
            "result_count": sum(int(report.get("result_count", 0)) for report in reports),
        }
    semantic_success_rate = max(
        (
            float(provider_status[name]["success_rate"])
            for name in ("semantic_scholar", "openalex")
            if name in provider_status
        ),
        default=0.0,
    )
    arxiv_success_rate = float(
        provider_status.get("arxiv", {}).get("success_rate", 0.0)
    )
    evaluation_valid = semantic_success_rate >= 0.95 and arxiv_success_rate >= 0.95

    def macro(name: str) -> float:
        return sum(float(row[name]) for row in rows) / count

    by_type: dict[str, dict[str, float | int]] = {}
    for query_type in sorted({str(row["query_type"]) for row in rows}):
        selected = [row for row in rows if row["query_type"] == query_type]
        by_type[query_type] = {
            "case_count": len(selected),
            "paper_recall_at_20": sum(
                float(row["paper_recall_at_20"]) for row in selected
            ) / len(selected),
            "paper_recall_at_50": sum(
                float(row["paper_recall_at_50"]) for row in selected
            ) / len(selected),
            "precision_at_10": sum(
                float(row["precision_at_10"]) for row in selected
            ) / len(selected),
            "ndcg_at_10": sum(float(row["ndcg_at_10"]) for row in selected)
            / len(selected),
        }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_type": "open_literature_discovery",
        "metric_boundary": (
            "Paper-level known-item discovery over open providers; these metrics are "
            "not bounded passage Recall@10 or Candidate@50."
        ),
        "live_external_requests": True,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(cases_path),
        "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "configuration": {
            "providers": sorted(selected_names),
            "results_per_query": 50,
            "provider_query_limits": service.provider_query_limits,
            "dense_model": dense_model,
            "query_rewrite_model": (
                query_rewriter.model if query_rewriter is not None else None
            ),
            "semantic_scholar_key_configured": bool(api_key),
            "openalex_key_configured": bool(openalex_api_key),
            "qrel_used_for_candidate_generation": False,
        },
        "case_count": count,
        "target_paper_count": total_targets,
        "summary": {
            "evaluation_valid": evaluation_valid,
            "paper_recall_at_20": macro("paper_recall_at_20"),
            "paper_recall_at_50": macro("paper_recall_at_50"),
            "paper_recall_at_20_micro": sum(
                int(row["relevant_at_20_count"]) for row in rows
            ) / max(1, total_targets),
            "paper_recall_at_50_micro": sum(
                int(row["relevant_at_50_count"]) for row in rows
            ) / max(1, total_targets),
            "recommendation_link_coverage": sum(
                int(row["recommendation_link_count_at_10"]) for row in rows
            ) / max(1, recommendation_count),
            "short_description_coverage": sum(
                int(row["short_description_count_at_10"]) for row in rows
            ) / max(1, recommendation_count),
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
                0.0
                if total_returned == 0
                else sum(int(row["duplicate_count"]) for row in rows) / total_returned
            ),
            "unverifiable_paper_rate": (
                0.0
                if total_returned == 0
                else sum(int(row["unverifiable_paper_count"]) for row in rows)
                / total_returned
            ),
            "arxiv_target_resolution_rate": (
                sum(
                    float(row["arxiv_target_resolution_rate"])
                    * int(row["target_count"])
                    for row in rows
                )
                / max(1, total_targets)
            ),
            "provider_failure_case_success_rate": (
                1.0
                if not failure_cases
                else sum(int(row["paper_count"]) > 0 for row in failure_cases)
                / len(failure_cases)
            ),
            "provider_status": provider_status,
            "semantic_provider_success_rate": semantic_success_rate,
            "arxiv_provider_success_rate": arxiv_success_rate,
            "query_rewrite_success_rate": (
                sum(bool(row["query_rewrite_applied"]) for row in rows) / count
                if query_rewriter is not None
                else 0.0
            ),
            "query_rewrite_request_count": (
                query_rewriter.request_count if query_rewriter is not None else 0
            ),
            "query_rewrite_prompt_tokens": (
                query_rewriter.prompt_tokens if query_rewriter is not None else 0
            ),
            "query_rewrite_completion_tokens": (
                query_rewriter.completion_tokens if query_rewriter is not None else 0
            ),
        },
        "by_query_type": by_type,
        "rows": rows,
    }
    report["quality_gate"] = quality_gate(report["summary"])  # type: ignore[arg-type]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "literature-discovery-cases.json",
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable the default one-call DeepSeek scholarly query rewrite.",
    )
    parser.add_argument("--rewrite-model")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "literature-discovery-live.json",
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("semantic_scholar", "openalex", "arxiv", "crossref"),
    )
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--dense-model",
        default="BAAI/bge-small-en-v1.5",
        help="Set to an empty string to disable local dense reranking.",
    )
    args = parser.parse_args()
    if args.state_dir is None:
        with tempfile.TemporaryDirectory(prefix="taskforge-literature-eval-") as temporary:
            report = asyncio.run(
                evaluate(
                    args.cases,
                    Path(temporary),
                    concurrency=args.concurrency,
                    provider_names=set(args.providers) if args.providers else None,
                    case_limit=args.case_limit,
                    case_ids=set(args.case_ids) if args.case_ids else None,
                    dense_model=args.dense_model.strip() or None,
                    query_rewrite=not args.no_query_rewrite,
                    rewrite_model=args.rewrite_model,
                )
            )
    else:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        report = asyncio.run(
            evaluate(
                args.cases,
                args.state_dir,
                concurrency=args.concurrency,
                provider_names=set(args.providers) if args.providers else None,
                case_limit=args.case_limit,
                case_ids=set(args.case_ids) if args.case_ids else None,
                dense_model=args.dense_model.strip() or None,
                query_rewrite=not args.no_query_rewrite,
                rewrite_model=args.rewrite_model,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
