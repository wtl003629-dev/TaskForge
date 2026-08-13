"""Run locked QASPER questions through the direct PDF upload RAG path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import textwrap
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.rag_evaluation import (  # noqa: E402
    EvalCorpusDocument,
    load_qasper_dataset,
)


def _settings(
    state: Path,
    *,
    backend: str,
    graph_enabled: bool,
    reranker_model: str | None,
    reranker_backend: str = "fastembed",
    rewrite_enabled: bool,
    feature_ranker_path: Path | None = None,
    structure_fusion_enabled: bool = False,
    structure_section_weight: float = 0.5,
    structure_query_coverage_weight: float = 0.1,
    preserve_head_k: int = 0,
    reranker_context_window: int = 0,
    lexical_fusion_weight: float = 0.0,
    intent_section_fusion_enabled: bool = False,
    intent_section_fusion_weight: float = 0.1,
    intent_query_overlap_weight: float = 0.05,
    intent_rank_fusion_weight: float = 0.45,
) -> Settings:
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
        retrieval_routing="profile" if backend == "fastembed" else "lexical",
        general_text_backend=backend,
        semantic_cache_path=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-upload-embeddings.sqlite3",
        research_graph_enabled=graph_enabled,
        research_reranker_model=reranker_model,
        research_reranker_backend=reranker_backend,
        research_feature_ranker_path=feature_ranker_path,
        research_structure_fusion_enabled=structure_fusion_enabled,
        research_structure_section_weight=structure_section_weight,
        research_structure_query_coverage_weight=structure_query_coverage_weight,
        research_preserve_head_k=preserve_head_k,
        research_reranker_context_window=reranker_context_window,
        research_lexical_fusion_weight=lexical_fusion_weight,
        research_intent_section_fusion_enabled=intent_section_fusion_enabled,
        research_intent_section_fusion_weight=intent_section_fusion_weight,
        research_intent_query_overlap_weight=intent_query_overlap_weight,
        research_intent_rank_fusion_weight=intent_rank_fusion_weight,
        research_rewrite_enabled=rewrite_enabled,
        provider="demo",
    )


def _printable(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    replacements = {
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    normalized = "".join(replacements.get(character, character) for character in normalized)
    return normalized.encode("latin-1", "replace").decode("latin-1")


def _render_paper(
    target: Path,
    paper_title: str,
    documents: list[EvalCorpusDocument],
) -> dict[str, list[int]]:
    pdf = canvas.Canvas(str(target), pagesize=A4, pageCompression=1, invariant=1)
    width, height = A4
    page_number = 0
    mapping: dict[str, list[int]] = {}
    for document_index, document in enumerate(documents):
        section = str(
            document.metadata.get("section_title")
            or document.metadata.get("section")
            or "Document"
        )
        lines = textwrap.wrap(
            _printable(document.text),
            width=96,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        first_part = True
        while lines:
            page_number += 1
            mapping.setdefault(document.document_id, []).append(page_number)
            y = height - 48
            if document_index == 0 and first_part:
                pdf.setFont("Helvetica-Bold", 14)
                for title_line in textwrap.wrap(_printable(paper_title), width=72)[:3]:
                    pdf.drawString(48, y, title_line)
                    y -= 18
                y -= 8
            pdf.setFont("Helvetica-Bold", 11)
            suffix = " (continued)" if not first_part else ""
            pdf.drawString(48, y, _printable(section + suffix)[:110])
            y -= 22
            pdf.setFont("Helvetica", 9)
            max_lines = max(1, int((y - 42) // 12))
            current, lines = lines[:max_lines], lines[max_lines:]
            for line in current:
                pdf.drawString(48, y, line)
                y -= 12
            pdf.setFont("Helvetica", 7)
            pdf.drawRightString(width - 48, 24, f"page {page_number}")
            pdf.showPage()
            first_part = False
    pdf.save()
    return mapping


def _page_numbers(value: object) -> set[int]:
    if value is None:
        return set()
    return {
        int(part)
        for part in str(value).split(",")
        if part.strip().isdigit()
    }


def _recall_at_k(
    relevant_ids: list[str],
    retrieved_pages: list[set[int]],
    evidence_pages: dict[str, list[int]],
    k: int,
) -> float:
    pages = set().union(*retrieved_pages[:k]) if retrieved_pages[:k] else set()
    hits = sum(bool(pages & set(evidence_pages[item])) for item in relevant_ids)
    return hits / len(relevant_ids)


def _ndcg_at_10(
    relevant_ids: list[str],
    retrieved_pages: list[set[int]],
    evidence_pages: dict[str, list[int]],
) -> float:
    gains: list[int] = []
    seen: set[str] = set()
    for pages in retrieved_pages[:10]:
        matched = {
            item
            for item in relevant_ids
            if item not in seen and pages & set(evidence_pages[item])
        }
        gains.append(len(matched))
        seen.update(matched)
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant_ids))))
    return dcg / ideal if ideal else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _paper_id(document: EvalCorpusDocument) -> str:
    value = str(document.metadata.get("paper_id") or "").strip()
    if not value:
        raise ValueError("QASPER document is missing paper_id")
    return value


def run(
    dataset_path: Path,
    split_path: Path,
    output: Path,
    *,
    limit: int,
    offset: int = 0,
    backend: str = "bm25",
    graph_enabled: bool = True,
    reranker_model: str | None = None,
    reranker_backend: str = "fastembed",
    rewrite_enabled: bool = True,
    candidate_k: int = 50,
    feature_ranker_path: Path | None = None,
    structure_fusion_enabled: bool = False,
    structure_section_weight: float = 0.5,
    structure_query_coverage_weight: float = 0.1,
    preserve_head_k: int = 0,
    reranker_context_window: int = 0,
    lexical_fusion_weight: float = 0.0,
    intent_section_fusion_enabled: bool = False,
    intent_section_fusion_weight: float = 0.1,
    intent_query_overlap_weight: float = 0.05,
    intent_rank_fusion_weight: float = 0.45,
) -> dict[str, object]:
    if candidate_k < 10 or candidate_k > 100:
        raise ValueError("candidate_k must be between 10 and 100")
    dataset = load_qasper_dataset(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    locked_ids = [str(item) for item in split["case_ids"]][offset : offset + limit]
    case_by_id = {case.case_id: case for case in dataset.cases}
    missing = [case_id for case_id in locked_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"locked QASPER cases are missing: {missing[:3]}")
    cases = [case_by_id[case_id] for case_id in locked_ids]
    paper_ids = {str(case.metadata["paper_id"]) for case in cases}
    documents_by_paper: dict[str, list[EvalCorpusDocument]] = defaultdict(list)
    for document in dataset.documents:
        paper_id = _paper_id(document)
        if paper_id in paper_ids:
            documents_by_paper[paper_id].append(document)

    started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="taskforge-qasper-upload-eval-",
        ignore_cleanup_errors=True,
    ) as raw:
        state = Path(raw)
        pdf_root = state / "pdfs"
        pdf_root.mkdir()
        rendered: dict[str, dict[str, Any]] = {}
        for paper_id, documents in documents_by_paper.items():
            title = str(documents[0].metadata.get("paper_title") or paper_id)
            path = pdf_root / f"{paper_id}.pdf"
            evidence_pages = _render_paper(path, title, documents)
            rendered[paper_id] = {
                "path": path,
                "title": title,
                "evidence_pages": evidence_pages,
                "page_count": max(page for pages in evidence_pages.values() for page in pages),
                "evidence_units": len(documents),
            }

        app = create_app(
            _settings(
                state,
                backend=backend,
                graph_enabled=graph_enabled,
                reranker_model=reranker_model,
                reranker_backend=reranker_backend,
                rewrite_enabled=rewrite_enabled,
                feature_ranker_path=feature_ranker_path,
                structure_fusion_enabled=structure_fusion_enabled,
                structure_section_weight=structure_section_weight,
                structure_query_coverage_weight=structure_query_coverage_weight,
                preserve_head_k=preserve_head_k,
                reranker_context_window=reranker_context_window,
                lexical_fusion_weight=lexical_fusion_weight,
                intent_section_fusion_enabled=intent_section_fusion_enabled,
                intent_section_fusion_weight=intent_section_fusion_weight,
                intent_query_overlap_weight=intent_query_overlap_weight,
                intent_rank_fusion_weight=intent_rank_fusion_weight,
            )
        )
        auth = {
            "X-TaskForge-Tenant": "qasper-upload-eval",
            "X-TaskForge-User": "evaluator",
        }
        scopes: dict[str, dict[str, object]] = {}
        ingestion: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        with TestClient(app) as client:
            for paper_id, item in rendered.items():
                path = Path(item["path"])
                uploaded = client.post(
                    "/api/research/uploads",
                    headers={
                        **auth,
                        "Content-Type": "application/pdf",
                        "X-Filename": path.name,
                    },
                    params={
                        "conversation_id": f"qasper-{paper_id}",
                        "user_intent": "Answer questions from this uploaded paper.",
                        "title": str(item["title"]),
                    },
                    content=path.read_bytes(),
                )
                uploaded.raise_for_status()
                scope = uploaded.json()["scope"]
                indexed = client.post(
                    f"/api/research/scopes/{scope['scope_id']}/ingest",
                    headers=auth,
                )
                indexed.raise_for_status()
                status = indexed.json()[0]
                if status["status"] != "indexed":
                    raise RuntimeError(f"QASPER PDF ingestion failed: {status}")
                scopes[paper_id] = scope
                ingestion.append(
                    {
                        "paper_id": paper_id,
                        "pages": item["page_count"],
                        "gold_evidence_units": item["evidence_units"],
                        "indexed_chunks": status["evidence_count"],
                    }
                )

            for case in cases:
                paper_id = str(case.metadata["paper_id"])
                scope = scopes[paper_id]
                query_started = perf_counter()
                response = client.post(
                    "/api/research/evidence/search",
                    headers=auth,
                    json={
                        "scope_id": scope["scope_id"],
                        "scope_version": scope["scope_version"],
                        "query": case.query,
                        "intent": "general_fact",
                        "top_k": min(candidate_k, 50),
                        "candidate_k": candidate_k,
                        "mode": "rigorous",
                    },
                )
                latency_ms = (perf_counter() - query_started) * 1_000
                response.raise_for_status()
                result = response.json()
                retrieved_pages = [
                    _page_numbers(item.get("page")) for item in result["evidence"]
                ]
                evidence_pages = rendered[paper_id]["evidence_pages"]
                recalls = {
                    str(k): _recall_at_k(
                        case.relevant_ids,
                        retrieved_pages,
                        evidence_pages,
                        k,
                    )
                    for k in (1, 5, 10, 50)
                }
                rows.append(
                    {
                        "case_id": case.case_id,
                        "paper_id": paper_id,
                        "query": case.query,
                        "gold_evidence_units": len(case.relevant_ids),
                        "recall_at_k": recalls,
                        "ndcg_at_10": _ndcg_at_10(
                            case.relevant_ids,
                            retrieved_pages,
                            evidence_pages,
                        ),
                        "latency_ms": latency_ms,
                        "retrieved_count": len(retrieved_pages),
                        "retrieval_rounds": result["retrieval_rounds"],
                        "retrieved_evidence": [
                            {
                                "evidence_id": item.get("evidence_id"),
                                "page": item.get("page"),
                                "score": item.get("score"),
                                "text": str(item.get("snippet") or item.get("text") or "")[:1_000],
                                "metadata": item.get("metadata", {}),
                                "retrieval_sources": item.get("retrieval_sources", []),
                            }
                            for item in result["evidence"]
                        ],
                    }
                )

    metrics = {
        f"recall_at_{k}": statistics.fmean(
            float(row["recall_at_k"][str(k)])  # type: ignore[index]
            for row in rows
        )
        for k in (1, 5, 10, 50)
    }
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics.update(
        {
            "ndcg_at_10": statistics.fmean(float(row["ndcg_at_10"]) for row in rows),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
        }
    )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_type": "qasper_direct_pdf_upload_retrieval",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "QASPER v0.3 official dev locked split",
        "license": "CC BY 4.0",
        "source_dataset": str(dataset_path),
        "split": str(split_path),
        "papers": len(rendered),
        "cases": len(rows),
        "case_offset": offset,
        "pipeline": ["render_pdf", "direct_upload", "parse", "chunk", "index", "search"],
        "retrieval": {
            "backend": backend,
            "graph_enabled": graph_enabled,
            "reranker_backend": reranker_backend,
            "reranker_model": reranker_model,
            "rewrite_enabled": rewrite_enabled,
            "candidate_k": candidate_k,
        },
        "metrics": metrics,
        "passed": float(metrics["recall_at_10"]) >= 0.80,
        "thresholds": {"recall_at_10": 0.80},
        "ingestion": ingestion,
        "rows": rows,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            "QASPER text and labels are real; the PDF layout is generated locally.",
            f"The run uses the {backend} product retrieval path with graph_enabled={graph_enabled}.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "eval" / "splits" / "qasper-dev-general-100-v1.json",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--backend", choices=("bm25", "fastembed"), default="bm25")
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional local fastembed cross-encoder model.",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers"),
        default="fastembed",
        help="Cross-encoder adapter; ensemble model names are comma-separated.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph feature reranking for an ablation run.",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable the confidence-triggered second retrieval query.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of candidates retrieved and reranked (10-100).",
    )
    parser.add_argument(
        "--feature-ranker-path",
        type=Path,
        default=None,
        help="Optional serialized supervised feature-ranker report.",
    )
    parser.add_argument("--structure-fusion", action="store_true")
    parser.add_argument("--structure-section-weight", type=float, default=0.5)
    parser.add_argument("--structure-query-coverage-weight", type=float, default=0.1)
    parser.add_argument("--preserve-head-k", type=int, default=0)
    parser.add_argument("--reranker-context-window", type=int, default=0)
    parser.add_argument("--lexical-fusion-weight", type=float, default=0.0)
    parser.add_argument("--intent-section-fusion", action="store_true")
    parser.add_argument("--intent-section-fusion-weight", type=float, default=0.1)
    parser.add_argument("--intent-query-overlap-weight", type=float, default=0.05)
    parser.add_argument("--intent-rank-fusion-weight", type=float, default=0.45)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "qasper-direct-upload.json",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    split_case_count = len(json.loads(args.split.read_text(encoding="utf-8"))["case_ids"])
    if args.offset < 0 or args.offset + args.limit > split_case_count:
        raise SystemExit(
            f"--offset + --limit must select cases within the split ({split_case_count})"
        )
    report = run(
        args.dataset,
        args.split,
        args.output,
        limit=args.limit,
        offset=args.offset,
        backend=args.backend,
        graph_enabled=not args.no_graph,
        reranker_model=args.reranker_model,
        reranker_backend=args.reranker_backend,
        rewrite_enabled=not args.no_rewrite,
        candidate_k=args.candidate_k,
        feature_ranker_path=args.feature_ranker_path,
        structure_fusion_enabled=args.structure_fusion,
        structure_section_weight=args.structure_section_weight,
        structure_query_coverage_weight=args.structure_query_coverage_weight,
        preserve_head_k=args.preserve_head_k,
                reranker_context_window=args.reranker_context_window,
        lexical_fusion_weight=args.lexical_fusion_weight,
        intent_section_fusion_enabled=args.intent_section_fusion,
        intent_section_fusion_weight=args.intent_section_fusion_weight,
        intent_query_overlap_weight=args.intent_query_overlap_weight,
        intent_rank_fusion_weight=args.intent_rank_fusion_weight,
    )
    # Keep the console output ASCII-safe on Windows; the complete Unicode
    # report is already persisted at --output.
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
