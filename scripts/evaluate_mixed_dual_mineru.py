"""Compare Flat and Flat+Child dual retrieval on the frozen 30+30 MinerU set.

This is an isolated evaluation entry point.  It does not change the product's
default retrieval mode or mutate the legacy Flat index.  Both modes consume
the same real PDFs, MinerU output, gold alignment and provider configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import evaluate_mixed_optimized_e as source_eval  # noqa: E402
import evaluate_mixed_optimized_e_mineru as mineru_eval  # noqa: E402

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.pdf_parsing.hierarchy import (  # noqa: E402
    build_boundary_aware_flat_units,
    build_flat_units,
)
from taskforge.pdf_parsing.structure_policy import (
    build_structure_aware_units,  # noqa: E402
)
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402
from taskforge.rag_experiment_profile import (
    resolve_rag_experiment_profile,  # noqa: E402
)
from taskforge.research_reranking import build_research_reranker  # noqa: E402
from taskforge.research_retrieval import (  # noqa: E402
    ResearchQuery,
    ResearchRetrievalService,
)
from taskforge.semantic_providers import BailianDenseEmbedder  # noqa: E402

Mode = Literal["flat", "dual", "boundary"]
RetrievalScope = Literal["global", "paper"]
RECALL_KS = source_eval.RECALL_KS
TENANT_ID = "mixed-mineru-flat-dual-30x30"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _materialize_lanes(
    parsed: Any,
    *,
    mode: Mode,
    paper_key: str,
    paper_id: str,
    language: str,
    title: str,
    source_block_to_ids: dict[str, set[str]],
    chunks: list[KnowledgeChunk],
    source_to_chunks: dict[tuple[str, str], set[str]],
) -> dict[str, Any]:
    flat_units = (
        build_boundary_aware_flat_units(
            parsed,
            target_chars=2_000,
            min_chars=1_000,
            max_chars=2_600,
            search_chars=400,
        )
        if mode == "boundary"
        else build_flat_units(parsed, target_chars=2_000, overlap_chars=0)
    )
    batches: list[tuple[str, str, tuple[Any, ...], dict[str, Any] | None, str | None]] = [
        (
            "flat_primary" if mode == "dual" else "single",
            "flat",
            flat_units,
            None,
            None,
        )
    ]
    structured_result = None
    if mode == "dual":
        structured_result = build_structure_aware_units(
            parsed,
            parent_target_tokens=2_000,
            parent_max_tokens=3_000,
            child_target_tokens=400,
            child_max_tokens=500,
            child_overlap_tokens=60,
            fallback_target_chars=2_000,
            fallback_overlap_chars=0,
        )
        batches.append(
            (
                "child_aux",
                "structure_aware",
                structured_result.units,
                structured_result.profile.as_metadata(),
                structured_result.policy.name,
            )
        )

    chunk_ids = {
        (lane, unit.unit_id): (
            f"{parsed.document_id}:hybrid:{lane}:{unit.role}:{index:05d}"
        )
        for lane, _, units, _, _ in batches
        for index, unit in enumerate(units)
    }
    lane_counts: dict[str, int] = defaultdict(int)
    parent_count = 0
    evidence_count = 0
    for lane, chunking_mode, units, structure_profile, chunk_policy in batches:
        for unit in units:
            chunk_id = chunk_ids[(lane, unit.unit_id)]
            parent_id = chunk_ids.get((lane, unit.parent_id), chunk_id)
            heading = " > ".join(unit.heading_path)
            metadata: dict[str, Any] = {
                "rag_profile": "optimized",
                "rag_ablation": "e",
                "rag_profile_label": "optimized-e",
                # A stable per-paper scope lets the evaluation mirror the
                # product path where the user has selected one document or
                # knowledge base.  Global runs omit the filter and therefore
                # continue searching the complete 60-paper corpus.
                "knowledge_base_id": f"{TENANT_ID}:{paper_key}",
                "parser": parsed.parser,
                "parser_version": parsed.parser_version,
                "parser_backend": parsed.parser_backend,
                "parse_quality": parsed.quality.model_dump(mode="json"),
                "source_sha256": parsed.sha256,
                "paper_id": paper_id,
                "title": title,
                "language": language,
                "retrieval_role": unit.role,
                # Keep authoritative text unchanged in both vector lanes.  The
                # previous title/section projection remains an independent
                # experiment and is deliberately absent from this comparison.
                "retrieval_text": None,
                "retrieval_text_version": None,
                "chunking_mode": (
                    "boundary_aware_flat_v1"
                    if mode == "boundary"
                    else chunking_mode
                ),
                "hybrid_route": lane if mode == "dual" else None,
                "chunk_policy": chunk_policy,
                "structure_profile": structure_profile,
                "flat_chunk_chars": 2_000,
                "flat_overlap_chars": 0,
                "boundary_min_chars": 1_000 if mode == "boundary" else None,
                "boundary_max_chars": 2_600 if mode == "boundary" else None,
                "boundary_search_chars": 400 if mode == "boundary" else None,
                "parent_chunk_id": parent_id,
                "heading": heading or None,
                "heading_path": list(unit.heading_path),
                "pages": list(unit.pages),
                "chunk_index": unit.order,
                "block_ids": list(unit.block_ids),
                "block_types": list(unit.block_types),
            }
            if unit.role == "child":
                metadata["previous_chunk_id"] = (
                    chunk_ids.get((lane, unit.previous_unit_id))
                    if unit.previous_unit_id
                    else None
                )
                metadata["next_chunk_id"] = (
                    chunk_ids.get((lane, unit.next_unit_id))
                    if unit.next_unit_id
                    else None
                )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    tenant_id=TENANT_ID,
                    text=unit.text,
                    source_uri=parsed.source_uri,
                    document_id=parsed.document_id,
                    acl=frozenset({"user:mixed-eval"}),
                    metadata=metadata,
                )
            )
            lane_counts[lane] += 1
            parent_count += int(unit.role == "parent")
            evidence_count += int(unit.role != "parent")
            if unit.role == "parent":
                continue
            unit_block_ids = set(unit.block_ids)
            for source_id, aligned_block_ids in source_block_to_ids.items():
                if unit_block_ids.intersection(aligned_block_ids):
                    source_to_chunks.setdefault((paper_key, source_id), set()).add(
                        chunk_id
                    )

    return {
        "paper_key": paper_key,
        "paper_id": paper_id,
        "language": language,
        "title": title,
        "mode": mode,
        "flat_units": len(flat_units),
        "structured_policy": (
            structured_result.policy.name if structured_result is not None else None
        ),
        "structured_profile": (
            structured_result.profile.as_metadata()
            if structured_result is not None
            else None
        ),
        "parents": parent_count,
        "evidence_chunks": evidence_count,
        "lane_counts": dict(lane_counts),
    }


def run(
    *,
    mode: Mode,
    retrieval_scope: RetrievalScope,
    english_dataset: Path,
    english_split: Path,
    english_pdf_manifest: Path,
    chinese_dataset_dir: Path,
    chinese_papers: Path,
    output_path: Path,
    state_dir: Path,
    max_cases: int | None = None,
    balanced_smoke: bool = False,
) -> dict[str, Any]:
    if retrieval_scope not in {"global", "paper"}:
        raise ValueError("retrieval_scope must be global or paper")
    chinese_queries_path = chinese_dataset_dir / "queries.jsonl"
    chinese_qrels_path = chinese_dataset_dir / "qrels.jsonl"
    chinese_chunks_path = chinese_dataset_dir / "chunks.jsonl.gz"
    for path in (
        english_dataset,
        english_split,
        english_pdf_manifest,
        chinese_papers,
        chinese_queries_path,
        chinese_qrels_path,
        chinese_chunks_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    qasper = load_qasper_dataset(english_dataset)
    split = json.loads(english_split.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in qasper.cases}
    selected_cases = [case_by_id[str(value)] for value in split["case_ids"]]
    english_paper_ids = sorted(
        {str(case.metadata["paper_id"]) for case in selected_cases}
    )
    if len(english_paper_ids) != 30:
        raise ValueError(f"expected 30 English papers, got {len(english_paper_ids)}")
    english_documents: dict[str, list[Any]] = defaultdict(list)
    for document in qasper.documents:
        paper_id = str(document.metadata.get("paper_id") or "")
        if paper_id in english_paper_ids:
            english_documents[paper_id].append(document)

    chinese_chunk_rows = _read_jsonl_gz(chinese_chunks_path)
    chinese_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chinese_chunk_rows:
        chinese_by_paper[str(row["paper_id"])].append(row)
    chinese_paper_ids = sorted(chinese_by_paper)
    if len(chinese_paper_ids) != 30:
        raise ValueError(f"expected 30 Chinese papers, got {len(chinese_paper_ids)}")
    qrels_by_query: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(chinese_qrels_path):
        if int(row.get("relevance", 0)) > 0:
            qrels_by_query[str(row["query_id"])].add(str(row["document_id"]))
    chinese_query_rows = _read_jsonl(chinese_queries_path)
    if len(chinese_query_rows) != 90:
        raise ValueError(f"expected 90 Chinese queries, got {len(chinese_query_rows)}")

    english_manifest = mineru_eval._load_english_pdf_manifest(
        english_pdf_manifest,
        set(english_paper_ids),
    )
    chinese_manifest = mineru_eval._load_chinese_pdf_manifest(
        chinese_papers,
        set(chinese_paper_ids),
    )
    specs: list[dict[str, Any]] = []
    for paper_id in english_paper_ids:
        title = str(
            english_documents[paper_id][0].metadata.get("paper_title") or paper_id
        )
        specs.append(
            {
                "paper_key": f"en:{paper_id}",
                "paper_id": paper_id,
                "language": "en",
                "title": title,
                "path": english_manifest[paper_id]["path"],
            }
        )
    for paper_id in chinese_paper_ids:
        specs.append(
            {
                "paper_key": f"zh:{paper_id}",
                "paper_id": paper_id,
                "language": "zh",
                "title": chinese_manifest[paper_id]["title"],
                "path": chinese_manifest[paper_id]["path"],
            }
        )

    settings = Settings()
    parsed_by_key, mineru_elapsed_ms = asyncio.run(
        mineru_eval._parse_pdfs(specs, settings)
    )
    if len(parsed_by_key) != 60:
        raise RuntimeError(f"MinerU parsed {len(parsed_by_key)} of 60 PDFs")

    chunks: list[KnowledgeChunk] = []
    source_to_chunks: dict[tuple[str, str], set[str]] = {}
    paper_stats: list[dict[str, Any]] = []
    alignment_stats: dict[str, dict[str, Any]] = {}
    for spec in specs:
        paper_key = str(spec["paper_key"])
        parsed = parsed_by_key[paper_key]
        if spec["language"] == "en":
            sources = [
                (str(document.document_id), str(document.text))
                for document in english_documents[str(spec["paper_id"])]
            ]
        else:
            sources = [
                (str(row["chunk_id"]), str(row["text"]))
                for row in chinese_by_paper[str(spec["paper_id"])]
            ]
        source_map, counts = mineru_eval._alignment_map(parsed, sources)
        alignment_stats[paper_key] = counts
        stat = _materialize_lanes(
            parsed,
            mode=mode,
            paper_key=paper_key,
            paper_id=str(spec["paper_id"]),
            language=str(spec["language"]),
            title=str(spec["title"]),
            source_block_to_ids=source_map,
            chunks=chunks,
            source_to_chunks=source_to_chunks,
        )
        stat.update(
            {
                "pdf_path": str(Path(spec["path"]).resolve()),
                "pdf_sha256": parsed.sha256,
                "page_count": parsed.page_count,
                "block_count": len(parsed.blocks),
                "parse_quality": parsed.quality.model_dump(mode="json"),
                "alignment": counts,
            }
        )
        paper_stats.append(stat)

    cases = mineru_eval._build_cases(
        selected_cases=selected_cases,
        chinese_query_rows=chinese_query_rows,
        qrels_by_query=qrels_by_query,
        source_to_children=source_to_chunks,
    )
    if balanced_smoke:
        if max_cases is None or max_cases < 2 or max_cases % 2:
            raise ValueError("balanced_smoke requires an even positive max_cases")
        per_language = max_cases // 2
        english_smoke = [case for case in cases if case["language"] == "en"]
        chinese_smoke = [case for case in cases if case["language"] == "zh"]
        if len(english_smoke) < per_language or len(chinese_smoke) < per_language:
            raise ValueError("balanced_smoke does not have enough cases per language")
        cases = english_smoke[:per_language] + chinese_smoke[:per_language]
    elif max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("evaluation has no cases")

    if settings.bailian_api_key is None:
        raise RuntimeError("TASKFORGE_BAILIAN_API_KEY is required")
    state_dir.mkdir(parents=True, exist_ok=True)
    api_key = settings.bailian_api_key.get_secret_value()
    embedder = BailianDenseEmbedder(
        api_key=api_key,
        base_url=settings.bailian_base_url,
        model_name=settings.bailian_model,
        dimension=settings.bailian_embedding_dimension,
        batch_size=settings.bailian_batch_size,
        timeout_seconds=settings.bailian_timeout_seconds,
        max_retries=settings.bailian_max_retries,
        cache_path=state_dir / "embeddings.sqlite3",
        index_name=f"mixed-mineru-{mode}-30x30",
    )
    reranker = build_research_reranker(
        "bailian",
        settings.bailian_rerank_model,
        bailian_api_key=api_key,
        bailian_base_url=settings.bailian_rerank_base_url,
        bailian_timeout_seconds=settings.bailian_rerank_timeout_seconds,
        bailian_max_retries=settings.bailian_rerank_max_retries,
    )
    request_candidate_k = 100 if mode == "dual" else 50
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        dense_embedder=embedder,
        reranker=reranker,
        multilingual_dense_embedder=embedder,
        multilingual_reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        parent_aware_rerank_enabled=False,
        # Dual already performs cross-lane deduplication and a two-children
        # per-Parent cap.  Running the generic O(K^2) token-overlap diversity
        # pass again adds seconds of CPU work for the same constraint.
        lineage_diversity_enabled=False,
        dual_route_enabled=mode == "dual",
        dual_route_flat_candidate_k=60,
        dual_route_child_candidate_k=40,
        dual_route_flat_head_k=2,
        dual_route_rerank_candidate_k=50,
        dual_route_tail_rerank_candidate_k=0,
        experiment_profile=resolve_rag_experiment_profile("optimized", "e"),
    )
    principal = AccessContext(tenant_id=TENANT_ID, user_id="mixed-eval")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    started = perf_counter()
    service.search(
        ResearchQuery(
            query=cases[0]["query"],
            top_k=50,
            candidate_k=request_candidate_k,
            knowledge_base_ids=(
                (f"{TENANT_ID}:{cases[0]['paper_key']}",)
                if retrieval_scope == "paper"
                else ()
            ),
        ),
        principal,
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        query_started = perf_counter()
        result = service.search(
            ResearchQuery(
                query=case["query"],
                top_k=50,
                candidate_k=request_candidate_k,
                knowledge_base_ids=(
                    (f"{TENANT_ID}:{case['paper_key']}",)
                    if retrieval_scope == "paper"
                    else ()
                ),
            ),
            principal,
        )
        latency_ms = (perf_counter() - query_started) * 1_000
        retrieved_ids = [item.chunk_id for item in result.evidence]
        if case["language"] == "en":
            recall = {
                str(k): source_eval._recall_for_sets(
                    retrieved_ids,
                    case["evidence_sets"],
                    k,
                )[0]
                for k in RECALL_KS
            }
        else:
            relevant_sources = set(case["relevant_source_ids"])
            recall = {
                str(k): (
                    len(
                        {
                            source_id
                            for source_id in relevant_sources
                            if set(retrieved_ids[:k]).intersection(
                                source_to_chunks.get(
                                    (case["paper_key"], source_id), set()
                                )
                            )
                        }
                    )
                    / len(relevant_sources)
                    if relevant_sources
                    else 0.0
                )
                for k in RECALL_KS
            }
        relevant_chunks = set(case["relevant_child_ids"])
        retrieved_lanes = [
            str(chunk_by_id[chunk_id].metadata.get("hybrid_route") or "single")
            for chunk_id in retrieved_ids
        ]
        evidence_sources = [
            list(item.retrieval_sources) for item in result.evidence
        ]
        rows.append(
            {
                "case_id": case["case_id"],
                "query_id": case["query_id"],
                "paper_id": case["paper_id"],
                "language": case["language"],
                "question_type": case["question_type"],
                "query": case["query"],
                "relevant_source_ids": case["relevant_source_ids"],
                "relevant_chunk_count": len(relevant_chunks),
                "retrieved_ids": retrieved_ids,
                "retrieved_lanes": retrieved_lanes,
                "retrieved_paper_ids": [
                    f"{chunk_by_id[chunk_id].metadata.get('language')}:"
                    f"{chunk_by_id[chunk_id].metadata.get('paper_id')}"
                    for chunk_id in retrieved_ids
                ],
                "retrieval_sources": evidence_sources,
                "recall_at_k": recall,
                "mrr_at_k": {
                    str(k): source_eval._mrr(retrieved_ids, relevant_chunks, k)
                    for k in RECALL_KS
                },
                "ndcg_at_k": {
                    str(k): source_eval._ndcg(retrieved_ids, relevant_chunks, k)
                    for k in RECALL_KS
                },
                "candidate_count": result.candidate_count,
                "retrieval_route": result.retrieval_route,
                "retrieval_scope": retrieval_scope,
                "latency_ms": latency_ms,
                "alignment_paper": alignment_stats[case["paper_key"]],
            }
        )
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(
                f"[retrieval] {index}/{len(cases)} mode={mode} "
                f"r10={recall['10']:.3f} latency_ms={latency_ms:.1f}",
                flush=True,
            )
    try:
        embedder.close()
    finally:
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    parse_status_counts: dict[str, int] = defaultdict(int)
    for item in paper_stats:
        parse_status_counts[str(item["parse_quality"]["status"])] += 1
    metrics_by_language = {
        language: source_eval._aggregate(
            [row for row in rows if row["language"] == language]
        )
        for language in ("en", "zh")
        if any(row["language"] == language for row in rows)
    }
    metrics_by_question_type = {
        question_type: source_eval._aggregate(
            [row for row in rows if row["question_type"] == question_type]
        )
        for question_type in sorted({str(row["question_type"]) for row in rows})
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluation_type": "mixed_annotated_mineru_flat_dual_retrieval",
        "mode": mode,
        "retrieval_scope": retrieval_scope,
        "created_at": datetime.now(UTC).isoformat(),
        "selection": {
            "english_papers": len(english_paper_ids),
            "chinese_papers": len(chinese_paper_ids),
            "total_papers": len(english_paper_ids) + len(chinese_paper_ids),
            "english_cases": sum(row["language"] == "en" for row in rows),
            "chinese_cases": sum(row["language"] == "zh" for row in rows),
            "total_cases": len(rows),
            "balanced_smoke": balanced_smoke,
            "english_split": str(english_split),
            "english_split_sha256": mineru_eval._sha256(english_split),
            "english_pdf_manifest": str(english_pdf_manifest),
            "english_pdf_manifest_sha256": mineru_eval._sha256(
                english_pdf_manifest
            ),
            "chinese_dataset": str(chinese_dataset_dir),
            "chinese_papers_manifest": str(chinese_papers),
            "chinese_papers_manifest_sha256": mineru_eval._sha256(chinese_papers),
            "chinese_queries_sha256": mineru_eval._sha256(chinese_queries_path),
            "chinese_qrels_sha256": mineru_eval._sha256(chinese_qrels_path),
            "chinese_chunks_sha256": mineru_eval._sha256(chinese_chunks_path),
        },
        "pipeline": [
            "real PDFs",
            f"MinerU {settings.mineru_expected_version}",
            (
                "Flat 2000 primary + structure-aware Child auxiliary"
                if mode == "dual"
                else (
                    "Flat 2000 with structure-safe boundary correction"
                    if mode == "boundary"
                    else "Flat 2000 control"
                )
            ),
            "BM25 + Bailian text-embedding-v4",
            "RRF",
            "single Bailian qwen3-rerank pass",
            "Parent context only",
        ],
        "parser": {
            "backend": settings.mineru_backend,
            "parse_method": settings.mineru_parse_method,
            "effort": settings.mineru_effort,
            "expected_version": settings.mineru_expected_version,
            "base_url": settings.mineru_base_url,
            "elapsed_ms": mineru_elapsed_ms,
            "status_counts": dict(sorted(parse_status_counts.items())),
        },
        "retrieval": {
            "backend": "bailian",
            "semantic_model": settings.bailian_model,
            "reranker_backend": "bailian",
            "reranker_model": settings.bailian_rerank_model,
            "top_k": 50,
            "candidate_k": request_candidate_k,
            "dual_route_enabled": mode == "dual",
            "dual_route_flat_candidate_k": 60 if mode == "dual" else None,
            "dual_route_child_candidate_k": 40 if mode == "dual" else None,
            "dual_route_rerank_candidate_k": 50 if mode == "dual" else None,
            "dual_route_flat_head_k": 2 if mode == "dual" else None,
            "parent_aware_rerank_enabled": False,
            "lineage_diversity_enabled": False,
            "dual_route_cross_lane_dedupe_enabled": mode == "dual",
            "retrieval_text_enabled": False,
            "boundary_aware_enabled": mode == "boundary",
            "boundary_target_chars": 2_000 if mode == "boundary" else None,
            "boundary_min_chars": 1_000 if mode == "boundary" else None,
            "boundary_max_chars": 2_600 if mode == "boundary" else None,
            "boundary_search_chars": 400 if mode == "boundary" else None,
            "graph_enabled": False,
            "mixed_corpus_scope": True,
            "scope_filter": (
                "knowledge_base_id_per_paper"
                if retrieval_scope == "paper"
                else None
            ),
        },
        "index_statistics": {
            "papers": len(paper_stats),
            "chunks_total_in_store": len(chunks),
            "evidence_chunks": sum(
                int(item["evidence_chunks"]) for item in paper_stats
            ),
            "parents": sum(int(item["parents"]) for item in paper_stats),
            "flat_chunks": sum(int(item["flat_units"]) for item in paper_stats),
            "child_aux_chunks": sum(
                int(item["lane_counts"].get("child_aux", 0))
                for item in paper_stats
            ),
            "mineru_blocks": sum(int(item["block_count"]) for item in paper_stats),
        },
        "alignment": {
            "papers": alignment_stats,
            "total_units": sum(
                item["total_units"] for item in alignment_stats.values()
            ),
            "exact_units": sum(
                item["exact_units"] for item in alignment_stats.values()
            ),
            "fuzzy_units": sum(
                item["fuzzy_units"] for item in alignment_stats.values()
            ),
            "ambiguous_units": sum(
                item["ambiguous_units"] for item in alignment_stats.values()
            ),
            "unaligned_units": sum(
                item["unaligned_units"] for item in alignment_stats.values()
            ),
        },
        "metrics": source_eval._aggregate(rows),
        "metrics_by_language": metrics_by_language,
        "metrics_by_question_type": metrics_by_question_type,
        "rows": rows,
        "paper_stats": paper_stats,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "mineru_elapsed_ms": mineru_elapsed_ms,
        "limitations": [
            "All PDFs and gold alignment are identical across Flat and Dual runs.",
            (
                "Paper-scoped queries assume the caller has selected the target paper; this is not a global document-discovery score."
                if retrieval_scope == "paper"
                else "Global queries search all 60 papers without a target-paper hint."
            ),
            "Chinese relevance labels are silver-curated and not human-final-reviewed.",
            "The benchmark searches one global mixed corpus; no paper-id oracle is used.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("flat", "dual", "boundary"),
        required=True,
        help="flat is the control; boundary is the isolated boundary-aware candidate",
    )
    parser.add_argument(
        "--scope",
        choices=("global", "paper"),
        default="global",
        help=(
            "Search all 60 papers, or restrict each query to its selected paper. "
            "The default preserves the original global benchmark."
        ),
    )
    parser.add_argument(
        "--english-dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--english-split",
        type=Path,
        default=PROJECT_ROOT
        / "eval"
        / "splits"
        / "qasper-dev-random-papers-30-v1.json",
    )
    parser.add_argument(
        "--english-pdf-manifest",
        type=Path,
        default=PROJECT_ROOT
        / ".taskforge"
        / "eval-cache"
        / "qasper-dev-random-papers-30-real-pdfs-v1.json",
    )
    parser.add_argument(
        "--chinese-dataset-dir",
        type=Path,
        default=PROJECT_ROOT
        / "eval"
        / "queries"
        / "chinese-paper-rag-30-v2-precision",
    )
    parser.add_argument(
        "--chinese-papers",
        type=Path,
        default=PROJECT_ROOT
        / ".taskforge"
        / "datasets"
        / "chinese-ai-oa-jos-v2"
        / "papers.jsonl.gz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / ".mixed-optimized-e-30x30-mineru-state-v2",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--balanced-smoke",
        action="store_true",
        help="with --max-cases, select an equal English/Chinese case sample",
    )
    args = parser.parse_args()
    report = run(
        mode=args.mode,
        retrieval_scope=args.scope,
        english_dataset=args.english_dataset,
        english_split=args.english_split,
        english_pdf_manifest=args.english_pdf_manifest,
        chinese_dataset_dir=args.chinese_dataset_dir,
        chinese_papers=args.chinese_papers,
        output_path=args.output,
        state_dir=args.state_dir,
        max_cases=args.max_cases,
        balanced_smoke=args.balanced_smoke,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": args.mode,
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
