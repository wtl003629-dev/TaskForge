"""Evaluate the optimized-e route on the annotated 30+30 paper corpus.

The English side is the fixed random 30-paper QASPER split and the Chinese
side is the curated 30-paper chunk-QA set.  The two languages are indexed in
one unrestricted corpus, so each query must rank evidence among all 60
papers.  Source paragraphs/chunks are converted into parser-neutral blocks
in order to exercise the same deterministic structure-aware policy even when
the selected 30+30 real-PDF manifest is not available.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.pdf_parsing.contracts import (  # noqa: E402
    DocumentBlock,
    ParsedDocument,
    ParseQualityReport,
)
from taskforge.pdf_parsing.structure_policy import (  # noqa: E402
    build_structure_aware_units,
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

RECALL_KS = (1, 5, 10, 20, 50)
TENANT_ID = "mixed-optimized-e-30x30"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _block(
    *,
    block_id: str,
    document_id: str,
    text: str,
    page: int,
    order: int,
    block_type: str = "paragraph",
    heading_level: int | None = None,
) -> DocumentBlock:
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError(f"empty synthetic block: {block_id}")
    return DocumentBlock(
        block_id=block_id,
        document_id=document_id,
        parser="source-record-blocks",
        parser_version="1",
        page=max(1, page),
        bbox=(0.0, 0.0, 612.0, 792.0),
        reading_order=order,
        block_type=block_type,  # type: ignore[arg-type]
        text=cleaned,
        content_hash=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        heading_level=heading_level,
    )


def _parsed_document(
    paper_key: str,
    source_uri: str,
    blocks: list[DocumentBlock],
) -> ParsedDocument:
    if not blocks:
        raise ValueError(f"paper has no blocks: {paper_key}")
    page_count = max(block.page for block in blocks)
    text = "\n".join(block.text for block in blocks)
    return ParsedDocument(
        document_id=paper_key,
        source_uri=source_uri,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        bytes_read=max(1, len(text.encode("utf-8"))),
        page_count=page_count,
        parser="source-record-blocks",
        parser_version="1",
        parser_backend="annotated-source-blocks",
        blocks=tuple(blocks),
        quality=ParseQualityReport(
            page_count=page_count,
            parsed_page_count=page_count,
            text_coverage=1.0,
            garbled_character_ratio=0.0,
            repeated_header_ratio=0.0,
            orphan_caption_count=0,
            empty_table_count=0,
            reading_order_warning_count=0,
            visual_unparsed_count=0,
            ocr_used=False,
            status="ready",
            recommended_parser="none",
        ),
    )


def _build_english_blocks(
    paper_id: str,
    title: str,
    documents: list[Any],
) -> tuple[ParsedDocument, dict[str, str]]:
    paper_key = f"en:{paper_id}"
    blocks: list[DocumentBlock] = []
    source_block_to_id: dict[str, str] = {}
    order = 0
    blocks.append(
        _block(
            block_id=f"{paper_key}:title",
            document_id=paper_key,
            text=title or paper_id,
            page=1,
            order=order,
            block_type="title",
            heading_level=1,
        )
    )
    order += 1
    last_section: str | None = None
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        section = str(
            metadata.get("section_title")
            or metadata.get("section")
            or ""
        ).strip()
        if section and section != last_section:
            blocks.append(
                _block(
                    block_id=f"{paper_key}:section:{index:05d}",
                    document_id=paper_key,
                    text=section,
                    page=1 + index // 8,
                    order=order,
                    block_type="title",
                    heading_level=2,
                )
            )
            order += 1
            last_section = section
        block_id = f"{paper_key}:source:{document.document_id}"
        blocks.append(
            _block(
                block_id=block_id,
                document_id=paper_key,
                text=document.text,
                page=1 + index // 8,
                order=order,
            )
        )
        source_block_to_id[document.document_id] = block_id
        order += 1
    return (
        _parsed_document(
            paper_key,
            f"mixed-optimized-e://{paper_key}",
            blocks,
        ),
        source_block_to_id,
    )


def _build_chinese_blocks(
    paper_id: str,
    title: str,
    rows: list[dict[str, Any]],
) -> tuple[ParsedDocument, dict[str, str]]:
    paper_key = f"zh:{paper_id}"
    blocks: list[DocumentBlock] = []
    source_block_to_id: dict[str, str] = {}
    blocks.append(
        _block(
            block_id=f"{paper_key}:title",
            document_id=paper_key,
            text=title or paper_id,
            page=1,
            order=0,
            block_type="title",
            heading_level=1,
        )
    )
    for order, row in enumerate(
        sorted(rows, key=lambda item: int(item.get("chunk_index", 0))),
        start=1,
    ):
        source_id = str(row["chunk_id"])
        block_id = f"{paper_key}:source:{source_id}"
        blocks.append(
            _block(
                block_id=block_id,
                document_id=paper_key,
                text=str(row["text"]),
                page=1 + (order - 1) // 8,
                order=order,
            )
        )
        source_block_to_id[source_id] = block_id
    return (
        _parsed_document(
            paper_key,
            f"mixed-optimized-e://{paper_key}",
            blocks,
        ),
        source_block_to_id,
    )


def _materialize_paper(
    parsed: ParsedDocument,
    *,
    paper_id: str,
    language: str,
    title: str,
    source_block_to_id: dict[str, str | set[str] | list[str]],
    chunks: list[KnowledgeChunk],
    source_to_children: dict[tuple[str, str], set[str]],
    tenant_id: str = TENANT_ID,
) -> dict[str, Any]:
    result = build_structure_aware_units(
        parsed,
        parent_target_tokens=2_000,
        parent_max_tokens=3_000,
        child_target_tokens=400,
        child_max_tokens=500,
        child_overlap_tokens=60,
        fallback_target_chars=2_000,
        fallback_overlap_chars=0,
    )
    units = list(result.units)
    unit_chunk_ids = {
        unit.unit_id: f"{parsed.document_id}:{unit.role}:{index:05d}"
        for index, unit in enumerate(units)
    }
    child_units = [unit for unit in units if unit.role == "child"]
    for unit in units:
        chunk_id = unit_chunk_ids[unit.unit_id]
        heading = " > ".join(unit.heading_path)
        retrieval_text = None
        if unit.role == "child":
            retrieval_parts = [f"Title: {title or paper_id}"]
            if heading:
                retrieval_parts.append(f"Section: {heading}")
            retrieval_parts.append(unit.text)
            retrieval_text = "\n\n".join(retrieval_parts)
        metadata: dict[str, Any] = {
            "rag_profile": "optimized",
            "rag_ablation": "e",
            "rag_profile_label": "optimized-e",
            "knowledge_base_id": "mixed-optimized-e-30x30",
            "parser": parsed.parser,
            "parser_version": parsed.parser_version,
            "parser_backend": parsed.parser_backend,
            "parse_quality": parsed.quality.model_dump(mode="json"),
            "source_sha256": parsed.sha256,
            "paper_id": paper_id,
            "title": title,
            "language": language,
            "retrieval_role": unit.role,
            "retrieval_text": retrieval_text,
            "retrieval_text_version": (
                "mixed-optimized-e-title-section-v1"
                if retrieval_text is not None
                else None
            ),
            "chunking_mode": "structure_aware",
            "chunk_policy": result.policy.name,
            "structure_profile": result.profile.as_metadata(),
            "parent_chunk_id": (
                unit_chunk_ids[unit.parent_id]
                if unit.role == "child"
                else None
            ),
            "heading": heading or None,
            "heading_path": list(unit.heading_path),
            "chunk_index": unit.order,
            "block_ids": list(unit.block_ids),
            "block_types": list(unit.block_types),
        }
        if unit.role == "child":
            metadata["previous_chunk_id"] = (
                unit_chunk_ids.get(unit.previous_unit_id)
                if unit.previous_unit_id
                else None
            )
            metadata["next_chunk_id"] = (
                unit_chunk_ids.get(unit.next_unit_id)
                if unit.next_unit_id
                else None
            )
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                text=unit.text,
                source_uri=parsed.source_uri,
                document_id=parsed.document_id,
                acl=frozenset({"user:mixed-eval"}),
                metadata=metadata,
            )
        )
        if unit.role == "child":
            for source_id, source_block_value in source_block_to_id.items():
                if isinstance(source_block_value, str):
                    source_block_ids = {source_block_value}
                else:
                    source_block_ids = set(source_block_value)
                if source_block_ids.intersection(unit.block_ids):
                    source_to_children.setdefault(
                        (parsed.document_id, source_id), set()
                    ).add(chunk_id)
    return {
        "paper_key": parsed.document_id,
        "paper_id": paper_id,
        "language": language,
        "title": title,
        "policy": result.policy.name,
        "profile": result.profile.as_metadata(),
        "parents": sum(unit.role == "parent" for unit in units),
        "children": len(child_units),
    }


def _recall_for_sets(
    retrieved: list[str],
    evidence_sets: list[tuple[str, list[set[str]]]],
    k: int,
) -> tuple[float, str]:
    head = set(retrieved[:k])
    scored: list[tuple[float, int, str]] = []
    for annotation_id, units in evidence_sets:
        score = sum(bool(head.intersection(unit)) for unit in units) / len(units)
        scored.append((score, len(units), annotation_id))
    score, _, annotation_id = max(
        scored,
        key=lambda item: (item[0], -item[1], item[2]),
    )
    return score, annotation_id


def _mrr(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    values: dict[str, float] = {}
    metric_prefixes = {
        "recall_at_k": "recall_at",
        "mrr_at_k": "mrr_at",
        "ndcg_at_k": "ndcg_at",
    }
    for metric, prefix in metric_prefixes.items():
        for k in RECALL_KS:
            values[f"{prefix}_{k}"] = statistics.fmean(
                float(row[metric][str(k)]) for row in rows
            )
    latencies = [float(row["latency_ms"]) for row in rows]
    values["p50_ms"] = _percentile(latencies, 0.50)
    values["p95_ms"] = _percentile(latencies, 0.95)
    return values


def run(
    *,
    english_dataset: Path,
    english_split: Path,
    chinese_dataset_dir: Path,
    output_path: Path,
    state_dir: Path,
    candidate_k: int = 50,
) -> dict[str, Any]:
    if candidate_k < 10 or candidate_k > 100:
        raise ValueError("candidate_k must be between 10 and 100")
    chinese_queries_path = chinese_dataset_dir / "queries.jsonl"
    chinese_qrels_path = chinese_dataset_dir / "qrels.jsonl"
    chinese_chunks_path = chinese_dataset_dir / "chunks.jsonl.gz"
    for path in (
        english_dataset,
        english_split,
        chinese_queries_path,
        chinese_qrels_path,
        chinese_chunks_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    qasper = load_qasper_dataset(english_dataset)
    split = json.loads(english_split.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in qasper.cases}
    selected_case_ids = [str(value) for value in split["case_ids"]]
    selected_cases = [case_by_id[value] for value in selected_case_ids]
    english_paper_ids = sorted({str(case.metadata["paper_id"]) for case in selected_cases})
    if len(english_paper_ids) != 30:
        raise ValueError(f"expected 30 English papers, got {len(english_paper_ids)}")
    english_documents: dict[str, list[Any]] = defaultdict(list)
    for document in qasper.documents:
        paper_id = str(document.metadata.get("paper_id") or "")
        if paper_id in english_paper_ids:
            english_documents[paper_id].append(document)

    chinese_chunk_rows = _read_jsonl_gz(chinese_chunks_path)
    chinese_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    titles: dict[str, str] = {}
    for row in chinese_chunk_rows:
        paper_id = str(row["paper_id"])
        chinese_by_paper[paper_id].append(row)
        titles[paper_id] = str(row.get("title") or paper_id)
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

    chunks: list[KnowledgeChunk] = []
    source_to_children: dict[tuple[str, str], set[str]] = {}
    paper_stats: list[dict[str, Any]] = []
    for paper_id in english_paper_ids:
        documents = english_documents[paper_id]
        title = str(documents[0].metadata.get("paper_title") or paper_id)
        parsed, source_map = _build_english_blocks(paper_id, title, documents)
        paper_stats.append(
            _materialize_paper(
                parsed,
                paper_id=paper_id,
                language="en",
                title=title,
                source_block_to_id=source_map,
                chunks=chunks,
                source_to_children=source_to_children,
            )
        )
    for paper_id in chinese_paper_ids:
        rows = chinese_by_paper[paper_id]
        parsed, source_map = _build_chinese_blocks(paper_id, titles[paper_id], rows)
        paper_stats.append(
            _materialize_paper(
                parsed,
                paper_id=paper_id,
                language="zh",
                title=titles[paper_id],
                source_block_to_id=source_map,
                chunks=chunks,
                source_to_children=source_to_children,
            )
        )

    settings = Settings()
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
        index_name="mixed-optimized-e-30x30",
    )
    reranker = build_research_reranker(
        "bailian",
        settings.bailian_rerank_model,
        bailian_api_key=api_key,
        bailian_base_url=settings.bailian_rerank_base_url,
        bailian_timeout_seconds=settings.bailian_rerank_timeout_seconds,
        bailian_max_retries=settings.bailian_rerank_max_retries,
    )
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        dense_embedder=embedder,
        reranker=reranker,
        multilingual_dense_embedder=embedder,
        multilingual_reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        experiment_profile=resolve_rag_experiment_profile("optimized", "e"),
    )
    principal = AccessContext(tenant_id=TENANT_ID, user_id="mixed-eval")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    cases: list[dict[str, Any]] = []
    for case in selected_cases:
        paper_id = str(case.metadata["paper_id"])
        paper_key = f"en:{paper_id}"
        if case.qasper_gold is None:
            raise ValueError(f"English case lacks QASPER gold: {case.case_id}")
        evidence_sets: list[tuple[str, list[set[str]]]] = []
        english_source_ids: set[str] = set()
        for evidence_set in case.qasper_gold.evidence_sets:
            mapped_units: list[set[str]] = []
            for unit in evidence_set.units:
                mapped: set[str] = set()
                for source_id in unit.alternative_paragraph_ids:
                    english_source_ids.add(source_id)
                    mapped.update(source_to_children.get((paper_key, source_id), set()))
                if not mapped:
                    raise ValueError(
                        f"QASPER gold paragraph did not map to a Child: {case.case_id}"
                    )
                mapped_units.append(mapped)
            evidence_sets.append((evidence_set.annotation_id, mapped_units))
        cases.append(
            {
                "case_id": case.case_id,
                "query_id": case.case_id,
                "paper_id": paper_id,
                "paper_key": paper_key,
                "language": "en",
                "question_type": case.category,
                "query": case.query,
                "evidence_sets": evidence_sets,
                "relevant_child_ids": set().union(
                    *(unit for _, units in evidence_sets for unit in units)
                ),
                "relevant_source_ids": sorted(
                    english_source_ids
                ),
            }
        )
    for query in chinese_query_rows:
        query_id = str(query["query_id"])
        paper_id = str(query["paper_id"])
        paper_key = f"zh:{paper_id}"
        relevant_sources = qrels_by_query.get(query_id, set())
        relevant_children = set().union(
            *(source_to_children.get((paper_key, source_id), set()) for source_id in relevant_sources)
        )
        if not relevant_children:
            raise ValueError(f"Chinese qrel did not map to a Child: {query_id}")
        cases.append(
            {
                "case_id": query_id,
                "query_id": query_id,
                "paper_id": paper_id,
                "paper_key": paper_key,
                "language": "zh",
                "question_type": str(query.get("question_type") or "unknown"),
                "query": str(query["query"]),
                "evidence_sets": [],
                "relevant_child_ids": relevant_children,
                "relevant_source_ids": sorted(relevant_sources),
            }
        )

    started = perf_counter()
    # Index creation is deliberately outside measured query latency.
    service.search(
        ResearchQuery(query=cases[0]["query"], top_k=candidate_k, candidate_k=candidate_k),
        principal,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        query_started = perf_counter()
        result = service.search(
            ResearchQuery(
                query=case["query"],
                top_k=candidate_k,
                candidate_k=candidate_k,
            ),
            principal,
        )
        latency_ms = (perf_counter() - query_started) * 1_000
        retrieved_ids = [item.chunk_id for item in result.evidence]
        if case["language"] == "en":
            recall = {
                str(k): _recall_for_sets(retrieved_ids, case["evidence_sets"], k)[0]
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
                                source_to_children.get((case["paper_key"], source_id), set())
                            )
                        }
                    )
                    / len(relevant_sources)
                    if relevant_sources
                    else 0.0
                )
                for k in RECALL_KS
            }
        relevant_children = set(case["relevant_child_ids"])
        rows.append(
            {
                "case_id": case["case_id"],
                "query_id": case["query_id"],
                "paper_id": case["paper_id"],
                "language": case["language"],
                "question_type": case["question_type"],
                "query": case["query"],
                "relevant_source_ids": case["relevant_source_ids"],
                "retrieved_ids": retrieved_ids,
                "retrieved_paper_ids": [
                    f"{chunk_by_id[item.chunk_id].metadata.get('language')}:{chunk_by_id[item.chunk_id].metadata.get('paper_id')}"
                    for item in result.evidence
                ],
                "recall_at_k": recall,
                "mrr_at_k": {
                    str(k): _mrr(retrieved_ids, relevant_children, k)
                    for k in RECALL_KS
                },
                "ndcg_at_k": {
                    str(k): _ndcg(retrieved_ids, relevant_children, k)
                    for k in RECALL_KS
                },
                "candidate_count": result.candidate_count,
                "retrieval_route": result.retrieval_route,
                "latency_ms": latency_ms,
            }
        )
    try:
        embedder.close()
    finally:
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    by_language = {
        language: _aggregate([row for row in rows if row["language"] == language])
        for language in ("en", "zh")
    }
    by_type = {
        question_type: _aggregate(
            [row for row in rows if row["question_type"] == question_type]
        )
        for question_type in sorted({str(row["question_type"]) for row in rows})
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluation_type": "mixed_annotated_optimized_e_retrieval",
        "benchmark_track": "mixed_30_chinese_30_english_global_chunk_retrieval",
        "created_at": datetime.now(UTC).isoformat(),
        "selection": {
            "english_papers": len(english_paper_ids),
            "chinese_papers": len(chinese_paper_ids),
            "total_papers": len(english_paper_ids) + len(chinese_paper_ids),
            "english_cases": sum(row["language"] == "en" for row in rows),
            "chinese_cases": sum(row["language"] == "zh" for row in rows),
            "english_split": str(english_split),
            "english_split_sha256": _sha256(english_split),
            "chinese_dataset": str(chinese_dataset_dir),
            "chinese_queries_sha256": _sha256(chinese_queries_path),
            "chinese_qrels_sha256": _sha256(chinese_qrels_path),
            "chinese_chunks_sha256": _sha256(chinese_chunks_path),
        },
        "pipeline": [
            "source-record-blocks",
            "optimized-e-structure-aware-chunking",
            "BM25",
            "Bailian text-embedding-v4",
            "RRF",
            "Bailian qwen3-rerank",
            "Parent-aware rerank",
            "lineage diversity",
        ],
        "rag_profile": {"name": "optimized", "ablation": "e"},
        "retrieval": {
            "backend": "bailian",
            "semantic_model": settings.bailian_model,
            "reranker_backend": "bailian",
            "reranker_model": settings.bailian_rerank_model,
            "candidate_k": candidate_k,
            "agent_visible_k": candidate_k,
            "graph_enabled": False,
            "query_profile": "original",
            "mixed_corpus_scope": True,
            "retrieval_route_counts": {
                route: sum(row["retrieval_route"] == route for row in rows)
                for route in sorted({row["retrieval_route"] for row in rows})
            },
        },
        "index_statistics": {
            "papers": len(paper_stats),
            "children": sum(int(item["children"]) for item in paper_stats),
            "parents": sum(int(item["parents"]) for item in paper_stats),
            "structured_papers": sum(
                item["policy"] == "structured_parent_child_v1"
                for item in paper_stats
            ),
            "chunks_total_in_store": len(chunks),
        },
        "metrics": _aggregate(rows),
        "metrics_by_language": by_language,
        "metrics_by_question_type": by_type,
        "rows": rows,
        "paper_stats": paper_stats,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            "The 30+30 mixed run uses source-record parser-neutral blocks because a real-PDF manifest for all 60 selected papers is not available.",
            "English relevance follows the best valid QASPER annotation; Chinese relevance follows the curated chunk qrels.",
            "All queries search the combined 60-paper corpus; metrics are therefore global mixed-corpus retrieval, not per-paper ranking.",
            "Chinese labels are silver-curated and not yet human-final-reviewed.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--english-dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--english-split",
        type=Path,
        default=PROJECT_ROOT / "eval" / "splits" / "qasper-dev-random-papers-30-v1.json",
    )
    parser.add_argument(
        "--chinese-dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v2-precision",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / ".mixed-optimized-e-30x30-state",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()
    report = run(
        english_dataset=args.english_dataset,
        english_split=args.english_split,
        chinese_dataset_dir=args.chinese_dataset_dir,
        output_path=args.output,
        state_dir=args.state_dir,
        candidate_k=args.candidate_k,
    )
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
