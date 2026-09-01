"""Evaluate optimized-e on the mixed 30+30 cohort using real MinerU PDFs.

Unlike ``evaluate_mixed_optimized_e.py``, this evaluator does not synthesize
parser blocks. It loads the frozen English PDF manifest, resolves the local
Chinese PDF paths from the dataset, parses all 60 PDFs through MinerU, and
then applies the same optimized-e chunking and retrieval route globally.
Gold paragraph/chunk labels are aligned to MinerU blocks before scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
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

import evaluate_mixed_optimized_e as source_eval  # noqa: E402

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.pdf_parsing.mineru_client import MinerUClient  # noqa: E402
from taskforge.qasper_alignment import (  # noqa: E402
    AlignmentChunk,
    GoldEvidenceUnit,
    align_gold_unit,
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

RECALL_KS = source_eval.RECALL_KS
TENANT_ID = "mixed-optimized-e-30x30-mineru"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_english_pdf_manifest(
    manifest_path: Path,
    paper_ids: set[str],
) -> dict[str, dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or raw.get("dataset") != "QASPER":
        raise ValueError("English real-PDF manifest is not a QASPER v1 manifest")
    rows = raw.get("papers")
    if not isinstance(rows, list):
        raise ValueError("English real-PDF manifest has no papers list")
    result: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("English real-PDF manifest row is not an object")
        paper_id = str(item.get("paper_id") or "").strip()
        relative = str(item.get("path") or "").strip()
        if not paper_id or not relative or paper_id in result:
            raise ValueError(f"invalid English real-PDF manifest row: {item!r}")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
            raise FileNotFoundError(f"English PDF missing or invalid: {path}")
        expected_sha = str(item.get("sha256") or "").strip().casefold()
        actual_sha = _sha256(path)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"English PDF checksum mismatch for {paper_id}")
        result[paper_id] = {
            **item,
            "path": path,
            "sha256": actual_sha,
        }
    missing = sorted(paper_ids.difference(result))
    if missing:
        raise ValueError(f"English real-PDF manifest missing papers: {missing[:5]}")
    return {paper_id: result[paper_id] for paper_id in paper_ids}


def _load_chinese_pdf_manifest(
    papers_path: Path,
    paper_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(papers_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            paper_id = str(item.get("paper_id") or "").strip()
            if paper_id not in paper_ids:
                continue
            local_pdf = str(item.get("local_pdf") or "").strip()
            path = Path(local_pdf).resolve()
            if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
                raise FileNotFoundError(f"Chinese PDF missing or invalid: {path}")
            result[paper_id] = {
                "paper_id": paper_id,
                "title": str(item.get("title") or paper_id),
                "path": path,
                "sha256": _sha256(path),
                "pages": item.get("pages"),
            }
    missing = sorted(paper_ids.difference(result))
    if missing:
        raise ValueError(f"Chinese PDF dataset missing papers: {missing[:5]}")
    return result


async def _parse_pdfs(
    specs: list[dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, Any], float]:
    if not settings.mineru_base_url or not settings.mineru_expected_version:
        raise RuntimeError("MinerU URL and exact expected version are required")
    cache_root = settings.mineru_cache_root or (
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "mineru-shared-v1"
    )
    client = MinerUClient(
        settings.mineru_base_url,
        cache_root,
        backend=settings.mineru_backend,
        parse_method=settings.mineru_parse_method,
        effort=settings.mineru_effort,
        expected_version=settings.mineru_expected_version,
        # A full pipeline pass can take several minutes on formula-heavy PDFs;
        # keep the service's bounded client wait above the 5-minute default.
        timeout_seconds=max(settings.mineru_timeout_seconds, 900.0),
        max_retries=settings.mineru_max_retries,
        concurrency=settings.mineru_concurrency,
    )
    started = perf_counter()

    async def one(index: int, spec: dict[str, Any]) -> tuple[str, Any]:
        parsed = await client.parse(
            Path(spec["path"]),
            source_uri=f"file:///{Path(spec['path']).name}",
        )
        print(
            f"[mineru] {index}/{len(specs)} {spec['language']}:{spec['paper_id']} "
            f"pages={parsed.page_count} blocks={len(parsed.blocks)} "
            f"quality={parsed.quality.status}",
            flush=True,
        )
        return str(spec["paper_key"]), parsed

    try:
        values = await asyncio.gather(
            *(one(index, spec) for index, spec in enumerate(specs, start=1))
        )
    finally:
        await client.aclose()
    return dict(values), (perf_counter() - started) * 1_000


def _alignment_map(
    parsed: Any,
    sources: list[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    candidates = [
        AlignmentChunk(
            child_id=block.block_id,
            text=block.text,
            order=block.reading_order,
            section=None,
        )
        for block in parsed.blocks
        if block.indexable and block.text.strip()
    ]
    mapping: dict[str, set[str]] = {}
    counts = {
        "total_units": 0,
        "exact_units": 0,
        "fuzzy_units": 0,
        "ambiguous_units": 0,
        "unaligned_units": 0,
    }
    for source_id, text in sources:
        if not str(text).strip():
            continue
        unit = GoldEvidenceUnit(
            unit_id=source_id,
            text=str(text),
            alternative_paragraph_ids=[source_id],
        )
        alignment = align_gold_unit(unit, candidates)
        counts["total_units"] += 1
        counts[f"{alignment.status}_units"] += 1
        aligned = {
            span.child_id for span in alignment.aligned_child_spans
        }
        if aligned:
            mapping[source_id] = aligned
    counts["alignment_coverage"] = (
        (counts["exact_units"] + counts["fuzzy_units"])
        / counts["total_units"]
        if counts["total_units"]
        else 0.0
    )
    return mapping, counts


def _build_cases(
    *,
    selected_cases: list[Any],
    chinese_query_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, set[str]],
    source_to_children: dict[tuple[str, str], set[str]],
) -> list[dict[str, Any]]:
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
                    mapped.update(
                        source_to_children.get((paper_key, source_id), set())
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
                "relevant_source_ids": sorted(english_source_ids),
            }
        )
    for query in chinese_query_rows:
        query_id = str(query["query_id"])
        paper_id = str(query["paper_id"])
        paper_key = f"zh:{paper_id}"
        relevant_sources = qrels_by_query.get(query_id, set())
        relevant_children = set().union(
            *(
                source_to_children.get((paper_key, source_id), set())
                for source_id in relevant_sources
            )
        )
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
    return cases


def run(
    *,
    english_dataset: Path,
    english_split: Path,
    english_pdf_manifest: Path,
    chinese_dataset_dir: Path,
    chinese_papers: Path,
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
    chinese_titles: dict[str, str] = {}
    for row in chinese_chunk_rows:
        paper_id = str(row["paper_id"])
        chinese_by_paper[paper_id].append(row)
        chinese_titles[paper_id] = str(row.get("title") or paper_id)
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

    english_manifest = _load_english_pdf_manifest(
        english_pdf_manifest,
        set(english_paper_ids),
    )
    chinese_manifest = _load_chinese_pdf_manifest(
        chinese_papers,
        set(chinese_paper_ids),
    )
    specs: list[dict[str, Any]] = []
    for paper_id in english_paper_ids:
        title = str(english_documents[paper_id][0].metadata.get("paper_title") or paper_id)
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
    parsed_by_key, mineru_elapsed_ms = asyncio.run(_parse_pdfs(specs, settings))
    if len(parsed_by_key) != 60:
        raise RuntimeError(f"MinerU parsed {len(parsed_by_key)} of 60 PDFs")

    chunks: list[KnowledgeChunk] = []
    source_to_children: dict[tuple[str, str], set[str]] = {}
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
        source_map, counts = _alignment_map(parsed, sources)
        alignment_stats[paper_key] = counts
        # MinerU assigns a content-addressed document id (``pdf:<sha>``),
        # while the benchmark cases use the stable ``en:<paper>`` /
        # ``zh:<paper>`` key.  Materialize into a per-paper map first, then
        # expose the lineage under the benchmark key used by _build_cases.
        paper_source_to_children: dict[tuple[str, str], set[str]] = {}
        stat = source_eval._materialize_paper(
            parsed,
            paper_id=str(spec["paper_id"]),
            language=str(spec["language"]),
            title=str(spec["title"]),
            source_block_to_id=source_map,
            chunks=chunks,
            source_to_children=paper_source_to_children,
            tenant_id=TENANT_ID,
        )
        for (_, source_id), child_ids in paper_source_to_children.items():
            source_to_children.setdefault((paper_key, source_id), set()).update(child_ids)
        stat.update(
            {
                "paper_key": paper_key,
                "pdf_path": str(Path(spec["path"]).resolve()),
                "pdf_sha256": parsed.sha256,
                "page_count": parsed.page_count,
                "block_count": len(parsed.blocks),
                "parse_quality": parsed.quality.model_dump(mode="json"),
                "alignment": counts,
            }
        )
        paper_stats.append(stat)

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
        index_name="mixed-optimized-e-30x30-mineru",
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
    cases = _build_cases(
        selected_cases=selected_cases,
        chinese_query_rows=chinese_query_rows,
        qrels_by_query=qrels_by_query,
        source_to_children=source_to_children,
    )

    started = perf_counter()
    service.search(
        ResearchQuery(query=cases[0]["query"], top_k=candidate_k, candidate_k=candidate_k),
        principal,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        query_started = perf_counter()
        result = service.search(
            ResearchQuery(query=case["query"], top_k=candidate_k, candidate_k=candidate_k),
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
                    f"{chunk_by_id[item.chunk_id].metadata.get('language')}:"
                    f"{chunk_by_id[item.chunk_id].metadata.get('paper_id')}"
                    for item in result.evidence
                ],
                "recall_at_k": recall,
                "mrr_at_k": {
                    str(k): source_eval._mrr(retrieved_ids, relevant_children, k)
                    for k in RECALL_KS
                },
                "ndcg_at_k": {
                    str(k): source_eval._ndcg(retrieved_ids, relevant_children, k)
                    for k in RECALL_KS
                },
                "candidate_count": result.candidate_count,
                "retrieval_route": result.retrieval_route,
                "latency_ms": latency_ms,
                "alignment_paper": alignment_stats[case["paper_key"]],
            }
        )
    try:
        embedder.close()
    finally:
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    by_language = {
        language: source_eval._aggregate(
            [row for row in rows if row["language"] == language]
        )
        for language in ("en", "zh")
    }
    by_type = {
        question_type: source_eval._aggregate(
            [row for row in rows if row["question_type"] == question_type]
        )
        for question_type in sorted({str(row["question_type"]) for row in rows})
    }
    parse_status_counts: dict[str, int] = defaultdict(int)
    for item in paper_stats:
        parse_status_counts[str(item["parse_quality"]["status"])] += 1
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluation_type": "mixed_annotated_optimized_e_mineru_retrieval",
        "benchmark_track": "mixed_30_chinese_30_english_global_chunk_retrieval_mineru",
        "created_at": datetime.now(UTC).isoformat(),
        "selection": {
            "english_papers": len(english_paper_ids),
            "chinese_papers": len(chinese_paper_ids),
            "total_papers": len(english_paper_ids) + len(chinese_paper_ids),
            "english_cases": sum(row["language"] == "en" for row in rows),
            "chinese_cases": sum(row["language"] == "zh" for row in rows),
            "english_split": str(english_split),
            "english_split_sha256": _sha256(english_split),
            "english_pdf_manifest": str(english_pdf_manifest),
            "english_pdf_manifest_sha256": _sha256(english_pdf_manifest),
            "chinese_dataset": str(chinese_dataset_dir),
            "chinese_papers_manifest": str(chinese_papers),
            "chinese_papers_manifest_sha256": _sha256(chinese_papers),
            "chinese_queries_sha256": _sha256(chinese_queries_path),
            "chinese_qrels_sha256": _sha256(chinese_qrels_path),
            "chinese_chunks_sha256": _sha256(chinese_chunks_path),
        },
        "pipeline": [
            "real PDFs",
            f"MinerU {settings.mineru_expected_version}",
            "optimized-e structure-aware chunking",
            "BM25",
            "Bailian text-embedding-v4",
            "RRF",
            "Bailian qwen3-rerank",
            "Parent-aware rerank",
            "lineage diversity",
        ],
        "rag_profile": {"name": "optimized", "ablation": "e"},
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
                item["policy"] == "structured_parent_child_v1" for item in paper_stats
            ),
            "chunks_total_in_store": len(chunks),
            "mineru_blocks": sum(int(item["block_count"]) for item in paper_stats),
        },
        "alignment": {
            "papers": alignment_stats,
            "total_units": sum(item["total_units"] for item in alignment_stats.values()),
            "exact_units": sum(item["exact_units"] for item in alignment_stats.values()),
            "fuzzy_units": sum(item["fuzzy_units"] for item in alignment_stats.values()),
            "ambiguous_units": sum(item["ambiguous_units"] for item in alignment_stats.values()),
            "unaligned_units": sum(item["unaligned_units"] for item in alignment_stats.values()),
        },
        "metrics": source_eval._aggregate(rows),
        "metrics_by_language": by_language,
        "metrics_by_question_type": by_type,
        "rows": rows,
        "paper_stats": paper_stats,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "mineru_elapsed_ms": mineru_elapsed_ms,
        "limitations": [
            "All 60 PDFs were parsed through MinerU and scored in one global mixed corpus.",
            "English relevance follows the best valid QASPER annotation; Chinese relevance follows curated chunk qrels.",
            "Chinese labels are silver-curated and not yet human-final-reviewed.",
            "Mixed-corpus metrics are not directly equivalent to per-language scope metrics.",
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
        "--english-pdf-manifest",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-random-papers-30-real-pdfs-v1.json",
    )
    parser.add_argument(
        "--chinese-dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v2-precision",
    )
    parser.add_argument(
        "--chinese-papers",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2" / "papers.jsonl.gz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / ".mixed-optimized-e-30x30-mineru-state",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()
    report = run(
        english_dataset=args.english_dataset,
        english_split=args.english_split,
        english_pdf_manifest=args.english_pdf_manifest,
        chinese_dataset_dir=args.chinese_dataset_dir,
        chinese_papers=args.chinese_papers,
        output_path=args.output,
        state_dir=args.state_dir,
        candidate_k=args.candidate_k,
    )
    print(
        json.dumps(
            {"output": str(args.output), "metrics": report["metrics"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
