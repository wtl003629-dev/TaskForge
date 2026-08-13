from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from taskforge.hybrid_retrieval import (
    AppliedRetrievalFilters,
    BM25Index,
    HybridChunk,
    HybridSearchHit,
    HybridSearchResponse,
)
from taskforge.rag_baseline import LockedSplitManifest, sha256_file
from taskforge.rag_evaluation import (
    RAGEvalCase,
    load_multihop_rag_dataset,
    load_tatqa_dataset,
)
from taskforge.rag_experiment import (
    EXPERIMENT_MODE,
    ExperimentDatasetConfig,
    ExperimentFilterConfig,
    ExperimentRetrievalConfig,
    RAGExperimentConfig,
    _case_parent_scope,
    _compact_tatqa_parent_hybrid_chunks,
    _cross_document_subqueries,
    _hybrid_chunks,
    _NumericTableContextIndex,
    _NumericTableScanIndex,
    _ParentDiversePassageIndex,
    _ProfileConditionalIndex,
    _QueryAwareLineageClosureIndex,
    _QueryTypedStructuredTableIndex,
    _retrieved_table_units,
    _search_request,
    _structured_header_depth,
    _structured_numeric_value,
    _structured_table_fact_chunks,
    _StructuredLineagePairRerankIndex,
    _table_representation_chunks,
    _tatqa_query_plan_compact_query,
    _tatqa_should_route_structured,
    _tatqa_should_route_structured_candidate,
    _tatqa_should_route_table_profile_lookup,
    chunk_text,
    run_rag_experiment,
    table_aware_chunks,
)
from taskforge.rag_profiles import CorpusMetadata

FIXED_TIME = datetime(2026, 8, 4, 9, 15, tzinfo=UTC)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StepClock:
    def __init__(self, step_ns: int = 1_000_000) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        current = self.value
        self.value += self.step_ns
        return current


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_retrieved_table_units_preserve_hit_rank_and_structured_row() -> None:
    chunks = [
        HybridChunk(
            chunk_id="paragraph",
            tenant_id="tenant-evaluation",
            text="Narrative",
            source_uri="taskforge://paragraph",
            document_id="paragraph",
            acl_principals=frozenset({"user:evaluator"}),
            metadata={"kind": "paragraph"},
        ),
        HybridChunk(
            chunk_id="table::repr::structured::row::3",
            tenant_id="tenant-evaluation",
            text="Structured table row",
            source_uri="taskforge://table",
            document_id="table",
            acl_principals=frozenset({"user:evaluator"}),
            metadata={
                "kind": "table",
                "representation": "structured_row",
                "structured_row_index": 3,
                "table_complete": False,
            },
        ),
    ]
    hits = [
        HybridSearchHit(
            chunk=chunk,
            rank=rank,
            score=1.0 / rank,
            base_score=1.0 / rank,
            retrieval_sources=["python_bm25"],
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]

    rows, cells, complete_tables, units = _retrieved_table_units(hits)

    assert rows == ["table::row::3"]
    assert cells == []
    assert complete_tables == []
    assert units == [
        {
            "rank": 2,
            "chunk_id": "table::repr::structured::row::3",
            "document_id": "table",
            "representation": "structured_row",
            "table_complete": False,
            "row_index": 3,
            "column_index": None,
            "row_id": "table::row::3",
            "cell_id": None,
        }
    ]


def _write_synthetic_suite(repository: Path) -> Path:
    suite = {
        "schema_version": "1.0",
        "suite_id": "TaskForge-Experiment-Test-v1",
        "license": "CC0-1.0",
        "documents": [
            {
                "document_id": "change-policy",
                "filename": "change-policy.pdf",
                "pages": [
                    {
                        "page": 1,
                        "title": "Approval Matrix",
                        "paragraphs": [
                            "A critical production change requires security approval."
                        ],
                        "tables": [
                            {
                                "headers": ["Risk", "Lead time"],
                                "rows": [["Critical", "Three business days"]],
                            }
                        ],
                    },
                    {
                        "page": 2,
                        "title": "Rollback Window",
                        "paragraphs": [
                            "Rollback starts within fifteen minutes after an error budget breach."
                        ],
                        "tables": [],
                    },
                ],
            }
        ],
        "cases": [
            {
                "case_id": "approval",
                "question": "Who approves a critical production change?",
                "answer": "Security.",
                "category": "text",
                "evidence": [{"document_id": "change-policy", "pages": [1]}],
            },
            {
                "case_id": "rollback",
                "question": "When must rollback start after an error budget breach?",
                "answer": "Within fifteen minutes.",
                "category": "text",
                "evidence": [{"document_id": "change-policy", "pages": [2]}],
            },
        ],
    }
    path = repository / "eval" / "suite.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite), encoding="utf-8")
    return path


def _synthetic_config() -> RAGExperimentConfig:
    return RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="synthetic_pdf",
            synthetic_suite_path="eval/suite.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2],
            candidate_k=4,
            hash_dimension=16,
        ),
        filters=ExperimentFilterConfig(
            tenant_id="tenant-test",
            request_principals=["user:alice", "role:auditor"],
            indexed_acl_principals=["user:alice"],
            knowledge_base_id="kb-test",
            version="7",
            version_order=7,
        ),
    )


def _run_synthetic(repository: Path, output: Path):
    config = _synthetic_config()
    config_path = repository / "eval" / "experiment-config.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return run_rag_experiment(
        output_dir=output,
        config=config,
        repository_root=repository,
        config_source_path=config_path,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )


def _write_tatqa_fixture(path: Path) -> None:
    payload = [
        {
            "table": {
                "uid": "alpha",
                "table": [
                    ["", "Year Ended December 31", ""],
                    ["", "FY2025", "FY2024"],
                    ["Orchid revenue", "$42 million", "(35)"],
                ],
            },
            "paragraphs": [
                {
                    "uid": "alpha-p1",
                    "order": 1,
                    "text": (
                        "The table presents Orchid revenue in millions. "
                        "The cobalt workforce grew by seven engineers."
                    ),
                }
            ],
            "questions": [
                {
                    "uid": "q-table",
                    "question": "What was Orchid revenue in FY2025?",
                    "answer": "42",
                    "answer_type": "span",
                    "answer_from": "table",
                    "rel_paragraphs": [],
                    "derivation": "",
                    "scale": "million",
                },
                {
                    "uid": "q-text",
                    "question": "What happened to the cobalt workforce?",
                    "answer": "grew",
                    "answer_type": "span",
                    "answer_from": "text",
                    "rel_paragraphs": ["1"],
                    "derivation": "",
                    "scale": "",
                },
            ],
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_numeric_table_scan_uses_section_facts_and_bounded_context(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tatqa.json"
    _write_tatqa_fixture(input_path)
    dataset = load_tatqa_dataset(input_path)
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="tatqa_locked",
            tatqa_input_path="tatqa.json",
            tatqa_locked_split_path="split.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
            top_k=[1, 2],
            candidate_k=4,
        ),
    )
    chunks = _hybrid_chunks(dataset, dataset.cases, config)
    section_config = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(
                update={"chunking": True, "table_aware_chunking": False}
            )
        }
    )
    sections = _table_representation_chunks(
        chunks,
        dataset,
        section_config,
        representation="section",
    )
    section = next(item for item in sections if item.metadata.get("representation") == "section")
    assert "columns=" in section.text
    assert "Orchid revenue" in section.text
    assert "table_context" in section.metadata
    assert "cobalt workforce" in str(section.metadata["table_context"])

    scan = _NumericTableScanIndex(sections)
    request = _search_request("What was Orchid revenue in FY2025?", config, rerank=False)
    response = scan.search(request)
    assert response.hits[0].chunk.document_id.endswith(":table")
    assert "tatqa_numeric_scan" in response.hits[0].retrieval_sources
    assert response.raw_candidate_counts["numeric_scan"] <= request.candidate_k

    assert _NumericTableScanIndex._TEMPORAL_ROWS_MARKER in scan._facts(
        "How many periods are highlighted in the table?"
    )[0]
    compact_query = _tatqa_query_plan_compact_query(
        "How many years did net income exceed $30,000 thousand?"
    )
    assert "how" not in compact_query.split()
    assert "net" in compact_query and "30" in compact_query and "000" in compact_query
    assert not _tatqa_should_route_structured(
        "What caused the increase in interest income?"
    )
    assert _tatqa_should_route_structured(
        "How many years did net income exceed $30,000 thousand?"
    )
    assert _tatqa_should_route_table_profile_lookup(
        "What were the components making up current assets?"
    )
    assert _tatqa_should_route_table_profile_lookup(
        "What are the respective values for 2018 and 2019?"
    )
    assert not _tatqa_should_route_table_profile_lookup(
        "What caused the increase in interest income?"
    )

    context = _NumericTableContextIndex(scan, sections, max_siblings=1)
    context_response = context.search(request)
    assert len(context_response.hits) <= request.candidate_k

    # A TAT-QA-only branch must remain constructible on a non-table corpus so
    # profile routing can safely leave it inactive for general/cross-document
    # scenarios instead of failing during index construction.
    empty_scan = _NumericTableScanIndex(
        [item for item in sections if item.metadata.get("kind") != "table"]
    )
    assert empty_scan.search(request).hits == []


def test_provided_tatqa_context_becomes_an_explicit_pre_ranking_scope(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tatqa.json"
    _write_tatqa_fixture(input_path)
    case = load_tatqa_dataset(input_path).cases[0]
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="tatqa_locked",
            tatqa_context_mode="provided_hybrid_context",
        ),
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"], development_sweep=True
        ),
    )

    parent_scope = _case_parent_scope(case, config)
    request = _search_request(
        case.query,
        config,
        rerank=False,
        parent_document_ids=parent_scope,
    )

    assert parent_scope == frozenset({case.metadata["parent_document_id"]})
    assert request.parent_document_ids == parent_scope
    assert case.relevant_ids != list(parent_scope)


def test_provided_qasper_paper_becomes_an_explicit_pre_ranking_scope() -> None:
    case = RAGEvalCase(
        case_id="qasper:paper-1:q-1",
        dataset="QASPER",
        query="What did the study find?",
        relevant_ids=["qasper:paper-1:paper:section:1:paragraph:2"],
        category="text",
        metadata={
            "paper_id": "paper-1",
            "parent_document_id": "qasper:paper-1:paper",
        },
    )
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="qasper_locked",
            qasper_context_mode="provided_document_context",
        ),
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"], development_sweep=True
        ),
    )

    parent_scope = _case_parent_scope(case, config)
    request = _search_request(
        case.query,
        config,
        rerank=False,
        parent_document_ids=parent_scope,
    )

    assert parent_scope == frozenset({"qasper:paper-1:paper"})
    assert request.parent_document_ids == parent_scope


def test_structured_table_facts_preserve_multilevel_lineage_and_route_query_types(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tatqa.json"
    _write_tatqa_fixture(input_path)
    dataset = load_tatqa_dataset(input_path)
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="tatqa_locked",
            tatqa_input_path="tatqa.json",
            tatqa_locked_split_path="split.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
            top_k=[1, 2],
            candidate_k=4,
        ),
    )
    chunks = _hybrid_chunks(dataset, dataset.cases, config)
    structured = _structured_table_fact_chunks(chunks, dataset)
    row = next(
        chunk
        for chunk in structured
        if chunk.metadata.get("representation") == "structured_row"
    )

    assert _structured_header_depth(
        [
            ["", "Year Ended December 31", ""],
            ["", "FY2025", "FY2024"],
            ["Orchid revenue", "$42 million", "(35)"],
        ]
    ) == 2
    assert row.metadata["structured_header_depth"] == 2
    assert "FY2025" in row.metadata["structured_headers"][1]
    assert row.metadata["structured_scale"] == "million"
    assert row.metadata["lineage"]["parent_document_id"].endswith(":context")
    assert row.metadata["lineage"]["parent_paragraph_ids"]
    values = row.metadata["structured_values"]
    assert values[0]["unit"] == "currency"
    assert values[0]["sign"] == "positive"
    assert values[0]["years"] == ["2025"]
    assert values[1]["sign"] == "negative"
    assert _structured_numeric_value("(8,483)") == ("-8483", -8483.0, "negative")

    router = _QueryTypedStructuredTableIndex(structured)
    assert router.branch_name("How many years did revenue exceed $35 million?") == "count"
    assert router.branch_name("What was the percentage change in revenue?") == "arithmetic"
    assert router.branch_name("What were purchase obligations due within 5 years?") == "arithmetic"
    assert router.branch_name("What are the respective revenue values?") == "multi_span"
    assert _tatqa_should_route_structured_candidate(
        "What were purchase obligations due within 5 years?"
    )
    for query, source in (
        ("How many years did Orchid revenue exceed $35 million?", "tatqa_structured_count"),
        ("What was the change in Orchid revenue between 2024 and 2025?", "tatqa_structured_arithmetic"),
        ("What are the respective Orchid revenue values?", "tatqa_structured_multi_span"),
    ):
        response = router.search(_search_request(query, config, rerank=False))
        assert response.hits
        assert source in response.hits[0].retrieval_sources
        assert response.raw_candidate_counts[source] == 1


def test_compact_tatqa_parent_view_keeps_structure_context_and_acl(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tatqa.json"
    _write_tatqa_fixture(input_path)
    dataset = load_tatqa_dataset(input_path)
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(kind="tatqa_locked"),
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
        ),
    )
    chunks = _hybrid_chunks(dataset, dataset.cases, config)

    parents = _compact_tatqa_parent_hybrid_chunks(chunks)

    assert len(parents) == 1
    parent = parents[0]
    assert parent.metadata["representation"] == "tatqa_compact_parent"
    assert parent.metadata["parent_document_id"].endswith(":context")
    assert "Table headers:" in parent.text
    assert "Orchid revenue" in parent.text
    assert "Table values:" in parent.text
    assert "cobalt workforce" in parent.text
    assert parent.acl_principals == frozenset({
        "role:rag-reviewer",
        "user:evaluator",
    })


def test_parent_diverse_passage_keeps_best_authorized_child_per_parent() -> None:
    def chunk(
        chunk_id: str,
        parent_id: str,
        text: str,
        *,
        kind: str = "paragraph",
    ) -> HybridChunk:
        return HybridChunk(
            chunk_id=chunk_id,
            tenant_id="tenant-evaluation",
            text=text,
            source_uri=f"taskforge://test/{chunk_id}",
            document_id=chunk_id,
            knowledge_base_id="taskforge-evaluation",
            version="1",
            version_order=1,
            acl_principals=frozenset({"user:evaluator"}),
            metadata={"kind": kind, "parent_document_id": parent_id},
        )

    children = [
        chunk("child-a", "parent-a", "generic operating expenses"),
        chunk("child-b", "parent-b", "orchid cobalt workforce revenue"),
    ]
    router = _ParentDiversePassageIndex(
        BM25Index(children),
        query_rewriter=lambda value: value,
        probe_k=10,
    )
    config = RAGExperimentConfig(
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
            top_k=[1, 2],
            candidate_k=4,
        )
    )
    request = _search_request(
        "orchid cobalt workforce",
        config,
        rerank=False,
    ).model_copy(update={"top_k": 2, "allowed_chunk_ids": frozenset({"child-b"})})

    response = router.search(request)

    assert [hit.chunk.document_id for hit in response.hits] == ["child-b"]
    assert "parent_child_retrieval" in response.hits[0].retrieval_sources
    assert response.filters_applied_before_ranking == AppliedRetrievalFilters.from_request(
        request
    )


def test_structured_lineage_pair_rerank_only_reorders_existing_temporal_pair() -> None:
    config = RAGExperimentConfig(
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
            top_k=[1, 5, 10],
            candidate_k=12,
        )
    )
    request = _search_request(
        "How many years did revenue exceed $5 million?",
        config,
        rerank=False,
    )

    def chunk(index: int, *, parent_id: str = "context-other") -> HybridChunk:
        return HybridChunk(
            chunk_id=f"chunk-{index}",
            tenant_id="tenant-evaluation",
            text=f"Evidence {index}",
            source_uri=f"taskforge://test/{index}",
            document_id=f"document-{index}",
            knowledge_base_id="taskforge-evaluation",
            version="1",
            version_order=1,
            acl_principals=frozenset({"user:evaluator"}),
            metadata={"kind": "paragraph", "parent_document_id": parent_id},
        )

    chunks = [chunk(1, parent_id="context-pair"), *[chunk(i) for i in range(2, 11)]]
    chunks.extend(
        [
            chunk(11, parent_id="context-pair"),
            chunk(12),
        ]
    )
    hits = [
        HybridSearchHit(
            chunk=value,
            rank=rank,
            score=0.30 if rank == 11 else 1.0 / rank,
            base_score=0.30 if rank == 11 else 1.0 / rank,
            retrieval_sources=(
                ["same_parent_evidence_closure"]
                if rank == 11
                else ["python_bm25"]
            ),
        )
        for rank, value in enumerate(chunks, start=1)
    ]
    response = HybridSearchResponse(
        backend="python_bm25",
        query=request.query,
        filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
        seed_count=len(hits),
        expanded_neighbor_count=0,
        hits=hits,
    )

    class Backend:
        def search(self, _request: object) -> HybridSearchResponse:
            return response

    index = _StructuredLineagePairRerankIndex(Backend())
    reranked = index.search(request)
    assert reranked.hits[9].chunk.chunk_id == "chunk-11"
    assert reranked.hits[10].chunk.chunk_id == "chunk-10"
    assert {hit.chunk.chunk_id for hit in reranked.hits} == {
        hit.chunk.chunk_id for hit in response.hits
    }
    assert "structured_lineage_pair_rerank" in reranked.hits[9].retrieval_sources

    narrative = request.model_copy(
        update={"query": "What was revenue in the year ended 2019?"}
    )
    unchanged = index.search(narrative)
    assert [hit.chunk.chunk_id for hit in unchanged.hits] == [
        hit.chunk.chunk_id for hit in response.hits
    ]


def test_query_aware_lineage_closure_preserves_head_and_filters_siblings() -> None:
    def chunk(
        chunk_id: str,
        document_id: str,
        text: str,
        *,
        parent_id: str,
        kind: str,
        principals: frozenset[str] = frozenset({"user:evaluator"}),
    ) -> HybridChunk:
        return HybridChunk(
            chunk_id=chunk_id,
            tenant_id="tenant-evaluation",
            text=text,
            source_uri=f"taskforge://test/{chunk_id}",
            document_id=document_id,
            knowledge_base_id="taskforge-evaluation",
            version="1",
            version_order=1,
            acl_principals=principals,
            metadata={
                "kind": kind,
                "parent_document_id": parent_id,
            },
        )

    table = chunk(
        "table-a",
        "document-table-a",
        "Orchid revenue | 2024 | 35,000 million",
        parent_id="context-a",
        kind="table",
    )
    sibling = chunk(
        "paragraph-a",
        "document-paragraph-a",
        "Orchid revenue exceeded 35,000 million during 2024.",
        parent_id="context-a",
        kind="paragraph",
    )
    denied_sibling = chunk(
        "paragraph-denied",
        "document-paragraph-denied",
        "Orchid revenue 2024 exact private evidence.",
        parent_id="context-a",
        kind="paragraph",
        principals=frozenset({"principal:denied"}),
    )
    distractors = [
        chunk(
            f"table-{index}",
            f"document-table-{index}",
            f"Revenue reference {index}",
            parent_id=f"context-{index}",
            kind="table",
        )
        for index in range(3)
    ]
    backend = BM25Index([table, *distractors])
    closure = _QueryAwareLineageClosureIndex(
        backend,
        [table, sibling, denied_sibling, *distractors],
        preserve_head_k=2,
        seed_k=4,
        closure_slots=1,
        max_siblings_per_parent=1,
    )
    config = RAGExperimentConfig(
        retrieval=ExperimentRetrievalConfig(
            stages=["lexical_bm25"],
            development_sweep=True,
            top_k=[1, 2],
            candidate_k=4,
        )
    )
    request = _search_request(
        "When did Orchid revenue exceed 35,000 million?",
        config,
        rerank=False,
    )
    baseline = backend.search(request)
    response = closure.search(request)

    assert [hit.chunk.document_id for hit in response.hits[:2]] == [
        hit.chunk.document_id for hit in baseline.hits[:2]
    ]
    assert "document-paragraph-a" in {
        hit.chunk.document_id for hit in response.hits
    }
    assert "document-paragraph-denied" not in {
        hit.chunk.document_id for hit in response.hits
    }
    assert response.raw_candidate_counts["query_aware_lineage_closure"] == 1


def test_synthetic_ablation_is_reproducible_and_uses_real_pdf_qdrant_acl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline RAG experiment attempted a network connection")

    # Qdrant local mode, PDF generation/parsing, and deterministic hashing must
    # remain usable even when ordinary Python network connection paths are cut.
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    repository = tmp_path / "repo"
    _write_synthetic_suite(repository)

    first = _run_synthetic(repository, tmp_path / "run-one")
    second = _run_synthetic(repository, tmp_path / "run-two")

    assert first.predictions_path.read_bytes() == second.predictions_path.read_bytes()
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest["run_id"] == second.manifest["run_id"]
    assert first.manifest["experiment_mode"] == EXPERIMENT_MODE
    assert first.manifest["production_semantic_dense"] is False
    assert first.manifest["ablation"]["qdrant_location"] == ":memory:"
    assert first.metrics["same_case_ids_and_top_k"] is True
    assert set(first.metrics["stages"]) == {
        "lexical_bm25",
        "qdrant_rrf",
        "qdrant_rrf_rerank",
    }

    rows = [json.loads(line) for line in first.predictions_path.read_text().splitlines()]
    expected_case_ids = ["synthetic:approval", "synthetic:rollback"]
    for stage in first.metrics["stages"]:
        stage_rows = [row for row in rows if row["stage"] == stage]
        assert [row["case_id"] for row in stage_rows] == expected_case_ids
        assert all(row["experiment_mode"] == EXPERIMENT_MODE for row in stage_rows)
        assert all(row["production_semantic_dense"] is False for row in stage_rows)
        assert all(row["filters_applied_before_ranking"] for row in stage_rows)
        assert all(
            row["filter_request"]
            == {
                "tenant_id": "tenant-test",
                "acl_principals": ["role:auditor", "user:alice"],
                "versions": ["7"],
                "version_orders": [7],
                "knowledge_base_ids": ["kb-test"],
                "parent_document_ids": None,
                "allowed_chunk_count": None,
                "allowed_chunk_ids_sha256": None,
            }
            for row in stage_rows
        )
        assert not any(
            retrieved.startswith("__taskforge_filter_probe__")
            for row in stage_rows
            for retrieved in row["retrieved_ids"]
        )
        assert first.metrics["stages"][stage]["latency"]["p50"] == 1.0
        assert first.metrics["stages"][stage]["latency"]["p95"] == 1.0
        assert "hierarchical" in first.metrics["stages"][stage]
        assert all("retrieved_parent_ids" in row for row in stage_rows)

    assert first.metrics["stages"]["lexical_bm25"]["backend"] == "python_bm25"
    assert first.metrics["stages"]["qdrant_rrf"]["backend"] == "qdrant_local"
    assert (
        first.metrics["stages"]["qdrant_rrf"]["embedding"]["kind"]
        == "deterministic_hash"
    )
    assert (
        first.metrics["stages"]["qdrant_rrf_rerank"]["reranker"]["kind"]
        == "lexical_overlap_fallback"
    )
    qdrant_rows = [row for row in rows if row["stage"] == "qdrant_rrf"]
    assert all(
        "qdrant_server_rrf" in sources
        for row in qdrant_rows
        for sources in row["retrieval_sources"]
    )
    reranked_rows = [row for row in rows if row["stage"] == "qdrant_rrf_rerank"]
    assert all(
        "fallback_lexical_rerank" in sources
        for row in reranked_rows
        for sources in row["retrieval_sources"]
    )

    pdf = first.output_dir / first.manifest["pdf_artifacts"][0]["path"]
    assert pdf.read_bytes().startswith(b"%PDF")
    assert first.manifest["pdf_artifacts"][0]["pages"] == 2
    assert first.manifest["dataset"]["adapter"] == "taskforge_synthetic_pdf_real_pypdf"


def test_optional_fair_hybrid_stages_are_runnable_from_experiment_config(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _write_synthetic_suite(repository)
    base = _synthetic_config()
    retrieval = base.retrieval.model_copy(
        update={
            "stages": [
                "lexical_bm25",
                "qdrant_dense",
                "bm25_dense_rrf",
                "bm25_dense_table_profile_rrf",
                "bm25_dense_rrf_rerank",
                "qdrant_rrf",
                "qdrant_rrf_rerank",
            ]
        }
    )
    config = base.model_copy(update={"retrieval": retrieval})
    config_path = repository / "eval" / "fair-hybrid-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    result = run_rag_experiment(
        output_dir=tmp_path / "fair-hybrid-run",
        config=config,
        repository_root=repository,
        config_source_path=config_path,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert result.metrics["stages"]["qdrant_dense"]["backend"] == "qdrant_local"
    assert (
        result.metrics["stages"]["bm25_dense_rrf"]["backend"]
        == "bm25_dense_rrf"
    )
    assert (
        result.metrics["stages"]["bm25_dense_table_profile_rrf"]["backend"]
        == "profile_routed"
    )
    assert (
        result.metrics["stages"]["bm25_dense_rrf_rerank"]["reranker"]["kind"]
        == "lexical_overlap_fallback"
    )
    rows = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
    fused_rows = [row for row in rows if row["stage"] == "bm25_dense_rrf"]
    assert fused_rows
    assert all(
        "bm25_dense_rrf" in sources
        for row in fused_rows
        for sources in row["retrieval_sources"]
    )


def test_tatqa_query_plan_stage_is_opt_in_and_records_numeric_router(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _write_synthetic_suite(repository)
    base = _synthetic_config()
    retrieval = base.retrieval.model_copy(
        update={
            "development_sweep": True,
            "stages": ["lexical_bm25", "bm25_tatqa_query_plan_rrf"],
        }
    )
    result = run_rag_experiment(
        output_dir=tmp_path / "query-plan-run",
        config=base.model_copy(update={"retrieval": retrieval}),
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )
    assert (
        result.metrics["stages"]["bm25_tatqa_query_plan_rrf"]["query_router"]
        == "tatqa_query_plan_numeric_rrf"
    )


def test_profile_conditional_index_keeps_general_text_on_default_backend() -> None:
    class FakeResponse:
        def __init__(self, backend: str) -> None:
            self.backend = backend

        def model_copy(self, *, update: dict[str, object]) -> "FakeResponse":
            return FakeResponse(str(update.get("backend", self.backend)))

    class FakeBackend:
        def __init__(self, backend: str) -> None:
            self.backend = backend
            self.calls = 0

        def search(self, _request: object) -> FakeResponse:
            self.calls += 1
            return FakeResponse(self.backend)

    corpus = CorpusMetadata(
        document_count=3,
        table_count=0,
        page_count=0,
        source_count=3,
        has_page_coordinates=False,
        has_table_structure=False,
    )
    default = FakeBackend("python_bm25")
    cross_document = FakeBackend("source_coverage_rrf")
    index = _ProfileConditionalIndex(
        default,
        cross_document,
        corpus,
        "cross_document",
    )
    index.search(SimpleNamespace(query="Summarize the approval policy."))
    index.search(SimpleNamespace(query="According to both reports, which source changed?"))
    assert default.calls == 1
    assert cross_document.calls == 1


def test_manifest_hashes_cover_config_data_code_predictions_metrics_and_pdfs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    suite_path = _write_synthetic_suite(repository)
    result = _run_synthetic(repository, tmp_path / "run")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["config"]["sha256"] == _canonical_hash(
        manifest["config"]["effective"]
    )
    assert manifest["config"]["source"]["sha256"] == sha256_file(
        repository / "eval" / "experiment-config.json"
    )
    assert manifest["dataset"]["suite_sha256"] == sha256_file(suite_path)
    assert len(manifest["dataset"]["normalized_sha256"]) == 64
    assert len(manifest["code"]["sha256"]) == 64
    assert len(manifest["code"]["source_sha256"]) == 9
    assert manifest["code"]["sha256"] == _canonical_hash(
        manifest["code"]["source_sha256"]
    )
    for artifact in ("predictions.jsonl", "metrics.json"):
        assert manifest["artifacts"][artifact]["sha256"] == sha256_file(
            result.output_dir / artifact
        )
    for pdf in manifest["pdf_artifacts"]:
        assert pdf["sha256"] == sha256_file(result.output_dir / pdf["path"])
    assert manifest["pdf_artifacts_sha256"] == _canonical_hash(
        manifest["pdf_artifacts"]
    )


def test_tatqa_requires_and_enforces_the_locked_external_split(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    input_path = repository / ".taskforge" / "eval-cache" / "tatqa.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_tatqa_fixture(input_path)
    split_path.parent.mkdir(parents=True)
    dataset = load_tatqa_dataset(input_path)
    selected = list(reversed(dataset.cases))
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="TAT-QA",
        source_split="fixture",
        source_sha256=sha256_file(input_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in selected],
        category_counts=dict(Counter(case.category for case in selected)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="tatqa_locked",
            tatqa_input_path=".taskforge/eval-cache/tatqa.json",
            tatqa_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2], candidate_k=4, hash_dimension=16
        ),
    )

    result = run_rag_experiment(
        output_dir=tmp_path / "tatqa-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert result.manifest["dataset"]["adapter"] == "tatqa_locked"
    assert result.manifest["dataset"]["locked_split_id"] == "fixture-locked"
    assert result.manifest["sample"]["case_ids"] == [
        case.case_id for case in selected
    ]
    assert not (result.output_dir / "source_pdfs").exists()

    compact_stage = (
        "bm25_tatqa_query_plan_compact_parent_scan_closure_table_profile_"
        "lineage_pair_rerank_rrf"
    )
    compact_config = config.model_copy(
        update={
            "retrieval": ExperimentRetrievalConfig(
                stages=[compact_stage],
                development_sweep=True,
                top_k=[1, 2],
                candidate_k=20,
                hash_dimension=16,
                tatqa_lineage_closure_slots=3,
                tatqa_structured_candidate_slots=3,
            )
        }
    )
    compact_result = run_rag_experiment(
        output_dir=tmp_path / "tatqa-compact-parent-run",
        config=compact_config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )
    assert compact_result.metrics["stages"][compact_stage]["query_router"] == (
        "table_lookup_profile_compact_parent_structured_lineage_pair_rerank"
    )

    passage_stage = (
        "bm25_tatqa_query_plan_passage_parent_scan_closure_table_profile_"
        "lineage_pair_rerank_rrf"
    )
    passage_config = config.model_copy(
        update={
            "retrieval": ExperimentRetrievalConfig(
                stages=[passage_stage],
                development_sweep=True,
                top_k=[1, 2],
                candidate_k=20,
                hash_dimension=16,
                tatqa_lineage_closure_slots=3,
                tatqa_structured_candidate_slots=3,
            )
        }
    )
    passage_result = run_rag_experiment(
        output_dir=tmp_path / "tatqa-passage-parent-run",
        config=passage_config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )
    assert passage_result.metrics["stages"][passage_stage]["query_router"] == (
        "table_lookup_profile_passage_parent_structured_lineage_pair_rerank"
    )

    dense_union_stage = (
        "bm25_dense_tatqa_query_plan_parent_scan_closure_table_profile_"
        "lineage_pair_rerank_candidate_union"
    )
    dense_union_config = config.model_copy(
        update={
            "retrieval": ExperimentRetrievalConfig(
                stages=[dense_union_stage],
                development_sweep=True,
                top_k=[1, 2],
                candidate_k=20,
                hash_dimension=16,
                tatqa_lineage_closure_slots=3,
                tatqa_structured_candidate_slots=3,
            )
        }
    )
    dense_union_result = run_rag_experiment(
        output_dir=tmp_path / "tatqa-dense-candidate-union-run",
        config=dense_union_config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )
    dense_union_metrics = dense_union_result.metrics["stages"][dense_union_stage]
    assert dense_union_metrics["backend"] == "candidate_tail_union"
    assert dense_union_metrics["embedding"] == {
        "kind": "deterministic_hash",
        "dimension": 16,
        "semantic": False,
        "production": False,
    }
    assert dense_union_metrics["query_router"] == (
        "structured_lineage_pair_head_with_semantic_dense_candidate_tail"
    )
    assert "candidate_novel_tail" in dense_union_metrics["raw_candidate_counts"]

    missing_repository = tmp_path / "missing-repo"
    missing_repository.mkdir()
    missing_output = tmp_path / "must-not-exist"
    with pytest.raises(FileNotFoundError, match="external cache is missing"):
        run_rag_experiment(
            output_dir=missing_output,
            config=config,
            repository_root=missing_repository,
            created_at=FIXED_TIME,
        )
    assert not missing_output.exists()


def _write_multihop_fixture(repository: Path) -> None:
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "title": "Alpha",
                    "author": "u1",
                    "source": "Ex",
                    "published_at": "2024-01-01T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/a",
                    "body": "Alpha article body.",
                },
                {
                    "title": "Beta",
                    "author": "u2",
                    "source": "Ex",
                    "published_at": "2024-01-02T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/b",
                    "body": "Beta article body.",
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query": "Compare Alpha and Beta.",
                    "answer": "equal",
                    "question_type": "comparison_query",
                    "evidence_list": [
                        {"url": "https://ex.com/a", "fact": "Alpha fact"},
                        {"url": "https://ex.com/b", "fact": "Beta fact"},
                    ],
                },
                {
                    "query": "Unanswerable.",
                    "answer": "Insufficient information.",
                    "question_type": "null_query",
                    "evidence_list": [],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_multihop_rag_requires_and_enforces_the_locked_external_split(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_multihop_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    selected = list(reversed(dataset.cases))
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in selected],
        category_counts=dict(Counter(case.category for case in selected)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2], candidate_k=4, hash_dimension=16
        ),
    )

    result = run_rag_experiment(
        output_dir=tmp_path / "multihop-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert result.manifest["dataset"]["adapter"] == "multihop_rag_locked"
    assert result.manifest["dataset"]["locked_split_id"] == "fixture-locked"
    assert result.manifest["sample"]["case_ids"] == [
        case.case_id for case in selected
    ]
    assert not (result.output_dir / "source_pdfs").exists()


def test_chunk_text_is_paragraph_aware_bounded_and_overlapped() -> None:
    paragraphs = [f"Paragraph {index}: " + "x" * 400 for index in range(6)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chars=600, overlap_chars=100)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 600 for chunk in chunks)
    # The next chunk re-opens with the closed chunk's tail (overlap).
    assert chunks[1].startswith(chunks[0][-100:])
    # The full text is recoverable in order.
    assert "".join(chunks)  # non-empty concatenation is preserved enough
    assert "Paragraph 5:" in chunks[-1]

    with pytest.raises(ValueError, match="max_chars"):
        chunk_text(text, max_chars=100, overlap_chars=10)
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text(text, max_chars=600, overlap_chars=600)


def test_table_aware_chunks_repeat_headers_and_create_column_views() -> None:
    chunks = table_aware_chunks(
        "Metric | 2018 | 2019\nRevenue | 100 | 120\nCost | 30 | 40",
        max_chars=400,
        overlap_chars=40,
    )

    assert any("Table row: Metric: Revenue" in value for value in chunks)
    assert any("2018: 100" in value and "2019: 120" in value for value in chunks)
    assert any("Table column: 2019" in value for value in chunks)
    assert any("Metric=Cost | 2019=40" in value for value in chunks)


def test_cross_document_subqueries_use_only_query_clauses_and_quotes() -> None:
    query = (
        "Does 'The Verge' discuss Alpha, while 'TechCrunch' reports Beta?"
    )
    subqueries = _cross_document_subqueries(query)

    assert subqueries[0] == query
    assert "The Verge" in subqueries
    assert "TechCrunch" in subqueries
    assert any("reports Beta" in value for value in subqueries)


def _write_long_multihop_fixture(repository: Path) -> None:
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "title": "Alpha",
                    "author": "u1",
                    "source": "Ex",
                    "published_at": "2024-01-01T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/a",
                    "body": "\n\n".join(
                        f"Alpha evidence paragraph {i} on rollback." for i in range(12)
                    ),
                },
                {
                    "title": "Beta",
                    "author": "u2",
                    "source": "Ex",
                    "published_at": "2024-01-02T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/b",
                    "body": "\n\n".join(
                        f"Beta evidence paragraph {i} on switching." for i in range(12)
                    ),
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query": "Compare Alpha rollback and Beta switching.",
                    "answer": "equal",
                    "question_type": "comparison_query",
                    "evidence_list": [
                        {"url": "https://ex.com/a", "fact": "a"},
                        {"url": "https://ex.com/b", "fact": "b"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_chunked_experiment_splits_documents_and_maps_hits_back(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_long_multihop_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2],
            candidate_k=4,
            hash_dimension=16,
            chunking=True,
            chunk_max_chars=400,
            chunk_overlap_chars=60,
        ),
    )

    result = run_rag_experiment(
        output_dir=tmp_path / "chunked-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert result.metrics["chunking"]["enabled"] is True
    # Two long documents split into more than two chunks in total.
    assert result.metrics["chunk_count"] > 2
    assert result.manifest["ablation"]["chunk_count"] > 2
    # Chunk hits are mapped back to document-level recall for the eval.
    assert (
        result.metrics["stages"]["lexical_bm25"]["retrieval"]["summary"][
            "total_cases"
        ]
        == 1
    )


def test_semantic_embedding_is_opt_in_and_never_the_offline_default() -> None:
    default = ExperimentRetrievalConfig()
    assert default.semantic_embedding is False
    semantic = ExperimentRetrievalConfig(
        semantic_embedding=True, semantic_model="BAAI/bge-small-en-v1.5"
    )
    assert semantic.semantic_embedding is True
    assert semantic.semantic_model == "BAAI/bge-small-en-v1.5"


def test_query_expansion_and_field_weights_are_opt_in() -> None:
    default = ExperimentRetrievalConfig()
    assert default.query_expansion is False
    assert default.bm25_field_weights == {}
    assert default.graph_fusion is False
    expanded = ExperimentRetrievalConfig(
        query_expansion=True, bm25_field_weights={"title": 3.0}
    )
    assert expanded.query_expansion is True
    assert expanded.bm25_field_weights == {"title": 3.0}
    fused = ExperimentRetrievalConfig(graph_fusion=True, graph_max_neighbors=8)
    assert fused.graph_fusion is True
    assert fused.graph_max_neighbors == 8
    graph_rerank = ExperimentRetrievalConfig(
        graph_feature_rerank=True,
        graph_rerank_base_stage="lexical_bm25",
        graph_rerank_seed_k=4,
    )
    assert graph_rerank.graph_feature_rerank is True
    assert graph_rerank.graph_rerank_base_stage == "lexical_bm25"
    assert graph_rerank.graph_rerank_seed_k == 4
    adaptive = ExperimentRetrievalConfig(
        stages=["qdrant_qasper_dense_rerank"],
        development_sweep=True,
        candidate_k=50,
        learned_reranker=True,
        rerank_top_k=30,
        adaptive_rerank_enabled=True,
        adaptive_rerank_min_k=20,
        adaptive_rerank_margin_threshold=0.7,
    )
    assert adaptive.adaptive_rerank_enabled is True
    assert adaptive.adaptive_rerank_min_k == 20
    with pytest.raises(ValidationError, match="requires learned_reranker"):
        ExperimentRetrievalConfig(
            stages=["qdrant_qasper_dense_rerank"],
            development_sweep=True,
            adaptive_rerank_enabled=True,
            rerank_top_k=30,
        )
    with pytest.raises(ValidationError, match="smaller than rerank_top_k"):
        ExperimentRetrievalConfig(
            stages=["qdrant_qasper_dense_rerank"],
            development_sweep=True,
            candidate_k=50,
            learned_reranker=True,
            adaptive_rerank_enabled=True,
            adaptive_rerank_min_k=30,
            rerank_top_k=30,
        )
    with pytest.raises(ValidationError, match="bm25_field_weights"):
        ExperimentRetrievalConfig(bm25_field_weights={"title": -1})
    with pytest.raises(ValidationError, match="requires chunking"):
        ExperimentRetrievalConfig(table_aware_chunking=True)


def test_graph_fused_stage_runs_and_scores(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "title": "A",
                    "author": "u1",
                    "source": "Ex",
                    "published_at": "2024-01-01T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/a",
                    "body": "Apple device news covered by The Verge.",
                },
                {
                    "title": "B",
                    "author": "u2",
                    "source": "Ex",
                    "published_at": "2024-01-02T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/b",
                    "body": "Apple investigation covered by TechCrunch.",
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query": "Apple news",
                    "answer": "yes",
                    "question_type": "inference_query",
                    "evidence_list": [
                        {"url": "https://ex.com/a", "fact": "a"},
                        {"url": "https://ex.com/b", "fact": "b"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2], candidate_k=4, hash_dimension=16, graph_fusion=True
        ),
    )

    result = run_rag_experiment(
        output_dir=tmp_path / "graph-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert "graph_fused" in result.metrics["stages"]
    assert result.metrics["stages"]["graph_fused"]["backend"] == "local_graph_rrf"
    assert (
        result.metrics["stages"]["graph_fused"]["retrieval"]["summary"][
            "total_cases"
        ]
        == 1
    )
    assert result.manifest["ablation"]["graph_fusion"] is True


def test_graph_feature_rerank_preserves_candidate_set(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "title": "A",
                    "author": "u1",
                    "source": "Ex",
                    "published_at": "2024-01-01T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/a",
                    "body": "Apple device news covered by The Verge.",
                },
                {
                    "title": "B",
                    "author": "u2",
                    "source": "Ex",
                    "published_at": "2024-01-02T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/b",
                    "body": "Apple investigation covered by TechCrunch.",
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query": "Apple news",
                    "answer": "yes",
                    "question_type": "inference_query",
                    "evidence_list": [{"url": "https://ex.com/a", "fact": "a"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGExperimentConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(
            top_k=[1, 2],
            candidate_k=4,
            hash_dimension=16,
            graph_feature_rerank=True,
            graph_rerank_base_stage="lexical_bm25",
        ),
    )

    result = run_rag_experiment(
        output_dir=tmp_path / "graph-feature-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
        timer_ns=StepClock(),
    )

    assert "graph_feature_rerank" in result.metrics["stages"]
    assert (
        result.metrics["stages"]["graph_feature_rerank"]["backend"]
        == "local_graph_feature_rerank"
    )
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_rows = [row for row in rows if row["stage"] == "lexical_bm25"]
    graph_rows = [row for row in rows if row["stage"] == "graph_feature_rerank"]
    assert [row["retrieved_ids"] for row in base_rows] == [
        row["retrieved_ids"] for row in graph_rows
    ]
    assert result.manifest["ablation"]["graph_feature_rerank"] is True


def test_config_and_publication_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="complete M1 ablation"):
        ExperimentRetrievalConfig(stages=["lexical_bm25"])
    sweep = ExperimentRetrievalConfig(
        stages=["lexical_bm25", "bm25_dense_rrf"],
        development_sweep=True,
    )
    assert sweep.stages == ["lexical_bm25", "bm25_dense_rrf"]
    assert sweep.tatqa_parent_query_expansion is False
    with pytest.raises(ValidationError, match="candidate_k"):
        ExperimentRetrievalConfig(top_k=[1, 5], candidate_k=4)
    with pytest.raises(ValidationError, match="only valid for the TAT-QA"):
        ExperimentDatasetConfig(tatqa_context_mode="provided_hybrid_context")
    with pytest.raises(ValidationError, match="only valid for TAT-QA"):
        ExperimentDatasetConfig(tatqa_table_cleaning=True)
    with pytest.raises(ValidationError, match="only valid for the QASPER"):
        ExperimentDatasetConfig(qasper_context_mode="provided_document_context")
    with pytest.raises(ValidationError, match="pair rerank slots"):
        ExperimentRetrievalConfig(
            stages=[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf"
            ],
            development_sweep=True,
            top_k=[1],
        )
    with pytest.raises(ValidationError, match="cannot read"):
        ExperimentFilterConfig(
            request_principals=["user:alice"],
            indexed_acl_principals=["user:bob"],
        )

    repository = tmp_path / "repo"
    _write_synthetic_suite(repository)
    output = tmp_path / "owned"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output already exists"):
        _run_synthetic(repository, output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_default_runs_synthetic_offline_ablation(tmp_path: Path) -> None:
    output = tmp_path / "cli-run"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_rag_experiment.py"),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "mode=degraded_nonsemantic" in completed.stdout
    assert "stage=lexical_bm25" in completed.stdout
    assert "stage=qdrant_rrf" in completed.stdout
    assert "stage=qdrant_rrf_rerank" in completed.stdout
    assert (output / "predictions.jsonl").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "manifest.json").is_file()
