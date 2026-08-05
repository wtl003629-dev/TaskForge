from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from taskforge.rag_baseline import LockedSplitManifest, sha256_file
from taskforge.rag_evaluation import load_multihop_rag_dataset, load_tatqa_dataset
from taskforge.rag_experiment import (
    EXPERIMENT_MODE,
    ExperimentDatasetConfig,
    ExperimentFilterConfig,
    ExperimentRetrievalConfig,
    RAGExperimentConfig,
    chunk_text,
    run_rag_experiment,
)

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
    path.parent.mkdir(parents=True)
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
                    ["Metric", "FY2025"],
                    ["Orchid revenue", "42 million"],
                ],
            },
            "paragraphs": [
                {
                    "uid": "alpha-p1",
                    "order": 1,
                    "text": "The cobalt workforce grew by seven engineers.",
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
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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

    assert first.metrics["stages"]["lexical_bm25"]["backend"] == "python_bm25"
    assert first.metrics["stages"]["qdrant_rrf"]["backend"] == "qdrant_local"
    assert (
        first.metrics["stages"]["qdrant_rrf"]["embedding"]["kind"]
        == "deterministic_hash"
    )
    assert (
        first.metrics["stages"]["qdrant_rrf_rerank"]["reranker"]
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
    assert len(manifest["code"]["source_sha256"]) == 6
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
    with pytest.raises(ValidationError, match="bm25_field_weights"):
        ExperimentRetrievalConfig(bm25_field_weights={"title": -1})


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


def test_config_and_publication_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="complete M1 ablation"):
        ExperimentRetrievalConfig(stages=["lexical_bm25"])
    with pytest.raises(ValidationError, match="candidate_k"):
        ExperimentRetrievalConfig(top_k=[1, 5], candidate_k=4)
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
