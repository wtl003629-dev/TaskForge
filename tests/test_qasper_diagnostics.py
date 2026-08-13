from __future__ import annotations

import json
from pathlib import Path

from taskforge import qasper_diagnostics as diagnostics
from taskforge.rag_baseline import LockedSplitManifest
from taskforge.rag_evaluation import EvalCorpusDocument, RAGEvalCase, RAGEvalDataset


def test_qasper_diagnostics_separates_candidate_and_ranking_misses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paper = "qasper:paper:paper"
    first = "qasper:paper:section:0:paragraph:0"
    second = "qasper:paper:section:0:paragraph:1"
    dataset = RAGEvalDataset(
        dataset="QASPER",
        license="CC BY 4.0",
        attribution_url="https://example.invalid/qasper",
        documents=[
            EvalCorpusDocument(
                document_id=first,
                text="The baseline is a lexical model.",
                source_uri="qasper://paper/0/0",
                metadata={"parent_document_id": paper},
            ),
            EvalCorpusDocument(
                document_id=second,
                text="The method improves recall.",
                source_uri="qasper://paper/0/1",
                metadata={"parent_document_id": paper},
            ),
        ],
        cases=[
            RAGEvalCase(
                case_id="case-covered",
                dataset="QASPER",
                query="What is the baseline?",
                relevant_ids=[first],
                category="text",
                answer=["lexical model"],
                metadata={"parent_document_id": paper},
            ),
            RAGEvalCase(
                case_id="case-missing",
                dataset="QASPER",
                query="What improves recall?",
                relevant_ids=[second],
                category="text",
                answer=["method"],
                metadata={"parent_document_id": paper},
            ),
        ],
    )
    split = LockedSplitManifest(
        split_id="qasper-test",
        dataset="QASPER",
        source_split="test",
        source_sha256="a" * 64,
        selection={},
        case_ids=["case-covered", "case-missing"],
        category_counts={"text": 2},
    )
    monkeypatch.setattr(diagnostics, "load_qasper_dataset", lambda _path: dataset)
    monkeypatch.setattr(diagnostics, "load_locked_split", lambda _path: split)
    monkeypatch.setattr(diagnostics, "sha256_file", lambda _path: "a" * 64)
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps({"stages": {"lexical_bm25": {}}}), encoding="utf-8"
    )
    (run / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"stage": "lexical_bm25", "case_id": "case-covered", "retrieved_ids": [first]}
                ),
                json.dumps(
                    {"stage": "lexical_bm25", "case_id": "case-missing", "retrieved_ids": []}
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = diagnostics.diagnose_qasper_run(
        run,
        dataset_path=tmp_path / "dataset.json",
        split_path=tmp_path / "split.json",
    )

    assert report["counts"]["covered_top10"] == 1
    assert report["counts"]["candidate_missing"] == 1
    assert report["counts"]["top10_ranking_failure"] == 0

