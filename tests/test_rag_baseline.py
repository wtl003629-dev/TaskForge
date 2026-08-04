from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from taskforge.rag_baseline import (
    BM25Config,
    LockedSplitManifest,
    RAGBaselineConfig,
    SamplingConfig,
    run_rag_baseline,
    select_locked_cases,
    sha256_file,
)


FIXED_TIME = datetime(2026, 8, 4, 8, 30, tzinfo=timezone.utc)


def _tatqa_fixture(path: Path) -> None:
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
        },
        {
            "table": {
                "uid": "beta",
                "table": [
                    ["Metric", "FY2024"],
                    ["Nimbus margin", "18 percent"],
                ],
            },
            "paragraphs": [
                {
                    "uid": "beta-p1",
                    "order": 1,
                    "text": "The quartz office opened in Oslo.",
                }
            ],
            "questions": [
                {
                    "uid": "q-arithmetic",
                    "question": "What was the Nimbus margin?",
                    "answer": "18",
                    "answer_type": "arithmetic",
                    "answer_from": "table",
                    "rel_paragraphs": [],
                    "derivation": "18",
                    "scale": "percent",
                },
                {
                    "uid": "q-count",
                    "question": "How many quartz offices opened?",
                    "answer": 1,
                    "answer_type": "count",
                    "answer_from": "text",
                    "rel_paragraphs": ["1"],
                    "derivation": "1",
                    "scale": "",
                },
            ],
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(*, limit: int = 4, seed: int = 73) -> RAGBaselineConfig:
    return RAGBaselineConfig(
        sampling=SamplingConfig(limit=limit, seed=seed),
        retrieval=BM25Config(k1=1.2, b=0.75),
        top_k=[1, 2],
    )


def test_baseline_is_reproducible_and_publishes_complete_evidence(tmp_path: Path) -> None:
    source = tmp_path / "tiny-tatqa.json"
    _tatqa_fixture(source)

    first = run_rag_baseline(
        input_path=source,
        output_dir=tmp_path / "run-one",
        config=_config(limit=3),
        repository_root=Path(__file__).resolve().parents[1],
        created_at=FIXED_TIME,
    )
    second = run_rag_baseline(
        input_path=source,
        output_dir=tmp_path / "run-two",
        config=_config(limit=3),
        repository_root=Path(__file__).resolve().parents[1],
        created_at=FIXED_TIME,
    )

    assert first.predictions_path.read_bytes() == second.predictions_path.read_bytes()
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert {item.name for item in first.output_dir.iterdir()} == {
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
    }
    rows = [json.loads(line) for line in first.predictions_path.read_text().splitlines()]
    assert len(rows) == 3
    assert len({row["category"] for row in rows}) == 3


def test_baseline_metrics_and_manifest_hashes_are_auditable(tmp_path: Path) -> None:
    source = tmp_path / "tiny-tatqa.json"
    _tatqa_fixture(source)
    result = run_rag_baseline(
        input_path=source,
        output_dir=tmp_path / "run",
        config=_config(),
        repository_root=Path(__file__).resolve().parents[1],
        created_at=FIXED_TIME,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert metrics["summary"]["total_cases"] == 4
    assert metrics["summary"]["missing_predictions"] == 0
    assert metrics["summary"]["recall_at_k"]["1"] == 1.0
    assert metrics["summary"]["mrr_at_k"]["1"] == 1.0
    assert manifest["dataset"]["sha256"] == sha256_file(source)
    assert manifest["dataset"]["license"] == "CC BY 4.0"
    assert manifest["created_at"] == "2026-08-04T08:30:00Z"
    assert manifest["top_k"] == [1, 2]
    assert len(manifest["config"]["sha256"]) == 64
    config_payload = json.dumps(
        manifest["config"]["effective"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert manifest["config"]["sha256"] == hashlib.sha256(config_payload).hexdigest()
    for artifact_name in ("predictions.jsonl", "metrics.json"):
        assert manifest["artifacts"][artifact_name]["sha256"] == sha256_file(
            result.output_dir / artifact_name
        )
    assert manifest["code"]["package_version"]
    assert manifest["code"]["source_sha256"]["taskforge.rag_baseline"]


def test_missing_input_fails_without_publishing_output(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(FileNotFoundError, match="input does not exist"):
        run_rag_baseline(
            input_path=tmp_path / "missing.json",
            output_dir=output,
            config=_config(),
            created_at=FIXED_TIME,
        )
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "tiny-tatqa.json"
    _tatqa_fixture(source)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        run_rag_baseline(
            input_path=source,
            output_dir=output,
            config=_config(),
            created_at=FIXED_TIME,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_locked_split_binds_checksum_order_and_categories(tmp_path: Path) -> None:
    source = tmp_path / "tiny-tatqa.json"
    _tatqa_fixture(source)
    from taskforge.rag_evaluation import load_tatqa_dataset

    dataset = load_tatqa_dataset(source)
    selected_ids = [dataset.cases[1].case_id, dataset.cases[0].case_id]
    counts: dict[str, int] = {}
    for case in dataset.cases[:2]:
        counts[case.category] = counts.get(case.category, 0) + 1
    manifest = LockedSplitManifest(
        split_id="locked",
        dataset="TAT-QA",
        source_split="fixture",
        source_sha256=sha256_file(source),
        selection={"fixture": True},
        case_ids=selected_ids,
        category_counts=counts,
    )
    selected = select_locked_cases(
        dataset.cases,
        manifest,
        dataset_sha256=sha256_file(source),
    )
    assert [case.case_id for case in selected] == selected_ids
    with pytest.raises(ValueError, match="checksum"):
        select_locked_cases(dataset.cases, manifest, dataset_sha256="0" * 64)


def test_run_uses_repository_locked_split_and_records_hash(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "eval" / "splits").mkdir(parents=True)
    source = tmp_path / "tiny-tatqa.json"
    _tatqa_fixture(source)
    from taskforge.rag_evaluation import load_tatqa_dataset

    dataset = load_tatqa_dataset(source)
    chosen = dataset.cases[:2]
    counts: dict[str, int] = {}
    for case in chosen:
        counts[case.category] = counts.get(case.category, 0) + 1
    split_path = repository / "eval" / "splits" / "locked.json"
    split_path.write_text(
        LockedSplitManifest(
            split_id="locked",
            dataset="TAT-QA",
            source_split="fixture",
            source_sha256=sha256_file(source),
            selection={"fixture": True},
            case_ids=[case.case_id for case in reversed(chosen)],
            category_counts=counts,
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = _config(limit=1).model_copy(update={"locked_split": "eval/splits/locked.json"})
    config = RAGBaselineConfig.model_validate(config.model_dump())
    result = run_rag_baseline(
        input_path=source,
        output_dir=tmp_path / "locked-run",
        config=config,
        repository_root=repository,
        created_at=FIXED_TIME,
    )
    sample = result.manifest["sample"]
    assert sample["case_ids"] == [case.case_id for case in reversed(chosen)]
    assert sample["locked_split"]["sha256"] == sha256_file(split_path)
