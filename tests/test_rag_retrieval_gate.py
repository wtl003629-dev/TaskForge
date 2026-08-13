from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from taskforge.rag_retrieval_gate import (
    BootstrapConfig,
    GateThresholds,
    RAGRetrievalGateError,
    compare_retrieval_runs,
    load_retrieval_run,
    paired_bootstrap,
)


def _write_run(
    root: Path,
    name: str,
    rows: list[dict[str, object]],
    *,
    p95: float = 100.0,
    dataset_sha: str = "dataset-sha",
) -> Path:
    run = root / name
    run.mkdir()
    predictions = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    metrics = {
        "schema_version": "1.0",
        "case_ids": [row["case_id"] for row in rows],
        "top_k": [1, 5, 10],
        "candidate_k": 50,
        "chunking": {"enabled": True, "max_chars": 1500},
        "stages": {
            "lexical_bm25": {
                "latency": {"p95": p95},
                "backend": "python_bm25",
                "raw_candidate_counts": {
                    "bm25": {"min": 1, "max": 1, "mean": 1.0}
                },
            }
        },
    }
    metrics_payload = (json.dumps(metrics, sort_keys=True) + "\n").encode()
    (run / "predictions.jsonl").write_bytes(predictions)
    (run / "metrics.json").write_bytes(metrics_payload)
    manifest = {
        "schema_version": "1.0",
        "run_id": name,
        "dataset": {
            "normalized_sha256": dataset_sha,
            "locked_split_sha256": "split-sha",
        },
        "sample": {"case_ids": [row["case_id"] for row in rows]},
        "config": {
            "effective": {
                "filters": {"tenant_id": "tenant", "knowledge_base_id": "kb"},
                "retrieval": {
                    "top_k": [1, 5, 10],
                    "candidate_k": 50,
                    "chunking": True,
                    "chunk_max_chars": 1500,
                },
            }
        },
        "ablation": {"stages": {"lexical_bm25": {"backend": "python_bm25"}}},
        "artifacts": {
            "predictions.jsonl": {
                "sha256": hashlib.sha256(predictions).hexdigest(),
                "size_bytes": len(predictions),
            },
            "metrics.json": {
                "sha256": hashlib.sha256(metrics_payload).hexdigest(),
                "size_bytes": len(metrics_payload),
            },
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _rows(*, degrade_category_a: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(20):
        case_id = f"case-{index}"
        relevant = f"gold-{index}"
        hit = not (degrade_category_a and index < 10)
        rows.append(
            {
                "stage": "lexical_bm25",
                "case_id": case_id,
                "category": "a" if index < 10 else "b",
                "relevant_ids": [relevant],
                "retrieved_ids": [relevant if hit else f"wrong-{index}"],
                "raw_candidate_counts": {"bm25": 1},
                "retrieval_profile": {"name": "general_text"},
            }
        )
    return rows


def test_paired_bootstrap_is_deterministic() -> None:
    config = BootstrapConfig(repetitions=200, seed=7)
    assert paired_bootstrap([0.0, 0.1, 0.2], config) == paired_bootstrap(
        [0.0, 0.1, 0.2], config
    )


def test_identical_run_passes_and_can_require_exact_retrieval(tmp_path: Path) -> None:
    rows = _rows()
    baseline_path = _write_run(tmp_path, "baseline", rows)
    candidate_path = _write_run(tmp_path, "candidate", rows)
    baseline = load_retrieval_run(baseline_path, label="base", stage="lexical_bm25")
    candidate = load_retrieval_run(candidate_path, label="new", stage="lexical_bm25")
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        bootstrap=BootstrapConfig(repetitions=200),
        require_identical_retrieval=True,
    )
    assert report["passed"] is True
    assert report["candidates"][0]["checks"]["identical_retrieval"] is True


def test_category_regression_fails_even_when_global_recall_is_still_acceptable(
    tmp_path: Path,
) -> None:
    baseline_path = _write_run(tmp_path, "baseline", _rows())
    candidate_path = _write_run(
        tmp_path, "candidate", _rows(degrade_category_a=True)
    )
    baseline = load_retrieval_run(baseline_path, label="base", stage="lexical_bm25")
    candidate = load_retrieval_run(candidate_path, label="new", stage="lexical_bm25")
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        thresholds=GateThresholds(
            max_category_degradation=0.03,
            require_nonnegative_ci_lower=False,
        ),
        bootstrap=BootstrapConfig(repetitions=200),
    )
    assert report["passed"] is False
    assert report["candidates"][0]["checks"]["category_non_regression"] is False


def test_global_and_provided_context_runs_cannot_be_promoted_against_each_other(
    tmp_path: Path,
) -> None:
    rows = _rows()
    baseline_path = _write_run(tmp_path, "global-scope", rows)
    candidate_path = _write_run(tmp_path, "provided-scope", rows)
    manifest_path = candidate_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["effective"]["dataset"] = {
        "kind": "tatqa_locked",
        "tatqa_context_mode": "provided_hybrid_context",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    baseline = load_retrieval_run(
        baseline_path, label="global", stage="lexical_bm25"
    )
    candidate = load_retrieval_run(
        candidate_path, label="provided", stage="lexical_bm25"
    )
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        bootstrap=BootstrapConfig(repetitions=200),
    )

    assert report["passed"] is False
    assert "retrieval scope differs" in report["candidates"][0]["metadata_errors"]


def test_qasper_global_and_provided_document_runs_cannot_be_compared(
    tmp_path: Path,
) -> None:
    rows = _rows()
    baseline_path = _write_run(tmp_path, "qasper-global", rows)
    candidate_path = _write_run(tmp_path, "qasper-provided", rows)
    for path, mode in (
        (baseline_path, "global_discovery"),
        (candidate_path, "provided_document_context"),
    ):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"]["effective"]["dataset"] = {
            "kind": "qasper_locked",
            "qasper_context_mode": mode,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    baseline = load_retrieval_run(
        baseline_path, label="global", stage="lexical_bm25"
    )
    candidate = load_retrieval_run(
        candidate_path, label="provided", stage="lexical_bm25"
    )
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        bootstrap=BootstrapConfig(repetitions=200),
    )

    assert report["passed"] is False
    assert "retrieval scope differs" in report["candidates"][0]["metadata_errors"]


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    path = _write_run(tmp_path, "run", _rows())
    predictions = path / "predictions.jsonl"
    predictions.write_bytes(predictions.read_bytes() + b" ")
    with pytest.raises(RAGRetrievalGateError, match="sha256 mismatch"):
        load_retrieval_run(path, label="run", stage="lexical_bm25")


def test_profile_local_absolute_latency_limit_is_supported(tmp_path: Path) -> None:
    rows = _rows()
    baseline_path = _write_run(tmp_path, "baseline", rows, p95=100.0)
    candidate_path = _write_run(tmp_path, "candidate", rows, p95=110.0)
    baseline = load_retrieval_run(baseline_path, label="base", stage="lexical_bm25")
    candidate = load_retrieval_run(candidate_path, label="new", stage="lexical_bm25")
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        thresholds=GateThresholds(max_p95_ms=300.0),
        bootstrap=BootstrapConfig(repetitions=200),
    )
    assert report["passed"] is True
    assert report["candidates"][0]["checks"]["latency"] is True


def test_absolute_latency_cap_does_not_disable_relative_budget(tmp_path: Path) -> None:
    rows = _rows()
    baseline_path = _write_run(tmp_path, "baseline-both", rows, p95=100.0)
    candidate_path = _write_run(tmp_path, "candidate-both", rows, p95=250.0)
    baseline = load_retrieval_run(
        baseline_path, label="base", stage="lexical_bm25"
    )
    candidate = load_retrieval_run(
        candidate_path, label="new", stage="lexical_bm25"
    )
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        thresholds=GateThresholds(max_p95_ratio=1.2, max_p95_ms=300.0),
        bootstrap=BootstrapConfig(repetitions=200),
    )
    assert report["passed"] is False
    assert report["candidates"][0]["checks"]["latency"] is False


def test_absolute_candidate_recall_floors_are_enforced(tmp_path: Path) -> None:
    baseline_rows = _rows()
    candidate_rows = _rows()
    candidate_rows[0]["retrieved_ids"] = ["wrong-0"]
    baseline_path = _write_run(tmp_path, "baseline-floor", baseline_rows)
    candidate_path = _write_run(tmp_path, "candidate-floor", candidate_rows)
    baseline = load_retrieval_run(
        baseline_path, label="base", stage="lexical_bm25"
    )
    candidate = load_retrieval_run(
        candidate_path, label="new", stage="lexical_bm25"
    )
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        thresholds=GateThresholds(
            min_candidate_recall_at_10=1.0,
            min_candidate_recall_at_candidate_k=1.0,
            require_nonnegative_ci_lower=False,
        ),
        bootstrap=BootstrapConfig(repetitions=200),
    )
    checks = report["candidates"][0]["checks"]
    assert report["passed"] is False
    assert checks["candidate_recall_floor"] is False
    assert checks["candidate_candidate_recall_floor"] is False


def test_synthetic_suite_hash_can_lock_the_pdf_smoke_set(tmp_path: Path) -> None:
    path = _write_run(tmp_path, "synthetic", _rows())
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = {
        "adapter": "taskforge_synthetic_pdf_real_pypdf",
        "normalized_sha256": "pdf-normalized-sha",
        "suite_sha256": "suite-sha",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run = load_retrieval_run(path, label="synthetic", stage="lexical_bm25")
    assert run.locked_split_sha256 == "suite:suite-sha"


def test_profile_filter_compares_only_the_selected_case_slice(tmp_path: Path) -> None:
    rows = _rows()
    for index, row in enumerate(rows):
        row["retrieval_profile"] = {"name": "general_text" if index < 10 else "table_numeric"}
    baseline_path = _write_run(tmp_path, "baseline-profile", rows)
    candidate_rows = [dict(row) for row in rows]
    for row in candidate_rows[:10]:
        row["retrieved_ids"] = [row["relevant_ids"][0]]
    candidate_path = _write_run(tmp_path, "candidate-profile", candidate_rows)
    baseline = load_retrieval_run(
        baseline_path,
        label="base",
        stage="lexical_bm25",
        profile_name="general_text",
    )
    candidate = load_retrieval_run(
        candidate_path,
        label="new",
        stage="lexical_bm25",
        profile_name="general_text",
    )
    report = compare_retrieval_runs(
        baseline,
        [candidate],
        require_identical_retrieval=True,
        bootstrap=BootstrapConfig(repetitions=200),
    )
    assert report["passed"] is True


def test_probe_id_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["retrieved_ids"] = ["__taskforge_filter_probe__:acl"]
    path = _write_run(tmp_path, "run", rows)
    with pytest.raises(RAGRetrievalGateError, match="filter probe"):
        load_retrieval_run(path, label="run", stage="lexical_bm25")
