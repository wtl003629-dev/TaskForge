from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from taskforge.rag_answer_gate import (
    BootstrapConfig,
    GateThresholds,
    RAGAnswerGateError,
    compare_answer_eval_runs,
    load_answer_eval_run,
)

DATASET_SHA = "d" * 64
ANSWER_CONTRACT = {"id": "bare-answer-v1", "format": "plain_text"}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _write_run(
    root: Path,
    name: str,
    *,
    mode: str,
    retriever: str,
    scores: list[float],
    recalls: list[float],
    latencies: list[float],
    categories: list[str] | None = None,
    errors: set[int] = frozenset(),
    fallbacks: set[int] = frozenset(),
    model: str = "model-v1",
    budgets: dict[str, object] | None = None,
) -> Path:
    target = root / name
    target.mkdir()
    category_values = categories or ["multi-hop"] * len(scores)
    rows = [
        {
            "case_id": f"case-{index}",
            "category": category_values[index],
            "token_f1": score,
            "evidence_recall": recalls[index],
            "retrieved_ids": [f"doc-{index}"],
            "latency_ms": {"total": latencies[index]},
            "execution_error": "timeout" if index in errors else None,
            "fallback_used": index in fallbacks,
            "model": model,
            "mode": mode,
            "retriever": retriever,
            "answer_contract": ANSWER_CONTRACT,
        }
        for index, score in enumerate(scores)
    ]
    predictions = _json_bytes(rows)
    metrics = _json_bytes(
        {
            "total_cases": len(rows),
            "avg_token_f1": sum(scores) / len(scores),
            "model": model,
            "mode": mode,
            "retriever": retriever,
        }
    )
    (target / "predictions.jsonl").write_bytes(predictions)
    (target / "metrics.json").write_bytes(metrics)
    effective_budgets = budgets or {
        "evidence_top_k": 5,
        "max_evidence_chars": 16_000,
        "agent_max_steps": 8,
        "retrieval": {"candidate_k": 25, "top_k": [1, 5, 10]},
    }
    manifest = {
        "run_id": name,
        "dataset": {"normalized_sha256": DATASET_SHA},
        "answer_contract": ANSWER_CONTRACT,
        "sample": {
            "case_ids": [row["case_id"] for row in rows],
            "selected_cases": len(rows),
        },
        "config": {
            "effective": {
                "model": model,
                "mode": mode,
                "retriever": retriever,
                **effective_budgets,
            }
        },
        "artifacts": {
            "predictions.jsonl": {
                "sha256": hashlib.sha256(predictions).hexdigest(),
                "size_bytes": len(predictions),
            },
            "metrics.json": {
                "sha256": hashlib.sha256(metrics).hexdigest(),
                "size_bytes": len(metrics),
            },
        },
    }
    (target / "manifest.json").write_bytes(_json_bytes(manifest))
    return target


def test_retriever_gate_verifies_artifacts_and_is_deterministic(tmp_path: Path) -> None:
    baseline_path = _write_run(
        tmp_path,
        "baseline",
        mode="naive",
        retriever="bm25",
        scores=[0.4] * 12,
        recalls=[0.5] * 12,
        latencies=[100.0] * 12,
    )
    candidate_path = _write_run(
        tmp_path,
        "candidate",
        mode="naive",
        retriever="qdrant_rrf_rerank",
        scores=[0.5] * 12,
        recalls=[0.6] * 12,
        latencies=[150.0] * 12,
    )
    baseline = load_answer_eval_run(baseline_path, label="bm25")
    candidate = load_answer_eval_run(candidate_path, label="hybrid")
    kwargs = {
        "comparison_type": "retriever",
        "bootstrap": BootstrapConfig(repetitions=200, seed=17),
    }

    first = compare_answer_eval_runs(baseline, [candidate], **kwargs)
    second = compare_answer_eval_runs(baseline, [candidate], **kwargs)

    assert first == second
    assert first["passed"] is True
    assert first["candidates"][0]["paired_bootstrap"]["token_f1"]["lower"] == pytest.approx(0.1)
    assert first["candidates"][0]["summary"]["candidate"]["evidence_recall"] == pytest.approx(0.6)


def test_loader_rejects_tampered_artifact(tmp_path: Path) -> None:
    path = _write_run(
        tmp_path,
        "tampered",
        mode="naive",
        retriever="bm25",
        scores=[0.5],
        recalls=[0.5],
        latencies=[10.0],
    )
    with (path / "predictions.jsonl").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(RAGAnswerGateError, match="sha256 mismatch"):
        load_answer_eval_run(path)


def test_invariant_mismatch_fails_closed(tmp_path: Path) -> None:
    baseline = load_answer_eval_run(
        _write_run(
            tmp_path,
            "baseline",
            mode="naive",
            retriever="bm25",
            scores=[0.2, 0.2],
            recalls=[0.2, 0.2],
            latencies=[10.0, 10.0],
        )
    )
    candidate = load_answer_eval_run(
        _write_run(
            tmp_path,
            "candidate",
            mode="naive",
            retriever="rrf",
            model="other-model",
            scores=[0.8, 0.8],
            recalls=[0.8, 0.8],
            latencies=[10.0, 10.0],
        )
    )

    report = compare_answer_eval_runs(
        baseline,
        [candidate],
        comparison_type="retriever",
        bootstrap=BootstrapConfig(repetitions=100),
    )

    assert report["passed"] is False
    assert "model_mismatch" in report["candidates"][0]["compatibility_errors"]


def test_agentic_gate_enforces_category_latency_errors_and_fallback(tmp_path: Path) -> None:
    categories = ["multi-hop"] * 10 + ["small"] * 2
    baseline = load_answer_eval_run(
        _write_run(
            tmp_path,
            "naive",
            mode="naive",
            retriever="rrf",
            scores=[0.8] * 12,
            recalls=[0.8] * 12,
            latencies=[100.0] * 12,
            categories=categories,
        )
    )
    candidate = load_answer_eval_run(
        _write_run(
            tmp_path,
            "agentic",
            mode="agentic",
            retriever="rrf",
            scores=[0.7] * 10 + [1.0] * 2,
            recalls=[0.7] * 10 + [1.0] * 2,
            latencies=[300.0] * 12,
            categories=categories,
            errors={0},
            fallbacks={1},
        )
    )

    report = compare_answer_eval_runs(
        baseline,
        [candidate],
        comparison_type="agentic",
        thresholds=GateThresholds.for_comparison("agentic"),
        bootstrap=BootstrapConfig(repetitions=100),
    )
    result = report["candidates"][0]

    assert result["passed"] is False
    assert result["gates"]["category_degradation"]["passed"] is False
    assert result["gates"]["p95_latency_ratio"]["passed"] is False
    assert result["gates"]["execution_errors"]["passed"] is False
    assert result["gates"]["fallback"]["passed"] is False
    assert next(row for row in result["categories"] if row["category"] == "small")["gated"] is False


def test_cli_prints_json_and_atomically_writes_optional_output(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "cli-baseline",
        mode="naive",
        retriever="bm25",
        scores=[0.4, 0.4],
        recalls=[0.4, 0.4],
        latencies=[50.0, 50.0],
    )
    candidate = _write_run(
        tmp_path,
        "cli-candidate",
        mode="naive",
        retriever="rrf",
        scores=[0.6, 0.6],
        recalls=[0.6, 0.6],
        latencies=[60.0, 60.0],
    )
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "reports" / "gate.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "compare_rag_answer_runs.py"),
            "--comparison",
            "retriever",
            "--baseline",
            f"bm25={baseline}",
            "--candidate",
            f"rrf={candidate}",
            "--bootstrap-repetitions",
            "100",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    stdout_report = json.loads(completed.stdout)
    assert stdout_report["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == stdout_report
