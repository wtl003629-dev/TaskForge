from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_rag_retrieval_matrix import main
from tests.test_rag_retrieval_gate import _rows, _write_run


def test_matrix_gate_requires_all_scenarios_to_pass(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", _rows())
    candidate = _write_run(tmp_path, "candidate", _rows())
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scenarios": [
                    {
                        "name": "general_text",
                        "baseline": {
                            "label": "base",
                            "path": baseline.name,
                            "stage": "lexical_bm25",
                        },
                        "candidate": {
                            "label": "new",
                            "path": candidate.name,
                            "stage": "lexical_bm25",
                        },
                        "policy": {
                            "profile": "general_text",
                            "require_identical_retrieval": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "matrix-report.json"
    assert (
        main(
            [
                "--matrix",
                str(matrix),
                "--bootstrap-repetitions",
                "200",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["scenario_count"] == 1


def test_matrix_policy_can_use_profile_local_absolute_p95(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", _rows(), p95=100.0)
    candidate = _write_run(tmp_path, "candidate", _rows(), p95=110.0)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scenarios": [
                    {
                        "name": "table_numeric",
                        "baseline": {
                            "label": "base",
                            "path": baseline.name,
                            "stage": "lexical_bm25",
                        },
                        "candidate": {
                            "label": "new",
                            "path": candidate.name,
                            "stage": "lexical_bm25",
                        },
                        "policy": {"max_p95_ms": 300.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(["--matrix", str(matrix), "--bootstrap-repetitions", "200"]) == 0
    )
