from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_qasper_pdf_ablation import ABLATIONS, build_command


def test_ablation_matrix_is_cumulative_and_frozen() -> None:
    assert [spec.ablation_id for spec in ABLATIONS] == [f"A{i}" for i in range(8)]
    assert ABLATIONS[0].parser == "native" and ABLATIONS[0].chunking == "flat"
    assert ABLATIONS[1].parser == "mineru" and ABLATIONS[1].chunking == "flat"
    assert ABLATIONS[2].chunking == "parent_child"
    assert [spec.query_mode for spec in ABLATIONS[2:5]] == [
        "original",
        "synonym",
        "full",
    ]
    assert ABLATIONS[5].rerank is True
    assert ABLATIONS[6].operator_budget == 2
    assert ABLATIONS[7].visual is True


def test_visual_ablation_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-visual-calls"):
        build_command(
            ABLATIONS[7],
            dataset=tmp_path / "dataset.json",
            split=tmp_path / "split.json",
            pdf_manifest=tmp_path / "pdfs.json",
            query_variants=tmp_path / "queries.json",
            mineru_base_url="http://127.0.0.1:8001",
            mineru_version="3.4.4",
            mineru_cache_root=tmp_path / "mineru-cache",
            reranker_backend="fastembed",
            reranker_model="reranker",
            output=tmp_path / "a7.json",
            limit=50,
            confirm_visual_calls=False,
        )
