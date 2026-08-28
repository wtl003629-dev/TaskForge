from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_qasper_direct_upload.py"
    spec = importlib.util.spec_from_file_location("evaluate_qasper_direct_upload", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pdf_manifest_requires_complete_checksum_pinned_cohort(tmp_path: Path) -> None:
    module = _module()
    pdf = tmp_path / "paper.pdf"
    body = b"%PDF-1.4\nfixture"
    pdf.write_bytes(body)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "QASPER",
                "cohort_id": "locked-real-pdf-v1",
                "papers": [
                    {
                        "paper_id": "1234.5678",
                        "path": "paper.pdf",
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "source_url": "https://arxiv.org/pdf/1234.5678.pdf",
                        "acquired_at": "2026-08-13T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows, raw = module._load_real_pdf_manifest(manifest, {"1234.5678"})

    assert rows["1234.5678"]["path"] == pdf.resolve()
    assert raw["cohort_id"] == "locked-real-pdf-v1"


def test_real_pdf_manifest_rejects_missing_locked_paper(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "QASPER",
                "papers": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        module._load_real_pdf_manifest(manifest, {"missing"})
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete locked cohort must be rejected")


def test_frozen_query_variants_are_bound_to_split_and_case_text(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "variants.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "split_sha256": "a" * 64,
                "generator": {"model": "frozen-test"},
                "variants": [
                    {
                        "case_id": "case-1",
                        "query": "automobile velocity",
                        "synonym_query": "car speed",
                        "keyword_query": "vehicle speed benchmark",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    variants, raw = module._load_query_variants(
        manifest,
        cases=[SimpleNamespace(case_id="case-1", query="automobile velocity")],
        split_sha256="a" * 64,
    )

    assert variants == {
        "automobile velocity": ("car speed", "vehicle speed benchmark")
    }
    assert raw["generator"] == {"model": "frozen-test"}


def test_retrieval_failure_stage_uses_first_actual_loss() -> None:
    module = _module()

    assert module._retrieval_failure_stage(
        candidate_recall=0.8,
        reranked_top_10_recall=0.8,
        presented_top_10_recall=0.8,
    ) == "candidate_missing"
    assert module._retrieval_failure_stage(
        candidate_recall=1.0,
        reranked_top_10_recall=0.5,
        presented_top_10_recall=0.5,
    ) == "rerank_top10_missing"
    assert module._retrieval_failure_stage(
        candidate_recall=1.0,
        reranked_top_10_recall=1.0,
        presented_top_10_recall=0.5,
    ) == "presentation_window_missing"
    assert module._retrieval_failure_stage(
        candidate_recall=1.0,
        reranked_top_10_recall=1.0,
        presented_top_10_recall=1.0,
    ) == "retrieval_success"


def test_top8_visible_budget_is_separate_from_complete_reranked_recall() -> None:
    module = _module()
    assert module.DEFAULT_AGENT_VISIBLE_K == 8
    assert module._retrieval_failure_stage_for_visible_k(
        candidate_recall=1.0,
        reranked_top_10_recall=1.0,
        reranked_visible_recall=0.75,
        presented_visible_recall=0.5,
    ) == "presentation_window_missing"
    assert module._retrieval_failure_stage_for_visible_k(
        candidate_recall=1.0,
        reranked_top_10_recall=1.0,
        reranked_visible_recall=0.75,
        presented_visible_recall=0.75,
    ) == "retrieval_success"


def test_trace_ranked_ids_uses_final_non_empty_trace() -> None:
    module = _module()
    traces = [
        {"reranked_hits": [{"chunk_id": "old"}]},
        {"reranked_hits": [{"chunk_id": "new"}, {"chunk_id": "new"}]},
    ]
    assert module._trace_ranked_ids(traces, "reranked_hits") == ["new"]
