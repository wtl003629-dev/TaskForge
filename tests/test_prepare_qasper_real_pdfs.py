from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    script = Path(__file__).parents[1] / "scripts" / "prepare_qasper_real_pdfs.py"
    spec = importlib.util.spec_from_file_location("prepare_qasper_real_pdfs", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_override_builds_explicit_arxiv_url(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "versions.json"
    path.write_text(json.dumps({"1902.09666": "v1"}), encoding="utf-8")

    overrides = module._load_version_overrides(path)

    assert overrides == {"1902.09666": "v1"}
    assert module._arxiv_pdf_url("1902.09666", overrides["1902.09666"]) == (
        "https://arxiv.org/pdf/1902.09666v1.pdf"
    )


def test_version_override_rejects_unbounded_values(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "versions.json"
    path.write_text(json.dumps({"1902.09666": "latest"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid arXiv version override"):
        module._load_version_overrides(path)
