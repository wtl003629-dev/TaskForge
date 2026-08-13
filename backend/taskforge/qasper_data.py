"""Pinned QASPER Parquet-to-JSON preparation with fail-closed validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

QASPER_HF_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
QASPER_HOMEPAGE = "https://huggingface.co/datasets/allenai/qasper"
QASPER_LICENSE = "CC BY 4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            if target.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite different output: {target}")
            temporary.unlink()
            return
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parallel_records(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object of parallel arrays")
    arrays: dict[str, Sequence[object]] = {}
    lengths: set[int] = set()
    for raw_key, raw_items in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{label} contains an invalid field name")
        if not isinstance(raw_items, list):
            raise ValueError(f"{label}.{raw_key} must be an array")
        arrays[raw_key] = raw_items
        lengths.add(len(raw_items))
    if len(lengths) != 1:
        raise ValueError(f"{label} parallel arrays have different lengths")
    count = next(iter(lengths))
    return [
        {key: items[index] for key, items in arrays.items()}
        for index in range(count)
    ]


def normalize_qasper_parquet_row(row: Mapping[str, object]) -> tuple[str, dict[str, Any]]:
    """Restore one Hugging Face Parquet row to QASPER v0.3 JSON shape."""

    paper_id = str(row.get("id", "")).strip()
    if not paper_id:
        raise ValueError("QASPER Parquet row requires a paper id")
    full_text = _parallel_records(row.get("full_text"), f"paper {paper_id}.full_text")
    qas = _parallel_records(row.get("qas"), f"paper {paper_id}.qas")
    for index, question in enumerate(qas):
        question["answers"] = _parallel_records(
            question.get("answers"),
            f"paper {paper_id}.qas[{index}].answers",
        )
    figures = _parallel_records(
        row.get("figures_and_tables"),
        f"paper {paper_id}.figures_and_tables",
    )
    return paper_id, {
        "title": str(row.get("title", "")),
        "abstract": str(row.get("abstract", "")),
        "full_text": full_text,
        "qas": qas,
        "figures_and_tables": figures,
    }


def _read_parquet(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "QASPER preparation requires the 'evaluation' optional dependency"
        ) from exc
    table = parquet.read_table(path)
    result: dict[str, dict[str, Any]] = {}
    for raw_row in table.to_pylist():
        if not isinstance(raw_row, Mapping):
            raise ValueError("QASPER Parquet row must be an object")
        paper_id, paper = normalize_qasper_parquet_row(raw_row)
        if paper_id in result:
            raise ValueError(f"duplicate QASPER paper id: {paper_id}")
        result[paper_id] = paper
    if not result:
        raise ValueError("QASPER Parquet input is empty")
    return result


def _partition_stats(papers: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    questions = 0
    for paper in papers.values():
        qas = paper.get("qas")
        if not isinstance(qas, list):
            raise ValueError("normalized QASPER paper has invalid qas")
        questions += len(qas)
    return {"paper_count": len(papers), "question_count": questions}


def prepare_qasper_data(
    *,
    train_parquet: str | Path,
    validation_parquet: str | Path,
    train_json: str | Path,
    validation_json: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Convert only train/validation and publish a provenance manifest."""

    inputs = {
        "train": Path(train_parquet).resolve(),
        "validation": Path(validation_parquet).resolve(),
    }
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("QASPER train and validation Parquet files are required")
    partitions = {name: _read_parquet(path) for name, path in inputs.items()}
    overlap = set(partitions["train"]).intersection(partitions["validation"])
    if overlap:
        raise ValueError(f"QASPER train/validation paper overlap: {min(overlap)}")
    outputs = {
        "train": Path(train_json).resolve(),
        "validation": Path(validation_json).resolve(),
    }
    for name, target in outputs.items():
        _atomic_write(target, _canonical_json(partitions[name]))
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "version": "0.3",
        "license": QASPER_LICENSE,
        "homepage": QASPER_HOMEPAGE,
        "source_revision": QASPER_HF_REVISION,
        "final_test_downloaded": False,
        "partitions": {},
    }
    for name in ("train", "validation"):
        manifest["partitions"][name] = {
            "source_parquet": {
                "path": str(inputs[name]),
                "sha256": _sha256(inputs[name]),
                "size_bytes": inputs[name].stat().st_size,
            },
            "normalized_json": {
                "path": str(outputs[name]),
                "sha256": _sha256(outputs[name]),
                "size_bytes": outputs[name].stat().st_size,
            },
            **_partition_stats(partitions[name]),
        }
    _atomic_write(Path(manifest_path), _canonical_json(manifest))
    return manifest


__all__ = [
    "QASPER_HF_REVISION",
    "QASPER_HOMEPAGE",
    "QASPER_LICENSE",
    "normalize_qasper_parquet_row",
    "prepare_qasper_data",
]
