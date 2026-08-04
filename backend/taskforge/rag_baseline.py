"""Reproducible, provider-free lexical baseline for TaskForge RAG evaluation.

This module intentionally has no network or model-provider dependency.  It is
the floor against which dense, hybrid, reranked, and graph-assisted retrieval
must be compared.  Runs are published as an atomic directory containing raw
predictions, evaluator output, and enough provenance to reproduce the sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from . import __version__
from .domain import StrictModel
from .rag_evaluation import (
    EvalCorpusDocument,
    RAGEvalCase,
    RetrievalEvaluationReport,
    RetrievalPrediction,
    evaluate_retrieval,
    load_tatqa_dataset,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")


class SamplingConfig(StrictModel):
    strategy: Literal["category_round_robin"] = "category_round_robin"
    limit: int = Field(default=100, ge=1, le=100_000)
    seed: int = 20_260_804
    categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def categories_are_unique(self) -> SamplingConfig:
        normalized = [item.strip() for item in self.categories]
        if any(not item for item in normalized):
            raise ValueError("sampling categories must be non-empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("sampling categories must not contain duplicates")
        object.__setattr__(self, "categories", normalized)
        return self


class BM25Config(StrictModel):
    kind: Literal["bm25"] = "bm25"
    k1: float = Field(default=1.2, gt=0.0, le=10.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class RAGBaselineConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_adapter: Literal["tatqa"] = "tatqa"
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    retrieval: BM25Config = Field(default_factory=BM25Config)
    top_k: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    locked_split: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def top_k_is_safe(self) -> RAGBaselineConfig:
        if not self.top_k:
            raise ValueError("top_k must not be empty")
        if any(value <= 0 or value > 10_000 for value in self.top_k):
            raise ValueError("top_k values must be between 1 and 10000")
        if len(self.top_k) != len(set(self.top_k)):
            raise ValueError("top_k must not contain duplicates")
        object.__setattr__(self, "top_k", sorted(self.top_k))
        if self.locked_split is not None:
            value = self.locked_split.strip().replace("\\", "/")
            if not value or value.startswith("/") or ":" in value or ".." in value.split("/"):
                raise ValueError("locked_split must be a safe repository-relative path")
            object.__setattr__(self, "locked_split", value)
        return self


class LockedSplitManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    split_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection: dict[str, Any]
    case_ids: list[str] = Field(min_length=1)
    category_counts: dict[str, int]

    @model_validator(mode="after")
    def cases_are_unique(self) -> LockedSplitManifest:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("locked split case IDs must be unique")
        if any(not value.strip() for value in self.case_ids):
            raise ValueError("locked split case IDs must be non-empty")
        if any(not key.strip() or value < 0 for key, value in self.category_counts.items()):
            raise ValueError("locked split category counts are invalid")
        if sum(self.category_counts.values()) != len(self.case_ids):
            raise ValueError("locked split category counts do not match case IDs")
        return self


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    score: float


@dataclass(frozen=True)
class BaselineRunResult:
    output_dir: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    report: RetrievalEvaluationReport
    manifest: Mapping[str, Any]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_baseline_config(path: str | Path) -> RAGBaselineConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"baseline config does not exist: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline config is not valid JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("baseline config must be a JSON object")
    return RAGBaselineConfig.model_validate(raw)


def load_locked_split(path: str | Path) -> LockedSplitManifest:
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"locked split does not exist: {split_path}")
    return LockedSplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))


def select_locked_cases(
    cases: Sequence[RAGEvalCase],
    manifest: LockedSplitManifest,
    *,
    dataset_sha256: str,
) -> list[RAGEvalCase]:
    if manifest.source_sha256 != dataset_sha256:
        raise ValueError("locked split source checksum does not match the dataset")
    available = {case.case_id: case for case in cases}
    missing = [case_id for case_id in manifest.case_ids if case_id not in available]
    if missing:
        raise ValueError(f"locked split contains unavailable cases: {missing[0]}")
    selected = [available[case_id] for case_id in manifest.case_ids]
    actual_counts = Counter(case.category for case in selected)
    if dict(sorted(actual_counts.items())) != dict(sorted(manifest.category_counts.items())):
        raise ValueError("locked split category counts do not match normalized cases")
    return selected


def _seeded_key(seed: int, category: str, case_id: str) -> tuple[str, str]:
    material = f"{seed}\0{category}\0{case_id}".encode()
    return hashlib.sha256(material).hexdigest(), case_id


def select_stratified_cases(
    cases: Sequence[RAGEvalCase],
    *,
    limit: int,
    seed: int,
    categories: Sequence[str] = (),
) -> list[RAGEvalCase]:
    """Select a balanced sample deterministically, independent of input order."""

    if limit <= 0:
        raise ValueError("sample limit must be positive")
    requested = tuple(categories)
    if len(requested) != len(set(requested)):
        raise ValueError("sample categories must not contain duplicates")

    grouped: dict[str, list[RAGEvalCase]] = defaultdict(list)
    for case in cases:
        if not requested or case.category in requested:
            grouped[case.category].append(case)
    if requested:
        missing = sorted(set(requested).difference(grouped))
        if missing:
            raise ValueError(f"requested categories are absent: {', '.join(missing)}")
    if not grouped:
        raise ValueError("dataset has no cases eligible for sampling")

    category_order = sorted(
        grouped,
        key=lambda value: (
            hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
            value,
        ),
    )
    for category, values in grouped.items():
        grouped[category] = sorted(
            values,
            key=lambda case: _seeded_key(seed, category, case.case_id),
        )

    selected: list[RAGEvalCase] = []
    offsets = {category: 0 for category in category_order}
    while len(selected) < limit:
        made_progress = False
        for category in category_order:
            offset = offsets[category]
            values = grouped[category]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[category] += 1
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return selected


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Small deterministic BM25 implementation used only as an offline floor."""

    def __init__(
        self,
        documents: Sequence[EvalCorpusDocument],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 corpus must not be empty")
        if k1 <= 0.0 or not 0.0 <= b <= 1.0:
            raise ValueError("invalid BM25 parameters")
        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("BM25 document identifiers must be unique")
        self._documents = tuple(sorted(documents, key=lambda item: item.document_id))
        self._k1 = float(k1)
        self._b = float(b)
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            tokens = tokenize(document.text)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())
        self._average_length = sum(self._lengths) / len(self._lengths)
        count = len(self._documents)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, limit: int) -> list[RankedDocument]:
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        query_terms = tuple(dict.fromkeys(tokenize(query)))
        results: list[RankedDocument] = []
        for document, frequencies, length in zip(
            self._documents,
            self._term_frequencies,
            self._lengths,
            strict=True,
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                if self._average_length:
                    normalization = 1.0 - self._b + self._b * length / self._average_length
                else:
                    normalization = 1.0
                denominator = frequency + self._k1 * normalization
                score += self._idf[term] * frequency * (self._k1 + 1.0) / denominator
            results.append(RankedDocument(document_id=document.document_id, score=score))
        results.sort(key=lambda item: (-item.score, item.document_id))
        return results[: min(limit, len(results))]


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = result.stdout.strip()
    return candidate if result.returncode == 0 and candidate else None


def _source_metadata(repository_root: Path) -> dict[str, Any]:
    baseline_path = Path(__file__).resolve()
    evaluator_path = baseline_path.with_name("rag_evaluation.py")
    return {
        "package": "taskforge-agent",
        "package_version": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": _git_commit(repository_root),
        "source_sha256": {
            "taskforge.rag_baseline": sha256_file(baseline_path),
            "taskforge.rag_evaluation": sha256_file(evaluator_path),
        },
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _write_staged_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_run(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"baseline output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for name, payload in files.items():
            if Path(name).name != name:
                raise ValueError(f"unsafe artifact name: {name}")
            _write_staged_file(staging / name, payload)
        os.replace(staging, output_dir)
    except Exception:
        # The path is generated by mkdtemp under the resolved output parent;
        # removing it cannot touch a caller-owned directory.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_rag_baseline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: RAGBaselineConfig,
    repository_root: str | Path | None = None,
    created_at: datetime | None = None,
) -> BaselineRunResult:
    """Run the baseline and atomically publish its three evidence artifacts."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"TAT-QA input does not exist: {source}")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"baseline output already exists: {target}")

    dataset_checksum = sha256_file(source)
    dataset = load_tatqa_dataset(source)
    if not dataset.documents:
        raise ValueError("TAT-QA adapter produced an empty corpus")
    repository = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    locked_split_path: Path | None = None
    locked_split: LockedSplitManifest | None = None
    if config.locked_split is not None:
        locked_split_path = (repository / config.locked_split).resolve()
        try:
            locked_split_path.relative_to(repository)
        except ValueError as exc:
            raise ValueError("locked split escapes the repository") from exc
        locked_split = load_locked_split(locked_split_path)
        if locked_split.dataset != dataset.dataset:
            raise ValueError("locked split belongs to another dataset")
        selected_cases = select_locked_cases(
            dataset.cases,
            locked_split,
            dataset_sha256=dataset_checksum,
        )
    else:
        selected_cases = select_stratified_cases(
            dataset.cases,
            limit=config.sampling.limit,
            seed=config.sampling.seed,
            categories=config.sampling.categories,
        )

    retriever = BM25Retriever(
        dataset.documents,
        k1=config.retrieval.k1,
        b=config.retrieval.b,
    )
    retrieval_limit = max(config.top_k)
    prediction_models: list[RetrievalPrediction] = []
    prediction_rows: list[dict[str, Any]] = []
    for case in selected_cases:
        ranked = retriever.search(case.query, limit=retrieval_limit)
        retrieved_ids = [item.document_id for item in ranked]
        prediction_models.append(
            RetrievalPrediction(case_id=case.case_id, retrieved_ids=retrieved_ids)
        )
        prediction_rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "query": case.query,
                "relevant_ids": case.relevant_ids,
                "retrieved_ids": retrieved_ids,
                "scores": [item.score for item in ranked],
            }
        )

    report = evaluate_retrieval(selected_cases, prediction_models, ks=config.top_k)
    predictions_payload = _jsonl_bytes(prediction_rows)
    metrics_payload = _canonical_json(report.model_dump(mode="json")) + b"\n"
    effective_config = config.model_dump(mode="json")
    config_hash = _sha256_bytes(_canonical_json(effective_config))
    code_metadata = _source_metadata(repository)
    code_hash = _sha256_bytes(_canonical_json(code_metadata["source_sha256"]))
    run_id = _sha256_bytes(
        f"{dataset_checksum}\0{config_hash}\0{code_hash}".encode("ascii")
    )[:20]
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    category_counts = Counter(case.category for case in selected_cases)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "dataset": {
            "name": dataset.dataset,
            "adapter": config.dataset_adapter,
            "input_name": source.name,
            "sha256": dataset_checksum,
            "size_bytes": source.stat().st_size,
            "license": dataset.license,
            "attribution_url": dataset.attribution_url,
            "corpus_documents": len(dataset.documents),
            "available_cases": len(dataset.cases),
        },
        "config": {
            "sha256": config_hash,
            "effective": effective_config,
        },
        "sample": {
            "selected_cases": len(selected_cases),
            "category_counts": dict(sorted(category_counts.items())),
            "case_ids": [case.case_id for case in selected_cases],
            "locked_split": (
                {
                    "split_id": locked_split.split_id,
                    "path": config.locked_split,
                    "sha256": sha256_file(locked_split_path),
                }
                if locked_split is not None and locked_split_path is not None
                else None
            ),
        },
        "top_k": config.top_k,
        "code": code_metadata,
        "artifacts": {
            "predictions.jsonl": {
                "sha256": _sha256_bytes(predictions_payload),
                "size_bytes": len(predictions_payload),
            },
            "metrics.json": {
                "sha256": _sha256_bytes(metrics_payload),
                "size_bytes": len(metrics_payload),
            },
        },
    }
    manifest_payload = _canonical_json(manifest) + b"\n"
    _publish_run(
        target,
        {
            "predictions.jsonl": predictions_payload,
            "metrics.json": metrics_payload,
            "manifest.json": manifest_payload,
        },
    )
    return BaselineRunResult(
        output_dir=target,
        predictions_path=target / "predictions.jsonl",
        metrics_path=target / "metrics.json",
        manifest_path=target / "manifest.json",
        report=report,
        manifest=manifest,
    )


__all__ = [
    "BM25Config",
    "BM25Retriever",
    "BaselineRunResult",
    "LockedSplitManifest",
    "RAGBaselineConfig",
    "RankedDocument",
    "SamplingConfig",
    "load_baseline_config",
    "load_locked_split",
    "run_rag_baseline",
    "select_locked_cases",
    "select_stratified_cases",
    "sha256_file",
    "tokenize",
]
