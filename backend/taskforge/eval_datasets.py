"""Pinned, license-aware download contracts for external evaluation labels.

Large third-party corpora are never vendored into the repository.  Operators
must select a catalog entry explicitly; downloads are HTTPS allowlisted,
bounded, checksum-verified, and kept under a caller-selected cache directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from .domain import StrictModel


class DatasetArtifact(StrictModel):
    filename: str = Field(min_length=1, max_length=180)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_bytes: int = Field(gt=0, le=2_000_000_000)

    @model_validator(mode="after")
    def safe_location(self) -> "DatasetArtifact":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.filename):
            raise ValueError("artifact filename must be a safe basename")
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username:
            raise ValueError("artifact URL must be credential-free HTTPS")
        return self


class DatasetSource(StrictModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    homepage: str = Field(min_length=1)
    license: str = Field(min_length=1)
    commercial_use: bool
    redistribution: Literal["catalog-only", "allowed-with-attribution"]
    automated: bool
    notes: str = ""
    artifacts: list[DatasetArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def automated_sources_have_artifacts(self) -> "DatasetSource":
        if self.automated and not self.artifacts:
            raise ValueError("automated dataset source requires an artifact")
        if urlsplit(self.homepage).scheme != "https":
            raise ValueError("dataset homepage must use HTTPS")
        return self


class DatasetCatalog(StrictModel):
    schema_version: str = "1.0"
    sources: list[DatasetSource]

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> "DatasetCatalog":
        identifiers = [source.dataset_id for source in self.sources]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("dataset source identifiers must be unique")
        return self

    def get(self, dataset_id: str) -> DatasetSource:
        for source in self.sources:
            if source.dataset_id == dataset_id:
                return source
        raise KeyError(dataset_id)


class DatasetDownloadError(RuntimeError):
    """Raised when an external artifact violates its pinned contract."""


class DownloadReceipt(StrictModel):
    dataset_id: str
    filename: str
    path: str
    sha256: str
    bytes_written: int = Field(ge=0)
    cached: bool


DEFAULT_DOWNLOAD_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
    }
)


def load_dataset_catalog(path: str | Path) -> DatasetCatalog:
    return DatasetCatalog.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            size += len(block)
            sha.update(block)
    return sha.hexdigest(), size


def download_dataset_source(
    source: DatasetSource,
    *,
    output_dir: str | Path,
    accept_noncommercial: bool = False,
    allowed_hosts: frozenset[str] = DEFAULT_DOWNLOAD_HOSTS,
    client: httpx.Client | None = None,
) -> list[DownloadReceipt]:
    if not source.automated:
        raise DatasetDownloadError(
            f"{source.dataset_id} requires manual acquisition from {source.homepage}"
        )
    if not source.commercial_use and not accept_noncommercial:
        raise DatasetDownloadError(
            f"{source.dataset_id} is non-commercial; pass explicit acceptance"
        )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(60.0),
        follow_redirects=False,
        trust_env=False,
    )
    receipts: list[DownloadReceipt] = []
    try:
        for artifact in source.artifacts:
            host = (urlsplit(artifact.url).hostname or "").casefold()
            if host not in allowed_hosts:
                raise DatasetDownloadError(f"download host is not allowlisted: {host}")
            target = (root / artifact.filename).resolve()
            if target.parent != root:
                raise DatasetDownloadError("artifact target escapes output directory")
            if target.exists():
                digest, size = _digest(target)
                if digest != artifact.sha256:
                    raise DatasetDownloadError(
                        f"existing artifact checksum mismatch: {artifact.filename}"
                    )
                receipts.append(
                    DownloadReceipt(
                        dataset_id=source.dataset_id,
                        filename=artifact.filename,
                        path=str(target),
                        sha256=digest,
                        bytes_written=size,
                        cached=True,
                    )
                )
                continue
            temporary = root / f".{artifact.filename}.part"
            if temporary.exists():
                temporary.unlink()
            sha = hashlib.sha256()
            size = 0
            try:
                with http.stream("GET", artifact.url) as response:
                    if not 200 <= response.status_code < 300:
                        raise DatasetDownloadError(
                            f"download returned HTTP {response.status_code}"
                        )
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > artifact.max_bytes:
                        raise DatasetDownloadError("artifact exceeds configured size limit")
                    with temporary.open("xb") as handle:
                        for block in response.iter_bytes(1024 * 256):
                            size += len(block)
                            if size > artifact.max_bytes:
                                raise DatasetDownloadError(
                                    "artifact exceeded configured size limit while streaming"
                                )
                            sha.update(block)
                            handle.write(block)
                digest = sha.hexdigest()
                if digest != artifact.sha256:
                    raise DatasetDownloadError(
                        f"download checksum mismatch: {artifact.filename}"
                    )
                temporary.replace(target)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            receipts.append(
                DownloadReceipt(
                    dataset_id=source.dataset_id,
                    filename=artifact.filename,
                    path=str(target),
                    sha256=digest,
                    bytes_written=size,
                    cached=False,
                )
            )
    finally:
        if owns_client:
            http.close()
    return receipts


__all__ = [
    "DEFAULT_DOWNLOAD_HOSTS",
    "DatasetArtifact",
    "DatasetCatalog",
    "DatasetDownloadError",
    "DatasetSource",
    "DownloadReceipt",
    "download_dataset_source",
    "load_dataset_catalog",
]
