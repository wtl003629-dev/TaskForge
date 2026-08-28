"""Read-only SQLite/NumPy reference helpers for the pgvector gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np

DIMENSION = 1_024
BAILIAN_MODEL_ID = "aliyun-bailian|text-embedding-v4|dense-v1|1024"
APPLICATION_MODEL_NAME = "text-embedding-v4"
_VERSION_PARTS = re.compile(r"([0-9]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _version_key(version_order: int, version: str) -> tuple[int, tuple[tuple[int, object], ...]]:
    parts: list[tuple[int, object]] = []
    for part in _VERSION_PARTS.split(version.casefold()):
        if part:
            parts.append((0, int(part)) if part.isdigit() else (1, part))
    return version_order, tuple(parts)


def _is_visible(row: sqlite3.Row, *, tenant_id: str, acl_principals: set[str]) -> bool:
    if str(row["tenant_id"]) != tenant_id:
        return False
    acl = json.loads(str(row["acl_json"]))
    if not isinstance(acl, list) or not acl or not acl_principals.intersection(acl):
        return False
    # The current authoritative corpus has open-ended validity. Keep the
    # same boundary semantics as KnowledgeChunk for future refreshes.
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for name, lower in (("valid_from", now), ("valid_until", now)):
        value = row[name]
        if value is None:
            continue
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if name == "valid_from" and parsed > lower:
            return False
        if name == "valid_until" and parsed <= lower:
            return False
    return True


def sqlite_document_vectors(
    source_root: Path,
    *,
    tenant_id: str,
    acl_principals: list[str],
) -> list[tuple[str, np.ndarray]]:
    """Return the authorized, latest-only vector corpus from immutable SQLite."""

    context_path = source_root / "context.sqlite3"
    cache_path = source_root / "embeddings-bailian-v4-1024.sqlite3"
    with _read_only(context_path) as context:
        chunks = [
            row
            for row in context.execute(
                "SELECT chunk_id, tenant_id, text_content, document_id, source_uri, "
                "version, version_order, acl_json, valid_from, valid_until "
                "FROM knowledge_chunks"
            )
            if _is_visible(row, tenant_id=tenant_id, acl_principals=set(acl_principals))
        ]
    latest: dict[str, tuple[int, tuple[tuple[int, object], ...]]] = {}
    for row in chunks:
        logical_id = str(row["document_id"] or row["source_uri"])
        key = _version_key(int(row["version_order"]), str(row["version"]))
        if key > latest.get(logical_id, (-1, ())):
            latest[logical_id] = key
    chunks = [
        row
        for row in chunks
        if _version_key(int(row["version_order"]), str(row["version"]))
        == latest[str(row["document_id"] or row["source_uri"])]
    ]
    hash_to_chunk: dict[str, str] = {}
    for row in chunks:
        text_hash = hashlib.sha256(str(row["text_content"]).encode("utf-8")).hexdigest()
        previous = hash_to_chunk.setdefault(text_hash, str(row["chunk_id"]))
        if previous != str(row["chunk_id"]):
            raise ValueError(f"text hash maps to multiple visible chunks: {text_hash}")

    result: list[tuple[str, np.ndarray]] = []
    with _read_only(cache_path) as cache:
        rows = cache.execute(
            "SELECT text_sha256, dimension, vector FROM "
            "embeddings_bailian_v4_1024_v1 "
            "WHERE embedding_kind = 'document' AND model_name = ? "
            "ORDER BY cache_key",
            (BAILIAN_MODEL_ID,),
        )
        for row in rows:
            chunk_id = hash_to_chunk.get(str(row["text_sha256"]))
            if chunk_id is None:
                continue
            if int(row["dimension"]) != DIMENSION or len(row["vector"]) != DIMENSION * 4:
                raise ValueError(f"invalid document vector for {chunk_id}")
            values = struct.unpack("<1024f", row["vector"])
            vector = np.asarray(values, dtype=np.float32)
            if not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError(f"invalid document vector values for {chunk_id}")
            result.append((chunk_id, vector))
    if not result:
        raise ValueError("SQLite reference corpus has no authorized document vectors")
    result.sort(key=lambda item: item[0])
    return result


def numpy_rank(
    query: list[float] | np.ndarray,
    documents: list[tuple[str, np.ndarray]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    query_array = np.asarray(query, dtype=np.float32)
    matrix = np.vstack([vector for _, vector in documents])
    query_norm = float(np.linalg.norm(query_array))
    document_norms = np.linalg.norm(matrix, axis=1)
    scores = (matrix @ query_array) / (document_norms * query_norm)
    ranked = sorted(
        range(len(documents)),
        key=lambda index: (-float(scores[index]), documents[index][0]),
    )[:top_k]
    return [
        {"chunk_id": documents[index][0], "score": float(scores[index])}
        for index in ranked
    ]


def frozen_query_vectors(cache_path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("query limit must be positive")
    result: list[dict[str, Any]] = []
    with _read_only(cache_path) as cache:
        for row in cache.execute(
            "SELECT cache_key, text_sha256, model_name, dimension, vector "
            "FROM embeddings_bailian_v4_1024_v1 "
            "WHERE embedding_kind = 'query' AND model_name = ? "
            "ORDER BY cache_key LIMIT ?",
            (BAILIAN_MODEL_ID, limit),
        ):
            if int(row["dimension"]) != DIMENSION or len(row["vector"]) != DIMENSION * 4:
                raise ValueError(f"invalid query vector for {row['cache_key']}")
            values = list(struct.unpack("<1024f", row["vector"]))
            if any(not math.isfinite(value) for value in values) or not any(values):
                raise ValueError(f"invalid query vector values for {row['cache_key']}")
            result.append(
                {
                    "id": str(row["cache_key"]),
                    "text_sha256": str(row["text_sha256"]),
                    "model": str(row["model_name"]),
                    "vector": values,
                }
            )
    if not result:
        raise ValueError("SQLite embedding cache has no query vectors")
    return result
