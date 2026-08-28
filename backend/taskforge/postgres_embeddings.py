"""PostgreSQL/pgvector cache port for dense embedding providers.

The provider adapters retain their existing deterministic cache identity and
vector validation.  This class only replaces the storage transport, allowing
the same model and chunking code to use ``vector.embedding_cache`` when the
host selects PostgreSQL.  It never creates schema objects; migrations own
that responsibility.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .postgres_runtime import PostgresRuntime


class PostgresEmbeddingCache:
    """Tenant-scoped cache implementing the provider cache transport."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        tenant_id: str = "local",
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
        tenant_resolver: Callable[[], str] | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        if runtime is None and not dsn:
            raise ValueError("dsn or runtime is required")
        self.tenant_id = tenant_id
        self._tenant_resolver = tenant_resolver
        self.runtime = runtime or PostgresRuntime(
            dsn or "",
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        self._owns_runtime = runtime is None

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    def load(
        self,
        *,
        model_name: str,
        identities: Sequence[tuple[str, str]],
        embedding_kind: str,
        dimension: int,
    ) -> dict[str, list[float]]:
        if not identities:
            return {}
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        expected = dict(identities)
        loaded: dict[str, list[float]] = {}
        keys = list(expected)
        tenant_id = self._tenant()
        with self.runtime.transaction(tenant_id) as (_, cursor):
            for offset in range(0, len(keys), 400):
                batch = keys[offset : offset + 400]
                placeholders = ", ".join(["%s"] * len(batch))
                cursor.execute(
                    "SELECT cache_key, model_name, embedding_kind, text_sha256, "
                    "dimension, embedding::text AS embedding "
                    "FROM vector.embedding_cache "
                    f"WHERE tenant_id = %s AND cache_key IN ({placeholders})",
                    (tenant_id, *batch),
                )
                for row in cursor.fetchall():
                    values = _row_values(row)
                    cache_key, actual_model, actual_kind, text_sha256, actual_dimension, embedding = values
                    if (
                        actual_model != model_name
                        or actual_kind != embedding_kind
                        or text_sha256 != expected.get(cache_key)
                        or int(actual_dimension) != dimension
                    ):
                        raise ValueError("PostgreSQL embedding metadata does not match the request")
                    loaded[str(cache_key)] = _vector_values(embedding, dimension)
        return loaded

    def store(
        self,
        rows: Iterable[tuple[object, ...]],
    ) -> None:
        values = list(rows)
        if not values:
            return
        if any(int(row[4]) <= 0 for row in values):
            raise ValueError("embedding cache dimensions must be positive")
        tenant_id = self._tenant()
        with self.runtime.transaction(tenant_id) as (_, cursor):
            cursor.executemany(
                "INSERT INTO vector.embedding_cache ("
                "tenant_id, cache_key, model_name, embedding_kind, text_sha256, "
                "dimension, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
                "ON CONFLICT (tenant_id, cache_key) DO NOTHING",
                [
                    (
                        tenant_id,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        int(row[4]),
                        _vector_literal(row[5], int(row[4])),
                    )
                    for row in values
                ],
            )

    def _tenant(self) -> str:
        tenant_id = (self._tenant_resolver() if self._tenant_resolver else self.tenant_id).strip()
        if not tenant_id:
            raise ValueError("tenant resolver returned a blank tenant")
        return tenant_id


def _row_values(row: Any) -> tuple[Any, ...]:
    if isinstance(row, dict):
        return (
            row["cache_key"],
            row["model_name"],
            row["embedding_kind"],
            row["text_sha256"],
            row["dimension"],
            row["embedding"],
        )
    return tuple(row)


def _vector_values(value: object, dimension: int) -> list[float]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw.startswith("[") or not raw.endswith("]"):
            raise ValueError("PostgreSQL vector has invalid brackets")
        parts = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        values = [float(item) for item in parts]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [float(item) for item in value]
    else:
        raise ValueError("PostgreSQL vector has an unsupported type")
    if len(values) != dimension or any(not math.isfinite(item) for item in values):
        raise ValueError("PostgreSQL vector dimension or values are invalid")
    return values


def _vector_literal(value: object, dimension: int) -> str:
    if isinstance(value, (bytes, bytearray)):
        expected_bytes = dimension * 4
        if len(value) != expected_bytes:
            raise ValueError("embedding cache blob dimension is invalid")
        values = struct.unpack(f"<{dimension}f", value)
    else:
        values = _vector_values(value, dimension)
    return "[" + ",".join(format(float(item), ".9g") for item in values) + "]"


__all__ = ["PostgresEmbeddingCache"]
