"""PostgreSQL-backed HTTP response cache for literature providers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from ...postgres_runtime import PostgresRuntime
from .base import ProviderError


class PostgresProviderCache:
    """Tenant-scoped cache matching the SQLite provider-cache contract."""

    def __init__(
        self,
        dsn: str,
        *,
        tenant_id: str = "local",
        ttl_seconds: int = 86_400,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
        tenant_resolver: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        self.tenant_id = tenant_id
        self._tenant_resolver = tenant_resolver
        self._owns_runtime = runtime is None
        self.ttl_seconds = int(ttl_seconds)
        self.runtime = runtime or PostgresRuntime(
            dsn,
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    @staticmethod
    def key(provider: str, url: str, params: Mapping[str, object] | None) -> str:
        material = json.dumps(
            [provider, url, sorted((params or {}).items())],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str, *, now: float | None = None) -> tuple[int, Any] | None:
        instant = time.time() if now is None else float(now)
        tenant_id = self._tenant()
        with self.runtime.transaction(tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT status_code, payload_json
                  FROM literature.provider_cache
                 WHERE tenant_id = %s AND cache_key = %s AND expires_at >= to_timestamp(%s)
                """,
                (tenant_id, key, instant),
            )
            row = cursor.fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        payload = _row(row, "payload_json", 1)
        try:
            if isinstance(payload, str):
                payload = json.loads(payload)
            return int(_row(row, "status_code", 0)), payload
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError("cached provider payload is invalid") from exc

    def put(
        self,
        key: str,
        provider: str,
        status_code: int,
        payload: Any,
        *,
        now: float | None = None,
    ) -> None:
        instant = time.time() if now is None else float(now)
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        tenant_id = self._tenant()
        with self.runtime.transaction(tenant_id) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO literature.provider_cache(
                    tenant_id, cache_key, provider, payload_json, status_code,
                    created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
                ON CONFLICT (tenant_id, cache_key) DO UPDATE SET
                    provider = EXCLUDED.provider,
                    payload_json = EXCLUDED.payload_json,
                    status_code = EXCLUDED.status_code,
                    created_at = EXCLUDED.created_at,
                    expires_at = EXCLUDED.expires_at
                """,
                (tenant_id, key, provider, _json(payload), int(status_code), instant, instant + self.ttl_seconds),
            )

    def _tenant(self) -> str:
        tenant_id = (self._tenant_resolver() if self._tenant_resolver else self.tenant_id).strip()
        if not tenant_id:
            raise ValueError("tenant resolver returned a blank tenant")
        return tenant_id


def _row(row: Any, key: str, index: int) -> Any:
    return row.get(key) if isinstance(row, Mapping) else row[index]


def _json(value: object) -> object:
    try:
        from psycopg.types.json import Jsonb
    except ImportError:  # pragma: no cover
        return value
    return Jsonb(value)


__all__ = ["PostgresProviderCache"]
