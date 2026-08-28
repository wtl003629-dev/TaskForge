"""PostgreSQL backend for signed verification records.

The verification payload remains an opaque, validated JSON document.  The
database owns tenant isolation and timestamp ordering; the signature is still
checked by the domain model before both writes and reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .postgres_runtime import PostgresRuntime
from .verification import (
    VerificationKind,
    VerificationRecord,
    VerificationSignatureError,
)


class PostgresVerificationStore:
    """Pooled, tenant-scoped store for immutable verification evidence."""

    def __init__(
        self,
        dsn: str,
        *,
        tenant_id: str = "local",
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        self.tenant_id = tenant_id
        self._owns_runtime = runtime is None
        self.runtime = runtime or PostgresRuntime(
            dsn,
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    @staticmethod
    def _json_payload(record: VerificationRecord) -> Any:
        payload = record.model_dump(mode="json")
        try:
            from psycopg.types.json import Json
        except ImportError:  # pragma: no cover - real runtime imports psycopg first
            # Test doubles and static migration checks may inject a runtime
            # without installing the optional adapter package.
            return payload
        return Json(payload)

    def save(
        self,
        record: VerificationRecord,
        *,
        tenant_id: str | None = None,
    ) -> VerificationRecord:
        if not record.verify_signature():
            raise VerificationSignatureError(
                "refusing to store a record with an invalid signature"
            )
        current_tenant = tenant_id or self.tenant_id
        if not current_tenant.strip():
            raise ValueError("tenant_id is required")
        produced_at = _as_utc(record.produced_at)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO verification.verification_records(
                    tenant_id, record_id, record_json, produced_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, record_id) DO UPDATE SET
                    record_json = EXCLUDED.record_json,
                    produced_at = EXCLUDED.produced_at
                """,
                (
                    current_tenant,
                    record.record_id,
                    self._json_payload(record),
                    produced_at,
                ),
            )
        return record

    def all(self, *, tenant_id: str | None = None) -> list[VerificationRecord]:
        current_tenant = tenant_id or self.tenant_id
        if not current_tenant.strip():
            raise ValueError("tenant_id is required")
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                """
                SELECT record_json
                  FROM verification.verification_records
                 WHERE tenant_id = %s
                 ORDER BY produced_at ASC, record_id ASC
                """,
                (current_tenant,),
            )
            rows = cursor.fetchall()
        return [_validate_row(row) for row in rows]

    def latest(
        self,
        kind: VerificationKind,
        *,
        provider: str | None = None,
        model: str | None = None,
        tenant_id: str | None = None,
    ) -> VerificationRecord | None:
        matches = [record for record in self.all(tenant_id=tenant_id) if record.kind == kind]
        if provider is not None:
            matches = [record for record in matches if record.provider == provider]
        if model is not None:
            matches = [record for record in matches if record.model == model]
        return max(matches, key=lambda record: record.produced_at) if matches else None


def _validate_row(row: Any) -> VerificationRecord:
    """Validate both psycopg tuple rows and dict-row test doubles."""

    payload = row.get("record_json") if isinstance(row, dict) else row[0]
    try:
        if isinstance(payload, str):
            record = VerificationRecord.model_validate_json(payload)
        else:
            record = VerificationRecord.model_validate(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationSignatureError(
            "persisted verification record is corrupt"
        ) from exc
    if not record.verify_signature():
        raise VerificationSignatureError(
            "persisted verification record was tampered with"
        )
    return record


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


__all__ = ["PostgresVerificationStore"]
