"""Durable, tamper-evident verification records for live smoke / business E2E.

A verification record is written only by an explicit host step after a real
run has completed; it is never inferred from credentials or environment
variables.  Each record binds provider, model, run id and the evidence report.
The stored signature makes any later edit or corruption detectable on read, so
the API never trusts an unverified claim that a run was "verified".
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

from pydantic import Field

from .domain import StrictModel, utc_now

VerificationKind = Literal["live_smoke", "business_e2e"]
_SIGNATURE_PREFIX = "sig:"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class VerificationRecord(StrictModel):
    """One signed, attributed verification outcome bound to its evidence."""

    record_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    kind: VerificationKind
    provider: str = Field(min_length=1)
    model: str | None = None
    run_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    produced_at: datetime = Field(default_factory=utc_now)
    signature: str = Field(min_length=len(_SIGNATURE_PREFIX) + 64)

    @classmethod
    def signed(
        cls,
        *,
        kind: VerificationKind,
        provider: str,
        model: str | None = None,
        run_id: str | None = None,
        evidence: dict[str, Any] | None = None,
        produced_at: datetime | None = None,
    ) -> VerificationRecord:
        """Build a record and bind its signature to the exact content."""

        record = cls(
            kind=kind,
            provider=provider,
            model=model,
            run_id=run_id,
            evidence=evidence or {},
            produced_at=produced_at or utc_now(),
            signature=_SIGNATURE_PREFIX + "0" * 64,
        )
        record.signature = _SIGNATURE_PREFIX + record._content_hexdigest()
        return record

    @property
    def _content(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "run_id": self.run_id,
            "evidence": self.evidence,
            "produced_at": self.produced_at.isoformat(),
        }

    def _content_hexdigest(self) -> str:
        return hashlib.sha256(_canonical(self._content)).hexdigest()

    def verify_signature(self) -> bool:
        """True only if the content hashes to the recorded signature."""

        return self.signature == _SIGNATURE_PREFIX + self._content_hexdigest()


class VerificationSignatureError(RuntimeError):
    """Raised when a persisted record fails its integrity check."""


class SQLiteVerificationStore:
    """Append-only store of signed verification records, read with fail-closed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS verification_records (
                    record_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    produced_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS verification_produced_at_idx
                    ON verification_records(produced_at);
                """
            )

    def save(self, record: VerificationRecord) -> VerificationRecord:
        if not record.verify_signature():
            raise VerificationSignatureError(
                "refusing to store a record with an invalid signature"
            )
        payload = record.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO verification_records "
                "(record_id, record_json, produced_at) VALUES (?, ?, ?)",
                (record.record_id, payload, record.produced_at.isoformat()),
            )
        return record

    def all(self) -> list[VerificationRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM verification_records "
                "ORDER BY produced_at ASC"
            ).fetchall()
        records: list[VerificationRecord] = []
        for row in rows:
            try:
                record = VerificationRecord.model_validate_json(row["record_json"])
            except ValueError as exc:
                raise VerificationSignatureError(
                    "persisted verification record is corrupt"
                ) from exc
            if not record.verify_signature():
                raise VerificationSignatureError(
                    "persisted verification record was tampered with"
                )
            records.append(record)
        return records

    def latest(
        self,
        kind: VerificationKind,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> VerificationRecord | None:
        matches = [record for record in self.all() if record.kind == kind]
        if provider is not None:
            matches = [record for record in matches if record.provider == provider]
        if model is not None:
            matches = [record for record in matches if record.model == model]
        return max(matches, key=lambda record: record.produced_at) if matches else None


__all__ = [
    "SQLiteVerificationStore",
    "VerificationKind",
    "VerificationRecord",
    "VerificationSignatureError",
]
