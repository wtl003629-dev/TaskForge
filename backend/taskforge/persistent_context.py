"""SQLite-backed knowledge and memory stores.

SQLite owns durability and performs the first tenant/time boundary filter.
The existing domain objects and in-memory stores remain the source of truth
for ACL/scope checks and deterministic ranking.  This keeps persistence from
quietly creating a second, weaker authorisation implementation.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from threading import RLock

from .knowledge import (
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeHit,
    as_utc,
)
from .memory import (
    InMemoryMemoryStore,
    MemoryHit,
    MemoryItem,
    MemoryProvenance,
    MemoryScope,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    tenant_id       TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    text_content    TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    document_id     TEXT,
    version         TEXT NOT NULL,
    version_order   INTEGER NOT NULL CHECK (version_order >= 0),
    acl_json        TEXT NOT NULL,
    valid_from      TEXT,
    valid_until     TEXT,
    created_at      TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, chunk_id),
    CHECK (valid_from IS NULL OR valid_until IS NULL OR valid_from < valid_until)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_tenant_validity
    ON knowledge_chunks (tenant_id, valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_knowledge_tenant_document_version
    ON knowledge_chunks (tenant_id, document_id, version_order DESC, version);

CREATE TABLE IF NOT EXISTS memory_items (
    tenant_id       TEXT NOT NULL,
    memory_id       TEXT NOT NULL,
    content         TEXT NOT NULL,
    scope           TEXT NOT NULL CHECK (scope IN ('tenant', 'org', 'user', 'agent', 'task')),
    scope_id        TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    importance      REAL NOT NULL CHECK (importance >= 0.0 AND importance <= 1.0),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    expires_at      TEXT,
    tags_json       TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_tenant_expiry
    ON memory_items (tenant_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_tenant_scope
    ON memory_items (tenant_id, scope, scope_id, updated_at DESC);
"""


_KNOWLEDGE_UPSERT = """
INSERT INTO knowledge_chunks (
    tenant_id, chunk_id, text_content, source_uri, document_id, version,
    version_order, acl_json, valid_from, valid_until, created_at, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
    text_content = excluded.text_content,
    source_uri = excluded.source_uri,
    document_id = excluded.document_id,
    version = excluded.version,
    version_order = excluded.version_order,
    acl_json = excluded.acl_json,
    valid_from = excluded.valid_from,
    valid_until = excluded.valid_until,
    created_at = excluded.created_at,
    metadata_json = excluded.metadata_json
"""


_MEMORY_UPSERT = """
INSERT INTO memory_items (
    tenant_id, memory_id, content, scope, scope_id, provenance_json,
    importance, created_at, updated_at, expires_at, tags_json, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (tenant_id, memory_id) DO UPDATE SET
    content = excluded.content,
    scope = excluded.scope,
    scope_id = excluded.scope_id,
    provenance_json = excluded.provenance_json,
    importance = excluded.importance,
    created_at = excluded.created_at,
    updated_at = excluded.updated_at,
    expires_at = excluded.expires_at,
    tags_json = excluded.tags_json,
    metadata_json = excluded.metadata_json
"""


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_load(value: object, *, expected: type) -> object:
    if not isinstance(value, str):
        raise ValueError("JSON column must be text")
    parsed = json.loads(
        value,
        object_pairs_hook=_duplicate_safe_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(parsed, expected):
        raise ValueError(f"JSON value must be {expected.__name__}")
    return parsed


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _timestamp_dump(value: datetime | None) -> str | None:
    if value is None:
        return None
    return as_utc(value).isoformat(timespec="microseconds")


def _timestamp_load(value: object, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp column must be non-empty text")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return as_utc(result)


def _strict_string_list(value: object, *, name: str) -> frozenset[str]:
    parsed = _json_load(value, expected=list)
    assert isinstance(parsed, list)
    if any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ValueError(f"{name} must contain non-empty strings")
    return frozenset(item.strip() for item in parsed)


def _strict_metadata(value: object) -> Mapping[str, object]:
    parsed = _json_load(value, expected=dict)
    assert isinstance(parsed, dict)
    return parsed


def _row_to_knowledge(row: sqlite3.Row) -> KnowledgeChunk:
    version_order = row["version_order"]
    if type(version_order) is not int or version_order < 0:
        raise ValueError("version_order must be a non-negative integer")
    return KnowledgeChunk(
        chunk_id=row["chunk_id"],
        tenant_id=row["tenant_id"],
        text=row["text_content"],
        source_uri=row["source_uri"],
        document_id=row["document_id"],
        version=row["version"],
        version_order=version_order,
        acl=_strict_string_list(row["acl_json"], name="acl"),
        valid_from=_timestamp_load(row["valid_from"], optional=True),
        valid_until=_timestamp_load(row["valid_until"], optional=True),
        created_at=_timestamp_load(row["created_at"]),  # type: ignore[arg-type]
        metadata=_strict_metadata(row["metadata_json"]),
    )


def _provenance_load(value: object) -> MemoryProvenance:
    parsed = _json_load(value, expected=dict)
    assert isinstance(parsed, dict)
    allowed = {"source_type", "source_id", "source_uri", "actor_id", "observed_at", "confidence"}
    if set(parsed).difference(allowed):
        raise ValueError("provenance contains unknown fields")
    if not {"source_type", "observed_at", "confidence"}.issubset(parsed):
        raise ValueError("provenance is missing required fields")
    if not isinstance(parsed["source_type"], str) or not parsed["source_type"].strip():
        raise ValueError("provenance.source_type must be non-empty text")
    for name in ("source_type", "source_id", "source_uri", "actor_id"):
        value_for_name = parsed.get(name)
        if value_for_name is not None and not isinstance(value_for_name, str):
            raise ValueError(f"provenance.{name} must be text or null")
    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise ValueError("provenance.confidence must be a finite number")
    return MemoryProvenance(
        source_type=parsed["source_type"],  # type: ignore[arg-type]
        source_id=parsed.get("source_id"),  # type: ignore[arg-type]
        source_uri=parsed.get("source_uri"),  # type: ignore[arg-type]
        actor_id=parsed.get("actor_id"),  # type: ignore[arg-type]
        observed_at=_timestamp_load(parsed["observed_at"]),  # type: ignore[arg-type]
        confidence=float(confidence),
    )


def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
    importance = row["importance"]
    if isinstance(importance, bool) or not isinstance(importance, (int, float)) or not math.isfinite(float(importance)):
        raise ValueError("importance must be a finite number")
    return MemoryItem(
        memory_id=row["memory_id"],
        tenant_id=row["tenant_id"],
        content=row["content"],
        scope=MemoryScope(row["scope"]),
        scope_id=row["scope_id"],
        provenance=_provenance_load(row["provenance_json"]),
        importance=float(importance),
        created_at=_timestamp_load(row["created_at"]),  # type: ignore[arg-type]
        updated_at=_timestamp_load(row["updated_at"]),
        expires_at=_timestamp_load(row["expires_at"], optional=True),
        tags=_strict_string_list(row["tags_json"], name="tags"),
        metadata=_strict_metadata(row["metadata_json"]),
    )


def _knowledge_values(chunk: KnowledgeChunk) -> tuple[object, ...]:
    return (
        chunk.tenant_id,
        chunk.chunk_id,
        chunk.text,
        chunk.source_uri,
        chunk.document_id,
        chunk.version,
        chunk.version_order,
        _json_dump(sorted(chunk.acl)),
        _timestamp_dump(chunk.valid_from),
        _timestamp_dump(chunk.valid_until),
        _timestamp_dump(chunk.created_at),
        _json_dump(dict(chunk.metadata)),
    )


def _memory_values(item: MemoryItem) -> tuple[object, ...]:
    provenance = {
        "source_type": item.provenance.source_type,
        "source_id": item.provenance.source_id,
        "source_uri": item.provenance.source_uri,
        "actor_id": item.provenance.actor_id,
        "observed_at": _timestamp_dump(item.provenance.observed_at),
        "confidence": item.provenance.confidence,
    }
    return (
        item.tenant_id,
        item.memory_id,
        item.content,
        item.scope.value,
        item.scope_id,
        _json_dump(provenance),
        item.importance,
        _timestamp_dump(item.created_at),
        _timestamp_dump(item.updated_at),
        _timestamp_dump(item.expires_at),
        _json_dump(sorted(item.tags)),
        _json_dump(dict(item.metadata)),
    )


class _SQLiteStore(AbstractContextManager["_SQLiteStore"]):
    def __init__(self, database: str | Path, *, timeout: float = 5.0) -> None:
        self.database = str(database)
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.database,
            timeout=timeout,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA trusted_schema = OFF")
            if self.database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _rows(self, sql: str, parameters: tuple[object, ...]) -> list[sqlite3.Row]:
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            return list(self._connection.execute(sql, parameters).fetchall())


class SQLiteKnowledgeStore(_SQLiteStore):
    """Persistent counterpart of :class:`InMemoryKnowledgeStore`."""

    def upsert(self, chunk: KnowledgeChunk) -> None:
        self.upsert_many((chunk,))

    add = upsert

    def upsert_many(self, chunks: Iterable[KnowledgeChunk]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            with self._connection:
                for chunk in chunks:
                    if not isinstance(chunk, KnowledgeChunk):
                        raise TypeError("all records must be KnowledgeChunk instances")
                    self._connection.execute(_KNOWLEDGE_UPSERT, _knowledge_values(chunk))

    batch_upsert = upsert_many

    def replace_document_version(self, chunks: Iterable[KnowledgeChunk]) -> int:
        """Atomically replace every chunk belonging to one document version."""

        materialised = list(chunks)
        if not materialised:
            raise ValueError("at least one chunk is required")
        first = materialised[0]
        identity = (first.tenant_id, first.logical_document_id, first.version)
        if any(
            (chunk.tenant_id, chunk.logical_document_id, chunk.version) != identity
            for chunk in materialised
        ):
            raise ValueError("all chunks must belong to one tenant/document/version")
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            with self._connection:
                self._connection.execute(
                    """
                    DELETE FROM knowledge_chunks
                    WHERE tenant_id = ?
                      AND COALESCE(document_id, source_uri) = ?
                      AND version = ?
                    """,
                    identity,
                )
                for chunk in materialised:
                    self._connection.execute(_KNOWLEDGE_UPSERT, _knowledge_values(chunk))
        return len(materialised)

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> KnowledgeChunk | None:
        instant_datetime = as_utc(now)
        instant = _timestamp_dump(instant_datetime)
        rows = self._rows(
            """
            SELECT * FROM knowledge_chunks
            WHERE tenant_id = ? AND chunk_id = ?
              AND (valid_from IS NULL OR valid_from <= ?)
              AND (valid_until IS NULL OR valid_until > ?)
            """,
            (principal.tenant_id, chunk_id, instant, instant),
        )
        if not rows:
            return None
        try:
            chunk = _row_to_knowledge(rows[0])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return chunk if chunk.is_visible_to(principal, instant_datetime) else None

    def search(
        self,
        query: str,
        principal: AccessContext,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
        semantic_scores: Mapping[str, float] | None = None,
        lexical_weight: float = 0.70,
        semantic_weight: float = 0.30,
    ) -> list[KnowledgeHit]:
        if top_k <= 0:
            return []
        instant_datetime = as_utc(now)
        instant = _timestamp_dump(instant_datetime)
        rows = self._rows(
            """
            SELECT * FROM knowledge_chunks
            WHERE tenant_id = ?
              AND (valid_from IS NULL OR valid_from <= ?)
              AND (valid_until IS NULL OR valid_until > ?)
            """,
            (principal.tenant_id, instant, instant),
        )
        chunks: list[KnowledgeChunk] = []
        for row in rows:
            try:
                chunks.append(_row_to_knowledge(row))
            except (KeyError, TypeError, ValueError, OverflowError):
                # Corrupt persistence must not become model context.
                continue
        return InMemoryKnowledgeStore(chunks).search(
            query,
            principal,
            top_k=top_k,
            now=instant_datetime,
            source_uris=source_uris,
            knowledge_base_ids=knowledge_base_ids,
            latest_only=latest_only,
            semantic_scores=semantic_scores,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )


class SQLiteMemoryStore(_SQLiteStore):
    """Persistent counterpart of :class:`InMemoryMemoryStore`."""

    def remember(self, item: MemoryItem) -> None:
        self.remember_many((item,))

    upsert = remember
    add = remember

    def remember_many(self, items: Iterable[MemoryItem]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            with self._connection:
                for item in items:
                    if not isinstance(item, MemoryItem):
                        raise TypeError("all records must be MemoryItem instances")
                    self._connection.execute(_MEMORY_UPSERT, _memory_values(item))

    upsert_many = remember_many
    batch_upsert = remember_many

    def get(
        self,
        memory_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> MemoryItem | None:
        instant_datetime = as_utc(now)
        instant = _timestamp_dump(instant_datetime)
        rows = self._rows(
            """
            SELECT * FROM memory_items
            WHERE tenant_id = ? AND memory_id = ?
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (principal.tenant_id, memory_id, instant),
        )
        if not rows:
            return None
        try:
            item = _row_to_memory(rows[0])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return item if item.is_visible_to(principal, instant_datetime) else None

    def forget(
        self,
        memory_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> bool:
        """CAS-delete only a principal-owned, non-shared memory scope."""

        instant_datetime = as_utc(now)
        instant = _timestamp_dump(instant_datetime)
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            with self._connection:
                row = self._connection.execute(
                    """
                    SELECT * FROM memory_items
                    WHERE tenant_id = ? AND memory_id = ?
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (principal.tenant_id, memory_id, instant),
                ).fetchone()
                if row is None:
                    return False
                try:
                    item = _row_to_memory(row)
                except (KeyError, TypeError, ValueError, OverflowError):
                    return False
                if not item.is_deletable_by(principal, instant_datetime):
                    return False
                cursor = self._connection.execute(
                    """
                    DELETE FROM memory_items
                    WHERE tenant_id = ? AND memory_id = ?
                      AND scope = ? AND scope_id = ? AND updated_at = ?
                    """,
                    (
                        principal.tenant_id,
                        memory_id,
                        item.scope.value,
                        item.scope_id,
                        _timestamp_dump(item.updated_at),
                    ),
                )
        return cursor.rowcount == 1

    def recall(
        self,
        query: str,
        principal: AccessContext,
        *,
        scopes: Iterable[MemoryScope | str] | None = None,
        top_k: int = 5,
        now: datetime | None = None,
        include_unmatched: bool = False,
    ) -> list[MemoryHit]:
        if top_k <= 0:
            return []
        instant_datetime = as_utc(now)
        instant = _timestamp_dump(instant_datetime)
        rows = self._rows(
            """
            SELECT * FROM memory_items
            WHERE tenant_id = ?
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (principal.tenant_id, instant),
        )
        items: list[MemoryItem] = []
        for row in rows:
            try:
                items.append(_row_to_memory(row))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return InMemoryMemoryStore(items).recall(
            query,
            principal,
            scopes=scopes,
            top_k=top_k,
            now=instant_datetime,
            include_unmatched=include_unmatched,
        )


# Friendly aliases for callers that prefer ``Sqlite`` casing or a backend-
# neutral persistent name.
SqliteKnowledgeStore = SQLiteKnowledgeStore
SqliteMemoryStore = SQLiteMemoryStore
PersistentKnowledgeStore = SQLiteKnowledgeStore
PersistentMemoryStore = SQLiteMemoryStore


__all__ = [
    "PersistentKnowledgeStore",
    "PersistentMemoryStore",
    "SQLiteKnowledgeStore",
    "SQLiteMemoryStore",
    "SqliteKnowledgeStore",
    "SqliteMemoryStore",
]
