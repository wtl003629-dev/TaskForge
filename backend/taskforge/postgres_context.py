"""PostgreSQL persistence primitives for tenant-scoped context.

This module deliberately does not present PostgreSQL as a search engine.  It
stores knowledge and memory durably and returns bounded, already-authorised
candidates for a separate lexical/vector retrieval layer.  Every operation
sets the transaction-local tenant used by PostgreSQL RLS *and* repeats the
tenant, user, conversation, ACL, time, and memory-scope predicates in SQL.

``psycopg`` is imported only by :meth:`PostgresContextRepository.connect`, so
the explicit SQLite compatibility installation remains dependency-free. Tests
may inject a strict DB-API/psycopg-shaped connection without weakening the
production path.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

from .knowledge import AccessContext, KnowledgeChunk, as_utc
from .memory import MemoryItem, MemoryProvenance, MemoryScope

MAX_JSON_BYTES = 1_000_000
MAX_JSON_DEPTH = 50
MAX_CANDIDATES = 10_000


class PostgresContextError(RuntimeError):
    """Base error for the PostgreSQL context backend."""


class PostgresDependencyError(PostgresContextError):
    """Raised when a DSN is used without the optional psycopg dependency."""


class PostgresDataError(PostgresContextError):
    """Raised when persisted data violates the strict context contract."""


def _safe_text(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in cleaned:
        raise ValueError(f"{name} contains a NUL character")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in cleaned):
        raise ValueError(f"{name} contains an invalid Unicode surrogate")
    return cleaned


def _validate_json_value(
    value: object,
    *,
    depth: int = 0,
    active: set[int] | None = None,
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON exceeds maximum depth {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool) or type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("JSON string contains a NUL character")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON string contains an invalid Unicode surrogate")
        return

    if active is None:
        active = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("JSON value contains a cycle")
        active.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                if "\x00" in key:
                    raise ValueError("JSON key contains a NUL character")
                if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                    raise ValueError("JSON key contains an invalid Unicode surrogate")
                _validate_json_value(child, depth=depth + 1, active=active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("JSON value contains a cycle")
        active.add(identity)
        try:
            for child in value:
                _validate_json_value(child, depth=depth + 1, active=active)
        finally:
            active.remove(identity)
        return
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_dump(value: object) -> str:
    _validate_json_value(value)
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds maximum size {MAX_JSON_BYTES} bytes")
    return payload


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
    if isinstance(value, str):
        parsed = json.loads(
            value,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    else:
        parsed = value
    if not isinstance(parsed, expected):
        raise ValueError(f"JSON value must be {expected.__name__}")
    _validate_json_value(parsed)
    # Round-trip through the bounded canonical serializer so decoded JSONB
    # values receive the same byte-size limit as writes.
    _json_dump(parsed)
    return parsed


def _timestamp_load(value: object, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value:
        try:
            result = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid timestamp") from exc
    else:
        raise ValueError("timestamp must be an aware datetime")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return as_utc(result)


def _timestamp_param(value: datetime | None) -> datetime | None:
    return None if value is None else as_utc(value)


def _bounded_limit(value: int) -> int:
    if type(value) is not int or value <= 0 or value > MAX_CANDIDATES:
        raise ValueError(f"candidate_limit must be between 1 and {MAX_CANDIDATES}")
    return value


def _optional_text_array(values: Iterable[str] | None, name: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} filter must be an iterable of strings")
    cleaned = sorted({_safe_text(value, name) for value in values})
    return [value for value in cleaned if value is not None]


@dataclass(frozen=True, slots=True)
class PostgresContextAccess:
    """Trusted host identity for one PostgreSQL context transaction.

    ``conversation_id`` maps to the existing ``task`` memory scope while ACLs
    recognise both ``conversation:<id>`` and the backwards-compatible
    ``task:<id>`` token.
    """

    tenant_id: str
    user_id: str
    conversation_id: str
    org_id: str | None = None
    agent_id: str | None = None
    acl_principals: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("tenant_id", "user_id", "conversation_id"):
            cleaned = _safe_text(getattr(self, name), name)
            assert cleaned is not None
            if len(cleaned) > 256:
                raise ValueError(f"{name} is too long")
            object.__setattr__(self, name, cleaned)
        for name in ("org_id", "agent_id"):
            value = getattr(self, name)
            if value is not None:
                cleaned = _safe_text(value, name)
                assert cleaned is not None
                if len(cleaned) > 256:
                    raise ValueError(f"{name} is too long")
                object.__setattr__(self, name, cleaned)
        if isinstance(self.acl_principals, (str, bytes)):
            raise TypeError("acl_principals must be an iterable of strings")
        principals: set[str] = set()
        for principal in self.acl_principals:
            cleaned = _safe_text(principal, "acl_principal")
            assert cleaned is not None
            if len(cleaned) > 256:
                raise ValueError("acl_principal is too long")
            principals.add(cleaned)
        object.__setattr__(self, "acl_principals", frozenset(principals))

    @property
    def acl_tokens(self) -> frozenset[str]:
        tokens = {
            "tenant",
            f"tenant:{self.tenant_id}",
            f"user:{self.user_id}",
            f"conversation:{self.conversation_id}",
            f"task:{self.conversation_id}",
        }
        if self.org_id:
            tokens.add(f"org:{self.org_id}")
        if self.agent_id:
            tokens.add(f"agent:{self.agent_id}")
        tokens.update(self.acl_principals)
        return frozenset(tokens)

    def as_domain_access(self) -> AccessContext:
        roles = frozenset(
            token.removeprefix("role:")
            for token in self.acl_principals
            if token.startswith("role:") and token != "role:"
        )
        groups = frozenset(
            token.removeprefix("group:")
            for token in self.acl_principals
            if token.startswith("group:") and token != "group:"
        )
        return AccessContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            org_id=self.org_id,
            agent_id=self.agent_id,
            task_id=self.conversation_id,
            roles=roles,
            groups=groups,
        )


_SET_TENANT_SQL = """
SELECT set_config('taskforge.tenant_id', %(tenant_id)s, true)
"""


_KNOWLEDGE_UPSERT_SQL = """
INSERT INTO taskforge.knowledge_chunks AS existing (
    tenant_id, chunk_id, text_content, source_uri, document_id, version,
    version_order, acl_json, valid_from, valid_until, created_at, metadata_json
) VALUES (
    %(tenant_id)s, %(chunk_id)s, %(text_content)s, %(source_uri)s,
    %(document_id)s, %(version)s, %(version_order)s,
    %(acl_json)s::jsonb, %(valid_from)s, %(valid_until)s, %(created_at)s,
    %(metadata_json)s::jsonb
)
ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
    text_content = EXCLUDED.text_content,
    source_uri = EXCLUDED.source_uri,
    document_id = EXCLUDED.document_id,
    version = EXCLUDED.version,
    version_order = EXCLUDED.version_order,
    acl_json = EXCLUDED.acl_json,
    valid_from = EXCLUDED.valid_from,
    valid_until = EXCLUDED.valid_until,
    created_at = EXCLUDED.created_at,
    metadata_json = EXCLUDED.metadata_json
WHERE existing.tenant_id = %(tenant_id)s
  AND EXCLUDED.version_order >= existing.version_order
"""


_KNOWLEDGE_CANDIDATES_SQL = """
WITH authorised AS (
    SELECT
        kc.*,
        MAX(kc.version_order) OVER (
            PARTITION BY COALESCE(kc.document_id, kc.source_uri)
        ) AS latest_version_order
    FROM taskforge.knowledge_chunks AS kc
    WHERE kc.tenant_id = %(tenant_id)s
      AND (kc.valid_from IS NULL OR kc.valid_from <= %(now)s)
      AND (kc.valid_until IS NULL OR kc.valid_until > %(now)s)
      AND (
            kc.acl_json ? 'tenant'
         OR kc.acl_json ? ('tenant:' || %(tenant_id)s::text)
         OR kc.acl_json ? ('user:' || %(user_id)s::text)
         OR kc.acl_json ? ('conversation:' || %(conversation_id)s::text)
         OR kc.acl_json ? ('task:' || %(conversation_id)s::text)
         OR (%(org_id)s::text IS NOT NULL AND kc.acl_json ? ('org:' || %(org_id)s::text))
         OR (%(agent_id)s::text IS NOT NULL AND kc.acl_json ? ('agent:' || %(agent_id)s::text))
         OR kc.acl_json ?| %(additional_acl)s::text[]
      )
      AND (
            %(source_uris)s::text[] IS NULL
         OR kc.source_uri = ANY (%(source_uris)s::text[])
         OR COALESCE(kc.document_id, kc.source_uri) = ANY (%(source_uris)s::text[])
      )
      AND (
            %(knowledge_base_ids)s::text[] IS NULL
         OR kc.metadata_json ->> 'knowledge_base_id'
            = ANY (%(knowledge_base_ids)s::text[])
      )
)
SELECT
    tenant_id, chunk_id, text_content, source_uri, document_id, version,
    version_order, acl_json, valid_from, valid_until, created_at, metadata_json
FROM authorised
WHERE (%(latest_only)s = false OR version_order = latest_version_order)
ORDER BY version_order DESC, created_at DESC, chunk_id ASC
LIMIT %(candidate_limit)s
"""


_MEMORY_UPSERT_SQL = """
INSERT INTO taskforge.memory_items AS existing (
    tenant_id, memory_id, content, scope, scope_id, provenance_json,
    importance, created_at, updated_at, expires_at, tags_json, metadata_json
) VALUES (
    %(tenant_id)s, %(memory_id)s, %(content)s, %(scope)s, %(scope_id)s,
    %(provenance_json)s::jsonb, %(importance)s, %(created_at)s,
    %(updated_at)s, %(expires_at)s, %(tags_json)s::jsonb,
    %(metadata_json)s::jsonb
)
ON CONFLICT (tenant_id, memory_id) DO UPDATE SET
    content = EXCLUDED.content,
    scope = EXCLUDED.scope,
    scope_id = EXCLUDED.scope_id,
    provenance_json = EXCLUDED.provenance_json,
    importance = EXCLUDED.importance,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at,
    expires_at = EXCLUDED.expires_at,
    tags_json = EXCLUDED.tags_json,
    metadata_json = EXCLUDED.metadata_json
WHERE existing.tenant_id = %(tenant_id)s
  AND EXCLUDED.updated_at >= existing.updated_at
"""


_MEMORY_CANDIDATES_SQL = """
SELECT
    tenant_id, memory_id, content, scope, scope_id, provenance_json,
    importance, created_at, updated_at, expires_at, tags_json, metadata_json
FROM taskforge.memory_items AS mi
WHERE mi.tenant_id = %(tenant_id)s
  AND (mi.expires_at IS NULL OR mi.expires_at > %(now)s)
  AND (
        (mi.scope = 'tenant' AND mi.scope_id = %(tenant_id)s)
     OR (mi.scope = 'user' AND mi.scope_id = %(user_id)s)
     OR (mi.scope = 'task' AND mi.scope_id = %(conversation_id)s)
     OR (
            %(org_id)s::text IS NOT NULL
        AND mi.scope = 'org'
        AND mi.scope_id = %(org_id)s
     )
     OR (
            %(agent_id)s::text IS NOT NULL
        AND mi.scope = 'agent'
        AND mi.scope_id = %(agent_id)s
     )
  )
  AND (%(scopes)s::text[] IS NULL OR mi.scope = ANY (%(scopes)s::text[]))
ORDER BY mi.updated_at DESC, mi.memory_id ASC
LIMIT %(candidate_limit)s
"""


def _knowledge_parameters(chunk: KnowledgeChunk) -> dict[str, object]:
    for name in ("tenant_id", "chunk_id", "text", "source_uri", "version"):
        _safe_text(getattr(chunk, name), name)
    if chunk.document_id is not None:
        _safe_text(chunk.document_id, "document_id")
    return {
        "tenant_id": chunk.tenant_id,
        "chunk_id": chunk.chunk_id,
        "text_content": chunk.text,
        "source_uri": chunk.source_uri,
        "document_id": chunk.document_id,
        "version": chunk.version,
        "version_order": chunk.version_order,
        "acl_json": _json_dump(sorted(chunk.acl)),
        "valid_from": _timestamp_param(chunk.valid_from),
        "valid_until": _timestamp_param(chunk.valid_until),
        "created_at": _timestamp_param(chunk.created_at),
        "metadata_json": _json_dump(dict(chunk.metadata)),
    }


def _memory_parameters(item: MemoryItem) -> dict[str, object]:
    for name in ("tenant_id", "memory_id", "content", "scope_id"):
        _safe_text(getattr(item, name), name)
    source_type = _safe_text(item.provenance.source_type, "provenance.source_type")
    optional_provenance: dict[str, str | None] = {}
    for name in ("source_id", "source_uri", "actor_id"):
        value = getattr(item.provenance, name)
        optional_provenance[name] = (
            None if value is None else _safe_text(value, f"provenance.{name}")
        )
    provenance = {
        "source_type": source_type,
        **optional_provenance,
        "observed_at": item.provenance.observed_at.isoformat(timespec="microseconds"),
        "confidence": item.provenance.confidence,
    }
    return {
        "tenant_id": item.tenant_id,
        "memory_id": item.memory_id,
        "content": item.content,
        "scope": item.scope.value,
        "scope_id": item.scope_id,
        "provenance_json": _json_dump(provenance),
        "importance": item.importance,
        "created_at": _timestamp_param(item.created_at),
        "updated_at": _timestamp_param(item.updated_at),
        "expires_at": _timestamp_param(item.expires_at),
        "tags_json": _json_dump(sorted(item.tags)),
        "metadata_json": _json_dump(dict(item.metadata)),
    }


def _row_mapping(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise PostgresDataError("PostgreSQL context rows must use psycopg dict_row")
    return row


def _row_to_knowledge(row: object) -> KnowledgeChunk:
    values = _row_mapping(row)
    acl = _json_load(values["acl_json"], expected=list)
    metadata = _json_load(values["metadata_json"], expected=dict)
    assert isinstance(acl, list) and isinstance(metadata, dict)
    if any(not isinstance(token, str) or not token.strip() for token in acl):
        raise ValueError("knowledge ACL must contain non-empty strings")
    version_order = values["version_order"]
    if type(version_order) is not int or version_order < 0:
        raise ValueError("version_order must be a non-negative integer")
    chunk_id = _safe_text(values["chunk_id"], "chunk_id")
    tenant_id = _safe_text(values["tenant_id"], "tenant_id")
    text_content = _safe_text(values["text_content"], "text_content")
    source_uri = _safe_text(values["source_uri"], "source_uri")
    document_id = _safe_text(values["document_id"], "document_id", optional=True)
    version = _safe_text(values["version"], "version")
    assert chunk_id is not None
    assert tenant_id is not None
    assert text_content is not None
    assert source_uri is not None
    assert version is not None
    return KnowledgeChunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        text=text_content,
        source_uri=source_uri,
        document_id=document_id,
        version=version,
        version_order=version_order,
        acl=frozenset(token.strip() for token in acl),
        valid_from=_timestamp_load(values["valid_from"], optional=True),
        valid_until=_timestamp_load(values["valid_until"], optional=True),
        created_at=_timestamp_load(values["created_at"]),  # type: ignore[arg-type]
        metadata=metadata,
    )


def _row_to_memory(row: object) -> MemoryItem:
    values = _row_mapping(row)
    provenance_json = _json_load(values["provenance_json"], expected=dict)
    tags_json = _json_load(values["tags_json"], expected=list)
    metadata_json = _json_load(values["metadata_json"], expected=dict)
    assert isinstance(provenance_json, dict)
    assert isinstance(tags_json, list)
    assert isinstance(metadata_json, dict)
    if any(not isinstance(tag, str) or not tag.strip() for tag in tags_json):
        raise ValueError("memory tags must contain non-empty strings")
    required = {"source_type", "observed_at", "confidence"}
    allowed = required | {"source_id", "source_uri", "actor_id"}
    if not required.issubset(provenance_json) or set(provenance_json).difference(allowed):
        raise ValueError("invalid memory provenance fields")
    confidence = provenance_json["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise ValueError("memory provenance confidence must be finite")
    importance = values["importance"]
    if isinstance(importance, bool) or not isinstance(importance, (int, float)) or not math.isfinite(float(importance)):
        raise ValueError("memory importance must be finite")
    memory_id = _safe_text(values["memory_id"], "memory_id")
    tenant_id = _safe_text(values["tenant_id"], "tenant_id")
    content = _safe_text(values["content"], "content")
    scope = _safe_text(values["scope"], "scope")
    scope_id = _safe_text(values["scope_id"], "scope_id")
    source_type = _safe_text(provenance_json["source_type"], "provenance.source_type")
    for optional_name in ("source_id", "source_uri", "actor_id"):
        optional_value = provenance_json.get(optional_name)
        if optional_value is not None:
            _safe_text(optional_value, f"provenance.{optional_name}")
    assert memory_id is not None
    assert tenant_id is not None
    assert content is not None
    assert scope is not None
    assert scope_id is not None
    assert source_type is not None
    return MemoryItem(
        memory_id=memory_id,
        tenant_id=tenant_id,
        content=content,
        scope=scope,
        scope_id=scope_id,
        provenance=MemoryProvenance(
            source_type=source_type,
            source_id=provenance_json.get("source_id"),  # type: ignore[arg-type]
            source_uri=provenance_json.get("source_uri"),  # type: ignore[arg-type]
            actor_id=provenance_json.get("actor_id"),  # type: ignore[arg-type]
            observed_at=_timestamp_load(provenance_json["observed_at"]),  # type: ignore[arg-type]
            confidence=float(confidence),
        ),
        importance=float(importance),
        created_at=_timestamp_load(values["created_at"]),  # type: ignore[arg-type]
        updated_at=_timestamp_load(values["updated_at"]),
        expires_at=_timestamp_load(values["expires_at"], optional=True),
        tags=frozenset(tag.strip() for tag in tags_json),
        metadata=metadata_json,
    )


def _knowledge_visible(chunk: KnowledgeChunk, access: PostgresContextAccess, now: datetime) -> bool:
    return (
        chunk.tenant_id == access.tenant_id
        and chunk.is_valid_at(now)
        and bool(chunk.acl.intersection(access.acl_tokens))
    )


def _memory_visible(item: MemoryItem, access: PostgresContextAccess, now: datetime) -> bool:
    return item.is_visible_to(access.as_domain_access(), now)


class PostgresContextRepository(AbstractContextManager["PostgresContextRepository"]):
    """Synchronous psycopg3-style context repository.

    Injected connections are caller-owned by default.  Connections created by
    :meth:`connect` are repository-owned and closed by ``close``/``with``.
    Each public operation owns a database transaction, installs a trusted
    transaction-local RLS tenant, and never interpolates identity into SQL.
    """

    def __init__(
        self,
        connection: object | None = None,
        *,
        owns_connection: bool = False,
        pool: object | None = None,
        owns_pool: bool = True,
    ) -> None:
        if (connection is None) == (pool is None):
            raise ValueError("exactly one of connection or pool is required")
        if connection is not None:
            if not callable(getattr(connection, "cursor", None)):
                raise TypeError("connection must provide cursor()")
            if not callable(getattr(connection, "transaction", None)):
                raise TypeError("connection must provide psycopg transaction()")
        if pool is not None and not callable(getattr(pool, "connection", None)):
            raise TypeError("pool must provide connection()")
        self._connection = connection
        self._pool = pool
        self._owns_connection = bool(owns_connection)
        self._owns_pool = bool(owns_pool)
        self._closed = False
        self._lock = RLock()

    @classmethod
    def connect(
        cls,
        dsn: str,
        *,
        connect_timeout: int = 5,
        application_name: str = "taskforge-context",
        ) -> PostgresContextRepository:
        cleaned_dsn = _safe_text(dsn, "dsn")
        cleaned_application = _safe_text(application_name, "application_name")
        if type(connect_timeout) is not int or not 1 <= connect_timeout <= 60:
            raise ValueError("connect_timeout must be between 1 and 60 seconds")
        try:
            psycopg = importlib.import_module("psycopg")
            rows = importlib.import_module("psycopg.rows")
        except (ImportError, ModuleNotFoundError) as exc:
            raise PostgresDependencyError(
                "PostgreSQL support requires the PostgreSQL dependencies "
                "(pip install taskforge-agent)"
            ) from exc
        connection = psycopg.connect(
            cleaned_dsn,
            autocommit=False,
            row_factory=rows.dict_row,
            connect_timeout=connect_timeout,
            application_name=cleaned_application,
        )
        return cls(connection, owns_connection=True)

    @classmethod
    def connect_pool(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout: int = 5,
        application_name: str = "taskforge-context",
    ) -> PostgresContextRepository:
        """Open a psycopg connection pool for concurrent API/worker access."""

        cleaned_dsn = _safe_text(dsn, "dsn")
        cleaned_application = _safe_text(application_name, "application_name")
        if type(min_size) is not int or type(max_size) is not int or not 1 <= min_size <= max_size <= 64:
            raise ValueError("pool sizes must satisfy 1 <= min_size <= max_size <= 64")
        if type(connect_timeout) is not int or not 1 <= connect_timeout <= 60:
            raise ValueError("connect_timeout must be between 1 and 60 seconds")
        try:
            pool_module = importlib.import_module("psycopg_pool")
            rows = importlib.import_module("psycopg.rows")
        except (ImportError, ModuleNotFoundError) as exc:
            raise PostgresDependencyError(
                "PostgreSQL pool support requires the PostgreSQL dependencies "
                "(pip install taskforge-agent)"
            ) from exc
        pool = pool_module.ConnectionPool(
            conninfo=cleaned_dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=connect_timeout,
            kwargs={
                "autocommit": False,
                "row_factory": rows.dict_row,
                "connect_timeout": connect_timeout,
                "application_name": cleaned_application,
            },
            open=False,
        )
        pool.open(wait=True)
        return cls(pool=pool, owns_pool=True)

    @property
    def owns_connection(self) -> bool:
        return self._owns_connection

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._pool is not None and self._owns_pool:
                self._pool.close()  # type: ignore[attr-defined]
            elif self._owns_connection and self._connection is not None:
                self._connection.close()
            self._closed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @contextmanager
    def transaction(self, access: PostgresContextAccess) -> Iterator[Any]:
        """Expose one tenant-scoped transaction to backend-specific adapters."""

        with self._transaction(access) as cursor:
            yield cursor

    @contextmanager
    def _transaction(self, access: PostgresContextAccess) -> Iterator[Any]:
        if not isinstance(access, PostgresContextAccess):
            raise TypeError("access must be PostgresContextAccess")
        with self._lock:
            if self._closed:
                raise PostgresContextError("repository is closed")
            if self._pool is not None:
                with self._pool.connection() as connection:  # type: ignore[attr-defined]
                    with self._connection_transaction(connection, access) as cursor:
                        yield cursor
                return
            assert self._connection is not None
            with self._connection_transaction(self._connection, access) as cursor:
                yield cursor

    @contextmanager
    def _connection_transaction(
        self,
        connection: object,
        access: PostgresContextAccess,
    ) -> Iterator[Any]:
        info = getattr(connection, "info", None)
        status = getattr(info, "transaction_status", None)
        status_name = getattr(status, "name", None)
        # psycopg.pq.TransactionStatus.IDLE has numeric value 0.  Reject
        # caller-owned connections already inside a transaction: a nested
        # savepoint would otherwise let the tenant-local setting outlive
        # this repository operation.
        if status is not None and status != 0 and status_name != "IDLE":
            raise PostgresContextError(
                "connection must be idle before a context operation"
            )
        with connection.transaction():  # type: ignore[attr-defined]
            with connection.cursor() as cursor:  # type: ignore[attr-defined]
                cursor.execute(_SET_TENANT_SQL, {"tenant_id": access.tenant_id})
                yield cursor

    def upsert_knowledge(
        self,
        chunks: Iterable[KnowledgeChunk],
        access: PostgresContextAccess,
    ) -> int:
        materialised = list(chunks)
        parameters: list[dict[str, object]] = []
        for chunk in materialised:
            if not isinstance(chunk, KnowledgeChunk):
                raise TypeError("all records must be KnowledgeChunk instances")
            if chunk.tenant_id != access.tenant_id:
                raise PermissionError("knowledge tenant does not match trusted access")
            parameters.append(_knowledge_parameters(chunk))
        if not parameters:
            return 0
        with self._transaction(access) as cursor:
            for values in parameters:
                cursor.execute(_KNOWLEDGE_UPSERT_SQL, values)
        return len(parameters)

    def replace_knowledge_version(
        self,
        chunks: Iterable[KnowledgeChunk],
        access: PostgresContextAccess,
    ) -> int:
        """Atomically replace one tenant/document/version in PostgreSQL."""

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
        if identity[0] != access.tenant_id:
            raise PermissionError("knowledge tenant does not match trusted access")
        with self._transaction(access) as cursor:
            cursor.execute(
                "DELETE FROM taskforge.knowledge_chunks "
                "WHERE tenant_id = %(tenant_id)s "
                "AND COALESCE(document_id, source_uri) = %(document_id)s "
                "AND version = %(version)s",
                {
                    "tenant_id": access.tenant_id,
                    "document_id": identity[1],
                    "version": identity[2],
                },
            )
            for chunk in materialised:
                cursor.execute(_KNOWLEDGE_UPSERT_SQL, _knowledge_parameters(chunk))
        return len(materialised)

    def fetch_knowledge_candidates(
        self,
        access: PostgresContextAccess,
        *,
        now: datetime | None = None,
        candidate_limit: int = 500,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
    ) -> list[KnowledgeChunk]:
        """Return bounded authorised candidates without claiming search rank."""

        if type(latest_only) is not bool:
            raise TypeError("latest_only must be a boolean")
        limit = _bounded_limit(candidate_limit)
        instant = as_utc(now)
        allowed_sources = _optional_text_array(source_uris, "source_uri")
        allowed_bases = _optional_text_array(
            knowledge_base_ids, "knowledge_base_id"
        )
        parameters: dict[str, object] = {
            "tenant_id": access.tenant_id,
            "user_id": access.user_id,
            "conversation_id": access.conversation_id,
            "org_id": access.org_id,
            "agent_id": access.agent_id,
            "additional_acl": sorted(access.acl_principals),
            "now": instant,
            "source_uris": allowed_sources,
            "knowledge_base_ids": allowed_bases,
            "latest_only": bool(latest_only),
            "candidate_limit": limit,
        }
        with self._transaction(access) as cursor:
            cursor.execute(_KNOWLEDGE_CANDIDATES_SQL, parameters)
            rows = list(cursor.fetchall())

        source_filter = None if allowed_sources is None else frozenset(allowed_sources)
        base_filter = None if allowed_bases is None else frozenset(allowed_bases)
        candidates: list[KnowledgeChunk] = []
        for row in rows:
            try:
                chunk = _row_to_knowledge(row)
                if (
                    _knowledge_visible(chunk, access, instant)
                    and (
                        source_filter is None
                        or chunk.source_uri in source_filter
                        or chunk.logical_document_id in source_filter
                    )
                    and (
                        base_filter is None
                        or str(chunk.metadata.get("knowledge_base_id", ""))
                        in base_filter
                    )
                ):
                    candidates.append(chunk)
            except (KeyError, TypeError, ValueError, OverflowError, PostgresDataError):
                # A corrupt or unexpectedly shaped row must never reach model
                # context, even if the database predicate claimed it visible.
                continue
        if latest_only:
            latest: dict[str, tuple[int, tuple[tuple[int, object], ...]]] = {}
            for chunk in candidates:
                current = latest.get(chunk.logical_document_id)
                if current is None or chunk.version_key > current:
                    latest[chunk.logical_document_id] = chunk.version_key
            candidates = [
                chunk
                for chunk in candidates
                if chunk.version_key == latest[chunk.logical_document_id]
            ]
        return candidates

    def upsert_memories(
        self,
        items: Iterable[MemoryItem],
        access: PostgresContextAccess,
        *,
        allow_shared_writes: bool = False,
    ) -> int:
        if type(allow_shared_writes) is not bool:
            raise TypeError("allow_shared_writes must be a boolean")
        materialised = list(items)
        parameters: list[dict[str, object]] = []
        for item in materialised:
            if not isinstance(item, MemoryItem):
                raise TypeError("all records must be MemoryItem instances")
            self._assert_memory_write_scope(
                item, access, allow_shared_writes=allow_shared_writes
            )
            parameters.append(_memory_parameters(item))
        if not parameters:
            return 0
        with self._transaction(access) as cursor:
            for values in parameters:
                cursor.execute(_MEMORY_UPSERT_SQL, values)
        return len(parameters)

    def forget_memory(
        self,
        memory_id: str,
        access: PostgresContextAccess,
        *,
        now: datetime | None = None,
    ) -> bool:
        """CAS-delete a principal-owned non-shared memory item."""

        instant = as_utc(now)
        with self._transaction(access) as cursor:
            cursor.execute(
                """
                SELECT tenant_id, memory_id, content, scope, scope_id,
                       provenance_json, importance, created_at, updated_at,
                       expires_at, tags_json, metadata_json
                  FROM taskforge.memory_items
                 WHERE tenant_id = %(tenant_id)s AND memory_id = %(memory_id)s
                   AND (expires_at IS NULL OR expires_at > %(now)s)
                """,
                {"tenant_id": access.tenant_id, "memory_id": memory_id, "now": instant},
            )
            row = cursor.fetchone()
            if row is None:
                return False
            item = _row_to_memory(row)
            if not item.is_deletable_by(
                AccessContext(
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    org_id=access.org_id,
                    agent_id=access.agent_id,
                    task_id=access.conversation_id,
                ),
                instant,
            ):
                return False
            cursor.execute(
                """
                DELETE FROM taskforge.memory_items
                 WHERE tenant_id = %(tenant_id)s AND memory_id = %(memory_id)s
                   AND scope = %(scope)s AND scope_id = %(scope_id)s
                   AND updated_at = %(updated_at)s
                """,
                {
                    "tenant_id": access.tenant_id,
                    "memory_id": memory_id,
                    "scope": item.scope.value,
                    "scope_id": item.scope_id,
                    "updated_at": _timestamp_param(item.updated_at),
                },
            )
            return cursor.rowcount == 1

    def fetch_memory_candidates(
        self,
        access: PostgresContextAccess,
        *,
        scopes: Iterable[MemoryScope | str] | None = None,
        now: datetime | None = None,
        candidate_limit: int = 500,
    ) -> list[MemoryItem]:
        """Return shared/private visible memories for a later ranking layer."""

        limit = _bounded_limit(candidate_limit)
        if scopes is None:
            scope_values: list[str] | None = None
        else:
            if isinstance(scopes, (str, bytes)):
                raise TypeError("scopes must be an iterable of MemoryScope values")
            scope_values = sorted({MemoryScope(scope).value for scope in scopes})
            if not scope_values:
                return []
        instant = as_utc(now)
        parameters: dict[str, object] = {
            "tenant_id": access.tenant_id,
            "user_id": access.user_id,
            "conversation_id": access.conversation_id,
            "org_id": access.org_id,
            "agent_id": access.agent_id,
            "scopes": scope_values,
            "now": instant,
            "candidate_limit": limit,
        }
        with self._transaction(access) as cursor:
            cursor.execute(_MEMORY_CANDIDATES_SQL, parameters)
            rows = list(cursor.fetchall())

        candidates: list[MemoryItem] = []
        allowed_scopes = None if scope_values is None else frozenset(scope_values)
        for row in rows:
            try:
                item = _row_to_memory(row)
                if (
                    _memory_visible(item, access, instant)
                    and (allowed_scopes is None or item.scope.value in allowed_scopes)
                ):
                    candidates.append(item)
            except (KeyError, TypeError, ValueError, OverflowError, PostgresDataError):
                continue
        return candidates

    @staticmethod
    def _assert_memory_write_scope(
        item: MemoryItem,
        access: PostgresContextAccess,
        *,
        allow_shared_writes: bool,
    ) -> None:
        if item.tenant_id != access.tenant_id:
            raise PermissionError("memory tenant does not match trusted access")
        expected_scope_ids: dict[MemoryScope, str | None] = {
            MemoryScope.TENANT: access.tenant_id,
            MemoryScope.ORG: access.org_id,
            MemoryScope.USER: access.user_id,
            MemoryScope.AGENT: access.agent_id,
            MemoryScope.TASK: access.conversation_id,
        }
        expected = expected_scope_ids[item.scope]
        if expected is None or item.scope_id != expected:
            raise PermissionError("memory scope does not match trusted access")
        if item.scope in {MemoryScope.TENANT, MemoryScope.ORG} and not allow_shared_writes:
            raise PermissionError("shared memory writes require explicit host authority")


# Naming aliases make the persistence-only contract discoverable without
# pretending that two independent connections/stores are required.
PostgresKnowledgeRepository = PostgresContextRepository
PostgresMemoryRepository = PostgresContextRepository


__all__ = [
    "MAX_CANDIDATES",
    "MAX_JSON_BYTES",
    "MAX_JSON_DEPTH",
    "PostgresContextAccess",
    "PostgresContextError",
    "PostgresContextRepository",
    "PostgresDataError",
    "PostgresDependencyError",
    "PostgresKnowledgeRepository",
    "PostgresMemoryRepository",
]
