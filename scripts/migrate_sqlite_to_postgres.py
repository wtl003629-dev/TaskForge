"""Migrate TaskForge SQLite state to PostgreSQL/pgvector.

The source is always opened read-only.  Schema creation is intentionally
separate: run the PostgreSQL migrations as ``migration_admin`` first, then run
this tool with the application database URL.  The execute path imports in
batches and uses ``ON CONFLICT DO NOTHING`` so an interrupted run can be
repeated safely without deleting or mutating SQLite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / ".taskforge"
JSON_COLUMNS = {
    "task_json",
    "profile_json",
    "state_json",
    "metadata_json",
    "receipt_json",
    "handoff_json",
    "memory_json",
    "claim_json",
    "fact_json",
    "plan_json",
    "case_json",
    "event_json",
    "result_case_json",
    "record_json",
    "request_json",
    "query_json",
    "card_json",
    "status_json",
    "scope_json",
    "details_json",
    "payload_json",
    "matched_queries_json",
    "acl_json",
    "provenance_json",
    "tags_json",
}
JSON_ARRAY_COLUMNS = frozenset({"acl_json", "tags_json", "matched_queries_json"})
ENUM_VALUES: dict[tuple[str, str], frozenset[str]] = {
    ("operations.operation_jobs", "status"): frozenset(
        {"queued", "leased", "completed", "dead_letter"}
    ),
    ("orchestration.speaker_plans", "status"): frozenset(
        {"ready", "running", "waiting_approval", "completed", "degraded", "failed", "cancelled"}
    ),
    ("orchestration.role_runs", "status"): frozenset(
        {"pending", "queued", "running", "waiting_approval", "succeeded", "failed", "cancelled"}
    ),
    ("orchestration.shared_facts", "status"): frozenset({"proposed", "verified"}),
    ("orchestration.fact_verification_receipts", "authority"): frozenset(
        {"tool", "user", "system"}
    ),
    ("review.review_cases", "status"): frozenset(
        {"draft", "submitted", "running", "waiting_human_review", "approved", "rejected", "failed"}
    ),
    ("review.review_case_audit_events", "event_type"): frozenset(
        {
            "case_created",
            "draft_updated",
            "case_submitted",
            "review_started",
            "model_recommendation_recorded",
            "case_approved",
            "case_rejected",
            "case_failed",
        }
    ),
    ("literature.paper_catalog", "verification_status"): frozenset(
        {"provider_verified", "cross_source_verified", "metadata_partial", "unverified"}
    ),
    ("literature.paper_catalog", "full_text_status"): frozenset(
        {"not_requested", "available", "abstract_only", "ingested", "failed"}
    ),
    ("literature.research_scopes", "status"): frozenset(
        {"draft", "confirmed", "ingesting", "ready", "expansion_requested", "closed"}
    ),
    ("literature.research_scope_papers", "selection_status"): frozenset(
        {"selected", "excluded"}
    ),
}
JSON_ENUM_VALUES: dict[tuple[str, str, str], frozenset[str]] = {
    ("core.runs", "state_json", "status"): frozenset(
        {"pending", "running", "waiting_approval", "completed", "failed", "step_limit"}
    ),
    ("literature.paper_ingestion_jobs", "status_json", "status"): frozenset(
        {"queued", "uploaded", "fetching", "parsing", "indexed", "abstract_only", "failed"}
    ),
    ("literature.evidence_cards", "card_json", "verification_status"): frozenset(
        {"unread", "read", "verified", "unsupported"}
    ),
    ("literature.claim_records", "claim_json", "citation_status"): frozenset(
        {"unverified", "verified", "unsupported", "scope_mismatch"}
    ),
    ("literature.claim_records", "claim_json", "verification_status"): frozenset(
        {"unverified", "verified", "needs_review"}
    ),
    ("literature.scope_expansion_requests", "request_json", "status"): frozenset(
        {"pending", "approved", "rejected"}
    ),
}
TIME_COLUMNS = {
    "created_at",
    "updated_at",
    "occurred_at",
    "produced_at",
    "available_at",
    "lease_expires_at",
    "started_at",
    "completed_at",
    "confirmed_at",
    "expires_at",
    "valid_from",
    "valid_until",
}
SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TableMapping:
    database: str
    source_table: str
    target_schema: str
    target_table: str
    add_tenant: bool = False
    skip_source_columns: tuple[str, ...] = ()


TABLES: tuple[TableMapping, ...] = (
    TableMapping("taskforge.sqlite3", "tasks", "core", "tasks", add_tenant=True),
    TableMapping("taskforge.sqlite3", "profiles", "core", "profiles", add_tenant=True),
    TableMapping("taskforge.sqlite3", "runs", "core", "runs", add_tenant=True),
    TableMapping("context.sqlite3", "knowledge_chunks", "taskforge", "knowledge_chunks"),
    TableMapping("context.sqlite3", "memory_items", "taskforge", "memory_items"),
    TableMapping("operations.sqlite3", "operation_jobs", "operations", "operation_jobs"),
    TableMapping("operations.sqlite3", "audit_events", "operations", "audit_events", skip_source_columns=("sequence",)),
    TableMapping("orchestration.sqlite3", "speaker_plans", "orchestration", "speaker_plans"),
    TableMapping("orchestration.sqlite3", "role_runs", "orchestration", "role_runs"),
    TableMapping("orchestration.sqlite3", "handoffs", "orchestration", "handoffs"),
    TableMapping("orchestration.sqlite3", "shared_facts", "orchestration", "shared_facts"),
    TableMapping("orchestration.sqlite3", "fact_verification_receipts", "orchestration", "fact_verification_receipts"),
    TableMapping("orchestration.sqlite3", "private_role_memories", "orchestration", "private_role_memories"),
    TableMapping("orchestration.sqlite3", "role_run_execution_claims", "orchestration", "role_run_execution_claims"),
    TableMapping("review-cases.sqlite3", "review_cases", "review", "review_cases"),
    TableMapping("review-cases.sqlite3", "review_case_audit_events", "review", "review_case_audit_events"),
    TableMapping("review-cases.sqlite3", "review_case_commands", "review", "review_case_commands"),
    TableMapping("verification.sqlite3", "verification_records", "verification", "verification_records", add_tenant=True),
    TableMapping("literature.sqlite3", "literature_requests", "literature", "literature_requests"),
    TableMapping("literature.sqlite3", "literature_queries", "literature", "literature_queries"),
    TableMapping("literature.sqlite3", "paper_catalog", "literature", "paper_catalog"),
    TableMapping("literature.sqlite3", "paper_identifiers", "literature", "paper_identifiers"),
    TableMapping("literature.sqlite3", "research_scopes", "literature", "research_scopes"),
    TableMapping("literature.sqlite3", "research_scope_papers", "literature", "research_scope_papers"),
    TableMapping("literature.sqlite3", "paper_ingestion_jobs", "literature", "paper_ingestion_jobs"),
    TableMapping("literature.sqlite3", "paper_search_results", "literature", "paper_search_results"),
    TableMapping("literature.sqlite3", "evidence_cards", "literature", "evidence_cards"),
    TableMapping("literature.sqlite3", "claim_records", "literature", "claim_records"),
    TableMapping("literature.sqlite3", "scope_expansion_requests", "literature", "scope_expansion_requests"),
    TableMapping("literature.sqlite3", "literature_audit_events", "literature", "audit_events"),
    TableMapping("literature-cache.sqlite3", "literature_provider_cache", "literature", "provider_cache", add_tenant=True),
)

# Both historical vector caches are migrated to the mixed-dimension
# ``vector.embedding_cache`` table. Only the 1024-dimensional Bailian
# document rows that hash-match a knowledge chunk are additionally copied to
# the fixed-dimension ``vector.knowledge_embeddings`` table.
VECTOR_SOURCES: tuple[tuple[str, str, str, int], ...] = (
    ("embeddings.sqlite3", "embeddings_v1", "BAAI/bge-small-en-v1.5", 384),
    (
        "embeddings-bailian-v4-1024.sqlite3",
        "embeddings_bailian_v4_1024_v1",
        "aliyun-bailian|text-embedding-v4|dense-v1|1024",
        1024,
    ),
)

# These are the cross-table references that must remain valid after the
# tenant-aware import.  PostgreSQL constraints enforce them at write time;
# verify repeats the checks explicitly so the migration report records the
# result rather than relying only on a successful INSERT.
FOREIGN_KEY_CHECKS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("core.runs", ("task_id",), "core.tasks", ("task_id",)),
    ("core.runs", ("profile_id",), "core.profiles", ("profile_id",)),
    ("operations.operation_jobs", ("run_id",), "core.runs", ("run_id",)),
    ("operations.audit_events", ("run_id",), "core.runs", ("run_id",)),
    ("orchestration.role_runs", ("plan_id",), "orchestration.speaker_plans", ("plan_id",)),
    ("orchestration.handoffs", ("plan_id",), "orchestration.speaker_plans", ("plan_id",)),
    ("orchestration.handoffs", ("from_role_run_id",), "orchestration.role_runs", ("role_run_id",)),
    ("orchestration.role_run_execution_claims", ("role_run_id",), "orchestration.role_runs", ("role_run_id",)),
    ("literature.literature_queries", ("request_id",), "literature.literature_requests", ("request_id",)),
    ("literature.paper_identifiers", ("paper_id",), "literature.paper_catalog", ("paper_id",)),
    ("literature.research_scopes", ("request_id",), "literature.literature_requests", ("request_id",)),
    ("literature.research_scope_papers", ("scope_id", "scope_version"), "literature.research_scopes", ("scope_id", "scope_version")),
    ("literature.research_scope_papers", ("paper_id",), "literature.paper_catalog", ("paper_id",)),
    ("literature.paper_ingestion_jobs", ("scope_id", "scope_version"), "literature.research_scopes", ("scope_id", "scope_version")),
    ("literature.paper_search_results", ("request_id",), "literature.literature_requests", ("request_id",)),
    ("literature.paper_search_results", ("paper_id",), "literature.paper_catalog", ("paper_id",)),
    ("literature.evidence_cards", ("scope_id", "scope_version"), "literature.research_scopes", ("scope_id", "scope_version")),
    ("literature.evidence_cards", ("paper_id",), "literature.paper_catalog", ("paper_id",)),
    ("literature.claim_records", ("scope_id", "scope_version"), "literature.research_scopes", ("scope_id", "scope_version")),
    ("literature.scope_expansion_requests", ("scope_id", "scope_version"), "literature.research_scopes", ("scope_id", "scope_version")),
    ("vector.knowledge_embeddings", ("chunk_id",), "taskforge.knowledge_chunks", ("chunk_id",)),
    ("review.review_case_audit_events", ("case_id",), "review.review_cases", ("case_id",)),
    ("review.review_case_commands", ("case_id",), "review.review_cases", ("case_id",)),
)

PRIMARY_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "core.tasks": ("task_id",),
    "core.profiles": ("profile_id",),
    "core.runs": ("run_id",),
    "taskforge.knowledge_chunks": ("chunk_id",),
    "taskforge.memory_items": ("memory_id",),
    "operations.operation_jobs": ("run_id",),
    "operations.audit_events": ("event_id",),
    "orchestration.speaker_plans": ("plan_id",),
    "orchestration.role_runs": ("role_run_id",),
    "orchestration.handoffs": ("handoff_id",),
    "orchestration.shared_facts": ("fact_id",),
    "orchestration.fact_verification_receipts": ("receipt_id",),
    "orchestration.private_role_memories": ("memory_id",),
    "orchestration.role_run_execution_claims": ("role_run_id",),
    "review.review_cases": ("case_id",),
    "review.review_case_audit_events": ("event_id",),
    "review.review_case_commands": ("owner_user_id", "conversation_id", "idempotency_key"),
    "verification.verification_records": ("record_id",),
    "literature.literature_requests": ("request_id",),
    "literature.literature_queries": ("request_id", "query_id"),
    "literature.paper_catalog": ("paper_id",),
    "literature.paper_identifiers": ("identifier_type", "identifier_value"),
    "literature.research_scopes": ("scope_id", "scope_version"),
    "literature.research_scope_papers": ("scope_id", "scope_version", "paper_id"),
    "literature.paper_ingestion_jobs": ("scope_id", "scope_version", "paper_id"),
    "literature.paper_search_results": ("request_id", "paper_id"),
    "literature.evidence_cards": ("evidence_id",),
    "literature.claim_records": ("claim_id",),
    "literature.scope_expansion_requests": ("expansion_id",),
    "literature.audit_events": ("event_id",),
    "literature.provider_cache": ("cache_key",),
    "vector.embedding_cache": ("cache_key",),
    "vector.knowledge_embeddings": ("chunk_id", "model"),
}
GLOBAL_UNIQUE_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "operations.audit_events": ("event_id",),
    "literature.audit_events": ("event_id",),
}


def quote_identifier(value: str) -> str:
    if not value or any(not (character.isalnum() or character == "_") for character in value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return '"' + value + '"'


def source_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def parse_json(value: object, *, column: str) -> object:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{column} must contain JSON text")
    return json.loads(value)


def validate_json_contract(
    value: object,
    *,
    target_name: str,
    column: str,
) -> object:
    """Validate the JSONB shape and constrained enum fields before import."""

    expected_array = column in JSON_ARRAY_COLUMNS
    if expected_array and not isinstance(value, list):
        raise ValueError(f"{target_name}.{column} must contain a JSON array")
    if not expected_array and not isinstance(value, dict):
        raise ValueError(f"{target_name}.{column} must contain a JSON object")
    if isinstance(value, dict):
        for (rule_target, rule_column, field), allowed in JSON_ENUM_VALUES.items():
            if (rule_target, rule_column) != (target_name, column):
                continue
            current = value.get(field)
            if not isinstance(current, str) or current not in allowed:
                raise ValueError(
                    f"{target_name}.{column}.{field} contains unsupported value: {current!r}"
                )
    return value


def parse_timestamp(value: object, *, column: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{column} contains a non-finite epoch")
        return datetime.fromtimestamp(float(value), timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{column} must contain an ISO timestamp or epoch")
    parsed = datetime.fromisoformat(value)
    # SQLite CURRENT_TIMESTAMP and legacy TaskForge profile rows are UTC but
    # timezone-naive. Preserve the instant explicitly instead of relying on
    # the host's local timezone during migration.
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_digest(values: Iterable[object]) -> str:
    encoded = json.dumps(list(values), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rows_digest(rows: Iterable[Iterable[object]]) -> str:
    """Digest a row multiset so PostgreSQL ordering cannot hide drift."""

    return canonical_digest(
        sorted(canonical_digest(row) for row in rows)
    )


def status_distribution(columns: list[str], rows: Iterable[tuple[Any, ...]]) -> dict[str, int]:
    """Return a stable status histogram for tables that expose ``status``."""

    try:
        index = columns.index("status")
    except ValueError:
        return {}
    result: dict[str, int] = {}
    for row in rows:
        value = row[index]
        key = "<NULL>" if value is None else str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def table_rows(mapping: TableMapping, source_root: Path, tenant_id: str) -> tuple[list[str], list[tuple[Any, ...]], dict[str, Any]]:
    path = source_root / mapping.database
    with source_connection(path) as connection:
        source_columns = [row["name"] for row in connection.execute(
            f"PRAGMA table_info({quote_identifier(mapping.source_table)})"
        )]
        if not source_columns:
            raise ValueError(f"source table not found: {mapping.database}:{mapping.source_table}")
        columns = [column for column in source_columns if column not in mapping.skip_source_columns]
        target_columns = (["tenant_id"] if mapping.add_tenant else []) + columns
        target_name = f"{mapping.target_schema}.{mapping.target_table}"
        rows: list[tuple[Any, ...]] = []
        for row in connection.execute(f"SELECT {', '.join(quote_identifier(column) for column in columns)} FROM {quote_identifier(mapping.source_table)}"):
            converted: list[Any] = [tenant_id] if mapping.add_tenant else []
            for column, value in zip(columns, row):
                if column in JSON_COLUMNS or column.endswith("_json"):
                    converted.append(
                        validate_json_contract(
                            parse_json(value, column=column),
                            target_name=target_name,
                            column=column,
                        )
                    )
                elif column in TIME_COLUMNS:
                    converted.append(parse_timestamp(value, column=column))
                elif column == "safety_violation":
                    converted.append(bool(value))
                else:
                    allowed = ENUM_VALUES.get((target_name, column))
                    if allowed is not None and (
                        not isinstance(value, str) or value not in allowed
                    ):
                        raise ValueError(
                            f"{target_name}.{column} contains unsupported value: {value!r}"
                        )
                    converted.append(value)
            rows.append(tuple(converted))
    summary = {
        "database": mapping.database,
        "source_table": mapping.source_table,
        "target": f"{mapping.target_schema}.{mapping.target_table}",
        "source_rows": len(rows),
        "columns": target_columns,
        "content_sha256": rows_digest(rows),
    }
    return target_columns, rows, summary


def vector_rows(source_root: Path, tenant_id: str) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    rows: list[tuple[Any, ...]] = []
    source_summaries: list[dict[str, Any]] = []
    cache_keys: set[str] = set()
    for database, table, expected_model, expected_dimension in VECTOR_SOURCES:
        path = source_root / database
        source_rows: list[tuple[Any, ...]] = []
        with source_connection(path) as connection:
            for row in connection.execute(
                f"SELECT cache_key, model_name, embedding_kind, text_sha256, dimension, vector FROM {quote_identifier(table)}"
            ):
                model_name, kind, dimension, blob = row[1], row[2], row[4], row[5]
                if str(model_name) != expected_model:
                    raise ValueError(
                        f"unexpected embedding model in {database}: {model_name!r}"
                    )
                if kind not in {"document", "query"}:
                    raise ValueError(f"unexpected embedding kind: {kind!r}")
                if (
                    not isinstance(row[0], str)
                    or not isinstance(row[3], str)
                    or len(row[3]) != 64
                    or any(character not in SHA256_HEX for character in row[3])
                ):
                    raise ValueError(f"invalid SHA-256 text hash for cache key {row[0]!r}")
                if row[0] in cache_keys:
                    raise ValueError(f"duplicate embedding cache key: {row[0]!r}")
                cache_keys.add(row[0])
                if dimension != expected_dimension or not isinstance(blob, (bytes, bytearray)) or len(blob) != expected_dimension * 4:
                    raise ValueError(
                        f"{database} embedding is not a {expected_dimension}-dimensional float32 BLOB"
                    )
                vector = struct.unpack(f"<{expected_dimension}f", blob)
                if not all(math.isfinite(value) for value in vector) or not any(value != 0.0 for value in vector):
                    raise ValueError(f"invalid vector values for cache key {row[0]!r}")
                source_rows.append((tenant_id, row[0], row[1], kind, row[3], dimension, vector))
        rows.extend(source_rows)
        source_summaries.append(
            {
                "database": path.name,
                "source_table": table,
                "model": expected_model,
                "dimension": expected_dimension,
                "source_rows": len(source_rows),
                "content_sha256": canonical_digest(source_rows),
            }
        )
    return rows, {
        "database": "multiple",
        "source_table": "multiple",
        "target": "vector.embedding_cache",
        "source_rows": len(rows),
        "dimensions": sorted({int(row[5]) for row in rows}),
        "sources": source_summaries,
        "content_sha256": canonical_digest(rows),
    }


def knowledge_vector_rows(
    source_root: Path,
    tenant_id: str,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    """Map document embeddings to chunks by the authoritative text hash."""

    chunk_hashes: dict[str, str] = {}
    with source_connection(source_root / "context.sqlite3") as connection:
        for row in connection.execute("SELECT chunk_id, text_content FROM knowledge_chunks"):
            digest = hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest()
            previous = chunk_hashes.get(digest)
            if previous is not None and previous != row[0]:
                raise ValueError(f"text hash maps to multiple chunks: {digest}")
            chunk_hashes[digest] = str(row[0])
    cache_rows, _ = vector_rows(source_root, tenant_id)
    mapped: list[tuple[Any, ...]] = []
    mapped_keys: set[tuple[str, str]] = set()
    for row in cache_rows:
        if (
            row[3] != "document"
            or int(row[5]) != 1024
            or str(row[2]) != "aliyun-bailian|text-embedding-v4|dense-v1|1024"
        ):
            continue
        chunk_id = chunk_hashes.get(str(row[4]))
        if chunk_id is None:
            continue
        # The cache key carries the provider namespace, while the application
        # contract identifies this route by its configured model name. Keep
        # the provider-qualified identity in vector.embedding_cache, but use
        # the application model name for knowledge_embeddings so the runtime
        # query and the migrated rows agree.
        model_name = str(row[2]).split("|", 2)[1] if "|" in str(row[2]) else str(row[2])
        mapped_key = (str(chunk_id), model_name)
        if mapped_key in mapped_keys:
            raise ValueError(
                "multiple vectors map to the same knowledge embedding key: "
                f"{mapped_key[0]}:{mapped_key[1]}"
            )
        mapped_keys.add(mapped_key)
        mapped.append((row[0], chunk_id, model_name, row[4], row[5], row[6]))
    return mapped, {
        "target": "vector.knowledge_embeddings",
        "source_rows": len(mapped),
        "mapped_document_rows": len(mapped),
        "unmapped_document_rows": sum(
            row[3] == "document"
            and int(row[5]) == 1024
            and str(row[2]) == "aliyun-bailian|text-embedding-v4|dense-v1|1024"
            and str(row[4]) not in chunk_hashes
            for row in cache_rows
        ),
        "dimension": 1024,
        "content_sha256": canonical_digest(mapped),
    }


def source_columns(mapping: TableMapping, source_root: Path) -> list[str]:
    with source_connection(source_root / mapping.database) as connection:
        columns = [
            row["name"]
            for row in connection.execute(
                f"PRAGMA table_info({quote_identifier(mapping.source_table)})"
            )
        ]
    if not columns:
        raise ValueError(f"source table not found: {mapping.database}:{mapping.source_table}")
    return columns


def inventory(source_root: Path, tenant_id: str) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    source_tables: dict[str, tuple[list[str], list[tuple[Any, ...]]]] = {}
    for mapping in TABLES:
        columns, rows, summary = table_rows(mapping, source_root, tenant_id)
        source_tables[summary["target"]] = (columns, rows)
        tables.append(summary)
    source_integrity = validate_source_integrity(source_tables)
    _, vector_summary = vector_rows(source_root, tenant_id)
    _, knowledge_vector_summary = knowledge_vector_rows(source_root, tenant_id)
    return {
        "schema_version": 2,
        "source_root": str(source_root),
        "tenant_id_for_legacy_rows": tenant_id,
        "tables": tables,
        "source_integrity": source_integrity,
        "vectors": vector_summary,
        "knowledge_embeddings": knowledge_vector_summary,
        "source_row_count": sum(item["source_rows"] for item in tables) + vector_summary["source_rows"],
    }


def import_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.types.json import Json
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL execution requires `pip install taskforge-agent`"
        ) from exc
    return psycopg, Json


def vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


def tenant_groups(columns: list[str], rows: list[tuple[Any, ...]]) -> dict[str, list[tuple[Any, ...]]]:
    tenant_index = columns.index("tenant_id")
    groups: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        groups.setdefault(str(row[tenant_index]), []).append(row)
    return groups


def validate_source_integrity(
    source_tables: dict[str, tuple[list[str], list[tuple[Any, ...]]]],
) -> dict[str, Any]:
    """Validate source PK/FK shape before the first PostgreSQL commit."""

    primary_keys: list[dict[str, Any]] = []
    for target_name, (columns, rows) in source_tables.items():
        key_columns = PRIMARY_KEY_COLUMNS.get(target_name)
        if not key_columns:
            continue
        effective_key_columns = GLOBAL_UNIQUE_KEY_COLUMNS.get(target_name, ("tenant_id", *key_columns))
        indexes = [columns.index(column) for column in effective_key_columns]
        keys = [tuple(row[index] for index in indexes) for row in rows]
        null_key_rows = sum(any(value is None for value in key) for key in keys)
        duplicate_keys = len(keys) - len(set(keys))
        check = {
            "target": target_name,
            "key_columns": list(effective_key_columns),
            "rows": len(rows),
            "null_key_rows": null_key_rows,
            "duplicate_keys": duplicate_keys,
            "passed": null_key_rows == 0 and duplicate_keys == 0,
        }
        primary_keys.append(check)
        if not check["passed"]:
            raise ValueError(
                f"{target_name} source primary key validation failed: "
                f"nulls={null_key_rows}, duplicates={duplicate_keys}"
            )

    foreign_keys: list[dict[str, Any]] = []
    for child, child_columns, parent, parent_columns in FOREIGN_KEY_CHECKS:
        # knowledge_embeddings are generated from the authoritative text-hash
        # mapping rather than a source table; that mapping is checked separately.
        if child not in source_tables or parent not in source_tables:
            continue
        child_columns_all, child_rows = source_tables[child]
        parent_columns_all, parent_rows = source_tables[parent]
        child_tenant_index = child_columns_all.index("tenant_id")
        parent_tenant_index = parent_columns_all.index("tenant_id")
        child_indexes = [child_columns_all.index(column) for column in child_columns]
        parent_indexes = [parent_columns_all.index(column) for column in parent_columns]
        parent_keys = {
            (row[parent_tenant_index], *(row[index] for index in parent_indexes))
            for row in parent_rows
        }
        orphan_rows = 0
        for row in child_rows:
            key = (row[child_tenant_index], *(row[index] for index in child_indexes))
            if key not in parent_keys:
                orphan_rows += 1
        check = {
            "child": child,
            "parent": parent,
            "orphan_rows": orphan_rows,
            "passed": orphan_rows == 0,
        }
        foreign_keys.append(check)
        if not check["passed"]:
            raise ValueError(
                f"{child} source foreign key validation failed for {parent}: "
                f"orphans={orphan_rows}"
            )

    return {
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "passed": all(item["passed"] for item in primary_keys + foreign_keys),
    }


def batch_count(row_count: int, batch_size: int) -> int:
    return (row_count + batch_size - 1) // batch_size if row_count else 0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a migration report without connection details."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def set_tenant(cursor: Any, tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be blank")
    cursor.execute("SELECT set_config('taskforge.tenant_id', %s, true)", (tenant_id,))


def execute_import(
    source_root: Path,
    dsn: str,
    tenant_id: str,
    batch_size: int,
    progress_report: Path | None = None,
) -> dict[str, Any]:
    psycopg, Json = import_psycopg()
    report = inventory(source_root, tenant_id)
    progress: dict[str, Any] = {
        "status": "in_progress",
        "tables_completed": 0,
        "tables_total": len(TABLES),
        "vectors_completed": 0,
        "vectors_total": 0,
        "knowledge_embeddings_completed": 0,
        "knowledge_embeddings_total": 0,
        "batches": [],
    }

    def checkpoint_progress(stage: str, **details: Any) -> None:
        progress["last_committed_batch"] = {"stage": stage, **details}
        if progress_report is not None:
            write_report(progress_report, {**report, "mode": "execute", "progress": progress})

    checkpoint_progress("start")
    imported: list[dict[str, Any]] = []
    with psycopg.connect(dsn, autocommit=False) as connection:
        for mapping in TABLES:
            columns, rows, summary = table_rows(mapping, source_root, tenant_id)
            table_progress = {
                "target": f"{mapping.target_schema}.{mapping.target_table}",
                "completed_rows": 0,
                "total_rows": len(rows),
                "batch_count": batch_count(len(rows), batch_size),
                "status": "in_progress",
            }
            progress.setdefault("tables", []).append(table_progress)
            if not rows:
                table_progress["status"] = "completed"
                progress["tables_completed"] += 1
                checkpoint_progress("table", target=table_progress["target"], completed_rows=0, total_rows=0)
                imported.append({**summary, "imported_rows": 0, "batch_count": 0})
                continue
            identifiers = f"{quote_identifier(mapping.target_schema)}.{quote_identifier(mapping.target_table)}"
            column_sql = ", ".join(quote_identifier(column) for column in columns)
            values_sql = ", ".join(["%s"] * len(columns))
            sql = f"INSERT INTO {identifiers} ({column_sql}) VALUES ({values_sql}) ON CONFLICT DO NOTHING"
            with connection.cursor() as cursor:
                for grouped_rows in tenant_groups(columns, rows).values():
                    for start in range(0, len(grouped_rows), batch_size):
                        set_tenant(cursor, str(grouped_rows[0][columns.index("tenant_id")]))
                        batch = grouped_rows[start : start + batch_size]
                        adapted = [
                            tuple(Json(value) if column in JSON_COLUMNS or column.endswith("_json") else value for column, value in zip(columns, row))
                            for row in batch
                        ]
                        cursor.executemany(sql, adapted)
                        connection.commit()
                        table_progress["completed_rows"] += len(batch)
                        progress["batches"].append(
                            {
                                "stage": "table",
                                "target": table_progress["target"],
                                "completed_rows": table_progress["completed_rows"],
                                "total_rows": table_progress["total_rows"],
                            }
                        )
                        checkpoint_progress(
                            "table",
                            target=table_progress["target"],
                            completed_rows=table_progress["completed_rows"],
                            total_rows=table_progress["total_rows"],
                        )
            table_progress["status"] = "completed"
            progress["tables_completed"] += 1
            imported.append(
                {
                    **summary,
                    "imported_rows": len(rows),
                    "batch_count": sum(
                        batch_count(len(group), batch_size)
                        for group in tenant_groups(columns, rows).values()
                    ),
                }
            )
            if mapping.target_schema == "literature" and mapping.target_table == "audit_events":
                source_max_event_id = max(
                    (int(row[columns.index("event_id")]) for row in rows),
                    default=0,
                )
                with connection.cursor() as sequence_cursor:
                    sequence_cursor.execute(
                        "SELECT setval("
                        "'literature.audit_event_id_seq'::regclass, "
                        "GREATEST(last_value, %s, 1), true) "
                        "FROM literature.audit_event_id_seq",
                        (source_max_event_id,),
                    )
                    connection.commit()

        vector_data, vector_summary = vector_rows(source_root, tenant_id)
        progress["vectors_total"] = len(vector_data)
        vector_sql = (
            "INSERT INTO vector.embedding_cache "
            "(tenant_id, cache_key, model_name, embedding_kind, text_sha256, dimension, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) ON CONFLICT DO NOTHING"
        )
        with connection.cursor() as cursor:
            grouped_vectors: dict[str, list[tuple[Any, ...]]] = {}
            for row in vector_data:
                grouped_vectors.setdefault(str(row[0]), []).append(row)
            for grouped_rows in grouped_vectors.values():
                for start in range(0, len(grouped_rows), batch_size):
                    set_tenant(cursor, str(grouped_rows[0][0]))
                    batch = grouped_rows[start : start + batch_size]
                    cursor.executemany(
                        vector_sql,
                        [(*row[:6], vector_literal(row[6])) for row in batch],
                    )
                    connection.commit()
                    progress["vectors_completed"] += len(batch)
                    progress["batches"].append(
                        {
                            "stage": "vectors",
                            "completed_rows": progress["vectors_completed"],
                            "total_rows": len(vector_data),
                        }
                    )
                    checkpoint_progress(
                        "vectors",
                        completed_rows=progress["vectors_completed"],
                        total_rows=len(vector_data),
                    )
        report["imported_tables"] = imported
        report["imported_vectors"] = {
            **vector_summary,
            "imported_rows": len(vector_data),
            "batch_count": batch_count(len(vector_data), batch_size),
        }
        knowledge_data, knowledge_summary = knowledge_vector_rows(source_root, tenant_id)
        knowledge_sql = (
            "INSERT INTO vector.knowledge_embeddings "
            "(tenant_id, chunk_id, model, text_sha256, dimension, embedding, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s::vector, CURRENT_TIMESTAMP) "
            "ON CONFLICT DO NOTHING"
        )
        with connection.cursor() as cursor:
            for start in range(0, len(knowledge_data), batch_size):
                batch = knowledge_data[start : start + batch_size]
                set_tenant(cursor, str(batch[0][0]))
                cursor.executemany(
                    knowledge_sql,
                    [(*row[:5], vector_literal(row[5])) for row in batch],
                )
                connection.commit()
                progress["knowledge_embeddings_completed"] += len(batch)
                progress["batches"].append(
                    {
                        "stage": "knowledge_embeddings",
                        "completed_rows": progress["knowledge_embeddings_completed"],
                        "total_rows": len(knowledge_data),
                    }
                )
                checkpoint_progress(
                    "knowledge_embeddings",
                    completed_rows=progress["knowledge_embeddings_completed"],
                    total_rows=len(knowledge_data),
                )
        report["imported_knowledge_embeddings"] = {
            **knowledge_summary,
            "imported_rows": len(knowledge_data),
            "batch_count": batch_count(len(knowledge_data), batch_size),
        }
    progress["vectors_completed"] = len(vector_data)
    progress["knowledge_embeddings_total"] = len(knowledge_data)
    progress["knowledge_embeddings_completed"] = len(knowledge_data)
    progress["status"] = "completed"
    report["progress"] = progress
    return report


def verify_import(source_root: Path, dsn: str, tenant_id: str) -> dict[str, Any]:
    psycopg, _ = import_psycopg()
    report = inventory(source_root, tenant_id)
    checks: list[dict[str, Any]] = []
    # Always inspect the explicitly selected tenant, even when a source table
    # is empty.  Otherwise a stale target row could survive a migration and
    # make an empty source look verified simply because there was no source
    # group to iterate over.
    verification_tenants: set[str] = {tenant_id}
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            for mapping in TABLES:
                columns, rows, summary = table_rows(mapping, source_root, tenant_id)
                source_tenants = tenant_groups(columns, rows)
                verification_tenants.update(source_tenants)
                target_rows = 0
                target_digest_parts: list[str] = []
                tenant_checks: list[dict[str, Any]] = []
                target_name = f"{mapping.target_schema}.{mapping.target_table}"
                primary_key_columns = PRIMARY_KEY_COLUMNS.get(target_name)
                table_tenants = set(source_tenants)
                table_tenants.add(tenant_id)
                for current_tenant in sorted(table_tenants):
                    current_rows = source_tenants.get(current_tenant, [])
                    set_tenant(cursor, current_tenant)
                    cursor.execute(
                        f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
                        f"FROM {quote_identifier(mapping.target_schema)}.{quote_identifier(mapping.target_table)}"
                    )
                    fetched = [tuple(row) for row in cursor.fetchall()]
                    target_rows += len(fetched)
                    target_digest = rows_digest(fetched)
                    if fetched or current_rows:
                        target_digest_parts.append(target_digest)
                    source_digest = rows_digest(current_rows)
                    source_statuses = status_distribution(columns, current_rows)
                    target_statuses = status_distribution(columns, fetched)
                    source_key_unique = (
                        len(current_rows)
                        == len(
                            {
                                tuple(row[columns.index(column)] for column in primary_key_columns)
                                for row in current_rows
                            }
                        )
                        if primary_key_columns
                        else True
                    )
                    target_key_duplicates = 0
                    if primary_key_columns:
                        key_sql = ", ".join(
                            quote_identifier(column) for column in primary_key_columns
                        )
                        cursor.execute(
                            f"SELECT COALESCE(SUM(duplicate_count - 1), 0) FROM ("
                            f"SELECT {key_sql}, COUNT(*) AS duplicate_count "
                            f"FROM {quote_identifier(mapping.target_schema)}.{quote_identifier(mapping.target_table)} "
                            f"WHERE tenant_id = %s GROUP BY {key_sql} HAVING COUNT(*) > 1"
                            f") AS duplicate_keys",
                            (current_tenant,),
                        )
                        target_key_duplicates = int(cursor.fetchone()[0])
                    tenant_checks.append(
                        {
                            "tenant_id": current_tenant,
                            "source_rows": len(current_rows),
                            "target_rows": len(fetched),
                            "row_count_match": len(fetched) == len(current_rows),
                            "content_sha256_match": target_digest == source_digest,
                            "source_status_distribution": source_statuses,
                            "target_status_distribution": target_statuses,
                            "status_distribution_match": target_statuses == source_statuses,
                            "source_primary_key_unique": source_key_unique,
                            "target_primary_key_duplicates": target_key_duplicates,
                            "primary_key_match": source_key_unique and target_key_duplicates == 0,
                        }
                    )
                    connection.rollback()
                content_match = (
                    canonical_digest(sorted(target_digest_parts))
                    == canonical_digest(
                        sorted(rows_digest(group) for group in source_tenants.values())
                    )
                )
                checks.append(
                    {
                        **summary,
                        "target_rows": target_rows,
                        "row_count_match": target_rows == summary["source_rows"],
                        "content_sha256_match": content_match,
                        "status_distribution_match": all(
                            item["status_distribution_match"] for item in tenant_checks
                        ),
                        "primary_key_match": all(
                            item["primary_key_match"] for item in tenant_checks
                        ),
                        "tenants": tenant_checks,
                    }
                )
            _, vector_summary = vector_rows(source_root, tenant_id)
            vector_tenants: dict[str, int] = {}
            vector_data, _ = vector_rows(source_root, tenant_id)
            for row in vector_data:
                vector_tenants[str(row[0])] = vector_tenants.get(str(row[0]), 0) + 1
            verification_tenants.update(vector_tenants)
            target_vectors = 0
            target_vector_rows: list[tuple[Any, ...]] = []
            vector_tenants.setdefault(tenant_id, 0)
            for current_tenant in vector_tenants:
                set_tenant(cursor, current_tenant)
                cursor.execute(
                    "SELECT cache_key, model_name, embedding_kind, text_sha256, "
                    "dimension, encode(vector_send(embedding), 'hex') "
                    "FROM vector.embedding_cache "
                    "ORDER BY cache_key",
                )
                fetched = [tuple(row) for row in cursor.fetchall()]
                target_vectors += len(fetched)
                target_vector_rows.extend(
                    (*row[:5], _pg_vector_values(row[5], expected_dimension=int(row[4])))
                    for row in fetched
                )
                connection.rollback()
            source_vector_rows = [tuple(row[1:]) for row in vector_data]
            vector_content_match = rows_digest(target_vector_rows) == rows_digest(
                source_vector_rows
            )

            foreign_key_checks: list[dict[str, Any]] = []
            for current_tenant in sorted(verification_tenants):
                set_tenant(cursor, current_tenant)
                for child, child_columns, parent, parent_columns in FOREIGN_KEY_CHECKS:
                    child_schema, child_table = child.split(".", 1)
                    parent_schema, parent_table = parent.split(".", 1)
                    join = " AND ".join(
                        [
                            "parent.tenant_id = child.tenant_id",
                            *(
                                f"parent.{quote_identifier(parent_column)} = child.{quote_identifier(child_column)}"
                                for child_column, parent_column in zip(child_columns, parent_columns, strict=True)
                            ),
                        ]
                    )
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {quote_identifier(child_schema)}.{quote_identifier(child_table)} AS child "
                        f"LEFT JOIN {quote_identifier(parent_schema)}.{quote_identifier(parent_table)} AS parent "
                        f"ON {join} WHERE child.tenant_id = %s AND parent.tenant_id IS NULL",
                        (current_tenant,),
                    )
                    orphan_count = int(cursor.fetchone()[0])
                    foreign_key_checks.append(
                        {
                            "tenant_id": current_tenant,
                            "child": child,
                            "parent": parent,
                            "orphan_rows": orphan_count,
                            "passed": orphan_count == 0,
                        }
                    )
                connection.rollback()
    report["verification"] = {
        "tables": checks,
        "vectors": {
            **vector_summary,
            "target_rows": target_vectors,
            "row_count_match": target_vectors == vector_summary["source_rows"],
            "content_sha256_match": vector_content_match,
        },
        "foreign_keys": foreign_key_checks,
        "knowledge_embeddings": {},
        "passed": all(
            item["row_count_match"]
            and item["content_sha256_match"]
            and item["status_distribution_match"]
            and item["primary_key_match"]
            for item in checks
        )
        and target_vectors == vector_summary["source_rows"]
        and vector_content_match
        and all(item["passed"] for item in foreign_key_checks),
    }
    knowledge_data, knowledge_summary = knowledge_vector_rows(source_root, tenant_id)
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            set_tenant(cursor, tenant_id)
            cursor.execute(
                "SELECT chunk_id, model, text_sha256, dimension, "
                "encode(vector_send(embedding), 'hex') "
                "FROM vector.knowledge_embeddings"
            )
            target_knowledge_rows = [tuple(row) for row in cursor.fetchall()]
            target_knowledge = len(target_knowledge_rows)
            target_knowledge_digest = rows_digest(
                (*row[:4], _pg_vector_values(row[4], expected_dimension=int(row[3])))
                for row in target_knowledge_rows
            )
            connection.rollback()
    source_knowledge_rows = [tuple(row[1:]) for row in knowledge_data]
    knowledge_content_match = target_knowledge_digest == rows_digest(
        source_knowledge_rows
    )
    report["verification"]["knowledge_embeddings"] = {
        **knowledge_summary,
        "target_rows": target_knowledge,
        "row_count_match": target_knowledge == knowledge_summary["source_rows"],
        "content_sha256_match": knowledge_content_match,
    }
    report["verification"]["passed"] = (
        report["verification"]["passed"]
        and report["verification"]["knowledge_embeddings"]["row_count_match"]
        and knowledge_content_match
    )
    return report


def _reverse_value(value: object, column: str, mapping: TableMapping) -> object:
    if value is None:
        return None
    if column in JSON_COLUMNS or column.endswith("_json"):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if not isinstance(value, str) else value
    if isinstance(value, datetime):
        if mapping.database in {"operations.sqlite3", "orchestration.sqlite3", "review-cases.sqlite3"}:
            return value.timestamp()
        return value.astimezone(timezone.utc).isoformat()
    return value


def _clone_sqlite_template(
    template: Path,
    target: Path,
    tables: Iterable[str],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source_connection(template) as source, sqlite3.connect(target) as connection:
        source.backup(connection)
        # The immutable SQLite templates intentionally protect audit tables
        # with append-only triggers.  A rollback export is a newly-created
        # database that must be populated from PostgreSQL, so suspend those
        # triggers only while clearing/reloading the export, then restore the
        # exact trigger definitions before returning the file.
        trigger_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name IN ("
            + ", ".join("?" for _ in tables)
            + ") AND sql IS NOT NULL",
            tuple(tables),
        ).fetchall()
        for name, _ in trigger_rows:
            connection.execute(f"DROP TRIGGER IF EXISTS {quote_identifier(name)}")
        for table in tables:
            connection.execute(f"DELETE FROM {quote_identifier(table)}")
        for _, trigger_sql in trigger_rows:
            connection.execute(trigger_sql)
        connection.commit()


def _pg_vector_values(
    value: object,
    *,
    expected_dimension: int | None = None,
) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            values = tuple(float(item) for item in raw[1:-1].split(",") if item.strip())
        elif len(raw) % 2 == 0 and raw and all(
            character in "0123456789abcdefABCDEF" for character in raw
        ):
            value = bytes.fromhex(raw)
        else:
            raise ValueError("PostgreSQL vector text has invalid brackets")
    if isinstance(value, bytes):
        if len(value) < 4:
            raise ValueError("PostgreSQL vector wire value is truncated")
        dimension = int.from_bytes(value[:2], "big")
        if value[2:4] != b"\x00\x00" or len(value) != 4 + dimension * 4:
            raise ValueError("PostgreSQL vector wire value has invalid header")
        values = struct.unpack(f">{dimension}f", value[4:])
    elif isinstance(value, (list, tuple)):
        values = tuple(float(item) for item in value)
    elif not isinstance(value, str):
        raise ValueError("PostgreSQL vector value has an unsupported type")
    if expected_dimension is not None and len(values) != expected_dimension:
        raise ValueError("PostgreSQL vector dimension does not match metadata")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("PostgreSQL vector contains a non-finite value")
    return values


def reverse_export(
    source_root: Path,
    export_root: Path,
    dsn: str,
    tenant_id: str,
) -> dict[str, Any]:
    psycopg, _ = import_psycopg()
    if export_root == source_root:
        raise ValueError("--export-root must differ from --source-root")
    exported: list[dict[str, Any]] = []
    by_database: dict[str, list[TableMapping]] = {}
    for mapping in TABLES:
        by_database.setdefault(mapping.database, []).append(mapping)
    for database, mappings in by_database.items():
        _clone_sqlite_template(
            source_root / database,
            export_root / database,
            [mapping.source_table for mapping in mappings],
        )
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            for mapping in TABLES:
                original_columns = source_columns(mapping, source_root)
                target_columns = ( ["tenant_id"] if mapping.add_tenant else [] ) + [
                    column for column in original_columns if column not in mapping.skip_source_columns
                ]
                set_tenant(cursor, tenant_id)
                cursor.execute(
                    f"SELECT {', '.join(quote_identifier(column) for column in target_columns)} "
                    f"FROM {quote_identifier(mapping.target_schema)}.{quote_identifier(mapping.target_table)}"
                )
                rows = cursor.fetchall()
                connection.rollback()
                output = export_root / mapping.database
                sqlite_columns = [
                    column for column in original_columns if column not in mapping.skip_source_columns
                ]
                insert_sql = (
                    f"INSERT INTO {quote_identifier(mapping.source_table)} "
                    f"({', '.join(quote_identifier(column) for column in sqlite_columns)}) "
                    f"VALUES ({', '.join('?' for _ in sqlite_columns)})"
                )
                with sqlite3.connect(output) as sqlite_connection:
                    sqlite_connection.executemany(
                        insert_sql,
                        [
                            tuple(
                                _reverse_value(value, column, mapping)
                                for column, value in zip(sqlite_columns, row[1:] if mapping.add_tenant else row)
                            )
                            for row in rows
                        ],
                    )
                    sqlite_connection.commit()
                exported.append(
                    {
                        "database": mapping.database,
                        "table": mapping.source_table,
                        "target": f"{mapping.target_schema}.{mapping.target_table}",
                        "exported_rows": len(rows),
                    }
                )

            connection.rollback()
    vector_exports: list[dict[str, Any]] = []
    for database, table, expected_model, expected_dimension in VECTOR_SOURCES:
        with psycopg.connect(dsn, autocommit=False) as connection:
            with connection.cursor() as cursor:
                set_tenant(cursor, tenant_id)
                cursor.execute(
                    "SELECT cache_key, model_name, embedding_kind, text_sha256, dimension, "
                    "encode(vector_send(embedding), 'hex') "
                    "FROM vector.embedding_cache "
                    "WHERE model_name = %s AND dimension = %s ORDER BY cache_key",
                    (expected_model, expected_dimension),
                )
                vector_data = cursor.fetchall()
                connection.rollback()
        vector_output = export_root / database
        _clone_sqlite_template(source_root / database, vector_output, [table])
        with sqlite3.connect(vector_output) as sqlite_connection:
            sqlite_connection.executemany(
                f"INSERT INTO {quote_identifier(table)} "
                "(cache_key, model_name, embedding_kind, text_sha256, dimension, vector) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        *row[:5],
                        struct.pack(
                            f"<{expected_dimension}f",
                            *_pg_vector_values(row[5], expected_dimension=expected_dimension),
                        ),
                    )
                    for row in vector_data
                ],
            )
            sqlite_connection.commit()
        vector_exports.append(
            {
                "database": database,
                "model": expected_model,
                "dimension": expected_dimension,
                "exported_rows": len(vector_data),
            }
        )
    return {
        "mode": "reverse-export",
        "source_root": str(source_root),
        "export_root": str(export_root),
        "tenant_id": tenant_id,
        "tables": exported,
        "vectors": vector_exports,
        "exported_row_count": sum(item["exported_rows"] for item in exported)
        + sum(item["exported_rows"] for item in vector_exports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--reverse-export", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT.parent / "migration" / "taskforge-migration-report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.tenant_id.strip():
        raise SystemExit("--tenant-id must not be blank")
    if not 1 <= args.batch_size <= 10_000:
        raise SystemExit("--batch-size must be between 1 and 10000")
    source_root = args.source_root.resolve()
    report_path = args.report.resolve()
    try:
        report_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise SystemExit("--report must be outside --source-root so SQLite sources cannot be overwritten")
    dsn = (args.database_url or os.getenv("TASKFORGE_DATABASE_URL", "")).strip()
    if args.execute or args.verify or args.reverse_export:
        if not dsn:
            raise SystemExit("--database-url or TASKFORGE_DATABASE_URL is required")
    if args.reverse_export and args.export_root is None:
        raise SystemExit("--export-root is required with --reverse-export")
    if args.reverse_export and args.export_root.resolve() == source_root:
        raise SystemExit("--export-root must differ from --source-root")
    if args.dry_run:
        result = inventory(source_root, args.tenant_id)
        result["mode"] = "dry-run"
    elif args.execute:
        result = execute_import(
            source_root,
            dsn,
            args.tenant_id,
            args.batch_size,
            progress_report=report_path,
        )
        result["mode"] = "execute"
    elif args.reverse_export:
        result = reverse_export(source_root, args.export_root.resolve(), dsn, args.tenant_id)
    else:
        result = verify_import(source_root, dsn, args.tenant_id)
        result["mode"] = "verify"
    write_report(report_path, result)
    print(json.dumps({"mode": result["mode"], "report": str(report_path), "source_row_count": result.get("source_row_count"), "passed": result.get("verification", {}).get("passed")}, ensure_ascii=False))
    return 0 if result.get("verification", {}).get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
