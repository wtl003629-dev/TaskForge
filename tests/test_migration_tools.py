from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.migrate_sqlite_to_postgres import (
    _clone_sqlite_template,
    status_distribution,
    validate_json_contract,
    validate_source_integrity,
)


def test_taskforge_reverse_export_clones_each_database_once(tmp_path) -> None:
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "rollback" / "source.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE first (value TEXT);
            CREATE TABLE second (value TEXT);
            INSERT INTO first VALUES ('old-first');
            INSERT INTO second VALUES ('old-second');
            """
        )

    _clone_sqlite_template(source, target, ["first", "second"])

    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM first").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM second").fetchone()[0] == 0


def test_taskforge_status_distribution_is_explicit_and_stable() -> None:
    columns = ["tenant_id", "status", "value"]
    rows = [
        ("local", "running", 1),
        ("local", "completed", 2),
        ("local", "running", 3),
    ]

    assert status_distribution(columns, rows) == {
        "completed": 1,
        "running": 2,
    }
    assert status_distribution(["tenant_id", "value"], rows) == {}


def test_taskforge_migration_validates_json_shape_and_enum_fields() -> None:
    assert validate_json_contract(
        {"status": "completed"},
        target_name="core.runs",
        column="state_json",
    ) == {"status": "completed"}
    with pytest.raises(ValueError, match="JSON object"):
        validate_json_contract(
            [],
            target_name="core.runs",
            column="state_json",
        )


def test_taskforge_migration_validates_source_primary_and_foreign_keys() -> None:
    source_tables = {
        "core.tasks": (
            ["tenant_id", "task_id"],
            [("local", "task-1")],
        ),
        "core.profiles": (
            ["tenant_id", "profile_id"],
            [("local", "profile-1")],
        ),
        "core.runs": (
            ["tenant_id", "run_id", "task_id", "profile_id"],
            [("local", "run-1", "task-1", "profile-1")],
        ),
    }
    result = validate_source_integrity(source_tables)
    assert result["passed"] is True
    assert len(result["foreign_keys"]) == 2

    source_tables["core.runs"] = (
        ["tenant_id", "run_id", "task_id", "profile_id"],
        [("local", "run-1", "missing", "profile-1")],
    )
    with pytest.raises(ValueError, match="foreign key validation failed"):
        validate_source_integrity(source_tables)
    with pytest.raises(ValueError, match="unsupported value"):
        validate_json_contract(
            {"status": "unknown"},
            target_name="core.runs",
            column="state_json",
        )


def test_taskforge_runtime_migration_covers_rls_audit_and_fixed_vector_index() -> None:
    root = Path(__file__).resolve().parents[1]
    sql = (root / "migrations/postgres/002_taskforge_runtime.sql").read_text(
        encoding="utf-8"
    )
    normalised = " ".join(sql.casefold().split())
    assert "create extension if not exists vector" in normalised
    assert "alter table %i.%i force row level security" in normalised
    assert "revoke update, delete on operations.audit_events" in normalised
    assert "request_hash text not null check (request_hash ~ '^[0-9a-f]{64}$')" in normalised
    assert "jsonb_typeof(task_json) = 'object'" in normalised
    assert "text_sha256 text not null check (text_sha256 ~ '^[0-9a-f]{64}$')" in normalised
    assert "check (vector_dims(embedding) = dimension)" in normalised
    hnsw_sql = (root / "migrations/postgres/003_taskforge_hnsw.sql").read_text(
        encoding="utf-8"
    )
    assert "knowledge_embeddings_hnsw_cosine_idx" in hnsw_sql.casefold()
    assert "knowledge_embeddings_hnsw_cosine_idx" not in normalised
    assert "embedding_cache_hnsw_cosine_idx" not in normalised
    assert "literature.audit_event_id_seq" in normalised


def test_taskforge_pgvector_search_keeps_authorization_predicates_before_ordering() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "backend/taskforge/postgres_context_store.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert "ke.embedding <=> %s::vector" in source
    assert "ke.tenant_id = %s" in source
    assert "kc.acl_json ?| %s::text[]" in source
    assert "order by {distance_expression}" in source


def test_taskforge_literature_audit_ids_are_preserved_for_idempotent_import() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/migrate_sqlite_to_postgres.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert '"literature.sqlite3", "literature_audit_events", "literature", "audit_events"' in source
    assert '"literature.audit_events": ("event_id",)' in source


def test_taskforge_rag_fixture_freezes_numpy_reference_and_model_identity() -> None:
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace / "migration" / "rag-query-vectors.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["fixture"] == "taskforge-rag-pgvector-gate"
    assert payload["model"] == "text-embedding-v4"
    assert payload["cache_model"] == "aliyun-bailian|text-embedding-v4|dense-v1|1024"
    assert payload["dimension"] == 1024
    assert payload["query_count"] == len(payload["queries"]) >= 64
    assert all(len(item["vector"]) == 1024 for item in payload["queries"])
    assert all(item["sqlite_numpy_top_k"] for item in payload["queries"])


def test_taskforge_rag_comparison_records_all_required_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/verify_pgvector_retrieval.py").read_text(
        encoding="utf-8"
    ).casefold()
    for marker in (
        "sqlite_numpy",
        "postgres_exact_vs_sqlite_numpy",
        "postgres_hnsw_vs_postgres_exact",
        "for k in (5, 10, 50)",
        "mrr_at_10",
        "ndcg_at_8",
        "ndcg_at_10",
        "agent_visible_recall_at_8",
        "bailian_api_calls",
        "database_query_count",
        "postgres_exact_p50",
        "postgres_hnsw_p95",
    ):
        assert marker in source


def test_taskforge_migration_includes_legacy_and_bailian_embedding_caches() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/migrate_sqlite_to_postgres.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert '"embeddings.sqlite3", "embeddings_v1", "baai/bge-small-en-v1.5", 384' in source
    assert '"embeddings-bailian-v4-1024.sqlite3"' in source
    assert '"vector.embedding_cache"' in source
    report = json.loads(
        (root.parent / "migration" / "taskforge-migration-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["vectors"]["source_rows"] == 4933
    assert report["vectors"]["dimensions"] == [384, 1024]
