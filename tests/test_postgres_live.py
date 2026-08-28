"""Opt-in integration checks for a real TaskForge PostgreSQL database.

Set ``TASKFORGE_LIVE_DATABASE_URL`` to the least-privileged ``taskforge_app``
DSN after the schema migration has been applied.  The test intentionally uses
one transaction and rolls it back so it does not create durable fixture rows.
"""

from __future__ import annotations

import os

import pytest

DSN = os.getenv("TASKFORGE_LIVE_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN,
    reason="TASKFORGE_LIVE_DATABASE_URL is not configured",
)


def test_taskforge_postgres_rls_and_pgvector() -> None:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - only when a live DSN is set
        pytest.fail(f"TASKFORGE_LIVE_DATABASE_URL is set but psycopg is unavailable: {exc}")
    tenant_a = "live-taskforge-a"
    tenant_b = "live-taskforge-b"
    with psycopg.connect(DSN, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config(%s, %s, true)", ("taskforge.tenant_id", tenant_a))
            cursor.execute("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
            assert float(cursor.fetchone()[0]) == pytest.approx(1.0)
            cursor.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'vector' "
                "AND indexname = 'knowledge_embeddings_hnsw_cosine_idx'"
            )
            assert cursor.fetchone() is not None

            cursor.execute(
                "INSERT INTO core.tasks(tenant_id, task_id, task_json, created_at) "
                "VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)",
                (tenant_a, "live-task", '{"goal":"integration"}'),
            )
            cursor.execute(
                "INSERT INTO core.profiles(tenant_id, profile_id, profile_json, updated_at) "
                "VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)",
                (tenant_a, "live-profile", '{"name":"integration"}'),
            )
            cursor.execute(
                "INSERT INTO core.runs(tenant_id, run_id, task_id, profile_id, state_json, version, updated_at) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, 1, CURRENT_TIMESTAMP)",
                (
                    tenant_a,
                    "live-run",
                    "live-task",
                    "live-profile",
                    '{"status":"pending"}',
                ),
            )
            cursor.execute("SELECT COUNT(*) FROM core.runs")
            assert cursor.fetchone()[0] == 1

            cursor.execute("SELECT set_config(%s, %s, true)", ("taskforge.tenant_id", tenant_b))
            cursor.execute("SELECT COUNT(*) FROM core.runs")
            assert cursor.fetchone()[0] == 0
        connection.rollback()
