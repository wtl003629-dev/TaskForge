from __future__ import annotations

import pytest

from taskforge.app import create_app
from taskforge.config import Settings
from taskforge.postgres_runtime import PostgresBackendNotReadyError


def test_database_backend_defaults_to_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASKFORGE_DATABASE_BACKEND", raising=False)
    monkeypatch.delenv("TASKFORGE_DATABASE_URL", raising=False)
    assert Settings.model_fields["database_backend"].default == "postgres"
    settings = Settings(
        _env_file=None,
        database_url="postgresql://taskforge_app:secret@localhost/taskforge",
    )
    assert settings.database_backend == "postgres"


def test_default_postgres_backend_requires_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASKFORGE_DATABASE_BACKEND", raising=False)
    monkeypatch.delenv("TASKFORGE_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="TASKFORGE_DATABASE_URL"):
        Settings(_env_file=None)


def test_postgres_backend_requires_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASKFORGE_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="TASKFORGE_DATABASE_URL"):
        Settings(_env_file=None, database_backend="postgres")


def test_postgres_pool_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="postgres_pool_max_size"):
        Settings(_env_file=None, postgres_pool_min_size=4, postgres_pool_max_size=2)


def test_postgres_application_switch_fails_closed_until_store_cutover() -> None:
    settings = Settings(
        _env_file=None,
        database_backend="postgres",
        database_url="postgresql://taskforge_app:secret@localhost/taskforge",
    )
    with pytest.raises(PostgresBackendNotReadyError, match="no SQLite fallback"):
        create_app(settings)
