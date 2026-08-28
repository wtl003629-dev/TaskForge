"""Pooled PostgreSQL runtime primitives shared by TaskForge stores."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_REQUEST_TENANT: ContextVar[str] = ContextVar("taskforge_request_tenant", default="local")


def set_request_tenant(tenant_id: str) -> None:
    """Bind the host-authenticated tenant for request-scoped cache adapters."""

    cleaned = tenant_id.strip()
    if not cleaned:
        raise ValueError("tenant_id is required")
    _REQUEST_TENANT.set(cleaned)


def current_request_tenant() -> str:
    """Return the current request tenant, defaulting to the local deployment."""

    return _REQUEST_TENANT.get()


class PostgresRuntimeError(RuntimeError):
    """Base error for the host-owned PostgreSQL runtime."""


class PostgresDependencyError(PostgresRuntimeError):
    """Raised when the PostgreSQL optional dependencies are unavailable."""


class PostgresBackendNotReadyError(PostgresRuntimeError):
    """Raised when the configured PostgreSQL backend cannot be started."""


class PostgresRuntime:
    """Own a psycopg connection pool without running schema DDL at startup."""

    def __init__(
        self,
        dsn: str,
        *,
        pool: object | None = None,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        if not dsn.strip():
            raise ValueError("PostgreSQL DSN is required")
        if not 1 <= min_size <= max_size <= 64:
            raise ValueError("PostgreSQL pool sizes are invalid")
        if not 0 < connect_timeout_seconds <= 60:
            raise ValueError("PostgreSQL connect timeout is invalid")
        self.dsn = dsn
        self._owns_pool = pool is None
        if pool is None:
            try:
                pool_type = importlib.import_module("psycopg_pool").ConnectionPool
                row_factory = importlib.import_module("psycopg.rows").dict_row
            except (ImportError, ModuleNotFoundError) as exc:
                raise PostgresDependencyError(
                    "PostgreSQL support requires `pip install taskforge-agent[postgres]`"
                ) from exc
            pool = pool_type(
                conninfo=dsn,
                min_size=min_size,
                max_size=max_size,
                kwargs={
                    "connect_timeout": connect_timeout_seconds,
                    "row_factory": row_factory,
                },
                # Keep request-side acquisition bounded by the same
                # fail-closed connection budget as new connections, while
                # allowing the background pool to survive a short outage and
                # reconnect after the database returns.
                timeout=connect_timeout_seconds,
                reconnect_timeout=max(30.0, connect_timeout_seconds * 6.0),
                # Validate idle connections before handing them to a request;
                # otherwise a socket broken while the pool was idle can be
                # returned to the caller and block inside cursor.execute().
                check=pool_type.check_connection,
                open=False,
            )
            pool.open(wait=True)
        self._pool = pool
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_pool:
            self._pool.close()  # type: ignore[attr-defined]
        self._closed = True

    @property
    def pool(self) -> object:
        """Return the pool for repositories that support pool injection."""

        if self._closed:
            raise PostgresRuntimeError("PostgreSQL runtime is closed")
        return self._pool

    @contextmanager
    def transaction(self, tenant_id: str) -> Iterator[tuple[Any, Any]]:
        """Checkout one connection and install a transaction-local tenant."""

        if self._closed:
            raise PostgresRuntimeError("PostgreSQL runtime is closed")
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        with self._pool.connection() as connection:  # type: ignore[attr-defined]
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config(%s, %s, true)",
                    ("taskforge.tenant_id", tenant_id),
                )
                yield connection, cursor


__all__ = [
    "PostgresBackendNotReadyError",
    "PostgresDependencyError",
    "PostgresRuntime",
    "PostgresRuntimeError",
    "current_request_tenant",
    "set_request_tenant",
]
