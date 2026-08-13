"""Tenant-isolated durable state for literature discovery and research scopes."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import utc_now
from ..research_protocol import (
    ClaimRecord,
    EvidenceCard,
    IngestionStatus,
    LiteratureRequest,
    PaperCard,
    ResearchScope,
    ScopeExpansionRequest,
    SearchQuery,
)
from .models import DiscoveryResult


class LiteratureRepositoryError(RuntimeError):
    pass


class LiteratureNotFoundError(LiteratureRepositoryError):
    pass


class LiteratureConflictError(LiteratureRepositoryError):
    pass


class LiteratureAccessError(LiteratureRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class LiteratureAccess:
    tenant_id: str
    user_id: str
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.user_id.strip():
            raise ValueError("tenant_id and user_id are required")


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS literature_requests (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id)
);

CREATE TABLE IF NOT EXISTS literature_queries (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    priority INTEGER NOT NULL,
    query_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, query_id),
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES literature_requests(tenant_id, request_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS paper_catalog (
    tenant_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    card_json TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    full_text_status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_identifiers (
    tenant_id TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    identifier_value TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id, identifier_type, identifier_value),
    FOREIGN KEY (tenant_id, paper_id)
        REFERENCES paper_catalog(tenant_id, paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS paper_search_results (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    relevance_score REAL NOT NULL,
    matched_queries_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, paper_id),
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES literature_requests(tenant_id, request_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, paper_id)
        REFERENCES paper_catalog(tenant_id, paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_scopes (
    tenant_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    owner_user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    status TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    confirmed_at TEXT,
    PRIMARY KEY (tenant_id, scope_id, scope_version),
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES literature_requests(tenant_id, request_id)
);

CREATE TABLE IF NOT EXISTS research_scope_papers (
    tenant_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    selection_status TEXT NOT NULL CHECK(selection_status IN ('selected', 'excluded')),
    PRIMARY KEY (tenant_id, scope_id, scope_version, paper_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version)
        REFERENCES research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, paper_id)
        REFERENCES paper_catalog(tenant_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_ingestion_jobs (
    tenant_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    status_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, scope_id, scope_version, paper_id),
    UNIQUE (tenant_id, job_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version)
        REFERENCES research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_cards (
    tenant_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    card_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, evidence_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version)
        REFERENCES research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_records (
    tenant_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    claim_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, claim_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version)
        REFERENCES research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scope_expansion_requests (
    tenant_id TEXT NOT NULL,
    expansion_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    scope_version INTEGER NOT NULL,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    PRIMARY KEY (tenant_id, expansion_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version)
        REFERENCES research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature_audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS literature_results_request_idx
    ON paper_search_results(tenant_id, request_id, rank);
CREATE INDEX IF NOT EXISTS research_scopes_owner_idx
    ON research_scopes(tenant_id, owner_user_id, conversation_id, scope_id, scope_version);
CREATE INDEX IF NOT EXISTS evidence_scope_idx
    ON evidence_cards(tenant_id, scope_id, scope_version, paper_id);
CREATE INDEX IF NOT EXISTS expansion_scope_idx
    ON scope_expansion_requests(tenant_id, scope_id, scope_version, created_at);
"""


def _json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteLiteratureRepository:
    """One host-owned repository; all reads require a tenant/user access projection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        access: LiteratureAccess,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO literature_audit_events (
                tenant_id, user_id, action, resource_type, resource_id,
                details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                access.tenant_id,
                access.user_id,
                action,
                resource_type,
                resource_id,
                _json(details or {}),
                utc_now().isoformat(),
            ),
        )

    def save_request(
        self,
        access: LiteratureAccess,
        request: LiteratureRequest,
    ) -> LiteratureRequest:
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO literature_requests (
                        tenant_id, request_id, user_id, conversation_id,
                        request_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.tenant_id,
                        request.request_id,
                        access.user_id,
                        access.conversation_id,
                        _json(request),
                        request.created_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LiteratureConflictError("literature request already exists") from exc
            self._audit(connection, access, "create", "literature_request", request.request_id)
        return request

    def get_request(self, access: LiteratureAccess, request_id: str) -> LiteratureRequest:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_json FROM literature_requests
                WHERE tenant_id = ? AND user_id = ? AND request_id = ?
                """,
                (access.tenant_id, access.user_id, request_id),
            ).fetchone()
        if row is None:
            raise LiteratureNotFoundError("literature request not found")
        return LiteratureRequest.model_validate_json(row["request_json"])

    def save_queries(
        self,
        access: LiteratureAccess,
        request_id: str,
        queries: list[SearchQuery],
    ) -> None:
        self.get_request(access, request_id)
        now = utc_now().isoformat()
        with self._lock, self._connect() as connection:
            for query in queries:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO literature_queries (
                        tenant_id, request_id, query_id, priority, query_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.tenant_id,
                        request_id,
                        query.query_id,
                        query.priority,
                        _json(query),
                        now,
                    ),
                )
            self._audit(
                connection,
                access,
                "save_queries",
                "literature_request",
                request_id,
                {"count": len(queries)},
            )

    @staticmethod
    def _identifiers(card: PaperCard) -> list[tuple[str, str]]:
        values = {
            "doi": card.doi,
            "arxiv": card.arxiv_id,
            "semantic_scholar": card.semantic_scholar_id,
            "openalex": card.openalex_id,
        }
        return [(kind, value.casefold()) for kind, value in values.items() if value]

    def _upsert_paper(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        card: PaperCard,
    ) -> None:
        for kind, value in self._identifiers(card):
            owner = connection.execute(
                """
                SELECT paper_id FROM paper_identifiers
                WHERE tenant_id = ? AND identifier_type = ? AND identifier_value = ?
                """,
                (tenant_id, kind, value),
            ).fetchone()
            if owner is not None and owner["paper_id"] != card.paper_id:
                raise LiteratureConflictError(
                    f"{kind} identifier already belongs to another canonical paper"
                )
        connection.execute(
            """
            INSERT INTO paper_catalog (
                tenant_id, paper_id, card_json, verification_status,
                full_text_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, paper_id) DO UPDATE SET
                card_json = excluded.card_json,
                verification_status = excluded.verification_status,
                full_text_status = excluded.full_text_status,
                updated_at = excluded.updated_at
            """,
            (
                tenant_id,
                card.paper_id,
                _json(card),
                card.verification_status,
                card.full_text_status,
                utc_now().isoformat(),
            ),
        )
        for kind, value in self._identifiers(card):
            connection.execute(
                """
                INSERT OR REPLACE INTO paper_identifiers (
                    tenant_id, identifier_type, identifier_value, paper_id
                ) VALUES (?, ?, ?, ?)
                """,
                (tenant_id, kind, value, card.paper_id),
            )

    def save_discovery(
        self,
        access: LiteratureAccess,
        result: DiscoveryResult,
    ) -> None:
        self.get_request(access, result.request_id)
        self.save_queries(access, result.request_id, result.queries)
        with self._lock, self._connect() as connection:
            for rank, card in enumerate(result.papers, start=1):
                self._upsert_paper(connection, access.tenant_id, card)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO paper_search_results (
                        tenant_id, request_id, paper_id, rank, relevance_score,
                        matched_queries_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.tenant_id,
                        result.request_id,
                        card.paper_id,
                        rank,
                        card.relevance_score,
                        _json(card.matched_queries),
                        result.created_at.isoformat(),
                    ),
                )
            self._audit(
                connection,
                access,
                "save_discovery",
                "literature_request",
                result.request_id,
                {"paper_count": len(result.papers)},
            )

    def list_papers(
        self,
        access: LiteratureAccess,
        request_id: str,
        *,
        limit: int = 100,
    ) -> list[PaperCard]:
        self.get_request(access, request_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.card_json FROM paper_search_results r
                JOIN paper_catalog p
                  ON p.tenant_id = r.tenant_id AND p.paper_id = r.paper_id
                WHERE r.tenant_id = ? AND r.request_id = ?
                ORDER BY r.rank ASC LIMIT ?
                """,
                (access.tenant_id, request_id, min(max(1, limit), 100)),
            ).fetchall()
        return [PaperCard.model_validate_json(row["card_json"]) for row in rows]

    def get_paper(self, access: LiteratureAccess, paper_id: str) -> PaperCard:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT card_json FROM paper_catalog
                WHERE tenant_id = ? AND paper_id = ?
                """,
                (access.tenant_id, paper_id),
            ).fetchone()
        if row is None:
            raise LiteratureNotFoundError("paper not found")
        return PaperCard.model_validate_json(row["card_json"])

    def upsert_paper(self, access: LiteratureAccess, card: PaperCard) -> None:
        with self._lock, self._connect() as connection:
            self._upsert_paper(connection, access.tenant_id, card)
            self._audit(connection, access, "upsert", "paper", card.paper_id)

    def create_scope(self, access: LiteratureAccess, scope: ResearchScope) -> ResearchScope:
        if scope.tenant_id != access.tenant_id or scope.owner_user_id != access.user_id:
            raise LiteratureAccessError("scope authority does not match the caller")
        if access.conversation_id and scope.conversation_id != access.conversation_id:
            raise LiteratureAccessError("scope conversation does not match the caller")
        self.get_request(access, scope.request_id)
        if scope.scope_version != 1:
            raise LiteratureConflictError("a new scope must start at version 1")
        with self._lock, self._connect() as connection:
            self._validate_scope_papers(connection, scope)
            try:
                self._insert_scope(connection, scope)
            except sqlite3.IntegrityError as exc:
                raise LiteratureConflictError("research scope already exists") from exc
            self._audit(connection, access, "create", "research_scope", scope.scope_id)
        return scope

    @staticmethod
    def _validate_scope_papers(connection: sqlite3.Connection, scope: ResearchScope) -> None:
        expected = set(scope.selected_paper_ids) | set(scope.excluded_paper_ids)
        if not expected:
            return
        placeholders = ",".join("?" for _ in expected)
        rows = connection.execute(
            f"SELECT paper_id FROM paper_catalog WHERE tenant_id = ? AND paper_id IN ({placeholders})",
            (scope.tenant_id, *sorted(expected)),
        ).fetchall()
        missing = expected - {row["paper_id"] for row in rows}
        if missing:
            raise LiteratureNotFoundError(
                "scope contains unknown paper IDs: " + ", ".join(sorted(missing))
            )

    @staticmethod
    def _insert_scope(connection: sqlite3.Connection, scope: ResearchScope) -> None:
        connection.execute(
            """
            INSERT INTO research_scopes (
                tenant_id, scope_id, scope_version, owner_user_id,
                conversation_id, request_id, status, scope_json,
                created_at, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope.tenant_id,
                scope.scope_id,
                scope.scope_version,
                scope.owner_user_id,
                scope.conversation_id,
                scope.request_id,
                scope.status,
                _json(scope),
                scope.created_at.isoformat(),
                scope.confirmed_at.isoformat() if scope.confirmed_at else None,
            ),
        )
        for paper_id in scope.selected_paper_ids:
            connection.execute(
                """
                INSERT INTO research_scope_papers (
                    tenant_id, scope_id, scope_version, paper_id, selection_status
                ) VALUES (?, ?, ?, ?, 'selected')
                """,
                (scope.tenant_id, scope.scope_id, scope.scope_version, paper_id),
            )
        for paper_id in scope.excluded_paper_ids:
            connection.execute(
                """
                INSERT INTO research_scope_papers (
                    tenant_id, scope_id, scope_version, paper_id, selection_status
                ) VALUES (?, ?, ?, ?, 'excluded')
                """,
                (scope.tenant_id, scope.scope_id, scope.scope_version, paper_id),
            )

    def _get_scope_row(
        self,
        connection: sqlite3.Connection,
        access: LiteratureAccess,
        scope_id: str,
        version: int | None = None,
    ) -> sqlite3.Row | None:
        conversation_sql = ""
        params: list[object] = [access.tenant_id, access.user_id, scope_id]
        if access.conversation_id is not None:
            conversation_sql = " AND conversation_id = ?"
            params.append(access.conversation_id)
        version_sql = ""
        if version is not None:
            version_sql = " AND scope_version = ?"
            params.append(version)
        return connection.execute(
            """
            SELECT scope_json FROM research_scopes
            WHERE tenant_id = ? AND owner_user_id = ? AND scope_id = ?
            """ + conversation_sql + version_sql + " ORDER BY scope_version DESC LIMIT 1",
            tuple(params),
        ).fetchone()

    def get_scope(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        version: int | None = None,
    ) -> ResearchScope:
        with self._connect() as connection:
            row = self._get_scope_row(connection, access, scope_id, version)
        if row is None:
            raise LiteratureNotFoundError("research scope not found")
        return ResearchScope.model_validate_json(row["scope_json"])

    def list_scopes(self, access: LiteratureAccess) -> list[ResearchScope]:
        conversation_sql = ""
        params: list[object] = [access.tenant_id, access.user_id]
        if access.conversation_id is not None:
            conversation_sql = " AND s.conversation_id = ?"
            params.append(access.conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.scope_json FROM research_scopes s
                JOIN (
                    SELECT tenant_id, scope_id, MAX(scope_version) AS latest_version
                    FROM research_scopes GROUP BY tenant_id, scope_id
                ) latest ON latest.tenant_id = s.tenant_id
                    AND latest.scope_id = s.scope_id
                    AND latest.latest_version = s.scope_version
                WHERE s.tenant_id = ? AND s.owner_user_id = ?
                """ + conversation_sql + " ORDER BY s.created_at DESC",
                tuple(params),
            ).fetchall()
        return [ResearchScope.model_validate_json(row["scope_json"]) for row in rows]

    def update_scope(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        selected_paper_ids: list[str] | None = None,
        excluded_paper_ids: list[str] | None = None,
        user_intent: str | None = None,
        allowed_expansion: bool | None = None,
        status: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchScope:
        current = self.get_scope(access, scope_id)
        if expected_version is not None and current.scope_version != expected_version:
            raise LiteratureConflictError("research scope version changed")
        target_status = status or current.status
        confirmed_at = current.confirmed_at
        if target_status != "draft" and confirmed_at is None:
            confirmed_at = utc_now()
        update: dict[str, Any] = {
            "scope_version": current.scope_version + 1,
            "created_at": utc_now(),
            "confirmed_at": confirmed_at,
            "status": target_status,
        }
        if selected_paper_ids is not None:
            update["selected_paper_ids"] = selected_paper_ids
        if excluded_paper_ids is not None:
            update["excluded_paper_ids"] = excluded_paper_ids
        if user_intent is not None:
            update["user_intent"] = user_intent
        if allowed_expansion is not None:
            update["allowed_expansion"] = allowed_expansion
        revised = current.model_copy(update=update)
        revised = ResearchScope.model_validate(revised.model_dump())
        with self._lock, self._connect() as connection:
            latest = self._get_scope_row(connection, access, scope_id)
            if latest is None:
                raise LiteratureNotFoundError("research scope not found")
            latest_scope = ResearchScope.model_validate_json(latest["scope_json"])
            if latest_scope.scope_version != current.scope_version:
                raise LiteratureConflictError("research scope version changed")
            self._validate_scope_papers(connection, revised)
            self._insert_scope(connection, revised)
            self._audit(
                connection,
                access,
                "new_version",
                "research_scope",
                scope_id,
                {"version": revised.scope_version, "status": revised.status},
            )
        return revised

    def transition_scope_status(
        self,
        access: LiteratureAccess,
        scope_id: str,
        status: str,
        *,
        expected_version: int | None = None,
    ) -> ResearchScope:
        """Change lifecycle state without changing the immutable paper boundary."""

        current = self.get_scope(access, scope_id)
        if expected_version is not None and current.scope_version != expected_version:
            raise LiteratureConflictError("research scope version changed")
        allowed: dict[str, set[str]] = {
            "draft": {"confirmed", "closed"},
            "confirmed": {"ingesting", "closed"},
            "ingesting": {"ready", "closed"},
            "ready": {"expansion_requested", "closed"},
            "expansion_requested": {"ready", "closed"},
            "closed": set(),
        }
        if status not in allowed.get(current.status, set()):
            raise LiteratureConflictError(
                f"invalid research scope transition: {current.status} -> {status}"
            )
        confirmed_at = current.confirmed_at or (utc_now() if status != "draft" else None)
        revised = ResearchScope.model_validate(
            current.model_copy(
                update={"status": status, "confirmed_at": confirmed_at}
            ).model_dump()
        )
        with self._lock, self._connect() as connection:
            latest = self._get_scope_row(connection, access, scope_id)
            if latest is None:
                raise LiteratureNotFoundError("research scope not found")
            latest_scope = ResearchScope.model_validate_json(latest["scope_json"])
            if latest_scope.scope_version != current.scope_version:
                raise LiteratureConflictError("research scope version changed")
            connection.execute(
                """
                UPDATE research_scopes
                SET status = ?, scope_json = ?, confirmed_at = ?
                WHERE tenant_id = ? AND scope_id = ? AND scope_version = ?
                """,
                (
                    revised.status,
                    _json(revised),
                    revised.confirmed_at.isoformat() if revised.confirmed_at else None,
                    access.tenant_id,
                    revised.scope_id,
                    revised.scope_version,
                ),
            )
            self._audit(
                connection,
                access,
                "transition_status",
                "research_scope",
                scope_id,
                {"from": current.status, "to": revised.status},
            )
        return revised

    def save_ingestion_status(
        self,
        access: LiteratureAccess,
        status: IngestionStatus,
        *,
        scope_version: int | None = None,
    ) -> None:
        scope = self.get_scope(access, status.scope_id, version=scope_version)
        if status.paper_id not in scope.selected_paper_ids:
            raise LiteratureAccessError("ingestion paper is outside the selected scope")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_ingestion_jobs (
                    tenant_id, scope_id, scope_version, paper_id,
                    job_id, status_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, scope_id, scope_version, paper_id) DO UPDATE SET
                    job_id = excluded.job_id,
                    status_json = excluded.status_json,
                    updated_at = excluded.updated_at
                """,
                (
                    access.tenant_id,
                    scope.scope_id,
                    scope.scope_version,
                    status.paper_id,
                    status.job_id,
                    _json(status),
                    status.updated_at.isoformat(),
                ),
            )
            self._audit(connection, access, "ingestion_status", "paper", status.paper_id)

    def list_ingestion_statuses(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        version: int | None = None,
    ) -> list[IngestionStatus]:
        scope = self.get_scope(access, scope_id, version=version)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status_json FROM paper_ingestion_jobs
                WHERE tenant_id = ? AND scope_id = ? AND scope_version = ?
                ORDER BY paper_id
                """,
                (access.tenant_id, scope.scope_id, scope.scope_version),
            ).fetchall()
        return [IngestionStatus.model_validate_json(row["status_json"]) for row in rows]

    def save_evidence(self, access: LiteratureAccess, cards: list[EvidenceCard]) -> None:
        if not cards:
            return
        scope_ids = {card.scope_id for card in cards}
        versions = {card.scope_version for card in cards}
        if len(scope_ids) != 1 or len(versions) != 1 or None in scope_ids or None in versions:
            raise LiteratureConflictError("an evidence batch must belong to one scope version")
        scope = self.get_scope(access, next(iter(scope_ids)), version=next(iter(versions)))  # type: ignore[arg-type]
        with self._lock, self._connect() as connection:
            for card in cards:
                if card.paper_id not in scope.selected_paper_ids:
                    raise LiteratureAccessError("evidence paper is outside the selected scope")
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evidence_cards (
                        tenant_id, evidence_id, scope_id, scope_version,
                        paper_id, card_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.tenant_id,
                        card.evidence_id,
                        scope.scope_id,
                        scope.scope_version,
                        card.paper_id,
                        _json(card),
                        utc_now().isoformat(),
                    ),
                )
            self._audit(
                connection,
                access,
                "save_evidence",
                "research_scope",
                scope.scope_id,
                {"count": len(cards), "version": scope.scope_version},
            )

    def list_evidence(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        version: int | None = None,
        paper_id: str | None = None,
    ) -> list[EvidenceCard]:
        scope = self.get_scope(access, scope_id, version=version)
        paper_sql = ""
        params: list[object] = [access.tenant_id, scope.scope_id, scope.scope_version]
        if paper_id is not None:
            paper_sql = " AND paper_id = ?"
            params.append(paper_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT card_json FROM evidence_cards
                WHERE tenant_id = ? AND scope_id = ? AND scope_version = ?
                """ + paper_sql + " ORDER BY evidence_id",
                tuple(params),
            ).fetchall()
        return [EvidenceCard.model_validate_json(row["card_json"]) for row in rows]

    def save_claims(self, access: LiteratureAccess, claims: list[ClaimRecord]) -> None:
        if not claims:
            return
        for claim in claims:
            if claim.scope_id is None or claim.scope_version is None:
                raise LiteratureConflictError("claim must carry a scope version")
            scope = self.get_scope(access, claim.scope_id, version=claim.scope_version)
            if set(claim.paper_ids) - set(scope.selected_paper_ids):
                raise LiteratureAccessError("claim cites a paper outside the selected scope")
            available = {card.evidence_id for card in self.list_evidence(access, scope.scope_id, version=scope.scope_version)}
            if set(claim.evidence_ids) - available:
                raise LiteratureNotFoundError("claim cites unknown evidence")
        with self._lock, self._connect() as connection:
            for claim in claims:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO claim_records (
                        tenant_id, claim_id, scope_id, scope_version,
                        claim_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.tenant_id,
                        claim.claim_id,
                        claim.scope_id,
                        claim.scope_version,
                        _json(claim),
                        utc_now().isoformat(),
                    ),
                )
            first = claims[0]
            self._audit(
                connection,
                access,
                "save_claims",
                "research_scope",
                first.scope_id or "unknown",
                {"count": len(claims)},
            )

    def list_claims(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        version: int | None = None,
    ) -> list[ClaimRecord]:
        scope = self.get_scope(access, scope_id, version=version)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT claim_json FROM claim_records
                WHERE tenant_id = ? AND scope_id = ? AND scope_version = ?
                ORDER BY claim_id
                """,
                (access.tenant_id, scope.scope_id, scope.scope_version),
            ).fetchall()
        return [ClaimRecord.model_validate_json(row["claim_json"]) for row in rows]

    def request_expansion(
        self,
        access: LiteratureAccess,
        request: ScopeExpansionRequest,
    ) -> ScopeExpansionRequest:
        scope = self.get_scope(access, request.scope_id)
        if request.status != "pending":
            raise LiteratureConflictError("new expansion request must be pending")
        if scope.status != "ready":
            raise LiteratureConflictError("scope must be ready before requesting expansion")
        if not scope.allowed_expansion:
            raise LiteratureAccessError("scope expansion is not enabled by the user")
        if set(request.proposed_paper_ids) & set(scope.selected_paper_ids):
            raise LiteratureConflictError("proposed papers are already selected")
        for paper_id in request.proposed_paper_ids:
            self.get_paper(access, paper_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scope_expansion_requests (
                    tenant_id, expansion_id, scope_id, scope_version,
                    request_json, created_at, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    access.tenant_id,
                    request.expansion_id,
                    scope.scope_id,
                    scope.scope_version,
                    _json(request),
                    request.created_at.isoformat(),
                ),
            )
            self._audit(
                connection,
                access,
                "request_expansion",
                "research_scope",
                scope.scope_id,
                {"expansion_id": request.expansion_id},
            )
        return request

    def decide_expansion(
        self,
        access: LiteratureAccess,
        expansion_id: str,
        *,
        approve: bool,
        expected_scope_id: str | None = None,
    ) -> ScopeExpansionRequest:
        with self._lock, self._connect() as connection:
            scope_filter = "" if expected_scope_id is None else " AND e.scope_id = ?"
            parameters: list[object] = [access.tenant_id, expansion_id, access.user_id]
            if expected_scope_id is not None:
                parameters.append(expected_scope_id)
            row = connection.execute(
                """
                SELECT e.request_json, e.scope_id, e.scope_version
                FROM scope_expansion_requests e
                JOIN research_scopes s ON s.tenant_id = e.tenant_id
                  AND s.scope_id = e.scope_id AND s.scope_version = e.scope_version
                WHERE e.tenant_id = ? AND e.expansion_id = ?
                  AND s.owner_user_id = ?
                """ + scope_filter,
                tuple(parameters),
            ).fetchone()
            if row is None:
                raise LiteratureNotFoundError("scope expansion request not found")
            current = ScopeExpansionRequest.model_validate_json(row["request_json"])
            if current.status != "pending":
                raise LiteratureConflictError("scope expansion request is already decided")
            decided_at = utc_now()
            revised = current.model_copy(
                update={
                    "status": "approved" if approve else "rejected",
                    "decided_at": decided_at,
                }
            )
            connection.execute(
                """
                UPDATE scope_expansion_requests
                SET request_json = ?, decided_at = ?
                WHERE tenant_id = ? AND expansion_id = ?
                """,
                (_json(revised), decided_at.isoformat(), access.tenant_id, expansion_id),
            )
            self._audit(
                connection,
                access,
                "approve_expansion" if approve else "reject_expansion",
                "scope_expansion",
                expansion_id,
            )
        return revised

    def list_audit_events(
        self,
        access: LiteratureAccess,
        *,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, user_id, action, resource_type, resource_id,
                       details_json, created_at
                FROM literature_audit_events
                WHERE tenant_id = ?
                ORDER BY event_id DESC LIMIT ?
                """,
                (access.tenant_id, min(max(1, limit), 1000)),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "user_id": row["user_id"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "details": json.loads(row["details_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


__all__ = [
    "LiteratureAccess",
    "LiteratureAccessError",
    "LiteratureConflictError",
    "LiteratureNotFoundError",
    "LiteratureRepositoryError",
    "SQLiteLiteratureRepository",
]
