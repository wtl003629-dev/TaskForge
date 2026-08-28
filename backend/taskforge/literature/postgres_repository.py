"""PostgreSQL transport for the TaskForge literature repository.

Literature models and authorization rules stay in ``repository.py``.  This
adapter uses the same JSON documents and version semantics as the SQLite
repository, while relying on PostgreSQL transactions, composite foreign keys,
and RLS for durable isolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ..domain import utc_now
from ..postgres_runtime import PostgresRuntime
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
from .repository import (
    LiteratureAccess,
    LiteratureAccessError,
    LiteratureConflictError,
    LiteratureNotFoundError,
)


class PostgresLiteratureRepository:
    """Pooled, tenant-scoped implementation of the literature store port."""

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

    def save_request(self, access: LiteratureAccess, request: LiteratureRequest) -> LiteratureRequest:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            try:
                cursor.execute(
                    """
                    INSERT INTO literature.literature_requests(
                        tenant_id, request_id, user_id, conversation_id,
                        request_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (access.tenant_id, request.request_id, access.user_id, access.conversation_id, _json(request), _as_utc(request.created_at)),
                )
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise LiteratureConflictError("literature request already exists") from exc
                raise
            self._audit(cursor, access, "create", "literature_request", request.request_id)
        return request

    def get_request(self, access: LiteratureAccess, request_id: str) -> LiteratureRequest:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT request_json FROM literature.literature_requests
                WHERE tenant_id = %s AND user_id = %s AND request_id = %s
                """,
                (access.tenant_id, access.user_id, request_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise LiteratureNotFoundError("literature request not found")
        return _validate(LiteratureRequest, _row(row, "request_json", 0))

    def save_queries(self, access: LiteratureAccess, request_id: str, queries: list[SearchQuery]) -> None:
        self.get_request(access, request_id)
        created_at = utc_now()
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            for query in queries:
                cursor.execute(
                    """
                    INSERT INTO literature.literature_queries(
                        tenant_id, request_id, query_id, priority, query_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, request_id, query_id) DO UPDATE SET
                        priority = EXCLUDED.priority,
                        query_json = EXCLUDED.query_json,
                        created_at = EXCLUDED.created_at
                    """,
                    (access.tenant_id, request_id, query.query_id, query.priority, _json(query), created_at),
                )
            self._audit(cursor, access, "save_queries", "literature_request", request_id, {"count": len(queries)})

    @staticmethod
    def _identifiers(card: PaperCard) -> list[tuple[str, str]]:
        values = {"doi": card.doi, "arxiv": card.arxiv_id, "semantic_scholar": card.semantic_scholar_id, "openalex": card.openalex_id}
        return [(kind, value.casefold()) for kind, value in values.items() if value]

    def _upsert_paper(self, cursor: Any, tenant_id: str, card: PaperCard) -> None:
        identifiers = self._identifiers(card)
        for kind, value in identifiers:
            cursor.execute(
                """
                SELECT paper_id FROM literature.paper_identifiers
                WHERE tenant_id = %s AND identifier_type = %s AND identifier_value = %s
                """,
                (tenant_id, kind, value),
            )
            owner = cursor.fetchone()
            if owner is not None and _row(owner, "paper_id", 0) != card.paper_id:
                raise LiteratureConflictError(f"{kind} identifier already belongs to another canonical paper")
        cursor.execute(
            """
            INSERT INTO literature.paper_catalog(
                tenant_id, paper_id, card_json, verification_status,
                full_text_status, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, paper_id) DO UPDATE SET
                card_json = EXCLUDED.card_json,
                verification_status = EXCLUDED.verification_status,
                full_text_status = EXCLUDED.full_text_status,
                updated_at = EXCLUDED.updated_at
            """,
            (tenant_id, card.paper_id, _json(card), card.verification_status, card.full_text_status, utc_now()),
        )
        for kind, value in identifiers:
            cursor.execute(
                """
                INSERT INTO literature.paper_identifiers(
                    tenant_id, identifier_type, identifier_value, paper_id
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, identifier_type, identifier_value) DO UPDATE SET
                    paper_id = EXCLUDED.paper_id
                """,
                (tenant_id, kind, value, card.paper_id),
            )

    def save_discovery(self, access: LiteratureAccess, result: DiscoveryResult) -> None:
        self.get_request(access, result.request_id)
        self.save_queries(access, result.request_id, result.queries)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            for rank, card in enumerate(result.papers, start=1):
                self._upsert_paper(cursor, access.tenant_id, card)
                cursor.execute(
                    """
                    INSERT INTO literature.paper_search_results(
                        tenant_id, request_id, paper_id, rank, relevance_score,
                        matched_queries_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, request_id, paper_id) DO UPDATE SET
                        rank = EXCLUDED.rank,
                        relevance_score = EXCLUDED.relevance_score,
                        matched_queries_json = EXCLUDED.matched_queries_json,
                        created_at = EXCLUDED.created_at
                    """,
                    (access.tenant_id, result.request_id, card.paper_id, rank, card.relevance_score, _json(card.matched_queries), _as_utc(result.created_at)),
                )
            self._audit(cursor, access, "save_discovery", "literature_request", result.request_id, {"paper_count": len(result.papers)})

    def list_papers(self, access: LiteratureAccess, request_id: str, *, limit: int = 100) -> list[PaperCard]:
        self.get_request(access, request_id)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT p.card_json FROM literature.paper_search_results r
                JOIN literature.paper_catalog p ON p.tenant_id = r.tenant_id AND p.paper_id = r.paper_id
                WHERE r.tenant_id = %s AND r.request_id = %s
                ORDER BY r.rank ASC LIMIT %s
                """,
                (access.tenant_id, request_id, min(max(1, limit), 100)),
            )
            rows = cursor.fetchall()
        return [_validate(PaperCard, _row(row, "card_json", 0)) for row in rows]

    def get_paper(self, access: LiteratureAccess, paper_id: str) -> PaperCard:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                "SELECT card_json FROM literature.paper_catalog WHERE tenant_id = %s AND paper_id = %s",
                (access.tenant_id, paper_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise LiteratureNotFoundError("paper not found")
        return _validate(PaperCard, _row(row, "card_json", 0))

    def upsert_paper(self, access: LiteratureAccess, card: PaperCard) -> None:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._upsert_paper(cursor, access.tenant_id, card)
            self._audit(cursor, access, "upsert", "paper", card.paper_id)

    def create_scope(self, access: LiteratureAccess, scope: ResearchScope) -> ResearchScope:
        self._assert_scope_access(access, scope)
        self.get_request(access, scope.request_id)
        if scope.scope_version != 1:
            raise LiteratureConflictError("a new scope must start at version 1")
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._validate_scope_papers(cursor, scope)
            try:
                self._insert_scope(cursor, scope)
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise LiteratureConflictError("research scope already exists") from exc
                raise
            self._audit(cursor, access, "create", "research_scope", scope.scope_id)
        return scope

    @staticmethod
    def _assert_scope_access(access: LiteratureAccess, scope: ResearchScope) -> None:
        if scope.tenant_id != access.tenant_id or scope.owner_user_id != access.user_id:
            raise LiteratureAccessError("scope authority does not match the caller")
        if access.conversation_id and scope.conversation_id != access.conversation_id:
            raise LiteratureAccessError("scope conversation does not match the caller")

    @staticmethod
    def _validate_scope_papers(cursor: Any, scope: ResearchScope) -> None:
        expected = set(scope.selected_paper_ids) | set(scope.excluded_paper_ids)
        if not expected:
            return
        cursor.execute(
            "SELECT paper_id FROM literature.paper_catalog WHERE tenant_id = %s AND paper_id = ANY(%s)",
            (scope.tenant_id, sorted(expected)),
        )
        missing = expected - {_row(row, "paper_id", 0) for row in cursor.fetchall()}
        if missing:
            raise LiteratureNotFoundError("scope contains unknown paper IDs: " + ", ".join(sorted(missing)))

    @staticmethod
    def _insert_scope(cursor: Any, scope: ResearchScope) -> None:
        cursor.execute(
            """
            INSERT INTO literature.research_scopes(
                tenant_id, scope_id, scope_version, owner_user_id,
                conversation_id, request_id, status, scope_json, created_at, confirmed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (scope.tenant_id, scope.scope_id, scope.scope_version, scope.owner_user_id, scope.conversation_id, scope.request_id, scope.status, _json(scope), _as_utc(scope.created_at), _optional_utc(scope.confirmed_at)),
        )
        for paper_id, selection in [(item, "selected") for item in scope.selected_paper_ids] + [(item, "excluded") for item in scope.excluded_paper_ids]:
            cursor.execute(
                """
                INSERT INTO literature.research_scope_papers(
                    tenant_id, scope_id, scope_version, paper_id, selection_status
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (scope.tenant_id, scope.scope_id, scope.scope_version, paper_id, selection),
            )

    def _scope_row(self, cursor: Any, access: LiteratureAccess, scope_id: str, version: int | None = None, *, lock: bool = False) -> Any:
        where = ["tenant_id = %s", "owner_user_id = %s", "scope_id = %s"]
        params: list[Any] = [access.tenant_id, access.user_id, scope_id]
        if access.conversation_id is not None:
            where.append("conversation_id = %s")
            params.append(access.conversation_id)
        if version is not None:
            where.append("scope_version = %s")
            params.append(version)
        cursor.execute(
            "SELECT scope_json FROM literature.research_scopes WHERE "
            + " AND ".join(where)
            + " ORDER BY scope_version DESC LIMIT 1"
            + (" FOR UPDATE" if lock else ""),
            tuple(params),
        )
        return cursor.fetchone()

    def get_scope(self, access: LiteratureAccess, scope_id: str, *, version: int | None = None) -> ResearchScope:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            row = self._scope_row(cursor, access, scope_id, version)
        if row is None:
            raise LiteratureNotFoundError("research scope not found")
        return _validate(ResearchScope, _row(row, "scope_json", 0))

    def list_scopes(self, access: LiteratureAccess) -> list[ResearchScope]:
        where = ["s.tenant_id = %s", "s.owner_user_id = %s"]
        params: list[Any] = [access.tenant_id, access.user_id]
        if access.conversation_id is not None:
            where.append("s.conversation_id = %s")
            params.append(access.conversation_id)
        cursor_sql = """
            SELECT s.scope_json FROM literature.research_scopes s
            JOIN (
                SELECT tenant_id, scope_id, MAX(scope_version) AS latest_version
                FROM literature.research_scopes GROUP BY tenant_id, scope_id
            ) latest ON latest.tenant_id = s.tenant_id AND latest.scope_id = s.scope_id
                AND latest.latest_version = s.scope_version
            WHERE """ + " AND ".join(where) + " ORDER BY s.created_at DESC"
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(cursor_sql, tuple(params))
            rows = cursor.fetchall()
        return [_validate(ResearchScope, _row(row, "scope_json", 0)) for row in rows]

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
        confirmed_at = current.confirmed_at or (utc_now() if target_status != "draft" else None)
        update: dict[str, Any] = {"scope_version": current.scope_version + 1, "created_at": utc_now(), "confirmed_at": confirmed_at, "status": target_status}
        if selected_paper_ids is not None:
            update["selected_paper_ids"] = selected_paper_ids
        if excluded_paper_ids is not None:
            update["excluded_paper_ids"] = excluded_paper_ids
        if user_intent is not None:
            update["user_intent"] = user_intent
        if allowed_expansion is not None:
            update["allowed_expansion"] = allowed_expansion
        revised = ResearchScope.model_validate(current.model_copy(update=update).model_dump())
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            latest = self._scope_row(cursor, access, scope_id, lock=True)
            if latest is None:
                raise LiteratureNotFoundError("research scope not found")
            latest_scope = _validate(ResearchScope, _row(latest, "scope_json", 0))
            if latest_scope.scope_version != current.scope_version:
                raise LiteratureConflictError("research scope version changed")
            self._validate_scope_papers(cursor, revised)
            self._insert_scope(cursor, revised)
            self._audit(cursor, access, "new_version", "research_scope", scope_id, {"version": revised.scope_version, "status": revised.status})
        return revised

    def transition_scope_status(self, access: LiteratureAccess, scope_id: str, status: str, *, expected_version: int | None = None) -> ResearchScope:
        current = self.get_scope(access, scope_id)
        if expected_version is not None and current.scope_version != expected_version:
            raise LiteratureConflictError("research scope version changed")
        allowed = {"draft": {"confirmed", "closed"}, "confirmed": {"ingesting", "closed"}, "ingesting": {"ready", "closed"}, "ready": {"expansion_requested", "closed"}, "expansion_requested": {"ready", "closed"}, "closed": set()}
        if status not in allowed.get(current.status, set()):
            raise LiteratureConflictError(f"invalid research scope transition: {current.status} -> {status}")
        confirmed_at = current.confirmed_at or (utc_now() if status != "draft" else None)
        # Lifecycle transitions do not create a new immutable scope version.
        # The SQLite repository keeps the uploaded-PDF artifact and ingestion
        # status under the same version; doing otherwise here changes the
        # filesystem key from vN to vN+1 and makes a successful upload appear
        # missing when ingestion starts.
        revised = ResearchScope.model_validate(
            current.model_copy(update={"status": status, "confirmed_at": confirmed_at}).model_dump()
        )
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            latest = self._scope_row(cursor, access, scope_id, lock=True)
            if latest is None:
                raise LiteratureNotFoundError("research scope not found")
            if _validate(ResearchScope, _row(latest, "scope_json", 0)).scope_version != current.scope_version:
                raise LiteratureConflictError("research scope version changed")
            cursor.execute(
                """
                UPDATE literature.research_scopes
                   SET status = %s, scope_json = %s, confirmed_at = %s
                 WHERE tenant_id = %s AND scope_id = %s AND scope_version = %s
                """,
                (
                    revised.status,
                    _json(revised),
                    _optional_utc(revised.confirmed_at),
                    access.tenant_id,
                    revised.scope_id,
                    revised.scope_version,
                ),
            )
            self._audit(cursor, access, "transition_status", "research_scope", scope_id, {"from": current.status, "to": revised.status})
        return revised

    def save_ingestion_status(self, access: LiteratureAccess, status: IngestionStatus, *, scope_version: int | None = None) -> None:
        scope = self.get_scope(access, status.scope_id, version=scope_version)
        if status.paper_id not in scope.selected_paper_ids:
            raise LiteratureAccessError("ingestion paper is outside the selected scope")
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO literature.paper_ingestion_jobs(
                    tenant_id, scope_id, scope_version, paper_id, job_id, status_json, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, scope_id, scope_version, paper_id) DO UPDATE SET
                    job_id = EXCLUDED.job_id, status_json = EXCLUDED.status_json, updated_at = EXCLUDED.updated_at
                """,
                (access.tenant_id, scope.scope_id, scope.scope_version, status.paper_id, status.job_id, _json(status), _as_utc(status.updated_at)),
            )
            self._audit(cursor, access, "ingestion_status", "paper", status.paper_id)

    def list_ingestion_statuses(self, access: LiteratureAccess, scope_id: str, *, version: int | None = None) -> list[IngestionStatus]:
        scope = self.get_scope(access, scope_id, version=version)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                "SELECT status_json FROM literature.paper_ingestion_jobs WHERE tenant_id = %s AND scope_id = %s AND scope_version = %s ORDER BY paper_id",
                (access.tenant_id, scope.scope_id, scope.scope_version),
            )
            rows = cursor.fetchall()
        return [_validate(IngestionStatus, _row(row, "status_json", 0)) for row in rows]

    def save_evidence(self, access: LiteratureAccess, cards: list[EvidenceCard]) -> None:
        if not cards:
            return
        scope_ids = {card.scope_id for card in cards}
        versions = {card.scope_version for card in cards}
        if len(scope_ids) != 1 or len(versions) != 1 or None in scope_ids or None in versions:
            raise LiteratureConflictError("an evidence batch must belong to one scope version")
        scope = self.get_scope(access, next(iter(scope_ids)), version=next(iter(versions)))  # type: ignore[arg-type]
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            for card in cards:
                if card.paper_id not in scope.selected_paper_ids:
                    raise LiteratureAccessError("evidence paper is outside the selected scope")
                cursor.execute(
                    """
                    INSERT INTO literature.evidence_cards(
                        tenant_id, evidence_id, scope_id, scope_version, paper_id, card_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, evidence_id) DO UPDATE SET
                        scope_id = EXCLUDED.scope_id, scope_version = EXCLUDED.scope_version,
                        paper_id = EXCLUDED.paper_id, card_json = EXCLUDED.card_json, created_at = EXCLUDED.created_at
                    """,
                    (access.tenant_id, card.evidence_id, scope.scope_id, scope.scope_version, card.paper_id, _json(card), utc_now()),
                )
            self._audit(cursor, access, "save_evidence", "research_scope", scope.scope_id, {"count": len(cards), "version": scope.scope_version})

    def list_evidence(self, access: LiteratureAccess, scope_id: str, *, version: int | None = None, paper_id: str | None = None) -> list[EvidenceCard]:
        scope = self.get_scope(access, scope_id, version=version)
        where = ["tenant_id = %s", "scope_id = %s", "scope_version = %s"]
        params: list[Any] = [access.tenant_id, scope.scope_id, scope.scope_version]
        if paper_id is not None:
            where.append("paper_id = %s")
            params.append(paper_id)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute("SELECT card_json FROM literature.evidence_cards WHERE " + " AND ".join(where) + " ORDER BY evidence_id", tuple(params))
            rows = cursor.fetchall()
        return [_validate(EvidenceCard, _row(row, "card_json", 0)) for row in rows]

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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            for claim in claims:
                cursor.execute(
                    """
                    INSERT INTO literature.claim_records(
                        tenant_id, claim_id, scope_id, scope_version, claim_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, claim_id) DO UPDATE SET
                        scope_id = EXCLUDED.scope_id, scope_version = EXCLUDED.scope_version,
                        claim_json = EXCLUDED.claim_json, created_at = EXCLUDED.created_at
                    """,
                    (access.tenant_id, claim.claim_id, claim.scope_id, claim.scope_version, _json(claim), utc_now()),
                )
            self._audit(cursor, access, "save_claims", "research_scope", claims[0].scope_id or "unknown", {"count": len(claims)})

    def list_claims(self, access: LiteratureAccess, scope_id: str, *, version: int | None = None) -> list[ClaimRecord]:
        scope = self.get_scope(access, scope_id, version=version)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute("SELECT claim_json FROM literature.claim_records WHERE tenant_id = %s AND scope_id = %s AND scope_version = %s ORDER BY claim_id", (access.tenant_id, scope.scope_id, scope.scope_version))
            rows = cursor.fetchall()
        return [_validate(ClaimRecord, _row(row, "claim_json", 0)) for row in rows]

    def request_expansion(self, access: LiteratureAccess, request: ScopeExpansionRequest) -> ScopeExpansionRequest:
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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            try:
                cursor.execute(
                    """
                    INSERT INTO literature.scope_expansion_requests(
                        tenant_id, expansion_id, scope_id, scope_version,
                        request_json, created_at, decided_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NULL)
                    """,
                    (access.tenant_id, request.expansion_id, scope.scope_id, scope.scope_version, _json(request), _as_utc(request.created_at)),
                )
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise LiteratureConflictError("scope expansion request already exists") from exc
                raise
            self._audit(cursor, access, "request_expansion", "research_scope", scope.scope_id, {"expansion_id": request.expansion_id})
        return request

    def decide_expansion(self, access: LiteratureAccess, expansion_id: str, *, approve: bool, expected_scope_id: str | None = None) -> ScopeExpansionRequest:
        where = ["e.tenant_id = %s", "e.expansion_id = %s", "s.owner_user_id = %s"]
        params: list[Any] = [access.tenant_id, expansion_id, access.user_id]
        if expected_scope_id is not None:
            where.append("e.scope_id = %s")
            params.append(expected_scope_id)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT e.request_json FROM literature.scope_expansion_requests e
                JOIN literature.research_scopes s ON s.tenant_id = e.tenant_id
                  AND s.scope_id = e.scope_id AND s.scope_version = e.scope_version
                WHERE """ + " AND ".join(where) + " FOR UPDATE",
                tuple(params),
            )
            row = cursor.fetchone()
            if row is None:
                raise LiteratureNotFoundError("scope expansion request not found")
            current = _validate(ScopeExpansionRequest, _row(row, "request_json", 0))
            if current.status != "pending":
                raise LiteratureConflictError("scope expansion request is already decided")
            decided_at = utc_now()
            revised = current.model_copy(update={"status": "approved" if approve else "rejected", "decided_at": decided_at})
            cursor.execute(
                "UPDATE literature.scope_expansion_requests SET request_json = %s, decided_at = %s WHERE tenant_id = %s AND expansion_id = %s",
                (_json(revised), decided_at, access.tenant_id, expansion_id),
            )
            self._audit(cursor, access, "approve_expansion" if approve else "reject_expansion", "scope_expansion", expansion_id)
        return revised

    def list_audit_events(self, access: LiteratureAccess, *, limit: int = 100) -> list[dict[str, object]]:
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT event_id, user_id, action, resource_type, resource_id, details_json, created_at
                FROM literature.audit_events WHERE tenant_id = %s
                ORDER BY event_id DESC LIMIT %s
                """,
                (access.tenant_id, min(max(1, limit), 1000)),
            )
            rows = cursor.fetchall()
        return [{"event_id": _row(row, "event_id", 0), "user_id": _row(row, "user_id", 1), "action": _row(row, "action", 2), "resource_type": _row(row, "resource_type", 3), "resource_id": _row(row, "resource_id", 4), "details": _row(row, "details_json", 5), "created_at": _row(row, "created_at", 6)} for row in rows]

    @staticmethod
    def _audit(cursor: Any, access: LiteratureAccess, action: str, resource_type: str, resource_id: str, details: dict[str, object] | None = None) -> None:
        cursor.execute(
            """
            INSERT INTO literature.audit_events(
                tenant_id, user_id, action, resource_type, resource_id, details_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (access.tenant_id, access.user_id, action, resource_type, resource_id, _json(details or {}), utc_now()),
        )


def _row(row: Any, key: str, index: int) -> Any:
    return row.get(key) if isinstance(row, Mapping) else row[index]


def _validate(model: Any, payload: Any) -> Any:
    if isinstance(payload, str):
        return model.model_validate_json(payload)
    return model.model_validate(payload)


def _json(value: object) -> object:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    try:
        from psycopg.types.json import Jsonb
    except ImportError:  # pragma: no cover
        return payload
    return Jsonb(payload)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _as_utc(value)


def _is_unique_violation(exc: Exception) -> bool:
    return getattr(exc, "sqlstate", None) == "23505" or exc.__class__.__name__ == "UniqueViolation"


__all__ = ["PostgresLiteratureRepository"]
