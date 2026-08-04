"""Scoped longitudinal memory with fail-closed tenant isolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .knowledge import AccessContext, as_utc, lexical_match, utc_now


class MemoryScope(str, Enum):
    TENANT = "tenant"
    ORG = "org"
    USER = "user"
    AGENT = "agent"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    source_type: str = "unknown"
    source_id: str | None = None
    source_uri: str | None = None
    actor_id: str | None = None
    observed_at: datetime = field(default_factory=utc_now)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        source_type = str(self.source_type).strip()
        if not source_type:
            raise ValueError("source_type must not be empty")
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "observed_at", as_utc(self.observed_at))
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class MemoryItem:
    memory_id: str
    tenant_id: str
    content: str
    scope: MemoryScope | str = MemoryScope.TENANT
    scope_id: str | None = None
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    importance: float = 0.5
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("memory_id", "tenant_id", "content"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        scope = MemoryScope(self.scope)
        object.__setattr__(self, "scope", scope)
        scope_id = self.scope_id
        if scope is MemoryScope.TENANT and scope_id is None:
            scope_id = self.tenant_id
        if scope_id is None or not str(scope_id).strip():
            raise ValueError(f"scope_id is required for {scope.value} memory")
        object.__setattr__(self, "scope_id", str(scope_id).strip())
        if scope is MemoryScope.TENANT and self.scope_id != self.tenant_id:
            raise ValueError("tenant memory scope_id must equal tenant_id")
        importance = float(self.importance)
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        object.__setattr__(self, "importance", importance)
        object.__setattr__(self, "created_at", as_utc(self.created_at))
        object.__setattr__(self, "updated_at", as_utc(self.updated_at or self.created_at))
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", as_utc(self.expires_at))
        object.__setattr__(self, "tags", frozenset(str(tag).strip() for tag in self.tags if str(tag).strip()))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def text(self) -> str:
        return self.content

    def is_expired_at(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and as_utc(now) >= self.expires_at

    def is_visible_to(self, principal: AccessContext, now: datetime | None = None) -> bool:
        if self.tenant_id != principal.tenant_id or self.is_expired_at(now):
            return False
        expected = {
            MemoryScope.TENANT: principal.tenant_id,
            MemoryScope.ORG: principal.org_id,
            MemoryScope.USER: principal.user_id,
            MemoryScope.AGENT: principal.agent_id,
            MemoryScope.TASK: principal.task_id,
        }[self.scope]
        return expected is not None and self.scope_id == expected

    def is_deletable_by(self, principal: AccessContext, now: datetime | None = None) -> bool:
        """Return true only for a principal-owned, non-shared scope.

        Tenant and organisation memories are shared context. Visibility alone
        must never imply delete authority; their retention requires a separate
        operator/admin capability that this local API does not expose.
        """

        return self.scope in {
            MemoryScope.USER,
            MemoryScope.AGENT,
            MemoryScope.TASK,
        } and self.is_visible_to(principal, now)


@dataclass(frozen=True, slots=True)
class MemoryHit:
    item: MemoryItem
    score: float
    lexical_score: float
    freshness_score: float
    matched_terms: tuple[str, ...] = ()


class InMemoryMemoryStore:
    def __init__(self, items: Iterable[MemoryItem] = ()) -> None:
        self._items: dict[tuple[str, str], MemoryItem] = {}
        for item in items:
            self.remember(item)

    def remember(self, item: MemoryItem) -> None:
        self._items[(item.tenant_id, item.memory_id)] = item

    upsert = remember
    add = remember

    def get(self, memory_id: str, principal: AccessContext, *, now: datetime | None = None) -> MemoryItem | None:
        item = self._items.get((principal.tenant_id, memory_id))
        if item is None or not item.is_visible_to(principal, now):
            return None
        return item

    def forget(self, memory_id: str, principal: AccessContext, *, now: datetime | None = None) -> bool:
        """Delete one principal-owned memory without revealing shared IDs."""

        item = self.get(memory_id, principal, now=now)
        if item is None or not item.is_deletable_by(principal, now):
            return False
        del self._items[(principal.tenant_id, memory_id)]
        return True

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
        allowed_scopes = None if scopes is None else frozenset(MemoryScope(scope) for scope in scopes)
        instant = as_utc(now)
        has_query = bool(query.strip())
        hits: list[MemoryHit] = []
        for item in self._items.values():
            if not item.is_visible_to(principal, instant):
                continue
            if allowed_scopes is not None and item.scope not in allowed_scopes:
                continue
            match = lexical_match(query, item.content)
            if has_query and match.score <= 0 and not include_unmatched:
                continue
            age_days = max(0.0, (instant - item.updated_at).total_seconds() / 86_400)
            freshness = 1.0 / (1.0 + age_days / 30.0)
            if has_query:
                score = 0.80 * match.score + 0.15 * item.importance + 0.05 * freshness
            else:
                score = 0.75 * item.importance + 0.25 * freshness
            hits.append(
                MemoryHit(
                    item=item,
                    score=score,
                    lexical_score=match.score,
                    freshness_score=freshness,
                    matched_terms=match.matched_terms,
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.lexical_score,
                -hit.item.importance,
                -hit.item.updated_at.timestamp(),
                hit.item.memory_id,
            )
        )
        return hits[:top_k]


MemoryStore = InMemoryMemoryStore


__all__ = [
    "InMemoryMemoryStore",
    "MemoryHit",
    "MemoryItem",
    "MemoryProvenance",
    "MemoryScope",
    "MemoryStore",
]
