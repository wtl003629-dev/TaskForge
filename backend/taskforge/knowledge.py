"""Tenant-safe knowledge retrieval contracts.

The MVP deliberately uses an in-memory lexical backend.  The public result
shape carries lexical and semantic scores separately so a vector backend can
be added without changing the context assembler or weakening ACL checks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

UTC = UTC
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime:
    """Return an aware UTC datetime, interpreting naive values as UTC."""

    if value is None:
        return utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _required(value: str, name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _normalise_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def tokenise(value: str) -> tuple[str, ...]:
    """Tokenise English identifiers and CJK text deterministically."""

    return tuple(_TOKEN_RE.findall(_normalise_text(value)))


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    score: float
    matched_terms: tuple[str, ...] = ()


def lexical_match(query: str, text: str) -> LexicalMatch:
    """Return a bounded lexical score suitable for deterministic fallback.

    This is intentionally small and explainable rather than pretending to be
    a production search engine.  It rewards term coverage, repeated evidence
    (with a cap), and an exact normalised phrase match.
    """

    query_tokens = tuple(dict.fromkeys(tokenise(query)))
    if not query_tokens:
        return LexicalMatch(0.0)

    text_tokens = tokenise(text)
    if not text_tokens:
        return LexicalMatch(0.0)

    counts: dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1
    matched = tuple(token for token in query_tokens if counts.get(token, 0) > 0)
    if not matched:
        return LexicalMatch(0.0)

    coverage = len(matched) / len(query_tokens)
    frequency = sum(min(counts[token], 3) / 3 for token in matched) / len(query_tokens)
    normalised_query = " ".join(query_tokens)
    normalised_text = " ".join(text_tokens)
    phrase = 1.0 if normalised_query in normalised_text else 0.0
    score = min(1.0, 0.60 * coverage + 0.25 * frequency + 0.15 * phrase)
    return LexicalMatch(score=score, matched_terms=matched)


@dataclass(frozen=True, slots=True)
class AccessContext:
    """Trusted identity used for both knowledge ACLs and memory scopes.

    Tenant equality is always checked before an ACL token is considered.  An
    ACL entry named ``tenant`` therefore means tenant-wide, never public across
    tenants.
    """

    tenant_id: str
    user_id: str | None = None
    org_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _required(self.tenant_id, "tenant_id"))
        for name in ("user_id", "org_id", "agent_id", "task_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, name))
        object.__setattr__(self, "roles", frozenset(str(item).strip() for item in self.roles if str(item).strip()))
        object.__setattr__(self, "groups", frozenset(str(item).strip() for item in self.groups if str(item).strip()))

    @property
    def acl_tokens(self) -> frozenset[str]:
        tokens = {"tenant", f"tenant:{self.tenant_id}"}
        for kind in ("user", "org", "agent", "task"):
            value = getattr(self, f"{kind}_id")
            if value:
                tokens.add(f"{kind}:{value}")
        tokens.update(f"role:{role}" for role in self.roles)
        tokens.update(f"group:{group}" for group in self.groups)
        return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    tenant_id: str
    text: str
    source_uri: str
    document_id: str | None = None
    version: str = "1"
    version_order: int = 1
    acl: frozenset[str] = field(default_factory=lambda: frozenset({"tenant"}))
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("chunk_id", "tenant_id", "text", "source_uri", "version"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.document_id is not None:
            object.__setattr__(self, "document_id", _required(self.document_id, "document_id"))
        if self.version_order < 0:
            raise ValueError("version_order must be non-negative")
        acl = frozenset(str(item).strip() for item in self.acl if str(item).strip())
        object.__setattr__(self, "acl", acl)
        object.__setattr__(self, "created_at", as_utc(self.created_at))
        if self.valid_from is not None:
            object.__setattr__(self, "valid_from", as_utc(self.valid_from))
        if self.valid_until is not None:
            object.__setattr__(self, "valid_until", as_utc(self.valid_until))
        if self.valid_from and self.valid_until and self.valid_from >= self.valid_until:
            raise ValueError("valid_from must be earlier than valid_until")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content(self) -> str:
        return self.text

    @property
    def version_key(self) -> tuple[int, tuple[tuple[int, object], ...]]:
        parts: list[tuple[int, object]] = []
        for part in re.split(r"([0-9]+)", self.version.casefold()):
            if not part:
                continue
            parts.append((0, int(part)) if part.isdigit() else (1, part))
        # Do not include chunk timestamps: all chunks belonging to the same
        # document version must survive latest-version filtering.
        return self.version_order, tuple(parts)

    @property
    def logical_document_id(self) -> str:
        return self.document_id or self.source_uri

    def is_valid_at(self, now: datetime | None = None) -> bool:
        instant = as_utc(now)
        return not (
            (self.valid_from is not None and instant < self.valid_from)
            or (self.valid_until is not None and instant >= self.valid_until)
        )

    def is_visible_to(self, principal: AccessContext, now: datetime | None = None) -> bool:
        if self.tenant_id != principal.tenant_id or not self.is_valid_at(now):
            return False
        return bool(self.acl.intersection(principal.acl_tokens))


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    chunk: KnowledgeChunk
    score: float
    lexical_score: float
    semantic_score: float = 0.0
    matched_terms: tuple[str, ...] = ()


def _bounded_score(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return max(0.0, min(1.0, result))


class InMemoryKnowledgeStore:
    """Deterministic ACL-enforcing store used by the phase-one runtime."""

    def __init__(self, chunks: Iterable[KnowledgeChunk] = ()) -> None:
        # IDs only need to be unique within a tenant.  A global ID key would
        # let one tenant overwrite another tenant's same-named chunk.
        self._chunks: dict[tuple[str, str], KnowledgeChunk] = {}
        for chunk in chunks:
            self.upsert(chunk)

    def upsert(self, chunk: KnowledgeChunk) -> None:
        self._chunks[(chunk.tenant_id, chunk.chunk_id)] = chunk

    add = upsert

    def get(self, chunk_id: str, principal: AccessContext, *, now: datetime | None = None) -> KnowledgeChunk | None:
        chunk = self._chunks.get((principal.tenant_id, chunk_id))
        if chunk is None or not chunk.is_visible_to(principal, now):
            return None
        return chunk

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
        allowed_sources = None if source_uris is None else frozenset(str(value) for value in source_uris)
        allowed_bases = None if knowledge_base_ids is None else frozenset(str(value) for value in knowledge_base_ids)
        candidates = [
            chunk
            for chunk in self._chunks.values()
            if chunk.is_visible_to(principal, now)
            and (allowed_sources is None or chunk.source_uri in allowed_sources or chunk.logical_document_id in allowed_sources)
            and (
                allowed_bases is None
                or str(chunk.metadata.get("knowledge_base_id", "")) in allowed_bases
            )
        ]

        if latest_only:
            latest: dict[str, tuple[int, tuple[tuple[int, object], ...]]] = {}
            for chunk in candidates:
                current = latest.get(chunk.logical_document_id)
                if current is None or chunk.version_key > current:
                    latest[chunk.logical_document_id] = chunk.version_key
            candidates = [chunk for chunk in candidates if chunk.version_key == latest[chunk.logical_document_id]]

        semantic_scores = semantic_scores or {}
        lexical_weight = max(0.0, float(lexical_weight))
        semantic_weight = max(0.0, float(semantic_weight)) if semantic_scores else 0.0
        total_weight = lexical_weight + semantic_weight
        if total_weight <= 0:
            lexical_weight, semantic_weight, total_weight = 1.0, 0.0, 1.0

        hits: list[KnowledgeHit] = []
        for chunk in candidates:
            match = lexical_match(query, chunk.text)
            semantic = _bounded_score(semantic_scores.get(chunk.chunk_id, 0.0))
            score = (lexical_weight * match.score + semantic_weight * semantic) / total_weight
            if score <= 0:
                continue
            hits.append(
                KnowledgeHit(
                    chunk=chunk,
                    score=score,
                    lexical_score=match.score,
                    semantic_score=semantic,
                    matched_terms=match.matched_terms,
                )
            )

        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.lexical_score,
                -hit.semantic_score,
                -hit.chunk.version_order,
                -hit.chunk.created_at.timestamp(),
                hit.chunk.chunk_id,
            )
        )
        return hits[:top_k]


KnowledgeStore = InMemoryKnowledgeStore


__all__ = [
    "AccessContext",
    "InMemoryKnowledgeStore",
    "KnowledgeChunk",
    "KnowledgeHit",
    "KnowledgeStore",
    "LexicalMatch",
    "as_utc",
    "lexical_match",
    "tokenise",
    "utc_now",
]
