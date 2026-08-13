"""Budgeted assembly of ACL-filtered knowledge and scoped memory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from .knowledge import AccessContext, InMemoryKnowledgeStore, KnowledgeHit
from .memory import InMemoryMemoryStore, MemoryHit, MemoryScope


def _value(obj: object | None, *names: str) -> object | None:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _string(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _string_set(value: object | None) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value}) if value else frozenset()
    result: set[str] = set()
    try:
        values: Iterable[object] = value  # type: ignore[assignment]
        for item in values:
            candidate = _string(_value(item, "source_uri", "id") or item)
            if candidate:
                result.add(candidate)
    except TypeError:
        candidate = _string(value)
        if candidate:
            result.add(candidate)
    return frozenset(result)


def _intersection_if_both(left: frozenset[str] | None, right: frozenset[str] | None) -> frozenset[str] | None:
    if left is None:
        return right
    if right is None:
        return left
    return left.intersection(right)


@dataclass(frozen=True, slots=True)
class Citation:
    label: str
    kind: str
    item_id: str
    source_uri: str
    version: str | None = None
    provenance_type: str | None = None


@dataclass(frozen=True, slots=True)
class AssembledContext:
    text: str
    citations: tuple[Citation, ...]
    knowledge_hits: tuple[KnowledgeHit, ...]
    memory_hits: tuple[MemoryHit, ...]
    retrieval_query: str
    retrieval_profile: str | None
    retrieval_backend: str | None
    used_chars: int
    char_budget: int
    truncated: bool

    @property
    def context(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    item_id: str
    score: float
    source_uri: str
    content: str
    version: str | None
    provenance_type: str | None
    hit: KnowledgeHit | MemoryHit


class ContextAssembler:
    """Create a small, cited observation without bypassing store policies."""

    def __init__(
        self,
        knowledge_store: InMemoryKnowledgeStore | None = None,
        memory_store: InMemoryMemoryStore | None = None,
        *,
        default_char_budget: int = 6_000,
        knowledge_limit: int = 8,
        memory_limit: int = 8,
    ) -> None:
        if default_char_budget < 0:
            raise ValueError("default_char_budget must be non-negative")
        self.knowledge_store = knowledge_store or InMemoryKnowledgeStore()
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.default_char_budget = default_char_budget
        self.knowledge_limit = max(0, knowledge_limit)
        self.memory_limit = max(0, memory_limit)

    def assemble(
        self,
        query: str | object | None = None,
        profile: object | None = None,
        task: object | None = None,
        *,
        principal: AccessContext | None = None,
        now: datetime | None = None,
        char_budget: int | None = None,
    ) -> AssembledContext:
        # Convenience form: assemble(task, profile=...).
        if query is not None and not isinstance(query, str):
            if task is not None:
                raise TypeError("task was provided both positionally and by keyword")
            task, query = query, None

        principal = self._principal(principal, task, profile)
        retrieval_query = self._retrieval_query(query, task, profile)
        budget = self._budget(char_budget, task, profile)

        profile_sources = _string_set(_value(profile, "knowledge_sources", "knowledge_source_uris"))
        task_sources = _string_set(_value(task, "knowledge_sources", "knowledge_source_uris"))
        source_uris = _intersection_if_both(profile_sources, task_sources)
        profile_bases = _string_set(_value(profile, "knowledge_bases", "knowledge_base_ids"))
        task_bases = _string_set(_value(task, "knowledge_bases", "knowledge_base_ids"))
        knowledge_bases = _intersection_if_both(profile_bases, task_bases)
        raw_scopes = _value(profile, "memory_scopes")
        memory_scopes = None
        if raw_scopes is not None:
            values = (raw_scopes,) if isinstance(raw_scopes, str) else raw_scopes
            memory_scopes = tuple(MemoryScope(value) for value in values)

        knowledge_hits = self.knowledge_store.search(
            retrieval_query,
            principal,
            top_k=self.knowledge_limit,
            now=now,
            source_uris=source_uris,
            knowledge_base_ids=knowledge_bases,
        )
        memory_hits = self.memory_store.recall(
            retrieval_query,
            principal,
            top_k=self.memory_limit,
            now=now,
            scopes=memory_scopes,
        )

        candidates = [
            _Candidate(
                kind="knowledge",
                item_id=hit.chunk.chunk_id,
                score=hit.score,
                source_uri=hit.chunk.source_uri,
                content=hit.chunk.text,
                version=hit.chunk.version,
                provenance_type=None,
                hit=hit,
            )
            for hit in knowledge_hits
        ]
        candidates.extend(
            _Candidate(
                kind="memory",
                item_id=hit.item.memory_id,
                score=hit.score,
                source_uri=hit.item.provenance.source_uri or f"memory:{hit.item.memory_id}",
                content=hit.item.content,
                version=None,
                provenance_type=hit.item.provenance.source_type,
                hit=hit,
            )
            for hit in memory_hits
        )
        candidates.sort(key=lambda item: (-item.score, 0 if item.kind == "knowledge" else 1, item.item_id))

        blocks: list[str] = []
        citations: list[Citation] = []
        selected_knowledge: list[KnowledgeHit] = []
        selected_memory: list[MemoryHit] = []
        counters = {"knowledge": 0, "memory": 0}
        any_truncated = False
        # A single very long chunk must not starve every other evidence type.
        # Reserve space for up to four top candidates, while still allowing a
        # lone candidate to consume the full budget.
        diversity_slots = min(len(candidates), 4)
        per_item_content_cap = budget
        if diversity_slots > 1:
            per_item_content_cap = max(24, budget // diversity_slots - 64)

        for candidate in candidates:
            prefix = "K" if candidate.kind == "knowledge" else "M"
            next_number = counters[candidate.kind] + 1
            label = f"{prefix}{next_number}"
            source = candidate.source_uri[:160]
            version = f"; version={candidate.version}" if candidate.version is not None else ""
            header = f"[{label}] {candidate.kind}: {source}{version}\n"
            separator = "" if not blocks else "\n\n"
            remaining = budget - len("".join(blocks)) - len(separator)
            if remaining <= len(header):
                any_truncated = True
                break
            content = candidate.content
            content_capped = len(content) > per_item_content_cap
            if content_capped:
                content = content[: max(0, per_item_content_cap - 1)] + "…"
                any_truncated = True
            block = header + content
            exhausted_budget = False
            if len(block) > remaining:
                available = remaining - len(header)
                if available <= 1:
                    any_truncated = True
                    break
                content = content[: max(0, available - 1)] + "…"
                block = header + content
                any_truncated = True
                exhausted_budget = True
            blocks.append(separator + block)
            counters[candidate.kind] = next_number
            citations.append(
                Citation(
                    label=label,
                    kind=candidate.kind,
                    item_id=candidate.item_id,
                    source_uri=candidate.source_uri,
                    version=candidate.version,
                    provenance_type=candidate.provenance_type,
                )
            )
            if isinstance(candidate.hit, KnowledgeHit):
                selected_knowledge.append(candidate.hit)
            else:
                selected_memory.append(candidate.hit)
            if exhausted_budget:
                break

        text = "".join(blocks)
        truncated = any_truncated or len(citations) < len(candidates)
        retrieval_profile = next(
            (
                hit.retrieval_profile
                for hit in knowledge_hits
                if hit.retrieval_profile is not None
            ),
            None,
        )
        retrieval_backend = next(
            (
                hit.retrieval_backend
                for hit in knowledge_hits
                if hit.retrieval_backend is not None
            ),
            None,
        )
        return AssembledContext(
            text=text,
            citations=tuple(citations),
            knowledge_hits=tuple(selected_knowledge),
            memory_hits=tuple(selected_memory),
            retrieval_query=retrieval_query,
            retrieval_profile=retrieval_profile,
            retrieval_backend=retrieval_backend,
            used_chars=len(text),
            char_budget=budget,
            truncated=truncated,
        )

    def _principal(
        self,
        principal: AccessContext | None,
        task: object | None,
        profile: object | None,
    ) -> AccessContext:
        claims = {
            "tenant_id": _string(_value(task, "tenant_id") or _value(profile, "tenant_id")),
            "user_id": _string(_value(task, "user_id")),
            "org_id": _string(_value(task, "org_id", "organization_id")),
            "agent_id": _string(_value(profile, "agent_id", "id") or _value(task, "agent_id")),
            "task_id": _string(_value(task, "task_id", "id")),
        }
        if principal is None:
            if claims["tenant_id"] is None:
                raise ValueError("principal or task.tenant_id is required")
            return AccessContext(**claims)  # type: ignore[arg-type]

        for name, claimed in claims.items():
            trusted = getattr(principal, name)
            if trusted is not None and claimed is not None and trusted != claimed:
                raise PermissionError(f"{name} does not match the trusted principal")
        # Missing scope identifiers may safely be enriched from host task/profile
        # objects after all conflicting trusted claims have been rejected.
        return AccessContext(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id or claims["user_id"],
            org_id=principal.org_id or claims["org_id"],
            agent_id=principal.agent_id or claims["agent_id"],
            task_id=principal.task_id or claims["task_id"],
            roles=principal.roles,
            groups=principal.groups,
        )

    def _retrieval_query(self, query: str | None, task: object | None, profile: object | None) -> str:
        pieces = [
            _string(query),
            _string(_value(task, "goal", "query", "objective")),
            _string(_value(profile, "retrieval_hints", "context_hints")),
        ]
        unique: list[str] = []
        seen: set[str] = set()
        for piece in pieces:
            if piece is not None and piece.casefold() not in seen:
                unique.append(piece)
                seen.add(piece.casefold())
        return " ".join(unique)

    def _budget(self, explicit: int | None, task: object | None, profile: object | None) -> int:
        raw = explicit
        if raw is None:
            raw = _value(task, "context_char_budget")  # type: ignore[assignment]
        if raw is None:
            raw = _value(profile, "context_char_budget")  # type: ignore[assignment]
        budget = self.default_char_budget if raw is None else int(raw)
        if budget < 0:
            raise ValueError("char_budget must be non-negative")
        return budget


__all__ = ["AssembledContext", "Citation", "ContextAssembler"]
