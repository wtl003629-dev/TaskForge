"""Business-facing context stores backed by :mod:`postgres_context`.

The application and retrieval layers depend on the same small store
contracts used by the SQLite implementation.  This adapter keeps psycopg and
RLS access construction at the persistence boundary, while lexical ranking
continues to use the existing deterministic in-memory ranker.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from .knowledge import (
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeHit,
)
from .memory import InMemoryMemoryStore, MemoryHit, MemoryItem, MemoryScope
from .postgres_context import PostgresContextAccess, PostgresContextRepository
from .postgres_runtime import PostgresRuntime


def _access(principal: AccessContext) -> PostgresContextAccess:
    return PostgresContextAccess(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id or "anonymous",
        conversation_id=principal.task_id or "__no_task__",
        org_id=principal.org_id,
        agent_id=principal.agent_id,
        acl_principals=principal.acl_tokens,
    )


def _system_access(tenant_id: str, *, conversation_id: str = "__system__") -> PostgresContextAccess:
    return PostgresContextAccess(
        tenant_id=tenant_id,
        user_id="system",
        conversation_id=conversation_id,
    )


class PostgresKnowledgeStore:
    """KnowledgeStore-compatible adapter over the PostgreSQL repository."""

    def __init__(self, repository: PostgresContextRepository) -> None:
        self.repository = repository

    def close(self) -> None:
        self.repository.close()

    def upsert(self, chunk: KnowledgeChunk) -> None:
        self.repository.upsert_knowledge(
            (chunk,), _system_access(chunk.tenant_id, conversation_id=chunk.logical_document_id)
        )

    add = upsert

    def upsert_many(self, chunks: Iterable[KnowledgeChunk]) -> None:
        grouped: dict[str, list[KnowledgeChunk]] = {}
        for chunk in chunks:
            if not isinstance(chunk, KnowledgeChunk):
                raise TypeError("all records must be KnowledgeChunk instances")
            grouped.setdefault(chunk.tenant_id, []).append(chunk)
        for tenant_id, values in grouped.items():
            self.repository.upsert_knowledge(values, _system_access(tenant_id))

    batch_upsert = upsert_many

    def replace_document_version(self, chunks: Iterable[KnowledgeChunk]) -> int:
        materialised = list(chunks)
        if not materialised:
            raise ValueError("at least one chunk is required")
        first = materialised[0]
        return self.repository.replace_knowledge_version(
            materialised,
            _system_access(first.tenant_id, conversation_id=first.logical_document_id),
        )

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> KnowledgeChunk | None:
        candidates = self.repository.fetch_knowledge_candidates(
            _access(principal), now=now, candidate_limit=10_000, latest_only=False
        )
        return next((chunk for chunk in candidates if chunk.chunk_id == chunk_id), None)

    def visible_chunks(
        self,
        principal: AccessContext,
        *,
        now: datetime | None = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
    ) -> tuple[KnowledgeChunk, ...]:
        return tuple(
            self.repository.fetch_knowledge_candidates(
                _access(principal),
                now=now,
                candidate_limit=10_000,
                source_uris=source_uris,
                knowledge_base_ids=knowledge_base_ids,
                latest_only=latest_only,
            )
        )

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
        candidates = self.visible_chunks(
            principal,
            now=now,
            source_uris=source_uris,
            knowledge_base_ids=knowledge_base_ids,
            latest_only=latest_only,
        )
        return InMemoryKnowledgeStore(candidates).search(
            query,
            principal,
            top_k=top_k,
            now=now,
            latest_only=False,
            semantic_scores=semantic_scores,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )

    def search_dense(
        self,
        chunks: Iterable[Any],
        request: Any,
        embedder: Any,
        *,
        approximate: bool = False,
    ) -> Any:
        """Search migrated knowledge vectors in PostgreSQL/pgvector.

        The caller supplies the already-authorized catalog projection so the
        existing retrieval profile can retain its exact output shape.  The
        SQL repeats tenant, ACL, version, document, knowledge-base, and allow
        list predicates before the cosine ordering expression.  ``approximate``
        is explicit and opt-in; the default correctness path is exact cosine.
        """

        from .hybrid_retrieval import (
            AppliedRetrievalFilters,
            HybridSearchHit,
            HybridSearchResponse,
        )

        catalog = tuple(chunks)
        by_id = {chunk.chunk_id: chunk for chunk in catalog}
        query_vector = [float(value) for value in embedder.embed_query(request.query)]
        if not query_vector or any(not _finite(value) for value in query_vector):
            raise ValueError("dense embedder returned an invalid query vector")
        dimension = len(query_vector)
        vector_literal = "[" + ",".join(format(value, ".9g") for value in query_vector) + "]"
        allowed_ids = None if request.allowed_chunk_ids is None else sorted(request.allowed_chunk_ids)
        versions = None if request.versions is None else sorted(request.versions)
        version_orders = None if request.version_orders is None else sorted(request.version_orders)
        knowledge_bases = None if request.knowledge_base_ids is None else sorted(request.knowledge_base_ids)
        parent_documents = None if request.parent_document_ids is None else sorted(request.parent_document_ids)
        # Keep the distance expression visible in ORDER BY. PostgreSQL can
        # only use the optional HNSW cosine index when the operator appears
        # directly in the ordered query; the WHERE clause still carries every
        # tenant/ACL/version/document predicate before ranking.
        distance_expression = "ke.embedding <=> %s::vector"
        backend_name = "postgres_pgvector_hnsw" if approximate else "postgres_pgvector_exact"
        source_name = "pgvector_hnsw" if approximate else "pgvector_exact"
        with self.repository.transaction(
            PostgresContextAccess(
                tenant_id=request.tenant_id,
                user_id="retrieval",
                conversation_id="__retrieval__",
                acl_principals=request.acl_principals,
            )
        ) as cursor:
            cursor.execute(
                f"""
                SELECT kc.chunk_id, {distance_expression} AS distance
                  FROM vector.knowledge_embeddings AS ke
                  JOIN taskforge.knowledge_chunks AS kc
                    ON kc.tenant_id = ke.tenant_id AND kc.chunk_id = ke.chunk_id
                 WHERE ke.tenant_id = %s
                   AND kc.tenant_id = %s
                   AND ke.model = %s
                   AND ke.dimension = %s
                   AND kc.acl_json ?| %s::text[]
                   AND (%s::text[] IS NULL OR kc.chunk_id = ANY(%s::text[]))
                   AND (%s::text[] IS NULL OR kc.version = ANY(%s::text[]))
                   AND (%s::integer[] IS NULL OR kc.version_order = ANY(%s::integer[]))
                   AND (%s::text[] IS NULL OR kc.metadata_json ->> 'knowledge_base_id' = ANY(%s::text[]))
                   AND (%s::text[] IS NULL OR COALESCE(kc.document_id, kc.source_uri) = ANY(%s::text[]))
                 ORDER BY {distance_expression}, kc.chunk_id
                 LIMIT %s
                """,
                (
                    vector_literal,
                    request.tenant_id,
                    request.tenant_id,
                    getattr(embedder, "model_name", "text-embedding-v4"),
                    dimension,
                    sorted(request.acl_principals),
                    allowed_ids,
                    allowed_ids,
                    versions,
                    versions,
                    version_orders,
                    version_orders,
                    knowledge_bases,
                    knowledge_bases,
                    parent_documents,
                    parent_documents,
                    vector_literal,
                    min(500, max(request.top_k, request.candidate_k)),
                ),
            )
            rows = cursor.fetchall()
        hits: list[Any] = []
        for rank, row in enumerate(rows, start=1):
            chunk_id = _row_value(row, "chunk_id", 0)
            chunk = by_id.get(chunk_id)
            if chunk is None:
                raise RuntimeError("pgvector returned a chunk outside the authorized catalog")
            distance = float(_row_value(row, "distance", 1))
            hits.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=rank,
                    score=1.0 - distance,
                    base_score=1.0 - distance,
                    retrieval_sources=[source_name],
                )
            )
        return HybridSearchResponse(
            backend=backend_name,
            collection_name=f"knowledge:{getattr(embedder, 'model_name', 'text-embedding-v4')}",
            query=request.query,
            filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
            seed_count=min(len(hits), request.top_k),
            expanded_neighbor_count=0,
            raw_candidate_counts={backend_name: len(hits)},
            hits=hits,
        )


class PostgresMemoryStore:
    """MemoryStore-compatible adapter with database-enforced visibility."""

    def __init__(self, repository: PostgresContextRepository) -> None:
        self.repository = repository

    def close(self) -> None:
        self.repository.close()

    def remember(self, item: MemoryItem) -> None:
        self.repository.upsert_memories(
            (item,), _memory_write_access(item), allow_shared_writes=True
        )

    upsert = remember
    add = remember

    def remember_many(self, items: Iterable[MemoryItem]) -> None:
        for item in items:
            if not isinstance(item, MemoryItem):
                raise TypeError("all records must be MemoryItem instances")
            self.remember(item)

    upsert_many = remember_many
    batch_upsert = remember_many

    def get(
        self,
        memory_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> MemoryItem | None:
        candidates = self.repository.fetch_memory_candidates(
            _access(principal), now=now, candidate_limit=10_000
        )
        return next((item for item in candidates if item.memory_id == memory_id), None)

    def forget(
        self,
        memory_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> bool:
        return self.repository.forget_memory(memory_id, _access(principal), now=now)

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
        candidates = self.repository.fetch_memory_candidates(
            _access(principal), scopes=scopes, now=now, candidate_limit=10_000
        )
        return InMemoryMemoryStore(candidates).recall(
            query,
            principal,
            scopes=scopes,
            top_k=top_k,
            now=now,
            include_unmatched=include_unmatched,
        )


def _memory_write_access(item: MemoryItem) -> PostgresContextAccess:
    scope = item.scope
    return PostgresContextAccess(
        tenant_id=item.tenant_id,
        user_id=item.scope_id if scope is MemoryScope.USER else "system",
        conversation_id=item.scope_id if scope is MemoryScope.TASK else "__system__",
        org_id=item.scope_id if scope is MemoryScope.ORG else None,
        agent_id=item.scope_id if scope is MemoryScope.AGENT else None,
    )


def _row_value(row: Any, key: str, index: int) -> Any:
    return row.get(key) if isinstance(row, Mapping) else row[index]


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)


class PostgresContextStores:
    """Convenience owner for one shared pooled repository and its adapters."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout: int = 5,
        runtime: PostgresRuntime | None = None,
    ) -> None:
        self._owns_runtime = runtime is None
        self.runtime = runtime or PostgresRuntime(
            dsn,
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout,
        )
        repository = PostgresContextRepository(
            pool=self.runtime.pool,
            owns_pool=False,
        )
        self.repository = repository
        self.knowledge = PostgresKnowledgeStore(repository)
        self.memory = PostgresMemoryStore(repository)

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()


__all__ = [
    "PostgresContextStores",
    "PostgresKnowledgeStore",
    "PostgresMemoryStore",
]
