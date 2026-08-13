"""Deterministic, ACL-safe evidence graph features for retrieval reranking.

The graph is deliberately local and candidate-preserving.  It is built from
the same ``HybridChunk`` records that feed the retrieval index and only uses
structural/provenance links already present in chunk metadata.  It never
creates a new evidence item during reranking, so Candidate@K is unchanged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .hybrid_retrieval import HybridChunk, HybridSearchHit
from .knowledge import tokenise
from .local_graph import entity_phrases


@dataclass(frozen=True)
class EvidenceNode:
    """Searchable chunk node and the provenance needed for graph features."""

    chunk_id: str
    document_id: str
    parent_document_id: str
    section_id: str | None
    source_uri: str
    entities: frozenset[str]
    terms: frozenset[str]


@dataclass(frozen=True)
class GraphRerankFeatures:
    """Auditable features for one candidate."""

    base_rank: int
    base_score: float
    graph_support: float
    entity_support: float
    section_support: float
    adjacency_support: float
    final_score: float
    raw_score: float | None = None
    raw_base_score: float | None = None
    reranker_score: float | None = None
    ppr_support: float = 0.0


@dataclass(frozen=True)
class EvidenceQueryPlan:
    """Small, serializable query contract consumed by graph features."""

    raw_query: str
    terms: frozenset[str]
    entities: frozenset[str]
    section_hints: frozenset[str]
    max_hops: int = 1

    @classmethod
    def from_text(cls, query: str, *, max_hops: int = 1) -> EvidenceQueryPlan:
        cleaned = str(query).strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        if not 1 <= max_hops <= 2:
            raise ValueError("max_hops must be between 1 and 2")
        terms = frozenset(tokenise(cleaned))
        entities = entity_phrases(cleaned)
        section_terms = {
            "abstract",
            "background",
            "method",
            "methods",
            "result",
            "results",
            "discussion",
            "conclusion",
        }
        return cls(
            raw_query=cleaned,
            terms=terms,
            entities=entities,
            section_hints=frozenset(terms.intersection(section_terms)),
            max_hops=max_hops,
        )


@dataclass(frozen=True)
class GraphRerankResult:
    """Candidate-preserving graph reranking result."""

    hits: tuple[HybridSearchHit, ...]
    features: Mapping[str, GraphRerankFeatures]
    seed_chunk_ids: tuple[str, ...]
    node_count: int
    edge_count: int


def _value(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(field, default)
    return getattr(item, field, default)


def _metadata(item: Any) -> Mapping[str, Any]:
    raw = _value(item, "metadata", {})
    return raw if isinstance(raw, Mapping) else {}


def _text(item: Any) -> str:
    return str(_value(item, "text", ""))


def _clean_id(value: Any) -> str | None:
    candidate = str(value or "").strip()
    return candidate or None


def _parent_id(item: Any) -> str:
    metadata = _metadata(item)
    return (
        _clean_id(metadata.get("parent_document_id"))
        or _clean_id(_value(item, "document_id"))
        or _clean_id(_value(item, "chunk_id"))
        or "unknown"
    )


def _section_id(item: Any) -> str | None:
    metadata = _metadata(item)
    return _clean_id(metadata.get("section_id")) or _clean_id(metadata.get("section"))


def _linked_chunk_ids(item: Any) -> tuple[str, ...]:
    metadata = _metadata(item)
    values = (
        _value(item, "previous_chunk_id"),
        _value(item, "next_chunk_id"),
        metadata.get("previous_document_id"),
        metadata.get("next_document_id"),
    )
    return tuple(value for value in (_clean_id(raw) for raw in values) if value)


class LocalEvidenceGraph:
    """Small in-memory graph over chunks, sections, entities and provenance.

    The graph is intentionally deterministic: no model calls, no network
    dependency and no answer/evidence labels.  Structural links are bounded
    to nearby chunks and shared sections/entities so construction remains
    linear in normal evaluation corpora.
    """

    def __init__(self, chunks: Iterable[HybridChunk | Mapping[str, Any]]) -> None:
        materialized = list(chunks)
        self._nodes: dict[str, EvidenceNode] = {}
        self._chunks: dict[str, HybridChunk | Mapping[str, Any]] = {}
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._section_nodes: dict[str, list[str]] = defaultdict(list)
        self._entity_nodes: dict[str, list[str]] = defaultdict(list)
        for chunk in materialized:
            chunk_id = _clean_id(_value(chunk, "chunk_id"))
            document_id = _clean_id(_value(chunk, "document_id"))
            if not chunk_id or not document_id or chunk_id in self._nodes:
                continue
            text = _text(chunk)
            node = EvidenceNode(
                chunk_id=chunk_id,
                document_id=document_id,
                parent_document_id=_parent_id(chunk),
                section_id=_section_id(chunk),
                source_uri=str(_value(chunk, "source_uri", "")),
                entities=entity_phrases(text),
                terms=frozenset(tokenise(text)),
            )
            self._nodes[chunk_id] = node
            self._chunks[chunk_id] = chunk
            if node.section_id:
                self._section_nodes[node.section_id].append(chunk_id)
            for entity in node.entities:
                self._entity_nodes[entity].append(chunk_id)

        # Explicit document links are the strongest structural signal.
        for chunk in materialized:
            chunk_id = _clean_id(_value(chunk, "chunk_id"))
            if chunk_id not in self._nodes:
                continue
            for linked_id in _linked_chunk_ids(chunk):
                if linked_id in self._nodes:
                    self._connect(chunk_id, linked_id, 1.0)

        # Same-section links are capped to the nearest two records on either
        # side.  This gives section continuity without an O(n^2) clique.
        for members in self._section_nodes.values():
            ordered = list(dict.fromkeys(members))
            for index, chunk_id in enumerate(ordered):
                for linked_id in ordered[max(0, index - 2) : index + 3]:
                    if linked_id != chunk_id:
                        self._connect(chunk_id, linked_id, 0.55)

        # Shared entities connect evidence across sections/documents.  Cap the
        # contribution so common entities cannot dominate the graph score.
        for members in self._entity_nodes.values():
            ordered = list(dict.fromkeys(members))
            for index, chunk_id in enumerate(ordered):
                for linked_id in ordered[index + 1 : index + 9]:
                    self._connect(chunk_id, linked_id, 0.35)

    def _connect(self, left: str, right: str, weight: float) -> None:
        if left == right or not math.isfinite(weight) or weight <= 0:
            return
        self._edges[left][right] = max(weight, self._edges[left].get(right, 0.0))
        self._edges[right][left] = max(weight, self._edges[right].get(left, 0.0))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self._edges.values()) // 2

    def _graph_support(
        self,
        candidate_id: str,
        seed_weights: Mapping[str, float],
        allowed_parent_document_ids: frozenset[str] | None = None,
    ) -> float:
        neighbors = self._edges.get(candidate_id, {})
        return sum(
            seed_weights.get(neighbor, 0.0) * weight
            for neighbor, weight in neighbors.items()
            if allowed_parent_document_ids is None
            or self._nodes.get(neighbor) is not None
            and self._nodes[neighbor].parent_document_id
            in allowed_parent_document_ids
        )

    def _ppr_support(
        self,
        seed_weights: Mapping[str, float],
        *,
        iterations: int = 2,
        damping: float = 0.85,
        allowed_parent_document_ids: frozenset[str] | None = None,
    ) -> Mapping[str, float]:
        """Compute a bounded personalized PageRank over the local graph."""

        scoped_nodes = {
            node_id
            for node_id, node in self._nodes.items()
            if allowed_parent_document_ids is None
            or node.parent_document_id in allowed_parent_document_ids
        }
        scores = {
            node_id: score
            for node_id, score in seed_weights.items()
            if node_id in scoped_nodes
        }
        teleport = dict(seed_weights)
        for _ in range(iterations):
            next_scores: dict[str, float] = {
                node_id: (1.0 - damping) * teleport.get(node_id, 0.0)
                for node_id in scoped_nodes
            }
            for node_id, score in scores.items():
                neighbors = self._edges.get(node_id, {})
                neighbors = {
                    neighbor: weight
                    for neighbor, weight in neighbors.items()
                    if neighbor in scoped_nodes
                }
                total_weight = sum(neighbors.values())
                if total_weight <= 0:
                    continue
                for neighbor, edge_weight in neighbors.items():
                    next_scores[neighbor] = next_scores.get(neighbor, 0.0) + (
                        damping * score * edge_weight / total_weight
                    )
            scores = next_scores
        maximum = max(scores.values(), default=0.0)
        if maximum <= 1e-12:
            return {}
        return {
            node_id: min(1.0, score / maximum)
            for node_id, score in scores.items()
            if score > 0
        }

    def expand_candidates(
        self,
        hits: Sequence[HybridSearchHit],
        *,
        max_add: int = 5,
        hops: int = 1,
        allowed_parent_document_ids: frozenset[str] | None = None,
    ) -> tuple[HybridSearchHit, ...]:
        """Return bounded graph neighbors as auditable candidate additions.

        Expansion is opt-in and never used by ``rerank``.  The caller owns the
        retrieval budget and ACL/filter contract; ``allowed_parent_document_ids``
        is therefore required by contextual routes so a graph edge cannot pull
        evidence from another paper or parent document.
        """

        if max_add < 0:
            raise ValueError("max_add must be non-negative")
        if not 1 <= hops <= 2:
            raise ValueError("hops must be between 1 and 2")
        if not hits or max_add == 0:
            return ()
        existing = {hit.chunk.chunk_id for hit in hits}
        frontier = list(existing)
        visited = set(existing)
        discovered: dict[str, tuple[float, int, str]] = {}
        for distance in range(1, hops + 1):
            next_frontier: list[str] = []
            for origin in frontier:
                for neighbor, edge_weight in self._edges.get(origin, {}).items():
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    node = self._nodes.get(neighbor)
                    if node is None:
                        continue
                    if (
                        allowed_parent_document_ids is not None
                        and node.parent_document_id not in allowed_parent_document_ids
                    ):
                        continue
                    score = float(edge_weight) / distance
                    prior = discovered.get(neighbor)
                    candidate = (score, distance, origin)
                    if prior is None or candidate > prior:
                        discovered[neighbor] = candidate
            frontier = next_frontier

        ordered = sorted(
            discovered.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )[:max_add]
        output: list[HybridSearchHit] = []
        next_rank = len(hits) + 1
        for offset, (chunk_id, (score, distance, origin)) in enumerate(ordered):
            raw_chunk = self._chunks[chunk_id]
            chunk = (
                raw_chunk
                if isinstance(raw_chunk, HybridChunk)
                else HybridChunk.model_validate(raw_chunk)
            )
            output.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=next_rank + offset,
                    score=score,
                    base_score=score,
                    retrieval_sources=["graph_feature_rerank"],
                    neighbor_of_chunk_id=origin,
                    neighbor_distance=distance,
                )
            )
        return tuple(output)

    def rerank(
        self,
        query: str,
        hits: Sequence[HybridSearchHit],
        *,
        query_plan: EvidenceQueryPlan | None = None,
        seed_k: int = 10,
        graph_weight: float = 0.20,
        entity_weight: float = 0.10,
        section_weight: float = 0.05,
        adjacency_weight: float = 0.05,
        ppr_weight: float = 0.0,
        allowed_parent_document_ids: frozenset[str] | None = None,
    ) -> GraphRerankResult:
        """Reorder only the supplied hits and return auditable features."""

        if not hits:
            return GraphRerankResult((), {}, (), self.node_count, self.edge_count)
        if not 1 <= seed_k <= len(hits):
            raise ValueError("seed_k must be between 1 and the number of hits")
        weights = (
            graph_weight,
            entity_weight,
            section_weight,
            adjacency_weight,
            ppr_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("graph feature weights must be finite and non-negative")
        if sum(weights) > 1.0:
            raise ValueError("graph feature weights must sum to at most 1")

        candidate_ids = [hit.chunk.chunk_id for hit in hits]
        known_hits = [
            hit
            for hit in hits
            if hit.chunk.chunk_id in self._nodes
            and (
                allowed_parent_document_ids is None
                or self._nodes[hit.chunk.chunk_id].parent_document_id
                in allowed_parent_document_ids
            )
        ]
        seed_hits = known_hits[:seed_k]
        seed_ids = tuple(hit.chunk.chunk_id for hit in seed_hits)
        seed_weights = {
            hit.chunk.chunk_id: 1.0 / (index + 1)
            for index, hit in enumerate(seed_hits)
        }
        max_seed_weight = max(seed_weights.values(), default=1.0)
        seed_entities = set().union(
            *(self._nodes[chunk_id].entities for chunk_id in seed_ids)
        )
        seed_sections = {
            self._nodes[chunk_id].section_id
            for chunk_id in seed_ids
            if self._nodes[chunk_id].section_id
        }
        plan = query_plan or EvidenceQueryPlan.from_text(query)
        query_entities = plan.entities
        query_terms = plan.terms
        ppr_scores = self._ppr_support(
            seed_weights,
            allowed_parent_document_ids=allowed_parent_document_ids,
        )

        features: dict[str, GraphRerankFeatures] = {}
        scored: list[tuple[float, int, str, HybridSearchHit]] = []
        for index, hit in enumerate(hits):
            node = self._nodes.get(hit.chunk.chunk_id)
            ppr_support = 0.0
            # Cross-encoders and hybrid backends do not share a calibrated
            # score scale (and some return near-identical scores for many
            # candidates).  Rank is the only stable baseline signal here;
            # using it prevents a graph feature from promoting a low-ranked
            # candidate solely because of an incomparable raw score.
            base_score = 1.0 / (index + 1)
            if node is None:
                graph_support = entity_support = section_support = adjacency_support = 0.0
            else:
                graph_support = min(
                    1.0,
                    self._graph_support(
                        hit.chunk.chunk_id,
                        seed_weights,
                        allowed_parent_document_ids,
                    )
                    / max_seed_weight,
                )
                entity_support = (
                    len(node.entities.intersection(seed_entities)) / len(seed_entities)
                    if seed_entities
                    else 0.0
                )
                section_support = (
                    1.0
                    if node.section_id is not None and node.section_id in seed_sections
                    else 0.0
                )
                linked = set(self._edges.get(node.chunk_id, {}))
                adjacency_support = min(
                    1.0,
                    len(linked.intersection(seed_ids)) / max(1, min(seed_k, 3)),
                )
                # Query entities are a direct semantic-structure signal.  It
                # is folded into entity support only when present, preserving
                # the deterministic score for normal questions.
                if query_entities:
                    entity_support = max(
                        entity_support,
                        len(node.entities.intersection(query_entities))
                        / len(query_entities),
                    )
                elif query_terms:
                    overlap = len(node.terms.intersection(query_terms))
                    entity_support = max(entity_support, overlap / len(query_terms))
            final_score = (
                (1.0 - sum(weights)) * base_score
                + graph_weight * graph_support
                + entity_weight * entity_support
                + section_weight * section_support
                + adjacency_weight * adjacency_support
            )
            if node is not None:
                ppr_support = ppr_scores.get(hit.chunk.chunk_id, 0.0)
            final_score += ppr_weight * ppr_support
            features[hit.chunk.chunk_id] = GraphRerankFeatures(
                base_rank=index + 1,
                base_score=base_score,
                graph_support=graph_support,
                entity_support=entity_support,
                section_support=section_support,
                adjacency_support=adjacency_support,
                final_score=final_score,
                raw_score=float(hit.score),
                raw_base_score=float(hit.base_score),
                reranker_score=(
                    None
                    if hit.reranker_score is None
                    else float(hit.reranker_score)
                ),
                ppr_support=ppr_support,
            )
            scored.append((final_score, index, hit.chunk.chunk_id, hit))

        ordered = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
        reranked: list[HybridSearchHit] = []
        for rank, (score, _, _, hit) in enumerate(ordered, start=1):
            sources = list(hit.retrieval_sources)
            if "graph_feature_rerank" not in sources:
                sources.append("graph_feature_rerank")
            reranked.append(
                hit.model_copy(
                    update={
                        "rank": rank,
                        "score": float(score),
                        "retrieval_sources": sources,
                    }
                )
            )
        if {hit.chunk.chunk_id for hit in reranked} != set(candidate_ids):
            raise RuntimeError("graph reranker changed the candidate set")
        return GraphRerankResult(
            tuple(reranked),
            features,
            seed_ids,
            self.node_count,
            self.edge_count,
        )


__all__ = [
    "EvidenceQueryPlan",
    "EvidenceNode",
    "GraphRerankFeatures",
    "GraphRerankResult",
    "LocalEvidenceGraph",
]
