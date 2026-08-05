"""Deterministic, local document co-occurrence graph for the RAG ablation.

This is an intentionally small, in-process graph (no Neo4j server).  Nodes are
documents; an edge joins two documents that share at least one capitalized
entity phrase.  A query is expanded from its own entity phrases to seed
documents, then one hop out to their graph neighbors, producing a ranked
document list that can be fused with lexical/dense retrieval by RRF.

The graph is built purely from corpus text, so it is reproducible and never
needs network access.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

_ENTITY_PHRASE = re.compile(r"\b[A-Z][a-zA-Z]*(?:[ -][A-Z][a-zA-Z]*){0,2}\b")


def entity_phrases(text: str) -> frozenset[str]:
    """Return deduplicated capitalized entity-like phrases in a text."""

    return frozenset(
        phrase.strip()
        for phrase in _ENTITY_PHRASE.findall(text)
        if phrase.strip()
    )


class LocalDocumentGraph:
    """Document co-occurrence graph built from corpus chunks."""

    def __init__(self, chunks: Iterable[Any]) -> None:
        phrases_by_document: dict[str, set[str]] = {}
        for chunk in chunks:
            document_id = getattr(chunk, "document_id", None)
            if document_id is None:
                document_id = chunk.get("document_id")
            text = getattr(chunk, "text", None)
            if text is None:
                text = chunk.get("text", "")
            phrases_by_document.setdefault(str(document_id), set()).update(
                entity_phrases(str(text))
            )
        self._phrases = {
            document_id: frozenset(phrases)
            for document_id, phrases in phrases_by_document.items()
        }
        phrase_documents: dict[str, list[str]] = defaultdict(list)
        for document_id, phrases in self._phrases.items():
            for phrase in phrases:
                phrase_documents[phrase].append(document_id)
        neighbors: dict[str, set[str]] = defaultdict(set)
        for members in phrase_documents.values():
            for index, left in enumerate(members):
                for right in members[index + 1 :]:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
        self._neighbors = {
            document_id: frozenset(linked)
            for document_id, linked in neighbors.items()
        }

    @property
    def node_count(self) -> int:
        return len(self._phrases)

    @property
    def edge_count(self) -> int:
        return sum(len(linked) for linked in self._neighbors.values()) // 2

    def search(
        self,
        query: str,
        *,
        seed_weight: int = 2,
        neighbor_weight: int = 1,
        max_results: int = 12,
    ) -> list[str]:
        """Rank seed documents and their one-hop neighbors by contribution."""

        query_phrases = entity_phrases(query)
        if not query_phrases:
            return []
        seeds = [
            document_id
            for document_id, phrases in self._phrases.items()
            if phrases & query_phrases
        ]
        if not seeds:
            return []
        contributions: dict[str, int] = {}
        for seed in seeds:
            contributions[seed] = contributions.get(seed, 0) + seed_weight
        for seed in seeds:
            for neighbor in self._neighbors.get(seed, ()):
                contributions[neighbor] = (
                    contributions.get(neighbor, 0) + neighbor_weight
                )
        ordered = sorted(
            contributions.items(), key=lambda item: (-item[1], item[0])
        )
        return [document_id for document_id, _ in ordered[:max_results]]
