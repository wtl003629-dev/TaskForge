from __future__ import annotations

from taskforge.local_graph import LocalDocumentGraph, entity_phrases


def test_entity_phrases_extract_capitalized_phrases() -> None:
    phrases = entity_phrases(
        "As reported by The Verge and TechCrunch, Apple filed the case."
    )
    assert "The Verge" in phrases
    assert "TechCrunch" in phrases
    assert "Apple" in phrases


def test_graph_seeds_and_one_hop_neighbors() -> None:
    chunks = [
        {
            "document_id": "a",
            "text": "Apple plans a new device and The Verge reported it.",
        },
        {
            "document_id": "b",
            "text": "Apple faces a new investigation per TechCrunch.",
        },
        {
            "document_id": "c",
            "text": "Unrelated baking recipe content.",
        },
    ]
    graph = LocalDocumentGraph(chunks)

    assert graph.node_count == 3
    assert graph.edge_count == 1  # only a-b share "Apple"

    hits = graph.search("Apple")
    # both a and b are seed documents (or neighbors of a seed); c is isolated.
    assert {"a", "b"}.issubset(set(hits))
    assert "c" not in hits

    # A query with no matching entity phrase returns nothing.
    assert graph.search("baking") == []
