"""Scenario-isolated retrieval profile selection.

Selection is deliberately based on query/corpus signals only.  Dataset names
are never consulted, so adding a new table or PDF corpus cannot silently route
through a benchmark-specific code path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .knowledge import KnowledgeChunk
from .rag_evaluation import EvalCorpusDocument

RetrievalProfileName = Literal[
    "general_text",
    "table_numeric",
    "cross_document",
    "pdf_layout",
]


@dataclass(frozen=True)
class CorpusMetadata:
    document_count: int
    table_count: int
    page_count: int
    source_count: int
    has_page_coordinates: bool
    has_table_structure: bool
    source_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryFeatures:
    has_numeric_operation: bool
    has_table_language: bool
    has_cross_document_language: bool
    has_pdf_layout_language: bool
    numeric_tokens: tuple[str, ...]


_NUMBER_RE = re.compile(r"\b(?:19|20)\d{2}\b|[-+]?\d+(?:\.\d+)?%?")
_TABLE_TERMS = frozenset(
    {
        "table",
        "row",
        "rows",
        "column",
        "columns",
        "revenue",
        "profit",
        "assets",
        "liabilities",
        "percentage",
        "percent",
        "growth",
        "margin",
        "average",
        "total",
    }
)
_CROSS_DOCUMENT_PHRASES = (
    "according to both",
    "across documents",
    "between the documents",
    "which source",
    "compare the reports",
    "first article",
    "second article",
    "article from",
    "reported by",
    "reports",
    "according to",
    "sources",
)
_PDF_LAYOUT_PHRASES = (
    "page ",
    "previous page",
    "next page",
    "across pages",
    "table on page",
    "footnote",
    "header and footer",
)


def corpus_metadata(documents: Iterable[EvalCorpusDocument]) -> CorpusMetadata:
    values = list(documents)
    kinds = [str(document.metadata.get("kind", "")).casefold() for document in values]
    sources = {
        str(document.metadata.get("source", "")).strip().casefold()
        for document in values
        if str(document.metadata.get("source", "")).strip()
    }
    page_values = [document.metadata.get("page") for document in values]
    page_count = len({str(value) for value in page_values if value is not None})
    has_page_coordinates = any(
        any(key in document.metadata for key in ("page", "bbox", "coordinates"))
        for document in values
    )
    table_count = sum(kind == "table" for kind in kinds)
    return CorpusMetadata(
        document_count=len(values),
        table_count=table_count,
        page_count=page_count,
        source_count=len(sources),
        has_page_coordinates=has_page_coordinates,
        has_table_structure=table_count > 0
        or any("table_rows" in document.metadata for document in values),
        source_labels=tuple(sorted(sources)),
    )


def knowledge_corpus_metadata(chunks: Iterable[KnowledgeChunk]) -> CorpusMetadata:
    """Derive routing signals from the already-authorized runtime corpus."""

    values = list(chunks)
    document_ids = {chunk.logical_document_id for chunk in values}
    source_labels: set[str] = set()
    pages: set[tuple[str, str]] = set()
    table_count = 0
    has_page_coordinates = False
    has_table_structure = False
    for chunk in values:
        metadata = chunk.metadata
        raw_source = metadata.get("source") or metadata.get("title") or chunk.source_uri
        source = str(raw_source).strip().casefold()
        if source:
            source_labels.add(source)

        raw_pages = metadata.get("pages")
        if isinstance(raw_pages, (list, tuple, set, frozenset)):
            pages.update(
                (chunk.logical_document_id, str(page))
                for page in raw_pages
                if page is not None
            )
        elif metadata.get("page") is not None:
            pages.add((chunk.logical_document_id, str(metadata["page"])))

        provenance = metadata.get("provenance")
        has_page_coordinates = has_page_coordinates or any(
            key in metadata for key in ("page", "pages", "bbox", "coordinates")
        ) or (
            isinstance(provenance, list)
            and any(
                isinstance(item, dict) and ("page" in item or "bbox" in item)
                for item in provenance
            )
        )
        block_types = metadata.get("block_types")
        types = (
            {str(item).casefold() for item in block_types}
            if isinstance(block_types, (list, tuple, set, frozenset))
            else set()
        )
        is_table = (
            str(metadata.get("kind", "")).casefold() == "table"
            or "table" in types
            or bool(metadata.get("table_rows"))
        )
        table_count += int(is_table)
        has_table_structure = has_table_structure or is_table

    return CorpusMetadata(
        document_count=len(document_ids),
        table_count=table_count,
        page_count=len(pages),
        source_count=len(source_labels),
        has_page_coordinates=has_page_coordinates,
        has_table_structure=has_table_structure,
        source_labels=tuple(sorted(source_labels)),
    )


def query_features(query: str) -> QueryFeatures:
    lowered = " ".join(query.casefold().split())
    tokens = tuple(sorted(set(_NUMBER_RE.findall(lowered))))
    words = set(re.findall(r"[a-z][a-z-]+", lowered))
    has_numeric_operation = bool(
        tokens
        or words.intersection(
            {
                "count",
                "many",
                "percentage",
                "percent",
                "ratio",
                "difference",
                "growth",
                "increase",
                "decrease",
                "average",
                "total",
            }
        )
    )
    return QueryFeatures(
        has_numeric_operation=has_numeric_operation,
        has_table_language=bool(words.intersection(_TABLE_TERMS)),
        has_cross_document_language=any(
            phrase in lowered for phrase in _CROSS_DOCUMENT_PHRASES
        ),
        has_pdf_layout_language=any(
            phrase in lowered for phrase in _PDF_LAYOUT_PHRASES
        ),
        numeric_tokens=tokens,
    )


def select_retrieval_profile(
    query: str,
    corpus: CorpusMetadata,
) -> RetrievalProfileName:
    """Select a profile without consulting a dataset or benchmark identifier."""
    features = query_features(query)
    lowered = " ".join(query.casefold().split())
    named_source_matches = {
        label
        for label in corpus.source_labels
        if len(label) >= 3 and label in lowered
    }
    if (
        corpus.page_count > 1
        and corpus.has_page_coordinates
        and (features.has_pdf_layout_language or not corpus.has_table_structure)
    ):
        return "pdf_layout"
    if corpus.has_table_structure and (
        features.has_numeric_operation or features.has_table_language
    ):
        return "table_numeric"
    if corpus.source_count > 1 and (
        features.has_cross_document_language or len(named_source_matches) >= 2
    ):
        return "cross_document"
    return "general_text"


def profile_metadata(
    profile: RetrievalProfileName,
    corpus: CorpusMetadata,
    features: QueryFeatures,
) -> dict[str, object]:
    """Return auditable selection inputs for an evaluation prediction row."""
    return {
        "name": profile,
        "selection": {
            "query_features": {
                "has_numeric_operation": features.has_numeric_operation,
                "has_table_language": features.has_table_language,
                "has_cross_document_language": features.has_cross_document_language,
                "has_pdf_layout_language": features.has_pdf_layout_language,
                "numeric_tokens": list(features.numeric_tokens),
            },
            "corpus": {
                "document_count": corpus.document_count,
                "table_count": corpus.table_count,
                "page_count": corpus.page_count,
                "source_count": corpus.source_count,
                "source_labels": list(corpus.source_labels),
                "has_page_coordinates": corpus.has_page_coordinates,
                "has_table_structure": corpus.has_table_structure,
            },
        },
    }


__all__ = [
    "CorpusMetadata",
    "QueryFeatures",
    "RetrievalProfileName",
    "corpus_metadata",
    "knowledge_corpus_metadata",
    "profile_metadata",
    "query_features",
    "select_retrieval_profile",
]
