"""Versioned PDF parser boundary used before RAG indexing."""

from .contracts import (
    DocumentBlock,
    ParsedDocument,
    ParseQualityReport,
    ParserAttempt,
    VisualEvidence,
)
from .quality_gate import ParseQualityPolicy, evaluate_parse_quality

__all__ = [
    "DocumentBlock",
    "ParsedDocument",
    "ParseQualityPolicy",
    "ParseQualityReport",
    "ParserAttempt",
    "VisualEvidence",
    "evaluate_parse_quality",
]
