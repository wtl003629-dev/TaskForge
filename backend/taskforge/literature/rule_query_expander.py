"""Deterministic, zero-API query expansion for paper retrieval."""

from __future__ import annotations

import re

from ..research_protocol import EvidenceIntent
from .evidence_query_expander import protected_query_terms

_TOKEN_RE = re.compile(r"[\w][\w.-]*", re.UNICODE)
_QUESTION_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "would",
    }
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "their",
        "this",
        "that",
        "to",
        "with",
    }
)


class RuleEvidenceQueryExpander:
    """Create a keyword/entity query without a model call.

    The original query is never changed by this class.  It returns a pair so
    it can satisfy the existing expander protocol; the retrieval service's
    ``keyword`` mode consumes only the second member.
    """

    async def expand(
        self,
        query: str,
        intent: EvidenceIntent,
    ) -> tuple[str, str]:
        del intent
        tokens = _TOKEN_RE.findall(query)
        protected = set(protected_query_terms(query))
        selected: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            folded = token.casefold()
            keep = (
                folded in protected
                or any(char.isdigit() for char in token)
                or token.isupper()
                or len(token) >= 4
            ) and folded not in _QUESTION_WORDS and folded not in _STOPWORDS
            if keep and folded not in seen:
                selected.append(token)
                seen.add(folded)
        # Keep the query searchable even when it contains only short/common
        # words.  A deterministic order makes manifests reproducible.
        if len(selected) < 2:
            for token in tokens:
                folded = token.casefold()
                if folded not in seen:
                    selected.append(token)
                    seen.add(folded)
                if len(selected) >= 2:
                    break
        keyword = " ".join(selected[:32]).strip()
        if not keyword:
            keyword = query.strip()
        return query.strip(), keyword

    async def aclose(self) -> None:
        return None


__all__ = ["RuleEvidenceQueryExpander"]
