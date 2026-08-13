"""Canonical scholarly identifiers and conservative paper fingerprints."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
_DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.I)
_ARXIV_PREFIX = re.compile(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", re.I)
_ARXIV = re.compile(r"^(?:[a-z-]+(?:\.[a-z-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalise_doi(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _DOI_PREFIX.sub("", value.strip()).rstrip(". ").casefold()
    return cleaned if _DOI.fullmatch(cleaned) else None


def normalise_arxiv_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _ARXIV_PREFIX.sub("", value.strip()).removesuffix(".pdf")
    return cleaned.casefold() if _ARXIV.fullmatch(cleaned) else None


def arxiv_id_from_doi(value: str | None) -> str | None:
    """Recover an arXiv ID from DataCite/Crossref's canonical arXiv DOI."""

    doi = normalise_doi(value)
    prefix = "10.48550/arxiv."
    if doi is None or not doi.startswith(prefix):
        return None
    return normalise_arxiv_id(doi[len(prefix) :])


def normalise_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_NON_WORD.sub(" ", text).split())


def paper_fingerprint(
    title: str,
    authors: list[str] | tuple[str, ...],
    year: int | None,
) -> str:
    first_author = normalise_title(authors[0]) if authors else ""
    payload = f"{normalise_title(title)}\0{first_author}\0{year or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_paper_id(
    *,
    title: str,
    authors: list[str] | tuple[str, ...],
    year: int | None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    semantic_scholar_id: str | None = None,
    openalex_id: str | None = None,
) -> str:
    normalised_doi = normalise_doi(doi)
    if normalised_doi:
        raw = f"doi:{normalised_doi}"
    else:
        normalised_arxiv = normalise_arxiv_id(arxiv_id)
        if normalised_arxiv:
            raw = f"arxiv:{normalised_arxiv}"
        elif semantic_scholar_id:
            raw = f"s2:{semantic_scholar_id.strip()}"
        elif openalex_id:
            raw = f"openalex:{openalex_id.strip().rsplit('/', 1)[-1]}"
        else:
            raw = f"fingerprint:{paper_fingerprint(title, authors, year)}"
    return "paper-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "arxiv_id_from_doi",
    "canonical_paper_id",
    "normalise_arxiv_id",
    "normalise_doi",
    "normalise_title",
    "paper_fingerprint",
]
