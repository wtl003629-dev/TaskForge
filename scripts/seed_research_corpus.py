"""Seed a real research corpus (arxiv abstracts) into the review knowledge store.

Fetches paper metadata from the arxiv API and indexes each title + abstract as a
versioned knowledge chunk readable by the demo user, under the review knowledge
base so research-survey roles can retrieve and cite them.  The chunk's
``evidence_id`` is ``arxiv:<id>`` so the host verification can check that a
survey citation was genuinely retrieved.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.knowledge import KnowledgeChunk
from taskforge.persistent_context import SQLiteKnowledgeStore

_ATOM = "http://www.w3.org/2005/Atom"


def fetch_arxiv(query: str, *, max_results: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    root = ET.fromstring(response.text)
    papers: list[dict[str, str]] = []
    for entry in root.findall(f"{{{_ATOM}}}entry"):
        title = " ".join(
            (entry.findtext(f"{{{_ATOM}}}title", "") or "").split()
        )
        summary = " ".join(
            (entry.findtext(f"{{{_ATOM}}}summary", "") or "").split()
        )
        raw_id = (entry.findtext(f"{{{_ATOM}}}id", "") or "").rsplit("/", 1)[-1]
        published = entry.findtext(f"{{{_ATOM}}}published", "") or ""
        if not title or not summary or not raw_id:
            continue
        papers.append(
            {
                "id": raw_id,
                "title": title,
                "abstract": summary,
                "published": published,
            }
        )
    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default=(
            'all:"retrieval augmented generation" AND '
            '(all:"evaluation" OR all:"benchmark")'
        ),
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--store",
        type=Path,
        default=REPOSITORY_ROOT / ".taskforge" / "context.sqlite3",
    )
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--user", default="demo")
    args = parser.parse_args()

    papers = fetch_arxiv(args.query, max_results=args.count)
    if not papers:
        raise SystemExit("arxiv returned no papers for the query")
    store = SQLiteKnowledgeStore(args.store)
    chunks = []
    for paper in papers:
        chunk_id = hashlib.sha256(paper["id"].encode("utf-8")).hexdigest()[:24]
        text = (
            f"TITLE: {paper['title']}\n"
            f"ABSTRACT: {paper['abstract']}\n"
            f"PUBLISHED: {paper['published']}"
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=args.tenant,
                text=text[:20_000],
                source_uri=f"https://arxiv.org/abs/{paper['id']}",
                document_id=paper["id"],
                acl=frozenset({f"user:{args.user}"}),
                metadata={
                    "knowledge_base_id": "enterprise-review",
                    "evidence_id": f"arxiv:{paper['id']}",
                    "title": paper["title"],
                    "source": "arxiv",
                    "category": "research",
                    "published_at": paper["published"],
                },
            )
        )
    store.upsert_many(chunks)
    print(
        f"seeded {len(chunks)} arxiv chunks into {args.store.resolve()} "
        f"(kb=enterprise-review, acl=user:{args.user})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
