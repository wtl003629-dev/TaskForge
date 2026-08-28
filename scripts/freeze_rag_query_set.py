"""Freeze a deterministic TaskForge RAG query-vector set and NumPy baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_reference import (  # noqa: E402
    APPLICATION_MODEL_NAME,
    BAILIAN_MODEL_ID,
    DIMENSION,
    frozen_query_vectors,
    numpy_rank,
    sha256_file,
    sqlite_document_vectors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".taskforge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--acl", action="append", dest="acl_principals", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.tenant_id.strip():
        raise SystemExit("--tenant-id must not be blank")
    if not 1 <= args.limit <= 10_000:
        raise SystemExit("--limit must be 1..10000")
    if not 1 <= args.top_k <= 100:
        raise SystemExit("--top-k must be 1..100")
    source_root = args.source_root.resolve()
    cache_path = source_root / "embeddings-bailian-v4-1024.sqlite3"
    documents = sqlite_document_vectors(
        source_root,
        tenant_id=args.tenant_id,
        acl_principals=sorted(set(args.acl_principals)),
    )
    queries = frozen_query_vectors(cache_path, limit=args.limit)
    for query in queries:
        query["sqlite_numpy_top_k"] = numpy_rank(
            query["vector"], documents, top_k=args.top_k
        )
    report = {
        "schema_version": 1,
        "fixture": "taskforge-rag-pgvector-gate",
        "tenant_id": args.tenant_id,
        "acl_principals": sorted(set(args.acl_principals)),
        "model": APPLICATION_MODEL_NAME,
        "cache_model": BAILIAN_MODEL_ID,
        "dimension": DIMENSION,
        "top_k": args.top_k,
        "reference": "SQLite knowledge_chunks + Bailian cache + NumPy cosine; latest-only and ACL-visible corpus",
        "source_files": {
            "context.sqlite3": sha256_file(source_root / "context.sqlite3"),
            "embeddings-bailian-v4-1024.sqlite3": sha256_file(cache_path),
        },
        "document_count": len(documents),
        "query_count": len(queries),
        "queries": queries,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "query_count": len(queries), "document_count": len(documents)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
