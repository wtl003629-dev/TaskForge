"""Ingest one workspace text or machine-readable PDF into TaskForge knowledge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.ingestion import ingest_workspace_document, ingest_workspace_pdf  # noqa: E402
from taskforge.persistent_context import SQLiteKnowledgeStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely ingest one UTF-8 text document or structure-preserving PDF."
        )
    )
    parser.add_argument("path", help="Relative file path below --workspace")
    parser.add_argument("--workspace", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--database", type=Path, default=REPOSITORY_ROOT / ".taskforge" / "context.sqlite3")
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--version-order", type=int, required=True)
    parser.add_argument("--acl", action="append", default=["tenant"])
    parser.add_argument("--chunk-chars", type=int, default=2_000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--max-pdf-bytes", type=int, default=20_000_000)
    parser.add_argument("--max-pdf-pages", type=int, default=200)
    parser.add_argument("--max-pdf-blocks", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with SQLiteKnowledgeStore(args.database) as store:
        common = {
            "workspace_root": args.workspace,
            "relative_path": args.path,
            "tenant_id": args.tenant,
            "knowledge_base_id": args.knowledge_base,
            "version": args.version,
            "version_order": args.version_order,
            "acl": tuple(dict.fromkeys(args.acl)),
            "chunk_chars": args.chunk_chars,
        }
        if Path(args.path).suffix.casefold() == ".pdf":
            result = ingest_workspace_pdf(
                store,
                **common,
                max_bytes=args.max_pdf_bytes,
                max_pages=args.max_pdf_pages,
                max_blocks=args.max_pdf_blocks,
            )
        else:
            result = ingest_workspace_document(
                store,
                **common,
                overlap_chars=args.overlap_chars,
            )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
