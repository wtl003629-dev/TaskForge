"""Operator-controlled ingestion for the persistent knowledge backend."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .document_ingestion import StructuredDocument, extract_pdf_document
from .knowledge import KnowledgeChunk
from .security import ToolInputError, resolve_workspace_path


class KnowledgeWriter(Protocol):
    """Minimal persistence port required by the operator ingestion flow."""

    def replace_document_version(self, chunks: Iterable[KnowledgeChunk]) -> int: ...


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source_uri: str
    document_id: str
    version: str
    version_order: int
    chunks: int
    bytes_read: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PDFIngestionResult(IngestionResult):
    """Durable summary of one structure-preserving PDF ingestion."""

    pages: int
    blocks: int


def ingest_workspace_document(
    store: KnowledgeWriter,
    *,
    workspace_root: str | Path,
    relative_path: str,
    tenant_id: str,
    knowledge_base_id: str,
    version: str,
    version_order: int,
    acl: tuple[str, ...] = ("tenant",),
    chunk_chars: int = 2_000,
    overlap_chars: int = 200,
    max_bytes: int = 1_000_000,
) -> IngestionResult:
    """Read one safe text file and atomically replace its selected version."""

    if not tenant_id.strip() or not knowledge_base_id.strip() or not version.strip():
        raise ValueError("tenant_id, knowledge_base_id, and version are required")
    if version_order < 0:
        raise ValueError("version_order must be non-negative")
    if not 256 <= chunk_chars <= 20_000:
        raise ValueError("chunk_chars must be between 256 and 20000")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than chunk_chars")
    if not acl or any(not item.strip() for item in acl):
        raise ValueError("acl must contain non-empty host-selected tokens")

    root = Path(workspace_root).resolve(strict=True)
    path = resolve_workspace_path(root, relative_path)
    if not path.is_file():
        raise ToolInputError("ingestion target is not a regular file")
    size = path.stat().st_size
    if size > max_bytes:
        raise ToolInputError(f"ingestion target exceeds {max_bytes} bytes")
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ToolInputError("binary files cannot be ingested")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolInputError("ingestion target must be valid UTF-8") from exc
    if not text.strip():
        raise ToolInputError("empty documents cannot be ingested")

    source_uri = path.relative_to(root).as_posix()
    document_id = f"workspace:{source_uri}"
    digest = hashlib.sha256(raw).hexdigest()
    chunks: list[KnowledgeChunk] = []
    for index, (start, end, content) in enumerate(
        _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    ):
        chunk_id = hashlib.sha256(
            f"{tenant_id}\0{document_id}\0{version}\0{index}".encode()
        ).hexdigest()[:24]
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                text=content,
                source_uri=source_uri,
                document_id=document_id,
                version=version,
                version_order=version_order,
                acl=frozenset(acl),
                metadata={
                    "knowledge_base_id": knowledge_base_id,
                    "sha256": digest,
                    "chunk_index": index,
                    "char_start": start,
                    "char_end": end,
                    "line_start": text.count("\n", 0, start) + 1,
                    "line_end": text.count("\n", 0, end) + 1,
                },
            )
        )
    store.replace_document_version(chunks)
    return IngestionResult(
        source_uri=source_uri,
        document_id=document_id,
        version=version,
        version_order=version_order,
        chunks=len(chunks),
        bytes_read=len(raw),
        sha256=digest,
    )


def ingest_workspace_pdf(
    store: KnowledgeWriter,
    *,
    workspace_root: str | Path,
    relative_path: str,
    tenant_id: str,
    knowledge_base_id: str,
    version: str,
    version_order: int,
    acl: tuple[str, ...] = ("tenant",),
    chunk_chars: int = 2_000,
    max_bytes: int = 20_000_000,
    max_pages: int = 200,
    max_blocks: int = 20_000,
) -> PDFIngestionResult:
    """Extract one workspace PDF and atomically publish its selected version.

    The source path, rather than the PDF byte hash, is the logical document
    identity.  Consequently a changed file becomes a newer version of the
    same document and ``latest_only`` retrieval cannot accidentally return
    stale pages.  The extractor's content-addressed identity remains in
    metadata for provenance and reproducibility.
    """

    _validate_common_ingestion_fields(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        version=version,
        version_order=version_order,
        acl=acl,
    )
    root = Path(workspace_root).resolve(strict=True)
    path = resolve_workspace_path(root, relative_path)
    if not path.is_file():
        raise ToolInputError("ingestion target is not a regular file")
    source_uri = path.relative_to(root).as_posix()
    if path.suffix.casefold() != ".pdf":
        raise ToolInputError("PDF ingestion target must use a .pdf extension")

    document = extract_pdf_document(
        path,
        source_uri=source_uri,
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_blocks=max_blocks,
        chunk_chars=chunk_chars,
    )
    stable_document_id = f"workspace-pdf:{source_uri}"
    chunks = _pdf_knowledge_chunks(
        document,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=stable_document_id,
        version=version,
        version_order=version_order,
        acl=acl,
    )
    store.replace_document_version(chunks)
    return PDFIngestionResult(
        source_uri=source_uri,
        document_id=stable_document_id,
        version=version,
        version_order=version_order,
        chunks=len(chunks),
        bytes_read=document.bytes_read,
        sha256=document.sha256,
        pages=document.page_count,
        blocks=len(document.blocks),
    )


def _pdf_knowledge_chunks(
    document: StructuredDocument,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    version: str,
    version_order: int,
    acl: tuple[str, ...],
) -> list[KnowledgeChunk]:
    blocks_by_id = {block.block_id: block for block in document.blocks}
    stored_ids = [
        hashlib.sha256(
            (
                f"{tenant_id}\0{document_id}\0{version}\0{index}\0"
                f"{chunk.content_hash}"
            ).encode()
        ).hexdigest()[:24]
        for index, chunk in enumerate(document.chunks)
    ]
    result: list[KnowledgeChunk] = []
    for index, (chunk, chunk_id) in enumerate(zip(document.chunks, stored_ids, strict=True)):
        chunk_blocks = [
            blocks_by_id[block_id]
            for block_id in chunk.block_ids
            if block_id in blocks_by_id
        ]
        block_types = sorted({block.kind for block in chunk_blocks})
        table_rows = [
            [list(row) for row in block.table_rows]
            for block in chunk_blocks
            if block.kind == "table"
        ]
        provenance = [
            {
                "block_id": item.block_id,
                "page": item.page,
                "bbox": list(item.bbox),
                "content_hash": item.content_hash,
                "previous_block_id": item.previous_block_id,
                "next_block_id": item.next_block_id,
            }
            for item in chunk.provenance
        ]
        result.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                text=chunk.text,
                source_uri=document.source_uri,
                document_id=document_id,
                version=version,
                version_order=version_order,
                acl=frozenset(acl),
                metadata={
                    "knowledge_base_id": knowledge_base_id,
                    "document_sha256": document.sha256,
                    "extractor_document_id": document.document_id,
                    "chunk_content_hash": chunk.content_hash,
                    "chunk_index": index,
                    "pages": list(chunk.pages),
                    "block_ids": list(chunk.block_ids),
                    "block_types": block_types,
                    "kind": (
                        "table"
                        if block_types == ["table"]
                        else "mixed"
                        if "table" in block_types
                        else "paragraph"
                    ),
                    "table_rows": table_rows,
                    "provenance": provenance,
                    "heading": chunk.heading,
                    "previous_chunk_id": stored_ids[index - 1] if index > 0 else None,
                    "next_chunk_id": stored_ids[index + 1] if index + 1 < len(stored_ids) else None,
                    "previous_block_id": chunk.previous_block_id,
                    "next_block_id": chunk.next_block_id,
                },
            )
        )
    return result


def _validate_common_ingestion_fields(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    version: str,
    version_order: int,
    acl: tuple[str, ...],
) -> None:
    if not tenant_id.strip() or not knowledge_base_id.strip() or not version.strip():
        raise ValueError("tenant_id, knowledge_base_id, and version are required")
    if version_order < 0:
        raise ValueError("version_order must be non-negative")
    if not acl or any(not item.strip() for item in acl):
        raise ValueError("acl must contain non-empty host-selected tokens")


def _chunk_text(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + chunk_chars)
        end = hard_end
        if hard_end < length:
            boundary = text.rfind("\n", start + chunk_chars // 2, hard_end)
            if boundary > start:
                end = boundary + 1
        content = text[start:end].strip()
        if content:
            chunks.append((start, end, content))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return chunks


__all__ = [
    "IngestionResult",
    "PDFIngestionResult",
    "ingest_workspace_document",
    "ingest_workspace_pdf",
]
