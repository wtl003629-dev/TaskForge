from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from taskforge.ingestion import ingest_workspace_pdf
from taskforge.knowledge import AccessContext
from taskforge.persistent_context import SQLiteKnowledgeStore
from taskforge.security import ToolInputError


def _pdf_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_text_pdf(path: Path, lines: list[str]) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    operations = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        if index:
            operations.append("0 -24 Td")
        operations.append(f"({_pdf_literal(line)}) Tj")
    operations.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(operations).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): writer._add_object(
                        DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    )
                }
            )
        }
    )
    with path.open("wb") as handle:
        writer.write(handle)


def test_pdf_ingestion_persists_provenance_neighbors_and_acl(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "policy.pdf"
    _write_text_pdf(pdf, ["# Change Policy", "Production changes require reviewer evidence."])

    with SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3") as store:
        result = ingest_workspace_pdf(
            store,
            workspace_root=workspace,
            relative_path="policy.pdf",
            tenant_id="tenant-a",
            knowledge_base_id="policy",
            version="2026.1",
            version_order=1,
            acl=("role:reviewer",),
            chunk_chars=256,
        )
        assert result.pages == 1
        assert result.blocks >= 1
        assert result.document_id == "workspace-pdf:policy.pdf"

        denied = store.search("reviewer evidence", AccessContext("tenant-a"), top_k=10)
        assert denied == []
        hits = store.search(
            "reviewer evidence",
            AccessContext("tenant-a", roles=frozenset({"reviewer"})),
            knowledge_base_ids=["policy"],
            top_k=10,
        )
        assert hits
        metadata = hits[0].chunk.metadata
        assert metadata["document_sha256"] == result.sha256
        assert metadata["pages"] == [1]
        assert metadata["provenance"][0]["bbox"]
        assert "previous_chunk_id" in metadata
        assert "next_chunk_id" in metadata


def test_pdf_versions_share_logical_document_and_latest_wins(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "policy.pdf"
    database = tmp_path / "knowledge.sqlite3"
    with SQLiteKnowledgeStore(database) as store:
        _write_text_pdf(pdf, ["Legacy approval alpha."])
        first = ingest_workspace_pdf(
            store,
            workspace_root=workspace,
            relative_path="policy.pdf",
            tenant_id="tenant-a",
            knowledge_base_id="policy",
            version="1",
            version_order=1,
        )
        _write_text_pdf(pdf, ["Current approval beta."])
        second = ingest_workspace_pdf(
            store,
            workspace_root=workspace,
            relative_path="policy.pdf",
            tenant_id="tenant-a",
            knowledge_base_id="policy",
            version="2",
            version_order=2,
        )
        assert first.document_id == second.document_id
        assert first.sha256 != second.sha256
        assert store.search("alpha", AccessContext("tenant-a"), top_k=10) == []
        current = store.search("beta", AccessContext("tenant-a"), top_k=10)
        assert len(current) == 1
        assert current[0].chunk.version == "2"


def test_pdf_ingestion_rejects_escape_and_wrong_extension(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("not a PDF", encoding="utf-8")
    with SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3") as store:
        common = dict(
            store=store,
            workspace_root=workspace,
            tenant_id="tenant-a",
            knowledge_base_id="policy",
            version="1",
            version_order=1,
        )
        with pytest.raises(ToolInputError):
            ingest_workspace_pdf(relative_path="../outside.pdf", **common)
        with pytest.raises(ToolInputError):
            ingest_workspace_pdf(relative_path="note.txt", **common)
