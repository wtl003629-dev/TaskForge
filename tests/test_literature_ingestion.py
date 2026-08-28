from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
import pytest
from reportlab.pdfgen import canvas

from taskforge.knowledge import AccessContext
from taskforge.literature.ingestion import (
    PaperDownloadError,
    PaperIngestionService,
    SafePDFDownloader,
)
from taskforge.literature.repository import LiteratureAccess, SQLiteLiteratureRepository
from taskforge.persistent_context import SQLiteKnowledgeStore
from taskforge.research_protocol import LiteratureRequest, PaperCard, ResearchScope


class _FailingDownloader:
    max_bytes = 1_000_000

    async def download(self, url: str, target: Path) -> int:
        raise PaperDownloadError("publisher denied access")

    async def aclose(self) -> None:
        return None


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve_pdf(self, doi: str) -> str | None:
        self.calls.append(doi)
        return "https://8.8.8.8/open.pdf"

    async def aclose(self) -> None:
        return None


class _EmbeddingPrewarmer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        values = list(texts)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("provider details must not reach ingestion status")
        return [[1.0, 0.0, 0.0] for _ in values]


def _pdf_bytes(text: str = "User uploaded evidence document.") -> bytes:
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    document.drawString(72, 720, text)
    document.save()
    return buffer.getvalue()


def _seed_scope(
    repository: SQLiteLiteratureRepository,
    access: LiteratureAccess,
) -> ResearchScope:
    repository.save_request(
        access,
        LiteratureRequest(request_id="request-1", query="scope-bound retrieval"),
    )
    repository.upsert_paper(
        access,
        PaperCard(
            paper_id="paper-1",
            canonical_title="A Scope-Bound Research System",
            authors=["Ada Researcher"],
            abstract="The system binds every evidence lookup to a user-confirmed paper set.",
            pdf_url="https://203.0.113.10/paper.pdf",
            verification_status="provider_verified",
            full_text_status="available",
        ),
    )
    scope = repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-1",
            selected_paper_ids=["paper-1"],
            user_intent="Explain how the scope boundary works.",
        ),
    )
    return repository.transition_scope_status(access, scope.scope_id, "confirmed")


@pytest.mark.asyncio
async def test_user_uploaded_pdf_is_required_and_preserves_user_acl(tmp_path: Path) -> None:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    scope = _seed_scope(repository, access)
    service = PaperIngestionService(
        repository,
        knowledge,
        tmp_path / "artifacts",
        chunking_mode="parent_child",
    )
    uploaded = service.upload_pdf(
        access,
        scope.scope_id,
        "paper-1",
        _pdf_bytes(),
        filename="paper.pdf",
    )
    statuses = await service.ingest_scope(access, scope.scope_id)
    assert uploaded.status == "uploaded"
    assert [status.status for status in statuses] == ["indexed"]
    assert statuses[0].error is None
    assert repository.get_scope(access, scope.scope_id).status == "ready"
    evidence = repository.list_evidence(access, scope.scope_id)
    assert len(evidence) == 1
    assert evidence[0].paper_id == "paper-1"
    owner_chunks = knowledge.visible_chunks(
        AccessContext(tenant_id="tenant-a", user_id="user-a")
    )
    other_chunks = knowledge.visible_chunks(
        AccessContext(tenant_id="tenant-a", user_id="user-b")
    )
    assert len(owner_chunks) == 2
    assert {chunk.metadata.get("retrieval_role") for chunk in owner_chunks} == {
        "parent",
        "child",
    }
    child = next(
        chunk
        for chunk in owner_chunks
        if chunk.metadata.get("retrieval_role") == "child"
    )
    parent = next(
        chunk
        for chunk in owner_chunks
        if chunk.metadata.get("retrieval_role") == "parent"
    )
    assert child.metadata["parent_chunk_id"] == parent.chunk_id
    assert evidence[0].chunk_id == child.chunk_id
    assert other_chunks == ()


@pytest.mark.asyncio
async def test_hybrid_pdf_ingestion_stores_flat_primary_and_child_auxiliary_lanes(
    tmp_path: Path,
) -> None:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    scope = _seed_scope(repository, access)
    service = PaperIngestionService(
        repository,
        knowledge,
        tmp_path / "artifacts",
        chunking_mode="hybrid",
    )
    service.upload_pdf(
        access,
        scope.scope_id,
        "paper-1",
        _pdf_bytes(),
        filename="paper.pdf",
    )

    statuses = await service.ingest_scope(access, scope.scope_id)

    assert statuses[0].status == "indexed"
    chunks = knowledge.visible_chunks(
        AccessContext(tenant_id="tenant-a", user_id="user-a")
    )
    lanes = {
        str(chunk.metadata.get("hybrid_route"))
        for chunk in chunks
        if chunk.metadata.get("hybrid_route")
    }
    assert lanes == {"flat_primary", "child_aux"}
    assert sum(
        chunk.metadata.get("hybrid_route") == "flat_primary" for chunk in chunks
    ) == 1
    assert sum(
        chunk.metadata.get("hybrid_route") == "child_aux"
        and chunk.metadata.get("retrieval_role") == "child"
        for chunk in chunks
    ) == 1


@pytest.mark.asyncio
async def test_bailian_embedding_is_prewarmed_before_pdf_is_indexed(tmp_path: Path) -> None:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    scope = _seed_scope(repository, access)
    prewarmer = _EmbeddingPrewarmer()
    service = PaperIngestionService(
        repository,
        knowledge,
        tmp_path / "artifacts",
        chunking_mode="parent_child",
        embedding_prewarmer=prewarmer,
    )
    service.upload_pdf(
        access,
        scope.scope_id,
        "paper-1",
        _pdf_bytes(),
        filename="paper.pdf",
    )

    statuses = await service.ingest_scope(access, scope.scope_id)

    assert statuses[0].status == "indexed"
    assert len(prewarmer.calls) == 1
    assert len(prewarmer.calls[0]) == 1


@pytest.mark.asyncio
async def test_embedding_prewarmer_failure_keeps_document_unindexed(tmp_path: Path) -> None:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    scope = _seed_scope(repository, access)
    service = PaperIngestionService(
        repository,
        knowledge,
        tmp_path / "artifacts",
        chunking_mode="parent_child",
        embedding_prewarmer=_EmbeddingPrewarmer(fail=True),
    )
    service.upload_pdf(
        access,
        scope.scope_id,
        "paper-1",
        _pdf_bytes(),
        filename="paper.pdf",
    )

    statuses = await service.ingest_scope(access, scope.scope_id)

    assert statuses[0].status == "failed"
    assert statuses[0].error == "embedding prewarm failed (RuntimeError)"
    assert knowledge.visible_chunks(
        AccessContext(tenant_id="tenant-a", user_id="user-a")
    ) == ()


@pytest.mark.asyncio
async def test_safe_downloader_rejects_private_targets_before_request(tmp_path: Path) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, content=b"%PDF-1.4\n%%EOF")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    downloader = SafePDFDownloader(client=client)
    with pytest.raises(PaperDownloadError, match="non-public"):
        await downloader.download("https://127.0.0.1/paper.pdf", tmp_path / "paper.pdf")
    assert requests == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_downloader_streams_bounded_pdf(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4\n%%EOF",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    downloader = SafePDFDownloader(client=client, max_bytes=1_000)
    target = tmp_path / "paper.pdf"
    size = await downloader.download("https://8.8.8.8/paper.pdf", target)
    assert size == target.stat().st_size
    assert target.read_bytes().startswith(b"%PDF-")
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_downloader_rejects_oversized_content_length(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": "1001"},
            content=b"%PDF-1.4\n%%EOF",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    downloader = SafePDFDownloader(client=client, max_bytes=1_000)
    target = tmp_path / "oversized.pdf"
    with pytest.raises(PaperDownloadError, match="exceeds"):
        await downloader.download("https://8.8.8.8/paper.pdf", target)
    assert not target.exists()
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_downloader_rejects_non_pdf_payload(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ignore prior instructions")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    downloader = SafePDFDownloader(client=client, max_bytes=1_000)
    target = tmp_path / "malicious.pdf"
    with pytest.raises(PaperDownloadError, match="did not return a PDF"):
        await downloader.download("https://8.8.8.8/paper.pdf", target)
    assert not target.exists()
    await client.aclose()


@pytest.mark.asyncio
async def test_ingestion_does_not_auto_download_or_fall_back_to_abstract(tmp_path: Path) -> None:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    repository.save_request(
        access,
        LiteratureRequest(request_id="request-oa", query="open paper"),
    )
    repository.upsert_paper(
        access,
        PaperCard(
            paper_id="paper-oa",
            canonical_title="An Open Paper",
            abstract="Fallback abstract.",
            doi="10.1000/open",
            verification_status="provider_verified",
        ),
    )
    scope = repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-oa",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-oa",
            selected_paper_ids=["paper-oa"],
            user_intent="Read the open paper.",
        ),
    )
    repository.transition_scope_status(access, scope.scope_id, "confirmed")
    resolver = _Resolver()
    downloader = _FailingDownloader()
    service = PaperIngestionService(
        repository,
        knowledge,
        tmp_path / "artifacts",
        downloader=downloader,  # type: ignore[arg-type]
        oa_resolver=resolver,
    )

    statuses = await service.ingest_scope(access, scope.scope_id)

    assert resolver.calls == []
    assert statuses[0].status == "failed"
    assert "user-uploaded PDF" in (statuses[0].error or "")
    assert repository.get_paper(access, "paper-oa").pdf_url is None
    assert repository.get_scope(access, scope.scope_id).status == "ingesting"
