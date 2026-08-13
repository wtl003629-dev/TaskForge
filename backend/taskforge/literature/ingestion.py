"""Safe paper acquisition and structure-preserving Scope-bound ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx

from ..document_ingestion import DocumentIngestionError, extract_pdf_document
from ..ingestion import _pdf_knowledge_chunks
from ..knowledge import KnowledgeChunk
from ..research_protocol import EvidenceCard, IngestionStatus, PaperCard
from .repository import LiteratureAccess, SQLiteLiteratureRepository


class KnowledgeWriter(Protocol):
    def replace_document_version(self, chunks: Sequence[KnowledgeChunk]) -> int: ...


class OAPDFResolver(Protocol):
    async def resolve_pdf(self, doi: str) -> str | None: ...

    async def aclose(self) -> None: ...


class PaperDownloadError(RuntimeError):
    pass


class SafePDFDownloader:
    """Bounded HTTPS downloader that rejects local/private network targets."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        max_bytes: int = 25_000_000,
        timeout_seconds: float = 30.0,
        max_redirects: int = 3,
    ) -> None:
        if not 1 <= max_bytes <= 100_000_000:
            raise ValueError("max_bytes must be between 1 and 100000000")
        if timeout_seconds <= 0 or not 0 <= max_redirects <= 10:
            raise ValueError("invalid download timeout or redirect limit")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "TaskForge/0.1 paper-ingestion"},
        )
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    @staticmethod
    async def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise PaperDownloadError("paper URLs must be unauthenticated HTTPS URLs")
        if parsed.port not in (None, 443):
            raise PaperDownloadError("paper URL uses a disallowed port")
        try:
            direct = ipaddress.ip_address(parsed.hostname)
            addresses = [direct]
        except ValueError:
            loop = asyncio.get_running_loop()
            try:
                values = await loop.run_in_executor(
                    None,
                    lambda: socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM),
                )
            except OSError as exc:
                raise PaperDownloadError("paper host could not be resolved") from exc
            addresses = [ipaddress.ip_address(value[4][0]) for value in values]
        if not addresses or any(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
            for address in addresses
        ):
            raise PaperDownloadError("paper URL resolves to a non-public address")

    async def download(self, url: str, target: Path) -> int:
        current = url
        for redirect in range(self.max_redirects + 1):
            await self._validate_url(current)
            async with self.client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location or redirect >= self.max_redirects:
                        raise PaperDownloadError("paper download exceeded redirect policy")
                    current = urljoin(current, location)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise PaperDownloadError("paper download failed") from exc
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) > self.max_bytes:
                    raise PaperDownloadError("paper PDF exceeds the download limit")
                raw = bytearray()
                async for part in response.aiter_bytes():
                    raw.extend(part)
                    if len(raw) > self.max_bytes:
                        raise PaperDownloadError("paper PDF exceeds the download limit")
                if not raw.startswith(b"%PDF-"):
                    raise PaperDownloadError("paper URL did not return a PDF")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bytes(raw))
                return len(raw)
        raise PaperDownloadError("paper download failed")

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class PaperIngestionService:
    def __init__(
        self,
        repository: SQLiteLiteratureRepository,
        knowledge_store: KnowledgeWriter,
        artifact_root: str | Path,
        *,
        downloader: SafePDFDownloader | None = None,
        oa_resolver: OAPDFResolver | None = None,
        concurrency: int = 3,
        max_pages: int = 300,
        max_upload_bytes: int = 25_000_000,
    ) -> None:
        if concurrency < 1 or concurrency > 10:
            raise ValueError("ingestion concurrency must be between 1 and 10")
        self.repository = repository
        self.knowledge_store = knowledge_store
        self.artifact_root = Path(artifact_root).resolve()
        self.downloader = downloader
        self.oa_resolver = oa_resolver
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_pages = max_pages
        if not 1 <= max_upload_bytes <= 100_000_000:
            raise ValueError("max_upload_bytes must be between 1 and 100000000")
        self.max_upload_bytes = int(max_upload_bytes)

    def _target(self, access: LiteratureAccess, scope_id: str, version: int, paper_id: str) -> Path:
        tenant = hashlib.sha256(access.tenant_id.encode()).hexdigest()[:16]
        paper = hashlib.sha256(paper_id.encode()).hexdigest()[:24]
        return self.artifact_root / "literature" / tenant / scope_id / f"v{version}" / f"{paper}.pdf"

    @staticmethod
    def _evidence_cards(
        scope_id: str,
        scope_version: int,
        paper: PaperCard,
        chunks: Sequence[KnowledgeChunk],
    ) -> list[EvidenceCard]:
        return [
            EvidenceCard(
                evidence_id=str(chunk.metadata["evidence_id"]),
                scope_id=scope_id,
                scope_version=scope_version,
                paper_id=paper.paper_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source_uri,
                title=paper.canonical_title,
                section=(str(chunk.metadata.get("heading")) if chunk.metadata.get("heading") else None),
                page=(
                    ",".join(str(item) for item in chunk.metadata.get("pages", []))
                    if chunk.metadata.get("pages")
                    else None
                ),
                evidence_type=str(chunk.metadata.get("kind") or "paragraph"),
                snippet=chunk.text[:500],
                score=0.0,
                retrieval_sources=["scope_ingestion"],
            )
            for chunk in chunks
        ]

    def _abstract_chunks(
        self,
        access: LiteratureAccess,
        scope_id: str,
        scope_version: int,
        paper: PaperCard,
    ) -> list[KnowledgeChunk]:
        text = paper.abstract.strip()
        if not text:
            return []
        chunk_id = hashlib.sha256(
            f"{access.tenant_id}\0{scope_id}\0{scope_version}\0{paper.paper_id}\0abstract".encode()
        ).hexdigest()[:24]
        evidence_id = f"evidence:{scope_id}:v{scope_version}:{chunk_id}"
        return [
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=access.tenant_id,
                text=text,
                source_uri=f"paper://{paper.paper_id}",
                document_id=f"research-paper:{scope_id}:{paper.paper_id}",
                version=str(scope_version),
                version_order=scope_version,
                acl=frozenset({f"user:{access.user_id}"}),
                metadata={
                    "knowledge_base_id": f"research-scope:{scope_id}:v{scope_version}",
                    "scope_id": scope_id,
                    "scope_version": scope_version,
                    "paper_id": paper.paper_id,
                    "title": paper.canonical_title,
                    "authors": paper.authors,
                    "doi": paper.doi,
                    "kind": "abstract",
                    "heading": "Abstract",
                    "evidence_id": evidence_id,
                    "full_text_available": False,
                },
            )
        ]

    def _pdf_chunks(
        self,
        access: LiteratureAccess,
        scope_id: str,
        scope_version: int,
        paper: PaperCard,
        path: Path,
    ) -> list[KnowledgeChunk]:
        source_uri = f"paper://{paper.paper_id}"
        document = extract_pdf_document(
            path,
            source_uri=source_uri,
            max_bytes=self.max_upload_bytes,
            max_pages=self.max_pages,
            chunk_chars=2_000,
            preserve_page_boundaries=True,
        )
        chunks = _pdf_knowledge_chunks(
            document,
            tenant_id=access.tenant_id,
            knowledge_base_id=f"research-scope:{scope_id}:v{scope_version}",
            document_id=f"research-paper:{scope_id}:{paper.paper_id}",
            version=str(scope_version),
            version_order=scope_version,
            acl=(f"user:{access.user_id}",),
        )
        return [
            replace(
                chunk,
                metadata={
                    **chunk.metadata,
                    "scope_id": scope_id,
                    "scope_version": scope_version,
                    "paper_id": paper.paper_id,
                    "title": paper.canonical_title,
                    "authors": paper.authors,
                    "doi": paper.doi,
                    "evidence_id": f"evidence:{scope_id}:v{scope_version}:{chunk.chunk_id}",
                    "full_text_available": True,
                },
            )
            for chunk in chunks
        ]

    def upload_pdf(
        self,
        access: LiteratureAccess,
        scope_id: str,
        paper_id: str,
        payload: bytes,
        *,
        filename: str | None = None,
    ) -> IngestionStatus:
        """Persist a user-supplied PDF inside an already confirmed Scope."""

        scope = self.repository.get_scope(access, scope_id)
        if scope.status not in {"confirmed", "ingesting"}:
            raise ValueError("scope must be confirmed before PDF upload")
        if paper_id not in scope.selected_paper_ids:
            raise ValueError("paper is outside the selected scope")
        if filename and not filename.casefold().endswith(".pdf"):
            raise ValueError("uploaded paper filename must end with .pdf")
        if not payload or len(payload) > self.max_upload_bytes:
            raise ValueError("uploaded PDF is empty or exceeds the upload limit")
        if not payload.startswith(b"%PDF-"):
            raise ValueError("uploaded paper is not a PDF")
        target = self._target(access, scope.scope_id, scope.scope_version, paper_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4()}.upload")
        try:
            temporary.write_bytes(payload)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        paper = self.repository.get_paper(access, paper_id)
        self.repository.upsert_paper(
            access,
            paper.model_copy(update={"full_text_status": "available"}),
        )
        status = IngestionStatus(
            job_id=f"paper-upload-{uuid4()}",
            scope_id=scope.scope_id,
            paper_id=paper_id,
            status="uploaded",
            evidence_count=0,
        )
        self.repository.save_ingestion_status(
            access,
            status,
            scope_version=scope.scope_version,
        )
        return status

    async def ingest_paper(
        self,
        access: LiteratureAccess,
        scope_id: str,
        paper_id: str,
        *,
        scope_version: int | None = None,
    ) -> IngestionStatus:
        scope = self.repository.get_scope(access, scope_id, version=scope_version)
        if scope.status not in {"confirmed", "ingesting"}:
            raise ValueError("scope must be confirmed before paper ingestion")
        if paper_id not in scope.selected_paper_ids:
            raise ValueError("paper is outside the selected scope")
        paper = self.repository.get_paper(access, paper_id)
        job_id = f"paper-ingestion-{uuid4()}"

        def publish(status: str, *, count: int = 0, error: str | None = None) -> IngestionStatus:
            value = IngestionStatus(
                job_id=job_id,
                scope_id=scope.scope_id,
                paper_id=paper.paper_id,
                status=status,
                evidence_count=count,
                error=error,
            )
            self.repository.save_ingestion_status(
                access,
                value,
                scope_version=scope.scope_version,
            )
            return value

        publish("queued")
        chunks: list[KnowledgeChunk] = []
        failure: str | None = None
        async with self.semaphore:
            target = self._target(
                access,
                scope.scope_id,
                scope.scope_version,
                paper.paper_id,
            )
            if not target.is_file():
                failure = "user-uploaded PDF is required before ingestion"
            else:
                publish("parsing")
                try:
                    chunks = self._pdf_chunks(
                        access,
                        scope.scope_id,
                        scope.scope_version,
                        paper,
                        target,
                    )
                except (DocumentIngestionError, OSError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
        if not chunks:
            failed = publish("failed", error=failure or "paper has no retrievable text")
            self.repository.upsert_paper(
                access,
                paper.model_copy(update={"full_text_status": "failed"}),
            )
            return failed
        self.knowledge_store.replace_document_version(chunks)
        cards = self._evidence_cards(scope.scope_id, scope.scope_version, paper, chunks)
        self.repository.save_evidence(access, cards)
        full_text = any(bool(chunk.metadata.get("full_text_available")) for chunk in chunks)
        final_status = "indexed"
        self.repository.upsert_paper(
            access,
            paper.model_copy(
                update={"full_text_status": "ingested" if full_text else "abstract_only"}
            ),
        )
        return publish(final_status, count=len(cards), error=failure)

    async def ingest_scope(
        self,
        access: LiteratureAccess,
        scope_id: str,
    ) -> list[IngestionStatus]:
        scope = self.repository.get_scope(access, scope_id)
        if scope.status == "confirmed":
            scope = self.repository.transition_scope_status(
                access,
                scope_id,
                "ingesting",
                expected_version=scope.scope_version,
            )
        if scope.status != "ingesting":
            raise ValueError("scope must be confirmed before ingestion")
        results = await asyncio.gather(
            *(
                self.ingest_paper(
                    access,
                    scope.scope_id,
                    paper_id,
                    scope_version=scope.scope_version,
                )
                for paper_id in scope.selected_paper_ids
            )
        )
        if results and all(result.status == "indexed" for result in results):
            self.repository.transition_scope_status(
                access,
                scope.scope_id,
                "ready",
                expected_version=scope.scope_version,
            )
        return list(results)

    async def aclose(self) -> None:
        if self.downloader is not None:
            await self.downloader.aclose()
        if self.oa_resolver is not None:
            await self.oa_resolver.aclose()


__all__ = [
    "PaperDownloadError",
    "PaperIngestionService",
    "SafePDFDownloader",
]
