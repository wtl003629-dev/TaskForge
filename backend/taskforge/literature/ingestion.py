"""Safe paper acquisition and structure-preserving Scope-bound ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx

from ..document_ingestion import DocumentIngestionError
from ..knowledge import KnowledgeChunk
from ..pdf_parsing.contracts import DocumentBlock
from ..pdf_parsing.hierarchy import (
    HierarchicalUnit,
    build_flat_units,
    build_parent_child_units,
    build_sliding_window_units,
)
from ..pdf_parsing.native_parser import NativePDFParser
from ..pdf_parsing.router import ParserRoutingError, PDFParserRouter
from ..pdf_parsing.structure_policy import build_structure_aware_units
from ..pdf_parsing.visual_evidence import (
    VisualEvidenceExtractor,
    enrich_visual_evidence,
)
from ..rag_experiment_profile import (
    RAGExperimentProfile,
    resolve_rag_experiment_profile,
)
from ..research_protocol import EvidenceCard, IngestionStatus, PaperCard
from .repository import LiteratureAccess, SQLiteLiteratureRepository


class KnowledgeWriter(Protocol):
    def replace_document_version(self, chunks: Sequence[KnowledgeChunk]) -> int: ...


class EmbeddingPrewarmer(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]: ...


class OAPDFResolver(Protocol):
    async def resolve_pdf(self, doi: str) -> str | None: ...

    async def aclose(self) -> None: ...


class PaperDownloadError(RuntimeError):
    pass


_REFERENTIAL_CHILD_START = re.compile(
    r"^(?:(?:this|that|these|those|such|it|they|we|our)\b|"
    r"the\s+(?:method|model|dataset|corpus|baseline|classifier|approach|system|"
    r"framework|experiment|evaluation|result|metric)\b)",
    re.IGNORECASE,
)
_PREVIOUS_CONTEXT_CHARS = 600


def _child_retrieval_text(
    unit: HierarchicalUnit,
    *,
    document_title: str,
    previous_child: HierarchicalUnit | None,
    blocks: Sequence[DocumentBlock] = (),
) -> str:
    """Build deterministic search text without changing citation text.

    Every Parent-Child retrieval unit receives document and section context.
    A bounded previous-child tail is included only for passages whose opening
    contains an obvious backward reference.  The authoritative Child body
    remains ``KnowledgeChunk.text`` and is therefore still the citation unit.
    """

    pieces: list[str] = []
    title = str(document_title).strip()
    if title:
        pieces.append(f"Document: {title}")
    heading = " > ".join(value.strip() for value in unit.heading_path if value.strip())
    if heading:
        pieces.append(f"Section: {heading}")
    content_types = [
        value for value in dict.fromkeys(unit.block_types) if value != "title"
    ]
    if content_types:
        pieces.append(f"Content type: {', '.join(content_types)}")
    captions = [
        block.text.strip()
        for block in blocks
        if block.block_type == "caption" and block.text.strip()
    ]
    if captions:
        pieces.append("Caption: " + " ".join(captions)[:1_000])
    table_headers: list[str] = []
    for block in blocks:
        if block.block_type != "table":
            continue
        rendered = (
            block.text.strip()
            or str(block.structured_content.get("textual_rendering") or "").strip()
        )
        first_line = next(
            (line.strip() for line in rendered.splitlines() if line.strip()), ""
        )
        if first_line:
            table_headers.append(first_line)
    if table_headers:
        pieces.append("Table header: " + " | ".join(table_headers)[:1_000])
    if (
        previous_child is not None
        and previous_child.role == "child"
        and previous_child.parent_id == unit.parent_id
        and _REFERENTIAL_CHILD_START.search(unit.text.lstrip())
    ):
        context = previous_child.text.strip()[-_PREVIOUS_CONTEXT_CHARS:]
        if context:
            pieces.append(f"Previous context:\n{context}")
    if not pieces:
        return unit.text
    pieces.append(f"Content:\n{unit.text}")
    return "\n\n".join(pieces)


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
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
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
                    lambda: socket.getaddrinfo(
                        parsed.hostname, 443, type=socket.SOCK_STREAM
                    ),
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
                        raise PaperDownloadError(
                            "paper download exceeded redirect policy"
                        )
                    current = urljoin(current, location)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise PaperDownloadError("paper download failed") from exc
                content_length = response.headers.get("Content-Length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > self.max_bytes
                ):
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
        parser_router: PDFParserRouter | None = None,
        parent_target_tokens: int = 2_000,
        parent_max_tokens: int = 3_000,
        child_target_tokens: int = 400,
        child_max_tokens: int = 500,
        child_overlap_tokens: int = 60,
        visual_evidence_extractor: VisualEvidenceExtractor | None = None,
        embedding_prewarmer: EmbeddingPrewarmer | None = None,
        # The Agent keeps the historical single-lane modes. ``hybrid`` is an
        # explicit opt-in that stores a Flat lane plus a Child/Parent lane in
        # the same document version; retrieval decides whether to use it.
        chunking_mode: str = "parent_child",
        flat_chunk_chars: int = 2_000,
        flat_overlap_chars: int = 0,
        experiment_profile: RAGExperimentProfile | None = None,
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
        self.parser_router = parser_router or PDFParserRouter(
            NativePDFParser(
                max_bytes=self.max_upload_bytes,
                max_pages=self.max_pages,
            ),
            backend="native",
        )
        self.parent_target_tokens = parent_target_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_target_tokens = child_target_tokens
        self.child_max_tokens = child_max_tokens
        self.child_overlap_tokens = child_overlap_tokens
        self.visual_evidence_extractor = visual_evidence_extractor
        self.embedding_prewarmer = embedding_prewarmer
        if chunking_mode not in {"flat", "parent_child", "hybrid", "sliding"}:
            raise ValueError(
                "PDF chunking mode must be flat, parent_child, hybrid, or sliding"
            )
        if not 256 <= flat_chunk_chars <= 50_000:
            raise ValueError("flat PDF chunk target is outside the supported range")
        if not 0 <= flat_overlap_chars < flat_chunk_chars:
            raise ValueError("flat PDF chunk overlap must be smaller than target")
        self.chunking_mode = chunking_mode
        self.flat_chunk_chars = flat_chunk_chars
        self.flat_overlap_chars = flat_overlap_chars
        self.experiment_profile = experiment_profile or resolve_rag_experiment_profile(
            "current"
        )

    def _target(
        self, access: LiteratureAccess, scope_id: str, version: int, paper_id: str
    ) -> Path:
        tenant = hashlib.sha256(access.tenant_id.encode()).hexdigest()[:16]
        paper = hashlib.sha256(paper_id.encode()).hexdigest()[:24]
        return (
            self.artifact_root
            / "literature"
            / tenant
            / scope_id
            / f"v{version}"
            / f"{paper}.pdf"
        )

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
                rag_profile=str(chunk.metadata.get("rag_profile") or "current"),
                rag_ablation=str(chunk.metadata.get("rag_ablation") or "a"),
                source=chunk.source_uri,
                title=paper.canonical_title,
                section=(
                    str(chunk.metadata.get("heading"))
                    if chunk.metadata.get("heading")
                    else None
                ),
                page=(
                    ",".join(str(item) for item in chunk.metadata.get("pages", []))
                    if chunk.metadata.get("pages")
                    else None
                ),
                evidence_type=str(chunk.metadata.get("kind") or "paragraph"),
                visual_artifact_ids=[
                    str(value)
                    for value in chunk.metadata.get("visual_artifact_ids", [])
                ],
                visual_pending=bool(chunk.metadata.get("visual_pending")),
                snippet=chunk.text[:3_000],
                score=0.0,
                retrieval_sources=["scope_ingestion"],
            )
            for chunk in chunks
            if chunk.metadata.get("retrieval_role") != "parent"
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
        current_document_id = f"research-paper:{scope_id}:{paper.paper_id}"
        stable_document_id = self.experiment_profile.document_id(current_document_id)
        current_knowledge_base_id = f"research-scope:{scope_id}:v{scope_version}"
        knowledge_base_id = self.experiment_profile.knowledge_base_id(
            current_knowledge_base_id
        )
        chunk_identity = (
            f"{access.tenant_id}\0{scope_id}\0{scope_version}\0"
            f"{paper.paper_id}\0abstract"
            if self.experiment_profile.name == "current"
            else f"{access.tenant_id}\0{stable_document_id}\0{scope_version}\0abstract"
        )
        chunk_id = hashlib.sha256(chunk_identity.encode()).hexdigest()[:24]
        evidence_id = f"evidence:{scope_id}:v{scope_version}:{chunk_id}"
        return [
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id=access.tenant_id,
                text=text,
                source_uri=f"paper://{paper.paper_id}",
                document_id=stable_document_id,
                version=str(scope_version),
                version_order=scope_version,
                acl=frozenset({f"user:{access.user_id}"}),
                metadata={
                    "knowledge_base_id": knowledge_base_id,
                    **self.experiment_profile.metadata(),
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

    async def _pdf_chunks(
        self,
        access: LiteratureAccess,
        scope_id: str,
        scope_version: int,
        paper: PaperCard,
        path: Path,
    ) -> list[KnowledgeChunk]:
        source_uri = f"paper://{paper.paper_id}"
        parsed = await self.parser_router.parse(
            path,
            source_uri=source_uri,
        )
        parsed = await enrich_visual_evidence(
            parsed,
            self.visual_evidence_extractor,
        )
        # A hybrid document contains two independent retrieval lanes. The
        # Flat lane is deliberately built from the same parser output and
        # parameters as the legacy control; the Child lane may add structure
        # context, but never replaces the Flat evidence or citation text.
        unit_batches: list[
            tuple[
                str,
                str,
                tuple[HierarchicalUnit, ...],
                dict[str, int | bool] | None,
                str | None,
            ]
        ] = []
        if self.chunking_mode in {"parent_child", "hybrid"}:
            structure_profile: dict[str, int | bool] | None = None
            chunk_policy: str | None = None
            effective_chunking_mode = "parent_child"
            if self.experiment_profile.structure_aware_chunking_enabled:
                result = build_structure_aware_units(
                    parsed,
                    parent_target_tokens=self.parent_target_tokens,
                    parent_max_tokens=self.parent_max_tokens,
                    child_target_tokens=self.child_target_tokens,
                    child_max_tokens=self.child_max_tokens,
                    child_overlap_tokens=self.child_overlap_tokens,
                    fallback_target_chars=self.flat_chunk_chars,
                    fallback_overlap_chars=self.flat_overlap_chars,
                )
                child_units = result.units
                structure_profile = result.profile.as_metadata()
                chunk_policy = result.policy.name
                effective_chunking_mode = "structure_aware"
            else:
                child_units = build_parent_child_units(
                    parsed,
                    parent_target_tokens=self.parent_target_tokens,
                    parent_max_tokens=self.parent_max_tokens,
                    child_target_tokens=self.child_target_tokens,
                    child_max_tokens=self.child_max_tokens,
                    child_overlap_tokens=self.child_overlap_tokens,
                )
            unit_batches.append(
                (
                    "child_aux" if self.chunking_mode == "hybrid" else "single",
                    effective_chunking_mode,
                    child_units,
                    structure_profile,
                    chunk_policy,
                )
            )
        elif self.chunking_mode == "flat":
            flat_units = build_flat_units(
                parsed,
                target_chars=self.flat_chunk_chars,
                overlap_chars=self.flat_overlap_chars,
            )
            unit_batches.append(("single", "flat", flat_units, None, None))
        else:
            sliding_units = build_sliding_window_units(
                parsed,
                window_chars=self.flat_chunk_chars,
                overlap_chars=self.flat_overlap_chars,
            )
            unit_batches.append(("single", "sliding", sliding_units, None, None))
        if self.chunking_mode == "hybrid":
            flat_units = build_flat_units(
                parsed,
                target_chars=self.flat_chunk_chars,
                overlap_chars=self.flat_overlap_chars,
            )
            # Flat is appended first so deterministic card/evidence ordering
            # keeps the legacy lane easy to inspect and compare.
            unit_batches.insert(0, ("flat_primary", "flat", flat_units, None, None))
        current_document_id = f"research-paper:{scope_id}:{paper.paper_id}"
        stable_document_id = self.experiment_profile.document_id(current_document_id)
        current_knowledge_base_id = f"research-scope:{scope_id}:v{scope_version}"
        knowledge_base_id = self.experiment_profile.knowledge_base_id(
            current_knowledge_base_id
        )
        units = tuple(unit for _, _, batch, _, _ in unit_batches for unit in batch)
        stored_ids = {
            unit.unit_id: hashlib.sha256(
                (
                    f"{access.tenant_id}\0{stable_document_id}\0{scope_version}\0"
                    f"{unit.unit_id}"
                ).encode()
            ).hexdigest()[:24]
            for unit in units
        }
        blocks_by_id = {block.block_id: block for block in parsed.blocks}
        units_by_id = {unit.unit_id: unit for unit in units}
        chunks: list[KnowledgeChunk] = []
        for (
            lane,
            effective_chunking_mode,
            batch,
            structure_profile,
            chunk_policy,
        ) in unit_batches:
            for unit in batch:
                chunk_id = stored_ids[unit.unit_id]
                selected_blocks = [
                    blocks_by_id[block_id]
                    for block_id in unit.block_ids
                    if block_id in blocks_by_id
                ]
                provenance = [
                    {
                        "block_id": block.block_id,
                        "page": block.page,
                        "bbox": list(block.bbox),
                        "block_type": block.block_type,
                        "content_hash": block.content_hash,
                        "parser": block.parser,
                        "parser_version": block.parser_version,
                    }
                    for block in selected_blocks
                ]
                kind = (
                    "parent"
                    if unit.role == "parent"
                    else unit.block_types[0]
                    if len(unit.block_types) == 1
                    else "mixed"
                )
                retrieval_text = (
                    _child_retrieval_text(
                        unit,
                        document_title=paper.canonical_title,
                        previous_child=(
                            units_by_id.get(unit.previous_unit_id)
                            if unit.previous_unit_id is not None
                            else None
                        ),
                        blocks=selected_blocks,
                    )
                    if self.experiment_profile.retrieval_text_enabled
                    and unit.role == "child"
                    and lane != "flat_primary"
                    else None
                )
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        tenant_id=access.tenant_id,
                        text=unit.text,
                        source_uri=source_uri,
                        document_id=stable_document_id,
                        version=str(scope_version),
                        version_order=scope_version,
                        acl=frozenset({f"user:{access.user_id}"}),
                        metadata={
                            "knowledge_base_id": knowledge_base_id,
                            **self.experiment_profile.metadata(),
                            "scope_id": scope_id,
                            "scope_version": scope_version,
                            "paper_id": paper.paper_id,
                            "title": paper.canonical_title,
                            "authors": paper.authors,
                            "doi": paper.doi,
                            "retrieval_role": unit.role,
                            "retrieval_text": retrieval_text,
                            "retrieval_text_version": (
                                "parent-child-title-context-v1"
                                if retrieval_text is not None
                                else None
                            ),
                            "chunking_mode": effective_chunking_mode,
                            "hybrid_route": lane
                            if self.chunking_mode == "hybrid"
                            else None,
                            "chunk_policy": chunk_policy,
                            "structure_profile": structure_profile,
                            "flat_chunk_chars": self.flat_chunk_chars,
                            "flat_overlap_chars": self.flat_overlap_chars,
                            "parent_chunk_id": stored_ids[unit.parent_id],
                            "hierarchy_order": unit.order,
                            "chunk_index": unit.order,
                            "heading": " > ".join(unit.heading_path) or None,
                            "heading_path": list(unit.heading_path),
                            "pages": list(unit.pages),
                            "block_ids": list(unit.block_ids),
                            "block_types": list(unit.block_types),
                            "visual_artifact_ids": [
                                block.image_artifact_id
                                for block in selected_blocks
                                if block.image_artifact_id is not None
                            ],
                            "visual_evidence": [
                                block.structured_content["visual_evidence"]
                                for block in selected_blocks
                                if isinstance(
                                    block.structured_content.get("visual_evidence"),
                                    dict,
                                )
                            ],
                            "visual_text_ready": all(
                                block.block_type not in {"image", "chart"}
                                or block.structured_content.get(
                                    "visual_analysis_status"
                                )
                                != "pending"
                                for block in selected_blocks
                            ),
                            "visual_pending": any(
                                block.block_type in {"image", "chart"}
                                and (
                                    block.structured_content.get(
                                        "visual_analysis_status"
                                    )
                                    == "pending"
                                    or (
                                        not block.text.strip()
                                        and not block.structured_content.get(
                                            "textual_rendering"
                                        )
                                    )
                                )
                                for block in selected_blocks
                            ),
                            "kind": kind,
                            "provenance": provenance,
                            "previous_chunk_id": (
                                stored_ids[unit.previous_unit_id]
                                if unit.previous_unit_id is not None
                                else None
                            ),
                            "next_chunk_id": (
                                stored_ids[unit.next_unit_id]
                                if unit.next_unit_id is not None
                                else None
                            ),
                            "oversized_atomic": unit.oversized_atomic,
                            "parser": parsed.parser,
                            "parser_version": parsed.parser_version,
                            "parser_backend": parsed.parser_backend,
                            "parse_quality": parsed.quality.model_dump(mode="json"),
                            "parser_attempts": [
                                attempt.model_dump(mode="json")
                                for attempt in parsed.attempts
                            ],
                            "raw_parse_artifact": parsed.raw_output_artifact,
                            "document_sha256": parsed.sha256,
                            "evidence_id": f"evidence:{scope_id}:v{scope_version}:{chunk_id}",
                            "full_text_available": True,
                        },
                    )
                )
        return chunks

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

        def publish(
            status: str, *, count: int = 0, error: str | None = None
        ) -> IngestionStatus:
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
                candidate_urls: list[str] = []
                if paper.pdf_url:
                    candidate_urls.append(paper.pdf_url)
                if paper.doi and self.oa_resolver is not None:
                    try:
                        resolved = await self.oa_resolver.resolve_pdf(paper.doi)
                    except Exception:
                        resolved = None
                    if resolved and resolved not in candidate_urls:
                        candidate_urls.append(resolved)
                if self.downloader is not None and candidate_urls:
                    publish("fetching")
                    downloaded_from: str | None = None
                    for candidate_url in candidate_urls:
                        try:
                            await self.downloader.download(candidate_url, target)
                        except (PaperDownloadError, OSError):
                            continue
                        downloaded_from = candidate_url
                        break
                    if downloaded_from is not None:
                        paper = paper.model_copy(
                            update={
                                "pdf_url": downloaded_from,
                                "full_text_status": "available",
                            }
                        )
                        self.repository.upsert_paper(access, paper)
                    else:
                        failure = (
                            "开放 PDF 自动下载失败或访问受限；"
                            "请通过论文来源链接自行下载后上传。"
                        )
                else:
                    failure = (
                        "未发现可合法自动下载的开放 PDF；"
                        "请通过论文来源链接自行下载后上传。"
                    )
            if target.is_file():
                publish("parsing")
                try:
                    chunks = await self._pdf_chunks(
                        access,
                        scope.scope_id,
                        scope.scope_version,
                        paper,
                        target,
                    )
                except (DocumentIngestionError, ParserRoutingError, OSError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
        if not chunks:
            failed = publish("failed", error=failure or "paper has no retrievable text")
            self.repository.upsert_paper(
                access,
                paper.model_copy(update={"full_text_status": "failed"}),
            )
            return failed
        if self.embedding_prewarmer is not None:
            searchable = [
                chunk
                for chunk in chunks
                if chunk.metadata.get("retrieval_role") != "parent"
            ]
            texts = [
                str(chunk.metadata.get("retrieval_text") or chunk.text)
                for chunk in searchable
            ]
            try:
                vectors = await asyncio.to_thread(
                    self.embedding_prewarmer.embed_documents,
                    texts,
                )
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        "embedding prewarmer returned an unexpected vector count"
                    )
            except Exception as exc:
                failed = publish(
                    "failed",
                    error=f"embedding prewarm failed ({type(exc).__name__})",
                )
                self.repository.upsert_paper(
                    access,
                    paper.model_copy(update={"full_text_status": "failed"}),
                )
                return failed
        self.knowledge_store.replace_document_version(chunks)
        cards = self._evidence_cards(scope.scope_id, scope.scope_version, paper, chunks)
        self.repository.save_evidence(access, cards)
        full_text = any(
            bool(chunk.metadata.get("full_text_available")) for chunk in chunks
        )
        final_status = "indexed"
        self.repository.upsert_paper(
            access,
            paper.model_copy(
                update={
                    "full_text_status": "ingested" if full_text else "abstract_only"
                }
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
        await self.parser_router.aclose()
        if self.downloader is not None:
            await self.downloader.aclose()
        if self.oa_resolver is not None:
            await self.oa_resolver.aclose()
        close = getattr(self.visual_evidence_extractor, "aclose", None)
        if callable(close):
            await close()


__all__ = [
    "PaperDownloadError",
    "PaperIngestionService",
    "SafePDFDownloader",
]
