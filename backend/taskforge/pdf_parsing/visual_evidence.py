"""Optional multimodal sidecar for visuals MinerU cannot structure reliably."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image

from .contracts import DocumentBlock, ParsedDocument, VisualEvidence
from .quality_gate import evaluate_parse_quality


class VisualEvidenceExtractionError(RuntimeError):
    pass


class VisualEvidenceExtractor(Protocol):
    async def extract(self, block: DocumentBlock) -> VisualEvidence: ...


_PROMPT_VERSION = "visual-evidence-v1"
_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class OpenAICompatibleVisualEvidenceExtractor:
    """Call a separately configured vision model and cache structured evidence."""

    name = "openai-compatible-vlm"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        cache_root: Path,
        artifact_root: Path,
        timeout_seconds: float = 120.0,
        concurrency: int = 2,
        max_image_bytes: int = 12_000_000,
        max_pixels: int = 40_000_000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            if not (
                parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError("remote visual extractor endpoints must use HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("visual extractor URL cannot contain credentials or query data")
        if not api_key.strip() or not model.strip():
            raise ValueError("visual extractor requires an API key and exact model ID")
        if not 1 <= concurrency <= 8 or timeout_seconds <= 0:
            raise ValueError("invalid visual extractor timeout or concurrency")
        self._api_key = api_key
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.cache_root = cache_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_image_bytes = max_image_bytes
        self.max_pixels = max_pixels
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False)

    def _image(self, artifact_id: str | None) -> tuple[bytes, str, str]:
        if not artifact_id:
            raise VisualEvidenceExtractionError("visual block has no materialized image")
        path = Path(artifact_id).resolve()
        if not path.is_relative_to(self.artifact_root) or not path.is_file():
            raise VisualEvidenceExtractionError("visual artifact is outside the parser cache")
        body = path.read_bytes()
        if not body or len(body) > self.max_image_bytes:
            raise VisualEvidenceExtractionError("visual artifact exceeds the configured limit")
        try:
            with Image.open(BytesIO(body)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise VisualEvidenceExtractionError("visual artifact is not a supported image") from exc
        mime = _MIME_TYPES.get(image_format)
        if mime is None or width * height > self.max_pixels:
            raise VisualEvidenceExtractionError("visual artifact format or dimensions are unsupported")
        return body, mime, hashlib.sha256(body).hexdigest()

    def _cache_path(self, image_sha256: str) -> Path:
        key = hashlib.sha256(
            f"{self.name}\0{self.model}\0{_PROMPT_VERSION}".encode()
        ).hexdigest()[:16]
        return self.cache_root / f"{image_sha256}.{key}.visual.json"

    @staticmethod
    def _decode_response(raw: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            content = raw["choices"][0]["message"]["content"]
            decoded = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise VisualEvidenceExtractionError("visual extractor returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise VisualEvidenceExtractionError("visual evidence payload must be an object")
        return decoded

    async def extract(self, block: DocumentBlock) -> VisualEvidence:
        body, mime, image_sha256 = self._image(block.image_artifact_id)
        cache_path = self._cache_path(image_sha256)
        async with self._semaphore:
            if cache_path.is_file():
                wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    wrapper.get("schema_version") != "1.0"
                    or wrapper.get("image_sha256") != image_sha256
                    or wrapper.get("model") != self.model
                    or wrapper.get("prompt_version") != _PROMPT_VERSION
                ):
                    raise VisualEvidenceExtractionError("visual cache metadata is inconsistent")
                evidence = VisualEvidence.model_validate(wrapper.get("evidence"))
                return evidence.model_copy(
                    update={
                        "visual_id": f"visual:{block.block_id}",
                        "page": block.page,
                        "bbox": block.bbox,
                        "image_artifact_id": str(block.image_artifact_id),
                    }
                )
            prompt = (
                "Analyze this research-paper visual as evidence. Return JSON only with "
                "exactly: visual_type (chart|diagram|figure|table), caption, axes "
                "(object or null), legends (array), data_points (array of objects), "
                "nodes (array of objects), edges (array of objects), textual_rendering, "
                "confidence (0..1), warnings (array). Preserve visible labels, values, "
                "units, directions and uncertainty. Do not infer unreadable or absent "
                "facts. textual_rendering must be a concise standalone description for "
                "a text-only reasoning model. Existing parser text follows: "
                + block.text[:4_000]
            )
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{base64.b64encode(body).decode()}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                        "max_tokens": 2_000,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                decoded = self._decode_response(response.json())
                evidence = VisualEvidence.model_validate(
                    {
                        **decoded,
                        "visual_id": f"visual:{block.block_id}",
                        "page": block.page,
                        "bbox": block.bbox,
                        "image_artifact_id": str(block.image_artifact_id),
                        "extractor": self.name,
                        "extractor_version": f"{self.model}:{_PROMPT_VERSION}",
                        "image_sha256": image_sha256,
                    }
                )
            except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
                if isinstance(exc, VisualEvidenceExtractionError):
                    raise
                raise VisualEvidenceExtractionError("visual extraction request failed") from exc
            wrapper = {
                "schema_version": "1.0",
                "image_sha256": image_sha256,
                "model": self.model,
                "prompt_version": _PROMPT_VERSION,
                "evidence": evidence.model_dump(mode="json"),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".json.part")
            temporary.write_text(
                json.dumps(wrapper, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
            return evidence

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


async def enrich_visual_evidence(
    document: ParsedDocument,
    extractor: VisualEvidenceExtractor | None,
) -> ParsedDocument:
    """Enrich pending materialized visuals without hiding per-block failures."""

    pending = [
        block
        for block in document.blocks
        if block.block_type in {"image", "chart"}
        and block.structured_content.get("visual_analysis_status") != "ready"
    ]
    if not pending or extractor is None:
        return document
    blocks = list(document.blocks)
    by_id = {block.block_id: index for index, block in enumerate(blocks)}

    async def extract_one(block: DocumentBlock) -> tuple[DocumentBlock, VisualEvidence | Exception]:
        try:
            return block, await extractor.extract(block)
        except Exception as exc:  # preserve a pending visual rather than dropping the PDF
            return block, exc

    for block, result in await asyncio.gather(*(extract_one(block) for block in pending)):
        structured = dict(block.structured_content)
        if isinstance(result, VisualEvidence):
            structured.update(
                {
                    "visual_analysis_status": "ready",
                    "visual_evidence": result.model_dump(mode="json"),
                    "textual_rendering": result.textual_rendering,
                }
            )
            text = "\n\n".join(
                dict.fromkeys(
                    value
                    for value in (block.text.strip(), result.textual_rendering.strip())
                    if value
                )
            )
            confidence = result.confidence
        else:
            structured["visual_analysis_status"] = "pending"
            structured["visual_analysis_warning"] = f"{type(result).__name__}: {result}"[:1_000]
            text = block.text
            confidence = block.confidence
        content_hash = hashlib.sha256(
            json.dumps(
                {"text": text, "structured_content": structured},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        blocks[by_id[block.block_id]] = block.model_copy(
            update={
                "text": text,
                "structured_content": structured,
                "confidence": confidence,
                "content_hash": content_hash,
            }
        )
    quality = evaluate_parse_quality(
        blocks,
        page_count=document.page_count,
        ocr_used=document.quality.ocr_used,
        parser=document.parser,
    )
    return document.model_copy(update={"blocks": tuple(blocks), "quality": quality})


__all__ = [
    "OpenAICompatibleVisualEvidenceExtractor",
    "VisualEvidenceExtractionError",
    "VisualEvidenceExtractor",
    "enrich_visual_evidence",
]
