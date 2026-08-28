"""Bounded HTTP adapter for a separately deployed MinerU FastAPI service."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .contracts import ParsedDocument
from .mineru_normalizer import normalize_mineru_response
from .quality_gate import ParseQualityPolicy


class MinerUError(RuntimeError):
    pass


class MinerUClient:
    name = "mineru"

    def __init__(
        self,
        base_url: str,
        cache_root: Path,
        *,
        backend: str = "pipeline",
        parse_method: str = "auto",
        effort: str = "high",
        expected_version: str | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        concurrency: int = 2,
        max_pdf_bytes: int = 25_000_000,
        max_pages: int = 300,
        max_response_bytes: int = 50_000_000,
        quality_policy: ParseQualityPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MinerU base URL must be HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("MinerU base URL cannot contain credentials or query data")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("non-loopback MinerU services must use HTTPS")
        if backend.strip() != backend or not backend:
            raise ValueError("MinerU backend must be non-empty and trimmed")
        if parse_method not in {"auto", "txt", "ocr"}:
            raise ValueError("MinerU parse method must be auto, txt, or ocr")
        if effort not in {"medium", "high"}:
            raise ValueError("MinerU effort must be medium or high")
        if not 0 <= max_retries <= 5 or not 1 <= concurrency <= 16:
            raise ValueError("invalid MinerU retry or concurrency limit")
        self.base_url = base_url.rstrip("/")
        self.cache_root = cache_root.resolve()
        self.backend = backend
        self.parse_method = parse_method
        self.effort = effort
        self.expected_version = (
            expected_version.strip() if expected_version is not None else None
        )
        if expected_version is not None and not self.expected_version:
            raise ValueError("expected MinerU version must be non-empty")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_pdf_bytes = max_pdf_bytes
        if not 1 <= max_pages <= 2_000:
            raise ValueError("MinerU page limit must be between 1 and 2000")
        self.max_pages = max_pages
        self.max_response_bytes = max_response_bytes
        self.quality_policy = quality_policy
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )
        self._health: dict[str, Any] | None = None

    async def health(self) -> dict[str, Any]:
        if self._health is not None:
            return dict(self._health)
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise MinerUError("MinerU health response must be an object")
        actual_version = str(
            value.get("version")
            or value.get("mineru_version")
            or value.get("protocol_version")
            or "unknown"
        )
        if self.expected_version is not None and actual_version != self.expected_version:
            raise MinerUError(
                "MinerU version mismatch: "
                f"expected {self.expected_version}, received {actual_version}"
            )
        self._health = value
        return dict(value)

    @staticmethod
    def _page_count(body: bytes) -> int:
        try:
            import pypdf

            reader = pypdf.PdfReader(BytesIO(body), strict=False)
            count = len(reader.pages)
        except Exception as exc:
            raise MinerUError("uploaded PDF page count could not be verified") from exc
        if count < 1:
            raise MinerUError("uploaded PDF contains no pages")
        return count

    def _cache_path(self, sha256: str, *, parse_method: str | None = None) -> Path:
        selected_method = parse_method or self.parse_method
        request_key = hashlib.sha256(
            (
                f"{self.backend}\0{selected_method}\0{self.effort}\0"
                f"{self.expected_version or ''}"
            ).encode()
        ).hexdigest()[:12]
        return self.cache_root / f"{sha256}.{request_key}.mineru.json"

    @staticmethod
    def _has_machine_text(body: bytes) -> bool:
        try:
            import pypdf

            reader = pypdf.PdfReader(BytesIO(body), strict=False)
            # A small threshold avoids treating an image-only scan with a
            # stray invisible watermark as a born-digital document.
            visible = "".join((page.extract_text() or "") for page in reader.pages)
            return len("".join(visible.split())) >= 32
        except Exception:
            return False

    def _materialize_images(
        self,
        raw: dict[str, Any],
        *,
        filename: str,
        sha256: str,
    ) -> dict[str, str]:
        results = raw.get("results")
        payload: object = results.get(filename) if isinstance(results, dict) else raw
        if not isinstance(payload, dict) and isinstance(results, dict):
            values = [value for value in results.values() if isinstance(value, dict)]
            payload = values[0] if len(values) == 1 else {}
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, dict):
            return {}
        artifact_root = self.cache_root / "images" / sha256
        artifacts: dict[str, str] = {}
        total_bytes = 0
        for name, encoded in images.items():
            raw_value: object = encoded
            if isinstance(encoded, dict):
                raw_value = encoded.get("img_base64", encoded.get("base64"))
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            value = raw_value.split(",", 1)[-1] if raw_value.startswith("data:") else raw_value
            try:
                body = base64.b64decode(value, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise MinerUError("MinerU returned an invalid image artifact") from exc
            total_bytes += len(body)
            if total_bytes > self.max_response_bytes:
                raise MinerUError("MinerU image artifacts exceed the configured limit")
            suffix = Path(str(name)).suffix.casefold()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                suffix = ".bin"
            digest = hashlib.sha256(body).hexdigest()
            target = artifact_root / f"{digest}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                temporary = target.with_suffix(f"{suffix}.part")
                temporary.write_bytes(body)
                temporary.replace(target)
            resolved = str(target.resolve())
            normalized_name = str(name).replace("\\", "/")
            basename = normalized_name.rsplit("/", 1)[-1]
            # MinerU 3.4.x returns image payload keys as basenames while its
            # content list commonly refers to the same file as images/foo.png.
            # Keep both stable aliases so a valid visual is never downgraded
            # to visual_pending because of that transport-only discrepancy.
            artifacts[normalized_name] = resolved
            artifacts.setdefault(basename, resolved)
            artifacts.setdefault(f"images/{basename}", resolved)
        return artifacts

    async def _request(
        self,
        filename: str,
        body: bytes,
        *,
        parse_method: str,
    ) -> tuple[dict[str, Any], str]:
        health = await self.health()
        parser_version = str(
            health.get("version")
            or health.get("protocol_version")
            or health.get("mineru_version")
            or "unknown"
        )
        data = {
            "backend": self.backend,
            "parse_method": parse_method,
            "effort": self.effort,
            "formula_enable": "true",
            "table_enable": "true",
            "image_analysis": "true",
            "return_md": "false",
            "return_middle_json": "true",
            "return_model_output": "false",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "false",
            "start_page_id": "0",
            "end_page_id": "99999",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/file_parse",
                    files=[("files", (filename, body, "application/pdf"))],
                    data=data,
                )
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > self.max_response_bytes:
                    raise MinerUError("MinerU response exceeds the configured limit")
                if len(response.content) > self.max_response_bytes:
                    raise MinerUError("MinerU response exceeds the configured limit")
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw, dict):
                    raise MinerUError("MinerU parse response must be an object")
                return raw, parser_version
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise MinerUError(
                        f"MinerU rejected the PDF with HTTP {exc.response.status_code}"
                    ) from exc
                last_error = exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise MinerUError("MinerU returned invalid JSON") from exc
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2**attempt))
        raise MinerUError("MinerU service is unavailable after bounded retries") from last_error

    async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument:
        body = path.read_bytes()
        if not body or len(body) > self.max_pdf_bytes:
            raise MinerUError("PDF is empty or exceeds the MinerU upload limit")
        if not body.startswith(b"%PDF-"):
            raise MinerUError("MinerU input is not a PDF")
        sha256 = hashlib.sha256(body).hexdigest()
        page_count = self._page_count(body)
        if page_count > self.max_pages:
            raise MinerUError(f"PDF exceeds the {self.max_pages} page limit")
        effective_parse_method = (
            "ocr"
            if self.parse_method == "auto" and not self._has_machine_text(body)
            else self.parse_method
        )
        cache_path = self._cache_path(sha256, parse_method=effective_parse_method)
        async with self._semaphore:
            if cache_path.is_file():
                wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    wrapper.get("schema_version") == "1.0"
                    and wrapper.get("sha256") == sha256
                    and wrapper.get("backend") == self.backend
                    and wrapper.get("parse_method") == effective_parse_method
                    and wrapper.get("effort") == self.effort
                    and (
                        self.expected_version is None
                        or wrapper.get("parser_version") == self.expected_version
                    )
                    and isinstance(wrapper.get("response"), dict)
                ):
                    raw = wrapper["response"]
                    parser_version = str(wrapper.get("parser_version") or "unknown")
                else:
                    raise MinerUError("MinerU cache metadata is inconsistent")
            else:
                raw, parser_version = await self._request(
                    path.name,
                    body,
                    parse_method=effective_parse_method,
                )
                wrapper = {
                    "schema_version": "1.0",
                    "sha256": sha256,
                    "backend": self.backend,
                    "parse_method": effective_parse_method,
                    "effort": self.effort,
                    "parser_version": parser_version,
                    "response": raw,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".json.part")
                temporary.write_text(
                    json.dumps(wrapper, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
        return normalize_mineru_response(
            raw,
            filename=path.name,
            source_uri=source_uri,
            sha256=sha256,
            bytes_read=len(body),
            page_count=page_count,
            parser_version=parser_version,
            parser_backend=self.backend,
            effective_parse_method=effective_parse_method,
            raw_output_artifact=str(cache_path),
            image_artifacts=self._materialize_images(
                raw,
                filename=path.name,
                sha256=sha256,
            ),
            quality_policy=self.quality_policy,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["MinerUClient", "MinerUError"]
