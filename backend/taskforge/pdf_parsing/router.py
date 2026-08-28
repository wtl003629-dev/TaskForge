"""Host-configured parser routing with explicit attempts and degradation."""

from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from .contracts import ParsedDocument, ParserAttempt


class PDFParser(Protocol):
    name: str

    async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument: ...


class ParserRoutingError(RuntimeError):
    pass


class PDFParserRouter:
    def __init__(
        self,
        native: PDFParser,
        mineru: PDFParser | None = None,
        *,
        backend: Literal["auto", "native", "mineru"] = "auto",
        mineru_for_structured_documents: bool = True,
    ) -> None:
        if backend == "mineru" and mineru is None:
            raise ValueError("MinerU backend requires a configured MinerU parser")
        self.native = native
        self.mineru = mineru
        self.backend = backend
        self.mineru_for_structured_documents = mineru_for_structured_documents

    @staticmethod
    async def _attempt(
        parser: PDFParser,
        path: Path,
        source_uri: str,
    ) -> tuple[ParsedDocument | None, ParserAttempt]:
        started = perf_counter()
        try:
            result = await parser.parse(path, source_uri=source_uri)
        except Exception as exc:
            return None, ParserAttempt(
                parser=parser.name,
                parser_version="unknown",
                outcome="failed",
                elapsed_ms=(perf_counter() - started) * 1_000,
                error=f"{type(exc).__name__}: {exc}",
            )
        return result, ParserAttempt(
            parser=result.parser,
            parser_version=result.parser_version,
            outcome="accepted",
            elapsed_ms=(perf_counter() - started) * 1_000,
            quality_status=result.quality.status,
        )

    @staticmethod
    def _usable(result: ParsedDocument) -> bool:
        # ``table_failed`` is a partial-content state: one or more table
        # regions could not be structured, but the parser may still have
        # recovered the rest of the paper.  Discarding every text block in
        # that case turns a local table defect into a document-wide outage.
        # Keep the explicit quality status/count so downstream evaluation can
        # report the gap without silently treating the parse as complete.
        return result.quality.status in {
            "ready",
            "degraded",
            "visual_pending",
            "table_failed",
        } and any(block.indexable and block.text.strip() for block in result.blocks)

    @staticmethod
    def _unusable_message(result: ParsedDocument | None, parser_name: str) -> str:
        if result is None:
            return f"{parser_name} PDF parse was not usable"
        reasons = "; ".join(result.quality.reasons) or "no diagnostic reason"
        return (
            f"{parser_name} PDF parse quality was {result.quality.status}: "
            f"{reasons}"
        )

    def _needs_mineru(self, native: ParsedDocument) -> bool:
        if native.quality.status != "ready":
            return True
        return self.mineru_for_structured_documents and any(
            block.block_type in {"table", "chart", "equation", "image"}
            for block in native.blocks
        )

    async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument:
        attempts: list[ParserAttempt] = []
        if self.backend == "native":
            result, attempt = await self._attempt(self.native, path, source_uri)
            attempts.append(attempt)
            if result is None or not self._usable(result):
                raise ParserRoutingError(
                    attempt.error or self._unusable_message(result, "native")
                )
            return result.model_copy(update={"attempts": tuple(attempts)})
        if self.backend == "mineru":
            assert self.mineru is not None
            result, attempt = await self._attempt(self.mineru, path, source_uri)
            attempts.append(attempt)
            if result is None or not self._usable(result):
                raise ParserRoutingError(
                    attempt.error or self._unusable_message(result, "MinerU")
                )
            return result.model_copy(update={"attempts": tuple(attempts)})

        native, native_attempt = await self._attempt(self.native, path, source_uri)
        attempts.append(native_attempt)
        if native is not None and not self._needs_mineru(native) and self._usable(native):
            return native.model_copy(update={"attempts": tuple(attempts)})
        if self.mineru is not None:
            if native is not None:
                attempts[-1] = native_attempt.model_copy(update={"outcome": "rejected"})
            mineru, mineru_attempt = await self._attempt(self.mineru, path, source_uri)
            attempts.append(mineru_attempt)
            if mineru is not None and self._usable(mineru):
                return mineru.model_copy(update={"attempts": tuple(attempts)})
        if native is not None and self._usable(native):
            return native.model_copy(update={"attempts": tuple(attempts)})
        summary = "; ".join(
            f"{attempt.parser}={attempt.error or attempt.quality_status}"
            for attempt in attempts
        )
        raise ParserRoutingError(f"no usable PDF parse: {summary}")

    async def aclose(self) -> None:
        for parser in (self.native, self.mineru):
            close = getattr(parser, "aclose", None)
            if callable(close):
                result = close()
                if isinstance(result, Awaitable):
                    await result


__all__ = ["PDFParser", "PDFParserRouter", "ParserRoutingError"]
