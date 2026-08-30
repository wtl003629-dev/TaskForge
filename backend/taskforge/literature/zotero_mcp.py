"""Host-controlled, read-only adapter for a Zotero MCP server.

The MCP client is intentionally injected.  This module never accepts a tool
name from model output: each public method maps to one of a small, fixed set
of allowlisted Zotero operations and passes only structured arguments.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import Field, field_validator

from ..domain import StrictModel


class ZoteroMCPClient(Protocol):
    async def discover_tools(self, **kwargs: Any) -> Any: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class ZoteroMCPError(RuntimeError):
    """Base error at the Zotero MCP trust boundary."""


class ZoteroMCPProtocolError(ZoteroMCPError):
    """Malformed or unsupported tool output."""


class ZoteroMCPToolError(ZoteroMCPError):
    """The remote Zotero tool reported an execution error."""


class ZoteroMCPOutputLimitError(ZoteroMCPProtocolError):
    """Tool output exceeded the host-configured budget."""


class ZoteroItem(StrictModel):
    """The bounded paper-like projection exposed to TaskForge callers."""

    item_key: str = Field(pattern=r"^[A-Z0-9]{8}$")
    title: str = Field(min_length=1, max_length=2_000)
    authors: list[str] = Field(default_factory=list, max_length=256)
    year: int | None = Field(default=None, ge=1000, le=3000)
    doi: str | None = Field(default=None, max_length=512)
    item_type: str = Field(default="", max_length=240)
    abstract: str = Field(default="", max_length=50_000)
    source_url: str | None = Field(default=None, max_length=4_096)
    has_fulltext: bool = False

    @property
    def key(self) -> str:
        """Compatibility view for Zotero's native ``key`` terminology."""

        return self.item_key

    @property
    def has_full_text(self) -> bool:
        return self.has_fulltext

    @field_validator("item_key", "title", mode="before")
    @classmethod
    def clean_required(cls, value: object) -> str:
        return " ".join(str(value).split())

    @field_validator("authors", mode="before")
    @classmethod
    def clean_authors(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result: list[str] = []
            for author in value:
                if isinstance(author, Mapping):
                    first = str(author.get("firstName") or author.get("first_name") or "").strip()
                    last = str(author.get("lastName") or author.get("last_name") or author.get("name") or "").strip()
                    author = " ".join(part for part in (first, last) if part)
                text = " ".join(str(author).split())
                if text:
                    result.append(text)
            return list(dict.fromkeys(result))
        text = str(value).strip()
        return [part.strip() for part in re.split(r"\s*(?:;|\||\band\b)\s*", text) if part.strip()]


ZoteroPaper = ZoteroItem


_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "search": (
        "zotero_search_items",
        "zotero_search",
        "search",
    ),
    "recent": (
        "zotero_get_recent",
        "zotero_recent",
        "zotero_recent_items",
        "recent",
    ),
    "metadata": (
        "zotero_get_item_metadata",
        "zotero_metadata",
        "zotero_get_metadata",
        "zotero_get_item",
        "metadata",
    ),
    "fulltext": (
        "zotero_get_item_fulltext",
        "zotero_fulltext",
        "zotero_get_fulltext",
        "zotero_get_full_text",
        "fulltext",
    ),
    "pages": (
        "zotero_read_pdf_pages",
        "read_pdf_pages",
    ),
}


def _tool_name(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in ("name", "remote_name", "tool_name"):
            name = value.get(key)
            if isinstance(name, str) and name.strip():
                return name.strip()
    for key in ("remote_name", "name", "tool_name"):
        name = getattr(value, key, None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _clean_key(value: object) -> str:
    return str(value or "").strip()


def _clean_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_clean_scalar(item) for item in value if _clean_scalar(item))
    return " ".join(str(value).replace("<br>", " ").split())


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).casefold().replace("_", ""): value for key, value in mapping.items()}
    for name in names:
        if name.casefold().replace("_", "") in lowered:
            return lowered[name.casefold().replace("_", "")]
    return None


def _year(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 3000 else None
    match = re.search(r"\b(1\d{3}|2\d{3}|3000)\b", str(value or ""))
    return int(match.group(1)) if match else None


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "available", "present"}


def _item_from_mapping(value: Mapping[str, Any]) -> ZoteroItem:
    nested = value.get("data")
    if isinstance(nested, Mapping):
        value = {**dict(nested), **{key: item for key, item in value.items() if key != "data"}}
    item_key = _clean_key(_first(value, "item_key", "itemKey", "key", "item id", "itemId"))
    if not item_key:
        raise ZoteroMCPProtocolError("Zotero item is missing its key")
    title = _clean_scalar(_first(value, "title", "name"))
    if not title:
        raise ZoteroMCPProtocolError(f"Zotero item {item_key!r} is missing its title")
    authors = _first(value, "authors", "author", "creators")
    return ZoteroItem(
        item_key=item_key,
        title=title,
        authors=authors,
        year=_year(_first(value, "year", "date", "issued", "publication_date")),
        doi=_clean_scalar(_first(value, "doi", "DOI")) or None,
        item_type=_clean_scalar(_first(value, "item_type", "itemType", "type")),
        abstract=_clean_scalar(_first(value, "abstract", "abstract_note", "abstractNote")),
        source_url=_clean_scalar(_first(value, "source_url", "sourceUrl", "url", "URL")) or None,
        has_fulltext=_bool(_first(value, "has_fulltext", "hasFulltext", "fulltext", "full_text")),
    )


def _strip_markdown(value: str) -> str:
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_`~]", "", value)
    return " ".join(value.split())


def _json_value(text: str) -> Any | None:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _item_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested = _first(value, "items", "results", "data", "records")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return [item for item in nested if isinstance(item, Mapping)]
        if any(_first(value, field) is not None for field in ("key", "item_key", "itemKey", "title")):
            return [value]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _markdown_records(text: str) -> list[dict[str, str]]:
    body = re.split(r"^#{1,6}\s+full\s+text\s*$", text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    lines = [line.rstrip() for line in body.splitlines()]
    records: list[dict[str, str]] = []

    # zotero-mcp commonly renders search results as a Markdown table.
    for index, line in enumerate(lines[:-1]):
        if "|" not in line or not re.search(r"item\s*key|\bkey\b", line, re.IGNORECASE):
            continue
        separator = lines[index + 1]
        if not re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", separator):
            continue
        headers = [_strip_markdown(cell).casefold().replace(" ", "_") for cell in line.strip().strip("|").split("|")]
        for row in lines[index + 2 :]:
            if "|" not in row or not row.strip():
                break
            cells = [_strip_markdown(cell) for cell in row.strip().strip("|").split("|")]
            if len(cells) != len(headers):
                raise ZoteroMCPProtocolError("malformed Zotero Markdown table row")
            records.append(dict(zip(headers, cells, strict=True)))
        if records:
            return records

    current: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        heading = re.match(r"^#{2,6}\s+(?:\d+[.)]\s+)?(.+?)\s*$", line)
        if heading:
            if current and any(
                name in current
                for name in ("key", "item_key", "itemkey", "item_id", "itemid")
            ):
                records.append(current)
                current = {}
            current["title"] = _strip_markdown(heading.group(1))
            continue
        match = re.match(
            r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*)?([^:*|]+?)(?:\*\*)?\s*:\s*(.*?)\s*$",
            line,
        )
        if not match:
            continue
        key = _strip_markdown(match.group(1)).casefold().replace(" ", "_")
        value = _strip_markdown(match.group(2))
        if key in {"key", "item_key", "itemkey", "item_id", "itemid"} and any(
            key_name in current for key_name in ("key", "item_key", "itemkey", "item_id", "itemid")
        ):
            records.append(current)
            current = {}
        current[key] = value
    if current:
        records.append(current)
    return records


def _parse_items(text: str) -> list[ZoteroItem]:
    if re.search(
        r"\bNo (?:items|results) (?:found|in your Zotero library)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return []
    parsed = _json_value(text)
    mappings = _item_mappings(parsed) if parsed is not None else _markdown_records(text)
    if not mappings:
        raise ZoteroMCPProtocolError("Zotero output contains no parseable items")
    items = [_item_from_mapping(mapping) for mapping in mappings]
    unique: dict[str, ZoteroItem] = {}
    for item in items:
        unique.setdefault(item.item_key, item)
    return list(unique.values())


def _parse_status(text: str) -> dict[str, Any]:
    parsed = _json_value(text)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    records = _markdown_records(text)
    if records:
        return dict(records[0])
    cleaned = " ".join(text.split())
    if cleaned:
        return {"status": cleaned}
    raise ZoteroMCPProtocolError("Zotero status output is empty")


def _parse_fulltext(text: str) -> str:
    parsed = _json_value(text)
    if isinstance(parsed, Mapping):
        value = _first(parsed, "fulltext", "full_text", "text", "content")
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ZoteroMCPProtocolError("Zotero fulltext JSON is missing text")
    match = re.search(r"^#{1,6}\s+full\s+text\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        value = text[match.end() :].strip()
        if value:
            return value
        raise ZoteroMCPProtocolError("Zotero Full Text section is empty")
    raise ZoteroMCPToolError("Zotero item has no readable full text attachment")


def _parse_pdf_pages(text: str) -> tuple[int, str]:
    total_match = re.search(
        r"^\*\*Total pages in PDF:\*\*\s*(\d+)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    first_page = re.search(r"^##\s+Page\s+\d+\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if total_match is None or first_page is None:
        raise ZoteroMCPProtocolError("Zotero PDF page output is malformed")
    total = int(total_match.group(1))
    if total < 1:
        raise ZoteroMCPProtocolError("Zotero PDF page count is invalid")
    body = text[first_page.start() :].strip()
    if not body:
        raise ZoteroMCPProtocolError("Zotero PDF page output is empty")
    return total, body


class ZoteroMCPService:
    """Fixed-operation, read-only facade over an injected MCP client."""

    def __init__(
        self,
        client: ZoteroMCPClient,
        *,
        max_output_chars: int = 100_000,
        max_document_chars: int = 5_000_000,
        max_document_pages: int = 300,
        page_batch_size: int = 20,
        tool_names: Mapping[str, str] | None = None,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self.client = client
        self.max_output_chars = max_output_chars
        if not 1 <= max_document_chars <= 10_000_000:
            raise ValueError("max_document_chars is outside the supported range")
        if not 1 <= max_document_pages <= 1_000:
            raise ValueError("max_document_pages is outside the supported range")
        if not 1 <= page_batch_size <= 50:
            raise ValueError("page_batch_size must be between 1 and 50")
        self.max_document_chars = max_document_chars
        self.max_document_pages = max_document_pages
        self.page_batch_size = page_batch_size
        self.tool_names = dict(tool_names or {})
        unknown = set(self.tool_names) - set(_TOOL_ALIASES)
        if unknown:
            raise ValueError(f"unknown Zotero operation: {sorted(unknown)!r}")
        self._available: set[str] | None = None

    async def _discover(self) -> set[str]:
        if self._available is None:
            raw = await _await_if_needed(self.client.discover_tools())
            if isinstance(raw, Mapping):
                raw = raw.get("tools")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ZoteroMCPProtocolError("Zotero tools discovery is malformed")
            names = {_tool_name(item) for item in raw}
            if None in names:
                raise ZoteroMCPProtocolError("Zotero tools discovery contains a malformed tool")
            self._available = {name for name in names if name is not None}
        return self._available

    async def _resolve(self, operation: str) -> str:
        available = await self._discover()
        configured = self.tool_names.get(operation)
        if configured:
            if configured not in available:
                raise ZoteroMCPProtocolError(f"configured Zotero tool is unavailable: {operation}")
            return configured
        for candidate in _TOOL_ALIASES[operation]:
            if candidate in available:
                return candidate
        raise ZoteroMCPProtocolError(f"Zotero operation is unavailable: {operation}")

    async def _call(self, operation: str, arguments: Mapping[str, Any]) -> list[str]:
        name = await self._resolve(operation)
        result = await _await_if_needed(self.client.call_tool(name, dict(arguments)))
        if not isinstance(result, Mapping):
            raise ZoteroMCPProtocolError("Zotero tool result must be an object")
        is_error = result.get("isError", False)
        if not isinstance(is_error, bool):
            raise ZoteroMCPProtocolError("Zotero tool isError must be boolean")
        if is_error or result.get("error") is not None:
            raise ZoteroMCPToolError("Zotero tool reported an error")
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise ZoteroMCPProtocolError("Zotero tool content must be a non-empty array")
        texts: list[str] = []
        total = 0
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise ZoteroMCPProtocolError("Zotero content must contain text blocks only")
            text = item["text"]
            total += len(text)
            if total > self.max_output_chars:
                raise ZoteroMCPOutputLimitError("Zotero tool output exceeded the configured limit")
            texts.append(text)
        joined = "\n".join(texts).lstrip()
        if re.match(
            r"^(?:Error:|Error\s|No PDF attachment|Could not read PDF|"
            r"File download failed|Start page|End page|Requested \d+ pages)",
            joined,
            flags=re.IGNORECASE,
        ):
            raise ZoteroMCPToolError("Zotero tool returned no readable document")
        return texts

    async def status(self) -> dict[str, Any]:
        available = await self._discover()
        required = {
            operation: await self._resolve(operation)
            for operation in ("search", "recent", "metadata", "fulltext")
        }
        # Tool discovery alone only proves the MCP process is alive.  A
        # bounded recent-items call also verifies that it can reach Zotero.
        await self.recent(limit=1)
        return {
            "connected": True,
            "available_tools": sorted(available),
            "operations": required,
        }

    library_status = status

    async def search(self, query: str, limit: int = 20) -> list[ZoteroItem]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        return _parse_items("\n".join(await self._call("search", {"query": query.strip(), "limit": limit})))

    async def recent(self, limit: int = 20) -> list[ZoteroItem]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        return _parse_items("\n".join(await self._call("recent", {"limit": limit})))

    async def metadata(self, item_key: str) -> ZoteroItem:
        key = _clean_key(item_key)
        if not key:
            raise ValueError("item_key must not be empty")
        items = _parse_items(
            "\n".join(
                await self._call(
                    "metadata",
                    {"item_key": key, "include_abstract": True, "format": "json"},
                )
            )
        )
        if len(items) != 1:
            raise ZoteroMCPProtocolError("Zotero metadata must contain exactly one item")
        return items[0]

    get_metadata = metadata
    get_item_metadata = metadata

    async def fulltext(self, item_key: str) -> str:
        key = _clean_key(item_key)
        if not key:
            raise ValueError("item_key must not be empty")
        return _parse_fulltext("\n".join(await self._call("fulltext", {"item_key": key})))

    async def paged_fulltext(self, item_key: str) -> str:
        """Read an entire PDF in bounded page batches with explicit page markers."""

        key = _clean_key(item_key)
        if not key:
            raise ValueError("item_key must not be empty")
        # Read one page first because the tool rejects an end_page beyond the
        # unknown document length; its header then supplies the total.
        first_end = 1
        first = "\n".join(
            await self._call(
                "pages",
                {"item_key": key, "start_page": 1, "end_page": first_end},
            )
        )
        total, body = _parse_pdf_pages(first)
        if total > self.max_document_pages:
            raise ZoteroMCPOutputLimitError(
                "Zotero PDF exceeds the configured page limit"
            )
        parts = [body]
        total_chars = len(body)
        for start in range(2, total + 1, self.page_batch_size):
            end = min(total, start + self.page_batch_size - 1)
            response = "\n".join(
                await self._call(
                    "pages",
                    {"item_key": key, "start_page": start, "end_page": end},
                )
            )
            batch_total, batch = _parse_pdf_pages(response)
            if batch_total != total:
                raise ZoteroMCPProtocolError(
                    "Zotero PDF page count changed during ingestion"
                )
            total_chars += len(batch)
            if total_chars > self.max_document_chars:
                raise ZoteroMCPOutputLimitError(
                    "Zotero PDF text exceeds the configured document limit"
                )
            parts.append(batch)
        return "\n\n".join(parts)

    async def document_text(self, item_key: str) -> str:
        """Prefer complete page-batched PDF text, then use Zotero's fulltext index."""

        available = await self._discover()
        if any(name in available for name in _TOOL_ALIASES["pages"]):
            try:
                return await self.paged_fulltext(item_key)
            except ZoteroMCPToolError:
                pass
        return await self.fulltext(item_key)

    get_fulltext = fulltext
    get_full_text = fulltext
    search_items = search
    recent_items = recent


__all__ = [
    "ZoteroItem",
    "ZoteroMCPClient",
    "ZoteroMCPError",
    "ZoteroMCPOutputLimitError",
    "ZoteroMCPProtocolError",
    "ZoteroMCPService",
    "ZoteroMCPToolError",
    "ZoteroPaper",
]
