"""Resilient HTTP and cache primitives for untrusted scholarly APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from ...research_protocol import SearchQuery
from ..models import ProviderPaper


class ProviderError(RuntimeError):
    pass


class ProviderUnavailableError(ProviderError):
    """The upstream is temporarily unavailable; callers should degrade fast."""


class LiteratureProvider(Protocol):
    name: str

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]: ...

    async def get_paper(self, paper_id: str) -> ProviderPaper | None: ...

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]: ...

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]: ...


class ProviderCache(Protocol):
    """Synchronous content-cache port shared by SQLite and PostgreSQL."""

    @staticmethod
    def key(provider: str, url: str, params: Mapping[str, object] | None) -> str: ...

    def get(self, key: str, *, now: float | None = None) -> tuple[int, Any] | None: ...

    def put(
        self,
        key: str,
        provider: str,
        status_code: int,
        payload: Any,
        *,
        now: float | None = None,
    ) -> None: ...


class SQLiteProviderCache:
    """Content cache that never stores authorization headers or API keys."""

    def __init__(self, path: str | Path, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = int(ttl_seconds)
        self.hits = 0
        self.misses = 0
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS literature_provider_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS literature_provider_cache_expiry_idx "
                "ON literature_provider_cache(expires_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def key(provider: str, url: str, params: Mapping[str, object] | None) -> str:
        material = json.dumps(
            [provider, url, sorted((params or {}).items())],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str, *, now: float | None = None) -> tuple[int, Any] | None:
        instant = time.time() if now is None else float(now)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status_code, payload_json FROM literature_provider_cache "
                "WHERE cache_key = ? AND expires_at >= ?",
                (key, instant),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return int(row["status_code"]), json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError("cached provider payload is invalid") from exc

    def put(
        self,
        key: str,
        provider: str,
        status_code: int,
        payload: Any,
        *,
        now: float | None = None,
    ) -> None:
        instant = time.time() if now is None else float(now)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO literature_provider_cache (
                    cache_key, provider, payload_json, status_code, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    status_code = excluded.status_code,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    provider,
                    encoded,
                    int(status_code),
                    instant,
                    instant + self.ttl_seconds,
                ),
            )


class ResilientHTTPProvider:
    name = "provider"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache: ProviderCache | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        concurrency: int = 3,
        min_interval_seconds: float = 0.0,
        rate_limit_failure_threshold: int = 2,
        max_inline_retry_seconds: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if (
            timeout_seconds <= 0
            or max_retries < 0
            or concurrency < 1
            or min_interval_seconds < 0
            or rate_limit_failure_threshold < 1
            or max_inline_retry_seconds < 0
        ):
            raise ValueError("provider timeout/retry/concurrency values are invalid")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": "TaskForge/0.1 literature-research"},
        )
        self.cache = cache
        self.max_retries = int(max_retries)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.min_interval_seconds = float(min_interval_seconds)
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0
        self._circuit_open_until = 0.0
        self._rate_limit_failures = 0
        self.rate_limit_failure_threshold = int(rate_limit_failure_threshold)
        self.max_inline_retry_seconds = float(max_inline_retry_seconds)
        self.headers = dict(headers or {})
        self.request_count = 0

    async def _wait_for_rate_slot(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._rate_lock:
                now = loop.time()
                if self._circuit_open_until > now:
                    remaining = self._circuit_open_until - now
                    raise ProviderUnavailableError(
                        f"{self.name} circuit open for {remaining:.1f}s after rate limiting"
                    )
                target = max(self._next_request_at, self._cooldown_until)
                if target <= now:
                    self._next_request_at = now + self.min_interval_seconds
                    return
                delay = target - now
            await asyncio.sleep(delay)

    async def _defer_requests(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            self._cooldown_until = max(
                self._cooldown_until,
                loop.time() + max(0.0, seconds),
            )

    async def _record_rate_limit(self, delay: float) -> bool:
        """Return true when the provider circuit should fail fast."""

        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            self._rate_limit_failures += 1
            should_open = (
                delay > self.max_inline_retry_seconds
                or self._rate_limit_failures >= self.rate_limit_failure_threshold
            )
            if should_open:
                # Unknown resets use a short probe window; explicit Retry-After
                # values (for example daily OpenAlex budgets) are honoured.
                open_seconds = max(delay, 120.0)
                self._circuit_open_until = max(
                    self._circuit_open_until,
                    loop.time() + open_seconds,
                )
            return should_open

    async def _record_success(self) -> None:
        async with self._rate_lock:
            self._rate_limit_failures = 0
            self._circuit_open_until = 0.0

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After", "").strip()
        if header:
            try:
                return min(86_400.0, max(0.0, float(header)))
            except ValueError:
                try:
                    instant = parsedate_to_datetime(header)
                    if instant.tzinfo is None:
                        instant = instant.replace(tzinfo=UTC)
                    return min(
                        86_400.0,
                        max(0.0, (instant - datetime.now(UTC)).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        if response.status_code == 429:
            return min(120.0, 15.0 * (2**attempt))
        return min(10.0, 1.0 * (2**attempt))

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> Any:
        cache_key = self.cache.key(self.name, url, params) if self.cache else None
        if self.cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                status_code, payload = cached
                if 200 <= status_code < 300:
                    return payload

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    await self._wait_for_rate_slot()
                    self.request_count += 1
                    response = await self.client.get(
                        url,
                        params=params,
                        headers=self.headers,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    delay = self._retry_delay(response, attempt)
                    if response.status_code == 429 and await self._record_rate_limit(delay):
                        raise ProviderUnavailableError(
                            f"{self.name} rate-limit budget exhausted; retry after {delay:.1f}s"
                        )
                    await self._defer_requests(delay)
                    if attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                await self._record_success()
                payload = response.json()
                if self.cache and cache_key:
                    self.cache.put(cache_key, self.name, response.status_code, payload)
                return payload
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(10.0, 1.0 * (2**attempt)))
                    continue
        raise ProviderError(f"{self.name} request failed") from last_error

    async def _get_text(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        cache_key = self.cache.key(self.name, url, params) if self.cache else None
        if self.cache and cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                status_code, payload = cached
                if 200 <= status_code < 300 and isinstance(payload, Mapping):
                    text = payload.get("text")
                    if isinstance(text, str):
                        return text

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    await self._wait_for_rate_slot()
                    self.request_count += 1
                    response = await self.client.get(url, params=params, headers=self.headers)
                if response.status_code == 429 or response.status_code >= 500:
                    delay = self._retry_delay(response, attempt)
                    if response.status_code == 429 and await self._record_rate_limit(delay):
                        raise ProviderUnavailableError(
                            f"{self.name} rate-limit budget exhausted; retry after {delay:.1f}s"
                        )
                    await self._defer_requests(delay)
                    if attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                await self._record_success()
                if self.cache and cache_key:
                    self.cache.put(cache_key, self.name, response.status_code, {"text": response.text})
                return response.text
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(10.0, 1.0 * (2**attempt)))
                    continue
        raise ProviderError(f"{self.name} request failed") from last_error


__all__ = [
    "LiteratureProvider",
    "ProviderCache",
    "ProviderError",
    "ProviderUnavailableError",
    "ResilientHTTPProvider",
    "SQLiteProviderCache",
]
