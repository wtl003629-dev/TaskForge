"""Unpaywall resolver for lawful open-access PDF locations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import quote

from .base import ResilientHTTPProvider

_BASE_URL = "https://api.unpaywall.org/v2"


class UnpaywallResolver(ResilientHTTPProvider):
    name = "unpaywall"

    def __init__(self, *, email: str, **kwargs: object) -> None:
        cleaned = email.strip()
        if "@" not in cleaned or len(cleaned) > 320:
            raise ValueError("Unpaywall requires a valid contact email")
        super().__init__(**kwargs)
        self.email = cleaned

    async def resolve_pdf(self, doi: str) -> str | None:
        normalised = doi.removeprefix("https://doi.org/").strip().lower()
        if not normalised or len(normalised) > 512:
            return None
        payload = await self._get_json(
            f"{_BASE_URL}/{quote(normalised, safe='')}",
            params={"email": self.email},
        )
        if not isinstance(payload, Mapping):
            return None
        locations: list[object] = [payload.get("best_oa_location")]
        raw_locations = payload.get("oa_locations")
        if isinstance(raw_locations, Sequence) and not isinstance(
            raw_locations, (str, bytes)
        ):
            locations.extend(raw_locations)
        for location in locations:
            if not isinstance(location, Mapping):
                continue
            url = str(location.get("url_for_pdf") or "").strip()
            if url.startswith("https://"):
                return url
        return None


__all__ = ["UnpaywallResolver"]
