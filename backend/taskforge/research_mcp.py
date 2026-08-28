"""Scope-safe MCP facade for TaskForge, Claude Code, and Hermes clients."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .literature.evidence import ScopeBoundEvidenceService
from .literature.repository import LiteratureAccess
from .literature.service import LiteratureDiscoveryService
from .mcp import MCP_PROTOCOL_VERSION
from .research_protocol import (
    EvidenceSearchRequest,
    LiteratureRequest,
    ScopeExpansionRequest,
)

SERVER_INFO = {"name": "taskforge-paper-research", "version": "0.3.0"}

_SCOPE_ID = {"type": "string", "minLength": 1, "maxLength": 240}
_SCOPE_VERSION = {"type": "integer", "minimum": 1}
_EVIDENCE_ID = {"type": "string", "minLength": 1, "maxLength": 1_024}
_INTENTS = [
    "general_fact",
    "method_definition",
    "experimental_setup",
    "numeric_table",
    "cross_paper_comparison",
    "figure_or_layout",
    "claim_verification",
    "related_work",
]

_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "literature_search",
        "description": "Discover candidate papers from multiple scholarly providers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string", "minLength": 1, "maxLength": 240},
                "query": {"type": "string", "minLength": 1, "maxLength": 4_000},
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 16,
                },
                "year_from": {"type": ["integer", "null"], "minimum": 1000, "maximum": 3000},
                "year_to": {"type": ["integer", "null"], "minimum": 1000, "maximum": 3000},
                "required_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 64},
                "excluded_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 64},
                "result_limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "required": ["conversation_id", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "literature_expand",
        "description": "Expand references/citations as candidates without mutating a ResearchScope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "minLength": 1, "maxLength": 240},
                "seed_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 20,
                },
                "include_references": {"type": "boolean", "default": True},
                "include_citations": {"type": "boolean", "default": True},
                "per_seed_limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
                "total_limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 100},
            },
            "required": ["request_id", "seed_paper_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "literature_get",
        "description": "Resolve one tenant-visible canonical PaperCard by Paper ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "minLength": 1, "maxLength": 240}
            },
            "required": ["paper_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scope_get",
        "description": "Read the host-owned paper boundary and lifecycle state.",
        "inputSchema": {
            "type": "object",
            "properties": {"scope_id": _SCOPE_ID, "scope_version": _SCOPE_VERSION},
            "required": ["scope_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "paper_search",
        "description": "Search citation-ready evidence only inside a ready ResearchScope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope_id": _SCOPE_ID,
                "scope_version": _SCOPE_VERSION,
                "query": {"type": "string", "minLength": 1, "maxLength": 4_000},
                "intent": {"type": "string", "enum": _INTENTS, "default": "general_fact"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "candidate_k": {"type": "integer", "minimum": 10, "maximum": 100, "default": 50},
                "mode": {"type": "string", "enum": ["standard", "rigorous"], "default": "standard"},
            },
            "required": ["scope_id", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "paper_read",
        "description": "Read one evidence passage after re-checking its Scope membership.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope_id": _SCOPE_ID,
                "scope_version": _SCOPE_VERSION,
                "evidence_id": _EVIDENCE_ID,
            },
            "required": ["scope_id", "evidence_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "citation_verify",
        "description": "Check evidence identity and lexical support within one Scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope_id": _SCOPE_ID,
                "scope_version": _SCOPE_VERSION,
                "claim": {"type": "string", "minLength": 1, "maxLength": 2_000},
                "evidence_ids": {
                    "type": "array",
                    "items": _EVIDENCE_ID,
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
            "required": ["scope_id", "claim", "evidence_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scope_expansion_request",
        "description": "Request user approval to expand a Scope; never applies expansion directly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope_id": _SCOPE_ID,
                "requested_by": {"type": "string", "enum": ["evaluator", "critic"]},
                "reason": {"type": "string", "minLength": 1, "maxLength": 2_000},
                "proposed_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 100,
                },
            },
            "required": ["scope_id", "requested_by", "reason"],
            "additionalProperties": False,
        },
    },
)


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id: Any, value: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(value)}


class ResearchMCPServer:
    """JSON-RPC dispatcher whose identity and services are bound by the host."""

    def __init__(
        self,
        service: ScopeBoundEvidenceService,
        principal: LiteratureAccess,
        *,
        discovery: LiteratureDiscoveryService | None = None,
    ) -> None:
        self.service = service
        self.principal = principal
        self.discovery = discovery

    async def handle_async(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        request_id = payload.get("id")
        method = payload.get("method")
        if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return _error(request_id, -32600, "invalid request")
        if method == "notifications/initialized" or method.startswith("notifications/"):
            return None
        if method == "initialize":
            return _result(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": [dict(tool) for tool in _TOOLS]})
        if method != "tools/call":
            return _error(request_id, -32601, "method not found")
        params = payload.get("params")
        if not isinstance(params, Mapping) or not isinstance(params.get("name"), str):
            return _error(request_id, -32602, "invalid tools/call parameters")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, Mapping):
            return _error(request_id, -32602, "tool arguments must be an object")
        try:
            value = await self._call(str(params["name"]), arguments)
        except KeyError:
            return _error(request_id, -32004, "research resource not found")
        except (TypeError, ValueError):
            return _error(request_id, -32602, "invalid tool arguments")
        except Exception:
            return _error(request_id, -32000, "tool execution failed")
        encoded = value.model_dump(mode="json")
        return _result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(encoded, ensure_ascii=False)}],
                "structuredContent": encoded,
                "isError": False,
            },
        )

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Synchronous compatibility entry point for stdlib clients and tests."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.handle_async(payload))
        raise RuntimeError("use handle_async from an active event loop")

    async def _call(self, name: str, arguments: Mapping[str, Any]):
        values = dict(arguments)
        if name == "literature_search":
            if self.discovery is None:
                raise ValueError("literature discovery is not configured")
            conversation_id = str(values.pop("conversation_id"))
            access = LiteratureAccess(
                self.principal.tenant_id,
                self.principal.user_id,
                conversation_id,
            )
            return await self.discovery.discover(
                access,
                LiteratureRequest.model_validate(values),
            )
        if name == "literature_expand":
            if self.discovery is None:
                raise ValueError("literature discovery is not configured")
            seed_ids = values.get("seed_paper_ids")
            if isinstance(seed_ids, (str, bytes)) or not isinstance(seed_ids, Sequence):
                raise ValueError("seed_paper_ids must be an array")
            return await self.discovery.expand_citations(
                self.principal,
                str(values["request_id"]),
                [str(item) for item in seed_ids],
                include_references=bool(values.get("include_references", True)),
                include_citations=bool(values.get("include_citations", True)),
                per_seed_limit=int(values.get("per_seed_limit", 20)),
                total_limit=int(values.get("total_limit", 100)),
            )
        if name == "literature_get":
            return self.service.repository.get_paper(
                self.principal,
                str(values["paper_id"]),
            )
        if name == "scope_get":
            return self.service.repository.get_scope(
                self.principal,
                str(values["scope_id"]),
                version=(int(values["scope_version"]) if values.get("scope_version") else None),
            )
        if name == "paper_search":
            return await self.service.search(
                self.principal,
                EvidenceSearchRequest.model_validate(values),
            )
        if name == "paper_read":
            return self.service.read_evidence(
                self.principal,
                str(values["scope_id"]),
                str(values["evidence_id"]),
                scope_version=(
                    int(values["scope_version"]) if values.get("scope_version") else None
                ),
            )
        if name == "citation_verify":
            evidence_ids = values.get("evidence_ids")
            if isinstance(evidence_ids, (str, bytes)) or not isinstance(evidence_ids, Sequence):
                raise ValueError("evidence_ids must be an array")
            return self.service.verify_citation(
                self.principal,
                str(values["scope_id"]),
                str(values["claim"]),
                [str(item) for item in evidence_ids],
                scope_version=(
                    int(values["scope_version"]) if values.get("scope_version") else None
                ),
            )
        if name == "scope_expansion_request":
            scope = self.service.repository.get_scope(
                self.principal,
                str(values["scope_id"]),
            )
            request = self.service.repository.request_expansion(
                self.principal,
                ScopeExpansionRequest.model_validate(values),
            )
            self.service.repository.transition_scope_status(
                self.principal,
                scope.scope_id,
                "expansion_requested",
                expected_version=scope.scope_version,
            )
            return request
        raise ValueError("unknown tool")


def create_mcp_app(server: ResearchMCPServer) -> FastAPI:
    app = FastAPI(title=SERVER_INFO["name"], version=SERVER_INFO["version"])

    @app.post("/mcp")
    async def mcp(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(_error(None, -32700, "invalid JSON"), status_code=400)
        if not isinstance(payload, Mapping):
            return JSONResponse(_error(None, -32600, "invalid request"), status_code=400)
        response = await server.handle_async(payload)
        if response is None:
            return JSONResponse({}, status_code=202)
        return JSONResponse(response)

    return app


async def run_stdio(server: ResearchMCPServer) -> None:
    """Serve newline-delimited JSON-RPC for local MCP clients."""

    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        try:
            payload = json.loads(line)
            response = (
                await server.handle_async(payload)
                if isinstance(payload, Mapping)
                else _error(None, -32600, "invalid request")
            )
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_error(None, -32700, "invalid JSON")) + "\n")
            sys.stdout.flush()


__all__ = ["ResearchMCPServer", "create_mcp_app", "run_stdio"]
