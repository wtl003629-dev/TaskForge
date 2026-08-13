from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from taskforge.knowledge import InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.literature.evidence import ScopeBoundEvidenceService
from taskforge.literature.repository import LiteratureAccess, SQLiteLiteratureRepository
from taskforge.mcp import MCPServerConfig, MCPStreamableHTTPClient, MCPToolPolicy
from taskforge.research_mcp import ResearchMCPServer, create_mcp_app, run_stdio
from taskforge.research_protocol import (
    EvidenceCard,
    LiteratureRequest,
    PaperCard,
    ResearchScope,
)
from taskforge.research_retrieval import ResearchRetrievalService
from taskforge.tooling import ToolRisk

TOOLS = {
    "literature_search",
    "literature_expand",
    "literature_get",
    "scope_get",
    "paper_search",
    "paper_read",
    "citation_verify",
    "scope_expansion_request",
}


def server(tmp_path: Path) -> ResearchMCPServer:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="p1-c1",
                tenant_id="tenant-a",
                text="Paper one reports recall 0.9.",
                source_uri="paper://paper-one",
                acl=frozenset({"user:agent-1"}),
                metadata={
                    "evidence_id": "p1:1",
                    "title": "Paper One",
                    "paper_id": "paper-one",
                    "knowledge_base_id": "research-scope:scope-1:v1",
                },
            )
        ]
    )
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    access = LiteratureAccess("tenant-a", "agent-1", "conversation-1")
    repository.save_request(
        access,
        LiteratureRequest(request_id="request-1", query="recall"),
    )
    repository.upsert_paper(
        access,
        PaperCard(
            paper_id="paper-one",
            canonical_title="Paper One",
            abstract="Paper one reports recall 0.9.",
            full_text_status="ingested",
        ),
    )
    repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-1",
            selected_paper_ids=["paper-one"],
            user_intent="Find recall.",
        ),
    )
    repository.transition_scope_status(access, "scope-1", "confirmed")
    repository.transition_scope_status(access, "scope-1", "ingesting")
    repository.transition_scope_status(access, "scope-1", "ready")
    repository.save_evidence(
        access,
        [
            EvidenceCard(
                evidence_id="p1:1",
                scope_id="scope-1",
                scope_version=1,
                paper_id="paper-one",
                chunk_id="p1-c1",
                source="paper://paper-one",
                snippet="Paper one reports recall 0.9.",
            )
        ],
    )
    service = ScopeBoundEvidenceService(
        repository,
        ResearchRetrievalService(store, graph_enabled=False),
    )
    return ResearchMCPServer(service, access)


def test_mcp_initialize_list_and_call(tmp_path: Path) -> None:
    value = server(tmp_path)
    initialized = value.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialized["result"]["capabilities"]["tools"] == {}
    listed = value.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert {item["name"] for item in listed["result"]["tools"]} == TOOLS
    called = value.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "paper_search",
                "arguments": {"scope_id": "scope-1", "query": "recall"},
            },
        }
    )
    assert called["result"]["structuredContent"]["evidence"][0]["evidence_id"] == "p1:1"
    paper = value.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "literature_get",
                "arguments": {"paper_id": "paper-one"},
            },
        }
    )
    assert paper["result"]["structuredContent"]["paper_id"] == "paper-one"


def test_mcp_http_and_forged_id_are_safe(tmp_path: Path) -> None:
    client = TestClient(create_mcp_app(server(tmp_path)))
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert response.status_code == 200
    assert len(response.json()["result"]["tools"]) == len(TOOLS)
    forged = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "paper_read",
                "arguments": {"scope_id": "scope-1", "evidence_id": "forged"},
            },
        },
    )
    assert forged.status_code == 200
    assert forged.json()["error"]["code"] == -32004


@pytest.mark.asyncio
async def test_existing_taskforge_mcp_client_interoperates_with_server(tmp_path: Path) -> None:
    app = create_mcp_app(server(tmp_path))
    transport = httpx.ASGITransport(app=app)
    config = MCPServerConfig(
        namespace="paper",
        endpoint="http://127.0.0.1:8765/mcp",
        enabled=True,
        allow_local_http=True,
        allowed_ports=(8765,),
        allowed_tools=tuple(sorted(TOOLS)),
        tool_policies={
            name: MCPToolPolicy(risk=ToolRisk.READ, requires_approval=False)
            for name in TOOLS
        },
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        async with MCPStreamableHTTPClient(
            config,
            client=client,
            resolver=lambda *_: ["127.0.0.1"],
        ) as mcp:
            tools = await mcp.discover_tools()
            assert {tool.remote_name for tool in tools} == TOOLS
            result = await mcp.call_tool(
                "paper_search",
                {"scope_id": "scope-1", "query": "recall"},
            )
            assert result["structuredContent"]["evidence"][0]["evidence_id"] == "p1:1"


def test_mcp_rejects_scope_less_search(tmp_path: Path) -> None:
    result = server(tmp_path).handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "paper_search", "arguments": {"query": "recall"}},
        }
    )
    assert result["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_stdio_transport_dispatches_newline_delimited_json_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n'
    )
    outgoing = StringIO()
    monkeypatch.setattr("taskforge.research_mcp.sys.stdin", incoming)
    monkeypatch.setattr("taskforge.research_mcp.sys.stdout", outgoing)

    await run_stdio(server(tmp_path))

    rows = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert rows[0]["result"]["serverInfo"]["version"] == "0.3.0"
    assert {item["name"] for item in rows[1]["result"]["tools"]} == TOOLS
