from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from taskforge.app import create_app
from taskforge.config import Settings
from taskforge.literature.models import ProviderPaper
from taskforge.literature.service import LiteratureDiscoveryService
from taskforge.orchestration import OrchestrationAccess
from taskforge.research_protocol import SearchQuery


class _Provider:
    name = "semantic_scholar"
    request_count = 0
    cache = None

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        self.request_count += 1
        return [
            ProviderPaper(
                provider="semantic_scholar",
                provider_id="s2-selected",
                semantic_scholar_id="s2-selected",
                title="Scope-Bound Evidence Retrieval",
                authors=["Ada Researcher"],
                abstract=(
                    "A host-confirmed scope boundary limits evidence retrieval to "
                    "the papers selected by the user."
                ),
                year=2025,
                source_url="https://example.test/selected",
                query_id=query.query_id,
                provider_rank=1,
            ),
            ProviderPaper(
                provider="semantic_scholar",
                provider_id="s2-expansion",
                semantic_scholar_id="s2-expansion",
                title="Auditable Research Agent Expansion",
                authors=["Grace Reviewer"],
                abstract="Expansion requires an explicit user decision.",
                year=2026,
                source_url="https://example.test/expansion",
                query_id=query.query_id,
                provider_rank=2,
            ),
        ]

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        return None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []


def _pdf_bytes() -> bytes:
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    document.drawString(72, 720, "Scope-bound evidence from a user-uploaded PDF.")
    document.save()
    return buffer.getvalue()


def _settings(tmp_path: Path) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("TaskForge", encoding="utf-8")
    state = tmp_path / "state"
    return Settings(
        _env_file=None,
        sqlite_path=state / "taskforge.sqlite3",
        context_sqlite_path=state / "context.sqlite3",
        operations_sqlite_path=state / "operations.sqlite3",
        orchestration_sqlite_path=state / "orchestration.sqlite3",
        review_case_sqlite_path=state / "review.sqlite3",
        verification_sqlite_path=state / "verification.sqlite3",
        literature_sqlite_path=state / "literature.sqlite3",
        literature_cache_path=state / "literature-cache.sqlite3",
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
        retrieval_routing="lexical",
        general_text_backend="bm25",
        research_reranker_model=None,
        provider="demo",
    )


def test_direct_pdf_upload_creates_scope_without_discovery(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    auth_headers = {
        "X-TaskForge-Tenant": "tenant-a",
        "X-TaskForge-User": "alice",
    }
    upload_headers = {
        **auth_headers,
        "Content-Type": "application/pdf",
        "X-Filename": "direct-upload.pdf",
    }
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/research/uploads",
            headers=upload_headers,
            params={
                "conversation_id": "direct-upload-conversation",
                "user_intent": "Find evidence inside my uploaded paper.",
                "title": "My Uploaded Research Paper",
            },
            content=_pdf_bytes(),
        )
        assert uploaded.status_code == 201, uploaded.text
        result = uploaded.json()
        assert result["paper"]["canonical_title"] == "My Uploaded Research Paper"
        assert result["paper"]["full_text_status"] == "available"
        assert result["upload"]["status"] == "uploaded"
        assert result["scope"]["status"] == "confirmed"
        assert result["scope"]["selected_paper_ids"] == [result["paper"]["paper_id"]]

        scope_id = result["scope"]["scope_id"]
        ingested = client.post(
            f"/api/research/scopes/{scope_id}/ingest",
            headers=auth_headers,
        )
        assert ingested.status_code == 200, ingested.text
        assert ingested.json()[0]["status"] == "indexed"

        evidence = client.post(
            "/api/research/evidence/search",
            headers=auth_headers,
            json={
                "scope_id": scope_id,
                "query": "Where does scope-bound evidence come from?",
                "top_k": 5,
                "candidate_k": 10,
            },
        )
        assert evidence.status_code == 200, evidence.text
        assert evidence.json()["evidence"]


def test_literature_selection_ingestion_and_bounded_evidence_api(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    intervals = {
        provider.name: provider.min_interval_seconds
        for provider in app.state.container.literature_discovery.providers
    }
    assert intervals == {
        "semantic_scholar": 1.0,
        "openalex": 1.05,
        "arxiv": 3.1,
        "crossref": 0.21,
    }
    research_profile = app.state.container.profiles["research-agent"]
    tool_names = {
        item["name"]
        for item in app.state.container.runtime.registry.list_specs(research_profile)
    }
    assert {"literature_search", "literature_expand", "literature_get"} <= tool_names
    headers = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "alice"}
    with TestClient(app) as client:
        repository = app.state.container.literature_repository
        app.state.container.literature_discovery = LiteratureDiscoveryService(
            repository,
            [_Provider()],
        )
        search = client.post(
            "/api/literature/search",
            headers=headers,
            json={
                "conversation_id": "conversation-1",
                "request": {
                    "request_id": "request-1",
                    "query": "scope bound evidence retrieval",
                    "result_limit": 10,
                },
            },
        )
        assert search.status_code == 201, search.text
        papers = search.json()["papers"]
        assert "abstract" not in papers[0]
        assert papers[0]["short_description"]
        selected = papers[0]["paper_id"]
        expansion = papers[1]["paper_id"]
        stored_request = client.get(
            "/api/literature/requests/request-1",
            headers=headers,
        )
        assert stored_request.status_code == 200
        assert stored_request.json()["query"] == "scope bound evidence retrieval"
        stored_paper = client.get(
            f"/api/literature/papers/{selected}",
            headers=headers,
        )
        assert stored_paper.status_code == 200
        assert stored_paper.json()["paper_id"] == selected
        scope_response = client.post(
            "/api/research/scopes",
            headers=headers,
            json={
                "request_id": "request-1",
                "conversation_id": "conversation-1",
                "selected_paper_ids": [selected],
                "excluded_paper_ids": [expansion],
                "user_intent": "Explain the host-confirmed scope boundary.",
                "allowed_expansion": True,
                "confirm": True,
            },
        )
        assert scope_response.status_code == 201, scope_response.text
        scope = scope_response.json()
        missing_upload = client.post(
            f"/api/research/scopes/{scope['scope_id']}/ingest",
            headers=headers,
        )
        assert missing_upload.status_code == 200
        assert missing_upload.json()[0]["status"] == "failed"
        upload = client.put(
            f"/api/research/scopes/{scope['scope_id']}/papers/{selected}/pdf",
            headers={
                **headers,
                "Content-Type": "application/pdf",
                "X-Filename": "selected.pdf",
            },
            content=_pdf_bytes(),
        )
        assert upload.status_code == 201, upload.text
        assert upload.json()["status"] == "uploaded"
        ingest = client.post(
            f"/api/research/scopes/{scope['scope_id']}/ingest",
            headers=headers,
        )
        assert ingest.status_code == 200, ingest.text
        assert ingest.json()[0]["status"] == "indexed"
        evidence = client.post(
            "/api/research/evidence/search",
            headers=headers,
            json={
                "scope_id": scope["scope_id"],
                "query": "What limits evidence retrieval to selected papers?",
                "top_k": 5,
                "candidate_k": 10,
            },
        )
        assert evidence.status_code == 200, evidence.text
        result = evidence.json()
        assert result["evidence"]
        assert {item["paper_id"] for item in result["evidence"]} == {selected}
        assert result["scope_version"] == 1

        agent_run = client.post(
            f"/api/research/scopes/{scope['scope_id']}/agent-run",
            headers={**headers, "Idempotency-Key": "bounded-research-run-1"},
            json={
                "title": "Scope-bound evidence survey",
                "context": "Use only the papers already selected by the user.",
                "survey_depth": "rigorous",
            },
        )
        assert agent_run.status_code == 201, agent_run.text
        case_id = agent_run.json()["case"]["case_id"]
        finished = client.post(
            f"/api/review-cases/{case_id}/run-until-review",
            headers=headers,
            json={"max_iterations": 4},
        )
        assert finished.status_code == 200, finished.text
        assert finished.json()["case"]["status"] == "waiting_human_review", finished.text
        access = OrchestrationAccess(
            tenant_id="tenant-a",
            user_id="alice",
            conversation_id=case_id,
        )
        plan_id = finished.json()["plan"]["plan_id"]
        role_runs = app.state.container.orchestration_store.list_role_runs(access, plan_id)
        assert [run.role_id for run in role_runs] == [
            "retrieval_planner",
            "source_evaluator",
            "synthesis_writer",
            "critical_reviewer",
        ]
        assert [run.output["blackboard"]["research_payload"]["protocol"] for run in role_runs] == [
            "research.planner_handoff.v1",
            "research.evaluator_handoff.v1",
            "research.writer_handoff.v1",
            "research.critic_handoff.v1",
        ]
        writer_task = app.state.container.store.load_task(
            f"case-task:{role_runs[2].role_run_id}"
        )
        critic_task = app.state.container.store.load_task(
            f"case-task:{role_runs[3].role_run_id}"
        )
        assert writer_task.metadata["research_scope_id"] == scope["scope_id"]
        writer_delta = writer_task.metadata["case_context"]["dependency_results"][0][
            "blackboard_delta"
        ]
        critic_delta = critic_task.metadata["case_context"]["dependency_results"][0][
            "blackboard_delta"
        ]
        assert set(writer_delta) >= {"evidence_ledger", "evidence_cards"}
        assert "research_plan" not in writer_delta
        assert set(critic_delta) >= {"draft", "claim_manifest"}
        assert "evidence_cards" not in critic_delta

        stranger = client.get(
            f"/api/research/scopes/{scope['scope_id']}",
            headers={**headers, "X-TaskForge-User": "mallory"},
        )
        assert stranger.status_code == 404

        requested = client.post(
            f"/api/research/scopes/{scope['scope_id']}/expansion-requests",
            headers=headers,
            json={
                "requested_by": "evaluator",
                "reason": "The selected paper does not cover audit evaluation.",
                "proposed_paper_ids": [expansion],
            },
        )
        assert requested.status_code == 201, requested.text
        expanded = client.post(
            (
                f"/api/research/scopes/{scope['scope_id']}/expansion-requests/"
                f"{requested.json()['expansion_id']}/decision"
            ),
            headers=headers,
            json={"approve": True},
        )
        assert expanded.status_code == 200, expanded.text
        assert expanded.json()["scope_version"] == 2
        assert set(expanded.json()["selected_paper_ids"]) == {selected, expansion}
        assert expanded.json()["status"] == "confirmed"


def test_draft_scope_has_explicit_idempotent_host_confirmation(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    headers = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "alice"}
    with TestClient(app) as client:
        app.state.container.literature_discovery = LiteratureDiscoveryService(
            app.state.container.literature_repository,
            [_Provider()],
        )
        search = client.post(
            "/api/literature/search",
            headers=headers,
            json={
                "conversation_id": "confirm-conversation",
                "request": {
                    "request_id": "confirm-request",
                    "query": "scope bound evidence retrieval",
                },
            },
        ).json()
        scope = client.post(
            "/api/research/scopes",
            headers=headers,
            json={
                "request_id": "confirm-request",
                "conversation_id": "confirm-conversation",
                "selected_paper_ids": [search["papers"][0]["paper_id"]],
                "user_intent": "Confirm this paper boundary.",
            },
        ).json()
        assert scope["status"] == "draft"
        endpoint = f"/api/research/scopes/{scope['scope_id']}/confirm?expected_version=1"
        confirmed = client.post(endpoint, headers=headers)
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "confirmed"
        assert client.post(endpoint, headers=headers).json()["status"] == "confirmed"
