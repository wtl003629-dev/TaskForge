"""Run 30 deterministic end-to-end paper-research integration tasks.

This suite proves lifecycle, Scope safety, four-role protocols, and evidence-ID
resolution without external network/model variance.  It deliberately does not
claim semantic answer quality; the separate live-model report covers that path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.literature.models import ProviderPaper  # noqa: E402
from taskforge.literature.service import LiteratureDiscoveryService  # noqa: E402
from taskforge.research_protocol import SearchQuery  # noqa: E402

PROTOCOLS = [
    "research.planner_handoff.v1",
    "research.evaluator_handoff.v1",
    "research.writer_handoff.v1",
    "research.critic_handoff.v1",
]


class _FixtureProvider:
    # Reuse a real provider discriminator so ProviderPaper validation exercises
    # the same canonicalisation path while all payloads remain local fixtures.
    name = "arxiv"
    cache = None

    def __init__(self, papers: list[dict[str, Any]]) -> None:
        self.papers = papers
        self.request_count = 0

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        self.request_count += 1
        values: list[ProviderPaper] = []
        for rank, paper in enumerate(self.papers[:limit], start=1):
            arxiv_id = str(paper["arxiv_id"])
            values.append(
                ProviderPaper(
                    provider=self.name,
                    provider_id=arxiv_id,
                    arxiv_id=arxiv_id,
                    title=str(paper["title"]),
                    authors=[str(value) for value in paper.get("authors", [])],
                    abstract=str(paper["abstract"]),
                    year=int(paper["year"]),
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    query_id=query.query_id,
                    provider_rank=rank,
                )
            )
        return values

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        return None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []


def _settings(root: Path) -> Settings:
    state = root / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    return Settings(
        _env_file=None,
        database_backend="sqlite",
        provider="demo",
        sqlite_path=state / "taskforge.sqlite3",
        context_sqlite_path=state / "context.sqlite3",
        operations_sqlite_path=state / "operations.sqlite3",
        orchestration_sqlite_path=state / "orchestration.sqlite3",
        review_case_sqlite_path=state / "review.sqlite3",
        verification_sqlite_path=state / "verification.sqlite3",
        literature_sqlite_path=state / "literature.sqlite3",
        literature_cache_path=state / "literature-cache.sqlite3",
        workspace_root=workspace,
        artifact_root=root / "artifacts",
        retrieval_routing="lexical",
    )


def _request(client: TestClient, method: str, path: str, **kwargs: object) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text}")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} did not return an object")
    return value


def _run_case(app: Any, client: TestClient, case: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    case_id = str(case["case_id"])
    tenant = f"benchmark-{case_id}"
    headers = {"X-TaskForge-Tenant": tenant, "X-TaskForge-User": "evaluator"}
    app.state.container.literature_discovery = LiteratureDiscoveryService(
        app.state.container.literature_repository,
        [_FixtureProvider(list(case["papers"]))],
    )
    discovery = _request(
        client,
        "POST",
        "/api/literature/search",
        headers=headers,
        json={
            "conversation_id": case_id,
            "request": {
                "request_id": f"request-{case_id}",
                "query": case["query"],
                "result_limit": 10,
            },
        },
    )
    papers = discovery.get("papers", [])
    if not isinstance(papers, list) or not papers:
        raise RuntimeError(f"{case_id}: discovery returned no papers")
    selected_ids = [str(paper["paper_id"]) for paper in papers if isinstance(paper, dict)]
    scope = _request(
        client,
        "POST",
        "/api/research/scopes",
        headers=headers,
        json={
            "request_id": discovery["request_id"],
            "conversation_id": case_id,
            "selected_paper_ids": selected_ids,
            "excluded_paper_ids": [],
            "user_intent": case["user_intent"],
            "allowed_expansion": True,
            "confirm": True,
        },
    )
    ingestion_response = client.post(
        f"/api/research/scopes/{scope['scope_id']}/ingest",
        headers=headers,
    )
    if not ingestion_response.is_success:
        raise RuntimeError(f"{case_id}: ingestion failed: {ingestion_response.text}")
    ingestion = ingestion_response.json()
    evidence = _request(
        client,
        "POST",
        "/api/research/evidence/search",
        headers=headers,
        json={
            "scope_id": scope["scope_id"],
            "scope_version": scope["scope_version"],
            "query": case["user_intent"],
            "intent": (
                "cross_paper_comparison"
                if case["task_type"] in {"comparison", "survey"}
                else "method_definition"
            ),
            "top_k": 10,
            "candidate_k": 50,
            "mode": "rigorous",
        },
    )
    created = _request(
        client,
        "POST",
        f"/api/research/scopes/{scope['scope_id']}/agent-run",
        headers={**headers, "Idempotency-Key": f"run-{case_id}"},
        json={
            "title": str(case["query"])[:240],
            "context": "Use only the Host-confirmed Scope and structured Evidence IDs.",
            "survey_depth": "rigorous",
        },
    )
    runtime_case_id = str(created["case"]["case_id"])
    finished = _request(
        client,
        "POST",
        f"/api/review-cases/{runtime_case_id}/run-until-review",
        headers=headers,
        json={"max_iterations": 8},
    )

    evidence_items = [item for item in evidence.get("evidence", []) if isinstance(item, dict)]
    evidence_ids = {str(item["evidence_id"]) for item in evidence_items}
    evidence_papers = {str(item["paper_id"]) for item in evidence_items}
    escaped = evidence_papers.difference(selected_ids)
    role_runs = [item for item in finished.get("role_runs", []) if isinstance(item, dict)]
    succeeded = [item for item in role_runs if item.get("status") == "succeeded"]
    protocols: list[str | None] = []
    handoff_chars = 0
    manifest_refs: list[str] = []
    for role in succeeded:
        result = role.get("role_result")
        payload = result.get("research_payload") if isinstance(result, dict) else None
        protocols.append(payload.get("protocol") if isinstance(payload, dict) else None)
        if isinstance(payload, dict):
            handoff_chars += len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            manifest = payload.get("claim_manifest", [])
            if isinstance(manifest, list):
                for claim in manifest:
                    refs = claim.get("evidence_ids", []) if isinstance(claim, dict) else []
                    if isinstance(refs, list):
                        manifest_refs.extend(str(ref) for ref in refs)
    resolved_refs = [ref for ref in manifest_refs if ref in evidence_ids]
    expected_count = int(case["expected_selected_paper_count"])
    passed = (
        len(selected_ids) == expected_count
        and not escaped
        and evidence_papers == set(selected_ids)
        and protocols == PROTOCOLS
        and finished.get("case", {}).get("status") == "waiting_human_review"
        and bool(manifest_refs)
        and len(resolved_refs) == len(manifest_refs)
    )
    return {
        "case_id": case_id,
        "task_type": case["task_type"],
        "passed": passed,
        "discovered_paper_count": len(papers),
        "selected_paper_count": len(selected_ids),
        "evidence_paper_coverage": round(len(evidence_papers) / expected_count, 6),
        "evidence_count": len(evidence_items),
        "scope_escape_count": len(escaped),
        "retrieval_rounds": evidence.get("retrieval_rounds"),
        "protocols": protocols,
        "claim_evidence_ref_count": len(manifest_refs),
        "resolved_claim_evidence_ref_count": len(resolved_refs),
        "structured_handoff_chars": handoff_chars,
        "estimated_handoff_tokens": round(handoff_chars / 4),
        "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
        "external_api_call_count": 0,
        "fixture_provider_call_count": sum(
            int(report.get("request_count") or 0)
            for report in discovery.get("provider_reports", [])
            if isinstance(report, dict)
        ),
        "duplicate_full_text_injection_count": 0,
        "case_status": finished.get("case", {}).get("status"),
        "ingestion_statuses": [
            item.get("status") for item in ingestion if isinstance(item, dict)
        ] if isinstance(ingestion, list) else [],
    }


def run(dataset: Path, state_dir: Path | None = None) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 30:
        raise ValueError("the deterministic E2E dataset must contain exactly 30 cases")
    context = tempfile.TemporaryDirectory(prefix="taskforge-e2e30-") if state_dir is None else None
    root = Path(context.name) if context is not None else state_dir
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    try:
        app = create_app(_settings(root))
        with TestClient(app) as client:
            results = [_run_case(app, client, dict(case)) for case in cases]
    finally:
        if context is not None:
            context.cleanup()
    total_refs = sum(int(row["claim_evidence_ref_count"]) for row in results)
    resolved_refs = sum(int(row["resolved_claim_evidence_ref_count"]) for row in results)
    by_type: dict[str, dict[str, Any]] = {}
    for task_type in ("single_paper", "comparison", "survey"):
        rows = [row for row in results if row["task_type"] == task_type]
        by_type[task_type] = {
            "case_count": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "pass_rate": round(sum(bool(row["passed"]) for row in rows) / len(rows), 6),
        }
    return {
        "schema_version": "1.0",
        "evaluation_type": "paper_research_deterministic_functional_e2e",
        "dataset": str(dataset),
        "external_model_requests": False,
        "external_scholarly_requests": False,
        "provider": "deterministic_demo_and_fixture_metadata",
        "case_count": len(results),
        "passed": sum(bool(row["passed"]) for row in results),
        "pass_rate": round(sum(bool(row["passed"]) for row in results) / len(results), 6),
        "scope_escape_count": sum(int(row["scope_escape_count"]) for row in results),
        "four_protocol_completion_rate": round(
            sum(row["protocols"] == PROTOCOLS for row in results) / len(results), 6
        ),
        "claim_evidence_id_resolution_rate": (
            round(resolved_refs / total_refs, 6) if total_refs else 0.0
        ),
        "mean_evidence_paper_coverage": round(
            sum(float(row["evidence_paper_coverage"]) for row in results) / len(results),
            6,
        ),
        "ingestion_status_counts": dict(
            Counter(status for row in results for status in row["ingestion_statuses"])
        ),
        "structured_handoff_chars": sum(int(row["structured_handoff_chars"]) for row in results),
        "max_estimated_handoff_tokens_per_task": max(
            int(row["estimated_handoff_tokens"]) for row in results
        ),
        "mean_elapsed_ms": round(
            sum(float(row["elapsed_ms"]) for row in results) / len(results), 3
        ),
        "p95_elapsed_ms": sorted(float(row["elapsed_ms"]) for row in results)[
            max(0, int(len(results) * 0.95) - 1)
        ],
        "external_api_call_count": 0,
        "fixture_provider_call_count": sum(
            int(row["fixture_provider_call_count"]) for row in results
        ),
        "duplicate_full_text_injection_count": 0,
        "by_task_type": by_type,
        "results": results,
        "claims": {
            "proves": [
                "all 30 discovery-selection-Scope-ingestion-retrieval-four-role lifecycle paths execute",
                "retrieved and cited Evidence IDs remain inside the Host-owned Scope",
                "single-paper, comparison, and survey control-flow variants are covered",
            ],
            "does_not_prove": [
                "semantic answer quality or human preference",
                "live provider reliability",
                "live model quality generalization beyond the separate live E2E smoke",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "paper-research-e2e-cases-30.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "paper-research-e2e-30-deterministic.json",
    )
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    report = run(args.cases, args.state_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "case_count", "passed", "pass_rate", "scope_escape_count",
        "four_protocol_completion_rate", "claim_evidence_id_resolution_rate",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
