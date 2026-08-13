"""Run and persist a real-provider paper-research business E2E smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402


def _request(client: TestClient, method: str, path: str, **kwargs: object) -> dict[str, object]:
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text}")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned a non-object response")
    return value


def run(state_root: Path | None = None) -> dict[str, object]:
    configured = Settings()
    if configured.provider == "demo":
        raise RuntimeError("a real model provider must be configured for this E2E")
    if configured.provider == "deepseek" and configured.deepseek_api_key is None:
        raise RuntimeError("DeepSeek is selected but no API key is configured")
    if configured.provider == "openai" and configured.openai_api_key is None:
        raise RuntimeError("OpenAI is selected but no API key is configured")

    if state_root is not None:
        state_root.mkdir(parents=True, exist_ok=False)
    workspace = (
        nullcontext(str(state_root))
        if state_root is not None
        else tempfile.TemporaryDirectory(
            prefix="taskforge-paper-e2e-",
            ignore_cleanup_errors=True,
        )
    )
    with workspace as temporary:
        root = Path(temporary)
        state = root / "state"
        settings = configured.model_copy(
            update={
                "sqlite_path": state / "taskforge.sqlite3",
                "context_sqlite_path": state / "context.sqlite3",
                "operations_sqlite_path": state / "operations.sqlite3",
                "orchestration_sqlite_path": state / "orchestration.sqlite3",
                "review_case_sqlite_path": state / "review.sqlite3",
                "verification_sqlite_path": state / "verification.sqlite3",
                "literature_sqlite_path": state / "literature.sqlite3",
                "literature_cache_path": state / "literature-cache.sqlite3",
                "workspace_root": PROJECT_ROOT,
                "artifact_root": root / "artifacts",
            }
        )
        app = create_app(settings)
        suffix = uuid4().hex[:12]
        headers = {
            "X-TaskForge-Tenant": "paper-research-live-eval",
            "X-TaskForge-User": "e2e-evaluator",
        }
        with TestClient(app) as client:
            discovery = _request(
                client,
                "POST",
                "/api/literature/search",
                headers=headers,
                json={
                    "conversation_id": f"conversation-{suffix}",
                    "request": {
                        "request_id": f"request-{suffix}",
                        "query": (
                            "Self-RAG learning to retrieve generate and critique "
                            "through self-reflection"
                        ),
                        "research_questions": [
                            "How does retrieval and self-critique improve factuality?"
                        ],
                        "year_from": 2022,
                        "year_to": 2025,
                        "result_limit": 20,
                    },
                },
            )
            papers = discovery.get("papers")
            if not isinstance(papers, list) or not papers:
                raise RuntimeError("live literature discovery returned no papers")
            usable = [
                paper
                for paper in papers
                if isinstance(paper, dict) and str(paper.get("abstract") or "").strip()
            ]
            selected = usable[:2] if len(usable) >= 2 else papers[:1]
            selected_ids = [str(paper["paper_id"]) for paper in selected]
            excluded_ids = [
                str(paper["paper_id"])
                for paper in papers
                if isinstance(paper, dict) and str(paper.get("paper_id")) not in selected_ids
            ]
            scope = _request(
                client,
                "POST",
                "/api/research/scopes",
                headers=headers,
                json={
                    "request_id": discovery["request_id"],
                    "conversation_id": f"conversation-{suffix}",
                    "selected_paper_ids": selected_ids,
                    "excluded_paper_ids": excluded_ids,
                    "user_intent": (
                        "Explain how the selected papers combine retrieval, generation, "
                        "and critique, and identify evidence-backed limitations."
                    ),
                    "allowed_expansion": True,
                    "confirm": True,
                },
            )
            ingestion_response = client.post(
                f"/api/research/scopes/{scope['scope_id']}/ingest",
                headers=headers,
            )
            if not ingestion_response.is_success:
                raise RuntimeError(
                    f"scope ingestion failed ({ingestion_response.status_code}): "
                    f"{ingestion_response.text}"
                )
            ingestion = ingestion_response.json()
            evidence = _request(
                client,
                "POST",
                "/api/research/evidence/search",
                headers=headers,
                json={
                    "scope_id": scope["scope_id"],
                    "scope_version": scope["scope_version"],
                    "query": (
                        "How do retrieval and self-critique support factual generation, "
                        "and what limitations are reported?"
                    ),
                    "intent": "cross_paper_comparison",
                    "top_k": 10,
                    "candidate_k": 50,
                    "mode": "rigorous",
                },
            )
            created = _request(
                client,
                "POST",
                f"/api/research/scopes/{scope['scope_id']}/agent-run",
                headers={**headers, "Idempotency-Key": f"research-e2e-{suffix}"},
                json={
                    "title": "Self-RAG evidence-grounded survey",
                    "context": (
                        "Use only the Host-confirmed Scope. Pass structured IDs and "
                        "bounded handoff objects rather than prior chat transcripts."
                    ),
                    "survey_depth": "rigorous",
                },
            )
            case_id = str(created["case"]["case_id"])
            finished = _request(
                client,
                "POST",
                f"/api/review-cases/{case_id}/run-until-review",
                headers=headers,
                json={"max_iterations": 12},
            )

        role_runs = finished.get("role_runs")
        if not isinstance(role_runs, list):
            raise RuntimeError("E2E response omitted role runs")
        role_rows: list[dict[str, object]] = []
        successful_roles: dict[str, dict[str, object]] = {}
        total_input = total_output = total_tokens = 0
        structured_handoff_chars = 0
        evidence_refs: set[str] = set()
        for role in role_runs:
            if not isinstance(role, dict):
                continue
            metrics = role.get("runtime_metrics")
            usage = metrics.get("usage") if isinstance(metrics, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            used_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
            total_input += input_tokens
            total_output += output_tokens
            total_tokens += used_tokens
            result = role.get("role_result")
            payload = result.get("research_payload") if isinstance(result, dict) else None
            payload_chars = len(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ) if payload is not None else 0
            structured_handoff_chars += payload_chars
            refs = role.get("retrieved_evidence_refs")
            if isinstance(refs, list):
                evidence_refs.update(str(value) for value in refs)
            role_rows.append(
                {
                    "role_id": role.get("role_id"),
                    "status": role.get("status"),
                    "runtime_status": role.get("runtime_status"),
                    "tool_success_count": (
                        metrics.get("tool_success_count") if isinstance(metrics, dict) else None
                    ),
                    "tool_failure_count": (
                        metrics.get("tool_failure_count") if isinstance(metrics, dict) else None
                    ),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": used_tokens,
                    "structured_handoff_chars": payload_chars,
                    "research_protocol": (
                        payload.get("protocol") if isinstance(payload, dict) else None
                    ),
                    "evidence_ref_count": len(refs) if isinstance(refs, list) else 0,
                }
            )
            if role.get("status") == "succeeded" and isinstance(role.get("role_id"), str):
                successful_roles[str(role["role_id"])] = role_rows[-1]

        evidence_items = evidence.get("evidence")
        evidence_items = evidence_items if isinstance(evidence_items, list) else []
        escaped = [
            item
            for item in evidence_items
            if isinstance(item, dict) and item.get("paper_id") not in selected_ids
        ]
        ordered_roles = [
            "retrieval_planner",
            "source_evaluator",
            "synthesis_writer",
            "critical_reviewer",
        ]
        protocols = [
            successful_roles.get(role_id, {}).get("research_protocol")
            for role_id in ordered_roles
        ]
        complete = (
            finished.get("case", {}).get("status") == "waiting_human_review"
            if isinstance(finished.get("case"), dict)
            else False
        ) and protocols == [
            "research.planner_handoff.v1",
            "research.evaluator_handoff.v1",
            "research.writer_handoff.v1",
            "research.critic_handoff.v1",
        ]
        return {
            "schema_version": "1.0",
            "evaluation_type": "paper_research_business_e2e_live_model",
            "created_at": datetime.now(UTC).isoformat(),
            "provider": configured.provider,
            "model": (
                configured.deepseek_model
                if configured.provider == "deepseek"
                else configured.openai_model
            ),
            "live_external_model_requests": True,
            "live_scholarly_provider_requests": True,
            "discovery": {
                "raw_candidates": discovery.get("total_raw_candidates"),
                "paper_count": len(papers),
                "provider_reports": discovery.get("provider_reports"),
                "selected_paper_ids": selected_ids,
            },
            "scope": {
                "scope_id": scope.get("scope_id"),
                "scope_version": scope.get("scope_version"),
                "selected_paper_count": len(selected_ids),
                "status_before_ingestion": scope.get("status"),
            },
            "ingestion": ingestion,
            "bounded_retrieval": {
                "retrieval_rounds": evidence.get("retrieval_rounds"),
                "routed_intent": evidence.get("routed_intent"),
                "evidence_count": len(evidence_items),
                "scope_escape_count": len(escaped),
                "confidence": evidence.get("confidence"),
            },
            "multi_agent": {
                "case_id": case_id,
                "case_status": finished.get("case", {}).get("status")
                if isinstance(finished.get("case"), dict)
                else None,
                "attempt_count": len(role_rows),
                "role_count": len(successful_roles),
                "protocols": protocols,
                "all_structured_protocols_present": complete,
                "unique_evidence_ref_count": len(evidence_refs),
                "structured_handoff_chars": structured_handoff_chars,
                "estimated_handoff_tokens": round(structured_handoff_chars / 4),
                "usage": {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_tokens,
                },
                "roles": role_rows,
            },
            "claims": {
                "proves": [
                    "live scholarly discovery request path executed",
                    "ResearchScope-bound evidence retrieval executed",
                    *(
                        ["four configured Agent roles completed through the runtime"]
                        if complete
                        else ["the persisted report records the incomplete role attempts"]
                    ),
                ],
                "does_not_prove": [
                    "production reliability or provider SLA",
                    "answer-quality generalization beyond this smoke case",
                    "a 40 percent total-token reduction without a paired legacy live run",
                ],
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "paper-research-business-e2e-live.json",
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "eval"
            / "reports"
            / "paper-research-business-e2e-prebudget-live.json"
        ),
        help="Immutable successful pre-optimization report used for Token A/B.",
    )
    args = parser.parse_args()
    state_dir = args.state_dir or (
        PROJECT_ROOT
        / ".taskforge"
        / "eval-runs"
        / f"paper-research-e2e-live-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    # Preserve the first successful report before overwriting the canonical
    # output.  Failed/partial runs are never promoted as an A/B baseline.
    if args.output.exists() and not args.baseline_output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_multi_agent = previous.get("multi_agent", {})
        if (
            isinstance(previous_multi_agent, dict)
            and previous_multi_agent.get("all_structured_protocols_present") is True
        ):
            args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
            args.baseline_output.write_text(
                json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    report = run(state_dir)
    report["state_dir"] = str(state_dir)
    if args.baseline_output.exists():
        baseline = json.loads(args.baseline_output.read_text(encoding="utf-8"))
        baseline_usage = baseline.get("multi_agent", {}).get("usage", {})
        current_usage = report.get("multi_agent", {}).get("usage", {})
        baseline_tokens = int(baseline_usage.get("total_tokens") or 0)
        current_tokens = int(current_usage.get("total_tokens") or 0)
        report["token_comparison"] = {
            "baseline_report": str(args.baseline_output),
            "same_task_and_model": (
                baseline.get("provider") == report.get("provider")
                and baseline.get("model") == report.get("model")
                and baseline.get("evaluation_type") == report.get("evaluation_type")
            ),
            "baseline_total_tokens": baseline_tokens,
            "current_total_tokens": current_tokens,
            "absolute_reduction": baseline_tokens - current_tokens,
            "reduction_rate": (
                round((baseline_tokens - current_tokens) / baseline_tokens, 6)
                if baseline_tokens > 0
                else None
            ),
            "target_at_most_90000": current_tokens <= 90_000,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "provider": report["provider"],
        "model": report["model"],
        "bounded_retrieval": report["bounded_retrieval"],
        "multi_agent": report["multi_agent"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
