"""A deterministic, offline provider for exercising the complete runtime.

This is intentionally not presented as an intelligent model.  It derives its
next proposal only from the durable trajectory supplied by ``AgentRuntime``,
which makes approval resumption work after a process restart without hidden
provider state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .domain import AgentProfile, ModelTurn, Task, ToolRequest

_DEMO_LABEL = "TaskForge deterministic offline demo provider"


def _trajectory(context: Any) -> list[Mapping[str, Any]]:
    if not isinstance(context, Mapping):
        return []
    value = context.get("trajectory", [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _repo_pattern(goal: str) -> str:
    candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}|[\u3400-\u9fff]{2,}", goal)
    if not candidates:
        return "TaskForge"
    # Identifiers and specific phrases generally carry more signal than the
    # first imperative verb in a repository question.
    return max(candidates, key=lambda item: (len(item), item.casefold()))[:128]


def _receipt_json(trajectory: Sequence[Mapping[str, Any]]) -> str:
    receipts: list[Any] = []
    for entry in trajectory:
        raw = entry.get("tool_results", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            receipts.extend(raw)
    rendered = json.dumps(receipts, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > 30_000:
        rendered = rendered[:30_000] + "\n... [bounded by demo provider]"
    return rendered


def _artifact_result(trajectory: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return _tool_result(trajectory, "artifact_write")


def _tool_result(
    trajectory: Sequence[Mapping[str, Any]], tool_name: str
) -> Mapping[str, Any] | None:
    for entry in reversed(trajectory):
        requests = entry.get("tool_requests", [])
        results = entry.get("tool_results", [])
        if not (
            isinstance(requests, Sequence)
            and not isinstance(requests, (str, bytes))
            and isinstance(results, Sequence)
            and not isinstance(results, (str, bytes))
        ):
            continue
        call_ids = {
            request.get("call_id")
            for request in requests
            if isinstance(request, Mapping) and request.get("name") == tool_name
        }
        for result in results:
            if isinstance(result, Mapping) and result.get("call_id") in call_ids:
                return result
    return None


def _case_evidence_refs(trajectory: Sequence[Mapping[str, Any]]) -> list[str]:
    result = _tool_result(trajectory, "knowledge_search")
    if isinstance(result, Mapping) and result.get("ok") is True:
        output = result.get("output")
        hits = output.get("hits") if isinstance(output, Mapping) else None
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
            refs: list[str] = []
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                evidence_id = hit.get("evidence_id")
                chunk_id = hit.get("chunk_id")
                source = hit.get("source")
                if isinstance(evidence_id, str) and evidence_id:
                    value = evidence_id
                else:
                    value = chunk_id if isinstance(chunk_id, str) and chunk_id else source
                if isinstance(value, str) and value and value not in refs:
                    refs.append(value[:500])
                if len(refs) >= 8:
                    break
            if refs:
                return refs
    # This label is deliberately explicit: it satisfies only the demo schema
    # and is never a host verification receipt.
    return ["demo:offline-unverified-evidence"]


def _submitted_case_evidence_refs(task: Task) -> list[str]:
    """Read only the host-delimited case IDs; never interpret case prose."""

    marker = "CASE_INPUT_JSON="
    start = task.goal.find(marker)
    if start < 0 or task.workspace_id is None:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(task.goal[start + len(marker) :])
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(payload, Mapping) or payload.get("case_id") != task.workspace_id:
        return []
    values = payload.get("evidence_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    refs: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 240
            or value in refs
        ):
            return []
        refs.append(value)
        if len(refs) > 100:
            return []
    return refs


def _case_evidence_refs_for_task(
    task: Task,
    trajectory: Sequence[Mapping[str, Any]],
) -> list[str]:
    # Submitted IDs are the only references the review service can bind to the
    # case record.  Retrieval chunk IDs remain useful only for generic demos.
    submitted = _submitted_case_evidence_refs(task)
    if not submitted:
        return _case_evidence_refs(trajectory)
    retrieved = _case_evidence_refs(trajectory)
    bound = [value for value in retrieved if value in set(submitted)]
    # A declared evidence ID is not evidence that the Agent actually retrieved
    # it.  Keep the explicit unverified marker so business reconciliation fails
    # closed when the governed knowledge tool returned nothing relevant.
    return bound or ["demo:offline-unverified-evidence"]


def _case_turn(task: Task, profile: AgentProfile, history: list[Mapping[str, Any]]) -> ModelTurn:
    if not history:
        return ModelTurn(
            kind="tool",
            tool_requests=[
                ToolRequest(
                    call_id=f"case-knowledge-{task.id}",
                    name="knowledge_search",
                    arguments={"query": task.goal[:1_000], "limit": 5},
                )
            ],
            assistant_text=f"{_DEMO_LABEL}: collect governed case evidence.",
            metadata={"provider": "demo", "deterministic": True, "case_mode": True},
        )
    if _tool_result(history, "submit_role_result") is None:
        role_id = str(profile.metadata.get("role_id", "review_role"))[:120]
        is_decision = role_id == "decision_synthesizer"
        return ModelTurn(
            kind="tool",
            tool_requests=[
                ToolRequest(
                    call_id=f"case-submit-{task.id}",
                    name="submit_role_result",
                    arguments={
                        "claims": [
                            {
                                "fact_key": (
                                    "decision.outcome"
                                    if is_decision
                                    else f"{role_id}.recommendation"
                                ),
                                "value": (
                                    "escalate"
                                    if is_decision
                                    else {
                                        "recommendation": "needs_human_review",
                                        "offline_demo": True,
                                    }
                                ),
                                "evidence_refs": _case_evidence_refs_for_task(
                                    task, history
                                ),
                                "confidence": 0.5,
                            }
                        ],
                        "summary": (
                            f"{_DEMO_LABEL} produced a schema-valid, unverified "
                            f"recommendation for {role_id}."
                        ),
                        "handoff_summary": (
                            "Offline deterministic result only; downstream roles must treat "
                            "it as unverified until host verification."
                        ),
                    },
                )
            ],
            assistant_text=f"{_DEMO_LABEL}: submit an explicitly unverified role result.",
            metadata={"provider": "demo", "deterministic": True, "case_mode": True},
        )
    answer = (
        f"{_DEMO_LABEL} completed the structured role-result flow. "
        "No live model was called and no fact was automatically verified."
    )
    return ModelTurn(
        kind="final",
        final_answer=answer,
        assistant_text=answer,
        metadata={"provider": "demo", "deterministic": True, "case_mode": True},
    )


class DemoProvider:
    """Stateless provider that demonstrates retrieval, approval, and resume."""

    async def complete(
        self,
        *,
        task: Task,
        profile: AgentProfile,
        context: Any,
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        history = _trajectory(context)
        visible_tools = {
            str(tool.get("name"))
            for tool in tools
            if isinstance(tool, Mapping) and tool.get("name")
        }
        if "submit_role_result" in visible_tools:
            return _case_turn(task, profile, history)

        if not history:
            if profile.id == "repo-agent":
                request = ToolRequest(
                    call_id=f"workspace-{task.id}",
                    name="workspace_grep",
                    arguments={
                        "pattern": _repo_pattern(task.goal),
                        "include": "*",
                        "regex": False,
                        "case_sensitive": False,
                        "limit": 8,
                    },
                )
            else:
                request = ToolRequest(
                    call_id=f"knowledge-{task.id}",
                    name="knowledge_search",
                    arguments={"query": task.goal[:1_000], "limit": 5},
                )
            return ModelTurn(
                kind="tool",
                tool_requests=[request],
                assistant_text=f"{_DEMO_LABEL}: collect one governed evidence receipt.",
                metadata={"provider": "demo", "deterministic": True},
            )

        if len(history) == 1:
            key = f"artifact-{task.id}"
            content = (
                "# TaskForge Offline Demo Report\n\n"
                f"> Generated by the {_DEMO_LABEL}; this is not a live LLM response.\n\n"
                f"## Goal\n\n{task.goal}\n\n"
                "## Host tool receipts\n\n"
                "The JSON below is the actual bounded receipt returned by the governed tool.\n\n"
                f"```json\n{_receipt_json(history)}\n```\n"
            )
            request = ToolRequest(
                call_id=f"artifact-{task.id}",
                name="artifact_write",
                idempotency_key=key,
                arguments={
                    "filename": f"{profile.id}-report.md",
                    "content": content,
                    "idempotency_key": key,
                },
            )
            return ModelTurn(
                kind="tool",
                tool_requests=[request],
                assistant_text=f"{_DEMO_LABEL}: propose a report artifact for human approval.",
                metadata={"provider": "demo", "deterministic": True},
            )

        receipt = _artifact_result(history)
        if receipt is None:
            answer = (
                f"{_DEMO_LABEL} finished, but no artifact_write receipt was present "
                "in the durable trajectory."
            )
        elif receipt.get("ok") is True:
            output = receipt.get("output")
            source = None
            if isinstance(output, Mapping) and isinstance(output.get("artifact"), Mapping):
                source = output["artifact"].get("source")
            suffix = f" at {source}" if isinstance(source, str) else ""
            answer = (
                f"{_DEMO_LABEL} completed. The real artifact_write receipt confirms "
                f"that the approved report was written{suffix}."
            )
        else:
            answer = (
                f"{_DEMO_LABEL} completed without writing a report. The real "
                f"artifact_write receipt records: {receipt.get('error', 'unknown_error')}."
            )
        return ModelTurn(
            kind="final",
            final_answer=answer,
            assistant_text=answer,
            metadata={"provider": "demo", "deterministic": True},
        )


__all__ = ["DemoProvider"]
