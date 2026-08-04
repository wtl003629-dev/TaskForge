"""Bounded, checkpointed Agent loop.

The runtime is intentionally provider- and infrastructure-neutral.  It asks a
model for proposals, delegates every proposal to host policy and tool ports,
and persists the resulting trajectory.  It never imports a concrete database,
retriever, MCP client, or provider SDK.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from .domain import (
    AgentProfile,
    ApprovalResponse,
    ModelTurn,
    PendingApproval,
    PolicyDecision,
    RunError,
    RunState,
    RunStatus,
    StepRecord,
    StepStatus,
    Task,
    ToolRequest,
    ToolResult,
    utc_now,
)
from .providers import ModelProvider


class ContextAssembler(Protocol):
    def assemble(
        self,
        query: str | None = None,
        profile: object | Mapping[str, Any] | None = None,
        task: object | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return governed task evidence; implementations may also be async."""


class ToolRegistry(Protocol):
    def list_specs(self, profile: AgentProfile) -> Sequence[Mapping[str, Any]]:
        """Return model-visible JSON schemas for allowed capabilities only."""

    def execute(
        self,
        request: ToolRequest,
        task: Task,
        profile: AgentProfile,
        state: RunState,
    ) -> ToolResult:
        """Execute a validated request; implementations may also be async."""


class PolicyEngine(Protocol):
    def evaluate(
        self,
        task: Task,
        profile: AgentProfile,
        request: ToolRequest,
    ) -> PolicyDecision:
        """Allow, deny, or pause for approval; may also be async."""


class CheckpointStore(Protocol):
    def save(self, state: RunState) -> None:
        """Durably save one state version; implementations may also be async."""


T = TypeVar("T")


async def _await_if_needed(value: T) -> T:
    if inspect.isawaitable(value):
        return await value  # type: ignore[no-any-return, misc]
    return value


def _state_update(state: RunState, **changes: Any) -> RunState:
    """Atomically update cross-validated RunState fields."""

    payload = state.model_dump()
    payload.update(changes)
    return RunState.model_validate(payload)


def _serialisable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
        return deepcopy(value)
    return str(value)


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]


def _is_retryable_provider_exception(exc: Exception) -> bool:
    """Classify only explicit/transient provider failures for durable retry."""

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return getattr(exc, "retryable", False) is True


def _request_fingerprint(request: ToolRequest) -> str:
    canonical = json.dumps(
        {"name": request.name, "arguments": request.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AgentRuntime:
    """Execute one Agent profile under bounded host control."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: ToolRegistry,
        policy: PolicyEngine,
        checkpoint: CheckpointStore,
        context: ContextAssembler,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.policy = policy
        self.checkpoint = checkpoint
        self.context = context

    async def run(
        self,
        task: Task,
        profile: AgentProfile,
        state: RunState | None = None,
        *,
        approval: ApprovalResponse | None = None,
    ) -> RunState:
        """Run until final, failure, approval pause, or the model-step budget.

        Resuming requires the previously checkpointed state.  When that state is
        awaiting approval, the caller must provide a response with the exact
        pending ``call_id``.  Receipts prevent repeated call IDs or idempotency
        keys from executing a tool twice.
        """

        if state is None:
            if approval is not None:
                raise ValueError("approval cannot be applied to a new run")
            state = RunState(
                task_id=task.id,
                agent_profile_id=profile.id,
                status=RunStatus.RUNNING,
                step_budget=profile.max_steps,
            )
            await self._save(state)
        else:
            self._validate_resume_identity(task, profile, state)
            state = state.model_copy(deep=True)
            if state.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.STEP_LIMIT,
            }:
                if approval is not None:
                    raise ValueError("approval cannot be applied to a terminal run")
                return state
            if state.status == RunStatus.WAITING_APPROVAL:
                if approval is None:
                    return state
                state = await self._resume_approval(task, profile, state, approval)
                if state.status != RunStatus.RUNNING:
                    return state
            elif approval is not None:
                raise ValueError("run has no pending approval")
            elif state.status == RunStatus.PENDING:
                state = _state_update(state, status=RunStatus.RUNNING)
                await self._save(state)

        while len(state.steps) < state.step_budget:
            step = StepRecord(index=len(state.steps))
            try:
                assembled = await _await_if_needed(
                    self.context.assemble(
                        query=task.goal,
                        profile=profile,
                        task=task,
                    )
                )
                tools = await self._schemas(profile)
                provider_context = self._provider_context(assembled, state)
            except Exception as exc:
                return await self._fail_model_step(
                    state,
                    step,
                    stage="runtime",
                    exc=exc,
                )

            try:
                turn = await _await_if_needed(
                    self.provider.complete(
                        task=task,
                        profile=profile,
                        context=provider_context,
                        tools=tools,
                    )
                )
                if not isinstance(turn, ModelTurn):
                    raise TypeError("provider returned a non-ModelTurn value")
            except Exception as exc:
                return await self._fail_model_step(
                    state,
                    step,
                    stage="provider",
                    exc=exc,
                )

            step.model_turn = turn
            state.steps.append(step)
            if turn.kind == "final":
                step.status = StepStatus.COMPLETED
                step.safe_summary = "model produced a final response"
                step.finished_at = utc_now()
                state = _state_update(
                    state,
                    status=RunStatus.COMPLETED,
                    final_answer=turn.final_answer,
                    error=None,
                    pending_approval=None,
                )
                await self._save(state)
                return state

            state = await self._process_requests(
                task,
                profile,
                state,
                step_index=step.index,
                start_index=0,
            )
            if state.status != RunStatus.RUNNING:
                return state
            step = state.steps[step.index]
            step.status = StepStatus.COMPLETED
            step.safe_summary = f"host processed {len(step.tool_results)} tool result(s)"
            step.finished_at = utc_now()
            await self._save(state)

        error = RunError(
            stage="runtime",
            code="step_budget_exhausted",
            message=f"run exhausted its {state.step_budget}-step model budget",
        )
        state = _state_update(
            state,
            status=RunStatus.STEP_LIMIT,
            error=error,
            pending_approval=None,
        )
        await self._save(state)
        return state

    @staticmethod
    def _validate_resume_identity(
        task: Task,
        profile: AgentProfile,
        state: RunState,
    ) -> None:
        if state.task_id != task.id:
            raise ValueError("checkpoint belongs to a different task")
        if state.agent_profile_id != profile.id:
            raise ValueError("checkpoint belongs to a different Agent profile")

    async def _schemas(
        self,
        profile: AgentProfile,
    ) -> list[Mapping[str, Any]]:
        schemas = await _await_if_needed(self.registry.list_specs(profile))
        if isinstance(schemas, (str, bytes)) or not isinstance(schemas, Sequence):
            raise TypeError("registry.list_specs must return a sequence")
        return [deepcopy(dict(schema)) for schema in schemas]

    def _provider_context(self, assembled: Any, state: RunState) -> dict[str, Any]:
        trajectory: list[dict[str, Any]] = []
        for step in state.steps:
            if step.model_turn is None:
                continue
            trajectory.append(
                {
                    "step": step.index,
                    "assistant_text": step.model_turn.assistant_text,
                    "provider_response_id": step.model_turn.provider_response_id,
                    "tool_requests": [
                        request.model_dump(mode="json")
                        for request in step.model_turn.tool_requests
                    ],
                    "tool_results": [
                        result.model_dump(mode="json") for result in step.tool_results
                    ],
                }
            )
        return {
            "assembled": _serialisable(assembled),
            "trajectory": trajectory,
        }

    async def _process_requests(
        self,
        task: Task,
        profile: AgentProfile,
        state: RunState,
        *,
        step_index: int,
        start_index: int,
    ) -> RunState:
        step = state.steps[step_index]
        assert step.model_turn is not None
        requests = step.model_turn.tool_requests
        for request_index in range(start_index, len(requests)):
            request = requests[request_index]
            receipt_or_error = self._existing_receipt(state, request)
            if isinstance(receipt_or_error, RunError):
                return await self._fail_tool_step(state, step_index, receipt_or_error)
            if receipt_or_error is not None:
                step.tool_results.append(receipt_or_error)
                continue

            if request.name not in profile.allowed_tools:
                denied = ToolResult(
                    call_id=request.call_id,
                    ok=False,
                    error="capability_denied",
                    metadata={"reason": "tool is not in AgentProfile.allowed_tools"},
                )
                denied = self._record_receipt(state, request, denied)
                step.tool_results.append(denied)
                continue

            try:
                raw_decision = await _await_if_needed(
                    self.policy.evaluate(
                        task,
                        profile,
                        request,
                    )
                )
                decision = (
                    raw_decision
                    if isinstance(raw_decision, PolicyDecision)
                    else PolicyDecision.model_validate(raw_decision)
                )
            except Exception as exc:
                error = RunError(
                    stage="runtime",
                    code=exc.__class__.__name__,
                    message=_exception_message(exc),
                )
                return await self._fail_tool_step(state, step_index, error)

            if decision.requires_approval:
                step.status = StepStatus.WAITING_APPROVAL
                step.safe_summary = f"tool {request.name!r} awaits approval"
                pending = PendingApproval(
                    step_index=step_index,
                    request_index=request_index,
                    request=request,
                    reason=decision.reason,
                )
                state = _state_update(
                    state,
                    status=RunStatus.WAITING_APPROVAL,
                    pending_approval=pending,
                    error=None,
                )
                await self._save(state)
                return state

            if not decision.allowed:
                denied = ToolResult(
                    call_id=request.call_id,
                    ok=False,
                    error="policy_denied",
                    metadata={"reason": decision.reason},
                )
                denied = self._record_receipt(state, request, denied)
                step.tool_results.append(denied)
                continue

            state = await self._execute_request(task, profile, state, step_index, request)
            if state.status != RunStatus.RUNNING:
                return state
            step = state.steps[step_index]
        return state

    async def _execute_request(
        self,
        task: Task,
        profile: AgentProfile,
        state: RunState,
        step_index: int,
        request: ToolRequest,
    ) -> RunState:
        try:
            raw_result = await _await_if_needed(
                self.registry.execute(
                    request,
                    task,
                    profile,
                    state.model_copy(deep=True),
                )
            )
        except Exception as exc:
            # Tool-domain failures are observations.  Giving the error back to
            # the model permits a bounded retry or an alternative plan.
            result = ToolResult(
                call_id=request.call_id,
                ok=False,
                error=f"{exc.__class__.__name__}: {_exception_message(exc)}",
            )
        else:
            try:
                result = (
                    raw_result
                    if isinstance(raw_result, ToolResult)
                    else ToolResult.model_validate(raw_result)
                )
                if result.call_id != request.call_id:
                    raise ValueError("tool result call_id does not match request")
            except Exception as exc:
                error = RunError(
                    stage="runtime",
                    code=exc.__class__.__name__,
                    message=_exception_message(exc),
                )
                return await self._fail_tool_step(state, step_index, error)
        result = self._record_receipt(state, request, result)
        state.steps[step_index].tool_results.append(result)
        self._collect_host_artifacts(state, result)
        return state

    @staticmethod
    def _collect_host_artifacts(state: RunState, result: ToolResult) -> None:
        """Promote a trusted tool's conventional artifact receipt into Run evidence."""

        if not result.ok or not isinstance(result.output, Mapping):
            return
        raw_artifact = result.output.get("artifact")
        if not isinstance(raw_artifact, Mapping):
            return
        artifact = deepcopy(dict(raw_artifact))
        if not isinstance(artifact.get("id"), str) or not artifact["id"]:
            return
        if not isinstance(artifact.get("source"), str) or not artifact["source"]:
            return
        if all(existing.get("id") != artifact["id"] for existing in state.artifacts):
            state.artifacts.append(deepcopy(artifact))
        if all(existing.get("id") != artifact["id"] for existing in state.evidence):
            state.evidence.append(deepcopy(artifact))

    async def _resume_approval(
        self,
        task: Task,
        profile: AgentProfile,
        state: RunState,
        approval: ApprovalResponse,
    ) -> RunState:
        pending = state.pending_approval
        assert pending is not None
        if approval.call_id != pending.request.call_id:
            raise ValueError("approval call_id does not match the pending request")
        step_index = pending.step_index
        request_index = pending.request_index
        state = _state_update(
            state,
            status=RunStatus.RUNNING,
            pending_approval=None,
            error=None,
        )
        step = state.steps[step_index]
        step.status = StepStatus.RUNNING
        step.finished_at = None

        if approval.approved:
            # Approval confirms the exact durable proposal; it does not freeze
            # authority forever. Re-evaluate the current profile and policy so
            # a capability revoked while a human was reviewing cannot execute
            # after restart or configuration rollout.
            if pending.request.name not in profile.allowed_tools:
                decision = PolicyDecision.deny(
                    "tool is no longer in the Agent capability set"
                )
            else:
                try:
                    raw_decision = await _await_if_needed(
                        self.policy.evaluate(task, profile, pending.request)
                    )
                    decision = (
                        raw_decision
                        if isinstance(raw_decision, PolicyDecision)
                        else PolicyDecision.model_validate(raw_decision)
                    )
                except Exception as exc:
                    error = RunError(
                        stage="runtime",
                        code=exc.__class__.__name__,
                        message=_exception_message(exc),
                    )
                    return await self._fail_tool_step(state, step_index, error)

            if decision.allowed or decision.requires_approval:
                state = await self._execute_request(
                    task,
                    profile,
                    state,
                    step_index,
                    pending.request,
                )
                if state.status != RunStatus.RUNNING:
                    return state
            else:
                result = ToolResult(
                    call_id=pending.request.call_id,
                    ok=False,
                    error="approval_invalidated",
                    metadata={"reason": decision.reason},
                )
                result = self._record_receipt(state, pending.request, result)
                step.tool_results.append(result)
        else:
            result = ToolResult(
                call_id=pending.request.call_id,
                ok=False,
                error="approval_denied",
                metadata={"reason": approval.reason},
            )
            result = self._record_receipt(state, pending.request, result)
            step.tool_results.append(result)

        state = await self._process_requests(
            task,
            profile,
            state,
            step_index=step_index,
            start_index=request_index + 1,
        )
        if state.status != RunStatus.RUNNING:
            return state
        step = state.steps[step_index]
        step.status = StepStatus.COMPLETED
        step.safe_summary = f"host processed {len(step.tool_results)} tool result(s)"
        step.finished_at = utc_now()
        await self._save(state)
        return state

    def _existing_receipt(
        self,
        state: RunState,
        request: ToolRequest,
    ) -> ToolResult | RunError | None:
        fingerprint = _request_fingerprint(request)
        existing = state.receipts.get(request.call_id)
        if existing is not None:
            if existing.metadata.get("request_fingerprint") != fingerprint:
                return RunError(
                    stage="runtime",
                    code="call_id_reused_with_different_request",
                    message="a tool call ID was reused with different arguments",
                )
            return existing.model_copy(deep=True)

        if request.idempotency_key:
            original_call_id = state.idempotency_receipts.get(request.idempotency_key)
            if original_call_id:
                original = state.receipts.get(original_call_id)
                if original is None:
                    return RunError(
                        stage="runtime",
                        code="invalid_idempotency_receipt",
                        message="idempotency index points to a missing receipt",
                    )
                if original.metadata.get("request_fingerprint") != fingerprint:
                    return RunError(
                        stage="runtime",
                        code="idempotency_key_reused_with_different_request",
                        message="an idempotency key was reused with different arguments",
                    )
                metadata = deepcopy(original.metadata)
                metadata["reused_from_call_id"] = original.call_id
                clone = original.model_copy(
                    update={"call_id": request.call_id, "metadata": metadata},
                    deep=True,
                )
                state.receipts[request.call_id] = clone
                return clone.model_copy(deep=True)
        return None

    @staticmethod
    def _record_receipt(
        state: RunState,
        request: ToolRequest,
        result: ToolResult,
    ) -> ToolResult:
        metadata = deepcopy(result.metadata)
        metadata["request_fingerprint"] = _request_fingerprint(request)
        recorded = result.model_copy(update={"metadata": metadata}, deep=True)
        state.receipts[request.call_id] = recorded
        if request.idempotency_key:
            state.idempotency_receipts[request.idempotency_key] = request.call_id
        return recorded.model_copy(deep=True)

    async def _fail_tool_step(
        self,
        state: RunState,
        step_index: int,
        error: RunError,
    ) -> RunState:
        step = state.steps[step_index]
        step.status = StepStatus.FAILED
        step.error = error
        step.safe_summary = "tool processing failed"
        step.finished_at = utc_now()
        state = _state_update(
            state,
            status=RunStatus.FAILED,
            error=error,
            pending_approval=None,
        )
        await self._save(state)
        return state

    async def _fail_model_step(
        self,
        state: RunState,
        step: StepRecord,
        *,
        stage: Literal["provider", "runtime"],
        exc: Exception,
    ) -> RunState:
        error = RunError(
            stage=stage,
            code=exc.__class__.__name__,
            message=_exception_message(exc),
            retryable=(
                stage == "provider" and _is_retryable_provider_exception(exc)
            ),
        )
        step.status = StepStatus.FAILED
        step.error = error
        step.safe_summary = "model turn failed"
        step.finished_at = utc_now()
        state.steps.append(step)
        state = _state_update(
            state,
            status=RunStatus.FAILED,
            error=error,
            pending_approval=None,
        )
        await self._save(state)
        return state

    async def _save(self, state: RunState) -> None:
        state.updated_at = utc_now()
        await _await_if_needed(self.checkpoint.save(state.model_copy(deep=True)))
