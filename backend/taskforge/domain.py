"""Provider-neutral domain contracts for the TaskForge runtime.

The objects in this module are deliberately small and serialisable.  Provider
payloads and executable Python callables do not cross this boundary: a model
can only propose a :class:`ToolRequest`, and the host records the resulting
:class:`ToolResult`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp (kept as a function for testability)."""

    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model used for durable runtime data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    STEP_LIMIT = "step_limit"


class StepStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    WAITING_APPROVAL = "waiting_approval"
    FAILED = "failed"


class Task(StrictModel):
    """A user goal plus the identity and scope in which it may run."""

    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    input_artifacts: list[str] = Field(default_factory=list)
    workspace_id: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def task_id(self) -> str:
        """Explicit alias useful to adapters without duplicating stored data."""

        return self.id


class AgentProfile(StrictModel):
    """Configuration that changes Agent behaviour without changing runtime code."""

    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: str = Field(default="scripted", min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    memory_scopes: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=8, ge=1, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_capabilities(self) -> AgentProfile:
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools must not contain duplicates")
        return self

    @property
    def profile_id(self) -> str:
        return self.id


class ToolRequest(StrictModel):
    """A proposed capability invocation.  It has no authority by itself."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1)


class ToolResult(StrictModel):
    """Host-produced receipt for one tool request."""

    call_id: str = Field(min_length=1)
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def result_is_unambiguous(self) -> ToolResult:
        if self.ok and self.error is not None:
            raise ValueError("a successful tool result cannot contain an error")
        if not self.ok and not self.error:
            raise ValueError("an unsuccessful tool result must contain an error")
        return self


class ModelTurn(StrictModel):
    """A normalised provider response understood by the core runtime."""

    kind: Literal["final", "tool"]
    final_answer: str | None = None
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=16)
    assistant_text: str | None = None
    provider_response_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exactly_one_action_kind(self) -> ModelTurn:
        if self.kind == "final":
            if not self.final_answer or not self.final_answer.strip():
                raise ValueError("a final turn requires a non-empty final_answer")
            if self.tool_requests:
                raise ValueError("a final turn cannot also request tools")
        else:
            if not self.tool_requests:
                raise ValueError("a tool turn requires at least one ToolRequest")
            if self.final_answer is not None:
                raise ValueError("a tool turn cannot contain final_answer")
            call_ids = [request.call_id for request in self.tool_requests]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("tool call IDs must be unique within a model turn")
        return self


class PolicyDecision(StrictModel):
    """Host policy outcome for a proposed tool call."""

    allowed: bool
    requires_approval: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def decision_is_unambiguous(self) -> PolicyDecision:
        if self.allowed and self.requires_approval:
            raise ValueError("a call cannot be allowed and awaiting approval")
        return self

    @classmethod
    def allow(cls, reason: str = "") -> PolicyDecision:
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> PolicyDecision:
        return cls(allowed=False, reason=reason)

    @classmethod
    def approval(cls, reason: str) -> PolicyDecision:
        return cls(allowed=False, requires_approval=True, reason=reason)


class ApprovalResponse(StrictModel):
    """A human decision correlated to exactly one pending call."""

    call_id: str = Field(min_length=1)
    approved: bool
    reason: str = ""


class PendingApproval(StrictModel):
    """Durable cursor allowing an interrupted tool turn to resume safely."""

    step_index: int = Field(ge=0)
    request_index: int = Field(ge=0)
    request: ToolRequest
    reason: str


class RunError(StrictModel):
    stage: Literal["provider", "tool", "runtime"]
    code: str
    message: str
    retryable: bool = False


class StepRecord(StrictModel):
    """Durable trajectory record for one model turn and its tool receipts."""

    index: int = Field(ge=0)
    status: StepStatus = StepStatus.RUNNING
    model_turn: ModelTurn | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    error: RunError | None = None
    safe_summary: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None

    @property
    def tool_request(self) -> ToolRequest | None:
        """Compatibility convenience for single-call turns."""

        if not self.model_turn or not self.model_turn.tool_requests:
            return None
        return self.model_turn.tool_requests[0]

    @property
    def tool_result(self) -> ToolResult | None:
        if not self.tool_results:
            return None
        return self.tool_results[0]


class RunState(StrictModel):
    """Checkpointable state of a bounded Agent run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    task_id: str = Field(min_length=1)
    agent_profile_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.PENDING
    step_budget: int = Field(ge=1, le=100)
    steps: list[StepRecord] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None
    receipts: dict[str, ToolResult] = Field(default_factory=dict)
    idempotency_receipts: dict[str, str] = Field(default_factory=dict)
    final_answer: str | None = None
    error: RunError | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def terminal_and_pending_state_is_consistent(self) -> RunState:
        if self.status == RunStatus.WAITING_APPROVAL and self.pending_approval is None:
            raise ValueError("waiting_approval requires pending_approval")
        if self.status != RunStatus.WAITING_APPROVAL and self.pending_approval is not None:
            raise ValueError("pending_approval is only valid while waiting_approval")
        if self.status == RunStatus.COMPLETED and not self.final_answer:
            raise ValueError("a completed run requires final_answer")
        return self
