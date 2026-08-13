"""Governed tool registry and capability policy.

Provider adapters receive only JSON schemas. They never receive Python callables,
and a model-produced request must pass both policy and argument validation before
the handler can run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import ValidationError as PydanticValidationError
from pydantic import Field, model_validator

from .domain import (
    AgentProfile,
    PolicyDecision,
    RunState,
    StrictModel,
    Task,
    ToolRequest,
    ToolResult,
)


class ToolRisk(str, Enum):
    READ = "read"
    COMPUTE = "compute"
    WRITE = "write"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class ToolSpec(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    description: str = Field(min_length=1, max_length=1_000)
    parameters: dict[str, Any]
    risk: ToolRisk = ToolRisk.READ
    side_effecting: bool = False
    requires_approval: bool = False
    strict: bool = True
    timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    max_output_chars: int = Field(default=12_000, ge=256, le=100_000)

    @model_validator(mode="after")
    def schema_and_risk_are_consistent(self) -> ToolSpec:
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as exc:
            raise ValueError(f"parameters is not valid JSON Schema: {exc.message}") from exc
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must describe a JSON object")
        if self.strict:
            if self.parameters.get("additionalProperties") is not False:
                raise ValueError("strict function schemas require additionalProperties=false")
            properties = set(self.parameters.get("properties", {}))
            required = set(self.parameters.get("required", []))
            optional = sorted(properties - required)
            if optional:
                raise ValueError(
                    "strict function schemas require every property; optional: "
                    + ", ".join(optional)
                )
        if self.side_effecting and self.risk in {ToolRisk.READ, ToolRisk.COMPUTE}:
            raise ValueError("side-effecting tools need write, external, or destructive risk")
        if self.risk == ToolRisk.DESTRUCTIVE and not self.requires_approval:
            raise ValueError("destructive tools must require approval")
        return self

    def provider_schema(self) -> dict[str, Any]:
        """Return the function-tool shape used by Responses-style providers."""

        parameters = dict(self.parameters)
        parameters.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "strict": self.strict,
        }


ToolHandler = Callable[
    [dict[str, Any], Task, AgentProfile, RunState],
    Any | Awaitable[Any],
]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def spec(self, name: str) -> ToolSpec | None:
        registered = self._tools.get(name)
        return registered[0] if registered else None

    def list_specs(self, profile: AgentProfile) -> Sequence[Mapping[str, Any]]:
        return [
            self._tools[name][0].provider_schema()
            for name in profile.allowed_tools
            if name in self._tools
        ]

    @staticmethod
    def _tool_use_count(state: RunState, name: str) -> int:
        return sum(
            1
            for step in state.steps
            if step.model_turn is not None
            for item in step.model_turn.tool_requests
            if item.name == name and item.call_id in state.receipts
        )

    async def execute(
        self,
        request: ToolRequest,
        task: Task,
        profile: AgentProfile,
        state: RunState,
    ) -> ToolResult:
        cached = state.receipts.get(request.call_id)
        if cached is not None:
            return cached.model_copy(deep=True)

        registered = self._tools.get(request.name)
        if registered is None:
            return ToolResult(call_id=request.call_id, ok=False, error="unknown_tool")
        spec, handler = registered
        if request.name not in profile.allowed_tools:
            return ToolResult(call_id=request.call_id, ok=False, error="tool_not_allowed")
        raw_limits = profile.metadata.get("tool_call_limits", {})
        limits = raw_limits if isinstance(raw_limits, Mapping) else {}
        raw_limit = limits.get(request.name)
        if isinstance(raw_limit, int) and raw_limit >= 0:
            if self._tool_use_count(state, request.name) >= raw_limit:
                result = ToolResult(
                    call_id=request.call_id,
                    ok=False,
                    error="tool_call_limit_exceeded",
                    metadata={"tool": request.name, "limit": raw_limit},
                )
                state.receipts[request.call_id] = result
                return result.model_copy(deep=True)
        if spec.side_effecting and not request.idempotency_key:
            return ToolResult(call_id=request.call_id, ok=False, error="idempotency_key_required")
        if request.idempotency_key:
            previous_call_id = state.idempotency_receipts.get(request.idempotency_key)
            if previous_call_id and previous_call_id in state.receipts:
                previous = state.receipts[previous_call_id]
                replay = previous.model_copy(deep=True)
                replay.call_id = request.call_id
                replay.metadata = {**replay.metadata, "idempotent_replay_of": previous_call_id}
                state.receipts[request.call_id] = replay
                return replay

        try:
            Draft202012Validator(spec.parameters).validate(request.arguments)
        except ValidationError as exc:
            return ToolResult(
                call_id=request.call_id,
                ok=False,
                error=f"invalid_arguments: {exc.message}",
            )

        async def invoke() -> Any:
            value = handler(request.arguments, task, profile, state)
            return await value if inspect.isawaitable(value) else value

        try:
            output = await asyncio.wait_for(invoke(), timeout=spec.timeout_seconds)
            output, truncated = _bounded_output(output, spec.max_output_chars)
            result = ToolResult(
                call_id=request.call_id,
                ok=True,
                output=output,
                metadata={
                    "tool": spec.name,
                    "risk": spec.risk.value,
                    "side_effecting": spec.side_effecting,
                    "truncated": truncated,
                },
            )
        except TimeoutError:
            result = ToolResult(call_id=request.call_id, ok=False, error="tool_timeout")
        except PydanticValidationError as exc:
            issues = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                for issue in exc.errors(include_url=False)[:4]
            )
            result = ToolResult(
                call_id=request.call_id,
                ok=False,
                error=f"tool_validation_error:{issues}",
            )
        except Exception as exc:  # handlers are an untrusted integration boundary
            result = ToolResult(
                call_id=request.call_id,
                ok=False,
                error=f"tool_error:{type(exc).__name__}",
            )

        state.receipts[request.call_id] = result
        if request.idempotency_key:
            state.idempotency_receipts[request.idempotency_key] = request.call_id
        return result.model_copy(deep=True)


class CapabilityPolicy:
    """Deterministic capability and approval gate."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def evaluate(
        self,
        task: Task,
        profile: AgentProfile,
        request: ToolRequest,
    ) -> PolicyDecision:
        del task  # identity checks can be injected without changing the runtime contract
        if request.name not in profile.allowed_tools:
            return PolicyDecision.deny("tool is not in the Agent capability set")
        spec = self._registry.spec(request.name)
        if spec is None:
            return PolicyDecision.deny("tool is not registered")
        if spec.side_effecting and not request.idempotency_key:
            return PolicyDecision.deny("side-effecting tools require an idempotency key")
        if spec.requires_approval or spec.risk in {
            ToolRisk.WRITE,
            ToolRisk.EXTERNAL,
            ToolRisk.DESTRUCTIVE,
        }:
            return PolicyDecision.approval(f"{spec.risk.value} capability requires approval")
        return PolicyDecision.allow("read/compute capability allowed")


def _bounded_output(output: Any, max_chars: int) -> tuple[Any, bool]:
    try:
        encoded = json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps(str(output), ensure_ascii=False)
    if len(encoded) <= max_chars:
        return output, False
    return {"truncated": True, "preview": encoded[:max_chars]}, True
