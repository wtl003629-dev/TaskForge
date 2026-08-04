"""Provider contracts and pure response/payload adapters.

No adapter in this module performs network I/O.  A production HTTP client can
use these conversion functions and still expose the same ``ModelProvider``
protocol to the runtime.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from .domain import AgentProfile, ModelTurn, Task, ToolRequest


class ProviderError(RuntimeError):
    """Base error raised at the provider boundary."""

    retryable = False


class RetryableProviderError(ProviderError):
    """Explicit adapter contract for a transient provider failure."""

    retryable = True


class ProviderResponseError(ProviderError):
    """Raised when a provider response cannot be normalised safely."""


class ModelProvider(Protocol):
    async def complete(
        self,
        *,
        task: Task,
        profile: AgentProfile,
        context: Any,
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        """Return one normalised model turn without executing any tool."""


def _strip_single_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ProviderResponseError("unterminated JSON code fence")
    if lines[0].strip().lower() not in {"```", "```json"}:
        raise ProviderResponseError("only a JSON code fence is accepted")
    return "\n".join(lines[1:-1]).strip()


def _object_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("tool arguments are not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderResponseError("tool arguments must be a JSON object")
    return deepcopy(dict(value))


def _request_from_mapping(
    value: Mapping[str, Any],
    *,
    default_name: str | None = None,
    call_id_factory: Callable[[], str],
) -> ToolRequest:
    name = value.get("name") or value.get("tool") or default_name
    if not isinstance(name, str) or not name.strip():
        raise ProviderResponseError("tool action requires a non-empty name")
    call_id = value.get("call_id") or value.get("id") or call_id_factory()
    if not isinstance(call_id, str) or not call_id:
        raise ProviderResponseError("tool call_id must be a non-empty string")
    arguments = _object_arguments(value.get("arguments", value.get("args")))
    # Native function-calling protocols expose only schema-declared arguments,
    # so an idempotency key commonly arrives inside that object.  Promote it to
    # the host contract while retaining the argument for normal schema checks.
    idempotency_key = value.get("idempotency_key", arguments.get("idempotency_key"))
    return ToolRequest(
        call_id=call_id,
        name=name,
        arguments=arguments,
        idempotency_key=idempotency_key,
    )


def parse_json_action(
    value: str | Mapping[str, Any],
    *,
    call_id_factory: Callable[[], str] | None = None,
) -> ModelTurn:
    """Parse the deterministic/offline JSON action protocol.

    Supported forms are intentionally narrow::

        {"action": "final", "answer": "..."}
        {"action": "tool", "tool": "search", "arguments": {...}}
        {"action": "search", "arguments": {...}}
        {"tool_calls": [{"name": "search", "arguments": {...}}]}

    Markdown JSON fences are accepted, but prose surrounding JSON is rejected.
    Parsing only creates a proposal; it never resolves or executes a callable.
    """

    factory = call_id_factory or (lambda: f"json-{uuid4()}")
    if isinstance(value, str):
        raw = _strip_single_json_fence(value)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("model output is not a single JSON object") from exc
    else:
        decoded = deepcopy(dict(value))
    if not isinstance(decoded, Mapping):
        raise ProviderResponseError("model action must be a JSON object")

    action = decoded.get("action", decoded.get("type"))
    if action in {"final", "finish"}:
        answer = decoded.get("answer", decoded.get("final_answer"))
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderResponseError("final action requires a non-empty answer")
        return ModelTurn(kind="final", final_answer=answer, assistant_text=answer)

    calls = decoded.get("tool_calls")
    if calls is None and action in {"tools", "tool_calls"}:
        calls = decoded.get("calls")
    if calls is not None:
        if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence) or not calls:
            raise ProviderResponseError("tool_calls must be a non-empty array")
        requests: list[ToolRequest] = []
        for call in calls:
            if not isinstance(call, Mapping):
                raise ProviderResponseError("each tool call must be an object")
            requests.append(
                _request_from_mapping(call, call_id_factory=factory)
            )
        return ModelTurn(kind="tool", tool_requests=requests)

    if action in {"tool", "tool_call"}:
        request = _request_from_mapping(decoded, call_id_factory=factory)
    elif isinstance(action, str) and action.strip():
        request = _request_from_mapping(
            decoded,
            default_name=action,
            call_id_factory=factory,
        )
    else:
        raise ProviderResponseError("action must be final or a tool proposal")
    return ModelTurn(kind="tool", tool_requests=[request])


def _normalise_openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a provider-neutral tool schema to a Responses function tool."""

    if tool.get("type") == "function" and isinstance(tool.get("name"), str):
        native = deepcopy(dict(tool))
        native.setdefault("parameters", {"type": "object", "properties": {}})
        return native

    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool schema requires a non-empty name")
    parameters = tool.get("parameters", tool.get("input_schema"))
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"parameters for tool {name!r} must be an object")
    converted: dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": deepcopy(dict(parameters)),
    }
    if tool.get("description") is not None:
        converted["description"] = str(tool["description"])
    if tool.get("strict") is not None:
        converted["strict"] = bool(tool["strict"])
    return converted


def build_openai_responses_payload(
    *,
    model: str,
    instructions: str,
    input_items: str | Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] = (),
    previous_response_id: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Responses API payload without sending it."""

    if not model:
        raise ValueError("model is required")
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items if isinstance(input_items, str) else deepcopy(list(input_items)),
    }
    if tools:
        payload["tools"] = [_normalise_openai_tool(tool) for tool in tools]
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if metadata is not None:
        payload["metadata"] = deepcopy(dict(metadata))
    return payload


def _materialise_response(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise ProviderResponseError("OpenAI response must be a mapping or model_dump object")


def parse_openai_responses_response(value: Any) -> ModelTurn:
    """Normalise a Responses API response into a provider-neutral turn."""

    response = _materialise_response(value)
    response_id = response.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise ProviderResponseError("response id must be a string")
    output = response.get("output", [])
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise ProviderResponseError("response output must be an array")

    requests: list[ToolRequest] = []
    text_parts: list[str] = []
    for index, raw_item in enumerate(output):
        item = _materialise_response(raw_item)
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            requests.append(
                _request_from_mapping(
                    item,
                    call_id_factory=lambda i=index: f"openai-call-{i}",
                )
            )
            continue
        if item_type != "message":
            continue
        content = item.get("content", [])
        if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
            raise ProviderResponseError("message content must be an array")
        for raw_part in content:
            part = _materialise_response(raw_part)
            if part.get("type") not in {"output_text", "text"}:
                continue
            text_value = part.get("text")
            if isinstance(text_value, Mapping):
                text_value = text_value.get("value")
            if isinstance(text_value, str):
                text_parts.append(text_value)

    shortcut_text = response.get("output_text")
    if not text_parts and isinstance(shortcut_text, str):
        text_parts.append(shortcut_text)
    assistant_text = "\n".join(part for part in text_parts if part).strip() or None
    metadata: dict[str, Any] = {}
    for key in ("model", "status", "usage"):
        if key in response:
            metadata[key] = deepcopy(response[key])

    if requests:
        return ModelTurn(
            kind="tool",
            tool_requests=requests,
            assistant_text=assistant_text,
            provider_response_id=response_id,
            metadata=metadata,
        )
    if assistant_text:
        return ModelTurn(
            kind="final",
            final_answer=assistant_text,
            assistant_text=assistant_text,
            provider_response_id=response_id,
            metadata=metadata,
        )
    raise ProviderResponseError("OpenAI response contains neither tool calls nor output text")


class OpenAIResponsesAdapter:
    """Namespaced pure conversion helpers for an eventual HTTP provider."""

    build_payload = staticmethod(build_openai_responses_payload)
    parse_response = staticmethod(parse_openai_responses_response)


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    task: Task
    profile: AgentProfile
    context: Any
    tools: tuple[Mapping[str, Any], ...]


class ScriptedProvider:
    """Deterministic provider used by tests and the offline demo."""

    def __init__(
        self,
        turns: Sequence[ModelTurn | str | Mapping[str, Any] | BaseException],
    ) -> None:
        self._turns = list(turns)
        self._cursor = 0
        self.calls: list[ScriptedCall] = []

    async def complete(
        self,
        *,
        task: Task,
        profile: AgentProfile,
        context: Any,
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        self.calls.append(
            ScriptedCall(
                task=task.model_copy(deep=True),
                profile=profile.model_copy(deep=True),
                context=deepcopy(context),
                tools=tuple(deepcopy(list(tools))),
            )
        )
        if self._cursor >= len(self._turns):
            raise ProviderError("scripted provider exhausted")
        scripted = self._turns[self._cursor]
        self._cursor += 1
        if isinstance(scripted, BaseException):
            raise scripted
        if isinstance(scripted, ModelTurn):
            return scripted.model_copy(deep=True)
        return parse_json_action(scripted)
