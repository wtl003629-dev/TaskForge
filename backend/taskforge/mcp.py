"""Governed MCP 2025-11-25 Streamable HTTP tool integration.

This is intentionally a narrow client: it implements JSON responses for
``initialize``, ``tools/list`` and ``tools/call``.  It advertises the required
Streamable HTTP media types but fails closed if a server selects SSE, so this
module never implies streaming support it does not have.

Server endpoints, tool allowlists, risks and approvals are host configuration.
None of them can be supplied by a model tool call.  Server descriptions,
annotations, schemas and results are all treated as untrusted input.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import json
import os
import re
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import ConfigDict, Field, model_validator

from .tooling import ToolRegistry, ToolRisk, ToolSpec
from .domain import StrictModel


MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_ACCEPT = "application/json, text/event-stream"
_REMOTE_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAMESPACE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,20}$")
_VISIBLE_ASCII = re.compile(r"^[\x21-\x7e]+$")
_JSON_SCHEMA_2020_12 = {
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2020-12/schema#",
}
_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_SINGLE_SCHEMA_KEYS = {
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_ARRAY_SCHEMA_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_MAPPING_SCHEMA_KEYS = {"$defs", "dependentSchemas", "patternProperties", "properties"}


class MCPError(RuntimeError):
    """Base error at the MCP trust boundary."""


class MCPConfigurationError(MCPError):
    """Invalid or disabled host-owned MCP configuration."""


class MCPSSRFError(MCPError):
    """Endpoint resolution crossed the configured network boundary."""


class MCPTransportError(MCPError):
    """Sanitised HTTP/timeout/media-type failure."""


class MCPUnsupportedTransportError(MCPTransportError):
    """The server selected an unsupported transport mode."""


class MCPProtocolError(MCPError):
    """Malformed or incompatible MCP/JSON-RPC response."""


class MCPRemoteProtocolError(MCPProtocolError):
    """A JSON-RPC error response (message/data intentionally not surfaced)."""


class MCPRemoteToolExecutionError(MCPError):
    """A remote tools/call completed with ``isError=true``."""


class MCPOutputLimitError(MCPProtocolError):
    """A remote response exceeded a host-owned output budget."""


class MCPToolPolicy(StrictModel):
    """Local policy for one allowlisted remote tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk: ToolRisk = ToolRisk.EXTERNAL
    side_effecting: bool = False
    requires_approval: bool = True
    description: str = Field(
        default="Host-approved MCP capability. Remote descriptions are not trusted.",
        min_length=1,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def consistent_risk(self) -> "MCPToolPolicy":
        if self.side_effecting and self.risk in {ToolRisk.READ, ToolRisk.COMPUTE}:
            raise ValueError("side-effecting MCP tools require write/external risk")
        if self.risk == ToolRisk.DESTRUCTIVE and not self.requires_approval:
            raise ValueError("destructive MCP tools require approval")
        return self


class MCPServerConfig(StrictModel):
    """Immutable-in-spirit host configuration; never accepted from model input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str = Field(min_length=1, max_length=21)
    endpoint: str = Field(min_length=1, max_length=2_048)
    enabled: bool = False
    allowed_tools: tuple[str, ...] = ()
    tool_policies: dict[str, MCPToolPolicy] = Field(default_factory=dict)
    secret_env_var: str | None = None
    allow_local_http: bool = False
    allow_private_network: bool = False
    allowed_ports: tuple[int, ...] = ()
    timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    max_output_chars: int = Field(default=12_000, ge=256, le=100_000)
    max_response_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    max_pages: int = Field(default=20, ge=1, le=100)
    max_tools: int = Field(default=128, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_host_configuration(self) -> "MCPServerConfig":
        if not _NAMESPACE.fullmatch(self.namespace):
            raise ValueError("namespace must be OpenAI-compatible ASCII")
        parsed = _validated_endpoint(self.endpoint)
        if parsed.scheme == "http" and not self.allow_local_http:
            raise ValueError("HTTP MCP endpoints require allow_local_http=true")
        explicit_port = _explicit_port(parsed)
        default_port = 443 if parsed.scheme == "https" else 80
        if explicit_port is not None and explicit_port != default_port:
            if explicit_port not in self.allowed_ports:
                raise ValueError("non-default endpoint port is not host-allowlisted")
        if len(self.allowed_ports) != len(set(self.allowed_ports)) or any(
            port < 1 or port > 65_535 for port in self.allowed_ports
        ):
            raise ValueError("allowed_ports must contain unique valid TCP ports")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools must not contain duplicates")
        for name in self.allowed_tools:
            if not _REMOTE_TOOL_NAME.fullmatch(name):
                raise ValueError("allowed MCP tool name is invalid")
        if set(self.tool_policies) != set(self.allowed_tools):
            raise ValueError("every allowed MCP tool needs exactly one local policy")
        if self.secret_env_var and not _ENV_NAME.fullmatch(self.secret_env_var):
            raise ValueError("secret_env_var must be an environment variable name")
        return self

    @property
    def parsed_endpoint(self) -> SplitResult:
        return _validated_endpoint(self.endpoint)


@dataclass(frozen=True, slots=True)
class MCPDiscoveredTool:
    remote_name: str
    mounted_name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    policy: MCPToolPolicy


Resolver = Callable[
    [str, int],
    Sequence[str | ipaddress.IPv4Address | ipaddress.IPv6Address]
    | Awaitable[Sequence[str | ipaddress.IPv4Address | ipaddress.IPv6Address]],
]


def _validated_endpoint(endpoint: str) -> SplitResult:
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in endpoint):
        raise ValueError("MCP endpoint must not contain whitespace or control characters")
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("MCP endpoint contains an invalid port or host") from exc
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("MCP endpoint must use HTTPS")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("MCP endpoint requires a host")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("MCP endpoint must not contain userinfo")
    if parsed.fragment:
        raise ValueError("MCP endpoint must not contain a fragment")
    if parsed.query:
        raise ValueError("MCP endpoint must not contain a query string")
    if parsed.path and not parsed.path.startswith("/"):
        raise ValueError("MCP endpoint requires an absolute path")
    host = parsed.hostname
    if "%" in host:
        raise ValueError("scoped IPv6 MCP endpoints are not supported")
    try:
        host.encode("idna")
    except UnicodeError as exc:
        raise ValueError("MCP endpoint host is invalid") from exc
    return parsed


def _explicit_port(parsed: SplitResult) -> int | None:
    return parsed.port


def _effective_port(parsed: SplitResult) -> int:
    return parsed.port or (443 if parsed.scheme == "https" else 80)


async def _default_resolver(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    try:
        records = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(host, port, type=socket.SOCK_STREAM),
        )
    except OSError as exc:
        raise MCPSSRFError("MCP endpoint DNS resolution failed") from None
    return tuple(record[4][0] for record in records)


def _safe_schema(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MCPProtocolError(f"{label} must be a JSON Schema object")
    schema = _strip_schema_annotations(deepcopy(dict(raw)))
    try:
        encoded = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MCPProtocolError(f"{label} is not JSON serialisable") from None
    if len(encoded) > 64_000:
        raise MCPProtocolError(f"{label} exceeds the schema budget")
    if schema.get("type") != "object":
        raise MCPProtocolError(f"{label} must have object at its root")
    dialect = schema.get("$schema")
    if dialect is not None and dialect not in _JSON_SCHEMA_2020_12:
        raise MCPProtocolError(f"{label} must use JSON Schema 2020-12")
    _reject_external_refs(schema, label=label)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MCPProtocolError(f"{label} is not valid JSON Schema 2020-12") from None
    schema.setdefault("additionalProperties", False)
    return schema


def _require_side_effect_idempotency(
    schema: Mapping[str, Any],
    *,
    tool_name: str,
) -> None:
    """Fail startup if a remote write cannot carry downstream idempotency."""

    properties = schema.get("properties")
    required = schema.get("required")
    property_schema = (
        properties.get("idempotency_key") if isinstance(properties, Mapping) else None
    )
    if (
        not isinstance(property_schema, Mapping)
        or property_schema.get("type") != "string"
        or not isinstance(required, list)
        or "idempotency_key" not in required
    ):
        raise MCPProtocolError(
            f"side-effecting MCP tool {tool_name!r} must declare a required string "
            "idempotency_key"
        )


def _reject_external_refs(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"}:
                if not isinstance(child, str) or not child.startswith("#"):
                    raise MCPProtocolError(f"{label} contains an external schema reference")
            _reject_external_refs(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_external_refs(child, label=label)


def _strip_schema_annotations(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove prose/hints before exposing an untrusted remote schema to a model."""

    cleaned = {
        key: deepcopy(value)
        for key, value in schema.items()
        if key not in _SCHEMA_ANNOTATION_KEYS
    }
    for key in _SINGLE_SCHEMA_KEYS:
        child = cleaned.get(key)
        if isinstance(child, Mapping):
            cleaned[key] = _strip_schema_annotations(dict(child))
    for key in _ARRAY_SCHEMA_KEYS:
        children = cleaned.get(key)
        if isinstance(children, list):
            cleaned[key] = [
                _strip_schema_annotations(dict(child))
                if isinstance(child, Mapping)
                else deepcopy(child)
                for child in children
            ]
    for key in _MAPPING_SCHEMA_KEYS:
        children = cleaned.get(key)
        if isinstance(children, Mapping):
            cleaned[key] = {
                name: _strip_schema_annotations(dict(child))
                if isinstance(child, Mapping)
                else deepcopy(child)
                for name, child in children.items()
            }
    return cleaned


def _mounted_name(namespace: str, remote_name: str) -> str:
    # OpenAI function names use the conservative ASCII alnum/underscore/dash
    # subset.  A digest prevents collisions such as ``a.b`` versus ``a_b``.
    prefix = f"mcp_{namespace}_"
    safe_remote = re.sub(r"[^A-Za-z0-9_-]", "_", remote_name)
    digest = hashlib.sha256(remote_name.encode("utf-8")).hexdigest()[:10]
    available = 64 - len(prefix) - len(digest) - 1
    return f"{prefix}{safe_remote[:available]}_{digest}"


class MCPStreamableHTTPClient:
    """Opt-in, JSON-response subset of MCP Streamable HTTP 2025-11-25."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._resolver = resolver or _default_resolver
        self._environment = os.environ if environment is None else environment
        self._timeout = httpx.Timeout(config.timeout_seconds)
        self._request_id = 0
        self._protocol_version: str | None = None
        self._session_id: str | None = None
        self._server_capabilities: dict[str, Any] | None = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._discovered: dict[str, MCPDiscoveredTool] = {}

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        self._ensure_enabled()
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            result, response = await self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "TaskForge", "version": "0.1.0"},
                },
                subsequent=False,
            )
            protocol_version = result.get("protocolVersion")
            if protocol_version != MCP_PROTOCOL_VERSION:
                raise MCPProtocolError("MCP protocol version negotiation failed")
            capabilities = result.get("capabilities")
            server_info = result.get("serverInfo")
            if not isinstance(capabilities, Mapping) or not isinstance(
                capabilities.get("tools"), Mapping
            ):
                raise MCPProtocolError("MCP server did not negotiate tools capability")
            if not isinstance(server_info, Mapping) or not all(
                isinstance(server_info.get(key), str) and server_info.get(key)
                for key in ("name", "version")
            ):
                raise MCPProtocolError("MCP initialize serverInfo is invalid")
            session_id = response.headers.get("MCP-Session-Id")
            if session_id is not None and (
                len(session_id) > 1_024 or not _VISIBLE_ASCII.fullmatch(session_id)
            ):
                raise MCPProtocolError("MCP session ID is invalid")

            self._protocol_version = protocol_version
            self._session_id = session_id
            self._server_capabilities = deepcopy(dict(capabilities))
            try:
                await self._notification("notifications/initialized")
            except Exception:
                self._protocol_version = None
                self._session_id = None
                self._server_capabilities = None
                raise
            self._initialized = True

    async def discover_tools(self, *, refresh: bool = False) -> list[MCPDiscoveredTool]:
        await self.initialize()
        if self._discovered and not refresh:
            return [deepcopy(tool) for tool in self._discovered.values()]

        discovered: dict[str, MCPDiscoveredTool] = {}
        all_names: set[str] = set()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        total_seen = 0
        for _ in range(self.config.max_pages):
            params = {"cursor": cursor} if cursor is not None else None
            result, _ = await self._request("tools/list", params, subsequent=True)
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise MCPProtocolError("tools/list result.tools must be an array")
            total_seen += len(raw_tools)
            if total_seen > self.config.max_tools:
                raise MCPProtocolError("tools/list exceeded the configured tool budget")
            for raw_tool in raw_tools:
                if not isinstance(raw_tool, Mapping):
                    raise MCPProtocolError("tools/list entries must be objects")
                name = raw_tool.get("name")
                if not isinstance(name, str) or not _REMOTE_TOOL_NAME.fullmatch(name):
                    raise MCPProtocolError("MCP server returned an invalid tool name")
                if name in all_names:
                    raise MCPProtocolError("MCP server returned a duplicate tool name")
                all_names.add(name)
                if name not in self.config.allowed_tools:
                    continue
                input_schema = _safe_schema(
                    raw_tool.get("inputSchema"), label=f"inputSchema for {name}"
                )
                policy = self.config.tool_policies[name]
                if policy.side_effecting:
                    _require_side_effect_idempotency(input_schema, tool_name=name)
                output_schema_raw = raw_tool.get("outputSchema")
                output_schema = (
                    _safe_schema(output_schema_raw, label=f"outputSchema for {name}")
                    if output_schema_raw is not None
                    else None
                )
                discovered[name] = MCPDiscoveredTool(
                    remote_name=name,
                    mounted_name=_mounted_name(self.config.namespace, name),
                    input_schema=input_schema,
                    output_schema=output_schema,
                    policy=policy,
                )
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor:
                raise MCPProtocolError("tools/list nextCursor must be a non-empty string")
            if next_cursor in seen_cursors:
                raise MCPProtocolError("tools/list repeated a pagination cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise MCPProtocolError("tools/list exceeded the pagination budget")

        missing = set(self.config.allowed_tools) - set(discovered)
        if missing:
            raise MCPProtocolError("one or more allowlisted MCP tools were not discovered")
        self._discovered = discovered
        return [deepcopy(tool) for tool in discovered.values()]

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        await self.initialize()
        if name not in self.config.allowed_tools:
            raise MCPConfigurationError("MCP tool is not host-allowlisted")
        if name not in self._discovered:
            await self.discover_tools()
        tool = self._discovered[name]
        if not isinstance(arguments, Mapping):
            raise MCPProtocolError("MCP tool arguments must be an object")
        try:
            Draft202012Validator(tool.input_schema).validate(dict(arguments))
        except ValidationError as exc:
            raise MCPProtocolError(
                "MCP tool arguments failed local schema validation"
            ) from None

        result, _ = await self._request(
            "tools/call",
            {"name": name, "arguments": deepcopy(dict(arguments))},
            subsequent=True,
        )
        bounded = _validate_tool_result(result, tool.output_schema)
        encoded = json.dumps(bounded, ensure_ascii=False, default=str)
        if len(encoded) > self.config.max_output_chars:
            raise MCPOutputLimitError("MCP tool output exceeded the configured limit")
        return bounded

    async def aclose(self) -> None:
        self._initialized = False
        self._protocol_version = None
        self._session_id = None
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "MCPStreamableHTTPClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _ensure_enabled(self) -> None:
        if not self.config.enabled:
            raise MCPConfigurationError("MCP server is disabled")

    async def _ensure_safe_endpoint(self) -> None:
        parsed = self.config.parsed_endpoint
        host = parsed.hostname
        assert host is not None
        value = self._resolver(host, _effective_port(parsed))
        resolved = await value if inspect.isawaitable(value) else value
        if isinstance(resolved, (str, bytes)) or not isinstance(resolved, Sequence):
            raise MCPSSRFError("MCP resolver returned an invalid result")
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for raw_address in resolved:
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError as exc:
                raise MCPSSRFError("MCP resolver returned an invalid IP address") from None
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                address = address.ipv4_mapped
            addresses.append(address)
        if not addresses:
            raise MCPSSRFError("MCP endpoint did not resolve to an IP address")

        for address in addresses:
            if (
                address.is_link_local
                or address.is_reserved
                or address.is_multicast
                or address.is_unspecified
            ):
                raise MCPSSRFError("MCP endpoint resolved to a blocked IP range")
            if address.is_loopback:
                if parsed.scheme == "http" and self.config.allow_local_http:
                    continue
                if self.config.allow_private_network:
                    continue
                raise MCPSSRFError("MCP endpoint resolved to loopback")
            if address.is_private:
                if self.config.allow_private_network:
                    continue
                raise MCPSSRFError("MCP endpoint resolved to a private network")
            if not address.is_global:
                raise MCPSSRFError("MCP endpoint resolved to a non-public address")
            if parsed.scheme == "http":
                raise MCPSSRFError("plain HTTP MCP is restricted to local endpoints")

    def _headers(self, *, subsequent: bool) -> dict[str, str]:
        headers = {
            "Accept": MCP_ACCEPT,
            "Content-Type": "application/json",
        }
        if self.config.secret_env_var:
            token = self._environment.get(self.config.secret_env_var)
            if not token:
                raise MCPConfigurationError("configured MCP credential is unavailable")
            headers["Authorization"] = f"Bearer {token}"
        if subsequent:
            if self._protocol_version is None:
                raise MCPProtocolError("MCP protocol version has not been negotiated")
            headers["MCP-Protocol-Version"] = self._protocol_version
            if self._session_id is not None:
                headers["MCP-Session-Id"] = self._session_id
        return headers

    async def _request(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        *,
        subsequent: bool,
    ) -> tuple[dict[str, Any], httpx.Response]:
        self._request_id += 1
        request_id = self._request_id
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            body["params"] = deepcopy(dict(params))
        response = await self._post(body, subsequent=subsequent)
        result = _jsonrpc_result(response, expected_id=request_id)
        return result, response

    async def _notification(self, method: str) -> None:
        response = await self._post(
            {"jsonrpc": "2.0", "method": method}, subsequent=True
        )
        if response.status_code != 202:
            raise MCPTransportError("MCP notification was not accepted")

    async def _post(
        self,
        body: Mapping[str, Any],
        *,
        subsequent: bool,
    ) -> httpx.Response:
        await self._ensure_safe_endpoint()
        try:
            async with self._client.stream(
                "POST",
                self.config.endpoint,
                headers=self._headers(subsequent=subsequent),
                json=body,
                timeout=self._timeout,
                follow_redirects=False,
            ) as streamed:
                if not 200 <= streamed.status_code < 300:
                    raise MCPTransportError(
                        f"MCP endpoint returned HTTP {streamed.status_code}"
                    )
                if "id" in body:
                    content_type = (
                        streamed.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .lower()
                    )
                    if content_type == "text/event-stream":
                        raise MCPUnsupportedTransportError(
                            "MCP SSE responses are unsupported by this client"
                        )
                    if content_type != "application/json":
                        raise MCPTransportError(
                            "MCP response media type is unsupported"
                        )
                content = bytearray()
                async for chunk in streamed.aiter_bytes():
                    if len(content) + len(chunk) > self.config.max_response_bytes:
                        raise MCPOutputLimitError(
                            "MCP response exceeded the byte limit"
                        )
                    content.extend(chunk)
                response = httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    content=bytes(content),
                    request=streamed.request,
                )
        except httpx.TimeoutException:
            raise MCPTransportError("MCP request timed out") from None
        except httpx.RequestError:
            raise MCPTransportError("MCP request failed") from None
        return response


def _jsonrpc_result(response: httpx.Response, *, expected_id: int) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise MCPProtocolError("MCP endpoint returned invalid JSON") from None
    if not isinstance(payload, Mapping):
        raise MCPProtocolError("MCP JSON-RPC response must be an object")
    if payload.get("jsonrpc") != "2.0":
        raise MCPProtocolError("MCP JSON-RPC version is invalid")
    response_id = payload.get("id")
    if type(response_id) is not type(expected_id) or response_id != expected_id:
        raise MCPProtocolError("MCP JSON-RPC response ID does not match the request")
    has_result = "result" in payload
    has_error = "error" in payload
    if has_result == has_error:
        raise MCPProtocolError("MCP JSON-RPC response must contain result or error")
    if has_error:
        error = payload["error"]
        if not isinstance(error, Mapping):
            raise MCPProtocolError("MCP JSON-RPC error object is invalid")
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, bool) or not isinstance(code, int) or not isinstance(message, str):
            raise MCPProtocolError("MCP JSON-RPC error object is invalid")
        raise MCPRemoteProtocolError(f"MCP server returned JSON-RPC error {code}")
    result = payload["result"]
    if not isinstance(result, Mapping):
        raise MCPProtocolError("MCP JSON-RPC result must be an object")
    return deepcopy(dict(result))


def _validate_tool_result(
    raw: Mapping[str, Any], output_schema: Mapping[str, Any] | None
) -> dict[str, Any]:
    content = raw.get("content")
    if not isinstance(content, list):
        raise MCPProtocolError("tools/call result.content must be an array")
    for item in content:
        if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
            raise MCPProtocolError("tools/call content entries must be typed objects")
        item_type = item["type"]
        if item_type == "text":
            if not isinstance(item.get("text"), str):
                raise MCPProtocolError("MCP text content requires text")
        elif item_type in {"image", "audio"}:
            if not isinstance(item.get("data"), str) or not isinstance(
                item.get("mimeType"), str
            ):
                raise MCPProtocolError("MCP binary content is invalid")
        elif item_type == "resource_link":
            if not isinstance(item.get("uri"), str) or not isinstance(
                item.get("name"), str
            ):
                raise MCPProtocolError("MCP resource link is invalid")
        elif item_type == "resource":
            resource = item.get("resource")
            if not isinstance(resource, Mapping) or not isinstance(
                resource.get("uri"), str
            ):
                raise MCPProtocolError("MCP embedded resource is invalid")
            has_text = isinstance(resource.get("text"), str)
            has_blob = isinstance(resource.get("blob"), str)
            if has_text == has_blob:
                raise MCPProtocolError("MCP embedded resource needs text or blob")
        else:
            raise MCPProtocolError("MCP tool result content type is unsupported")
    is_error = raw.get("isError", False)
    if not isinstance(is_error, bool):
        raise MCPProtocolError("tools/call isError must be boolean")
    structured = raw.get("structuredContent")
    if structured is not None and not isinstance(structured, Mapping):
        raise MCPProtocolError("tools/call structuredContent must be an object")
    if output_schema is not None and not is_error:
        if structured is None:
            raise MCPProtocolError("tool outputSchema requires structuredContent")
        try:
            Draft202012Validator(output_schema).validate(dict(structured))
        except ValidationError as exc:
            raise MCPProtocolError(
                "structuredContent failed outputSchema validation"
            ) from None
    return deepcopy(dict(raw))


class MCPToolRegistryAdapter:
    """Mount allowlisted discoveries into TaskForge's governed registry."""

    def __init__(self, client: MCPStreamableHTTPClient) -> None:
        self.client = client

    async def mount(self, registry: ToolRegistry) -> dict[str, str]:
        tools = await self.client.discover_tools()
        for tool in tools:
            if tool.policy.side_effecting:
                _require_side_effect_idempotency(
                    tool.input_schema,
                    tool_name=tool.remote_name,
                )
            if registry.spec(tool.mounted_name) is not None:
                raise ValueError(f"tool already registered: {tool.mounted_name}")

        mounted: dict[str, str] = {}
        for tool in tools:
            spec = ToolSpec(
                name=tool.mounted_name,
                description=tool.policy.description,
                parameters=deepcopy(tool.input_schema),
                risk=tool.policy.risk,
                side_effecting=tool.policy.side_effecting,
                requires_approval=tool.policy.requires_approval,
                strict=False,
                timeout_seconds=self.client.config.timeout_seconds,
                max_output_chars=self.client.config.max_output_chars,
            )
            remote_name = tool.remote_name

            async def handler(arguments: dict[str, Any], *_, _name: str = remote_name) -> Any:
                result = await self.client.call_tool(_name, arguments)
                if result.get("isError") is True:
                    raise MCPRemoteToolExecutionError(
                        "MCP tool reported a remote execution error"
                    )
                return result

            registry.register(spec, handler)
            mounted[remote_name] = tool.mounted_name
        return mounted


async def mount_mcp_tools(
    registry: ToolRegistry, client: MCPStreamableHTTPClient
) -> dict[str, str]:
    """Convenience adapter entrypoint."""

    return await MCPToolRegistryAdapter(client).mount(registry)
