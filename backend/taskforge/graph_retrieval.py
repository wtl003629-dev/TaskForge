"""Optional, evaluation-gated Neo4j retrieval and controlled RRF fusion.

Graph retrieval is deliberately a host-selected retrieval route, not a model
capability.  Models cannot provide Cypher, labels, relationship types, or an
unbounded traversal depth.  The two static query templates apply identity,
ACL, knowledge-base, version, and validity-window predicates to every node and
relationship before Neo4j returns a candidate.  Returned records are checked
again in Python and any scope violation fails the whole query closed.

The feature gate defaults to disabled and requires evidence from one exact,
locked evaluation artifact.  A configured quality improvement is never enough
when the safety suite reports a regression.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .domain import StrictModel
from .hybrid_retrieval import HybridSearchHit

# These are code-owned.  They are intentionally absent from GraphSearchRequest.
# Extending the graph vocabulary therefore requires a reviewed code change.
RELATIONSHIP_ALLOWLIST: tuple[str, ...] = (
    "CITES",
    "DEPENDS_ON",
    "REFERENCES",
    "SUPPORTS",
)


class GraphRetrievalError(RuntimeError):
    """Base class for explicit graph retrieval failures."""


class Neo4jUnavailableError(GraphRetrievalError):
    """Raised when a real Neo4j connection is requested without its extra."""


class GraphBackendError(GraphRetrievalError):
    """Raised for a driver failure or an unsafe result shape."""


class GraphGateConfigurationError(GraphRetrievalError):
    """Raised when a graph gate file is absent or invalid."""


class GraphFusionScopeError(GraphRetrievalError):
    """Raised when retrieval sources do not share the gated graph scope."""


def _required(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, str):
        source: Iterable[object] = (value,)
    elif value is None:
        source = ()
    else:
        source = value  # type: ignore[assignment]
    cleaned = frozenset(str(item).strip() for item in source if str(item).strip())
    if not cleaned:
        raise ValueError(f"{field_name} must contain at least one non-empty value")
    return cleaned


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _sha256_strings(values: Iterable[str]) -> str:
    return hashlib.sha256("\0".join(sorted(values)).encode("utf-8")).hexdigest()


class GraphAccess(StrictModel):
    """Trusted graph scope resolved by the host before retrieval."""

    tenant_id: str = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=256)
    acl_principals: frozenset[str] = Field(min_length=1, max_length=256)
    knowledge_base_ids: frozenset[str] = Field(min_length=1, max_length=128)
    versions: frozenset[str] = Field(min_length=1, max_length=128)
    as_of: datetime

    @field_validator("tenant_id", "user_id", mode="before")
    @classmethod
    def clean_identity(cls, value: object, info: Any) -> str:
        return _required(value, info.field_name)

    @field_validator("acl_principals", "knowledge_base_ids", "versions", mode="before")
    @classmethod
    def clean_sets(cls, value: object, info: Any) -> frozenset[str]:
        return _string_set(value, info.field_name)

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        return _utc(value)


class AppliedGraphScope(StrictModel):
    """Safe telemetry proving which trusted filters were applied."""

    tenant_id: str
    user_id: str
    acl_principal_count: int = Field(ge=1, le=256)
    acl_principals_sha256: str = Field(min_length=64, max_length=64)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=128)
    versions: list[str] = Field(min_length=1, max_length=128)
    as_of: datetime
    max_hops: Literal[1, 2]
    relationship_allowlist: list[str]

    @classmethod
    def from_request(cls, request: GraphSearchRequest) -> AppliedGraphScope:
        access = request.access
        return cls(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            acl_principal_count=len(access.acl_principals),
            acl_principals_sha256=_sha256_strings(access.acl_principals),
            knowledge_base_ids=sorted(access.knowledge_base_ids),
            versions=sorted(access.versions),
            as_of=access.as_of,
            max_hops=request.max_hops,
            relationship_allowlist=list(RELATIONSHIP_ALLOWLIST),
        )


class GraphSearchRequest(StrictModel):
    """A bounded traversal rooted at a host-resolved entity identifier."""

    entity_id: str = Field(min_length=1, max_length=512)
    access: GraphAccess
    max_hops: Literal[1, 2] = 2
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("entity_id", mode="before")
    @classmethod
    def clean_entity_id(cls, value: object) -> str:
        return _required(value, "entity_id")


class GraphGateConfig(StrictModel):
    """Host configuration for the locked graph-retrieval quality gate."""

    version: Literal[1] = 1
    enabled: bool = False
    locked_evaluation_id: str | None = Field(default=None, min_length=1, max_length=256)
    locked_evaluation_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    metric_name: str = Field(default="recall_at_5", min_length=1, max_length=128)
    minimum_improvement: float = Field(default=0.03, ge=0.0, le=1.0)
    require_zero_safety_regressions: Literal[True] = True
    graph_result_limit: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=10_000)
    hybrid_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    graph_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    fused_result_budget: int = Field(default=10, ge=1, le=100)
    fused_character_budget: int = Field(default=16_000, ge=1, le=2_000_000)

    @field_validator("metric_name", "locked_evaluation_id", mode="before")
    @classmethod
    def clean_optional_strings(cls, value: object, info: Any) -> object:
        if value is None and info.field_name == "locked_evaluation_id":
            return None
        return _required(value, info.field_name)

    @field_validator("minimum_improvement", "hybrid_weight", "graph_weight")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("graph gate numeric values must be finite")
        return float(value)

    @model_validator(mode="after")
    def enabled_gate_is_locked(self) -> GraphGateConfig:
        if self.enabled and (
            self.locked_evaluation_id is None
            or self.locked_evaluation_sha256 is None
        ):
            raise ValueError(
                "an enabled graph gate requires a locked evaluation ID and SHA-256"
            )
        return self

    @classmethod
    def load(cls, path: str | Path) -> GraphGateConfig:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except Exception as exc:
            raise GraphGateConfigurationError(
                f"invalid graph gate configuration at {source}: {type(exc).__name__}"
            ) from exc


class GraphEvaluationEvidence(StrictModel):
    """One immutable result from the exact evaluation artifact in the gate."""

    evaluation_id: str = Field(min_length=1, max_length=256)
    evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_name: str = Field(min_length=1, max_length=128)
    baseline_score: float = Field(ge=0.0, le=1.0)
    graph_score: float = Field(ge=0.0, le=1.0)
    safety_regressions: int = Field(ge=0)

    @field_validator("evaluation_id", "metric_name", mode="before")
    @classmethod
    def clean_strings(cls, value: object, info: Any) -> str:
        return _required(value, info.field_name)

    @field_validator("baseline_score", "graph_score")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("evaluation scores must be finite")
        return float(value)

    @property
    def improvement(self) -> float:
        return self.graph_score - self.baseline_score


GateReason = Literal[
    "config_disabled",
    "evaluation_missing",
    "evaluation_identity_mismatch",
    "evaluation_hash_mismatch",
    "metric_mismatch",
    "safety_regression",
    "improvement_below_threshold",
    "gate_passed",
]


class GraphGateDecision(StrictModel):
    enabled: bool
    reason: GateReason
    measured_improvement: float | None = None
    required_improvement: float

    @field_validator("measured_improvement", "required_improvement")
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("gate decision values must be finite")
        return value


class GraphRetrievalGate:
    """Evaluates enablement without consulting mutable runtime state."""

    def __init__(self, config: GraphGateConfig) -> None:
        self.config = config

    def evaluate(
        self, evidence: GraphEvaluationEvidence | None
    ) -> GraphGateDecision:
        threshold = self.config.minimum_improvement
        if not self.config.enabled:
            return GraphGateDecision(
                enabled=False,
                reason="config_disabled",
                required_improvement=threshold,
            )
        if evidence is None:
            return GraphGateDecision(
                enabled=False,
                reason="evaluation_missing",
                required_improvement=threshold,
            )
        measured = evidence.improvement
        if evidence.evaluation_id != self.config.locked_evaluation_id:
            reason: GateReason = "evaluation_identity_mismatch"
        elif evidence.evaluation_sha256 != self.config.locked_evaluation_sha256:
            reason = "evaluation_hash_mismatch"
        elif evidence.metric_name != self.config.metric_name:
            reason = "metric_mismatch"
        elif evidence.safety_regressions != 0:
            reason = "safety_regression"
        elif measured + 1e-12 < threshold:
            reason = "improvement_below_threshold"
        else:
            return GraphGateDecision(
                enabled=True,
                reason="gate_passed",
                measured_improvement=measured,
                required_improvement=threshold,
            )
        return GraphGateDecision(
            enabled=False,
            reason=reason,
            measured_improvement=measured,
            required_improvement=threshold,
        )


class _GraphNodeRecord(StrictModel):
    entity_id: str = Field(min_length=1, max_length=512)
    tenant_id: str = Field(min_length=1, max_length=256)
    allowed_user_ids: frozenset[str] = Field(min_length=1, max_length=256)
    acl_principals: frozenset[str] = Field(min_length=1, max_length=256)
    knowledge_base_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_uri: str = Field(min_length=1, max_length=2_048)
    chunk_id: str | None = Field(default=None, min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=2_000_000)

    @field_validator(
        "entity_id",
        "tenant_id",
        "knowledge_base_id",
        "version",
        "source_uri",
        "chunk_id",
        "text",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, value: object, info: Any) -> object:
        if value is None and info.field_name == "chunk_id":
            return None
        return _required(value, info.field_name)

    @field_validator("allowed_user_ids", "acl_principals", mode="before")
    @classmethod
    def clean_access_sets(cls, value: object, info: Any) -> frozenset[str]:
        return _string_set(value, info.field_name)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def normalize_dates(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def window_is_ordered(self) -> _GraphNodeRecord:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from >= self.valid_until
        ):
            raise ValueError("node valid_from must be earlier than valid_until")
        return self


class _GraphRelationshipRecord(StrictModel):
    relationship_id: str = Field(min_length=1, max_length=512)
    relationship_type: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=256)
    allowed_user_ids: frozenset[str] = Field(min_length=1, max_length=256)
    acl_principals: frozenset[str] = Field(min_length=1, max_length=256)
    knowledge_base_id: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_uri: str = Field(min_length=1, max_length=2_048)

    @field_validator(
        "relationship_id",
        "relationship_type",
        "tenant_id",
        "knowledge_base_id",
        "version",
        "source_uri",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, value: object, info: Any) -> str:
        return _required(value, info.field_name)

    @field_validator("allowed_user_ids", "acl_principals", mode="before")
    @classmethod
    def clean_access_sets(cls, value: object, info: Any) -> frozenset[str]:
        return _string_set(value, info.field_name)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def normalize_dates(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def window_is_ordered(self) -> _GraphRelationshipRecord:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from >= self.valid_until
        ):
            raise ValueError("relationship valid_from must be earlier than valid_until")
        return self


class GraphNodeProvenance(StrictModel):
    entity_id: str
    source_uri: str
    chunk_id: str | None = None
    knowledge_base_id: str
    version: str


class GraphRelationshipProvenance(StrictModel):
    relationship_id: str
    relationship_type: str
    source_uri: str
    knowledge_base_id: str
    version: str


class GraphPathProvenance(StrictModel):
    hop_count: Literal[1, 2]
    nodes: list[GraphNodeProvenance] = Field(min_length=2, max_length=3)
    relationships: list[GraphRelationshipProvenance] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def path_shape_matches_hops(self) -> GraphPathProvenance:
        if len(self.relationships) != self.hop_count:
            raise ValueError("relationship count must equal hop_count")
        if len(self.nodes) != self.hop_count + 1:
            raise ValueError("node count must equal hop_count + 1")
        return self


class GraphEvidence(StrictModel):
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_id: str
    chunk_id: str | None = None
    text: str
    source_uri: str
    knowledge_base_id: str
    version: str
    hop_count: Literal[1, 2]
    backend_score: float
    retrieval_source: Literal["neo4j_graph"] = "neo4j_graph"
    path: GraphPathProvenance

    @field_validator("backend_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("graph score must be finite")
        return float(value)


class GraphSearchResponse(StrictModel):
    gate: GraphGateDecision
    query_executed: bool
    scope: AppliedGraphScope
    hits: list[GraphEvidence]

    @model_validator(mode="after")
    def closed_gate_cannot_carry_results(self) -> GraphSearchResponse:
        if not self.gate.enabled and (self.query_executed or self.hits):
            raise ValueError("a closed graph gate cannot execute or return hits")
        if self.gate.enabled and not self.query_executed:
            raise ValueError("a passed graph gate must execute its bounded query")
        return self


_NODE_PREDICATE = """
  node.tenant_id = $tenant_id
  AND ($user_id IN coalesce(node.allowed_user_ids, [])
       OR 'tenant' IN coalesce(node.allowed_user_ids, []))
  AND any(principal IN $acl_principals
          WHERE principal IN coalesce(node.acl_principals, []))
  AND node.knowledge_base_id IN $knowledge_base_ids
  AND node.version IN $versions
  AND (node.valid_from IS NULL OR datetime(node.valid_from) <= datetime($as_of))
  AND (node.valid_until IS NULL OR datetime(node.valid_until) > datetime($as_of))
""".strip()

_RELATIONSHIP_PREDICATE = """
  type(rel) IN $relationship_allowlist
  AND rel.tenant_id = $tenant_id
  AND ($user_id IN coalesce(rel.allowed_user_ids, [])
       OR 'tenant' IN coalesce(rel.allowed_user_ids, []))
  AND any(principal IN $acl_principals
          WHERE principal IN coalesce(rel.acl_principals, []))
  AND rel.knowledge_base_id IN $knowledge_base_ids
  AND rel.version IN $versions
  AND (rel.valid_from IS NULL OR datetime(rel.valid_from) <= datetime($as_of))
  AND (rel.valid_until IS NULL OR datetime(rel.valid_until) > datetime($as_of))
""".strip()


def _query_for_hops(max_hops: Literal[1, 2]) -> str:
    # max_hops has already been reduced to Literal[1, 2].  Only this numeric
    # token differs between reviewed templates; no caller text is interpolated.
    path_pattern = "[*1..1]" if max_hops == 1 else "[*1..2]"
    return f"""
MATCH path = (seed:KnowledgeEntity {{entity_id: $entity_id}})-{path_pattern}-(target:KnowledgeEntity)
WHERE all(node IN nodes(path) WHERE {_NODE_PREDICATE})
  AND all(rel IN relationships(path) WHERE {_RELATIONSHIP_PREDICATE})
RETURN
  [node IN nodes(path) | {{
    entity_id: node.entity_id,
    tenant_id: node.tenant_id,
    allowed_user_ids: coalesce(node.allowed_user_ids, []),
    acl_principals: coalesce(node.acl_principals, []),
    knowledge_base_id: node.knowledge_base_id,
    version: node.version,
    valid_from: node.valid_from,
    valid_until: node.valid_until,
    source_uri: node.source_uri,
    chunk_id: node.chunk_id,
    text: node.text
  }}] AS nodes,
  [rel IN relationships(path) | {{
    relationship_id: rel.relationship_id,
    relationship_type: type(rel),
    tenant_id: rel.tenant_id,
    allowed_user_ids: coalesce(rel.allowed_user_ids, []),
    acl_principals: coalesce(rel.acl_principals, []),
    knowledge_base_id: rel.knowledge_base_id,
    version: rel.version,
    valid_from: rel.valid_from,
    valid_until: rel.valid_until,
    source_uri: rel.source_uri
  }}] AS relationships,
  length(path) AS hop_count,
  coalesce(target.graph_score, 1.0) AS score
ORDER BY score DESC, target.entity_id ASC
LIMIT $limit
""".strip()


def _parameters(request: GraphSearchRequest, result_limit: int) -> dict[str, Any]:
    access = request.access
    return {
        "entity_id": request.entity_id,
        "tenant_id": access.tenant_id,
        "user_id": access.user_id,
        "acl_principals": sorted(access.acl_principals),
        "knowledge_base_ids": sorted(access.knowledge_base_ids),
        "versions": sorted(access.versions),
        "as_of": access.as_of.isoformat(),
        "relationship_allowlist": list(RELATIONSHIP_ALLOWLIST),
        "limit": min(request.limit, result_limit),
    }


def _mapping(record: object) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    data = getattr(record, "data", None)
    if callable(data):
        value = data()
        if isinstance(value, Mapping):
            return value
    raise GraphBackendError("Neo4j returned a non-mapping record")


def _visible_user(allowed_user_ids: frozenset[str], user_id: str) -> bool:
    return user_id in allowed_user_ids or "tenant" in allowed_user_ids


def _valid_at(
    valid_from: datetime | None,
    valid_until: datetime | None,
    instant: datetime,
) -> bool:
    return not (
        (valid_from is not None and instant < valid_from)
        or (valid_until is not None and instant >= valid_until)
    )


def _node_in_scope(node: _GraphNodeRecord, access: GraphAccess) -> bool:
    return bool(
        node.tenant_id == access.tenant_id
        and _visible_user(node.allowed_user_ids, access.user_id)
        and node.acl_principals.intersection(access.acl_principals)
        and node.knowledge_base_id in access.knowledge_base_ids
        and node.version in access.versions
        and _valid_at(node.valid_from, node.valid_until, access.as_of)
    )


def _relationship_in_scope(
    relationship: _GraphRelationshipRecord,
    access: GraphAccess,
) -> bool:
    return bool(
        relationship.relationship_type in RELATIONSHIP_ALLOWLIST
        and relationship.tenant_id == access.tenant_id
        and _visible_user(relationship.allowed_user_ids, access.user_id)
        and relationship.acl_principals.intersection(access.acl_principals)
        and relationship.knowledge_base_id in access.knowledge_base_ids
        and relationship.version in access.versions
        and _valid_at(
            relationship.valid_from,
            relationship.valid_until,
            access.as_of,
        )
    )


def _parse_evidence(
    raw: Mapping[str, Any],
    request: GraphSearchRequest,
) -> GraphEvidence:
    try:
        raw_nodes = raw["nodes"]
        raw_relationships = raw["relationships"]
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
            raise TypeError("nodes must be a sequence")
        if not isinstance(raw_relationships, Sequence) or isinstance(
            raw_relationships, (str, bytes)
        ):
            raise TypeError("relationships must be a sequence")
        nodes = [_GraphNodeRecord.model_validate(item) for item in raw_nodes]
        relationships = [
            _GraphRelationshipRecord.model_validate(item) for item in raw_relationships
        ]
        hop_count = int(raw["hop_count"])
        score = float(raw["score"])
    except Exception as exc:
        raise GraphBackendError(
            f"Neo4j returned an invalid graph record: {type(exc).__name__}"
        ) from exc

    if hop_count not in (1, 2) or hop_count > request.max_hops:
        raise GraphBackendError("Neo4j returned a path outside the fixed hop bound")
    if len(nodes) != hop_count + 1 or len(relationships) != hop_count:
        raise GraphBackendError("Neo4j returned an inconsistent path shape")
    if nodes[0].entity_id != request.entity_id:
        raise GraphBackendError("Neo4j returned a path rooted at another entity")
    if not math.isfinite(score):
        raise GraphBackendError("Neo4j returned a non-finite graph score")
    if not all(_node_in_scope(node, request.access) for node in nodes):
        raise GraphBackendError("Neo4j returned an out-of-scope graph node")
    if not all(
        _relationship_in_scope(relationship, request.access)
        for relationship in relationships
    ):
        raise GraphBackendError("Neo4j returned an out-of-scope graph relationship")

    node_provenance = [
        GraphNodeProvenance(
            entity_id=node.entity_id,
            source_uri=node.source_uri,
            chunk_id=node.chunk_id,
            knowledge_base_id=node.knowledge_base_id,
            version=node.version,
        )
        for node in nodes
    ]
    relationship_provenance = [
        GraphRelationshipProvenance(
            relationship_id=relationship.relationship_id,
            relationship_type=relationship.relationship_type,
            source_uri=relationship.source_uri,
            knowledge_base_id=relationship.knowledge_base_id,
            version=relationship.version,
        )
        for relationship in relationships
    ]
    target = nodes[-1]
    identity_payload = json.dumps(
        {
            "nodes": [node.entity_id for node in nodes],
            "relationships": [item.relationship_id for item in relationships],
            "version": target.version,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return GraphEvidence(
        evidence_id=hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
        entity_id=target.entity_id,
        chunk_id=target.chunk_id,
        text=target.text,
        source_uri=target.source_uri,
        knowledge_base_id=target.knowledge_base_id,
        version=target.version,
        hop_count=hop_count,  # type: ignore[arg-type]
        backend_score=score,
        path=GraphPathProvenance(
            hop_count=hop_count,  # type: ignore[arg-type]
            nodes=node_provenance,
            relationships=relationship_provenance,
        ),
    )


class Neo4jGraphRetriever:
    """Execute static, gated traversal queries through an injectable driver."""

    def __init__(
        self,
        driver: Any,
        *,
        gate: GraphRetrievalGate,
        database: str = "neo4j",
        owns_driver: bool = False,
    ) -> None:
        if driver is None:
            raise ValueError("driver must not be None")
        self._driver = driver
        self.gate = gate
        self.database = _required(database, "database")
        self._owns_driver = bool(owns_driver)
        self._closed = False

    @classmethod
    def connect(
        cls,
        uri: str,
        *,
        auth: tuple[str, str],
        gate: GraphRetrievalGate,
        database: str = "neo4j",
    ) -> Neo4jGraphRetriever:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise Neo4jUnavailableError(
                "Neo4j graph retrieval requires the 'graph' optional dependency"
            ) from exc
        driver = GraphDatabase.driver(_required(uri, "uri"), auth=auth)
        return cls(
            driver,
            gate=gate,
            database=database,
            owns_driver=True,
        )

    def search(
        self,
        request: GraphSearchRequest,
        *,
        evaluation: GraphEvaluationEvidence | None = None,
    ) -> GraphSearchResponse:
        if self._closed:
            raise GraphBackendError("graph retriever is closed")
        decision = self.gate.evaluate(evaluation)
        scope = AppliedGraphScope.from_request(request)
        if not decision.enabled:
            return GraphSearchResponse(
                gate=decision,
                query_executed=False,
                scope=scope,
                hits=[],
            )

        query = _query_for_hops(request.max_hops)
        parameters = _parameters(request, self.gate.config.graph_result_limit)
        try:
            with self._driver.session(database=self.database) as session:
                records = list(session.run(query, parameters))
        except Exception as exc:
            raise GraphBackendError(
                f"Neo4j traversal failed: {type(exc).__name__}"
            ) from exc
        if len(records) > parameters["limit"]:
            raise GraphBackendError("Neo4j exceeded the host result budget")
        hits = [_parse_evidence(_mapping(record), request) for record in records]
        # A backend must not create duplicate evidence paths to consume budget.
        evidence_ids = [item.evidence_id for item in hits]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise GraphBackendError("Neo4j returned duplicate graph evidence paths")
        return GraphSearchResponse(
            gate=decision,
            query_executed=True,
            scope=scope,
            hits=hits,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_driver:
            try:
                self._driver.close()
            except Exception as exc:
                raise GraphBackendError(
                    f"Neo4j driver close failed: {type(exc).__name__}"
                ) from exc

    def __enter__(self) -> Neo4jGraphRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


FusionSource = Literal["hybrid", "graph"]


class FusionContribution(StrictModel):
    source: FusionSource
    source_rank: int = Field(ge=1)
    source_score: float
    weight: float = Field(gt=0.0)
    reciprocal_rank_contribution: float = Field(gt=0.0)
    explanation: dict[str, Any]

    @field_validator("source_score", "weight", "reciprocal_rank_contribution")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fusion scores must be finite")
        return float(value)


class FusedRetrievalHit(StrictModel):
    retrieval_id: str
    rank: int = Field(ge=1)
    score: float
    text: str
    source_uri: str
    chunk_id: str | None = None
    entity_id: str | None = None
    contributions: list[FusionContribution] = Field(min_length=1)
    graph_evidence: list[GraphEvidence] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fused score must be finite")
        return float(value)


class FusedRetrievalResponse(StrictModel):
    gate: GraphGateDecision
    rrf_k: int = Field(ge=1)
    result_budget: int = Field(ge=1)
    character_budget: int = Field(ge=1)
    characters_used: int = Field(ge=0)
    graph_contributed: bool
    hits: list[FusedRetrievalHit]


def fuse_hybrid_and_graph(
    hybrid_hits: Sequence[HybridSearchHit],
    graph_response: GraphSearchResponse,
    *,
    config: GraphGateConfig,
) -> FusedRetrievalResponse:
    """Fuse ranked lists with weighted RRF, deduplication, and hard budgets."""

    tenant_id = graph_response.scope.tenant_id
    for hit in hybrid_hits:
        if hit.chunk.tenant_id != tenant_id:
            raise GraphFusionScopeError(
                "hybrid and graph results must belong to the same tenant"
            )

    entries: dict[str, dict[str, Any]] = {}

    def get_entry(
        identity: str,
        *,
        text: str,
        source_uri: str,
        chunk_id: str | None,
        entity_id: str | None,
    ) -> dict[str, Any]:
        entry = entries.get(identity)
        if entry is None:
            entry = {
                "retrieval_id": identity,
                "score": 0.0,
                "text": text,
                "source_uri": source_uri,
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "contributions": [],
                "graph_evidence": [],
            }
            entries[identity] = entry
        return entry

    for source_rank, hit in enumerate(hybrid_hits, start=1):
        identity = f"chunk:{tenant_id}:{hit.chunk.chunk_id}"
        entry = get_entry(
            identity,
            text=hit.chunk.text,
            source_uri=hit.chunk.source_uri,
            chunk_id=hit.chunk.chunk_id,
            entity_id=None,
        )
        contribution = config.hybrid_weight / (config.rrf_k + source_rank)
        entry["score"] += contribution
        entry["contributions"].append(
            FusionContribution(
                source="hybrid",
                source_rank=source_rank,
                source_score=hit.score,
                weight=config.hybrid_weight,
                reciprocal_rank_contribution=contribution,
                explanation={
                    "retrieval_sources": list(hit.retrieval_sources),
                    "hybrid_rank": hit.rank,
                },
            )
        )

    if graph_response.gate.enabled:
        for source_rank, evidence in enumerate(graph_response.hits, start=1):
            identity = (
                f"chunk:{tenant_id}:{evidence.chunk_id}"
                if evidence.chunk_id is not None
                else f"entity:{tenant_id}:{evidence.entity_id}"
            )
            entry = get_entry(
                identity,
                text=evidence.text,
                source_uri=evidence.source_uri,
                chunk_id=evidence.chunk_id,
                entity_id=evidence.entity_id,
            )
            if entry["entity_id"] is None:
                entry["entity_id"] = evidence.entity_id
            contribution = config.graph_weight / (config.rrf_k + source_rank)
            entry["score"] += contribution
            entry["graph_evidence"].append(evidence)
            entry["contributions"].append(
                FusionContribution(
                    source="graph",
                    source_rank=source_rank,
                    source_score=evidence.backend_score,
                    weight=config.graph_weight,
                    reciprocal_rank_contribution=contribution,
                    explanation={
                        "evidence_id": evidence.evidence_id,
                        "hop_count": evidence.hop_count,
                        "relationship_types": [
                            item.relationship_type
                            for item in evidence.path.relationships
                        ],
                    },
                )
            )

    ordered = sorted(entries.values(), key=lambda item: (-item["score"], item["retrieval_id"]))
    selected: list[FusedRetrievalHit] = []
    characters_used = 0
    for entry in ordered:
        if len(selected) >= config.fused_result_budget:
            break
        text_length = len(entry["text"])
        if characters_used + text_length > config.fused_character_budget:
            continue
        characters_used += text_length
        selected.append(
            FusedRetrievalHit(
                retrieval_id=entry["retrieval_id"],
                rank=len(selected) + 1,
                score=entry["score"],
                text=entry["text"],
                source_uri=entry["source_uri"],
                chunk_id=entry["chunk_id"],
                entity_id=entry["entity_id"],
                contributions=entry["contributions"],
                graph_evidence=entry["graph_evidence"],
            )
        )

    return FusedRetrievalResponse(
        gate=graph_response.gate,
        rrf_k=config.rrf_k,
        result_budget=config.fused_result_budget,
        character_budget=config.fused_character_budget,
        characters_used=characters_used,
        graph_contributed=any(
            contribution.source == "graph"
            for hit in selected
            for contribution in hit.contributions
        ),
        hits=selected,
    )


__all__ = [
    "RELATIONSHIP_ALLOWLIST",
    "AppliedGraphScope",
    "FusedRetrievalHit",
    "FusedRetrievalResponse",
    "FusionContribution",
    "GraphAccess",
    "GraphBackendError",
    "GraphEvaluationEvidence",
    "GraphEvidence",
    "GraphFusionScopeError",
    "GraphGateConfig",
    "GraphGateConfigurationError",
    "GraphGateDecision",
    "GraphNodeProvenance",
    "GraphPathProvenance",
    "GraphRelationshipProvenance",
    "GraphRetrievalError",
    "GraphRetrievalGate",
    "GraphSearchRequest",
    "GraphSearchResponse",
    "Neo4jGraphRetriever",
    "Neo4jUnavailableError",
    "fuse_hybrid_and_graph",
]
