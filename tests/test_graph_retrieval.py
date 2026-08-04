from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from taskforge.graph_retrieval import (
    RELATIONSHIP_ALLOWLIST,
    GraphAccess,
    GraphBackendError,
    GraphEvaluationEvidence,
    GraphFusionScopeError,
    GraphGateConfig,
    GraphRetrievalGate,
    GraphSearchRequest,
    Neo4jGraphRetriever,
    Neo4jUnavailableError,
    fuse_hybrid_and_graph,
)
from taskforge.hybrid_retrieval import HybridChunk, HybridSearchHit

NOW = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
LOCKED_SHA = "a" * 64


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> FakeSession:
        self.driver.sessions_entered += 1
        return self

    def __exit__(self, *_: object) -> None:
        self.driver.sessions_exited += 1

    def run(self, query: str, parameters: dict[str, object]) -> list[dict[str, object]]:
        self.driver.calls.append((query, parameters))
        if self.driver.run_error is not None:
            raise self.driver.run_error
        return list(self.driver.records)


class FakeDriver:
    def __init__(
        self,
        records: list[dict[str, object]] | None = None,
        *,
        run_error: Exception | None = None,
    ) -> None:
        self.records = records or []
        self.run_error = run_error
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.session_databases: list[str] = []
        self.sessions_entered = 0
        self.sessions_exited = 0
        self.close_calls = 0

    def session(self, *, database: str) -> FakeSession:
        self.session_databases.append(database)
        return FakeSession(self)

    def close(self) -> None:
        self.close_calls += 1


def access(**updates: object) -> GraphAccess:
    values: dict[str, object] = {
        "tenant_id": "tenant-a",
        "user_id": "alice",
        "acl_principals": {"user:alice", "role:reviewer"},
        "knowledge_base_ids": {"governance"},
        "versions": {"v2"},
        "as_of": NOW,
    }
    values.update(updates)
    return GraphAccess.model_validate(values)


def enabled_config(**updates: object) -> GraphGateConfig:
    values: dict[str, object] = {
        "enabled": True,
        "locked_evaluation_id": "graph-locked-v1",
        "locked_evaluation_sha256": LOCKED_SHA,
        "metric_name": "recall_at_5",
        "minimum_improvement": 0.03,
        "graph_result_limit": 10,
    }
    values.update(updates)
    return GraphGateConfig.model_validate(values)


def evaluation(**updates: object) -> GraphEvaluationEvidence:
    values: dict[str, object] = {
        "evaluation_id": "graph-locked-v1",
        "evaluation_sha256": LOCKED_SHA,
        "metric_name": "recall_at_5",
        "baseline_score": 0.60,
        "graph_score": 0.64,
        "safety_regressions": 0,
    }
    values.update(updates)
    return GraphEvaluationEvidence.model_validate(values)


def node(
    entity_id: str,
    *,
    chunk_id: str | None,
    text: str,
    tenant_id: str = "tenant-a",
    users: list[str] | None = None,
    acl: list[str] | None = None,
    knowledge_base_id: str = "governance",
    version: str = "v2",
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "tenant_id": tenant_id,
        "allowed_user_ids": users or ["alice"],
        "acl_principals": acl or ["user:alice"],
        "knowledge_base_id": knowledge_base_id,
        "version": version,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "source_uri": f"kb://{entity_id}",
        "chunk_id": chunk_id,
        "text": text,
    }


def relationship(
    relationship_id: str = "r1",
    *,
    relationship_type: str = "SUPPORTS",
    tenant_id: str = "tenant-a",
    users: list[str] | None = None,
    acl: list[str] | None = None,
    knowledge_base_id: str = "governance",
    version: str = "v2",
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict[str, object]:
    return {
        "relationship_id": relationship_id,
        "relationship_type": relationship_type,
        "tenant_id": tenant_id,
        "allowed_user_ids": users or ["alice"],
        "acl_principals": acl or ["user:alice"],
        "knowledge_base_id": knowledge_base_id,
        "version": version,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "source_uri": f"kb://relationship/{relationship_id}",
    }


def one_hop_record(
    *,
    start: dict[str, object] | None = None,
    target: dict[str, object] | None = None,
    edge: dict[str, object] | None = None,
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "nodes": [
            start or node("policy", chunk_id="seed", text="Policy root"),
            target or node("control", chunk_id="shared", text="Control evidence"),
        ],
        "relationships": [edge or relationship()],
        "hop_count": 1,
        "score": score,
    }


def search(
    driver: FakeDriver,
    *,
    config: GraphGateConfig | None = None,
    request: GraphSearchRequest | None = None,
    eval_result: GraphEvaluationEvidence | None = None,
):
    retriever = Neo4jGraphRetriever(
        driver,
        gate=GraphRetrievalGate(config or enabled_config()),
    )
    return retriever.search(
        request or GraphSearchRequest(entity_id="policy", access=access()),
        evaluation=eval_result or evaluation(),
    )


def test_repository_gate_file_is_disabled_and_driver_is_not_touched() -> None:
    config_path = Path(__file__).parents[1] / "config" / "graph_retrieval_gate.json"
    config = GraphGateConfig.load(config_path)
    driver = FakeDriver([one_hop_record()])

    response = search(driver, config=config)

    assert config.enabled is False
    assert response.gate.reason == "config_disabled"
    assert response.query_executed is False
    assert response.hits == []
    assert driver.calls == []
    assert driver.session_databases == []


@pytest.mark.parametrize(
    ("eval_result", "reason"),
    [
        (None, "evaluation_missing"),
        (evaluation(evaluation_id="other"), "evaluation_identity_mismatch"),
        (evaluation(evaluation_sha256="b" * 64), "evaluation_hash_mismatch"),
        (evaluation(metric_name="mrr_at_5"), "metric_mismatch"),
        (
            evaluation(graph_score=0.90, safety_regressions=1),
            "safety_regression",
        ),
        (evaluation(graph_score=0.629), "improvement_below_threshold"),
    ],
)
def test_locked_evaluation_must_pass_quality_and_safety(
    eval_result: GraphEvaluationEvidence | None,
    reason: str,
) -> None:
    driver = FakeDriver([one_hop_record()])
    retriever = Neo4jGraphRetriever(
        driver,
        gate=GraphRetrievalGate(enabled_config()),
    )

    response = retriever.search(
        GraphSearchRequest(entity_id="policy", access=access()),
        evaluation=eval_result,
    )

    assert response.gate.reason == reason
    assert not response.gate.enabled
    assert not response.query_executed
    assert driver.calls == []


def test_query_is_static_parameterized_and_provenance_is_preserved() -> None:
    malicious = "policy'}) MATCH (stolen) DETACH DELETE stolen //"
    driver = FakeDriver([one_hop_record(start=node(malicious, chunk_id="seed", text="root"))])

    response = search(
        driver,
        request=GraphSearchRequest(
            entity_id=malicious,
            access=access(),
            max_hops=2,
            limit=7,
        ),
    )

    assert response.query_executed
    assert len(response.hits) == 1
    query, parameters = driver.calls[0]
    assert malicious not in query
    assert parameters["entity_id"] == malicious
    assert parameters["tenant_id"] == "tenant-a"
    assert parameters["user_id"] == "alice"
    assert parameters["relationship_allowlist"] == list(RELATIONSHIP_ALLOWLIST)
    assert parameters["limit"] == 7
    assert "(seed:KnowledgeEntity {entity_id: $entity_id})" in query
    assert "[*1..2]" in query
    assert "$acl_principals" in query
    assert "$knowledge_base_ids" in query
    assert "$versions" in query
    assert "$as_of" in query
    assert "type(rel) IN $relationship_allowlist" in query
    hit = response.hits[0]
    assert hit.chunk_id == "shared"
    assert hit.path.nodes[0].entity_id == malicious
    assert hit.path.relationships[0].relationship_type == "SUPPORTS"
    assert hit.evidence_id
    assert response.scope.acl_principal_count == 2
    assert len(response.scope.acl_principals_sha256) == 64


@pytest.mark.parametrize(
    "record",
    [
        one_hop_record(target=node("control", chunk_id="x", text="x", tenant_id="tenant-b")),
        one_hop_record(target=node("control", chunk_id="x", text="x", users=["bob"])),
        one_hop_record(target=node("control", chunk_id="x", text="x", acl=["user:bob"])),
        one_hop_record(
            target=node("control", chunk_id="x", text="x", knowledge_base_id="secret")
        ),
        one_hop_record(target=node("control", chunk_id="x", text="x", version="v1")),
        one_hop_record(
            target=node(
                "control",
                chunk_id="x",
                text="x",
                valid_from=(NOW + timedelta(seconds=1)).isoformat(),
            )
        ),
        one_hop_record(
            target=node(
                "control",
                chunk_id="x",
                text="x",
                valid_until=NOW.isoformat(),
            )
        ),
        one_hop_record(edge=relationship(relationship_type="MODEL_CHOSEN_EDGE")),
        one_hop_record(edge=relationship(users=["bob"])),
    ],
)
def test_backend_scope_violations_fail_the_entire_query_closed(
    record: dict[str, object],
) -> None:
    driver = FakeDriver([record])

    with pytest.raises(GraphBackendError, match="out-of-scope"):
        search(driver)

    assert driver.sessions_entered == 1
    assert driver.sessions_exited == 1


def test_depth_and_request_surface_are_closed_to_model_supplied_query_parts() -> None:
    with pytest.raises(ValidationError):
        GraphSearchRequest.model_validate(
            {
                "entity_id": "policy",
                "access": access(),
                "max_hops": 3,
            }
        )
    with pytest.raises(ValidationError):
        GraphSearchRequest.model_validate(
            {
                "entity_id": "policy",
                "access": access(),
                "cypher": "MATCH (n) RETURN n",
                "relationship_types": ["OWNS_EVERYTHING"],
                "label": "Admin",
            }
        )

    driver = FakeDriver([one_hop_record()])
    search(
        driver,
        request=GraphSearchRequest(entity_id="policy", access=access(), max_hops=1),
    )
    query, _ = driver.calls[0]
    assert "[*1..1]" in query
    assert "[*1..2]" not in query


def test_invalid_driver_result_and_driver_error_are_sanitized() -> None:
    invalid = one_hop_record()
    invalid["hop_count"] = 2
    with pytest.raises(GraphBackendError, match="inconsistent path shape"):
        search(FakeDriver([invalid]))

    with pytest.raises(GraphBackendError, match="RuntimeError") as caught:
        search(FakeDriver(run_error=RuntimeError("secret-password")))
    assert "secret-password" not in str(caught.value)


def test_injected_driver_is_caller_owned_and_owned_driver_closes_once() -> None:
    caller_driver = FakeDriver()
    caller_owned = Neo4jGraphRetriever(
        caller_driver,
        gate=GraphRetrievalGate(GraphGateConfig()),
    )
    caller_owned.close()
    caller_owned.close()
    assert caller_driver.close_calls == 0

    managed_driver = FakeDriver()
    managed = Neo4jGraphRetriever(
        managed_driver,
        gate=GraphRetrievalGate(GraphGateConfig()),
        owns_driver=True,
    )
    managed.close()
    managed.close()
    assert managed_driver.close_calls == 1
    with pytest.raises(GraphBackendError, match="closed"):
        managed.search(
            GraphSearchRequest(entity_id="policy", access=access()),
            evaluation=evaluation(),
        )


def test_missing_optional_neo4j_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "neo4j", None)
    with pytest.raises(Neo4jUnavailableError, match="optional dependency"):
        Neo4jGraphRetriever.connect(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
            gate=GraphRetrievalGate(GraphGateConfig()),
        )


def hybrid_hit(
    chunk_id: str,
    text: str,
    *,
    tenant_id: str = "tenant-a",
    rank: int = 1,
) -> HybridSearchHit:
    chunk = HybridChunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        text=text,
        source_uri=f"kb://{chunk_id}",
        document_id="doc",
        knowledge_base_id="governance",
        version="v2",
        version_order=2,
        acl_principals={"user:alice"},
    )
    return HybridSearchHit(
        chunk=chunk,
        rank=rank,
        score=1.0 / rank,
        base_score=1.0 / rank,
        retrieval_sources=["python_bm25"],
    )


def test_weighted_rrf_fusion_deduplicates_and_keeps_both_explanations() -> None:
    driver = FakeDriver(
        [
            one_hop_record(),
            one_hop_record(
                target=node(
                    "independent-control",
                    chunk_id=None,
                    text="Independent graph evidence",
                ),
                edge=relationship("r2", relationship_type="REFERENCES"),
                score=0.8,
            ),
        ]
    )
    graph_response = search(driver)
    config = enabled_config(
        fused_result_budget=2,
        fused_character_budget=100,
        hybrid_weight=1.0,
        graph_weight=2.0,
        rrf_k=10,
    )

    fused = fuse_hybrid_and_graph(
        [hybrid_hit("shared", "Hybrid authoritative text")],
        graph_response,
        config=config,
    )

    assert len(fused.hits) == 2
    shared = next(hit for hit in fused.hits if hit.chunk_id == "shared")
    assert shared.text == "Hybrid authoritative text"
    assert {item.source for item in shared.contributions} == {"hybrid", "graph"}
    assert shared.graph_evidence[0].path.relationships[0].relationship_type == "SUPPORTS"
    assert fused.graph_contributed
    assert fused.characters_used == sum(len(hit.text) for hit in fused.hits)


def test_fusion_respects_closed_gate_scope_and_hard_budgets() -> None:
    disabled_response = search(FakeDriver(), config=GraphGateConfig())
    config = GraphGateConfig(
        fused_result_budget=1,
        fused_character_budget=len("short"),
    )
    fused = fuse_hybrid_and_graph(
        [
            hybrid_hit("too-long", "longer than budget", rank=1),
            hybrid_hit("fits", "short", rank=2),
        ],
        disabled_response,
        config=config,
    )

    assert [hit.chunk_id for hit in fused.hits] == ["fits"]
    assert fused.characters_used == len("short")
    assert not fused.graph_contributed

    with pytest.raises(GraphFusionScopeError, match="same tenant"):
        fuse_hybrid_and_graph(
            [hybrid_hit("wrong", "x", tenant_id="tenant-b")],
            disabled_response,
            config=config,
        )


def test_gate_cannot_be_enabled_without_a_locked_artifact() -> None:
    with pytest.raises(ValidationError, match="locked evaluation"):
        GraphGateConfig(enabled=True)

