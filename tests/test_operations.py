from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.context import ContextAssembler
from taskforge.domain import AgentProfile, ModelTurn, RunState, RunStatus, Task
from taskforge.operations import (
    AuditSecretError,
    AuditUsage,
    DuplicateJobError,
    JobStatus,
    LeaseLostError,
    OperationsError,
    OperationsStore,
)
from taskforge.providers import ProviderError, ScriptedProvider
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy, ToolRegistry
from taskforge.worker import DurableWorker

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


def test_two_worker_connections_atomically_claim_only_once(tmp_path: Path) -> None:
    path = tmp_path / "operations.db"
    first_store = OperationsStore(path)
    second_store = OperationsStore(path)
    first_store.enqueue("run-1", "tenant-a", now=NOW)
    barrier = Barrier(2)

    def compete(store: OperationsStore, owner: str):
        barrier.wait(timeout=5)
        return store.claim(owner, lease_seconds=30, now=NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(compete, first_store, "worker-a")
        second = pool.submit(compete, second_store, "worker-b")
        results = [first.result(timeout=5), second.result(timeout=5)]

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    assert claimed[0].attempt == 1
    assert claimed[0].lease_version == 1
    assert first_store.get_job("run-1").owner == claimed[0].owner
    with pytest.raises(DuplicateJobError):
        second_store.enqueue("run-1", "tenant-a", now=NOW)


def test_expired_lease_reclaim_and_every_mutation_uses_token_version_cas(
    tmp_path: Path,
) -> None:
    store = OperationsStore(tmp_path / "operations.db")
    store.enqueue("run-1", "tenant-a", max_attempts=3, now=NOW)
    stale = store.claim("worker-a", lease_seconds=10, now=NOW)
    assert stale is not None
    assert store.claim("worker-b", lease_seconds=10, now=NOW + timedelta(seconds=9)) is None

    reclaimed = store.claim(
        "worker-b",
        lease_seconds=10,
        now=NOW + timedelta(seconds=10),
    )
    assert reclaimed is not None
    assert reclaimed.owner == "worker-b"
    assert reclaimed.attempt == 2
    assert reclaimed.lease_version == stale.lease_version + 1
    assert reclaimed.lease_token != stale.lease_token

    for operation in (
        lambda: store.heartbeat(stale, now=NOW + timedelta(seconds=10)),
        lambda: store.complete(
            stale,
            result_status="completed",
            now=NOW + timedelta(seconds=10),
        ),
        lambda: store.fail(stale, "old worker", now=NOW + timedelta(seconds=10)),
    ):
        with pytest.raises(LeaseLostError):
            operation()

    wrong_owner = reclaimed.model_copy(update={"owner": "intruder"})
    with pytest.raises(LeaseLostError):
        store.heartbeat(wrong_owner, now=NOW + timedelta(seconds=11))

    renewed = store.heartbeat(
        reclaimed,
        lease_seconds=20,
        now=NOW + timedelta(seconds=11),
    )
    assert renewed.lease_version == reclaimed.lease_version + 1
    with pytest.raises(LeaseLostError):
        store.complete(
            reclaimed,
            result_status="completed",
            now=NOW + timedelta(seconds=12),
        )
    completed = store.complete(
        renewed,
        result_status="completed",
        now=NOW + timedelta(seconds=12),
    )
    assert completed.status == JobStatus.COMPLETED
    assert completed.owner is None and completed.lease_token is None


def test_retry_backoff_dead_letter_and_sanitized_failure(tmp_path: Path) -> None:
    store = OperationsStore(
        tmp_path / "operations.db",
        base_backoff_seconds=10,
        max_backoff_seconds=60,
    )
    store.enqueue("run-1", "tenant-a", max_attempts=2, now=NOW)
    first = store.claim("worker", lease_seconds=30, now=NOW)
    assert first is not None
    queued = store.fail(
        first,
        "provider password=hunter2 and sk-abcdef123456",
        now=NOW + timedelta(seconds=1),
    )
    assert queued.status == JobStatus.QUEUED
    assert queued.result_status == "retry_scheduled"
    assert "hunter2" not in (queued.last_error or "")
    assert "sk-abcdef" not in (queued.last_error or "")
    assert "REDACTED" in (queued.last_error or "")
    assert store.claim("worker", now=NOW + timedelta(seconds=10)) is None

    second = store.claim("worker", now=NOW + timedelta(seconds=11))
    assert second is not None and second.attempt == 2
    dead = store.fail(second, RuntimeError("still unavailable"), now=NOW + timedelta(seconds=12))
    assert dead.status == JobStatus.DEAD_LETTER
    assert dead.result_status == "failed"
    assert store.claim("worker", now=NOW + timedelta(days=1)) is None


def test_reopened_store_recovers_expired_inflight_job(tmp_path: Path) -> None:
    path = tmp_path / "operations.db"
    original = OperationsStore(path)
    original.enqueue("run-1", "tenant-a", now=NOW)
    abandoned = original.claim("old-process", lease_seconds=5, now=NOW)
    assert abandoned is not None

    reopened = OperationsStore(path)
    loaded = reopened.get_job("run-1", tenant_id="tenant-a")
    assert loaded.lease_token == abandoned.lease_token
    assert reopened.claim("new-process", now=NOW + timedelta(seconds=4)) is None
    recovered = reopened.claim("new-process", now=NOW + timedelta(seconds=5))
    assert recovered is not None
    assert recovered.owner == "new-process"
    assert recovered.attempt == 2


def test_audit_is_append_only_tenant_scoped_secret_safe_and_measurable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.db"
    store = OperationsStore(path)
    events = [
        {
            "event_id": "a-run-failed",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "action": "run.execute",
            "outcome": "failed",
            "duration_ms": 100,
            "usage": AuditUsage(input_tokens=10, output_tokens=2, total_tokens=12, cost_usd=0.1),
            "occurred_at": NOW,
        },
        {
            "event_id": "a-tool-ok",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "action": "tool.execute",
            "outcome": "ok",
            "duration_ms": 20,
            "tool": "search",
            "usage": {"input_tokens": 3, "total_tokens": 3, "cost_usd": 0.02},
            "occurred_at": NOW + timedelta(seconds=1),
        },
        {
            "event_id": "a-tool-error",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "action": "tool.execute",
            "outcome": "error",
            "duration_ms": 40,
            "tool": "write",
            "safety_violation": True,
            "occurred_at": NOW + timedelta(seconds=2),
        },
        {
            "event_id": "a-run-complete",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "action": "run.execute",
            "outcome": "completed",
            "duration_ms": 300,
            "provider": "model-a",
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cost_usd": 0.2},
            "occurred_at": NOW + timedelta(seconds=3),
        },
        {
            "event_id": "b-run-complete",
            "tenant_id": "tenant-b",
            "run_id": "run-b",
            "action": "run.execute",
            "outcome": "completed",
            "duration_ms": 1,
            "occurred_at": NOW,
        },
    ]
    for event in events:
        store.append_audit(event)

    assert [event.event_id for event in store.list_audit("tenant-a")] == [
        "a-run-failed",
        "a-tool-ok",
        "a-tool-error",
        "a-run-complete",
    ]
    assert store.list_audit("tenant-a", run_id="run-b") == []
    assert [event.event_id for event in store.list_audit("tenant-b")] == [
        "b-run-complete"
    ]
    assert [
        event.event_id
        for event in store.list_audit("tenant-a", limit=2, latest=True)
    ] == ["a-tool-error", "a-run-complete"]

    with pytest.raises(AuditSecretError):
        store.append_audit(
            {
                "tenant_id": "tenant-a",
                "run_id": "run-a",
                "action": "test",
                "outcome": "rejected",
                "metadata": {"api_key": "abc"},
            }
        )
    with pytest.raises(AuditSecretError):
        store.append_audit(
            {
                "tenant_id": "tenant-a",
                "run_id": "run-a",
                "action": "test",
                "outcome": "rejected",
                "metadata": {"note": "Bearer abcdefghijklmnop"},
            }
        )
    with pytest.raises(AuditSecretError):
        store.append_audit(
            {
                "tenant_id": "tenant-a",
                "run_id": "run-a",
                "action": "test",
                "outcome": "rejected",
                "metadata": {"token": "opaque"},
            }
        )
    with pytest.raises(ValueError, match="finite JSON"):
        store.append_audit(
            {
                "tenant_id": "tenant-a",
                "run_id": "run-a",
                "action": "test",
                "outcome": "rejected",
                "metadata": {"opaque_object": object()},
            }
        )

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE audit_events SET outcome = 'changed' WHERE event_id = ?",
                ("a-run-complete",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM audit_events WHERE event_id = ?",
                ("a-run-complete",),
            )

    metrics = store.metrics("tenant-a")
    assert metrics.run_count == 1
    assert metrics.run_success_count == 1
    assert metrics.run_success_rate == 1
    assert metrics.tool_count == 2
    assert metrics.tool_success_count == 1
    assert metrics.tool_success_rate == 0.5
    assert metrics.duration_p50_ms == 70
    assert metrics.duration_p95_ms == pytest.approx(270)
    assert metrics.input_tokens == 33
    assert metrics.output_tokens == 7
    assert metrics.total_tokens == 40
    assert metrics.cost_usd == pytest.approx(0.32)
    assert metrics.safety_violation_count == 1
    assert store.metrics("tenant-b").safety_violation_count == 0

    duplicate = store.append_audit_once(events[1])
    assert duplicate.event_id == "a-tool-ok"
    assert len(store.list_audit("tenant-a")) == 4
    with pytest.raises(OperationsError, match="collision"):
        store.append_audit_once(
            {
                **events[1],
                "tenant_id": "tenant-b",
            }
        )
    with pytest.raises(OperationsError, match="collision"):
        store.append_audit_once(
            {
                **events[1],
                "outcome": "failed",
                "safety_violation": True,
            }
        )


def _checkpointed_run(path: Path) -> tuple[SQLiteCheckpointStore, Task, AgentProfile, RunState]:
    checkpoints = SQLiteCheckpointStore(path)
    task = Task(id="task-1", tenant_id="tenant-a", user_id="user-a", goal="finish")
    profile = AgentProfile(
        id="profile-1",
        name="worker-test",
        instructions="Finish deterministically.",
        max_steps=2,
    )
    state = RunState(
        run_id="run-1",
        task_id=task.id,
        agent_profile_id=profile.id,
        status=RunStatus.PENDING,
        step_budget=profile.max_steps,
    )
    checkpoints.save_task(task)
    checkpoints.save_profile(profile)
    checkpoints.save(state)
    return checkpoints, task, profile, state


@pytest.mark.asyncio
async def test_worker_loads_checkpoint_executes_and_completes_job(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path)
    operations.enqueue("run-1", "tenant-a")
    registry = ToolRegistry()
    provider = ScriptedProvider(
        [
            ModelTurn(
                kind="final",
                final_answer="done",
                metadata={
                    "usage": {
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "total_tokens": 10,
                        "cost_usd": 0.04,
                    }
                },
            )
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoints,
        context=ContextAssembler(),
    )
    worker = DurableWorker(
        owner="worker-1",
        operations=operations,
        checkpoints=checkpoints,
        runtime=runtime,
        lease_seconds=5,
        heartbeat_interval=1,
    )

    result = await worker.run_once()

    assert result is not None
    assert result.outcome == RunStatus.COMPLETED.value
    assert result.job.status == JobStatus.COMPLETED
    assert result.job.result_status == RunStatus.COMPLETED.value
    assert result.state is not None and result.state.final_answer == "done"
    assert checkpoints.load("run-1").status == RunStatus.COMPLETED
    audit = operations.list_audit("tenant-a", run_id="run-1")
    assert len(audit) == 1
    assert audit[0].provider == "scripted"
    assert audit[0].usage is not None and audit[0].usage.total_tokens == 10
    assert operations.metrics("tenant-a").run_success_rate == 1
    assert await worker.run_once() is None


@pytest.mark.asyncio
async def test_worker_retries_provider_failure_from_a_reopened_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=3)
    provider = ScriptedProvider(
        [
            TimeoutError("temporary provider timeout"),
            ModelTurn(kind="final", final_answer="recovered"),
        ]
    )
    registry = ToolRegistry()
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoints,
        context=ContextAssembler(),
    )
    worker = DurableWorker(
        owner="retry-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=runtime,
        lease_seconds=5,
        heartbeat_interval=1,
    )

    first = await worker.run_once()
    assert first is not None and first.outcome == "retry_scheduled"
    assert first.job.status == JobStatus.QUEUED
    assert checkpoints.load("run-1").status == RunStatus.PENDING

    second = await worker.run_once()
    assert second is not None and second.outcome == RunStatus.COMPLETED.value
    assert second.job.status == JobStatus.COMPLETED
    assert second.state is not None and second.state.final_answer == "recovered"
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_worker_does_not_retry_non_retryable_provider_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=3)
    provider = ScriptedProvider([ProviderError("invalid provider configuration")])
    registry = ToolRegistry()
    worker = DurableWorker(
        owner="non-retry-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=checkpoints,
            context=ContextAssembler(),
        ),
        lease_seconds=5,
        heartbeat_interval=1,
    )

    result = await worker.run_once()

    assert result is not None and result.outcome == RunStatus.FAILED.value
    assert result.job.status == JobStatus.COMPLETED
    assert result.job.attempt == 1
    assert result.state is not None and result.state.error is not None
    assert result.state.error.retryable is False
    assert len(provider.calls) == 1
    assert await worker.run_once() is None


@pytest.mark.asyncio
async def test_retryable_provider_failure_dead_letters_at_attempt_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=1)
    provider = ScriptedProvider([TimeoutError("transient but exhausted")])
    registry = ToolRegistry()
    worker = DurableWorker(
        owner="final-attempt-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=checkpoints,
            context=ContextAssembler(),
        ),
        lease_seconds=5,
        heartbeat_interval=1,
    )

    result = await worker.run_once()

    assert result is not None and result.outcome == "failed"
    assert result.job.status == JobStatus.DEAD_LETTER
    assert result.job.attempt == 1
    assert len(provider.calls) == 1
    assert await worker.run_once() is None


@pytest.mark.asyncio
async def test_terminal_reconciliation_does_not_double_count_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=3)
    provider = ScriptedProvider(
        [
            ModelTurn(
                kind="final",
                final_answer="durable",
                metadata={"usage": {"total_tokens": 10}},
            )
        ]
    )
    registry = ToolRegistry()
    worker = DurableWorker(
        owner="billing-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=checkpoints,
            context=ContextAssembler(),
        ),
        lease_seconds=5,
        heartbeat_interval=1,
    )
    real_complete = operations.complete
    calls = 0

    def fail_complete_once(job, *, result_status, now=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("completion connection interrupted")
        return real_complete(job, result_status=result_status, now=now)

    monkeypatch.setattr(operations, "complete", fail_complete_once)
    first = await worker.run_once()
    second = await worker.run_once()

    assert first is not None and first.job.status == JobStatus.QUEUED
    assert second is not None and second.job.status == JobStatus.COMPLETED
    assert len(provider.calls) == 1
    metrics = operations.metrics("tenant-a", run_id="run-1")
    assert metrics.total_tokens == 10
    assert len(
        [event for event in operations.list_audit("tenant-a") if event.outcome == "completed"]
    ) == 1


@pytest.mark.asyncio
async def test_expired_final_lease_reconciles_a_terminal_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, pending = _checkpointed_run(path)
    operations = OperationsStore(path)
    operations.enqueue("run-1", "tenant-a", max_attempts=1)
    abandoned = operations.claim("crashed-worker", lease_seconds=0.01)
    assert abandoned is not None and abandoned.attempt == 1
    checkpoints.save(
        RunState.model_validate(
            {
                **pending.model_dump(),
                "status": RunStatus.COMPLETED,
                "final_answer": "durable before crash",
            }
        )
    )
    await asyncio.sleep(0.02)

    provider = ScriptedProvider([])
    registry = ToolRegistry()
    worker = DurableWorker(
        owner="reconcile-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=checkpoints,
            context=ContextAssembler(),
        ),
        lease_seconds=5,
        heartbeat_interval=1,
    )
    result = await worker.run_once()

    assert result is not None and result.job.status == JobStatus.COMPLETED
    assert result.job.attempt == 2
    assert result.state is not None and result.state.final_answer == "durable before crash"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_expired_final_lease_dead_letters_non_terminal_without_provider(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path)
    operations.enqueue("run-1", "tenant-a", max_attempts=1)
    assert operations.claim("crashed-worker", lease_seconds=0.01) is not None
    await asyncio.sleep(0.02)

    provider = ScriptedProvider([])
    registry = ToolRegistry()
    worker = DurableWorker(
        owner="reconcile-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=checkpoints,
            context=ContextAssembler(),
        ),
        lease_seconds=5,
        heartbeat_interval=1,
    )
    result = await worker.run_once()

    assert result is not None and result.job.status == JobStatus.DEAD_LETTER
    assert result.job.attempt == 2
    assert provider.calls == []


@pytest.mark.asyncio
async def test_worker_restart_finalizes_terminal_checkpoint_without_rerunning_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, pending = _checkpointed_run(path)
    completed = RunState.model_validate(
        {
            **pending.model_dump(),
            "status": RunStatus.COMPLETED,
            "final_answer": "already durable",
        }
    )
    checkpoints.save(completed)
    operations = OperationsStore(path)
    operations.enqueue("run-1", "tenant-a")
    abandoned = operations.claim("crashed-worker", lease_seconds=0.01)
    assert abandoned is not None
    await asyncio.sleep(0.02)

    provider = ScriptedProvider([])
    registry = ToolRegistry()
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoints,
        context=ContextAssembler(),
    )
    restarted = DurableWorker(
        owner="replacement-worker",
        operations=operations,
        checkpoints=checkpoints,
        runtime=runtime,
        lease_seconds=5,
        heartbeat_interval=1,
    )

    result = await restarted.run_once()

    assert result is not None and result.job.status == JobStatus.COMPLETED
    assert result.state is not None and result.state.final_answer == "already durable"
    assert provider.calls == []


class FailingRuntime:
    async def run(self, *_):
        raise RuntimeError("upstream password=hunter2 sk-abcdef123456")


@pytest.mark.asyncio
async def test_worker_exception_is_sanitized_and_dead_lettered(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=1)
    worker = DurableWorker(
        owner="worker-1",
        operations=operations,
        checkpoints=checkpoints,
        runtime=FailingRuntime(),  # type: ignore[arg-type]
        lease_seconds=5,
        heartbeat_interval=1,
    )

    result = await worker.run_once()

    assert result is not None
    assert result.outcome == "failed"
    assert result.job.status == JobStatus.DEAD_LETTER
    assert "hunter2" not in (result.error or "")
    assert "sk-abcdef" not in (result.job.last_error or "")
    event = operations.list_audit("tenant-a", run_id="run-1")[0]
    assert "REDACTED" in event.metadata["error"]
    assert "hunter2" not in event.metadata["error"]


class DelayedFailingRuntime:
    async def run(self, *_):
        await asyncio.sleep(0.2)
        raise RuntimeError("failed after heartbeat")


@pytest.mark.asyncio
async def test_worker_failure_after_heartbeat_uses_latest_cas_version(tmp_path: Path) -> None:
    path = tmp_path / "durable.db"
    checkpoints, _, _, _ = _checkpointed_run(path)
    operations = OperationsStore(path, base_backoff_seconds=0)
    operations.enqueue("run-1", "tenant-a", max_attempts=1)
    worker = DurableWorker(
        owner="worker-1",
        operations=operations,
        checkpoints=checkpoints,
        runtime=DelayedFailingRuntime(),  # type: ignore[arg-type]
        lease_seconds=1.0,
        heartbeat_interval=0.05,
    )

    result = await worker.run_once()

    assert result is not None
    assert result.outcome == "failed"
    assert result.job.status == JobStatus.DEAD_LETTER
    # Claim version 1, at least one heartbeat, then fail CAS increments again.
    assert result.job.lease_version >= 3
