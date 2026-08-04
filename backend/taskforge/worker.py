"""Durable queue worker for checkpointed Agent runs."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from .checkpoints import SQLiteCheckpointStore
from .domain import AgentProfile, RunState, RunStatus, Task, utc_now
from .operations import (
    AuditEvent,
    LeaseLostError,
    OperationJob,
    OperationsStore,
    audit_usage_from_state,
    sanitize_failure,
    tool_result_is_safety_violation,
)
from .runtime import AgentRuntime


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    job: OperationJob
    state: RunState | None
    outcome: str
    error: str | None = None


class DurableWorker:
    """Claim and execute durable runs without trusting in-process ownership."""

    def __init__(
        self,
        *,
        owner: str,
        operations: OperationsStore,
        checkpoints: SQLiteCheckpointStore,
        runtime: AgentRuntime,
        lease_seconds: float = 30.0,
        heartbeat_interval: float | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if not owner:
            raise ValueError("worker owner is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        interval = heartbeat_interval
        if interval is None:
            interval = max(0.05, lease_seconds / 3)
        if interval <= 0 or interval >= lease_seconds:
            raise ValueError("heartbeat_interval must be positive and shorter than lease")
        self.owner = owner
        self.operations = operations
        self.checkpoints = checkpoints
        self.runtime = runtime
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_interval = float(interval)
        self.tenant_id = tenant_id

    async def run_once(self) -> WorkerOutcome | None:
        claimed = self.operations.claim(
            self.owner,
            lease_seconds=self.lease_seconds,
            tenant_id=self.tenant_id,
        )
        if claimed is None:
            return None

        started = time.monotonic()
        current_job = claimed
        lease: list[OperationJob] = [claimed]
        state: RunState | None = None
        profile: AgentProfile | None = None
        job_completed = False
        try:
            state = self.checkpoints.load(claimed.run_id)
            task = self.checkpoints.load_task(state.task_id)
            profile = self.checkpoints.load_profile(state.agent_profile_id)
            self._validate_identity(claimed, task, profile, state)

            terminal = {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.STEP_LIMIT,
                RunStatus.WAITING_APPROVAL,
            }
            if claimed.attempt > claimed.max_attempts and state.status not in terminal:
                raise RuntimeError("final attempt lease expired before a terminal checkpoint")

            state = await self._run_with_heartbeat(
                task,
                profile,
                state,
                lease,
            )
            current_job = lease[0]
            if state.status == RunStatus.FAILED and state.error is not None:
                if state.error.stage == "provider" and state.error.retryable:
                    error_code = state.error.code
                    if current_job.attempt < current_job.max_attempts:
                        state = RunState.model_validate(
                            {
                                **state.model_dump(),
                                "status": RunStatus.PENDING,
                                "error": None,
                                "updated_at": utc_now(),
                            }
                        )
                        self.checkpoints.save(state)
                    raise RuntimeError(
                        f"retryable provider failure: {error_code}"
                    )
            if state.status not in terminal:
                raise RuntimeError(f"runtime returned non-terminal status: {state.status.value}")

            duration_ms = (time.monotonic() - started) * 1_000
            self._append_tool_audit(claimed, profile, state)
            self.operations.append_audit_once(
                AuditEvent(
                    event_id=self._event_id(
                        claimed.run_id,
                        "run.terminal",
                        state.status.value,
                    ),
                    tenant_id=claimed.tenant_id,
                    run_id=claimed.run_id,
                    action="run.execute",
                    outcome=state.status.value,
                    duration_ms=duration_ms,
                    provider=profile.model,
                    usage=audit_usage_from_state(state),
                    metadata={
                        "steps": len(state.steps),
                    },
                )
            )
            current_job = self.operations.complete(
                current_job,
                result_status=state.status.value,
            )
            job_completed = True
            return WorkerOutcome(
                job=current_job,
                state=state,
                outcome=state.status.value,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Heartbeats may have advanced the CAS version before the runtime
            # failed; always use the newest successfully persisted lease.
            current_job = lease[0]
            safe_error = sanitize_failure(exc)
            duration_ms = (time.monotonic() - started) * 1_000
            if job_completed:
                # The CAS completion is authoritative.  Never turn an already
                # completed job back into a retry because a later observer failed.
                raise
            if isinstance(exc, LeaseLostError):
                outcome = "lease_lost"
            else:
                try:
                    current_job = self.operations.fail(current_job, exc)
                    outcome = current_job.result_status or current_job.status.value
                except LeaseLostError:
                    outcome = "lease_lost"
            self.operations.append_audit_once(
                AuditEvent(
                    event_id=self._event_id(
                        claimed.run_id,
                        "run.execute",
                        str(claimed.attempt),
                        outcome,
                    ),
                    tenant_id=claimed.tenant_id,
                    run_id=claimed.run_id,
                    action="run.execute",
                    outcome=outcome,
                    duration_ms=duration_ms,
                    provider=profile.model if profile else None,
                    metadata={
                        "worker_id_hash": self._worker_hash(),
                        "attempt": claimed.attempt,
                        "error_type": type(exc).__name__,
                        "error": safe_error,
                    },
                )
            )
            return WorkerOutcome(
                job=current_job,
                state=state,
                outcome=outcome,
                error=safe_error,
            )

    async def run_forever(
        self,
        stop: asyncio.Event,
        *,
        poll_interval: float = 0.5,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        while not stop.is_set():
            outcome = await self.run_once()
            if outcome is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_interval)
                except TimeoutError:
                    pass

    async def _run_with_heartbeat(
        self,
        task: Task,
        profile: AgentProfile,
        state: RunState,
        lease: list[OperationJob],
    ) -> RunState:
        stop = asyncio.Event()

        async def heartbeat() -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.heartbeat_interval)
                    return
                except TimeoutError:
                    lease[0] = self.operations.heartbeat(
                        lease[0],
                        lease_seconds=self.lease_seconds,
                    )

        runtime_task = asyncio.create_task(self.runtime.run(task, profile, state))
        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait(
                {runtime_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_error = heartbeat_task.exception()
                if heartbeat_error is not None:
                    runtime_task.cancel()
                    await asyncio.gather(runtime_task, return_exceptions=True)
                    raise heartbeat_error
            if runtime_task not in done:
                # A heartbeat loop only exits normally after stop is set, which
                # cannot happen before the runtime completes.
                raise RuntimeError("heartbeat stopped before runtime completion")
            result = runtime_task.result()
            stop.set()
            await heartbeat_task
            return result
        finally:
            stop.set()
            if not heartbeat_task.done():
                heartbeat_task.cancel()
            if not runtime_task.done():
                runtime_task.cancel()
            await asyncio.gather(runtime_task, heartbeat_task, return_exceptions=True)

    @staticmethod
    def _validate_identity(
        job: OperationJob,
        task: Task,
        profile: AgentProfile,
        state: RunState,
    ) -> None:
        if task.tenant_id != job.tenant_id:
            raise PermissionError("job tenant does not match checkpoint task tenant")
        if state.run_id != job.run_id:
            raise ValueError("job run_id does not match checkpoint")
        if state.task_id != task.id:
            raise ValueError("checkpoint task identity mismatch")
        if state.agent_profile_id != profile.id:
            raise ValueError("checkpoint profile identity mismatch")

    def _append_tool_audit(
        self,
        job: OperationJob,
        profile: AgentProfile,
        state: RunState,
    ) -> None:
        """Record each durable receipt once across lease-expiry recovery."""

        for step in state.steps:
            if step.model_turn is None:
                continue
            requests = {
                request.call_id: request for request in step.model_turn.tool_requests
            }
            for result in step.tool_results:
                request = requests.get(result.call_id)
                if request is None:
                    continue
                reused = any(
                    key in result.metadata
                    for key in ("reused_from_call_id", "idempotent_replay_of")
                )
                self.operations.append_audit_once(
                    AuditEvent(
                        event_id=self._event_id(
                            job.run_id,
                            "tool.receipt_reused" if reused else "tool.execute",
                            result.call_id,
                        ),
                        tenant_id=job.tenant_id,
                        run_id=job.run_id,
                        action="tool.receipt_reused" if reused else "tool.execute",
                        outcome="reused" if reused else ("success" if result.ok else "failed"),
                        tool=None if reused else request.name,
                        provider=profile.model,
                        safety_violation=(
                            False if reused else tool_result_is_safety_violation(result)
                        ),
                        metadata={"step_index": step.index},
                    )
                )

    def _worker_hash(self) -> str:
        return hashlib.sha256(self.owner.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _event_id(run_id: str, *parts: str) -> str:
        material = "\x00".join(("taskforge", run_id, *parts))
        return str(uuid5(NAMESPACE_URL, material))


__all__ = ["DurableWorker", "WorkerOutcome"]
