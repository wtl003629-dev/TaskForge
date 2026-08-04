from __future__ import annotations

import json
import sqlite3

import pytest

from taskforge.checkpoints import (
    CheckpointCorruptError,
    CheckpointNotFoundError,
    SQLiteCheckpointStore,
)
from taskforge.domain import AgentProfile, RunError, RunState, RunStatus, Task


def _records() -> tuple[Task, AgentProfile, RunState]:
    task = Task(tenant_id="tenant-a", user_id="user-a", goal="inspect the repository")
    profile = AgentProfile(name="Repo analyst", instructions="Use read-only evidence.", max_steps=4)
    state = RunState(
        task_id=task.id,
        agent_profile_id=profile.id,
        status=RunStatus.RUNNING,
        step_budget=profile.max_steps,
    )
    return task, profile, state


def test_sqlite_checkpoint_round_trip_survives_reopen(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = SQLiteCheckpointStore(path)
    task, profile, state = _records()
    store.save_task(task)
    store.save_profile(profile)

    assert store.save(state) == 1
    state.artifacts.append({"kind": "report", "uri": "artifact://one"})
    assert store.save(state) == 2

    reopened = SQLiteCheckpointStore(path)
    loaded = reopened.load(state.run_id)
    assert reopened.load_task(task.id) == task
    assert reopened.load_profile(profile.id) == profile
    assert reopened.version(state.run_id) == 2
    assert loaded.artifacts == [{"kind": "report", "uri": "artifact://one"}]


def test_checkpoint_load_returns_an_independent_model(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    task, profile, state = _records()
    store.save_task(task)
    store.save_profile(profile)
    store.save(state)

    first = store.load(state.run_id)
    first.artifacts.append({"mutated": True})

    assert store.load(state.run_id).artifacts == []


def test_legacy_checkpoint_without_retryable_defaults_fail_closed(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = SQLiteCheckpointStore(path)
    task, profile, state = _records()
    state.status = RunStatus.FAILED
    state.error = RunError(
        stage="provider",
        code="LegacyProviderError",
        message="legacy failure",
    )
    store.save_task(task)
    store.save_profile(profile)
    store.save(state)
    payload = state.model_dump(mode="json")
    del payload["error"]["retryable"]
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET state_json = ? WHERE run_id = ?",
            (json.dumps(payload), state.run_id),
        )

    loaded = SQLiteCheckpointStore(path).load(state.run_id)

    assert loaded.error is not None
    assert loaded.error.retryable is False


def test_checkpoint_rejects_identity_reuse(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    task, profile, state = _records()
    store.save_task(task)
    store.save_profile(profile)
    store.save(state)
    other = state.model_copy(update={"task_id": "other-task"})

    with pytest.raises(ValueError, match="another task"):
        store.save(other)


def test_checkpoint_reports_missing_and_corrupt_rows(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = SQLiteCheckpointStore(path)
    with pytest.raises(CheckpointNotFoundError):
        store.load("missing")

    task, profile, state = _records()
    store.save_task(task)
    store.save_profile(profile)
    store.save(state)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET state_json = ? WHERE run_id = ?",
            ('{"status":"broken"}', state.run_id),
        )

    with pytest.raises(CheckpointCorruptError):
        store.load(state.run_id)
