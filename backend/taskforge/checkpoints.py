"""SQLite-backed durable state for resumable Agent runs.

The runtime writes a complete, validated :class:`RunState` snapshot after each
transition.  Tasks are stored separately so an approval can resume after an API
process restart without trusting data supplied again by the client.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from .domain import AgentProfile, RunState, Task


class CheckpointNotFoundError(KeyError):
    """Raised when a requested durable object does not exist."""


class CheckpointCorruptError(RuntimeError):
    """Raised when persisted JSON no longer satisfies the domain contract."""


class SQLiteCheckpointStore:
    """Small durable store with atomic upserts and monotonic snapshot versions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_task_id_idx ON runs(task_id);
                CREATE INDEX IF NOT EXISTS runs_updated_at_idx ON runs(updated_at DESC);
                """
            )

    def save_task(self, task: Task) -> None:
        payload = task.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(task_id, task_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET task_json = excluded.task_json
                """,
                (task.id, payload, task.created_at.isoformat()),
            )

    def save_profile(self, profile: AgentProfile) -> None:
        payload = profile.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO profiles(profile_id, profile_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(profile_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (profile.id, payload),
            )

    def save(self, state: RunState) -> int:
        """Atomically persist a snapshot and return its new version number."""

        payload = state.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, task_id, profile_id, state_json, version, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    version = runs.version + 1,
                    updated_at = excluded.updated_at
                WHERE runs.task_id = excluded.task_id
                  AND runs.profile_id = excluded.profile_id
                """,
                (
                    state.run_id,
                    state.task_id,
                    state.agent_profile_id,
                    payload,
                    state.updated_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT task_id, profile_id, version FROM runs WHERE run_id = ?",
                (state.run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("checkpoint upsert did not create a row")
            if row["task_id"] != state.task_id or row["profile_id"] != state.agent_profile_id:
                raise ValueError("run_id already belongs to another task or profile")
            return int(row["version"])

    def load(self, run_id: str) -> RunState:
        row = self._one("SELECT state_json FROM runs WHERE run_id = ?", run_id, "run")
        return self._validate(RunState, row["state_json"], f"run {run_id}")

    def load_task(self, task_id: str) -> Task:
        row = self._one("SELECT task_json FROM tasks WHERE task_id = ?", task_id, "task")
        return self._validate(Task, row["task_json"], f"task {task_id}")

    def load_profile(self, profile_id: str) -> AgentProfile:
        row = self._one(
            "SELECT profile_json FROM profiles WHERE profile_id = ?",
            profile_id,
            "profile",
        )
        return self._validate(AgentProfile, row["profile_json"], f"profile {profile_id}")

    def version(self, run_id: str) -> int:
        row = self._one("SELECT version FROM runs WHERE run_id = ?", run_id, "run")
        return int(row["version"])

    def _one(self, query: str, identifier: str, kind: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(query, (identifier,)).fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"{kind} not found: {identifier}")
        return row

    @staticmethod
    def _validate(model: type[RunState | Task | AgentProfile], payload: str, label: str):
        try:
            return model.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise CheckpointCorruptError(f"invalid persisted {label}") from exc


__all__ = [
    "CheckpointCorruptError",
    "CheckpointNotFoundError",
    "SQLiteCheckpointStore",
]
