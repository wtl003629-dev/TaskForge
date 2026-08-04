from __future__ import annotations

from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.domain import RunState, RunStatus


def test_checkpoint_protocol_can_save_state_before_optional_task_catalog(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    state = RunState(
        task_id="task-owned-by-caller",
        agent_profile_id="profile-owned-by-caller",
        status=RunStatus.RUNNING,
        step_budget=2,
    )

    assert store.save(state) == 1
    assert store.load(state.run_id) == state
