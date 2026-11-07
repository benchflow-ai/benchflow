"""Collection is not solver time."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import _review


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_requested", [False, True])
async def test_all_final_collections_stop_writers_before_the_copy(
    monkeypatch, capture_requested
):
    """Guards 12f002dc: a late writer cannot finish during a slow copy."""
    writer_alive = True

    async def stop(*args):
        nonlocal writer_alive
        writer_alive = False

    trial = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=object() if capture_requested else None,
        _config=SimpleNamespace(
            purpose="task", sandbox_user="agent", freeze_workspace=False
        ),
        _planes=SimpleNamespace(quiesce_agent=stop),
        _env=object(),
        _task=object(),
        disconnect=AsyncMock(),
    )
    monkeypatch.setattr(_review, "_freeze_requested", lambda _: False)
    # The task keeps no services (K2), so the kill is the plain one.
    monkeypatch.setattr(_review.keep_services, "trial_keep", lambda *_: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _: "no recovery contract",
    )
    # Evidence is optional and can fail. Quiescence is not optional.
    await _review.capture_terminal_workspace(trial)
    assert not writer_alive, "the detached writer still has the copy's time budget"


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_requested", [False, True])
async def test_failed_quiescence_cannot_admit_artifacts_or_verification(
    monkeypatch, capture_requested
):
    """Guards 12f002dc, including evidence and collection-timeout paths."""
    trial = SimpleNamespace(
        _branch_child_active=False,
        _review_plan=object() if capture_requested else None,
        _config=SimpleNamespace(purpose="task", sandbox_user="agent"),
        _planes=SimpleNamespace(
            quiesce_agent=AsyncMock(side_effect=RuntimeError("live writer"))
        ),
        _env=object(),
        _task=object(),
        disconnect=AsyncMock(),
    )
    monkeypatch.setattr(_review, "_freeze_requested", lambda _: False)
    # The task keeps no services (K2), so the kill is the plain one.
    monkeypatch.setattr(_review.keep_services, "trial_keep", lambda *_: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _: "no recovery contract",
    )
    with pytest.raises(RuntimeError, match="live writer"):
        await _review.capture_terminal_workspace(trial)


@pytest.mark.asyncio
@pytest.mark.parametrize("copy_timeout", [False, True])
async def test_a_slow_or_timed_out_collection_cannot_complete_a_late_answer(
    tmp_path, monkeypatch, copy_timeout
):
    """Guards 12f002dc through verify(), not just the evidence helper."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task import RolloutPaths, Task

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Produce an answer.")
    trial = Rollout(RolloutConfig(task_path=task, sandbox_user="agent"))
    trial._task = Task(task)
    trial._rollout_dir = tmp_path / "trial"
    trial._rollout_dir.mkdir()
    trial._rollout_paths = RolloutPaths(rollout_dir=trial._rollout_dir)
    trial._started_at = datetime.now()
    trial._env = SimpleNamespace()
    trial._agent_cwd = "/app"
    trial._trajectory = [{"type": "agent_message", "content": "done"}]
    trial.disconnect = AsyncMock()
    answer = False
    copy_started = asyncio.Event()

    async def late_writer():
        nonlocal answer
        await copy_started.wait()
        answer = True

    writer = asyncio.create_task(late_writer())

    async def stop(*args):
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)

    trial._planes = SimpleNamespace(quiesce_agent=stop)

    async def collect(*args):
        copy_started.set()
        await asyncio.sleep(0)
        if copy_timeout:
            trial._artifact_manifest = {"collections": [{"status": "error"}]}

    async def verify(*args, **kwargs):
        return {"reward": float(answer)}, None, None

    monkeypatch.setattr("benchflow.rollout.collect_rollout_artifacts", collect)
    monkeypatch.setattr("benchflow.rollout._verify_rollout", verify)
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    try:
        assert await trial.verify() == {"reward": 0.0}
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
