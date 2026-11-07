"""The agent sandbox's teardown after the separate verifier moved ``verifier/`` aside.

DF-1 (dogfood p2-task-author, 2026-10-18): the separate verifier moves the
agent run's ``verifier/`` to ``verifier-sandbox/agent-verifier-dir/`` while the
agent's container still runs (the serial order, a stop not yet confirmed).
Teardown then serialized the Compose variables, read BenchFlow's own move as
a swap, printed ``verifier was swapped while BenchFlow wrote in it`` twice and
skipped ``compose down``. DF-2: that failed down then left the task image the
trial built behind, under a random name.
"""

from __future__ import annotations

import logging
import shutil

import pytest

from benchflow._utils.reachable_write import ReachableWriteError
from benchflow.rollout._separate_verifier import (
    AGENT_VERIFIER_DIR,
    RECORD_DIR,
    _publish_verifier_outputs,
)
from tests.test_compose_teardown import _docker_sandbox


def _publish(tmp_path, rollout_dir):
    outputs = tmp_path / "verifier-outputs"
    outputs.mkdir()
    (outputs / "reward.txt").write_text("1\n")
    record: dict = {}
    _publish_verifier_outputs(outputs, rollout_dir, record)
    return record


def _compose_with_env(sandbox, daemon):
    """Compose calls build their environment first, as the real runner does."""

    async def compose(command, check=True, timeout_sec=None, **_):
        sandbox._docker_compose_env()
        return await daemon.compose(command, check=check, timeout_sec=timeout_sec)

    return compose


async def test_teardown_after_the_move_aside_is_clean(tmp_path, caplog):
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:main", sandbox._env_vars.main_image_name + ":latest", "main")
    rollout_dir = tmp_path / "run"
    # compose up serialized the mounts while the folder was in place.
    before = sandbox._env_vars.to_env_dict(include_os_env=False)
    assert before["HOST_VERIFIER_LOGS_PATH"] == str(rollout_dir / "verifier")

    record = _publish(tmp_path, rollout_dir)
    assert record["planted_in_verifier_dir"]["moved_to"] == (
        f"{RECORD_DIR}/{AGENT_VERIFIER_DIR}"
    )
    sandbox._run_docker_compose_command = _compose_with_env(sandbox, daemon)

    with caplog.at_level(logging.DEBUG):
        await sandbox.stop(delete=True)

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "swapped" not in messages
    assert "force-killing" not in messages
    assert [c[0] for c in daemon.compose_calls] == ["kill", "down"]
    # The built image went with the sandbox (DF-2).
    assert daemon.images == {}


def test_the_mount_variable_names_the_folder_where_benchflow_moved_it(tmp_path):
    sandbox, _ = _docker_sandbox(tmp_path)
    rollout_dir = tmp_path / "run"
    sandbox._env_vars.to_env_dict(include_os_env=False)

    _publish(tmp_path, rollout_dir)

    env = sandbox._env_vars.to_env_dict(include_os_env=False)
    assert env["HOST_VERIFIER_LOGS_PATH"] == str(
        rollout_dir / RECORD_DIR / AGENT_VERIFIER_DIR
    )
    # The published folder is not what the agent's container mounts.
    assert (rollout_dir / "verifier" / "reward.txt").read_text() == "1\n"


def test_a_real_swap_is_still_refused(tmp_path):
    sandbox, _ = _docker_sandbox(tmp_path)
    rollout_dir = tmp_path / "run"
    sandbox._env_vars.to_env_dict(include_os_env=False)

    # Not BenchFlow's move: someone else replaced the folder.
    (rollout_dir / "verifier").rename(tmp_path / "elsewhere")
    (rollout_dir / "verifier").mkdir()

    with pytest.raises(ReachableWriteError, match="swapped"):
        sandbox._env_vars.to_env_dict(include_os_env=False)


def test_a_swap_after_benchflows_move_is_still_refused(tmp_path):
    sandbox, _ = _docker_sandbox(tmp_path)
    rollout_dir = tmp_path / "run"
    sandbox._env_vars.to_env_dict(include_os_env=False)
    _publish(tmp_path, rollout_dir)

    moved = rollout_dir / RECORD_DIR / AGENT_VERIFIER_DIR
    shutil.move(str(moved), str(tmp_path / "elsewhere"))
    moved.mkdir()

    with pytest.raises(ReachableWriteError, match="swapped"):
        sandbox._env_vars.to_env_dict(include_os_env=False)


async def test_a_failed_down_still_removes_the_image_the_trial_built(tmp_path):
    sandbox, daemon = _docker_sandbox(tmp_path)
    daemon.built("sha256:main", sandbox._env_vars.main_image_name + ":latest", "main")
    compose = daemon.compose

    async def failing_down(command, check=True, timeout_sec=None, **_):
        if command[0] == "down":
            raise RuntimeError("Docker compose command failed")
        return await compose(command, check=check, timeout_sec=timeout_sec)

    async def force_kill():
        daemon.containers.clear()

    sandbox._run_docker_compose_command = failing_down
    sandbox._force_kill_project = force_kill

    await sandbox.stop(delete=True)

    assert daemon.images == {}
