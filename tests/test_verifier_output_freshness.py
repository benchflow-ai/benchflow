"""Regression tests for verifier reward-output freshness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier import RewardFileNotFoundError


def _make_task(tmp_path: Path) -> MagicMock:
    task_dir = tmp_path / "task"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/sh\ntrue\n")
    task = MagicMock()
    task.task_dir = task_dir
    task.paths.task_dir = task_dir
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\nservice = "target"\n'
    )
    task.instruction = "Verify target-side reward freshness."
    return task


class _StatefulTargetSandbox:
    """Mounted Docker-like sandbox with stale target-service verifier output."""

    is_mounted = True

    def __init__(self, *, fail_on_mountpoint_delete: bool = False) -> None:
        self.exec_calls: list[dict] = []
        self.remote_target_reward: str | None = "1.0"
        self.fail_on_mountpoint_delete = fail_on_mountpoint_delete

    async def upload_dir(self, source_dir, target_dir, service: str = "main") -> None:
        del source_dir, target_dir, service

    async def download_dir(self, source_dir, target_dir, service: str = "main") -> None:
        del source_dir
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if service == "target" and self.remote_target_reward is not None:
            (dest / "reward.txt").write_text(self.remote_target_reward)

    async def exec(self, command, service: str = "main", **kwargs) -> ExecResult:
        self.exec_calls.append({"command": command, "service": service, **kwargs})
        if self.fail_on_mountpoint_delete and "rm -rf /logs/verifier &&" in command:
            return ExecResult(
                stdout="",
                stderr="rm: cannot remove '/logs/verifier': Device or resource busy",
                return_code=1,
            )
        if service == "target" and "find /logs/verifier" in command:
            self.remote_target_reward = None
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.mark.asyncio
async def test_mounted_target_verifier_clears_stale_remote_reward(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@1256d8b against stale target-service rewards."""
    task = _make_task(tmp_path)
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = _StatefulTargetSandbox(fail_on_mountpoint_delete=True)

    with pytest.raises(RewardFileNotFoundError):
        await Verifier(task, rollout_paths, sandbox).verify()

    commands = [call["command"] for call in sandbox.exec_calls]
    clear_index = next(
        i for i, command in enumerate(commands) if "find /logs/verifier" in command
    )
    test_index = next(
        i for i, command in enumerate(commands) if "test-stdout.txt" in command
    )
    assert clear_index < test_index
    assert sandbox.exec_calls[clear_index]["service"] == "target"


@pytest.mark.asyncio
async def test_non_mounted_verifier_clears_contents_without_removing_mountpoint(
    tmp_path: Path,
) -> None:
    """Guards v0.5-integration@2014952 from deleting Daytona DinD log mounts."""
    task = _make_task(tmp_path)
    task.config.verifier.service = "main"
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = _StatefulTargetSandbox()
    sandbox.is_mounted = False

    with pytest.raises(RewardFileNotFoundError):
        await Verifier(task, rollout_paths, sandbox).verify()

    clear_command = next(
        call["command"]
        for call in sandbox.exec_calls
        if "find /logs/verifier" in call["command"]
    )
    assert "rm -rf /logs/verifier &&" not in clear_command
    assert "mkdir -p /logs/verifier" in clear_command
    assert "find /logs/verifier -mindepth 1" in clear_command
    assert "-exec rm -rf -- {} +" in clear_command
    assert "mkdir -p /app" not in clear_command
