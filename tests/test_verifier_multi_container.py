"""Regression tests for target-service test-script verification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig


class TestVerifierServiceConfig:
    def test_service_defaults_to_main(self) -> None:
        """Guards #248 compatibility for existing single-container tasks."""
        cfg = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        assert cfg.verifier.service == "main"

    def test_service_can_target_named_container(self) -> None:
        """Guards #248 target-side verifier service selection."""
        cfg = TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier]\nservice = "target"\n'
        )
        assert cfg.verifier.service == "target"


def _make_task(tmp_path: Path, toml: str) -> MagicMock:
    task_dir = tmp_path / "task"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    task = MagicMock()
    task.task_dir = task_dir
    task.paths.task_dir = task_dir
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.config = TaskConfig.model_validate_toml(toml)
    task.instruction = "Exploit the target."
    return task


class _RecordingSandbox:
    def __init__(self, reward: str = "1.0", *, is_mounted: bool = False) -> None:
        self.is_mounted = is_mounted
        self.reward = reward
        self.upload_calls: list[dict] = []
        self.download_calls: list[dict] = []
        self.exec_calls: list[dict] = []

    async def upload_dir(self, source_dir, target_dir, service: str = "main") -> None:
        self.upload_calls.append(
            {"source": source_dir, "target": target_dir, "service": service}
        )

    async def download_dir(self, source_dir, target_dir, service: str = "main") -> None:
        self.download_calls.append(
            {"source": source_dir, "target": target_dir, "service": service}
        )
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "reward.txt").write_text(self.reward)

    async def exec(self, command, service: str = "main", **kwargs) -> ExecResult:
        self.exec_calls.append({"command": command, "service": service, **kwargs})
        return ExecResult(stdout="", stderr="", return_code=0)


class TestTargetServiceVerification:
    @pytest.mark.asyncio
    async def test_verifier_runs_test_script_in_target_service(
        self, tmp_path: Path
    ) -> None:
        """Guards #248 target-side verifier execution."""
        task = _make_task(
            tmp_path, 'version = "1.0"\n[verifier]\nservice = "target"\n'
        )
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox()

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert result.rewards == {"reward": 1.0}
        assert {call["service"] for call in sandbox.upload_calls} == {"target"}
        assert {call["service"] for call in sandbox.download_calls} == {"target"}
        assert {call["service"] for call in sandbox.exec_calls} == {"target"}

    @pytest.mark.asyncio
    async def test_target_verifier_dir_created_before_test_redirect(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #321: target service gets /logs/verifier before test.sh."""
        task = _make_task(
            tmp_path, 'version = "1.0"\n[verifier]\nservice = "target"\n'
        )
        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox()

        await Verifier(task, rollout_paths, sandbox).verify()

        commands = [call["command"] for call in sandbox.exec_calls]
        mkdir_index = next(
            i
            for i, command in enumerate(commands)
            if "mkdir -p /logs/verifier" in command
        )
        test_index = next(
            i for i, command in enumerate(commands) if "test-stdout.txt" in command
        )
        assert mkdir_index < test_index
        assert sandbox.exec_calls[mkdir_index]["user"] == "root"
