"""The verifier's environment sets TEST_DIR (#1179).

Registry test.sh scripts find their files through ``$TEST_DIR``, the folder
the verifier package is mounted at. BenchFlow did not set it: DevEval's
``source $TEST_DIR/setup-uv-pytest.sh`` read ``/setup-uv-pytest.sh`` and
exited without a reward, and MLGym's ``> $TEST_DIR/results.json`` under
``set -u`` stopped at once. It is now the mount: ``/tests`` for a task.toml
``tests/`` package, ``/verifier`` for a native ``verifier/`` one; a task's own
``[verifier.env]`` still wins.

The sandbox model runs test.sh's command through ``/bin/sh`` with the
environment the verifier passes, sandbox paths mapped into the test's folder.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests.reachable_fixture import on_disk

# Writes TEST_DIR as the reward's detail and fails under set -u without it,
# as MLGym's test.sh does.
TEST_SH = """#!/bin/sh
set -u
printf '%s' "$TEST_DIR" > "$OUT/test-dir"
echo 1 > "$REWARD_DIR/reward.txt"
"""


class ShSandbox:
    is_mounted = True

    def __init__(self, tmp_path: Path, paths: RolloutPaths) -> None:
        self.mount = tmp_path / "mount"
        self.out = tmp_path / "out"
        self.out.mkdir()
        self.paths = paths
        self.envs: list[dict | None] = []

    async def upload_dir(self, source_dir, target_dir, **kwargs):
        subprocess.run(["cp", "-R", str(source_dir), str(self.mount)], check=True)

    def local(self, command: str) -> str:
        # The mounts first, where a path starts, then /logs/verifier.
        command = re.sub(r"(?<![\w/])/(?:tests|verifier)/", f"{self.mount}/", command)
        return command.replace("/logs/verifier", str(on_disk(self.paths.verifier_dir)))

    async def exec(self, command, env=None, **kwargs):
        self.envs.append(env)
        if "ln -s" in command:
            # The /tests alias of a native mount: nothing to alias here.
            return ExecResult(stdout="", stderr="", return_code=0)
        result = subprocess.run(
            ["/bin/sh", "-c", self.local(command)],
            env={
                "PATH": "/usr/bin:/bin",
                "OUT": str(self.out),
                "REWARD_DIR": str(on_disk(self.paths.verifier_dir)),
                **(env or {}),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def _task(tmp_path: Path, *, native: bool, toml: str = "") -> MagicMock:
    package = tmp_path / "task" / ("verifier" if native else "tests")
    package.mkdir(parents=True)
    (package / "test.sh").write_text(TEST_SH)
    task = MagicMock()
    task.name = "task"
    task.task_dir = package.parent
    task.paths.task_dir = package.parent
    task.paths.tests_dir = package
    task.paths.test_path = package / "test.sh"
    task.paths.uses_native_verifier_dir = native
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n' + toml)
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native", "expected"), [(False, "/tests"), (True, "/verifier")]
)
async def test_test_dir_is_the_verifier_mount(tmp_path, native, expected):
    """Guards #1179: test.sh sees TEST_DIR, the folder its package is mounted at."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = ShSandbox(tmp_path, paths)

    result = await Verifier(
        task=_task(tmp_path, native=native), rollout_paths=paths, sandbox=sandbox
    ).verify()

    assert result.rewards == {"reward": 1.0}
    assert (sandbox.out / "test-dir").read_text() == expected


@pytest.mark.asyncio
async def test_a_tasks_own_test_dir_wins(tmp_path):
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = ShSandbox(tmp_path, paths)
    task = _task(tmp_path, native=False, toml='env = { TEST_DIR = "/srv/tests" }\n')

    await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()

    assert (sandbox.out / "test-dir").read_text() == "/srv/tests"
