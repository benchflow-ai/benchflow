"""Guards the fix against 96155b555, which scored every verifier timeout 0
once the test command was issued, without reading anything.

A test command whose tests ended but whose exec never reported back (a lost
Daytona poll, a hung transport) was scored 0 although test.sh had written
its verdict, and a timeout the sandbox could not even be asked about was
scored 0 too. The command now records its tests' exit status beside its start
receipt: tests that ended are scored from their outputs by the usual rules,
tests still running at the deadline still score 0 (even after writing a
passing reward first), and an unanswered sandbox leaves the trial unscored
as a verifier timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from benchflow._utils.scoring import classify_verifier_error
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests._receipt_fakes import receipt_aware
from tests.reachable_fixture import on_disk
from tests.test_verifier_environment_checks import _MappedSandbox

BUDGET_SEC = 3


class _Sandbox(_MappedSandbox):
    """The real shell, with the test command's exec lost after it ends."""

    def __init__(self, mapping, *, lose_exec=False, probe_fails=False):
        super().__init__(mapping)
        self.lose_exec = lose_exec
        self.probe_fails = probe_fails

    @receipt_aware
    async def exec(self, command, env=None, **kwargs):
        self.commands.append(command)
        if self.probe_fails and "printf 'ended '" in command:
            raise ConnectionError("dummy: the sandbox stopped answering")
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            self._map(command),
            env={"PATH": "/usr/bin:/bin", **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise
        if self.lose_exec and "printf started" in command:
            await asyncio.Event().wait()  # the exec never reports back
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )


def _setup(tmp_path: Path, script: str, **sandbox):
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/sh\n" + script)
    task = MagicMock()
    task.name = "lost-exec"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        f'version = "1.0"\n[verifier]\ntimeout_sec = {BUDGET_SEC}\n'
    )
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    env = _Sandbox(
        {
            "/logs/verifier": on_disk(paths.verifier_dir),
            "/tests": tmp_path / "sandbox-tests",
            "/run/benchflow": tmp_path / "sandbox-run-benchflow",
        },
        **sandbox,
    )
    return task, paths, env


async def _score(task, paths, env):
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(),
        verifier=lambda **kw: Verifier(
            kw["task"],
            kw["rollout_paths"],
            kw["sandbox"],
            execution_receipt=kw.get("execution_receipt", False),
        ),
    )
    return await _verify_rollout(env, task, paths, {}, planes)


async def test_tests_that_ended_behind_a_lost_exec_are_scored(tmp_path, caplog):
    task, paths, env = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\n", lose_exec=True
    )
    with caplog.at_level("WARNING"):
        rewards, error, diagnostic = await _score(task, paths, env)
    assert (rewards, error, diagnostic) == ({"reward": 1.0}, None, None)
    assert "scoring its outputs" in caplog.text


async def test_tests_still_running_at_the_deadline_score_0(tmp_path):
    # A passing reward written first is no verdict while the tests still run.
    task, paths, env = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\nsleep 60\n"
    )
    rewards, error, diagnostic = await _score(task, paths, env)
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic is not None and diagnostic.scored is True
    assert diagnostic.format_issue("lost-exec").endswith("scored 0")


async def test_an_unanswered_sandbox_leaves_the_timeout_unscored(tmp_path):
    task, paths, env = _setup(
        tmp_path,
        "echo 1 > /logs/verifier/reward.txt\n",
        lose_exec=True,
        probe_fails=True,
    )
    rewards, error, diagnostic = await _score(task, paths, env)
    assert rewards is None
    assert error.startswith(f"verifier timed out after {BUDGET_SEC}")
    assert "did not answer whether the test command had ended" in error
    assert classify_verifier_error(error) == "verifier_timeout"
    assert diagnostic is not None and diagnostic.scored is False


async def test_the_command_records_its_tests_exit_status(tmp_path):
    task, paths, env = _setup(tmp_path, "echo 0 > /logs/verifier/reward.txt\nexit 3\n")
    verifier = Verifier(task, paths, env, execution_receipt=True)
    result = await verifier.verify()
    assert result.rewards == {"reward": 0.0}
    assert verifier.test_return_code == 3
    receipt, _ = verifier.execution_receipt
    assert Path(env._map(receipt + ".exit")).read_text() == "3"
