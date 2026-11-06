"""Support for the unchanged Sol acceptance input at 7bc81e6ba.

These helpers retain the original R5 probe's serializer and prior-command
assertions. The original full external R5 probe remains a separate input.
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.models import RolloutResult
from benchflow.rollout import _setup
from benchflow.rollout._results import _build_rollout_result
from benchflow.sandbox._base import ExecResult
from benchflow.task import Task
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier


def observe(root, rewards, error, expected, cause=None):
    """Use the product result serializer and reader, retaining observed reward."""
    folder = root / "saved-result"
    folder.mkdir()
    _build_rollout_result(
        folder,
        task_name="independent-review",
        rollout_name="review",
        agent="nop",
        agent_name="nop",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=error,
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        started_at=datetime.now(),
        timing={},
        result_filename="review-result.json",
    )
    path = folder / "review-result.json"
    saved = RolloutResult.load(path)
    print(
        json.dumps(
            {
                "record": str(path),
                "rewards": saved.rewards,
                "error": saved.verifier_error,
                "outcome": saved.score_outcome,
            }
        )
    )
    if expected == "unscored":
        assert saved.rewards is None, "controller failure must not record a score"
        assert saved.reward is None and saved.verifier_error
        assert saved.score_outcome == "verifier_errored"
        if cause:
            assert cause in saved.verifier_error
    else:
        assert saved.rewards == rewards
        assert saved.reward == float(expected) and saved.verifier_error is None
        assert saved.score_outcome == ("passed" if expected else "failed")


def task_and_env(root, env=None):
    folder = root / "task"
    (folder / "tests").mkdir(parents=True)
    (folder / "instruction.md").write_text("Grade the saved output.\n")
    (folder / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (folder / "task.toml").write_text('version = "1.0"\n[verifier]\ntimeout_sec = 5\n')
    paths = RolloutPaths(root / "trial")
    paths.mkdir()
    env = env or SimpleNamespace()
    env.is_mounted = True
    env.supports_system_shell = True
    env.upload_dir = AsyncMock()

    async def execute(command, **kwargs):
        if "test-stdout.txt" in command:
            paths.reward_text_path.write_text("1")
        return ExecResult(stdout="", stderr="", return_code=0)

    env.exec = AsyncMock(side_effect=execute)
    verifier = Verifier(Task(folder), paths, env)
    return env, verifier


async def assess(env, verifier, harden=None):
    planes = SimpleNamespace(
        harden_before_verify=harden or AsyncMock(), verifier=lambda **_: verifier
    )
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await _setup._verify_rollout(
            env, verifier._task, verifier._rollout_paths, {}, planes
        )
    return rewards, error, warnings


async def prime(root, env, prior):
    if prior:
        env, verifier = task_and_env(root, env)
        rewards, error, _ = await assess(env, verifier)
        assert rewards == {"reward": 1.0} and error is None
        assert env._benchflow_test_command_issued is True
        assert any(
            "test-stdout.txt" in (c.args[0] if c.args else c.kwargs.get("command", ""))
            for c in env.exec.await_args_list
        )
