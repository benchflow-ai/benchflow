"""Mounted verifier outputs are handed to the host user before they are read.

On Linux rootful docker the rollout's verifier dir is bind-mounted at
/logs/verifier and the test command runs as container root, so a reward written
through ``mkstemp`` (mode 0600, as some tasks' reward writers do) is root's
and the host cannot read it (#1202). The download that would chown it is
skipped for mounted outputs, so the verifier must hand the folder over itself.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.task.verifier_errors import VerifierOutputParseError

pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a mode-000 file, so the unreadable reward cannot be staged",
)


def _verifier(tmp_path, *, reward_mode: int):
    (tmp_path / "verifier").mkdir()
    rollout_paths = RolloutPaths(tmp_path)
    # The host's own view of the mounted file (rollout paths are capabilities).
    reward = tmp_path / "verifier" / "reward.txt"
    task = MagicMock()
    task.config.verifier.type = "test-script"
    task.config.verifier.service = "main"
    task.config.verifier.user = "root"
    task.config.verifier.env = None
    task.config.verifier.timeout_sec = 60
    task.paths.tests_dir = tmp_path / "tests"
    (tmp_path / "tests").mkdir()
    task.paths.test_path = tmp_path / "tests" / "test.sh"
    task.paths.uses_native_verifier_dir = False

    sandbox = MagicMock()
    sandbox.is_mounted = True

    async def fake_exec(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", "")
        if "test.sh" in cmd and "chmod" not in cmd:
            # What root's mkstemp + rename leaves on the bind mount: a reward
            # the host user cannot read.
            reward.write_text("1")
            reward.chmod(reward_mode)
        return MagicMock(exit_code=0, returncode=0)

    async def chown_to_host_user(path, recursive=False, service="main"):
        reward.chmod(0o644)

    sandbox.upload_dir = AsyncMock()
    sandbox.upload_file = AsyncMock()
    sandbox.exec = AsyncMock(side_effect=fake_exec)
    sandbox._chown_to_host_user = AsyncMock(side_effect=chown_to_host_user)
    verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)
    return verifier, sandbox


@pytest.mark.asyncio
async def test_root_owned_reward_on_mounted_outputs_is_handed_to_the_host(tmp_path):
    """Guards the fix for #1202: a finished verifier whose reward root wrote
    0600 on the bind mount was recorded as unreadable (scored 0 on 0.9);
    a benchmark's own infrastructure carries the same chown as a patch."""
    verifier, sandbox = _verifier(tmp_path, reward_mode=0o000)

    result = await verifier._verify_test_script()

    assert result.rewards == {"reward": 1.0}
    sandbox._chown_to_host_user.assert_awaited_once_with(
        "/logs/verifier", recursive=True, service="main"
    )


@pytest.mark.asyncio
async def test_a_reward_the_host_reads_changes_no_owner(tmp_path):
    """Guards the fix for #1202 against handing the folder over when nothing
    needs it: where the host reads the reward already (rootless docker maps
    container root to the host user, whose chown would give the folder away),
    the outputs stay as the test command left them and no exec is added."""
    verifier, sandbox = _verifier(tmp_path, reward_mode=0o644)

    result = await verifier._verify_test_script()

    assert result.rewards == {"reward": 1.0}
    sandbox._chown_to_host_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_outputs_stay_root_s_while_agent_code_may_run(tmp_path):
    """Guards the fix for #1202 against the kept-service write race: while the
    agent's code may run beside the verifier, whose uid can be the host
    user's, the mounted outputs are not handed over, and an unreadable reward
    still fails the parse as before."""
    verifier, sandbox = _verifier(tmp_path, reward_mode=0o000)

    async def release(code_dir: str) -> None:
        return None

    verifier.before_test = release
    with pytest.raises(VerifierOutputParseError, match="PermissionError"):
        await verifier._verify_test_script()

    sandbox._chown_to_host_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_root_owned_reward_from_a_criteria_runner_is_handed_to_the_host(
    tmp_path,
):
    """Guards the fix for #1202 on the criteria-runner path, which reads the
    mounted reward.json the same way."""
    from tests.test_verifier_strategies import _criteria_runner_verifier, _criteria_toml

    verifier, _ = _criteria_runner_verifier(
        tmp_path,
        criteria_toml=_criteria_toml("weighted_mean", {"c1": 1.0}),
        reward_payload={"metrics": {"c1": 1.0}},
    )
    verifier._task.config.verifier.user = "root"
    runner = verifier._sandbox.exec.side_effect

    def reward_files():
        return [p for p in tmp_path.rglob("reward.json") if "verifier" in p.parts]

    async def exec_(*args, **kwargs):
        result = await runner(*args, **kwargs)
        if "criteria_runner/reward.py" in (kwargs.get("command") or ""):
            for reward in reward_files():
                reward.chmod(0o000)
        return result

    async def chown_to_host_user(path, recursive=False, service="main"):
        for reward in reward_files():
            reward.chmod(0o644)

    verifier._sandbox.exec = AsyncMock(side_effect=exec_)
    verifier._sandbox.upload_file = AsyncMock()
    verifier._sandbox._chown_to_host_user = AsyncMock(side_effect=chown_to_host_user)

    result = await verifier.verify()

    assert result.rewards["reward"] == 1.0
    verifier._sandbox._chown_to_host_user.assert_awaited_once_with(
        "/logs/verifier", recursive=True, service="main"
    )
