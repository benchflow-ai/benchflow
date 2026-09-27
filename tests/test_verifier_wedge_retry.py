"""Verifier retry on the zero-output timeout (exec-layer wedge) signature.

Verifiers that complete in under a second when they run could burn their whole timeout budget with an EMPTY
test-stdout.txt — the exec session wedged before test.sh ever started. Those
rollouts scored ``reward=None`` (lost data) even though the workspace held
scoreable work. ``_verify_rollout`` now retries the verifier exactly once when
a timeout carries the no-output signature; a timeout WITH output (a genuinely
slow or hung verifier) is never retried.

That in-place retry (PR #949, 66054d5b) still applies to every task without a
verifier-only recovery contract; solver-evidence preservation had removed it for all tasks. Tasks
with an eligible recovery contract (#1136) instead use command start receipts
and recover in a fresh sandbox, never replaying in place.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.rollout import Rollout, RolloutConfig, _setup
from benchflow.rollout._setup import _verify_rollout
from benchflow.task import RolloutPaths, Task


def _mk_task():
    return SimpleNamespace(
        name="wedge-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=0.2, reward_range=None, env={})
        ),
    )


def _mk_paths(tmp_path):
    return SimpleNamespace(verifier_dir=tmp_path / "verifier")


def _mk_planes(verifier_stub):
    planes = SimpleNamespace()
    planes.harden_before_verify = AsyncMock()
    planes.verifier = lambda **_: verifier_stub
    return planes


def _env_with_probe_output(stdout: str):
    env = SimpleNamespace()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(stdout=stdout, stderr="", return_code=0)
    )
    return env


@pytest.mark.asyncio
async def test_zero_output_timeout_retries_once_and_scores(tmp_path):
    """Guards PR #949 against the solver-evidence preservation removal of the in-place retry."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)  # wedge: never returns
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")  # probe: empty test-stdout

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert verr is None
    assert vtimeout is None


@pytest.mark.asyncio
async def test_timeout_with_output_is_not_retried(tmp_path):
    """Guards PR #949: a slow verifier that produced output is never replayed."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("4242\n")  # probe: real output was produced

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 1
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None


@pytest.mark.asyncio
async def test_wedged_retry_that_also_times_out_reports_timeout(tmp_path):
    """Guards PR #949: exactly one in-place retry, never more."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task(), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 2  # exactly one retry, never more
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None


def _rollout(tmp_path, verify, *, recovery: bool):
    task = tmp_path / "task"
    task.mkdir()
    toml = 'version = "1.0"\n[verifier]\ntimeout_sec = 0.2\n'
    if recovery:
        toml += "workspace_recovery = true\n"
        toml += '[sandbox]\ndocker_image = "python@sha256:' + "a" * 64 + '"\n'
    (task / "task.toml").write_text(toml)
    (task / "instruction.md").write_text("Produce a file.")
    rollout = Rollout(RolloutConfig(task_path=task, task_digest=task_digest(task)))
    rollout._task = Task(task)
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._started_at = datetime.now()
    rollout._rollout_paths = RolloutPaths(rollout_dir=rollout._rollout_dir)
    rollout._env = _env_with_probe_output("0\n")
    # The real verifier records a start receipt on the recovery path.
    rollout._planes = _mk_planes(
        SimpleNamespace(verify=verify, execution_receipt=("/logs/verifier/s", "main"))
    )
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]
    return rollout


@pytest.mark.asyncio
async def test_rollout_without_recovery_contract_retries_wedge_in_place(
    tmp_path, monkeypatch
):
    """Guards PR #949 through Rollout.verify against the solver-evidence preservation regression."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    rollout = _rollout(tmp_path, verify, recovery=False)
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )

    assert await rollout.verify() == {"reward": 1.0}
    assert len(attempts) == 2
    assert rollout._verifier_error is None


@pytest.mark.asyncio
async def test_rollout_with_recovery_contract_leaves_wedge_to_recovery(
    tmp_path, monkeypatch
):
    """Guards #1136: an eligible task reports the wedge instead of replaying."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    rollout = _rollout(tmp_path, verify, recovery=True)
    rollout._env.exec.return_value.return_code = 1  # no start receipt
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    monkeypatch.setattr("benchflow.rollout.capture_terminal_workspace", AsyncMock())

    assert await rollout.verify() is None
    assert len(attempts) == 1
    assert rollout._verifier_error.startswith("[solver-preserved] ")
    assert "verifier_wedge:" in rollout._verifier_error


@pytest.mark.asyncio
async def test_timeout_without_start_receipt_is_not_replayed(tmp_path):
    """Guards #1136: an eligible task never replays its verifier in place."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)  # wedge: never returns
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )

    assert len(attempts) == 1
    assert rewards is None
    assert "timed out" in verr
    assert vtimeout is not None
    env.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_command_receipt_reports_wedge_without_in_place_retry(tmp_path):
    """Guards #1136: an eligible task reports startup loss for fresh recovery."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start-unique", "main")
    )
    env = _env_with_probe_output("0\n")

    rewards, verr, vtimeout = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )

    assert len(attempts) == 1
    assert rewards is None
    assert verr is not None and "verifier_wedge:" in verr
    assert vtimeout is None


@pytest.mark.asyncio
async def test_quiet_started_verifier_is_a_real_timeout(tmp_path):
    """Guards #1136: a start receipt beats lack of stdout."""

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/unique", "main")
    )
    rewards, error, diagnostic = await _verify_rollout(
        _env_with_probe_output("started"),
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert rewards is None
    assert error == "verifier timed out after 0.2s"
    assert diagnostic is not None


@pytest.mark.asyncio
async def test_startup_wedge_uses_short_grace_not_full_budget(tmp_path, monkeypatch):
    """Guards #1136: startup loss does not consume two verifier budgets."""
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)
    task = _mk_task()
    task.config.verifier.timeout_sec = 60

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    rewards, error, diagnostic = await asyncio.wait_for(
        _verify_rollout(
            _env_with_probe_output(""),
            task,
            _mk_paths(tmp_path),
            {},
            _mk_planes(verifier),
            recovery_eligible=True,
        ),
        timeout=1,
    )
    assert rewards is None and "verifier_wedge:" in error
    assert diagnostic is None


@pytest.mark.asyncio
async def test_receipt_probe_error_is_unknown_not_a_wedge(tmp_path, monkeypatch):
    """Guards the fix for solver-evidence preservation's false wedge verdict on a transient probe error.

    A failed start-receipt probe says nothing about the verifier; treating it as
    "not started" cancelled a running verifier after one grace interval.
    """
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)
    task = _mk_task()
    task.config.verifier.timeout_sec = 5
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(0.2)
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    env = SimpleNamespace(
        exec=AsyncMock(side_effect=RuntimeError("Failed to execute session command"))
    )
    rewards, error, diagnostic = await _verify_rollout(
        env,
        task,
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert (rewards, error, diagnostic) == ({"reward": 1.0}, None, None)
    assert len(attempts) == 1
    assert env.exec.await_count >= 1


@pytest.mark.asyncio
async def test_unknown_start_until_deadline_is_a_timeout(tmp_path, monkeypatch):
    """Guards the same fix: probe errors until the budget ends are a timeout."""
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 0.01)

    async def verify():
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(
        verify=verify, execution_receipt=("/logs/verifier/start", "main")
    )
    env = SimpleNamespace(exec=AsyncMock(side_effect=TimeoutError()))
    rewards, error, diagnostic = await _verify_rollout(
        env,
        _mk_task(),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
        recovery_eligible=True,
    )
    assert rewards is None
    assert error == "verifier timed out after 0.2s"
    assert diagnostic is not None
