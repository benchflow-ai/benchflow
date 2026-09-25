"""Verifier retry on the zero-output timeout (exec-layer wedge) signature.

Observed on Daytona (2026-08-07/08): verifiers that complete in under a second
when they run sometimes burned their whole timeout budget with an EMPTY
test-stdout.txt — the exec session wedged before test.sh ever started. Those
rollouts scored ``reward=None`` (lost data) even though the workspace held
scoreable work. ``_verify_rollout`` now retries the verifier exactly once when
a timeout carries the no-output signature; a timeout WITH output (a genuinely
slow or hung verifier) is never retried.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout._setup import _verify_rollout


def _mk_task(service: str = "main"):
    return SimpleNamespace(
        name="wedge-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                timeout_sec=0.2,
                reward_range=None,
                env={},
                service=service,
            )
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
@pytest.mark.parametrize("verifier_service", ["main", "scorer"])
async def test_zero_output_timeout_retries_once_and_scores(tmp_path, verifier_service):
    """Retries a silent timeout on the selected verifier service."""
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)  # wedge: never returns
        return SimpleNamespace(rewards={"reward": 1.0})

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("0\n")  # probe: empty test-stdout

    rewards, verr, vtimeout = await _verify_rollout(
        env,
        _mk_task(verifier_service),
        _mk_paths(tmp_path),
        {},
        _mk_planes(verifier),
    )

    assert len(attempts) == 2
    assert rewards == {"reward": 1.0}
    assert verr is None
    assert vtimeout is None
    assert [call.kwargs.get("service") for call in env.exec.await_args_list] == [
        verifier_service,
        verifier_service,
    ]


@pytest.mark.asyncio
async def test_timeout_with_output_is_not_retried(tmp_path):
    """Does not retry a timeout after the verifier produced scorer output."""
    attempts = []

    async def verify():
        attempts.append(1)
        await asyncio.sleep(3600)

    verifier = SimpleNamespace(verify=verify)
    env = _env_with_probe_output("4242\n")  # probe: real output was produced

    rewards, verr, vtimeout = await _verify_rollout(
        env, _mk_task("scorer"), _mk_paths(tmp_path), {}, _mk_planes(verifier)
    )

    assert len(attempts) == 1
    assert rewards is None
    assert verr is not None and "timed out" in verr
    assert vtimeout is not None
    assert env.exec.await_args.kwargs.get("service") == "scorer"


@pytest.mark.asyncio
async def test_wedged_retry_that_also_times_out_reports_timeout(tmp_path):
    """Reports a timeout when both verifier attempts time out."""
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
