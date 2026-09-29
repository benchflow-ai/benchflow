"""A verifier setup exec that fails on a stalled exec layer is retried.

The idea of #1043 (fixes #948), applied to the verifier's own setup: those
commands (chmod of test.sh, clearing the output dir, the target verifier dir,
the reward-kit manifest) run after the agent has finished, with a 10 s exec.
On a loaded Docker host or during a Daytona API stall one failed, and took
the finished rollout's verification with it ("Verifier setup failed: chmod
exited with rc=1" after a 3.5 h solve). Each now gets 60 s and two more
tries; only when every try fails is it ``Verifier setup failed`` (verifier
infrastructure).
"""

from __future__ import annotations

import pytest

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.sandbox._base import ExecResult
from benchflow.task import verifier_core
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_core import Verifier
from benchflow.task.verifier_errors import VerifierOutputParseError
from tests.test_verifier_environment_checks import (
    _MappedSandbox,
    _receipt_sandbox,
    _receipt_task,
)


class _FlakySetup(_MappedSandbox):
    """Fails the first calls of one setup command, then runs it for real."""

    def __init__(self, base: _MappedSandbox, prefix: str, failures: list) -> None:
        self.__dict__.update(base.__dict__)
        self.prefix = prefix
        self.failures = list(failures)
        self.timeouts: list[int | None] = []

    async def exec(self, command, env=None, **kwargs):
        if command.startswith(self.prefix):
            self.timeouts.append(kwargs.get("timeout_sec"))
            if self.failures:
                failure = self.failures.pop(0)
                if isinstance(failure, BaseException):
                    raise failure
                return ExecResult(stdout="", stderr="stalled", return_code=failure)
        return await super().exec(command, env=env, **kwargs)


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(verifier_core, "_SETUP_RETRY_DELAYS_SEC", (0.0, 0.0))


def _verifier(tmp_path, prefix: str, failures: list):
    task = _receipt_task(tmp_path)
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    sandbox = _FlakySetup(_receipt_sandbox(tmp_path, paths), prefix, failures)
    return Verifier(task, paths, sandbox), sandbox


async def test_a_chmod_that_fails_twice_is_retried_and_the_run_scores(tmp_path):
    verifier, sandbox = _verifier(
        tmp_path, "chmod +x", [1, RuntimeError("Command timed out after 60 seconds")]
    )
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    assert sandbox.timeouts == [60, 60, 60]


async def test_a_chmod_that_keeps_failing_is_verifier_infrastructure(tmp_path):
    verifier, sandbox = _verifier(tmp_path, "chmod +x", [1, 1, 1])
    with pytest.raises(VerifierOutputParseError) as caught:
        await verifier.verify()
    assert str(caught.value) == "Verifier setup failed: chmod exited with rc=1"
    assert classify_verifier_error(f"verifier crashed: {caught.value}") == (
        VERIFIER_INFRA
    )
    assert len(sandbox.timeouts) == 3


async def test_a_lost_exec_names_its_cause(tmp_path):
    verifier, _ = _verifier(
        tmp_path, "chmod +x", [ConnectionResetError("session lost")] * 3
    )
    with pytest.raises(VerifierOutputParseError) as caught:
        await verifier.verify()
    assert str(caught.value).startswith("Verifier setup failed: chmod did not run: ")
    assert "session lost" in str(caught.value)
