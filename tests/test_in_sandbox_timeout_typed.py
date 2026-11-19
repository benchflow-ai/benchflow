"""A command its own time limit ended is a timeout on every sandbox.

Guards the audit fix on branch ``audit-in-sandbox-timeout``.
Docker raises ``ExecTimeoutError`` for a command that runs past its limit.
Daytona and Modal answer with an exit status instead (124 from GNU ``timeout``
here; the other statuses are in ``tests/test_time_limit_status_backends.py``).
So on Daytona an oracle that ran out of time went on to the verifier as a
finished run (a scored 0 with no error), and a setup command that ran out of
time was "failed with exit code 124" (not retried), where Docker records a
timeout and an infrastructure error.

The exec layer keeps returning 124 as a result (a command may exit 124 itself,
``tests/test_daytona_channel.py``); the three callers that set the limit now ask
:func:`ended_by_time_limit`.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import INFRA_ERROR, classify_error
from benchflow.evaluation import RetryConfig
from benchflow.rollout import _run_one_environment_setup_command
from benchflow.rollout._setup import OracleTimeoutError, _run_oracle
from benchflow.sandbox.protocol import ExecTimeoutError, ended_by_time_limit
from tests.test_dx_a_errors import _oracle_task


class _Answer:
    stdout, stderr = "", ""

    def __init__(self, return_code: int) -> None:
        self.return_code = return_code


def _daytona_like(monkeypatch, *, limit: int, took: float, status: int = 124):
    """An env whose exec with *limit* answers *status* after *took* seconds,
    as a Daytona session command under ``timeout`` does."""
    real = time.monotonic
    passed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + passed[0])

    class Env:
        async def exec(self, command, **kwargs):
            if kwargs.get("timeout_sec") == limit:
                passed[0] += took
                return _Answer(status)
            return _Answer(0)

    return Env()


def test_the_predicate_needs_the_status_and_the_whole_limit():
    assert ended_by_time_limit(_Answer(124), 5, 5.2)
    assert not ended_by_time_limit(_Answer(124), 5, 1.0)  # the command's own 124
    assert not ended_by_time_limit(_Answer(1), 5, 5.2)
    assert not ended_by_time_limit(_Answer(124), None, 9999)
    assert not ended_by_time_limit(_Answer(124), 0, 9999)


def test_an_oracle_the_sandbox_timed_out_is_the_agent_timeout(monkeypatch):
    root, task = _oracle_task()
    env = _daytona_like(monkeypatch, limit=5, took=5.3)

    with pytest.raises(OracleTimeoutError) as caught:
        asyncio.run(_run_oracle(env, root, 5, task=task))

    text = str(caught.value)
    assert text == "oracle/solve.sh timed out after 5s, the task's agent timeout"
    assert classify_error(text) == "timeout"


def test_an_oracle_that_exits_124_itself_is_still_a_finished_run(monkeypatch):
    root, task = _oracle_task()
    env = _daytona_like(monkeypatch, limit=5, took=0.5)

    _trajectory, agent = asyncio.run(_run_oracle(env, root, 5, task=task))

    assert agent == "oracle"


def _setup_command(limit: int) -> SimpleNamespace:
    return SimpleNamespace(
        command="npm ci",
        cwd=None,
        timeout_sec=limit,
        user=None,
        service=None,
        capture_dir=None,
        capture_dir_b64_env=None,
    )


def test_a_setup_command_the_sandbox_timed_out_is_retried_as_infrastructure(
    monkeypatch,
):
    env = _daytona_like(monkeypatch, limit=600, took=600.4)

    with pytest.raises(ExecTimeoutError) as caught:
        asyncio.run(
            _run_one_environment_setup_command(
                env, _setup_command(600), env_vars=None, index=1
            )
        )

    text = str(caught.value)
    assert text == "Command timed out after 600 seconds"
    assert classify_error(text) == INFRA_ERROR
    assert RetryConfig().should_retry(text)


def test_a_setup_command_that_exits_124_itself_is_its_own_failure(monkeypatch):
    env = _daytona_like(monkeypatch, limit=600, took=2.0)

    with pytest.raises(RuntimeError, match="failed with exit code 124") as caught:
        asyncio.run(
            _run_one_environment_setup_command(
                env, _setup_command(600), env_vars=None, index=1
            )
        )

    assert not isinstance(caught.value, ExecTimeoutError)


def _user_setup_env(monkeypatch, *, took: float, status: int):
    """A sandbox whose user-setup exec (the one that creates the user and
    grants the workspace) answers *status* after *took* seconds. Reads
    answer with receipts, as a sandbox's shell prints them."""
    from unittest.mock import MagicMock

    from tests._receipt_fakes import receipt_aware

    real = time.monotonic
    passed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + passed[0])
    commands: list[str] = []

    async def run(command, **kwargs):
        commands.append(command)
        answer = MagicMock(return_code=0, stdout="", stderr="")
        if "chown -R agent:agent /home/agent" in command:
            passed[0] += took
            answer.return_code = status
            answer.stderr = "Terminated"
        return answer

    env = MagicMock()
    env.supports_system_shell = True
    env.exec = receipt_aware(run)
    return env, commands


def test_a_workspace_grant_the_sandbox_timed_out_stops_the_setup(monkeypatch):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env, commands = _user_setup_env(monkeypatch, took=600.5, status=124)

    with pytest.raises(ExecTimeoutError) as caught:
        asyncio.run(setup_sandbox_user(env, "agent", "/app", timeout_sec=600))

    text = str(caught.value)
    assert text == "Command timed out after 600 seconds"
    assert classify_error(text) == INFRA_ERROR
    assert RetryConfig().should_retry(text)
    # Nothing ran after the cut-short setup: no `id -u`, no PATH lock.
    assert "chown -R agent:agent /home/agent" in commands[-1]


def test_a_user_setup_step_that_fails_in_time_is_still_tolerated(monkeypatch):
    """The existing rule (tests/test_sandbox_user_busybox.py): a step that
    fails once the user exists, such as a chown on a read-only mount, does
    not stop the setup."""
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env, _commands = _user_setup_env(monkeypatch, took=3.0, status=1)

    assert asyncio.run(setup_sandbox_user(env, "agent", "/app", timeout_sec=600)) == (
        "/app"
    )
