"""A command its time limit ended is a timeout on every backend, whatever
status the backend answers it with.

Guards the review of PR #132, 2026-11-13: the PR typed only status 124 as the
time limit, which is GNU ``timeout`` on Daytona and nothing else. What each
backend answers for a command that runs its whole limit, read from the code
(no backend runs live here):

- Docker raises ``ExecTimeoutError`` (the docker client's wait in
  ``DockerSandbox._run_docker_process``).
- Daytona runs the command under the image's ``timeout``
  (``DaytonaSandbox._session_command``): GNU's exits 124; BusyBox's (an Alpine
  image) kills the command with SIGTERM, which reads 143.
- Modal passes the limit to ``Sandbox.exec``. The SDK's
  ``ContainerProcess.wait`` answers -1 once its exec deadline has passed, and
  128 + the signal when the process was killed (137 for SIGKILL).

Each backend case runs the backend's real exec under a task setup command,
with a fake at the SDK boundary (the docker client program, Daytona's session
API, the command router under Modal's own ``wait``), next to its negative: the
same status from a command that ended well within its limit is the command's
own failure. The oracle and the sandbox-user setup ask the same predicate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import INFRA_ERROR, classify_error
from benchflow.rollout import _run_one_environment_setup_command
from benchflow.rollout._setup import OracleTimeoutError, _run_oracle
from benchflow.sandbox.protocol import ExecTimeoutError, ended_by_time_limit
from benchflow.task.config import SetupCommandConfig
from tests.test_dx_a_errors import _oracle_task
from tests.test_in_sandbox_timeout_typed import _daytona_like, _user_setup_env

LIMIT = 600


def _clock(monkeypatch):
    """The monotonic clock, moved on by a fake SDK call for the seconds the
    command ran."""
    real = time.monotonic
    passed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + passed[0])

    def run_for(seconds: float) -> None:
        passed[0] += seconds

    return run_for


def _setup(env, limit: int = LIMIT) -> None:
    asyncio.run(
        _run_one_environment_setup_command(
            env,
            SetupCommandConfig(command="npm ci", timeout_sec=limit),
            env_vars=None,
            index=1,
        )
    )


def _assert_timed_out(env, limit: int = LIMIT) -> None:
    with pytest.raises(ExecTimeoutError) as caught:
        _setup(env, limit)
    text = str(caught.value)
    assert text == f"Command timed out after {limit} seconds"
    assert classify_error(text) == INFRA_ERROR


def _assert_own_failure(env, status: int) -> None:
    with pytest.raises(
        RuntimeError, match=f"failed with exit code {status}:"
    ) as caught:
        _setup(env)
    assert not isinstance(caught.value, ExecTimeoutError)


@pytest.mark.parametrize("status", [124, 137, 143, -1])
def test_a_time_limit_status_is_the_limit_only_once_it_has_elapsed(status):
    answer = SimpleNamespace(return_code=status)
    assert ended_by_time_limit(answer, LIMIT, LIMIT)
    assert ended_by_time_limit(answer, LIMIT, LIMIT + 0.4)
    assert not ended_by_time_limit(answer, LIMIT, LIMIT - 0.5)
    assert not ended_by_time_limit(answer, LIMIT, 2.0)


@pytest.mark.parametrize("status", [0, 1, 2, 125, 126, 127, 130, 255])
def test_another_status_after_the_whole_limit_is_the_commands_own(status):
    assert not ended_by_time_limit(
        SimpleNamespace(return_code=status), LIMIT, LIMIT + 1
    )


# Docker: the client program stands in for ``docker``.


def _docker(tmp_path, client: str):
    from benchflow.sandbox.docker import DockerSandbox

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    program = bin_dir / "docker"
    program.write_text(f"#!/bin/sh\n{client}\n")
    program.chmod(0o755)
    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = "ctr"
    sandbox._launch = SimpleNamespace(
        docker_args=[], env={"PATH": f"{bin_dir}:/usr/bin:/bin"}
    )
    return sandbox


def test_docker_raises_its_timeout_at_the_limit_unchanged(tmp_path):
    _assert_timed_out(_docker(tmp_path, "exec sleep 30"), limit=1)


def test_docker_command_that_exits_124_itself_is_its_own_failure(tmp_path):
    _assert_own_failure(_docker(tmp_path, "exit 124"), 124)


# Daytona: the toolbox's session API.


def _daytona(monkeypatch, *, took: float, status: int):
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import (
        DaytonaSandbox,
        _DaytonaDirect,
        _load_daytona_sdk,
    )

    _load_daytona_sdk()
    run_for = _clock(monkeypatch)
    sent: list[str] = []

    class Process:
        async def create_session(self, session_id):
            pass

        async def execute_session_command(self, session_id, request, timeout=None):
            sent.append(request.command)
            run_for(took)
            return SimpleNamespace(cmd_id="cmd-1")

        async def get_session_command(self, session_id, command_id):
            return SimpleNamespace(id="cmd-1", exit_code=status)

        async def get_session_command_logs(self, session_id, command_id):
            return SimpleNamespace(stdout="", stderr="Terminated")

        async def delete_session(self, session_id):
            pass

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox.logger = logging.getLogger("test-time-limit-daytona")
    sandbox._sandbox = SimpleNamespace(process=Process())
    sandbox._strategy = _DaytonaDirect(sandbox)
    return sandbox, sent


DAYTONA_TIMEOUTS = pytest.mark.parametrize(
    "status", [124, 143], ids=["gnu-timeout", "busybox-timeout"]
)


@DAYTONA_TIMEOUTS
def test_daytona_timeout_status_after_the_whole_limit_is_a_timeout(monkeypatch, status):
    env, sent = _daytona(monkeypatch, took=LIMIT + 0.4, status=status)

    _assert_timed_out(env)
    # The status is the image's timeout program's, which the command runs under.
    assert f"/usr/bin/timeout {LIMIT} " in sent[0]


@DAYTONA_TIMEOUTS
def test_daytona_same_status_from_a_command_that_ended_early_is_its_own(
    monkeypatch, status
):
    env, _sent = _daytona(monkeypatch, took=2.0, status=status)

    _assert_own_failure(env, status)


# Modal: the SDK's own ContainerProcess.wait over a fake command router.


def _modal(monkeypatch, *, took: float, signal: int | None = None):
    modal = pytest.importorskip("modal")
    from modal.container_process import _ContainerProcess
    from modal.exception import ExecTimeoutError as ModalDeadline

    from benchflow.sandbox.modal_impl import ModalSandbox

    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(dict))
    run_for = _clock(monkeypatch)
    timeouts: list[int | None] = []

    class Router:
        """``exec_wait`` is the RPC under ``ContainerProcess.wait``."""

        async def exec_wait(self, task_id, exec_id, deadline):
            if signal is not None:
                # The process was killed, and its status came back before
                # the client's deadline.
                return SimpleNamespace(WhichOneof=lambda _: "signal", signal=signal)
            if deadline is not None and time.monotonic() >= deadline:
                raise ModalDeadline(f"Deadline exceeded while waiting for {exec_id}")
            return SimpleNamespace(WhichOneof=lambda _: "code", code=0)

    async def read():
        return ""

    async def sandbox_exec(*argv, workdir=None, secrets=(), timeout=None):
        timeouts.append(timeout)
        process = _ContainerProcess.__new__(_ContainerProcess)
        process._returncode = None
        process._task_id = "ta-1"
        process._process_id = "ex-1"
        process._command_router_client = Router()
        # As Sandbox._exec: the deadline is taken once the exec has started.
        process._exec_deadline = time.monotonic() + int(timeout) if timeout else None
        run_for(took)
        stream = SimpleNamespace(read=SimpleNamespace(aio=read))
        return SimpleNamespace(
            stdout=stream, stderr=stream, wait=SimpleNamespace(aio=process.wait)
        )

    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox.logger = logging.getLogger("test-time-limit-modal")
    sandbox._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=sandbox_exec))
    return sandbox, timeouts


MODAL_ENDS = pytest.mark.parametrize(
    ("signal", "status"), [(None, -1), (9, 137)], ids=["deadline", "killed"]
)


@MODAL_ENDS
def test_modal_answers_its_time_limit_with_a_status(monkeypatch, signal, status):
    env, timeouts = _modal(monkeypatch, took=LIMIT + 0.4, signal=signal)

    result = asyncio.run(env.exec("npm ci", timeout_sec=LIMIT))

    assert result.return_code == status
    assert timeouts == [LIMIT]


@MODAL_ENDS
def test_modal_status_after_the_whole_limit_is_a_timeout(monkeypatch, signal, status):
    env, _timeouts = _modal(monkeypatch, took=LIMIT + 0.4, signal=signal)

    _assert_timed_out(env)


def test_modal_process_killed_early_is_its_own_failure(monkeypatch):
    env, _timeouts = _modal(monkeypatch, took=2.0, signal=9)

    _assert_own_failure(env, 137)


def test_a_minus_one_from_a_command_that_ended_early_is_its_own(monkeypatch):
    """Modal's SDK answers -1 only past its deadline; any other -1 is not the
    limit's."""
    env = _daytona_like(monkeypatch, limit=LIMIT, took=2.0, status=-1)

    _assert_own_failure(env, -1)


# The oracle and the sandbox-user setup, with the statuses 124 did not cover.

OTHER_STATUSES = pytest.mark.parametrize("status", [137, 143, -1])


@OTHER_STATUSES
def test_an_oracle_the_limit_ended_is_the_agent_timeout(monkeypatch, status):
    root, task = _oracle_task()
    env = _daytona_like(monkeypatch, limit=5, took=5.3, status=status)

    with pytest.raises(OracleTimeoutError) as caught:
        asyncio.run(_run_oracle(env, root, 5, task=task))

    assert str(caught.value) == (
        "oracle/solve.sh timed out after 5s, the task's agent timeout"
    )


@OTHER_STATUSES
def test_an_oracle_that_ends_early_with_the_status_is_a_finished_run(
    monkeypatch, status
):
    root, task = _oracle_task()
    env = _daytona_like(monkeypatch, limit=5, took=0.5, status=status)

    _trajectory, agent = asyncio.run(_run_oracle(env, root, 5, task=task))

    assert agent == "oracle"


@OTHER_STATUSES
def test_a_user_setup_the_limit_ended_stops_the_setup(monkeypatch, status):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env, commands = _user_setup_env(monkeypatch, took=600.5, status=status)

    with pytest.raises(
        ExecTimeoutError, match=r"^Command timed out after 600 seconds$"
    ):
        asyncio.run(setup_sandbox_user(env, "agent", "/app", timeout_sec=600))
    assert "chown -R agent:agent /home/agent" in commands[-1]


@OTHER_STATUSES
def test_a_user_setup_step_that_ends_early_with_the_status_is_tolerated(
    monkeypatch, status
):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    env, _commands = _user_setup_env(monkeypatch, took=3.0, status=status)

    assert asyncio.run(setup_sandbox_user(env, "agent", "/app", timeout_sec=600)) == (
        "/app"
    )
