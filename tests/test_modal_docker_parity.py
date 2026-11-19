"""A Modal sandbox matches the Docker backend's where Modal lets it (#1245).

Guards the fix for issue #1245 against the behaviour of 5a29c66 (0.10.0rc5),
where ``sandbox/modal_impl.py`` was unchanged since 0.9.1. All provider calls
are fakes; what Modal itself does was measured with modal 1.6.2 and is quoted
where a test relies on it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("tenacity")

from benchflow._utils.modal_lifetime import SANDBOX_TIMEOUT_ENV, modal_sandbox_timeout
from benchflow._utils.scoring import INFRA_ERROR, PIPE_CLOSED, classify_error
from benchflow.sandbox import modal_impl
from benchflow.sandbox.modal_impl import (
    MODAL_ENDED,
    ModalSandbox,
    ModalSandboxEndedError,
)


class _Aio:
    """A Modal method: callable, with an ``aio`` coroutine twin."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[tuple[tuple, dict]] = []

    async def aio(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = self._fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


class NotFoundError(Exception):
    """modal.exception.NotFoundError, as an exec in an ended sandbox raises it:
    "Modal Sandbox with container ID ta-... not found. This means this Sandbox
    has already shut down." (measured, modal 1.6.2)."""


def _gone(*_args, **_kwargs):
    raise NotFoundError(
        "Modal Sandbox with container ID ta-1 not found. This means this "
        "Sandbox has already shut down."
    )


def _modal_sandbox(remote) -> ModalSandbox:
    """A started ModalSandbox around the fake Modal handle *remote*."""
    sandbox = object.__new__(ModalSandbox)
    sandbox.default_user = "root"
    sandbox._persistent_env = {}
    sandbox._sandbox = remote
    sandbox._app = sandbox._image = None
    return sandbox


def _remote(*, poll=None, exec_=_gone):
    return SimpleNamespace(
        object_id="sb-1",
        poll=_Aio(lambda: poll),
        exec=_Aio(exec_),
        terminate=_Aio(lambda: None),
        wait=_Aio(lambda raise_on_termination=True: None),
    )


@pytest.fixture
def fake_secret(monkeypatch):
    import modal

    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda env: env))


# Gap 5: a sandbox Modal ended is recorded and failed as infrastructure.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "said"),
    [(124, "lifetime ran out"), (137, "memory limit"), (3, "exited with 3")],
)
async def test_stop_records_an_end_modal_made_before_the_terminate(code, said):
    remote = _remote(poll=code)
    sandbox = _modal_sandbox(remote)
    await sandbox.stop(delete=True)
    assert sandbox.platform_end == {
        "returncode": code,
        "reason": modal_impl.platform_end_reason(code),
        "seen_at": "stop",
    }
    assert said in sandbox.platform_end["reason"]
    # Asked before any terminate, which a sandbox Modal ended does not need.
    assert len(remote.poll.calls) == 1
    assert remote.terminate.calls == []
    assert sandbox._sandbox is None


@pytest.mark.asyncio
async def test_stop_of_a_running_sandbox_records_no_end():
    remote = _remote(poll=None)
    sandbox = _modal_sandbox(remote)
    await sandbox.stop(delete=True)
    assert sandbox.platform_end is None
    assert len(remote.terminate.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_exec_in_an_ended_sandbox_is_an_infrastructure_error(
    fake_secret,
):
    """The issue's repro: a timeout=60 sandbox running a 120 s exec. Modal
    ends it, the exec's wait raises NotFoundError and poll returns 124."""
    sandbox = _modal_sandbox(_remote(poll=124))
    with pytest.raises(ModalSandboxEndedError) as raised:
        await sandbox.exec("sleep 120", user="root")
    message = str(raised.value)
    assert message.startswith(MODAL_ENDED)
    assert "returncode 124" in message
    assert isinstance(raised.value.__cause__, NotFoundError)
    assert sandbox.platform_end["seen_at"] == "command"
    # Modal's own text classifies as no infrastructure error; ours does.
    assert classify_error(str(raised.value.__cause__)) != INFRA_ERROR
    assert classify_error(message) == INFRA_ERROR
    assert classify_error(f"ModalSandboxEndedError: {message}") == INFRA_ERROR


@pytest.mark.asyncio
async def test_a_failed_exec_in_a_running_sandbox_raises_as_before(
    monkeypatch, fake_secret
):
    monkeypatch.setattr(modal_impl, "PLATFORM_END_SETTLE_SEC", 0.05)
    monkeypatch.setattr(modal_impl, "PLATFORM_END_SETTLE_STEP_SEC", 0.01)
    remote = _remote(poll=None)
    sandbox = _modal_sandbox(remote)
    with pytest.raises(NotFoundError):
        await sandbox.exec("true", user="root")
    assert sandbox.platform_end is None
    # Modal's "not found" is waited out, within the bound.
    assert len(remote.poll.calls) > 1


@pytest.mark.asyncio
async def test_no_answer_from_poll_keeps_the_exec_error(monkeypatch, fake_secret):
    async def hang():
        await asyncio.sleep(60)

    monkeypatch.setattr(modal_impl, "PLATFORM_END_POLL_SEC", 0.01)
    remote = _remote()
    remote.poll = _Aio(hang)
    sandbox = _modal_sandbox(remote)
    with pytest.raises(NotFoundError):
        await sandbox.exec("true", user="root")
    assert sandbox.platform_end is None


class ConflictError(Exception):
    """modal.exception.ConflictError: "Modal Sandbox is shutting down." """


@pytest.mark.asyncio
async def test_a_shutdown_is_waited_out_before_the_end_is_read(
    monkeypatch, fake_secret
):
    """Measured: while Modal shuts a sandbox down, an exec fails with
    ConflictError "Modal Sandbox is shutting down." and poll still answers
    None for a while."""
    answers = iter([None, None, 124])

    def shutting_down(*_args, **_kwargs):
        raise ConflictError("Modal Sandbox is shutting down.")

    monkeypatch.setattr(modal_impl, "PLATFORM_END_SETTLE_STEP_SEC", 0.0)
    remote = _remote(exec_=shutting_down)
    remote.poll = _Aio(lambda: next(answers))
    sandbox = _modal_sandbox(remote)
    with pytest.raises(ModalSandboxEndedError, match="returncode 124"):
        await sandbox.exec("true", user="root")
    assert len(remote.poll.calls) == 3


@pytest.mark.asyncio
async def test_another_failure_asks_once(monkeypatch, fake_secret):
    def broken(*_args, **_kwargs):
        raise OSError("connection reset")

    remote = _remote(poll=None, exec_=broken)
    sandbox = _modal_sandbox(remote)
    with pytest.raises(OSError, match="connection reset"):
        await sandbox.exec("true", user="root")
    assert len(remote.poll.calls) == 1


@pytest.mark.asyncio
async def test_an_end_seen_after_stop_began_is_benchflows_own():
    sandbox = _modal_sandbox(_remote(poll=137))
    sandbox._stopping = True
    assert await sandbox.check_platform_end("command") is None
    assert sandbox.platform_end is None


class _Chunks:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                await asyncio.sleep(0)
                yield chunk

        return gen()


def _ended_agent_process():
    """The agent's exec when Modal ends its sandbox: stdout ends, and wait
    and poll raise NotFoundError (measured)."""
    return SimpleNamespace(
        stdout=_Chunks([]),
        stderr=_Chunks([]),
        stdin=SimpleNamespace(
            write=lambda data: None,
            write_eof=lambda: None,
            drain=_Aio(lambda: None),
        ),
        wait=_Aio(_gone),
        poll=_Aio(_gone),
    )


class _Env:
    """The BenchFlow sandbox a ModalProcess asks (check_platform_end)."""

    def __init__(self, end):
        self._end = end
        self.asked: list[str] = []

    async def check_platform_end(self, seen_at, *, settle_sec=0.0):
        self.asked.append(seen_at)
        self.settle_sec = settle_sec
        return self._end


@pytest.mark.asyncio
@pytest.mark.parametrize("ended", [True, False], ids=["modal-ended", "running"])
async def test_the_agent_process_names_an_end_modal_made(monkeypatch, ended):
    import sys
    import types

    from benchflow.diagnostics import TransportClosedError
    from benchflow.sandbox.process.modal import ModalProcess

    fake = types.ModuleType("modal")
    fake.Secret = SimpleNamespace(from_dict=lambda d: d)
    monkeypatch.setitem(sys.modules, "modal", fake)
    end = {"returncode": 137, "reason": "it was killed", "seen_at": "agent process"}
    env = _Env(end if ended else None)
    remote = SimpleNamespace(exec=_Aio(lambda *a, **k: _ended_agent_process()))
    process = ModalProcess(remote, env=env)

    async def route(_env):
        return None

    monkeypatch.setattr("benchflow.sandbox.process.modal.launch_route", route)
    await process.start("agent --acp")
    with pytest.raises(TransportClosedError) as raised:
        await process.readline()
    message = str(raised.value)
    assert env.asked == ["agent process"]
    # wait raised Modal's "not found": a shutdown is waited out.
    assert env.settle_sec == modal_impl.PLATFORM_END_SETTLE_SEC
    if ended:
        assert message.startswith(MODAL_ENDED)
        assert "closed stdout" not in message
        assert classify_error(message) == INFRA_ERROR
    else:
        assert message.startswith("Modal process closed stdout (rc=None)")
        assert classify_error(message) == PIPE_CLOSED
    assert raised.value.diagnostic.transport_diagnosis == "remote_session_killed"
    await process.close()


# Gap 4: a settable lifetime, checked by the plan gate.


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [(None, 86400), ("", 86400), ("60", 60), (" 3600 ", 3600), ("86400", 86400)],
)
def test_the_lifetime_setting(raw, seconds):
    environ = {} if raw is None else {SANDBOX_TIMEOUT_ENV: raw}
    assert modal_sandbox_timeout(environ) == seconds


@pytest.mark.parametrize("raw", ["59", "86401", "0", "-60", "3600.5", "1h", "١٢٣"])
def test_a_lifetime_out_of_range_is_refused(raw):
    with pytest.raises(ValueError, match=SANDBOX_TIMEOUT_ENV):
        modal_sandbox_timeout({SANDBOX_TIMEOUT_ENV: raw})


def _gpu_task(**sections):
    from benchflow.task import TaskConfig

    return TaskConfig.model_validate(
        {"sandbox": {"gpus": 1, "gpu_types": ["H100"]}, **sections}
    )


def _lifetime_issues(config):
    from benchflow.task.runtime_capabilities import validate_task_runtime_support

    return [
        issue
        for issue in validate_task_runtime_support(config, sandbox="modal")
        if issue.path == "agent.timeout_sec"
    ]


def test_the_plan_gate_refuses_a_task_that_outlives_a_modal_sandbox(monkeypatch):
    """The issue's repro: a 30 h agent budget passed the gate and Modal ended
    the sandbox at 24 h."""
    monkeypatch.delenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", raising=False)
    issues = _lifetime_issues(_gpu_task(agent={"timeout_sec": 30 * 3600}))
    assert len(issues) == 1
    assert issues[0].sandbox == "modal"
    assert "lives 86400 s" in issues[0].reason
    assert "agent (108000 s), verifier (600 s) and setup (600 s)" in issues[0].reason


def test_the_plan_gate_counts_agent_verifier_and_setup(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", "4000")
    fits = _gpu_task(
        agent={"timeout_sec": 2000},
        verifier={"timeout_sec": 900},
        sandbox={"gpus": 1, "setup_timeout_sec": 600},
    )
    assert _lifetime_issues(fits) == []
    over = _gpu_task(
        agent={"timeout_sec": 2000},
        verifier={"timeout_sec": 900},
        sandbox={
            "gpus": 1,
            "setup_timeout_sec": 600,
            "setup_commands": [{"command": "true", "timeout_sec": 600}],
        },
    )
    (issue,) = _lifetime_issues(over)
    assert "lives 4000 s" in issue.reason
    assert "add up to 4100 s" in issue.reason


def test_a_separate_verifier_does_not_count_against_the_agents_sandbox(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", "3000")
    shared = _gpu_task(agent={"timeout_sec": 1800}, verifier={"timeout_sec": 900})
    assert len(_lifetime_issues(shared)) == 1
    separate = _gpu_task(
        agent={"timeout_sec": 1800},
        verifier={"timeout_sec": 900, "sandbox_mode": "separate"},
    )
    assert _lifetime_issues(separate) == []


def test_other_backends_have_no_lifetime_check(monkeypatch):
    from benchflow.task import TaskConfig
    from benchflow.task.runtime_capabilities import validate_task_runtime_support

    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", "60")
    config = TaskConfig.model_validate({"agent": {"timeout_sec": 30 * 3600}})
    assert validate_task_runtime_support(config, sandbox="docker") == []


def test_the_launcher_creates_modal_sandboxes_with_the_setting(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from benchflow.sandbox import setup as sandbox_setup
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "task.toml").write_text(
        'schema_version = "1.3"\n[environment]\ngpus = 1\n'
    )
    (task_dir / "instruction.md").write_text("Solve it.\n")
    monkeypatch.setattr(ModalSandbox, "preflight", classmethod(lambda cls: None))

    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", "7200")
    task = Task(task_dir)
    made = sandbox_setup._create_environment("modal", task, task_dir, "t", MagicMock())
    assert made._sandbox_timeout == 7200

    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", "90000")
    with pytest.raises(ValueError, match="BENCHFLOW_MODAL_SANDBOX_TIMEOUT"):
        sandbox_setup._create_environment("modal", task, task_dir, "t", MagicMock())


# Gap 1: the task's CPUs and memory are hard limits.


@pytest.mark.parametrize(
    ("value", "environ", "given"),
    [
        (12, {}, (12, 12)),
        (0.5, {}, (0.5, 0.5)),
        (131072, {}, (131072, 131072)),
        (12, {"BENCHFLOW_MODAL_HARD_LIMITS": "0"}, 12),
        (12, {"BENCHFLOW_MODAL_HARD_LIMITS": "off"}, 12),
        (12, {"BENCHFLOW_MODAL_HARD_LIMITS": "1"}, (12, 12)),
        (None, {}, None),
        (0, {}, 0),
    ],
)
def test_modal_is_given_the_tasks_resources_as_limits(value, environ, given):
    assert modal_impl.modal_resource(value, environ) == given


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
async def test_both_launchers_create_with_hard_limits(tmp_path, monkeypatch, factory):
    """Docker sets cpus and mem_limit as limits
    (_compose_files/docker-compose-base.yaml); a bare number is a request on
    Modal, so 28 CPUs and 1.28 TB were visible at cpu=12."""
    create = await _create_kwargs(
        tmp_path, monkeypatch, factory=factory, cpus=12, memory_mb=131072
    )
    assert create["cpu"] == (12, 12)
    assert create["memory"] == (131072, 131072)


async def _create_kwargs(tmp_path, monkeypatch, *, factory, dockerfile=None, **config):
    """``Sandbox.create``'s keyword arguments for a start of a sandbox of a task
    with *config* (SandboxConfig fields) and *dockerfile*, all provider calls
    fakes."""
    import sys
    from types import ModuleType
    from unittest.mock import AsyncMock, MagicMock

    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox.setup import _create_benchflow_modal_environment_class
    from benchflow.task.config import SandboxConfig

    module = ModuleType("modal")
    module.App = SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock()))
    module.Image = MagicMock()
    module.Secret = MagicMock()
    module.Volume = MagicMock()
    remote = SimpleNamespace(
        filesystem=SimpleNamespace(make_directory=SimpleNamespace(aio=AsyncMock()))
    )
    module.Sandbox = SimpleNamespace(
        create=SimpleNamespace(aio=AsyncMock(return_value=remote))
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setattr(
        "benchflow.sandbox.modal_impl.bfx_delivery.provider_enabled", lambda name: False
    )
    (tmp_path / "Dockerfile").write_text(dockerfile or "FROM python:3.12-slim\n")
    cls = _create_benchflow_modal_environment_class() if factory else ModalSandbox
    sandbox = cls(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="trial",
        rollout_paths=None,
        task_env_config=SandboxConfig(**config),
    )
    sandbox._refuse_startup_names = AsyncMock()
    sandbox.exec = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    await sandbox.start(force_build=False)
    module.Sandbox.create.aio.assert_awaited_once()
    return module.Sandbox.create.aio.await_args.kwargs
