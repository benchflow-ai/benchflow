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


# Gap 2: the image's thread counts, PYTHONHASHSEED and PYTHONPATH.


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        ("FROM python:3.12\nENV OMP_NUM_THREADS=1 PYTHONHASHSEED=7\n", None),
        # Legacy form, quotes, continuation lines.
        (
            'FROM a\nENV K V W\nENV J="two words" \\\n    L=3\n',
            {"K": "V W", "J": "two words", "L": "3"},
        ),
        # A reference to a value the base image may set resolves empty, as
        # Docker resolves one the base image leaves unset.
        ("FROM a\nENV PYTHONPATH=/app:$PYTHONPATH\n", {"PYTHONPATH": "/app:"}),
        # ARG defaults (a global one redeclared in the stage), braces.
        (
            "ARG T=4\nFROM a\nARG T\nARG U=2\nENV MKL_NUM_THREADS=${T} B=$U\n",
            {"MKL_NUM_THREADS": "4", "B": "2"},
        ),
        # Another ${...} form cannot be resolved here.
        ("FROM a\nENV A=${X:-y}\nENV B=$A\n", {"A": None, "B": None}),
        # The final stage, through the stage it is built FROM; not another.
        (
            "FROM a AS one\nENV X=1\nFROM b AS two\nENV Y=2\nFROM one\nENV Z=$X$Y\n",
            {"X": "1", "Z": "1"},
        ),
        # A heredoc body is not an instruction.
        ("FROM a\nRUN <<EOF\nENV FAKE=1\nEOF\nENV REAL=1\n", {"REAL": "1"}),
        ("# no stage\n", {}),
    ],
)
def test_the_dockerfile_env_reader(dockerfile, expected):
    from benchflow.sandbox._dockerfile_env import final_stage_env

    if expected is None:
        expected = {"OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "7"}
    assert final_stage_env(dockerfile) == expected


def test_the_image_env_given_at_create():
    image = modal_impl.modal_image_env(
        "FROM python:3.12\nENV OMP_NUM_THREADS=1 PYTHONPATH=/opt/lib\n"
    )
    assert image.create == {
        "OMP_NUM_THREADS": "1",
        # Unset in the image: empty, which Python reads as unset.
        "PYTHONHASHSEED": "",
        # The copy root's task commands get back, in place of PID 1's.
        "_BF_IMAGE_PYTHONPATH": "/opt/lib",
    }
    assert image.pythonpath == "/opt/lib"
    # Thread counts the image leaves unset keep Modal's (no client unset).
    assert image.kept == modal_impl.MODAL_THREAD_VARS[1:]


def test_the_tasks_env_goes_over_the_image():
    image = modal_impl.modal_image_env(
        "FROM a\nENV OMP_NUM_THREADS=1\n",
        {"OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONPATH": "/t"},
    )
    assert image.create["OMP_NUM_THREADS"] == "2"
    assert image.create["PYTHONHASHSEED"] == "0"
    assert image.pythonpath == "/t"


def test_a_prebuilt_image_keeps_what_cannot_be_read():
    image = modal_impl.modal_image_env(None, {"MKL_NUM_THREADS": "3"})
    assert image.create == {"MKL_NUM_THREADS": "3"}
    assert image.pythonpath is None
    assert "PYTHONHASHSEED" in image.kept and "PYTHONPATH" in image.kept


def test_an_unresolvable_pythonpath_is_left_as_modal_sets_it():
    image = modal_impl.modal_image_env("FROM a\nENV PYTHONPATH=${P:-/x}\n")
    assert image.pythonpath is None
    assert "_BF_IMAGE_PYTHONPATH" not in image.create
    assert "PYTHONPATH" in image.kept


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [False, True], ids=["direct", "factory"])
async def test_both_launchers_create_with_the_image_env(tmp_path, monkeypatch, factory):
    """The issue's repro image: ENV OMP_NUM_THREADS=1 at cpu=12, where the
    agent and test.sh saw OMP_NUM_THREADS=12 and PYTHONHASHSEED=0."""
    create = await _create_kwargs(
        tmp_path,
        monkeypatch,
        factory=factory,
        dockerfile="FROM python:3.12-slim\nENV OMP_NUM_THREADS=1\n",
        cpus=12,
    )
    assert create["env"] == {
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "",
        "_BF_IMAGE_PYTHONPATH": "",
    }


def _exec_secrets(remote) -> dict:
    (_args, kwargs) = remote.exec.calls[-1]
    merged = {}
    for secret in kwargs["secrets"]:
        merged.update(secret)
    return merged


def _ok(*_args, **_kwargs):
    async def read():
        return ""

    async def wait():
        return 0

    return SimpleNamespace(
        stdout=SimpleNamespace(read=SimpleNamespace(aio=read)),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=read)),
        wait=SimpleNamespace(aio=wait),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user", "system_shell"), [("root", False), ("root", True), (None, False)]
)
async def test_every_exec_starts_with_the_images_pythonpath(
    fake_secret, user, system_shell
):
    remote = _remote(exec_=_ok)
    sandbox = _modal_sandbox(remote)
    sandbox._image_env = modal_impl.ModalImageEnv(pythonpath="/opt/lib")
    await sandbox.exec("true", user=user, system_shell=system_shell)
    assert _exec_secrets(remote)["PYTHONPATH"] == "/opt/lib"


@pytest.mark.asyncio
async def test_a_sandbox_without_an_image_env_execs_as_before(fake_secret):
    remote = _remote(exec_=_ok)
    sandbox = _modal_sandbox(remote)
    await sandbox.exec("true", user="root")
    assert "PYTHONPATH" not in _exec_secrets(remote)


@pytest.mark.asyncio
async def test_the_agent_process_starts_with_the_images_pythonpath(monkeypatch):
    import sys
    import types

    from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
    from benchflow.sandbox.process.modal import ModalProcess

    fake = types.ModuleType("modal")
    fake.Secret = SimpleNamespace(from_dict=lambda d: d)
    monkeypatch.setitem(sys.modules, "modal", fake)

    async def route(_env):
        return None

    monkeypatch.setattr("benchflow.sandbox.process.modal.launch_route", route)
    env = SimpleNamespace(exec_process_env=lambda: {"PYTHONPATH": ""})
    remote = SimpleNamespace(exec=_Aio(lambda *a, **k: _ended_agent_process()))
    process = ModalProcess(remote, env=env)
    await process.start("agent --acp")
    (_args, kwargs) = remote.exec.calls[0]
    assert kwargs["secrets"] == [
        {**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "PYTHONPATH": ""}
    ]
    await process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("image", "seen"), [("/opt/lib", "/opt/lib"), ("", "")])
async def test_roots_task_route_gets_the_images_pythonpath_back(
    monkeypatch, image, seen
):
    """On a real shell: the process starts as Modal starts it (its
    PYTHONPATH=/pkg/:/root/ in the sandbox env, with the image's copy given at
    create); root's task route (test.sh's) ends with the image's value."""
    import os
    import subprocess

    import modal

    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda env: env))

    async def aio(*argv, secrets=(), **kwargs):
        payload = argv[-1]
        if os.geteuid() != 0:
            # Modal runs as root; a non-root host emulates only that switch.
            payload = payload.replace("/bin/su root -s /bin/bash -c", "/bin/sh -c")
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/pkg/:/root/",
            **modal_impl.modal_image_env(f"FROM a\nENV PYTHONPATH={image}\n").create,
        }
        for secret in secrets:
            env.update(secret)
        done = subprocess.run(
            [*argv[:-1], payload], env=env, capture_output=True, text=True
        )
        return _done(done)

    sandbox = _modal_sandbox(SimpleNamespace(exec=SimpleNamespace(aio=aio)))
    sandbox._image_env = modal_impl.modal_image_env(f"FROM a\nENV PYTHONPATH={image}\n")
    with sandbox.task_command():
        result = await sandbox.exec('printf "[%s]" "${PYTHONPATH-unset}"', user="root")
    assert result.stdout == f"[{seen}]", result


def _done(done):
    async def read(name):
        return getattr(done, name)

    async def wait():
        return done.returncode

    return SimpleNamespace(
        stdout=SimpleNamespace(read=SimpleNamespace(aio=lambda: read("stdout"))),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=lambda: read("stderr"))),
        wait=SimpleNamespace(aio=wait),
    )


# Gap 3: Docker's open-file and stack limits, opt-in.

_HELD = [
    "/usr/bin/prlimit",
    "--nofile=1024:524288",
    "--stack=8388608:unlimited",
    "--",
    "/usr/bin/timeout",
    "--foreground",
    "0",
    "/usr/bin/setpriv",
    "--pdeathsig",
    "KILL",
    "/usr/bin/prlimit",
    "--nofile=1024:524288",
    "--stack=8388608:unlimited",
    "--",
]


@pytest.mark.parametrize(
    ("raw", "limits"),
    [
        (None, None),
        ("", None),
        ("0", None),
        ("off", None),
        ("1", modal_impl.DOCKER_RLIMITS),
        ("docker", modal_impl.DOCKER_RLIMITS),
        ("nofile=4096:8192", {"nofile": (4096, 8192)}),
        (
            "nofile=1024:524288, stack=unlimited:unlimited",
            {"nofile": (1024, 524288), "stack": (None, None)},
        ),
    ],
)
def test_the_rlimits_setting(raw, limits):
    environ = {} if raw is None else {"BENCHFLOW_MODAL_RLIMITS": raw}
    assert modal_impl.modal_rlimits(environ) == limits


@pytest.mark.parametrize(
    "raw",
    [
        "2",
        "cpu=1:1",
        "nofile=2:1",
        "nofile=unlimited:5",
        "nofile=1",
        "nofile=1:2,nofile=1:2",
    ],
)
def test_an_unreadable_rlimits_setting_is_refused(raw):
    with pytest.raises(ValueError, match="BENCHFLOW_MODAL_RLIMITS"):
        modal_impl.modal_rlimits({"BENCHFLOW_MODAL_RLIMITS": raw})


def test_the_held_exec_prefix():
    """The payload is a forked child (timeout) with its own copy of the
    limits, which no later exec can reset; setpriv ends it with its waiter."""
    argv = ["/bin/bash", "-c", "true"]
    assert modal_impl.held_argv(argv, modal_impl.DOCKER_RLIMITS) == [*_HELD, *argv]


def test_the_limits_file_text():
    assert modal_impl.rlimits_conf(modal_impl.DOCKER_RLIMITS) == (
        "# BenchFlow (BENCHFLOW_MODAL_RLIMITS): the limits su and runuser give\n"
        "*\tsoft\tnofile\t1024\n"
        "*\thard\tnofile\t524288\n"
        "*\tsoft\tstack\t8192\n"
        "*\thard\tstack\tunlimited\n"
        "root\tsoft\tnofile\t1024\n"
        "root\thard\tnofile\t524288\n"
        "root\tsoft\tstack\t8192\n"
        "root\thard\tstack\tunlimited\n"
    )


def _run_conf_command(root):
    import subprocess

    command = modal_impl._rlimits_conf_command(modal_impl.DOCKER_RLIMITS).replace(
        "/etc/security", f"{root}/etc/security"
    )
    return subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)


def test_the_limits_file_is_written(tmp_path):
    (tmp_path / "etc/security/limits.d").mkdir(parents=True)
    (tmp_path / "etc/security/limits.conf").write_text(
        "# defaults\n*  soft  core  0\n@staff hard nproc 50\n"
    )
    done = _run_conf_command(tmp_path)
    assert done.returncode == 0, done
    written = tmp_path / "etc/security/limits.d/benchflow-rlimits.conf"
    assert written.read_text() == modal_impl.rlimits_conf(modal_impl.DOCKER_RLIMITS)
    assert oct(written.stat().st_mode & 0o777) == "0o644"


def test_an_image_whose_limits_config_names_one_is_refused(tmp_path):
    (tmp_path / "etc/security/limits.d").mkdir(parents=True)
    (tmp_path / "etc/security/limits.d/90-app.conf").write_text(
        "*   soft   nofile   65536\n"
    )
    done = _run_conf_command(tmp_path)
    assert done.returncode == 3
    assert "90-app.conf: * soft nofile 65536" in done.stdout
    assert not (tmp_path / "etc/security/limits.d/benchflow-rlimits.conf").exists()


_DOCKERS = (
    "Max open files            1024                 524288               files\n"
    "Max stack size            8388608              unlimited            bytes\n"
)
_MODALS = (
    "Max open files            1048576              1048576              files\n"
    "Max stack size            unlimited            unlimited            bytes\n"
)


def test_the_su_check_reads_each_route():
    limits = modal_impl.DOCKER_RLIMITS
    good = f"== su nobody rc=0\n{_DOCKERS}== runuser nobody rc=0\n{_DOCKERS}"
    assert modal_impl.rlimit_check_problems(good, limits) == []
    bad = f"== su nobody rc=0\n{_MODALS}== su root rc=1\nsu: Authentication failure\n"
    assert modal_impl.rlimit_check_problems(bad, limits) == [
        "su root failed (exit status 1)",
        "su nobody gives the nofile limit (1048576, 1048576), not (1024, 524288)",
        "su nobody gives the stack limit (None, None), not (8388608, None)",
    ]


def _exec_result(stdout="", stderr="", code=0):
    from benchflow.sandbox._base import ExecResult

    return ExecResult(stdout=stdout, stderr=stderr, return_code=code)


@pytest.mark.asyncio
async def test_no_setting_holds_nothing(monkeypatch, fake_secret):
    from unittest.mock import AsyncMock

    monkeypatch.delenv("BENCHFLOW_MODAL_RLIMITS", raising=False)
    remote = _remote(exec_=_ok)
    sandbox = _modal_sandbox(remote)
    sandbox.exec = AsyncMock()
    await sandbox._hold_rlimits()
    assert sandbox._rlimits is None
    assert remote.exec.calls == [] and not sandbox.exec.await_count
    assert sandbox.hold_argv(["true"]) == ["true"]


@pytest.mark.asyncio
async def test_the_setting_holds_every_later_exec(monkeypatch, fake_secret):
    """Every step is a root command on the system-shell route; the file is
    written before the prefix holds, and the prefix's first command proves
    it runs in the image."""
    monkeypatch.setenv("BENCHFLOW_MODAL_RLIMITS", "1")
    remote = _remote(exec_=_ok)
    sandbox = _modal_sandbox(remote)
    su_check = f"== su nobody rc=0\n{_DOCKERS}== su root rc=0\n{_DOCKERS}"
    seen = []

    async def exec_(command, **kwargs):
        seen.append((command, kwargs, sandbox._rlimits))
        return _exec_result(su_check if "/proc/self/limits" in command else "")

    sandbox.exec = exec_
    await sandbox._hold_rlimits()
    (written, *_), (proof, *_), (checked, *_) = seen
    assert "benchflow-rlimits.conf" in written and seen[0][2] is None
    assert proof == "true" and seen[1][2] == modal_impl.DOCKER_RLIMITS
    assert "su -s /bin/sh" in checked
    for _command, kwargs, _held in seen:
        assert kwargs == {"user": "root", "system_shell": True, "timeout_sec": 60}
    assert sandbox._rlimits == modal_impl.DOCKER_RLIMITS
    # Every exec from now on: the route's argv behind the prefix.
    del sandbox.exec
    await ModalSandbox.exec(sandbox, "true", user="root")
    (held, _kwargs) = remote.exec.calls[-1]
    sandbox._rlimits = None
    await ModalSandbox.exec(sandbox, "true", user="root")
    (plain, _kwargs) = remote.exec.calls[-1]
    assert list(held) == [*_HELD, *plain]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("results", "said"),
    [
        ([_exec_result("x.conf: * soft nofile 9", code=3)], "sets one of these"),
        ([_exec_result(), _exec_result(code=127)], "could not run"),
        ([_exec_result(), RuntimeError("no such file: prlimit")], "could not run"),
        (
            [
                _exec_result(),
                _exec_result(),
                _exec_result(f"== su nobody rc=0\n{_MODALS}"),
            ],
            "su nobody",
        ),
    ],
)
async def test_a_sandbox_that_cannot_hold_them_is_refused_and_ended(
    monkeypatch, fake_secret, results, said
):
    from unittest.mock import AsyncMock

    monkeypatch.setenv("BENCHFLOW_MODAL_RLIMITS", "1")
    remote = _remote(exec_=_ok)
    sandbox = _modal_sandbox(remote)
    sandbox.exec = AsyncMock(side_effect=results)
    with pytest.raises(modal_impl.ModalRlimitsRefused, match=said) as raised:
        await sandbox._hold_rlimits()
    assert str(raised.value).startswith(
        "sandbox startup refused: BENCHFLOW_MODAL_RLIMITS"
    )
    assert len(remote.terminate.calls) == 1
    assert sandbox._sandbox is None


@pytest.mark.asyncio
async def test_the_agent_process_runs_behind_the_held_prefix(monkeypatch):
    import sys
    import types

    from benchflow.sandbox.process.modal import ModalProcess

    fake = types.ModuleType("modal")
    fake.Secret = SimpleNamespace(from_dict=lambda d: d)
    monkeypatch.setitem(sys.modules, "modal", fake)

    async def route(_env):
        return None

    monkeypatch.setattr("benchflow.sandbox.process.modal.launch_route", route)
    env = SimpleNamespace(
        hold_argv=lambda argv: modal_impl.held_argv(argv, modal_impl.DOCKER_RLIMITS)
    )
    remote = SimpleNamespace(exec=_Aio(lambda *a, **k: _ended_agent_process()))
    process = ModalProcess(remote, env=env)
    await process.start("agent --acp")
    (args, _kwargs) = remote.exec.calls[0]
    assert list(args[: len(_HELD)]) == _HELD
    assert args[len(_HELD) : len(_HELD) + 4] == (
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-p",
    )
    await process.close()


@pytest.mark.asyncio
async def test_rlimits_are_off_by_default_on_both_launchers(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCHFLOW_MODAL_RLIMITS", raising=False)
    for factory in (False, True):
        folder = tmp_path / str(factory)
        folder.mkdir()
        await _create_kwargs(folder, monkeypatch, factory=factory)


def _separate_verifier_task(tmp_path):
    """A GPU task whose verifier has a sandbox of its own: agent 1800 s,
    verifier 900 s, setup 600 s (the review's example)."""
    from benchflow.task import Task

    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "verifier").mkdir()
    (task_dir / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "task.md").write_text(
        '---\nschema_version: "1.3"\n'
        "agent:\n  timeout_sec: 1800\n"
        "verifier:\n  timeout_sec: 900\n  sandbox_mode: separate\n"
        "sandbox:\n  gpus: 1\n  setup_timeout_sec: 600\n"
        "---\n\n## prompt\n\nSolve it.\n"
    )
    return Task(task_dir), task_dir


@pytest.mark.parametrize(
    ("lifetime", "agent_refused", "verifier_refused"),
    [("3000", False, False), ("2000", True, False), ("1200", True, True)],
)
def test_each_sandbox_of_a_separate_verifier_task_is_checked_alone(
    tmp_path, monkeypatch, lifetime, agent_refused, verifier_refused
):
    """Guards PR #1248's first plan-gate check against the review finding
    that it refused the separate verifier's sandbox: that launch's task view
    (_separate_verifier._verifier_task) keeps the finished agent's timeout,
    so 1800 + 900 + 600 s was counted against a sandbox that runs 900 + 600."""
    from benchflow.rollout._separate_verifier import _verifier_task
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task import UnsupportedTaskFeatureError

    monkeypatch.setenv("BENCHFLOW_MODAL_SANDBOX_TIMEOUT", lifetime)
    task, task_dir = _separate_verifier_task(tmp_path)
    plan = SimpleNamespace(sandbox=task.config.sandbox, source="Dockerfile")
    for view, refused, path in (
        (task, agent_refused, "agent.timeout_sec"),
        (_verifier_task(task, plan), verifier_refused, "verifier.timeout_sec"),
    ):
        if not refused:
            _validate_task_runtime_for_launch(
                view, sandbox_type="modal", task_path=task_dir, check_compose=False
            )
            continue
        with pytest.raises(UnsupportedTaskFeatureError) as raised:
            _validate_task_runtime_for_launch(
                view, sandbox_type="modal", task_path=task_dir, check_compose=False
            )
        assert [f.path for f in raised.value.features] == [path]


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        # A comment line inside a continued ENV is skipped, as Docker skips it.
        (
            "FROM a\nENV A=1 \\\n# the seed\n    PYTHONHASHSEED=0\n",
            {"A": "1", "PYTHONHASHSEED": "0"},
        ),
        # A tab between the keyword and its arguments.
        ("FROM a\nENV\tPYTHONHASHSEED=0\n", {"PYTHONHASHSEED": "0"}),
        ("FROM a\nENV\tK\tV W\n", {"K": "V W"}),
        # A parent stage's ARG carries into the stage built FROM it...
        (
            "FROM python:3.12 AS base\nARG SEED=0\nFROM base\nENV PYTHONHASHSEED=$SEED\n",
            {"PYTHONHASHSEED": "0"},
        ),
        # ... and a re-declaration without a default keeps it.
        (
            "ARG SEED=9\nFROM a AS base\nARG SEED=0\nFROM base\nARG SEED\n"
            "ENV PYTHONHASHSEED=$SEED\n",
            {"PYTHONHASHSEED": "0"},
        ),
        # Another stage's ARG does not.
        ("FROM a AS one\nARG S=1\nFROM b\nENV X=$S\n", {"X": ""}),
        # \$ is a literal $, in an ENV value and in an ARG default; nothing in
        # single quotes is expanded.
        ("FROM a\nENV PYTHONPATH=/opt/\\$vendor\n", {"PYTHONPATH": "/opt/$vendor"}),
        ("FROM a\nARG P=/x/\\$y\nENV Q=$P\n", {"Q": "/x/$y"}),
        ("FROM a\nENV A='$HOME' B=\"\\$x\"\n", {"A": "$HOME", "B": "$x"}),
    ],
)
def test_the_dockerfile_env_reader_follows_docker(dockerfile, expected):
    """Guards PR #1248's Dockerfile reader against the review findings on
    comments in continuations, tabs, inherited ARGs and escaped dollars."""
    from benchflow.sandbox._dockerfile_env import final_stage_env

    assert final_stage_env(dockerfile) == expected


@pytest.mark.parametrize(
    "dockerfile",
    [
        # Another escape character: the reader does not follow the file.
        "# escape=`\nFROM a\nENV PYTHONHASHSEED=0\n",
        # A quote that never closes.
        'FROM a\nENV PYTHONPATH="/opt/lib\n',
    ],
)
def test_a_dockerfile_the_reader_cannot_follow_keeps_modals_values(dockerfile):
    """Fail safe: what the reader cannot tell keeps Modal's value; it never
    gives a value Docker would not."""
    from benchflow.sandbox._dockerfile_env import final_stage_env

    assert final_stage_env(dockerfile) is None
    image = modal_impl.modal_image_env(dockerfile)
    assert image.pythonpath is None
    assert image.create == {}
    assert {"PYTHONHASHSEED", "PYTHONPATH"} <= set(image.kept)
