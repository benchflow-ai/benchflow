"""How a Modal sandbox's end is named and classed (follow-up to #1245).

Measured with modal 1.6.2 on CPU sandboxes:

- a command running when the sandbox's lifetime ran out, or when the sandbox
  was terminated from outside, returned 137 rather than raising, and ``poll``
  answered None right after;
- ``poll`` then gave 124 with status ``TIMEOUT`` ("Sandbox terminated because
  it reached its timeout") for the lifetime, 137 with ``TERMINATED`` ("Sandbox
  terminated by user request") for the terminate, and 137 with ``FAILURE``
  (exit code 137, no message) for a process over a hard memory limit.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("tenacity")
pytest.importorskip("modal")

from modal_proto import api_pb2

from benchflow._utils.scoring import INFRA_ERROR, OTHER_ERROR, classify_error
from benchflow.sandbox import modal_impl
from benchflow.sandbox.modal_impl import (
    MODAL_ENDED,
    MODAL_LIFETIME_END,
    MODAL_MEMORY_KILL,
    ModalSandbox,
    ModalSandboxEndedError,
)


class _Aio:
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    async def aio(self, *args, **kwargs):
        self.calls += 1
        result = self._fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


def _result(status: str, exitcode: int, exception: str = ""):
    return api_pb2.GenericResult(
        status=getattr(api_pb2.GenericResult, f"GENERIC_STATUS_{status}"),
        exitcode=exitcode,
        exception=exception,
    )


def _killed_process(code=137):
    async def read():
        return ""

    async def wait():
        return code

    return SimpleNamespace(
        stdout=SimpleNamespace(read=SimpleNamespace(aio=read)),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=read)),
        wait=SimpleNamespace(aio=wait),
    )


def _remote(answers, result=None, code=137):
    """A Modal sandbox handle whose poll gives *answers* in turn (the last
    one again after that), keeping *result* once it answers a code."""
    remote = SimpleNamespace(object_id="sb-1", _result=None)
    queue = list(answers)

    def poll():
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if answer is not None:
            remote._result = result
        return answer

    remote.poll = _Aio(poll)
    remote.exec = _Aio(lambda *a, **k: _killed_process(code))
    remote.terminate = _Aio(lambda: None)
    remote.wait = _Aio(lambda raise_on_termination=True: None)
    return remote


def _sandbox(remote) -> ModalSandbox:
    sandbox = object.__new__(ModalSandbox)
    sandbox.default_user = "root"
    sandbox._persistent_env = {}
    sandbox._sandbox = remote
    sandbox._app = sandbox._image = None
    return sandbox


@pytest.fixture
def fake_secret(monkeypatch):
    import modal

    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda env: env))
    monkeypatch.setattr(modal_impl, "PLATFORM_END_SETTLE_STEP_SEC", 0.0)


@pytest.mark.parametrize(
    ("status", "code", "exception", "memory_kill", "category", "said"),
    [
        ("FAILURE", 137, "", True, OTHER_ERROR, "hard memory limit"),
        (
            "TIMEOUT",
            124,
            "Sandbox terminated because it reached its timeout",
            False,
            OTHER_ERROR,
            "lifetime ran out",
        ),
        (
            "TERMINATED",
            137,
            "Sandbox terminated by user request",
            False,
            INFRA_ERROR,
            "terminated by user request",
        ),
        ("MEMORY_MANAGER_EVICTION", 137, "", False, INFRA_ERROR, "eviction"),
        (None, 137, "", False, INFRA_ERROR, "memory limit, or Modal evicted"),
    ],
)
def test_modals_status_names_the_end(
    status, code, exception, memory_kill, category, said
):
    handle = SimpleNamespace(
        _result=_result(status, code if status != "TERMINATED" else 130, exception)
        if status
        else None
    )
    got_status, detail = modal_impl.modal_end_status(handle)
    assert got_status == status
    end = modal_impl.platform_end_record(code, got_status, detail, "command")
    assert end["memory_kill"] is memory_kill
    assert said in end["reason"]
    message = modal_impl.modal_ended_message(end, "an exec failed with it")
    start = (
        MODAL_MEMORY_KILL
        if memory_kill
        else MODAL_LIFETIME_END
        if status == "TIMEOUT"
        else MODAL_ENDED
    )
    assert message.startswith(start)
    assert classify_error(f"ModalSandboxEndedError: {message}") == category


def test_a_client_without_a_result_reads_as_unknown():
    assert modal_impl.modal_end_status(object()) == (None, "")


@pytest.mark.asyncio
async def test_a_command_killed_by_the_lifetime_end_is_not_rerun_as_infrastructure(
    fake_secret,
):
    """The issue's repro as measured: the 120 s command in a 60 s sandbox
    returned 137, and poll answered None before it gave 124. A rerun would
    meet the same lifetime, so the end is named and not retried (as ruled in
    review)."""
    remote = _remote(
        [None, None, 124],
        _result("TIMEOUT", 0, "Sandbox terminated because it reached its timeout"),
    )
    sandbox = _sandbox(remote)
    with pytest.raises(ModalSandboxEndedError) as raised:
        await sandbox.exec("sleep 120", user="root")
    message = str(raised.value)
    assert message.startswith(MODAL_LIFETIME_END)
    assert "returncode 124" in message and "smaller budgets" in message
    assert sandbox.platform_end["status"] == "TIMEOUT"
    assert sandbox.platform_end["lifetime"] is True
    assert classify_error(message) != INFRA_ERROR
    assert remote.poll.calls == 3


@pytest.mark.asyncio
async def test_a_memory_kill_is_the_tasks_not_the_infrastructures(fake_secret):
    remote = _remote([137], _result("FAILURE", 137))
    sandbox = _sandbox(remote)
    with pytest.raises(ModalSandboxEndedError) as raised:
        await sandbox.exec("python3 hog.py", user="root")
    message = str(raised.value)
    assert message.startswith(MODAL_MEMORY_KILL)
    assert "sandbox not found" not in message.lower()
    assert classify_error(message) == OTHER_ERROR
    assert sandbox.platform_end["memory_kill"] is True


@pytest.mark.asyncio
async def test_a_command_killed_in_a_running_sandbox_returns_137_as_before(
    monkeypatch, fake_secret
):
    monkeypatch.setattr(modal_impl, "PLATFORM_END_KILLED_SETTLE_SEC", 0.05)
    monkeypatch.setattr(modal_impl, "PLATFORM_END_SETTLE_STEP_SEC", 0.01)
    remote = _remote([None])
    sandbox = _sandbox(remote)
    result = await sandbox.exec("kill -9 $$", user="root")
    assert result.return_code == 137
    assert sandbox.platform_end is None
    assert remote.poll.calls > 1


@pytest.mark.asyncio
async def test_other_exit_codes_ask_nothing(fake_secret):
    remote = _remote([124], code=1)
    sandbox = _sandbox(remote)
    result = await sandbox.exec("false", user="root")
    assert result.return_code == 1
    assert remote.poll.calls == 0


class _Chunks:
    def __aiter__(self):
        async def gen():
            if False:
                yield b""

        return gen()


def _agent_exec(code):
    return SimpleNamespace(
        stdout=_Chunks(),
        stderr=_Chunks(),
        stdin=SimpleNamespace(
            write=lambda data: None,
            write_eof=lambda: None,
            drain=_Aio(lambda: None),
        ),
        wait=_Aio(lambda: code),
        poll=_Aio(lambda: code),
    )


class _Env:
    def __init__(self, end):
        self._end = end
        self.settle: list[float] = []

    async def check_platform_end(self, seen_at, *, settle_sec=0.0):
        self.settle.append(settle_sec)
        return self._end


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_kill", [False, True], ids=["evicted", "memory-kill"])
async def test_an_agent_process_killed_with_its_sandbox_names_the_end(
    monkeypatch, memory_kill
):
    import sys
    import types

    from benchflow.diagnostics import TransportClosedError
    from benchflow.sandbox.process.modal import ModalProcess

    fake = types.ModuleType("modal")
    fake.Secret = SimpleNamespace(from_dict=lambda d: d)
    monkeypatch.setitem(sys.modules, "modal", fake)
    end = modal_impl.platform_end_record(
        137,
        "FAILURE" if memory_kill else "TERMINATED",
        "" if memory_kill else "Sandbox terminated by user request",
        "agent process",
    )
    env = _Env(end)
    remote = SimpleNamespace(exec=_Aio(lambda *a, **k: _agent_exec(137)))
    process = ModalProcess(remote, env=env)

    async def route(_env):
        return None

    monkeypatch.setattr("benchflow.sandbox.process.modal.launch_route", route)
    await process.start("agent --acp")
    with pytest.raises(TransportClosedError) as raised:
        await process.readline()
    message = str(raised.value)
    assert env.settle == [modal_impl.PLATFORM_END_KILLED_SETTLE_SEC]
    if memory_kill:
        assert message.startswith(MODAL_MEMORY_KILL)
        assert classify_error(message) != INFRA_ERROR
        assert raised.value.diagnostic.transport_diagnosis == "process_exited"
    else:
        assert message.startswith(MODAL_ENDED)
        assert classify_error(message) == INFRA_ERROR
        assert raised.value.diagnostic.transport_diagnosis == "remote_session_killed"
    await process.close()


@pytest.mark.asyncio
async def test_an_agent_process_at_the_lifetime_end_is_not_rerun_as_infrastructure(
    monkeypatch,
):
    """The agent's process killed when the lifetime ran out: the end is
    named, the transport reads as the process's own exit, and the trial is
    not rerun as an infrastructure error (as ruled in review)."""
    import sys
    import types

    from benchflow.diagnostics import TransportClosedError
    from benchflow.sandbox.process.modal import ModalProcess

    fake = types.ModuleType("modal")
    fake.Secret = SimpleNamespace(from_dict=lambda d: d)
    monkeypatch.setitem(sys.modules, "modal", fake)
    end = modal_impl.platform_end_record(
        124,
        "TIMEOUT",
        "Sandbox terminated because it reached its timeout",
        "agent process",
    )
    env = _Env(end)
    remote = SimpleNamespace(exec=_Aio(lambda *a, **k: _agent_exec(137)))
    process = ModalProcess(remote, env=env)

    async def route(_env):
        return None

    monkeypatch.setattr("benchflow.sandbox.process.modal.launch_route", route)
    await process.start("agent --acp")
    with pytest.raises(TransportClosedError) as raised:
        await process.readline()
    message = str(raised.value)
    assert message.startswith(MODAL_LIFETIME_END), message
    assert classify_error(message) != INFRA_ERROR
    assert raised.value.diagnostic.transport_diagnosis == "process_exited"
    await process.close()
