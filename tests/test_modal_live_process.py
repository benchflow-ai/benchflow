"""Modal hosts ACP agents through ``Sandbox.exec`` stdio.

Guards the 0.9 port of 96d225614 and 2f74fc42f: before
it, ``ModalSandbox.live_process`` was BaseSandbox's refusal and plan
validation refused every ACP and native agent on Modal.

A fake Modal sandbox stands in for the real one: its ``exec.aio`` returns a
process whose stdout and stderr are async iterators of byte chunks and whose
stdin records what is written, so the transport's contract is checked without
a Modal account. What the launch runs is checked on a real shell in
``tests/test_agent_launch_root_path.py``.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import types

import pytest

from benchflow.sandbox import process as process_pkg
from benchflow.sandbox._base import SYSTEM_PATH, SYSTEM_SHELL_EMPTY_VARS, ExecRoute
from benchflow.sandbox.process import modal as modal_process
from benchflow.sandbox.process.modal import ModalProcess
from tests._trust_admission import admitted


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *a, **k):
        return self._fn(*a, **k)

    async def aio(self, *a, **k):
        result = self._fn(*a, **k)
        if asyncio.iscoroutine(result):
            return await result
        return result


class _Chunks:
    """An async iterable of chunks: *gate* holds the first one back, *hold*
    keeps the stream open after the last (a live agent), *fail* ends it with
    an error."""

    def __init__(self, chunks, gate=None, hold=None, fail=None):
        self._chunks = list(chunks)
        self._gate = gate
        self._hold = hold
        self._fail = fail

    def __aiter__(self):
        async def gen():
            if self._gate is not None:
                await self._gate.wait()
            for chunk in self._chunks:
                await asyncio.sleep(0)
                yield chunk
            if self._hold is not None:
                await self._hold.wait()
            if self._fail is not None:
                raise self._fail

        return gen()


class _Stdin:
    def __init__(self):
        self.writes: list[bytes] = []
        self.eof = False
        self.drains = 0
        self.drain = _Aio(self._drain)

    def _drain(self):
        self.drains += 1

    def write(self, b):
        self.writes.append(bytes(b))

    def write_eof(self):
        self.eof = True

    @property
    def data(self) -> bytes:
        return b"".join(self.writes)


class _Proc:
    def __init__(self, out=(), err=(), rc=0, gate=None, hold=None, fail=None):
        self.stdout = _Chunks(out, gate=gate, hold=hold, fail=fail)
        self.stderr = _Chunks(err)
        self.stdin = _Stdin()
        self.wait = _Aio(lambda: rc)
        self.poll = _Aio(lambda: rc)


class _Sandbox:
    def __init__(self, proc):
        self.proc = proc
        self.calls = []

        def _exec(*args, **kwargs):
            self.calls.append((args, kwargs))
            return self.proc

        self.exec = _Aio(_exec)


@pytest.fixture(autouse=True)
def fake_modal(monkeypatch):
    mod = types.ModuleType("modal")

    class Secret:
        def __init__(self, d):
            self.d = d

        @classmethod
        def from_dict(cls, d):
            return cls(dict(d))

    mod.Secret = Secret
    monkeypatch.setitem(sys.modules, "modal", mod)
    return mod


def _env_line(stdin: _Stdin) -> str:
    """The env line the launch reads first, decoded."""
    first, _, _ = stdin.data.partition(b"\n")
    return base64.b64decode(first).decode()


def test_exported():
    assert process_pkg.ModalProcess is ModalProcess


@pytest.mark.asyncio
async def test_launch_is_roots_bootstrap_and_env_travels_on_stdin():
    proc = _Proc(
        out=[b'{"jsonrpc": "2.0", "id": 1, "result": {}}\nsec', b"ond\n"],
        hold=asyncio.Event(),
    )
    sb = _Sandbox(proc)
    lp = ModalProcess(sb)
    await lp.start("agent --acp", env={"API_KEY": "secret-value"}, cwd="/app")

    args, kwargs = sb.calls[0]
    # Root's bash by absolute path, no profile, rc or BASH_ENV file (C8).
    assert args[:5] == ("/bin/bash", "--noprofile", "--norc", "-p", "-c")
    script = args[5]
    assert len(args) == 6
    # The env line is read, then the route's PATH pinned, before the command.
    assert "IFS= read -r __bf_live_env" in script
    assert f"PATH={SYSTEM_PATH}" in script
    assert script.index("read -r __bf_live_env") < script.index("agent --acp")
    # No value on the command line or in the Modal Secret: only the loader
    # and start-up names, empty.
    assert not any("secret-value" in arg for arg in args)
    assert [s.d for s in kwargs["secrets"]] == [
        dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, "")
    ]
    assert kwargs["workdir"] == "/app"
    assert kwargs["text"] is False
    assert _env_line(proc.stdin) == ("export API_KEY=secret-value\n")

    assert lp.is_running
    assert await lp.readline() == b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n'
    # A line cut across chunks comes back whole.
    assert await lp.readline() == b"second\n"
    await lp.writeline('{"jsonrpc": "2.0", "id": 2}')
    assert proc.stdin.data.endswith(b'\n{"jsonrpc": "2.0", "id": 2}\n')
    await lp.close()
    assert proc.stdin.eof and not lp.is_running
    await lp.close()  # idempotent


@pytest.mark.asyncio
async def test_the_last_line_without_a_newline_is_read_before_the_close():
    """As read_whole_line: the partial last line, then the closed stream."""
    from benchflow.diagnostics import TransportClosedError

    lp = ModalProcess(_Sandbox(_Proc(out=[b"done", b" now"])))
    await lp.start("agent")
    assert await lp.readline() == b"done now"
    with pytest.raises(TransportClosedError):
        await lp.readline()


@pytest.mark.asyncio
async def test_launch_env_loader_names_are_inert_until_the_drop():
    """PT-C8-3 on Modal: an LD_PRELOAD in the launch env never reaches root."""
    proc = _Proc()
    sb = _Sandbox(proc)
    await ModalProcess(sb).start(
        "agent", env={"LD_PRELOAD": "/app/lib/x.so", "NODE_OPTIONS": "--inspect"}
    )
    exports = _env_line(proc.stdin)
    assert "export _BENCHFLOW_SET_LD_PRELOAD=/app/lib/x.so" in exports
    assert "export _BENCHFLOW_SET_NODE_OPTIONS=--inspect" in exports
    assert "export LD_PRELOAD=" not in exports
    assert sb.calls[0][1]["secrets"][0].d["LD_PRELOAD"] == ""


@pytest.mark.asyncio
async def test_invalid_env_name_is_refused_before_any_exec():
    sb = _Sandbox(_Proc())
    with pytest.raises(ValueError, match="Invalid Modal environment variable name"):
        await ModalProcess(sb).start("agent", env={"BAD-NAME": "x"})
    assert sb.calls == []


@pytest.mark.asyncio
async def test_no_env_means_no_env_line():
    proc = _Proc(out=[b"x\n"])
    sb = _Sandbox(proc)
    lp = ModalProcess(sb)
    await lp.start("agent")
    assert "__bf_live_env" not in sb.calls[0][0][5]
    assert proc.stdin.data == b""
    assert await lp.readline() == b"x\n"


@pytest.mark.asyncio
async def test_launch_uses_the_routes_bash_and_path():
    """On the protected kit, root's shell and PATH are the kit's (C8)."""
    kit_path = "/opt/benchflow/kit/bin:" + SYSTEM_PATH
    route = ExecRoute(
        pin=kit_path,
        task_root=False,
        system_shell=True,
        bash="/opt/benchflow/kit/bin/bash",
        tools=kit_path,
        kit=True,
    )

    class _Env:
        _sandbox = None

        async def exec_route(self, user, *, system_shell=False, service="main"):
            assert (user, system_shell, service) == ("root", True, "main")
            return route

    bf_env = _Env()
    bf_env._sandbox = _Sandbox(_Proc())
    lp = ModalProcess.from_sandbox_env(bf_env)
    await lp.start("agent")
    args = bf_env._sandbox.calls[0][0]
    assert args[0] == "/opt/benchflow/kit/bin/bash"
    assert f"PATH={kit_path}" in args[5]


@pytest.mark.asyncio
async def test_closed_stdout_raises_transport_closed_with_exit_code_and_stderr():
    from benchflow.diagnostics import TransportClosedError

    proc = _Proc(err=[b"boom: agent failed\n"], rc=1)
    lp = ModalProcess(_Sandbox(proc))
    await lp.start("agent")
    with pytest.raises(TransportClosedError) as excinfo:
        await lp.readline()
    assert "rc=1" in str(excinfo.value)
    assert "boom" in str(excinfo.value)
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.process_exit_code == 1
    assert diagnostic.transport_diagnosis == "process_exited"
    assert "boom" in (diagnostic.stderr_snippet or "")
    assert not lp.is_running
    # Every later read, and a write, says the same.
    with pytest.raises(TransportClosedError, match="rc=1"):
        await lp.readline()
    with pytest.raises(TransportClosedError, match="rc=1"):
        await lp.writeline("{}")


@pytest.mark.asyncio
async def test_a_failed_stdout_stream_is_a_lost_session():
    from benchflow.diagnostics import TransportClosedError

    proc = _Proc(out=[b"one\n"], fail=ConnectionError("stream reset"), rc=None)
    lp = ModalProcess(_Sandbox(proc))
    await lp.start("agent")
    assert await lp.readline() == b"one\n"
    with pytest.raises(TransportClosedError) as excinfo:
        await lp.readline()
    assert excinfo.value.diagnostic.transport_diagnosis == "remote_session_killed"
    assert "stream reset" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_cancelled_read_loses_no_line():
    """The reader task owns the stream: a read the caller cancels (a watchdog,
    a turn's timeout) leaves the next line for the next read."""
    gate = asyncio.Event()
    lp = ModalProcess(_Sandbox(_Proc(out=[b"late\n"], gate=gate)))
    await lp.start("agent")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(lp.readline(), timeout=0.05)
    gate.set()
    assert await asyncio.wait_for(lp.readline(), timeout=5) == b"late\n"


@pytest.mark.asyncio
async def test_a_large_message_is_written_in_bounded_chunks(monkeypatch):
    monkeypatch.setattr(modal_process, "_STDIN_CHUNK", 8)
    proc = _Proc()
    lp = ModalProcess(_Sandbox(proc))
    await lp.start("agent")
    await lp.writeline("x" * 20)
    assert [len(w) for w in proc.stdin.writes] == [8, 8, 5]
    assert proc.stdin.data == b"x" * 20 + b"\n"
    assert proc.stdin.drains == 3


@pytest.mark.asyncio
async def test_an_oversized_line_is_skipped_whole(monkeypatch):
    monkeypatch.setattr(modal_process, "_OVERSIZED_LINE_CAP", 8)
    lp = ModalProcess(_Sandbox(_Proc(out=[b"ok\n", b"0123456789", b"abc\nnext\n"])))
    await lp.start("agent")
    assert await lp.readline() == b"ok\n"
    assert await lp.readline() == b"next\n"


@pytest.mark.asyncio
async def test_unstarted_process_refuses_io():
    lp = ModalProcess(object())
    with pytest.raises(RuntimeError):
        await lp.readline()
    with pytest.raises(RuntimeError):
        await lp.writeline("x")
    assert not lp.is_running


@pytest.mark.asyncio
async def test_modal_sandbox_live_process_uses_modal_transport():
    from benchflow.sandbox.modal_impl import ModalSandbox

    env = types.SimpleNamespace(_sandbox=object())
    lp = await ModalSandbox.live_process(
        env, agent="codex-acp", admission=admitted(env)
    )
    assert isinstance(lp, ModalProcess)


@pytest.mark.asyncio
async def test_modal_sandbox_live_process_needs_a_started_sandbox():
    from benchflow.sandbox.modal_impl import ModalSandbox

    env = types.SimpleNamespace(_sandbox=None)
    with pytest.raises(RuntimeError, match="start the environment"):
        await ModalSandbox.live_process(env, admission=admitted(env))


def test_registry_says_modal_has_a_live_transport_and_still_no_denylist():
    from benchflow.sandbox.providers import PROVIDERS_BY_NAME

    modal = PROVIDERS_BY_NAME["modal"]
    assert modal.supports_live_process is True
    assert modal.enforces_denylist is False
    assert modal.enforces_allowlist is False
