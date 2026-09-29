"""A Daytona PTY read fails as soon as the websocket is gone, with its close code.

Guards #1144 and #1143. ``DaytonaPtyProcess.readline`` waited only on its
line queue, so a websocket the Daytona side closed (1006 after a reset, 1008
on an oversized message) or a path that died silently surfaced as ``PTY
readline timeout`` once the whole silence budget had passed: 900 s, or hours
after #1143 raised the budget. The reviewer's transport retry skips a
readline timeout by design, so a dead reviewer websocket was never retried.
The SDK's PTY handle knows at once (``AsyncPtyHandle.wait()`` returns when
its websocket reader ends); BenchFlow now asks it, and pings the websocket so
a silently dead path closes too.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox.process.daytona import DaytonaPtyProcess

_BOOTSTRAP_OK = MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")


class FakeWs:
    """The aiohttp websocket attributes BenchFlow reads for the close reason."""

    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self._exception: BaseException | None = None

    def exception(self) -> BaseException | None:
        return self._exception


class FakePtyHandle:
    """daytona.handle.async_pty_handle.AsyncPtyHandle, as far as BenchFlow uses it."""

    def __init__(self, on_data) -> None:
        self._on_data = on_data
        self._ws = FakeWs()
        self._done = asyncio.get_running_loop().create_future()
        self.exit_code: int | None = None
        self.error: str | None = None
        self.inputs: list[str] = []
        self.killed = False
        self.disconnected = False

    async def wait_for_connection(self) -> None:
        return None

    def is_connected(self) -> bool:
        return not self._ws.closed

    async def wait(self):
        await self._done
        return SimpleNamespace(exit_code=self.exit_code, error=self.error)

    async def send_input(self, data: str) -> None:
        if self._ws.closed:
            raise ConnectionError("PTY is not connected")
        self.inputs.append(data)
        if "echo '" in data:  # the start marker
            marker = data.split("echo '", 1)[1].split("'", 1)[0]
            await self._on_data(f"{marker}\n".encode())

    async def kill(self) -> None:
        self.killed = True

    async def disconnect(self) -> None:
        self.disconnected = True
        if not self._done.done():
            self._done.cancel()

    # What the SDK's websocket reader does when the channel ends.
    def server_close(
        self,
        close_code: int,
        *,
        socket_error: BaseException | None = None,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        self._ws.closed = True
        self._ws.close_code = close_code
        self._ws._exception = socket_error
        self.exit_code = exit_code
        self.error = error
        self._done.set_result(None)


def _sandbox(handles: list[FakePtyHandle]):
    sandbox = MagicMock()

    async def create_pty_session(*, id, on_data, envs=None):
        handle = FakePtyHandle(on_data)
        handles.append(handle)
        return handle

    sandbox.process.create_pty_session = AsyncMock(side_effect=create_pty_session)
    sandbox.process.exec = AsyncMock(return_value=_BOOTSTRAP_OK)
    return sandbox


async def _started(monkeypatch) -> tuple[DaytonaPtyProcess, FakePtyHandle]:
    monkeypatch.delenv("BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT", raising=False)
    handles: list[FakePtyHandle] = []
    proc = DaytonaPtyProcess(_sandbox(handles), "", "docker compose -p t")
    await proc.start(command="codex-acp")
    proc.expect_silence(7200)  # a long idle budget: the read guard is hours
    return proc, handles[0]


async def test_a_closed_websocket_fails_the_read_at_once_with_its_close_code(
    monkeypatch,
):
    proc, pty = await _started(monkeypatch)
    reader = asyncio.ensure_future(proc.readline())
    await asyncio.sleep(0)
    assert not reader.done()
    pty.server_close(
        1006, socket_error=ConnectionResetError("Cannot write to closing transport")
    )
    with pytest.raises(TransportClosedError) as caught:
        await asyncio.wait_for(reader, timeout=2)
    diagnostic = caught.value.diagnostic
    assert str(caught.value).startswith("PTY closed by the peer: websocket closed")
    assert "close_code=1006" in diagnostic.raw_message
    assert "Cannot write to closing transport" in diagnostic.raw_message
    assert diagnostic.transport_diagnosis == "pty_closed"
    assert "readline timeout" not in diagnostic.raw_message
    assert not proc.is_running
    await proc.close()


async def test_lines_sent_before_the_close_are_read_first(monkeypatch):
    proc, pty = await _started(monkeypatch)
    await pty._on_data(b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n')
    pty.server_close(1000, exit_code=0)
    assert await proc.readline() == b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n'
    with pytest.raises(TransportClosedError, match="agent process exited with code 0"):
        await proc.readline()
    await proc.close()


async def test_the_sdk_error_and_exit_code_name_the_cause(monkeypatch):
    proc, pty = await _started(monkeypatch)
    pty.server_close(1008, exit_code=137, error="message too big")
    with pytest.raises(TransportClosedError) as caught:
        await proc.readline()
    assert caught.value.diagnostic.raw_message == (
        "PTY closed by the peer: message too big (close_code=1008)"
    )
    assert caught.value.diagnostic.process_exit_code == 137
    await proc.close()


async def test_writing_to_a_closed_websocket_is_a_transport_loss(monkeypatch):
    proc, pty = await _started(monkeypatch)
    pty.server_close(1006)
    with pytest.raises(TransportClosedError, match="close_code=1006"):
        await proc.writeline('{"jsonrpc": "2.0"}')
    await proc.close()


async def test_a_close_during_startup_is_reported_without_the_marker_timeout(
    monkeypatch,
):
    handles: list[FakePtyHandle] = []
    proc = DaytonaPtyProcess(_sandbox(handles), "", "docker compose -p t")
    original = FakePtyHandle.send_input

    async def close_instead_of_marker(self, data):
        if "echo '" in data:
            self.server_close(1006)
            return
        await original(self, data)

    monkeypatch.setattr(FakePtyHandle, "send_input", close_instead_of_marker)
    with pytest.raises(TransportClosedError, match="PTY closed by the peer"):
        await asyncio.wait_for(proc.start(command="codex-acp"), timeout=5)
    assert handles[0].disconnected


async def test_our_own_close_is_not_reported_as_the_peers(monkeypatch):
    proc, _pty = await _started(monkeypatch)
    await proc.close()
    with pytest.raises(TransportClosedError) as caught:
        await proc.readline()
    assert caught.value.diagnostic.raw_message == "PTY closed"


async def test_a_handle_without_wait_keeps_the_plain_read(monkeypatch):
    """Other SDK layouts (and older fakes) read the queue as before."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT", "0.05")
    handles: list[FakePtyHandle] = []
    proc = DaytonaPtyProcess(_sandbox(handles), "", "docker compose -p t")
    monkeypatch.setattr(FakePtyHandle, "wait", None)
    await proc.start(command="codex-acp")
    with pytest.raises(TransportClosedError, match="readline timeout"):
        await proc.readline()
    await proc.close()


# ---------------------------------------------------------------------------
# The real SDK handle over a real websocket, with the heartbeat BenchFlow sets
# ---------------------------------------------------------------------------


async def test_a_silently_dead_channel_is_detected_by_the_heartbeat(monkeypatch):
    """A websocket whose peer stops answering pings (a path that died without
    a close frame) closes after the heartbeat, and the read fails at once."""
    aiohttp = pytest.importorskip("aiohttp")
    web = pytest.importorskip("aiohttp.web")
    pty_module = pytest.importorskip("daytona.handle.async_pty_handle")

    async def handler(request):
        ws = web.WebSocketResponse(autoping=False)  # never answers a ping
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "control", "status": "connected"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and b"echo '" in msg.data:
                marker = msg.data.split(b"echo '", 1)[1].split(b"'", 1)[0]
                await ws.send_bytes(marker + b"\n")
            elif msg.type == aiohttp.WSMsgType.BINARY and b"exec sh" in msg.data:
                await ws.send_bytes(b"agent line\n")
        return ws

    from aiohttp.test_utils import unused_port

    app = web.Application()
    app.router.add_get("/pty", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = aiohttp.ClientSession()
    try:
        sandbox = MagicMock()

        async def create_pty_session(*, id, on_data, envs=None):
            ws = await session.ws_connect(f"http://127.0.0.1:{port}/pty", heartbeat=0.2)
            handle = pty_module.AsyncPtyHandle(ws, on_data, session_id=id)
            await handle.wait_for_connection()
            return handle

        sandbox.process.create_pty_session = create_pty_session
        sandbox.process.exec = AsyncMock(return_value=_BOOTSTRAP_OK)
        monkeypatch.delenv("BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT", raising=False)
        proc = DaytonaPtyProcess(sandbox, "", "docker compose -p t")
        await proc.start(command="codex-acp")
        proc.expect_silence(7200)
        assert await asyncio.wait_for(proc.readline(), timeout=5) == b"agent line\n"
        with pytest.raises(TransportClosedError) as caught:
            await asyncio.wait_for(proc.readline(), timeout=5)
        assert caught.value.diagnostic.transport_diagnosis == "pty_closed"
        assert "close_code=1006" in caught.value.diagnostic.raw_message
        await proc.close()
    finally:
        await session.close()
        await runner.cleanup()


def _fake_sdk_process(calls: list[dict]):
    """``daytona._async.process`` as far as the heartbeat patch reads it."""

    class Session:
        async def ws_connect(self, url, **kwargs):
            calls.append({"url": url, **kwargs})
            return "ws"

    module = SimpleNamespace(http_session_of=lambda client: Session())

    class AsyncProcess:
        def __init__(self) -> None:
            self._api_client = SimpleNamespace(api_client=object())

        async def _open_ws(self, url: str, headers: dict[str, str]):
            # The SDK's body (0.184): no heartbeat.
            session = module.http_session_of(self._api_client.api_client)
            return await session.ws_connect(url, headers=headers)

    module.AsyncProcess = AsyncProcess
    return module


async def test_the_sdk_opens_its_websockets_with_a_heartbeat(monkeypatch):
    from benchflow.sandbox import _sdk_ops

    calls: list[dict] = []
    sdk = _fake_sdk_process(calls)
    _sdk_ops._patch_websocket_heartbeat(sdk)
    _sdk_ops._patch_websocket_heartbeat(sdk)  # idempotent
    process = sdk.AsyncProcess()

    monkeypatch.delenv(_sdk_ops.DAYTONA_WS_HEARTBEAT_ENV, raising=False)
    assert await process._open_ws("wss://pty", {"a": "b"}) == "ws"
    monkeypatch.setenv(_sdk_ops.DAYTONA_WS_HEARTBEAT_ENV, "30")
    await process._open_ws("wss://pty", {})
    monkeypatch.setenv(_sdk_ops.DAYTONA_WS_HEARTBEAT_ENV, "0")
    await process._open_ws("wss://pty", {})
    monkeypatch.setenv(_sdk_ops.DAYTONA_WS_HEARTBEAT_ENV, "soon")
    await process._open_ws("wss://pty", {})
    assert [c.get("heartbeat") for c in calls] == [120.0, 30.0, None, 120.0]
    assert calls[0] == {"url": "wss://pty", "headers": {"a": "b"}, "heartbeat": 120.0}


def test_another_sdk_layout_is_left_alone():
    from benchflow.sandbox import _sdk_ops

    sdk = _fake_sdk_process([])

    async def _open_ws(self, url, headers, subprotocols=None):
        return None

    sdk.AsyncProcess._open_ws = _open_ws
    _sdk_ops._patch_websocket_heartbeat(sdk)
    assert sdk.AsyncProcess._open_ws is _open_ws
    _sdk_ops._patch_websocket_heartbeat(SimpleNamespace())  # no AsyncProcess


def test_the_installed_sdk_still_has_the_layout_the_patch_wraps():
    """If the SDK changes ``_open_ws``, the heartbeat silently stops: fail."""
    import inspect

    try:
        from daytona._async import process as sdk_process
    except Exception as exc:  # not installed, or not importable with this lock
        pytest.skip(f"daytona._async.process not importable: {exc!r}")
    original = getattr(
        sdk_process.AsyncProcess._open_ws,
        "__wrapped__",
        sdk_process.AsyncProcess._open_ws,
    )
    assert list(inspect.signature(original).parameters) == ["self", "url", "headers"]
    assert callable(sdk_process.http_session_of)
