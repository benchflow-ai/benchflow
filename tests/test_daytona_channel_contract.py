"""Transport outcomes at the public exec and pool boundaries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._runtime_tree import KIT, KIT_BIN
from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox import daytona_channel as module
from benchflow.sandbox._base import KIT_PROBE_ROUTE, ExecResult
from benchflow.sandbox._exec_channel import ChannelUnavailable, ChannelUnconfirmed
from benchflow.sandbox.daytona import _ExecCounts
from benchflow.sandbox.protocol import ExecTimeoutError
from tests.daytona_channel_fakes import _lifecycle_sandbox
from tests.test_daytona_channel import channel, local_pty, output  # noqa: F401
from tests.test_daytona_channel_lifecycle import _channel

pytestmark = pytest.mark.timeout(30)


@pytest.mark.parametrize("ident,pid", [(1, b"123"), (0, b"1"), (0, b"0")])
async def test_ready_frame_is_validated_before_any_command(ident, pid):
    """Invalid startup identities fail immediately, without a deadline."""
    made = module.DaytonaChannel(SimpleNamespace(), _ExecCounts())
    await made.on_data(made.marker + output(b"R", ident, pid))
    assert made.dead and not made.alive
    assert made.ready.done() and made.ready.exception() is not None


async def test_cancelling_a_partial_multi_message_frame_retires_framing():
    """A cancelled chunk cannot become the next frame's prefix."""
    made = channel()
    blocked = asyncio.Event()

    async def send(data):
        blocked.set()
        await asyncio.Event().wait()

    made.pty.send_input.side_effect = send
    work = asyncio.create_task(made.run("x" * module.CHUNK_SIZE * 2, "/bin/sh", 10))
    await blocked.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert made.dead
    with pytest.raises(ChannelUnavailable):
        await made.run("next", "/bin/sh", 10)
    assert made.pty.send_input.await_count == 1


@pytest.mark.parametrize("timeout", [1, None, 0])
async def test_confirmed_timeout_keeps_the_session_error_class_and_text(
    monkeypatch, timeout
):
    """A running command's timeout is not a transport failure."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    made.run = AsyncMock(side_effect=TimeoutError())
    expected = box._poll_timeout_error(timeout, timeout is None or timeout <= 0)
    with pytest.raises(type(expected)) as error:
        await pool.run("sleep 100", KIT_PROBE_ROUTE, timeout)
    assert str(error.value) == str(expected)
    assert box._exec_counts().channel_fallbacks == 0


@pytest.mark.parametrize("uncertain", [False, True])
async def test_no_fallback_after_generation_changes_during_delivery(
    monkeypatch, uncertain
):
    """Even no-start proof cannot move work to another generation."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)

    async def failed(*args, **kwargs):
        box._exec_generation += 1
        if uncertain:
            raise ChannelUnconfirmed("delivery lost", 1)
        raise ChannelUnavailable("never sent")

    made.run = failed
    made.started_ids = AsyncMock(return_value=frozenset())
    with pytest.raises(TransientSandboxTransportError):
        await pool.run("side-effect", KIT_PROBE_ROUTE, 5)
    assert box._exec_counts().channel_fallbacks == 0
    assert box._sandbox.process.created == []


def _exec_box(monkeypatch):
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    box._sandbox.process.create_pty_session = AsyncMock()
    box.exec_route = AsyncMock(return_value=KIT_PROBE_ROUTE)
    box._default_root_path = lambda command, *args: command
    box._merge_env = lambda env, *args, **kwargs: env
    box._resolve_user = lambda user: user
    return box, pool, made


@pytest.mark.parametrize("stdin", [False, True])
async def test_public_exec_reaches_the_channel_with_the_wrapped_command(
    monkeypatch, stdin
):
    """Both plumbing paths execute through the channel."""
    box, _pool, made = _exec_box(monkeypatch)
    captured = []

    async def send(data):
        head = json.loads(data[4:])
        captured.append(head)
        await made.on_data(output(b"S", head["i"]))
        await made.on_data(output(b"O", head["i"], b"channel"))
        await made.on_data(output(b"X", head["i"], b"7"))

    made.pty.send_input.side_effect = send
    if stdin:
        box._stage_payloads = AsyncMock(return_value={0: "/root/stage/payload"})
        box._unstage_payload = AsyncMock()
        result = await box._sandbox_exec(
            "cat",
            user="root",
            timeout_sec=5,
            route=KIT_PROBE_ROUTE,
            stdin_payload=b"private input",
        )
        assert "/root/stage/payload" in captured[0]["a"][-1]
    else:
        result = await box.exec("echo channel", user="root", timeout_sec=5)
        assert "echo channel" in captured[0]["a"][-1]
    assert captured[0]["a"][:2] == [KIT_PROBE_ROUTE.shell, "-c"]
    assert result.return_code == 7 and result.stdout == "channel\n"
    assert box._sandbox.process.created == []
    counts = box._exec_counts()
    assert counts.exec_calls == counts.channel_exec_calls == 1
    assert counts.exec_seconds >= counts.channel_exec_seconds > 0
    assert counts.status_requests == counts.channel_fallbacks == 0


@pytest.mark.parametrize("disabled", ["setting", "environment"])
async def test_public_exec_honors_both_channel_switches(monkeypatch, disabled):
    """Disabling the channel really selects a session exec."""
    box, _pool, made = _exec_box(monkeypatch)
    box._sandbox.process.get_session_command_logs = AsyncMock(
        return_value=SimpleNamespace(stdout="out", stderr="err")
    )
    if disabled == "setting":
        box.exec_channel = False
    else:
        monkeypatch.setenv("BENCHFLOW_EXEC_CHANNEL", "0")
    result = await box.exec("echo session", user="root", timeout_sec=5)
    assert result.stdout == "out"
    assert box._sandbox.process.executed
    made.pty.send_input.assert_not_awaited()
    assert box._exec_counts().channel_exec_calls == 0
    box._sandbox.release.set()
    await box.stop(True)


async def test_transient_exec_passes_its_cleanup_contract(monkeypatch):
    """Exec_transient forwards session-deletion semantics to the frame."""
    box, _pool, made = _exec_box(monkeypatch)
    made.run = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(ExecTimeoutError):
        await box.exec_transient("sleep 100", user="root", timeout_sec=1)
    assert made.run.await_args.kwargs == {"cleanup_session": True}
    assert box._exec_counts().transient_exec_calls == 1
    assert box._exec_counts().exec_calls == 0


async def test_transient_timeout_ends_only_its_own_process_group(local_pty, tmp_path):  # noqa: F811
    """Session cleanup kills descendants without disturbing a sibling."""
    made, _ = local_pty
    marker = tmp_path / "leaked"
    release = tmp_path / "release"
    script = (
        f"(while test ! -f {release}; do sleep .01; done; echo leak > {marker}) & wait"
    )
    with pytest.raises(TimeoutError):
        await made.run(script, "/bin/sh", 0.05, cleanup_session=True)
    release.touch()
    result = await made.run("sleep .1; echo sibling", "/bin/sh", 5)
    assert result.stream(1) == b"sibling\n"
    assert not marker.exists()


async def test_helper_recognizes_heartbeat_frames(local_pty):  # noqa: F811
    """A heartbeat is acknowledged without crashing the helper."""
    made, process = local_pty
    await made._send(module.frame({"p": 1, "n": -1}))
    result = await made.run("echo after-heartbeat", "/bin/sh", 5)
    assert result.stream(1) == b"after-heartbeat\n"
    assert process.returncode is None


async def _connect(monkeypatch, *, failed=False):
    box, _ = _lifecycle_sandbox(monkeypatch)
    box._session_command = Mock(wraps=box._session_command)
    provider = box._sandbox
    provider.fs = SimpleNamespace(upload_file=AsyncMock())
    provider.process.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(
                exit_code=0,
                result=f'#!{KIT}/bin/sh\nexec {KIT}/libexec/python3 -I -S -B "$@"\n',
            ),
            SimpleNamespace(exit_code=0, result=""),
        ]
    )
    pty = SimpleNamespace(kill=AsyncMock(), disconnect=AsyncMock())
    watcher = asyncio.create_task(asyncio.Event().wait())
    monkeypatch.setattr(module, "_watch_pty", lambda _: watcher)

    async def create(**kwargs):
        made = kwargs["on_data"].__self__

        async def send(data):
            if isinstance(data, str):
                if failed:
                    raise ConnectionError("launch failed")
                await made.on_data(made.marker + output(b"R", 0, b"123"))
            else:
                head = json.loads(data[4:])
                if "i" in head:
                    await made.on_data(output(b"S", head["i"]))
                    await made.on_data(output(b"X", head["i"], b"0"))

        pty.send_input = AsyncMock(side_effect=send)
        return pty

    provider.process.create_pty_session = AsyncMock(side_effect=create)
    pool = module.DaytonaChannels(box, KIT_PROBE_ROUTE)
    box._command_channels = pool
    await pool.starting
    return box, pool, pool.channels[0], watcher


async def test_start_commands_fallback_and_close_count_every_request(monkeypatch):
    """Each channel counter pins traffic and command accounting."""
    box, pool, made, watcher = await _connect(monkeypatch)
    counts = box._exec_counts()
    assert {k: v for k, v in asdict(counts).items() if k.startswith("channel_")} == {
        "channel_starts": 1,
        "channel_setup_requests": 2,
        "channel_send_requests": 1,
        "channel_exec_calls": 0,
        "channel_exec_seconds": 0,
        "channel_fallbacks": 0,
        "channel_exec_pending_at_release": 0,
        "channel_upload_requests": 1,
        "channel_pty_create_requests": 1,
        "channel_kill_requests": 0,
        "channel_disconnect_requests": 0,
        "channel_cleanup_timeouts": 0,
    }
    assert (await pool.run("true", KIT_PROBE_ROUTE, 5)).return_code == 0
    assert counts.channel_send_requests == 2
    assert counts.channel_exec_calls == 1 and counts.channel_exec_seconds > 0
    before = counts.channel_exec_seconds
    made.run = AsyncMock(side_effect=ChannelUnavailable("not sent"))
    assert await pool.run("fallback", KIT_PROBE_ROUTE, 5) is None
    assert counts.channel_exec_calls == 2 and counts.channel_fallbacks == 1
    assert counts.channel_exec_seconds > before
    # The verdict and folder removal use captured-provider controls.
    box._sandbox.process.exec.side_effect = None
    box._sandbox.process.exec.return_value = SimpleNamespace(exit_code=0, result="J")
    await pool.close()
    assert counts.channel_setup_requests == 4
    assert counts.channel_kill_requests == counts.channel_disconnect_requests == 1
    assert (
        counts.channel_exec_pending_at_release == counts.channel_cleanup_timeouts == 0
    )
    assert counts.channel_send_requests == 2 and watcher.done()


async def test_no_handle_close_counts_the_accepted_pty_cleanup_request():
    """A lost create response still accounts for kill_pty_session."""
    made = channel()
    made.pty = None
    made.provider = SimpleNamespace(
        process=SimpleNamespace(kill_pty_session=AsyncMock())
    )
    await made.close()
    assert made.counts.channel_kill_requests == 1
    assert made.counts.channel_disconnect_requests == 0
    made.provider.process.kill_pty_session.assert_awaited_once_with(made.session_id)


async def test_capture_proves_staging_before_writing_environment(monkeypatch):
    """Capture cannot write through an unproved staging directory."""
    box, pool, _made, _watcher = await _connect(monkeypatch)
    capture = box._session_command.call_args_list[1].args[0]
    proof = module._transport.stage_proof(
        [pool.folders[0] + "/payload"],
        shell=KIT_PROBE_ROUTE.shell,
        path=KIT_PROBE_ROUTE.pin,
    )
    # Inspect the inner capture, before session shell quoting.
    assert capture.startswith(proof + " || exit 126; set -C; ")
    box._sandbox.release.set()
    await box.stop(True)


async def test_failed_start_closes_channel_and_later_removes_staged_folder(monkeypatch):
    """A failed launch retains no live PTY or staged payload."""
    _box, pool, made, watcher = await _connect(monkeypatch, failed=True)
    made.pty.kill.assert_awaited_once()
    made.pty.disconnect.assert_awaited_once()
    assert made.dead and watcher.done() and pool.starting.result() is None
    folder = pool.folders[0]
    pool._control = AsyncMock(return_value=ExecResult(return_code=0))
    await pool.close()
    assert not pool.folders
    assert any(
        f"{KIT_BIN}/rm -rf -- {folder}" == call.args[0]
        for call in pool._control.call_args_list
    )


async def test_connected_channel_sends_heartbeats_and_fails_on_send_loss(monkeypatch):
    """The live channel owns a heartbeat and retires on failure."""
    monkeypatch.setattr(module, "HEARTBEAT_INTERVAL", 0.01)
    box, _pool, made, _watcher = await _connect(monkeypatch)
    failed = asyncio.Event()

    async def send(data):
        assert json.loads(data[4:])["p"] == 1
        failed.set()
        raise ConnectionError("lost heartbeat")

    made.pty.send_input.side_effect = send
    await failed.wait()
    await made.heartbeat
    assert made.dead
    box._sandbox.release.set()
    await box.stop(True)


async def test_websocket_loss_retires_every_pending_frame(monkeypatch):
    """SDK watcher completion immediately closes channel eligibility."""
    box, _pool, made, watcher = await _connect(monkeypatch)
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    assert made.dead and not made.alive
    box._sandbox.release.set()
    await box.stop(True)


async def test_provider_release_joins_cancelled_owners_without_a_timeout(monkeypatch):
    """Stop joins its owned tasks before returning, without Runner help."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)
    made.heartbeat = asyncio.create_task(asyncio.Event().wait())
    made.gone = asyncio.create_task(asyncio.Event().wait())
    made.death_check = asyncio.create_task(asyncio.Event().wait())
    made.withdrawals.add(asyncio.create_task(asyncio.Event().wait()))
    box._sandbox.release.set()
    await box.stop(True)
    assert all(task.done() for task in made.tasks())
    assert box._exec_counts().channel_cleanup_timeouts == 0
    assert box._exec_counts().channel_disconnect_requests == 1
    assert not box._released_command_channels


async def test_disconnect_deadline_is_recorded_and_joined_at_stop(monkeypatch):
    """An unresponsive disconnect is cancelled and awaited at release."""
    monkeypatch.setattr(module, "CLOSE_TIMEOUT", 0.01)
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)
    finished = asyncio.Event()

    async def disconnect():
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    made.pty.disconnect = disconnect
    box._sandbox.release.set()
    await box.stop(True)
    assert finished.is_set() and made.disconnecting.done()
    assert box._exec_counts().channel_cleanup_timeouts == 1
    assert box._exec_counts().channel_disconnect_requests == 1


async def test_finalization_cancels_owned_tasks_synchronously(monkeypatch):
    """Retirement cancels each owner before the asynchronous join."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    made.heartbeat = asyncio.create_task(asyncio.Event().wait())
    made.gone = asyncio.create_task(asyncio.Event().wait())
    made.death_check = asyncio.create_task(asyncio.Event().wait())
    withdrawal = asyncio.create_task(asyncio.Event().wait())
    made.withdrawals.add(withdrawal)
    owners = [made.heartbeat, made.gone, made.death_check, withdrawal]
    try:
        pool.finalize()
        assert all(task.cancelling() for task in owners)
    finally:
        for task in owners:
            task.cancel()
        await pool.join()


async def test_stop_joins_a_stalled_frame_without_cancelling_its_caller(monkeypatch):
    """A released provider cannot retain an in-progress WebSocket send."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    entered, finished = asyncio.Event(), asyncio.Event()

    async def send(data):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    made.pty.send_input.side_effect = send
    work = asyncio.create_task(pool.run("side-effect", KIT_PROBE_ROUTE, 100))
    await entered.wait()
    box._sandbox.release.set()
    await box.stop(True)
    assert finished.is_set() and not made.sends
    with pytest.raises(TransientSandboxTransportError):
        await work
    assert not work.cancelled()
    assert box._exec_counts().channel_exec_pending_at_release == 1
    assert box._exec_counts().channel_fallbacks == 0


async def test_surviving_provider_cleanup_cannot_drop_a_live_writer(monkeypatch):
    """Retiring a live provider keeps ownership until sends are drained."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    entered = asyncio.Event()

    async def send(data):
        entered.set()
        await asyncio.Event().wait()

    made.pty.send_input.side_effect = send
    work = asyncio.create_task(pool.run("side-effect", KIT_PROBE_ROUTE, 100))
    await entered.wait()
    box._retire_command_channels()
    box._queue_command_channel_cleanup()
    await pool.cleanup_task
    assert not made.sends and not box._retired_command_channels
    with pytest.raises(TransientSandboxTransportError):
        await work
    box._sandbox.release.set()
    await box.stop(True)


async def test_start_acknowledgement_remains_proof_when_send_reports_failure(
    monkeypatch,
):
    """A start received during send is authoritative even if the send fails."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)

    async def send(data):
        await made.on_data(output(b"S", 1))
        raise ConnectionError("send failed after acknowledgement")

    made.pty.send_input.side_effect = send
    made.started_ids = AsyncMock(return_value=frozenset())
    with pytest.raises(TransientSandboxTransportError):
        await pool.run("side-effect", KIT_PROBE_ROUTE, 5)
    made.started_ids.assert_not_awaited()
    assert box._exec_counts().channel_fallbacks == 0
    assert box._sandbox.process.created == []
