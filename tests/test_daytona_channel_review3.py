"""Command outcomes and cancellation at cleanup boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchflow.sandbox import daytona as module
from benchflow.sandbox import daytona_channel as channel_module
from benchflow.sandbox._base import KIT_PROBE_ROUTE
from benchflow.sandbox._exec_channel import ChannelLost
from benchflow.sandbox.protocol import ExecTimeoutError
from tests.daytona_channel_fakes import (
    _checkpoint,
    _lifecycle_sandbox,
    _Remote,
    _restore_setup,
)
from tests.test_daytona_channel import channel, local_pty, output  # noqa: F401
from tests.test_daytona_channel_lifecycle import _channel

pytestmark = pytest.mark.timeout(30)


def _timed_out_transient(made, *, stall_termination):
    """Confirm a timeout, then hold the termination send or acknowledgement."""
    terminating = asyncio.Event()

    async def send(data):
        head = json.loads(data[4:])
        if "i" in head:
            await made.on_data(output(b"S", head["i"]))
        elif "q" in head:
            await made.on_data(output(b"T", head["q"]))
        elif "k" in head:
            terminating.set()
            if stall_termination:
                await asyncio.Event().wait()

    made.pty.send_input.side_effect = send
    return terminating


async def test_review3_release_during_termination_frame_does_not_cancel_the_caller(
    monkeypatch,
):
    """Release of the termination writer preserves its caller's timeout."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    terminating = _timed_out_transient(made, stall_termination=True)
    work = asyncio.create_task(
        pool.run("sleep 100", KIT_PROBE_ROUTE, 1, cleanup_session=True)
    )
    await terminating.wait()
    box._sandbox.release.set()
    await box.stop(True)
    (outcome,) = await asyncio.gather(work, return_exceptions=True)
    assert isinstance(outcome, ExecTimeoutError), repr(outcome)
    assert str(outcome) == str(box._poll_timeout_error(1, False))


async def test_review3_release_does_not_hold_a_caller_awaiting_termination(
    monkeypatch,
):
    """Release wakes the termination waiter without changing its timeout."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    terminating = _timed_out_transient(made, stall_termination=False)
    work = asyncio.create_task(
        pool.run("sleep 100", KIT_PROBE_ROUTE, 1, cleanup_session=True)
    )
    await terminating.wait()
    box._sandbox.release.set()
    await box.stop(True)
    loop = asyncio.get_running_loop()
    released = loop.time()
    (outcome,) = await asyncio.gather(work, return_exceptions=True)
    waited = loop.time() - released
    assert isinstance(outcome, ExecTimeoutError), repr(outcome)
    assert str(outcome) == str(box._poll_timeout_error(1, False))
    assert waited < 2, f"caller held {waited:.1f} s after release"


async def test_review3_cancelling_stop_during_its_join_still_cancels_the_caller(
    monkeypatch,
):
    """A caller cancelled during stop's join still receives its cancel."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)
    disconnecting = asyncio.Event()

    async def slow_disconnect():
        disconnecting.set()
        await asyncio.sleep(0.5)

    made.pty.disconnect = slow_disconnect
    box._sandbox.release.set()
    work = asyncio.create_task(box.stop(True))
    await disconnecting.wait()
    await asyncio.sleep(0.05)  # stop() is now joining the released disconnect
    assert not work.done()
    work.cancel()
    (outcome,) = await asyncio.gather(work, return_exceptions=True)
    assert made.disconnecting is not None and made.disconnecting.done()
    assert isinstance(outcome, asyncio.CancelledError), repr(outcome)


async def test_review3_slow_send_within_the_session_request_bound():
    """The send has the session request's timeout + 5 allowance."""
    made = channel()

    async def send(data):
        await asyncio.sleep(0.5)
        head = json.loads(data[4:])
        await made.on_data(output(b"S", head["i"]))
        await made.on_data(output(b"X", head["i"], b"0"))

    made.pty.send_input.side_effect = send
    result = await made.run("true", "/bin/sh", 0.2)
    assert result.return_code == 0


async def test_review3_stop_writes_counts_after_the_join(monkeypatch, tmp_path):
    """Stop's record includes the disconnect and its timeout after join."""
    monkeypatch.setattr(channel_module, "CLOSE_TIMEOUT", 0.01)
    box, _ = _lifecycle_sandbox(monkeypatch)
    box.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    _pool, made = _channel(box)

    async def stuck_disconnect():
        await asyncio.Event().wait()

    made.pty.disconnect = stuck_disconnect
    box._sandbox.release.set()
    await box.stop(True)
    record = json.loads((tmp_path / module.EXEC_COUNTS_FILE).read_text())
    assert record["channel_disconnect_requests"] == 1
    assert record["channel_cleanup_timeouts"] == 1


async def test_review3_start_returns_with_released_work_joined(monkeypatch):
    """Start joins every channel task of a provider it released."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    old = box._sandbox
    _pool, made = _channel(box)

    async def slow_disconnect():
        await asyncio.sleep(0.2)

    made.pty.disconnect = slow_disconnect
    new = _Remote("replacement")

    async def start(force_build):
        box._release_command_channels(old)  # as the create retry does
        box._sandbox = new
        box._on_sandbox_created()

    box._strategy.start = start
    await box.start(False)
    assert made.disconnecting is not None and made.disconnecting.done()


async def test_review3_restore_returns_with_released_work_joined(monkeypatch):
    """Restore joins every channel task of the old provider."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    old, new = box._sandbox, _Remote("replacement")
    old.release.set()
    _restore_setup(monkeypatch, box, new)
    _pool, made = _channel(box)

    async def slow_disconnect():
        await asyncio.sleep(0.2)

    made.pty.disconnect = slow_disconnect
    await box.restore(_checkpoint())
    assert box._sandbox is new
    assert made.disconnecting is not None and made.disconnecting.done()


async def test_review3_join_bound_cancels_a_stuck_owner_and_counts_it(monkeypatch):
    """A bounded join cancels its stuck owner and counts the timeout."""
    monkeypatch.setattr(channel_module, "START_TIMEOUT", 0.05)
    monkeypatch.setattr(channel_module, "CLOSE_TIMEOUT", 0.05)
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)
    entered = asyncio.Event()

    async def stubborn_watcher():
        entered.set()
        # An SDK wait that absorbs the first cancellation.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        await asyncio.Event().wait()

    made.gone = asyncio.create_task(stubborn_watcher())
    await entered.wait()
    box._sandbox.release.set()
    await box.stop(True)
    assert made.gone.done()
    assert box._exec_counts().channel_cleanup_timeouts == 1


async def test_review3_termination_waits_for_the_helpers_acknowledgement(
    monkeypatch,
):
    """Transient timeout reporting waits for the helper's kill reply."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    acknowledged = asyncio.Event()
    later = []

    async def send(data):
        head = json.loads(data[4:])
        if "i" in head:
            await made.on_data(output(b"S", head["i"]))
        elif "q" in head:
            await made.on_data(output(b"T", head["q"]))
        elif "k" in head:

            async def acknowledge():
                await asyncio.sleep(0.3)
                acknowledged.set()
                await made.on_data(output(b"K", head["k"]))

            later.append(asyncio.get_running_loop().create_task(acknowledge()))

    made.pty.send_input.side_effect = send
    with pytest.raises(ExecTimeoutError):
        await pool.run("sleep 100", KIT_PROBE_ROUTE, 1, cleanup_session=True)
    done = acknowledged.is_set()
    await asyncio.gather(*later)
    assert done, "the timeout was reported before the helper acknowledged"


async def test_review3_missing_start_acknowledgement_is_transport_loss():
    """A missing start acknowledgement is bounded transport loss."""
    made = channel()
    with pytest.raises(ChannelLost):
        await made.run("side-effect", "/bin/sh", 0.1)
    assert not made.alive


@pytest.mark.parametrize("stall", [True, False], ids=["send", "acknowledgement"])
async def test_caller_cancel_during_termination_propagates_even_at_release(
    monkeypatch, stall
):
    """Release never hides a termination waiter's own cancellation."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    pool, made = _channel(box)
    terminating = _timed_out_transient(made, stall_termination=stall)
    work = asyncio.create_task(
        pool.run("sleep 100", KIT_PROBE_ROUTE, 1, cleanup_session=True)
    )
    await terminating.wait()
    work.cancel("caller cancelled termination")
    box._sandbox.release.set()
    await box.stop(True)
    with pytest.raises(asyncio.CancelledError) as error:
        await work
    assert error.value.args == ("caller cancelled termination",)


async def test_cancelled_stop_joins_every_pool_and_preserves_first_cancel(
    monkeypatch, tmp_path
):
    """Repeated cancellation drains all pools and preserves the first."""
    box, _ = _lifecycle_sandbox(monkeypatch)
    box.rollout_paths = SimpleNamespace(rollout_dir=tmp_path)
    queued = Mock(wraps=box._queue_command_channel_cleanup)
    monkeypatch.setattr(box, "_queue_command_channel_cleanup", queued)
    joining = [asyncio.Event(), asyncio.Event()]
    draining = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    channels = []
    caught = []
    for index in range(2):
        pool, made = _channel(box)
        channels.append(made)

        async def disconnect(i=index):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draining[i].set()
                await release[i].wait()
                raise

        async def join(i=index, original=pool.join):
            joining[i].set()
            try:
                await original()
            except asyncio.CancelledError as exc:
                caught.append(exc)
                raise

        made.pty.disconnect = disconnect
        pool.join = join
        box._retire_command_channels()
    box._sandbox.release.set()
    work = asyncio.create_task(box.stop(True))
    try:
        await joining[0].wait()
        work.cancel("original cancellation")
        await draining[0].wait()
        work.cancel("later cancellation")
        release[0].set()
        await joining[1].wait()
        work.cancel("another pool cancellation")
        await draining[1].wait()
        release[1].set()
        with pytest.raises(asyncio.CancelledError) as error:
            await work
        assert error.value is caught[0]
        assert error.value.args == ("original cancellation",)
        assert len(caught) == 2
        assert all(made.disconnecting.done() for made in channels)
        assert box._released_command_channels == []
        record = json.loads((tmp_path / module.EXEC_COUNTS_FILE).read_text())
        assert record["channel_disconnect_requests"] == 2
        queued.assert_called_once()
    finally:
        for event in release:
            event.set()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)


async def test_cancellation_during_join_timeout_drain_still_propagates(monkeypatch):
    """A caller cancellation during a timed-out join's drain is kept."""
    monkeypatch.setattr(channel_module, "START_TIMEOUT", 0.01)
    monkeypatch.setattr(channel_module, "CLOSE_TIMEOUT", 0.01)
    box, _ = _lifecycle_sandbox(monkeypatch)
    _pool, made = _channel(box)
    entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def watcher():
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()

    made.gone = asyncio.create_task(watcher())
    await entered.wait()
    box._release_command_channels(box._sandbox)
    work = asyncio.create_task(box._join_released_command_channels())
    try:
        await draining.wait()
        work.cancel("cancel during timeout drain")
        release.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await work
        assert error.value.args == ("cancel during timeout drain",)
        assert made.gone.done()
        assert box._exec_counts().channel_cleanup_timeouts == 1
        assert box._released_command_channels == []
    finally:
        release.set()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)


async def test_helper_acknowledges_reaped_child_without_signalling(local_pty, tmp_path):  # noqa: F811
    """A reaped leader's open pipes do not authorize a group signal."""
    made, _ = local_pty
    leader, release, survived = (tmp_path / name for name in ("leader", "go", "alive"))
    script = f"""
import os, time
if os.fork():
    with open({str(leader)!r}, 'w') as f:
        f.write(str(os.getpid()))
    os._exit(0)
end = time.monotonic() + 8
while not os.path.exists({str(release)!r}) and time.monotonic() < end:
    time.sleep(0.01)
if os.path.exists({str(release)!r}):
    with open({str(survived)!r}, 'w') as f:
        f.write('survived')
os._exit(0)
"""
    work = asyncio.create_task(
        made.run(
            "exec " + shlex.join([sys.executable, "-I", "-S", "-c", script]),
            "/bin/sh",
            10,
        )
    )
    try:
        async with asyncio.timeout(5):
            while not leader.exists() or not leader.read_text():
                await asyncio.sleep(0.01)
            while Path("/proc", leader.read_text()).exists():
                await asyncio.sleep(0.01)
        assert not work.done()
        pending = made.pending[1]
        await made._send(channel_module.frame({"k": 1, "n": -1}))
        await pending.terminated.wait()
        release.touch()
        assert (await work).return_code == 0
        assert survived.read_text() == "survived"
    finally:
        release.touch()
        await asyncio.gather(work, return_exceptions=True)
