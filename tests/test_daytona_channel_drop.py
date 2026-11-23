"""a bfx read that lost its exec to the sandbox's transport is asked again.

On Daytona the proxy path can cut the command channel's websocket while an exec's output streams (close 1006, at
times 1008); the PTY session and its helper live on. The exec in flight then fails with
``TransientSandboxTransportError``. bfx's reads (the mount table, ``scan``, ``walk``, the look at a removed path)
change nothing and only a whole answer is used, so such a read is asked again, up to
``_verifier_links.READ_ATTEMPTS`` times in all, inside the walk pause where there is one. A failure that is not the
transport's (a cap, a refusal, a timeout) is not asked again. The channel's own message keeps why the websocket
closed (its close code and socket error), so the next drop can be told from a trial's log.
"""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox import _verifier_links

# The walks' own bfx answers are given here: no stand-in (tests/_fake_bfx.py).
pytestmark = [pytest.mark.bfx_refusal, pytest.mark.timeout(30)]

R = "/logs/verifier/reward.txt"
DROP = "the sandbox's command channel stopped: websocket closed (close_code=1006)"


def _env(events, *, drops=None, pause=True):
    """A sandbox whose bfx reads and root execs are logged in order. ``drops`` maps a bfx read ("mounts",
    "scan", "walk", "look") to how many of its first calls lose their exec to the transport. The agent's link
    /app/x leads to the reward."""
    from benchflow.sandbox._identities import Identities

    drops = dict(drops or {})

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        kind = (
            "mounts"
            if args.get("mounts")
            else "look"
            if "paths" in args or "paths_b64" in args
            else cmd
        )
        events.append(f"bfx {kind}")
        if drops.get(kind, 0) > 0:
            drops[kind] -= 1
            events.append(f"drop {kind}")
            raise TransientSandboxTransportError(DROP)
        if cmd == "walk":
            link = {"path": "app/x", "type": "link", "link": R, "dev": 7, "ino": 42}
            return {
                "entries": [link] if args.get("root") == "/" else [],
                "vanished": [],
                "deeper": [],
            }
        if cmd == "remove":
            return {"removed": True}
        if kind == "mounts":
            return {"entries": [], "mounts": [{"id": 1, "mount_point": "/"}]}
        return {"entries": [], "errors": [], "mounts": []}

    async def exec_(command, *args, **kwargs):
        text = str(command)
        if "BF pause" in text and "kill -STOP" in text:
            events.append("pause")
            return SimpleNamespace(
                return_code=0,
                stdout="BF pause pid 77\nBF pause at 5.0\nBF pause still 1\n",
                stderr="",
            )
        if "BF resume" in text:
            events.append("resume")
            return SimpleNamespace(
                return_code=0,
                stdout="BF resume at 6.0\nBF resume pid 77\nBF resume done\n",
                stderr="",
            )
        if "readlink" in text or "realpath" in text:
            return SimpleNamespace(return_code=0, stdout=f"/app/x\0{R}\0", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    env = SimpleNamespace(
        bfx_call=bfx_call, bfx_delivery="per-call", exec=AsyncMock(side_effect=exec_)
    )
    env._benchflow_identities = Identities(
        "root:x:0:0::/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n",
        "root:x:0:\nagent:x:1000:\n",
        "0:0",
    )
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    return env


# -- after the turn: one cut, inside the pause ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["mounts", "walk"])
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_a_walk_after_the_turn_that_loses_one_read_to_the_transport_still_runs(
    check, read
):
    """(dogfood C3 part 2: 3 of 3 Daytona model trials scored 0 this way). One of the walk's bfx reads
    loses its exec to a dropped command channel: it is asked again, inside the same pause, and the check runs
    whole; before test.sh the turn's link is still found and removed.

    Mutant: a transport failure ends the walk (no read asked again)."""
    events: list[str] = []
    env = _env(events, drops={read: 1})
    found = await getattr(_verifier_links, check)(
        env, timeout_sec=5, pause_user="agent"
    )
    assert found["ran"] is True and found["why"] is None, found
    assert f"drop {read}" in events, events
    assert events.count("pause") == 1 and events.count("resume") == 1, events
    dropped, resumed = events.index(f"drop {read}"), events.index("resume")
    again = events.index(f"bfx {read}", dropped)
    assert events[0] == "pause" and dropped < again < resumed, events
    assert [r["path"] for r in found["links"]] == ["/app/x"], found
    if check == "remove_turn_links":
        assert found["links"][0]["removed"] is True, found


@pytest.mark.asyncio
async def test_a_look_after_a_removal_that_loses_its_exec_is_made_again():
    """The confirming look at a removed link (bfx ``scan`` of the path, after the resume) is a read too:
    one lost exec does not leave the removal unconfirmed.

    Mutant: the look's transport failure taken as an unconfirmed removal."""
    events: list[str] = []
    env = _env(events, drops={"look": 1})
    found = await _verifier_links.remove_turn_links(
        env, timeout_sec=5, pause_user="agent"
    )
    assert "drop look" in events, events
    row = found["links"][0]
    assert row["removed"] is True and "error" not in row, found


@pytest.mark.asyncio
async def test_a_record_before_the_turn_survives_two_cuts_in_a_row():
    """Before the turn the whole walk was already tried twice; a read that loses its exec twice in a row
    (the restarted channel dropped again) no longer leaves the record incomplete, so the trial is not refused.

    Mutant: no read asked again (two walks, each failing on its first scan)."""
    events: list[str] = []
    env = _env(events, drops={"scan": 2})
    delattr(env, _verifier_links.INVENTORY)
    walked = await _verifier_links.record_before_turn(env, timeout_sec=5)
    assert walked.complete is True and walked.why is None, walked
    assert events.count("drop scan") == 2, events


# -- the bounds -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_a_read_that_keeps_losing_its_exec_is_asked_a_bounded_number_of_times(
    check,
):
    """A transport that never comes back: the read is asked READ_ATTEMPTS times in all, then the walk is
    incomplete with the transport's reason, and what the pause stopped is continued.

    Mutant: unbounded asking; the failure swallowed as a whole walk."""
    events: list[str] = []
    env = _env(events, drops={"walk": 99})
    found = await getattr(_verifier_links, check)(
        env, timeout_sec=5, pause_user="agent"
    )
    assert found["ran"] is False and "websocket closed" in str(found["why"]), found
    assert events.count("bfx walk") == 3, events
    assert _verifier_links.READ_ATTEMPTS == 3
    assert events[-1] == "resume", events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        "too-many",
        "timeout",
        "other",
    ],
)
async def test_a_failure_that_is_not_the_transport_s_is_not_asked_again(error):
    """A cap (bfx too-many), a timeout or any other failure would only fail the same way again, or is not
    the transport's: the walk is incomplete after one call, as before.

    Mutant: every failure asked again."""
    from benchflow._bfx.client import BfxError

    calls: list[str] = []
    raised = {
        "too-many": BfxError("too-many", "more than 4000000 names"),
        "timeout": TimeoutError("bfx walk did not return"),
        "other": RuntimeError("boom"),
    }[error]

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        if args.get("mounts"):
            return {"entries": [], "mounts": [{"id": 1, "mount_point": "/"}]}
        calls.append(cmd)
        raise raised

    env = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="per-call")
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    found = await _verifier_links.links_after_grading(env, timeout_sec=5)
    assert found["ran"] is False, found
    assert calls == ["walk"], calls


# -- the channel says why its websocket closed ----------------------------------------------------------------


class _Pty:
    """A Daytona PTY handle: ``wait()`` ends when its websocket is gone; ``_ws`` keeps the close code as the SDK's
    aiohttp socket does."""

    def __init__(self, on_data, marker):
        self.on_data = on_data
        self.marker = marker
        self.ended = asyncio.Event()
        self.exit_code = None
        self.error = None
        self._ws = SimpleNamespace(close_code=None, exception=lambda: None)
        self.started = 0

    async def wait(self):
        await self.ended.wait()
        return SimpleNamespace(exit_code=None, error=None)

    async def send_input(self, data):
        if isinstance(data, str):  # the helper's launch line: it answers ready
            pid = b"4242"
            await self.on_data(
                self.marker + b"R" + struct.pack(">II", 0, len(pid)) + pid
            )
            return
        if (
            b'"i"' in data
        ):  # a command: it starts, streams, and the proxy cuts the websocket
            self.started += 1
            ident = 1
            await self.on_data(b"S" + struct.pack(">II", ident, 0))
            await self.on_data(b"O" + struct.pack(">II", ident, 5) + b"half-")
            self._ws.close_code = 1008
            self.ended.set()

    async def kill(self):
        return None

    async def disconnect(self):
        return None


@pytest.mark.asyncio
async def test_the_channel_s_lost_command_says_why_the_websocket_closed():
    """The cut came with close code 1008 (or 1006 and a socket error): the exec's transport failure says
    so, still starting "websocket closed", where it said "websocket closed" alone.

    Mutant: the close code dropped from the reason."""
    from benchflow.sandbox._base import KIT_PROBE_ROUTE
    from benchflow.sandbox._exec_channel import ChannelLost
    from benchflow.sandbox.daytona_channel import DaytonaChannel

    made: dict = {}

    async def create_pty_session(*, id, on_data, envs):
        made["pty"] = _Pty(on_data, made["channel"].marker)
        return made["pty"]

    provider = SimpleNamespace(
        process=SimpleNamespace(create_pty_session=create_pty_session)
    )
    counts = SimpleNamespace(channel_pty_create_requests=0, channel_send_requests=0)
    channel = DaytonaChannel(provider, counts)
    made["channel"] = channel
    await channel.connect("launch", KIT_PROBE_ROUTE)
    assert channel.alive and channel.pid == 4242
    with pytest.raises(ChannelLost) as lost:
        await channel.run("bfx walk", "/bin/sh", 5)
    said = str(lost.value)
    assert "websocket closed" in said and "close_code=1008" in said, said
    if channel.heartbeat is not None:
        channel.heartbeat.cancel()
        await asyncio.gather(channel.heartbeat, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_removal_that_loses_its_exec_is_never_asked_again():
    """S-3 (B2R's rule). A ``remove`` changes the sandbox, so unlike a read it is never asked again when its
    exec is lost to the transport: one call, the link left not removed with the error, and the check counts it
    (the Q6 0 for a non-root agent).

    Mutant: ``remove`` asked again as a read is."""
    events: list[str] = []
    env = _env(events, drops={"remove": 1})
    found = await _verifier_links.remove_turn_links(
        env, timeout_sec=5, pause_user="agent"
    )
    assert events.count("bfx remove") == 1 and "drop remove" in events, events
    (row,) = found["links"]
    assert row["path"] == "/app/x" and row["removed"] is False, found
    assert row.get("error"), found
