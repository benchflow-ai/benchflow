"""Root plumbing over a Daytona PTY.

The PTY carries frames only. The protected kit's Python puts it in raw mode,
then starts each unchanged session command with separate pipes and a new
session. Its environment, working folder and umask come from a normal root
exec, not the terminal. Unlike Docker, Daytona waits for both output pipes
to close, even after the foreground process exits.

A send failure is ambiguous even without a starting frame. Only the final
journal, read after killing the identified helper, permits a fallback. Every
resource belongs to the provider handle that created it, including startup
and teardown calls racing a sandbox replacement.

As on Docker, killing the helper can break a timed-out child's output pipes.
A root task can also kill the helper by process pattern; in-flight commands
then report transport loss, without replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from benchflow._runtime_tree import KIT_BIN, KIT_CHANNEL_PYTHON, KIT_PYTHON
from benchflow.diagnostics import TransientSandboxTransportError
from benchflow.sandbox import _transport
from benchflow.sandbox._base import ExecResult, ExecRoute
from benchflow.sandbox._channel_python_probe import startup_source
from benchflow.sandbox._exec_channel import (
    HELPER_SOURCE,
    VERDICT_SOURCE,
    ChannelDeadline,
    ChannelLost,
    ChannelOutput,
    ChannelUnavailable,
    ChannelUnconfirmed,
    helper_python_argv,
    parse_journal,
)
from benchflow.sandbox._root_launch import terminal_env, typed_root_prelude
from benchflow.sandbox.process.daytona import _watch_pty

CHUNK_SIZE = 16 * 1024
START_TIMEOUT = 30.0
HEARTBEAT_INTERVAL = 30.0
CLOSE_TIMEOUT = 15.0

# Files are in a freshly staged, root-only folder, proved before the PTY starts.
_PTY_PRELUDE = r"""
import os, sys, tty
tty.setraw(0)
# stdin becomes nonblocking; stdout needs its own open file description.
fd = os.open('/dev/tty', os.O_WRONLY)
os.dup2(fd, 1); os.close(fd)
base = sys.argv[2]
with open(base + '.env', 'rb') as f:
    BF_GIVEN_ENV = dict(e.split(b'=', 1) for e in f.read().split(b'\0') if b'=' in e)
with open(base + '.cwd', 'rb') as f:
    BF_WORKDIR = f.read()[:-1]
    os.chdir(BF_WORKDIR)
with open(base + '.mask') as f:
    os.umask(int(f.read().strip(), 8))
BF_STREAM_GRACE = None
BF_KEEP_STDIN = True
os.write(1, BF_READY_MARK)
"""


@dataclass
class _Pending:
    future: asyncio.Future[int]
    started: bool = False
    accepted: asyncio.Event = field(default_factory=asyncio.Event)
    terminated: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: list[tuple[int, bytes]] = field(default_factory=list)


def frame(head: dict[str, Any]) -> bytes:
    data = json.dumps(head).encode()
    return struct.pack(">I", len(data)) + data


def session_output(data: bytes) -> str:
    """Daytona's session-log text: no NULs, and a final newline if needed."""
    text = data.replace(b"\0", b"").decode("utf-8", errors="surrogateescape")
    text = re.sub("[\udc80-\udcff]", "\ufffd", text)
    return text + "\n" if text and not text.endswith("\n") else text


class DaytonaChannel:
    """One helper and its PTY, permanently bound to one provider handle."""

    def __init__(self, provider: Any, counts: Any) -> None:
        self.provider = provider
        self.counts = counts
        self.pty: Any = None
        self.gone: asyncio.Task[Any] | None = None
        self.heartbeat: asyncio.Task[None] | None = None
        self.ready = asyncio.get_running_loop().create_future()
        self.pending: dict[int, _Pending] = {}
        self.write_lock = asyncio.Lock()
        self.next_id = 1
        self.dead = False
        self.finalized = False
        self.disconnecting: asyncio.Task[None] | None = None
        self.withdrawals: set[asyncio.Task[None]] = set()
        self.sends: set[asyncio.Task[Any]] = set()
        self.buffer = b""
        self.marker = ("BF_CHANNEL_" + uuid.uuid4().hex).encode()
        self.synchronized = False
        self.pid: int | None = None
        self.journal: str | None = None
        self.verdict: Any = None
        self.death_check: asyncio.Task[frozenset[int] | None] | None = None
        self.session_id = "bf-channel-" + uuid.uuid4().hex

    @property
    def alive(self) -> bool:
        return not self.dead and self.ready.done() and self.ready.exception() is None

    async def on_data(self, data: bytes) -> None:
        self.buffer += data
        if not self.synchronized:
            _before, found, after = self.buffer.partition(self.marker)
            if not found:
                self.buffer = self.buffer[-len(self.marker) :]
                return
            self.buffer = after
            self.synchronized = True
        try:
            while len(self.buffer) >= 9:
                kind = self.buffer[:1]
                ident, size = struct.unpack(">II", self.buffer[1:9])
                if size > 65536 or kind not in (
                    b"R",
                    b"S",
                    b"O",
                    b"E",
                    b"X",
                    b"F",
                    b"H",
                    b"T",
                    b"K",
                ):
                    raise ValueError("invalid command channel frame")
                if len(self.buffer) < 9 + size:
                    break
                payload = self.buffer[9 : 9 + size]
                self.buffer = self.buffer[9 + size :]
                if kind == b"R":
                    self.pid = int(payload)
                    if ident != 0 or self.pid <= 1 or self.ready.done():
                        raise ValueError("invalid ready frame")
                    self.ready.set_result(None)
                    continue
                if kind == b"H":
                    continue
                pending = self.pending.get(ident)
                if pending is None:
                    continue  # A caller that timed out leaves its child running.
                if kind == b"S":
                    pending.started = True
                    pending.accepted.set()
                elif kind in (b"O", b"E"):
                    pending.chunks.append((1 if kind == b"O" else 2, payload))
                elif kind == b"K":
                    pending.terminated.set()
                elif not pending.future.done():
                    if kind == b"X":
                        pending.future.set_result(int(payload))
                    elif kind == b"T":
                        pending.future.set_exception(TimeoutError())
                    elif kind == b"F":
                        why = payload.decode(errors="replace")
                        pending.future.set_exception(ChannelUnavailable(why))
                    if pending.future.done():
                        pending.accepted.set()
        except Exception as exc:
            self.die(str(exc))

    def die(self, reason: str) -> None:
        self.dead = True
        message = f"the sandbox's command channel stopped: {reason}"
        if not self.ready.done():
            self.ready.set_exception(ChannelLost(message))
            self.ready.exception()
        for ident, pending in self.pending.items():
            pending.accepted.set()
            pending.terminated.set()
            if not pending.future.done():
                pending.future.set_exception(
                    ChannelLost(message)
                    if pending.started
                    else ChannelUnconfirmed(message, ident)
                )

    async def connect(self, command: str, route: ExecRoute) -> None:
        try:
            if self.dead:
                raise ChannelUnavailable("channel retired before startup")
            self.counts.channel_pty_create_requests += 1
            self.pty = await self.provider.process.create_pty_session(
                id=self.session_id,
                on_data=self.on_data,
                envs=terminal_env(route),
            )
            if self.dead:
                raise ChannelUnavailable("channel retired during startup")
            self.gone = _watch_pty(self.pty)
            if self.gone is not None:
                self.gone.add_done_callback(lambda _: self.die("websocket closed"))
            self.counts.channel_send_requests += 1
            await self.pty.send_input(command + "\n")
            await asyncio.wait_for(asyncio.shield(self.ready), START_TIMEOUT)
            self.heartbeat = asyncio.create_task(self._keepalive())
        except BaseException:
            self.die("startup failed")
            raise

    async def _send(self, data: bytes) -> None:
        owner = asyncio.create_task(self._write(data))
        self.sends.add(owner)
        try:
            await owner
        finally:
            self.sends.discard(owner)

    async def _write(self, data: bytes) -> None:
        for start in range(0, len(data), CHUNK_SIZE):
            if self.dead:
                raise ConnectionError("command channel closed during delivery")
            self.counts.channel_send_requests += 1
            await self.pty.send_input(data[start : start + CHUNK_SIZE])

    async def _keepalive(self) -> None:
        try:
            while self.alive:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                async with self.write_lock:
                    if self.alive:
                        await asyncio.wait_for(self._send(frame({"p": 1, "n": -1})), 10)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.die("heartbeat failed")

    async def run(
        self, command: str, shell: str, budget: float, *, cleanup_session: bool = False
    ) -> ChannelOutput:
        ident = self.next_id
        self.next_id += 1
        pending = _Pending(asyncio.get_running_loop().create_future())
        sent = False
        delivered = False
        multi_chunk = False
        cancelled = False
        try:
            try:
                await asyncio.wait_for(self.write_lock.acquire(), budget + 5)
            except TimeoutError:
                # No byte left this host: the command can use its own session.
                raise ChannelUnavailable("command channel writer busy") from None
            try:
                if not self.alive:
                    raise ChannelUnavailable("channel not ready")
                self.pending[ident] = pending
                data = frame(
                    {
                        "i": ident,
                        "a": [shell, "-c", command],
                        "d": None,
                        "n": -1,
                        "r": None,
                    }
                )
                multi_chunk = len(data) > CHUNK_SIZE
                sent = True  # Even a failed send may have delivered every byte.
                try:
                    await asyncio.wait_for(self._send(data), budget + 5)
                    delivered = True
                except Exception as exc:
                    self.die("input delivery failed")
                    if pending.started:
                        raise ChannelLost(
                            "command channel input delivery failed"
                        ) from exc
                    raise ChannelUnconfirmed(
                        "command channel input delivery failed", ident
                    ) from exc
            finally:
                self.write_lock.release()
            # Session polling starts its budget after the provider accepts
            # the exec, not when the host begins sending it.
            try:
                await asyncio.wait_for(pending.accepted.wait(), budget + 5)
            except TimeoutError:
                self.die("command acknowledgement missing")
                raise ChannelLost("command channel acknowledgement missing") from None
            try:
                result = await asyncio.wait_for(asyncio.shield(pending.future), budget)
            except TimeoutError:
                # As _poll_response: ask once more at the deadline. In
                # particular, GNU timeout's exit 124 must remain a result.
                try:
                    async with asyncio.timeout(10):
                        async with self.write_lock:
                            await self._send(frame({"q": ident, "n": -1}))
                except Exception as exc:
                    self.die("status request failed")
                    raise ChannelLost("command channel status request failed") from exc
                try:
                    result = await asyncio.wait_for(asyncio.shield(pending.future), 10)
                except TimeoutError:
                    if pending.future.done():
                        raise
                    self.die("status response missing")
                    raise ChannelLost(
                        "command channel status response missing"
                    ) from None
            return ChannelOutput(result, pending.chunks)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if self.dead and caller is not None and not caller.cancelling():
                raise ChannelLost("command channel provider released") from None
            cancelled = True
            raise
        except TimeoutError:
            if cleanup_session and pending.started and self.alive:
                # Deleting a transient session ends its process group too.
                try:
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(10):
                            async with self.write_lock:
                                await self._send(frame({"k": ident, "n": -1}))
                            await pending.terminated.wait()
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    if not (
                        self.dead and caller is not None and not caller.cancelling()
                    ):
                        raise
            raise
        finally:
            self.pending.pop(ident, None)
            if sent and not delivered and multi_chunk:
                self.die("incomplete input delivery")
            if pending.future.done() and not pending.future.cancelled():
                pending.future.exception()
            if (
                sent
                and self.alive
                and (cancelled or (not pending.started and not pending.future.done()))
            ):
                task = asyncio.create_task(self._withdraw(ident))
                self.withdrawals.add(task)
                task.add_done_callback(self.withdrawals.discard)

    async def _withdraw(self, ident: int) -> None:
        try:
            async with self.write_lock:
                if self.alive:
                    await asyncio.wait_for(self._send(frame({"c": ident, "n": -1})), 10)
        except Exception:
            self.die("cancel delivery failed")

    async def started_ids(self) -> frozenset[int] | None:
        if self.finalized:
            return None
        if self.death_check is None:
            self.death_check = asyncio.create_task(self._verdict())
        try:
            return await asyncio.shield(self.death_check)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if self.finalized and current is not None and not current.cancelling():
                return None
            raise

    async def _verdict(self) -> frozenset[int] | None:
        if self.pid is None or self.journal is None or self.verdict is None:
            return None
        try:
            result = await self.verdict(self.journal, self.pid)
            return parse_journal(result.stdout) if result.return_code == 0 else None
        except Exception:
            return None

    async def close(self) -> None:
        self.die("closed")
        if self.finalized:
            if self.disconnecting is not None:
                await asyncio.shield(self.disconnecting)
            return
        sends = list(self.sends)
        for task in sends:
            task.cancel()
        await asyncio.gather(*sends, return_exceptions=True)
        if self.withdrawals:
            for task in self.withdrawals:
                task.cancel()
            await asyncio.gather(*self.withdrawals, return_exceptions=True)
        if self.heartbeat is not None:
            self.heartbeat.cancel()
            await asyncio.gather(self.heartbeat, return_exceptions=True)
        # A final journal kills only our helper, not its session-leader children.
        await self.started_ids()
        if self.pty is not None:
            with contextlib.suppress(Exception):
                self.counts.channel_kill_requests += 1
                await asyncio.wait_for(self.pty.kill(), 15)
            await self._disconnect()
        else:
            # A create response can be lost after the server accepted it.
            with contextlib.suppress(Exception):
                self.counts.channel_kill_requests += 1
                await asyncio.wait_for(
                    self.provider.process.kill_pty_session(self.session_id), 15
                )
        if self.gone is not None:
            if not self.gone.done():
                self.gone.cancel()
            await asyncio.gather(self.gone, return_exceptions=True)

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        self.die("retired")
        for task in (self.heartbeat, self.gone, self.death_check, *self.withdrawals):
            if task is not None and not task.done():
                task.cancel()
        for task in self.sends:
            task.cancel()
        if self.pty is not None:
            # Registered before public teardown returns, never from cancellation.
            self.disconnecting = asyncio.create_task(self._disconnect())

    async def _disconnect(self) -> None:
        try:
            self.counts.channel_disconnect_requests += 1
            async with asyncio.timeout(CLOSE_TIMEOUT):
                await self.pty.disconnect()
        except TimeoutError:
            self.counts.channel_cleanup_timeouts += 1
        except Exception:
            pass

    def tasks(self) -> list[asyncio.Task[Any]]:
        return [
            task
            for task in (
                self.heartbeat,
                self.gone,
                self.death_check,
                self.disconnecting,
                *self.withdrawals,
                *self.sends,
            )
            if task is not None
        ]


class DaytonaChannels:
    """Retire synchronously; clean up each provider generation independently."""

    finalized = False

    def __init__(self, sandbox: Any, route: ExecRoute) -> None:
        self.sandbox = sandbox
        self.provider = sandbox._sandbox
        self.generation = sandbox._exec_generation
        self.route = route
        self.closed = False
        self.channels: list[DaytonaChannel] = []
        self.folders: list[str] = []
        self.cleanup_task: asyncio.Task[None] | None = None
        self.inflight: dict[object, float] = {}
        self.starting = asyncio.create_task(self._start())

    def current(self) -> bool:
        return (
            not self.closed
            and self.sandbox._sandbox is self.provider
            and self.generation == self.sandbox._exec_generation
        )

    async def _control(self, text: str) -> ExecResult:
        if self.finalized:
            raise ChannelUnavailable("channel provider released")
        command = self.sandbox._session_command(
            text,
            timeout_sec=30,
            shell="/bin/sh -c",
            cwd=None,
            user="root",
            route=self.route,
        )
        _transport.sent("Daytona", command)
        self.sandbox._exec_counts().channel_setup_requests += 1
        result = await asyncio.wait_for(
            self.provider.process.exec(command, timeout=30), 35
        )
        return ExecResult(return_code=result.exit_code, stdout=result.result, stderr="")

    async def _start(self) -> DaytonaChannel | None:
        channel: DaytonaChannel | None = None
        stage = "reading the kit interpreter"
        try:
            preflight = ""
            read = await self._control(f"{KIT_BIN}/cat {KIT_PYTHON}")
            python = (
                helper_python_argv(read.stdout or "") if read.return_code == 0 else None
            )
            if read.return_code != 0 and self.current():
                stage = "reading the channel interpreter"
                read = await self._control(f"{KIT_BIN}/cat {KIT_CHANNEL_PYTHON}")
                python = (
                    helper_python_argv(read.stdout or "", channel_python=True)
                    if read.return_code == 0
                    else None
                )
                preflight = startup_source(
                    (_PTY_PRELUDE, HELPER_SOURCE, VERDICT_SOURCE)
                )
            if python is None:
                self.sandbox.logger.info(
                    "Daytona command channel did not start: %s",
                    "interpreter wrapper not recognized"
                    if read.return_code == 0
                    else "no authenticated channel interpreter available",
                )
                return None
            if not self.current():
                self.sandbox.logger.info(
                    "Daytona command channel did not start: provider generation changed"
                )
                return None
            channel = DaytonaChannel(self.provider, self.sandbox._exec_counts())
            self.channels.append(channel)
            source = (
                preflight
                + f"BF_READY_MARK = {channel.marker!r}\n"
                + _PTY_PRELUDE
                + HELPER_SOURCE
            )
            stage = "staging the helper"
            path = await self._stage_source(source.encode())
            if not self.current():
                self.sandbox.logger.info(
                    "Daytona command channel did not start: provider generation changed"
                )
                return None
            assert self.route.pin is not None
            proof = _transport.stage_proof(
                [path], shell=self.route.shell, path=self.route.pin
            )
            capture = (
                f"{proof} || exit 126; set -C; "
                f"{KIT_BIN}/env -0 > {path}.env && "
                f"pwd -P > {path}.cwd && umask > {path}.mask"
            )
            stage = "proving the execution context"
            captured = await self._control(capture)
            if captured.return_code != 0 or not self.current():
                self.sandbox.logger.info(
                    "Daytona command channel did not start: %s",
                    "execution context proof failed"
                    if captured.return_code != 0
                    else "provider generation changed",
                )
                return None
            channel.journal = path + ".started"

            async def verdict(journal: str, pid: int) -> ExecResult:
                return await self._control(
                    shlex.join([*python, "-c", VERDICT_SOURCE, journal, str(pid)])
                )

            channel.verdict = verdict
            launch = shlex.join([*python, path, channel.journal, path])
            assert self.route.pin is not None
            command = typed_root_prelude(self.route.pin) + "; exec " + launch
            stage = "connecting the helper"
            await asyncio.wait_for(channel.connect(command, self.route), START_TIMEOUT)
            if not self.current():
                self.sandbox.logger.info(
                    "Daytona command channel did not start: provider generation changed"
                )
                return None
            self.sandbox._exec_counts().channel_starts += 1
            return channel
        except Exception as exc:
            self.sandbox.logger.info(
                "Daytona command channel did not start: %s (%s)",
                stage,
                type(exc).__name__,
            )
            if channel is not None and self.current():
                await channel.close()
            return None
        finally:
            if self.finalized and channel is not None and channel.pty is not None:
                # A late accepted create still belongs to this startup task.
                if channel.disconnecting is None:
                    await channel._disconnect()
                else:
                    await asyncio.shield(channel.disconnecting)

    async def _stage_source(self, source: bytes) -> str:
        if self.finalized:
            raise ChannelUnavailable("channel provider released")
        folder = _transport.stage_folder()
        self.folders.append(folder)
        path = folder + "/payload"
        # A retired startup must never retry through the facade's new provider.
        self.sandbox._exec_counts().channel_upload_requests += 1
        await asyncio.wait_for(
            self.provider.fs.upload_file(source, path), START_TIMEOUT
        )
        return path

    async def run(
        self,
        command: str,
        route: ExecRoute,
        timeout_sec: int | None,
        *,
        cleanup_session: bool = False,
    ) -> ExecResult | None:
        if not self.current() or not self.starting.done():
            return None
        channel = self.starting.result()
        if channel is None:
            return None
        if not channel.alive:
            if len(self.channels) <= 3:
                self.starting = asyncio.create_task(self._restart(channel))
            return None
        from benchflow.sandbox.daytona import _DAYTONA_EXEC_HARD_CAP_SEC

        capped = timeout_sec is None or timeout_sec <= 0
        budget = float(
            timeout_sec
            if timeout_sec is not None and timeout_sec > 0
            else _DAYTONA_EXEC_HARD_CAP_SEC
        )
        counts = self.sandbox._exec_counts()
        started = time.monotonic()
        key = object()
        inflight = getattr(self, "inflight", None)
        if inflight is None:
            inflight = self.inflight = {}
        inflight[key] = started
        try:
            counts.channel_exec_calls += 1
            result = await channel.run(
                command, route.shell, budget, cleanup_session=cleanup_session
            )
            return ExecResult(
                return_code=result.return_code,
                stdout=session_output(result.stream(1)),
                stderr=session_output(result.stream(2)),
            )
        except (TimeoutError, ChannelDeadline):
            raise self.sandbox._poll_timeout_error(timeout_sec, capped) from None
        except ChannelUnavailable:
            if not self.current():
                raise TransientSandboxTransportError(
                    "command channel generation changed"
                ) from None
            counts.channel_fallbacks += 1
            return None
        except ChannelUnconfirmed as exc:
            ids = await channel.started_ids()
            if self.current() and ids is not None and exc.ident not in ids:
                counts.channel_fallbacks += 1
                return None
            raise TransientSandboxTransportError(str(exc)) from exc
        except ChannelLost as exc:
            raise TransientSandboxTransportError(str(exc)) from exc
        finally:
            if inflight.pop(key, None) is not None:
                counts.channel_exec_seconds += time.monotonic() - started

    async def _restart(self, channel: DaytonaChannel) -> DaytonaChannel | None:
        ids = await channel.started_ids()
        await channel.close()
        if ids is None or not self.current():
            return None
        return await self._start()

    def retire(self) -> None:
        self.closed = True
        for channel in self.channels:
            channel.die("retired")

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        self.retire()
        counts = self.sandbox._exec_counts()
        inflight = getattr(self, "inflight", {})
        counts.channel_exec_pending_at_release += len(inflight)
        counts.channel_exec_seconds += sum(
            time.monotonic() - start for start in inflight.values()
        )
        inflight.clear()
        for task in (self.starting, self.cleanup_task):
            if task is not None and not task.done():
                task.cancel()
        for channel in self.channels:
            channel.finalize()
        self.folders = []

    async def join(self) -> None:
        """Drain released resources after the provider's bookkeeping is complete."""
        tasks = [self.starting]
        if self.cleanup_task is not None:
            tasks.append(self.cleanup_task)
        for channel in self.channels:
            tasks.extend(channel.tasks())
        # A cancelled startup can still receive and disconnect a late PTY.
        joined = asyncio.gather(*tasks, return_exceptions=True)
        cancelled = None
        try:
            await asyncio.wait_for(
                asyncio.shield(joined), START_TIMEOUT + CLOSE_TIMEOUT
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            cancelled = exc if isinstance(exc, asyncio.CancelledError) else None
            if isinstance(exc, TimeoutError):
                self.sandbox._exec_counts().channel_cleanup_timeouts += 1
            for task in tasks:
                if not task.done():
                    task.cancel()
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError as later:
                    if cancelled is None:
                        cancelled = later
        if cancelled is not None:
            raise cancelled

    async def close(self) -> None:
        self.retire()
        # Startup owns provider requests that may already have been accepted.
        # Join it rather than losing a late handle through cancellation.
        cancelled = False
        while not self.starting.done():
            try:
                await asyncio.shield(self.starting)
            except asyncio.CancelledError:
                if self.finalized:
                    raise
                if not self.starting.cancelled():
                    cancelled = True
        for channel in self.channels:
            await channel.close()
        for folder in getattr(self, "folders", ()):
            with contextlib.suppress(Exception):
                await self._control(f"{KIT_BIN}/rm -rf -- {shlex.quote(folder)}")
        self.folders = []
        if cancelled:
            raise asyncio.CancelledError


def eligible(sandbox: Any, route: ExecRoute) -> bool:
    process = getattr(getattr(sandbox, "_sandbox", None), "process", None)
    return (
        sandbox.exec_channel
        and callable(getattr(process, "create_pty_session", None))
        and route.kit
        and route.root
        and route.pin is not None
        and not route.task_root
        and not route.own_vm
        and not getattr(sandbox, "_command_channels_closing", False)
        and os.environ.get("BENCHFLOW_EXEC_CHANNEL", "1") != "0"
    )
