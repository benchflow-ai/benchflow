"""One long-lived command channel per Docker sandbox (PERF-B).

Every ``docker exec`` starts the Docker CLI, makes the daemon create and
start an exec process and asks it for the exit code: about 0.05 s on an idle
build VM and 0.1 s on a busy one, and a trial runs about fifty of root's
steps that way. The channel pays that once: one ``docker exec -i`` starts a
small helper in the container (bfx, :func:`helper_argv`, where it is
delivered; else the kit's Python running :data:`HELPER_SOURCE`), and each later
command is a frame on its stdin that the helper starts as a child process.

The helper changes nothing about the command it starts:

- **The same process.** The helper is started with the ``-u`` and the
  empty loader values (``-e LD_PRELOAD=`` and the rest) the per-command
  ``docker exec`` would have used, so it has the environment Docker gives
  such an exec, and it hands that environment on unchanged: it is bfx,
  BenchFlow's static helper, started by absolute path from the read-only
  ``/.bfx`` (TRUSTED-RUNTIME.md §6), which reads nothing from that
  environment and changes nothing in it, so no shell adds ``SHLVL`` or
  ``_`` on the way and no interpreter adds a locale. Where bfx is not
  delivered it is the kit's Python (:data:`~benchflow._runtime_tree.KIT_PYTHON`),
  started through the loader line of the kit's own wrapper rather than the
  wrapper, and ``-I -S -B`` keep every ``PYTHON*`` value and site folder out
  of it; what Python adds to its own environment (``LC_CTYPE`` under a C
  locale) is undone from ``/proc/self/environ`` before any command starts.
  The command line is
  the one ``docker exec`` would run (``<route shell> -c <routed command>``):
  the route's absolute shell, PATH pin and ``env -i`` restart (C8), the
  env on stdin, never a value in argv. The child is a
  session leader, as Docker's exec is; its stdin is ``/dev/null`` unless
  the command has input, its stdout and stderr two pipes.
- **K1's barrier.** On Docker, no_new_privs is set when the container is
  created (``agent_container_no_new_privileges``), so the helper and every
  child have it, exactly as a per-command exec does.
- **The working folder.** Docker changes to ``-w`` or the image's
  ``WorkingDir`` by path for each exec; the helper does too, per command,
  never from a folder it holds open. A folder that is not there makes the
  command fall back to its own ``docker exec``, whose error is Docker's.
- **The results.** The exit status is the child's, ``128 + n`` when signal
  *n* ended it (Docker's 137 for SIGKILL). After the child exits the helper
  keeps reading what a background process still writes to its pipes for
  2 s, as Docker does, then stops reading.
- **Timeouts and Ctrl-C.** The host stops waiting and raises, as it does
  when it kills a ``docker exec`` client; the command goes on in the
  container in both cases (killing the client never stops the exec).

Only BenchFlow's own root plumbing (uid 0, a pinned route) on a sandbox
that holds the protected kit and has bfx delivered, or whose kit has a
Python, goes through the channel. The
task's own commands as root (test.sh, an oracle's solve) keep their own
``docker exec``: such a command may kill processes by pattern (``pkill -f
python``), and the helper is a process in the container it could kill,
while a ``docker exec`` reports through a parent outside the container
(review PERF-B-M1).
A sandbox without bfx whose kit has no Python (an image with no Python 3)
but GNU ``head`` gets :class:`WarmExecPool` instead: a few ``docker exec -i`` processes
started ahead, each of which reads one plumbing step and becomes it, so
every step is still Docker's own exec, without waiting for its start.
Anything else, and any command neither can start, runs through its own
``docker exec`` as before. Before a step can exist, the helper appends
``started <id>`` to its journal, a new root-only file in root's run folder
(fsync'd), and then sends an S frame. If the channel dies, a step whose S
frame arrived fails with an error that says so: it may or may not have run,
so it is never started a second time. For a step without its S frame the
host cannot tell (the transport, a killed docker client or a dropped remote
connection, can lose the frame while the helper lives on), so the sandbox
kills the old helper and reads its journal with its own ``docker exec``:
only a step the journal does not list takes its own ``docker exec``; a
listed step, or a journal that cannot be read, fails as "may or may not have
run". Later commands use ``docker exec`` while the
sandbox starts a new helper (at most three times), once the old one is gone.

A step is sent with the time its caller has left, which the helper counts on
its own monotonic clock from the frame's arrival; when the caller gives up
(a timeout, Ctrl-C) before the S frame, it also sends a cancel frame, and the
helper reads all its input before starting anything, so a step that waited
in the pipe past its caller never starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from benchflow._runtime_tree import KIT
from benchflow.sandbox import _transport

#: How long the host waits for a new channel's helper to say it is ready.
START_TIMEOUT_SEC = 15.0
#: How long the host waits for the helper to leave after its stdin closes.
CLOSE_GRACE_SEC = 1.0

#: The two wrappers ``kit_prepare_command`` writes at ``KIT_PYTHON``: the
#: kit's Python, through the system loader with the kit's libraries when the
#: interpreter loads one from outside the system folders, or directly.
_WRAPPER_WITH_LOADER = re.compile(
    rf"#!{re.escape(KIT)}/bin/sh\n"
    r"exec (?P<ld>/(?:lib|lib64|usr/lib|usr/lib64)/ld-(?:linux[-.\w]*\.so\.\d+|musl-\w+\.so\.1))"
    rf" --library-path {re.escape(KIT)}/lib {re.escape(KIT)}/libexec/python3"
    r' -I -S -B "\$@"\n?'
)
_WRAPPER_DIRECT = re.compile(
    rf"#!{re.escape(KIT)}/bin/sh\n"
    rf"exec {re.escape(KIT)}/libexec/python3"
    r' -I -S -B "\$@"\n?'
)


def helper_python_argv(wrapper: str) -> list[str] | None:
    """The kit Python's argv without its shell wrapper, or None.

    *wrapper* is the text of :data:`KIT_PYTHON`; only the two forms
    ``kit_prepare_command`` writes are accepted.
    """
    match = _WRAPPER_WITH_LOADER.fullmatch(wrapper)
    if match is not None:
        return [
            match["ld"],
            "--library-path",
            f"{KIT}/lib",
            f"{KIT}/libexec/python3",
            "-I",
            "-S",
            "-B",
        ]
    if _WRAPPER_DIRECT.fullmatch(wrapper) is not None:
        return [f"{KIT}/libexec/python3", "-I", "-S", "-B"]
    return None


#: The helper where bfx is delivered: bfx (``bfx/src/channel.rs``), started
#: by absolute path from the read-only ``/.bfx``. Frames in, from the host: a 4-byte big-endian
#: length, that much JSON (``{"i": id, "a": argv, "d": cwd, "n": stdin bytes
#: or -1 for none, "r": seconds left}``, or a cancel ``{"c": id}``), then the
#: stdin bytes. Frames out: one byte kind, a 4-byte id, a 4-byte length, the
#: payload. Kinds: R ready (the helper's pid), S starting (after the journal
#: lists the step), O and E output chunks, X exit (the status as text), F not
#: started (the reason). Its last argument, when given, is the journal's path.
def helper_argv(bfx: str, journal: str | None = None) -> list[str]:
    """The helper's argv after the ``docker exec`` prefix."""
    return [bfx, "serve", "--frames", *([journal] if journal is not None else [])]


#: After a channel died, run on the same route as its helper with the
#: journal's path and the helper's pid: bfx kills that helper (only while its
#: command line still names this journal), waits until it is gone, then
#: prints "J" and the journal. Any other output, or exit status, means the
#: host learned nothing.
def verdict_argv(bfx: str) -> list[str]:
    return [bfx, "serve", "--verdict"]


# The helper where bfx is not delivered and the kit has a Python (today's
# route, kept: TRUSTED-RUNTIME.md v3 scope), run by the kit's Python by its
# loader line (helper_python_argv), with the same frames and journal as
# bfx's. It runs on whatever Python 3 the image has (3.5 and up), so it
# uses no f-strings, walrus or newer library calls. Frames in, from the host:
# a 4-byte big-endian length, that much JSON ({"i": id, "a": argv, "d": cwd,
# "n": stdin bytes or -1 for none}), then the stdin bytes. Frames out: one
# byte kind, a 4-byte id, a 4-byte length, the payload. Kinds: R ready (the
# helper's pid), S starting (after the journal lists the step), O and E
# output chunks, X exit (the status as text), F not started (the reason).
# Its one argument, when given, is the journal's path.
HELPER_SOURCE = r"""
import errno, json, os, selectors, signal, struct, subprocess, sys, time
GRACE = 2.0
# Children inherit the environment Docker gave this exec, byte for byte:
# Python itself adds to its own (PEP 538 sets LC_CTYPE under a C locale,
# and -I cannot stop it), so undo whatever differs from the block the
# kernel holds from the exec.
with open('/proc/self/environ', 'rb') as f:
    given = [e.split(b'=', 1) for e in f.read().split(b'\0') if b'=' in e]
given = dict((k, v) for k, v in given)
for k in list(os.environb):
    if k not in given:
        del os.environb[k]
for k, v in given.items():
    if os.environb.get(k) != v:
        os.environb[k] = v
sel = selectors.DefaultSelector()
rfd, wfd = os.pipe()
os.set_blocking(rfd, False); os.set_blocking(wfd, False)
signal.set_wakeup_fd(wfd)
signal.signal(signal.SIGCHLD, lambda *a: None)
os.set_blocking(0, False)
inbuf = b''
jobs = {}
# The write-ahead start record: a step is listed here before it can exist,
# so after a death the host reads which steps may have run. Root's alone, a
# new file per helper; no journal (a local test) means none is written.
journal = None
if len(sys.argv) > 1:
    journal = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
def out(kind, i, data):
    data = kind + struct.pack('>II', i, len(data)) + data
    while data:
        try:
            n = os.write(1, data)
        except InterruptedError:
            continue
        data = data[n:]
def start(head, body):
    i = head['i']
    # The caller's time left, counted on this monotonic clock from the
    # frame's arrival: a step past it (its caller gave up) never starts.
    if head.get('due') is not None and time.monotonic() > head['due']:
        out(b'F', i, b'deadline passed before the step started')
        return
    if journal is not None:
        line = ('started %d\n' % i).encode()
        try:
            if os.write(journal, line) != len(line):
                raise OSError('short write')
            os.fsync(journal)
        except OSError as e:
            out(b'F', i, ('could not record the start: %r' % (e,)).encode()[:2000])
            return
    # Said before the step can exist: the host that reads it knows the step
    # may have started.
    out(b'S', i, b'')
    try:
        p = subprocess.Popen(head['a'], cwd=head['d'],
            stdin=subprocess.PIPE if head['n'] >= 0 else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True)
    except BaseException as e:
        out(b'F', i, repr(e).encode('utf-8', 'replace')[:2000])
        return
    job = {'p': p, 'open': 2, 'end': None, 'in': body}
    jobs[i] = job
    sel.register(p.stdout, selectors.EVENT_READ, (i, b'O'))
    sel.register(p.stderr, selectors.EVENT_READ, (i, b'E'))
    if p.stdin is not None:
        os.set_blocking(p.stdin.fileno(), False)
        if body:
            sel.register(p.stdin, selectors.EVENT_WRITE, (i, b'I'))
        else:
            p.stdin.close()
def feed(i, job):
    p = job['p']
    try:
        n = os.write(p.stdin.fileno(), job['in'][:65536])
    except (BlockingIOError, InterruptedError):
        return
    except OSError:
        n = len(job['in'])
    job['in'] = job['in'][n:]
    if not job['in']:
        sel.unregister(p.stdin)
        p.stdin.close()
def close_stdin(job):
    p = job['p']
    if p.stdin is not None and not p.stdin.closed:
        with_reg = job['in']
        job['in'] = b''
        if with_reg:
            sel.unregister(p.stdin)
        p.stdin.close()
def finish(i, job):
    for f in (job['p'].stdout, job['p'].stderr):
        if not f.closed:
            sel.unregister(f)
            f.close()
    close_stdin(job)
    rc = job['p'].returncode
    out(b'X', i, str(128 - rc if rc < 0 else rc).encode())
    del jobs[i]
def reap():
    now = time.monotonic()
    for i, job in list(jobs.items()):
        if job['end'] is None and job['p'].poll() is not None:
            job['end'] = now
        if job['end'] is not None and (job['open'] == 0 or now - job['end'] >= GRACE):
            finish(i, job)
def take():
    # Every whole frame in, then the steps no cancel frame withdrew: a
    # step whose caller gave up while it waited in the pipe never starts.
    global inbuf
    steps = []
    cancelled = set()
    while len(inbuf) >= 4:
        size = struct.unpack('>I', inbuf[:4])[0]
        if len(inbuf) < 4 + size:
            break
        head = json.loads(inbuf[4:4 + size].decode('utf-8'))
        n = max(head['n'], 0)
        if len(inbuf) < 4 + size + n:
            break
        body = inbuf[4 + size:4 + size + n]
        inbuf = inbuf[4 + size + n:]
        if head.get('c') is not None:
            cancelled.add(head['c'])
            continue
        if head.get('r') is not None:
            head['due'] = time.monotonic() + head['r']
        steps.append((head, body))
    for head, body in steps:
        if head['i'] in cancelled:
            out(b'F', head['i'], b'deadline passed before the step started')
        else:
            start(head, body)
sel.register(0, selectors.EVENT_READ, (0, b'H'))
sel.register(rfd, selectors.EVENT_READ, (0, b'W'))
out(b'R', 0, str(os.getpid()).encode())
while True:
    wait = None
    ends = [j['end'] for j in jobs.values() if j['end'] is not None]
    if ends:
        wait = max(0.0, min(ends) + GRACE - time.monotonic())
    for key, ev in sel.select(wait):
        i, kind = key.data
        if kind == b'H':
            # Read all there is before starting anything, so a cancel that
            # is already here is seen.
            got = False
            while True:
                try:
                    data = os.read(0, 1 << 20)
                except InterruptedError:
                    continue
                except BlockingIOError:
                    break
                if not data:
                    os._exit(0)
                inbuf += data
                got = True
            if got:
                take()
        elif kind == b'W':
            try:
                while os.read(rfd, 512):
                    pass
            except (BlockingIOError, InterruptedError):
                pass
        elif kind == b'I':
            if i in jobs:
                feed(i, jobs[i])
        else:
            job = jobs.get(i)
            if job is None:
                continue
            try:
                data = os.read(key.fd, 65536)
            except InterruptedError:
                continue
            if data:
                out(kind, i, data)
            else:
                sel.unregister(key.fileobj)
                key.fileobj.close()
                job['open'] -= 1
    reap()
"""


# After a channel died: run on the same route as its helper, with the
# journal's path and the helper's pid. Kills that helper (only while its
# command line still names this journal), waits until it is gone, then
# prints "J" and the journal. Any other output, or exit status, means the
# host learned nothing.
VERDICT_SOURCE = r"""
import os, signal, sys, time
path, pid = sys.argv[1], int(sys.argv[2])
mark = path.encode()
def ours():
    try:
        with open('/proc/%d/cmdline' % pid, 'rb') as f:
            return mark in f.read().split(b'\0')
    except OSError:
        return False
if pid > 1 and ours():
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
end = time.monotonic() + 10
while pid > 1 and ours():
    if time.monotonic() > end:
        sys.exit(3)
    time.sleep(0.01)
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
data = b''
while True:
    b = os.read(fd, 65536)
    if not b:
        break
    data += b
os.write(1, b'J' + data)
"""


def parse_journal(output: bytes | str | None) -> frozenset[int] | None:
    """The step ids a dead helper's journal lists, or None when the output
    is not a whole journal (then every unconfirmed step may have run)."""
    if isinstance(output, str):
        output = output.encode()
    if not output or not output.startswith(b"J"):
        return None
    body = output[1:]
    if body and not body.endswith(b"\n"):
        return None  # a line cut short: anything may have started
    ids = set()
    for line in body.splitlines():
        kind, _, ident = line.partition(b" ")
        if kind != b"started" or not ident.isdigit():
            return None
        ids.add(int(ident))
    return frozenset(ids)


class ChannelUnavailable(Exception):
    """The channel cannot run this command; it was not started. Run it
    through its own ``docker exec``."""


class ChannelDeadline(ChannelUnavailable):
    """The helper got the step after its caller's deadline and did not
    start it: the step timed out, as a ``docker exec`` that never began."""


class ChannelLost(RuntimeError):
    """The channel died while a command ran; the command may or may not
    have run."""


class ChannelUnconfirmed(ChannelLost):
    """The channel died before the host read the step's S frame. The step
     may still have started (the transport can lose the frame while the
     helper lives); only the dead helper's journal can tell
    ."""

    def __init__(self, message: str, ident: int) -> None:
        super().__init__(message)
        self.ident = ident


@dataclass
class _Pending:
    future: asyncio.Future[int]
    chunks: list[tuple[int, bytes]] = field(default_factory=list)
    # The helper said it is starting the step (an S frame).
    started: bool = False


@dataclass
class ChannelOutput:
    """What one command wrote, in the order the helper read it."""

    return_code: int
    chunks: list[tuple[int, bytes]]

    def stream(self, fd: int) -> bytes:
        return b"".join(data for which, data in self.chunks if which == fd)

    def merged(self) -> bytes:
        return b"".join(data for _, data in self.chunks)


class ExecChannel:
    """The host's end of one helper (see the module docstring)."""

    def __init__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
        *,
        workdir: str,
        logger: Any,
        journal: str | None = None,
    ) -> None:
        self._argv = list(argv)
        #: The helper's start journal (its path is the helper's last
        #: argument), and its pid in the container, from the ready frame.
        self.journal = journal
        self.helper_pid: int | None = None
        #: The argv that runs bfx's verdict (:func:`verdict_argv`) on the helper's own route.
        self.verify_argv: list[str] | None = None
        #: What the sandbox learned after this channel died (docker.py).
        self.death_check: asyncio.Future[frozenset[int] | None] | None = None
        self._env = dict(env) if env is not None else None
        self.workdir = workdir
        self._logger = logger
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[bytes] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._dead: str | None = None

    @property
    def alive(self) -> bool:
        return self._dead is None and self._process is not None

    async def start(self, timeout_sec: float = START_TIMEOUT_SEC) -> None:
        """Start the helper and wait until it says it is ready; raises
        :class:`ChannelUnavailable` when it does not."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        try:
            # Its own session, like every docker client BenchFlow starts:
            # Ctrl-C reaches bench, never the channel the clean-up still uses.
            _transport.sent("Docker", self._argv)
            self._process = await asyncio.create_subprocess_exec(
                *self._argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            self._dead = f"could not start: {exc}"
            raise ChannelUnavailable(self._dead) from exc
        self._stderr_task = asyncio.create_task(self._read_stderr())
        self._reader = asyncio.create_task(self._read())
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), timeout=timeout_sec)
        except (TimeoutError, ChannelLost) as exc:
            await self.close()
            raise ChannelUnavailable(f"the helper did not start: {exc}") from exc

    async def _read_stderr(self) -> bytes:
        assert self._process is not None and self._process.stderr is not None
        return await self._process.stderr.read()

    async def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stdout = self._process.stdout
        reason = "the helper closed its output"
        try:
            while True:
                head = await stdout.readexactly(9)
                kind = head[:1]
                ident, size = struct.unpack(">II", head[1:])
                payload = await stdout.readexactly(size) if size else b""
                if kind == b"R":
                    with contextlib.suppress(ValueError):
                        self.helper_pid = int(payload.decode())
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(None)
                    continue
                pending = self._pending.get(ident)
                if pending is None:
                    # A command whose caller stopped waiting (a timeout, a
                    # cancel): it runs on, its output goes nowhere.
                    continue
                if kind == b"S":
                    pending.started = True
                elif kind in (b"O", b"E"):
                    pending.chunks.append((1 if kind == b"O" else 2, payload))
                elif pending.future.done():
                    continue
                elif kind == b"X":
                    pending.future.set_result(int(payload.decode()))
                elif kind == b"F":
                    why = payload.decode(errors="replace")
                    late = why.startswith("deadline passed")
                    pending.future.set_exception(
                        (ChannelDeadline if late else ChannelUnavailable)(why)
                    )
        except asyncio.IncompleteReadError:
            pass
        except asyncio.CancelledError:
            reason = "the channel was closed"
            raise
        except Exception as exc:  # a frame the host cannot read
            reason = f"unreadable frame: {exc}"
        finally:
            await self._die(reason)

    async def _die(self, reason: str) -> None:
        if self._dead is None:
            detail = ""
            if self._stderr_task is not None and self._stderr_task.done():
                with contextlib.suppress(Exception):
                    detail = self._stderr_task.result().decode(errors="replace")
            self._dead = reason + (
                f" ({detail.strip()[-300:]})" if detail.strip() else ""
            )
        lost = ChannelLost(f"the sandbox's command channel stopped: {self._dead}")
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(lost)
            # Retrieved by start(); keep asyncio quiet if nobody waits.
            self._ready.exception()
        for ident, pending in self._pending.items():
            if not pending.future.done():
                # No S frame read is no proof the step never started: the
                # transport may have lost it while the helper ran the step
                # (review R3-M1). The sandbox reads the journal to decide.
                pending.future.set_exception(
                    lost
                    if pending.started
                    else ChannelUnconfirmed(
                        "the sandbox's command channel stopped before the step "
                        f"was seen to start: {self._dead}",
                        ident,
                    )
                )

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None,
        stdin_data: bytes | None,
        deadline: float | None = None,
    ) -> ChannelOutput:
        """Run *argv* in *cwd* (the image's working folder when None) and
        return its output and exit status. *deadline* (``time.monotonic()``
        here): the step is sent with the time left, and the helper does not
        start it once that has run out on its own clock, or once a cancel
        frame (sent when this call is given up) has reached it
        (:class:`ChannelDeadline`).

        Raises :class:`ChannelUnavailable` when the command was not started
        (the channel is down, or the helper could not start it), and
        :class:`ChannelLost` when the channel died while it ran.
        """
        if not self.alive:
            raise ChannelUnavailable(self._dead or "not started")
        process = self._process
        assert process is not None and process.stdin is not None
        ident = self._next_id
        self._next_id += 1
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        pending = _Pending(future)
        self._pending[ident] = pending

        def frame(head: dict[str, Any], body: bytes = b"") -> bytes:
            data = json.dumps(head).encode()
            return struct.pack(">I", len(data)) + data + body

        try:
            try:
                async with self._write_lock:
                    # The time left, not a clock reading: the helper counts
                    # it on its own monotonic clock (review R2-L1).
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        raise ChannelDeadline(
                            "deadline passed before the step was sent"
                        )
                    process.stdin.write(
                        frame(
                            {
                                "i": ident,
                                "a": list(argv),
                                "d": cwd or self.workdir,
                                "n": -1 if stdin_data is None else len(stdin_data),
                                "r": left,
                            },
                            stdin_data or b"",
                        )
                    )
                    await process.stdin.drain()
            except (ConnectionError, RuntimeError) as exc:
                # The helper starts a command only once its whole frame is
                # in: this one never started.
                self._pending.pop(ident, None)
                await self._die(f"could not write to the helper: {exc}")
                raise ChannelUnavailable(str(exc)) from exc
            return_code = await future
            return ChannelOutput(return_code, pending.chunks)
        finally:
            self._pending.pop(ident, None)
            if future.done() and not future.cancelled():
                future.exception()
            elif self.alive and not pending.started:
                # The caller gave up (a timeout, Ctrl-C) before the step
                # started: withdraw it, so a helper that has not read it
                # yet never starts it.
                with contextlib.suppress(Exception):
                    process.stdin.write(frame({"c": ident, "n": -1}))

    async def close(self) -> None:
        """Close the helper's stdin (it leaves) and stop the client."""
        process = self._process
        if process is None:
            return
        if self._dead is None:
            self._dead = "closed"
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=CLOSE_GRACE_SEC)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if os.name == "posix" and process.pid > 1:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=CLOSE_GRACE_SEC)
        for task in (self._reader, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task


#: Pre-started execs a :class:`WarmExecPool` keeps ready.
WARM_POOL_SIZE = 3


def warm_reader(shell: str, head: str) -> str:
    """The shell text a pre-started exec runs: read one command, then become
    ``<shell> -c <command>``, the process the command's own ``docker exec``
    would have started.

    Only builtins and the kit's GNU ``head`` run first: ``read`` takes the
    header line (the command's length, whether input follows), ``head -c``
    exactly that many bytes (GNU head never reads past its count, so the
    command's input stays on stdin for it), and a last ``OK`` line proves
    the command arrived whole; a cut-off write exits 125 and runs nothing.
    Without input, stdin becomes /dev/null, as for ``docker exec`` with no
    ``-i``. The reader's variables are unset before the ``exec``.
    """
    return (
        "IFS=' ' read -r __bf_warm_n __bf_warm_in || exit 125; "
        f'__bf_warm_c=$({head} -c "$__bf_warm_n"; printf x) || exit 125; '
        "__bf_warm_c=${__bf_warm_c%x}; "
        "IFS= read -r __bf_warm_ok || exit 125; "
        '[ "$__bf_warm_ok" = OK ] || exit 125; '
        '[ "$__bf_warm_in" = 1 ] || exec </dev/null; '
        'set -- "$__bf_warm_c"; '
        "unset __bf_warm_n __bf_warm_in __bf_warm_c __bf_warm_ok; "
        f'exec {shell} -c "$1"'
    )


class WarmExecPool:
    """Pre-started ``docker exec`` processes for a kit without a Python.

    Each is the ``docker exec -i`` a root plumbing step would start, begun
    before the step is known: :func:`warm_reader` reads the step and
    ``exec``s ``<route shell> -c <command>``, so the process, its exit
    status, Docker's 2 s wait, a timeout and Ctrl-C (the client's process
    group is killed, the command goes on) are Docker's own. Only plumbing
    steps on the route shell in the image's working folder use it; each
    process runs one command and a new one starts in its place.
    """

    def __init__(
        self,
        argv_prefix: Sequence[str],
        env: Mapping[str, str] | None,
        *,
        shell: str,
        head: str,
        size: int = WARM_POOL_SIZE,
    ) -> None:
        self._argv = [*argv_prefix, shell, "-c", warm_reader(shell, head)]
        self._env = dict(env) if env is not None else None
        self.shell = shell
        self._size = size
        self._ready: list[asyncio.subprocess.Process] = []
        self._spawning: set[asyncio.Task[None]] = set()
        self._closed = False
        self.workdir = None

    @property
    def alive(self) -> bool:
        return not self._closed

    def fill(self) -> None:
        while not self._closed and len(self._ready) + len(self._spawning) < self._size:
            task = asyncio.ensure_future(self._spawn())
            self._spawning.add(task)
            task.add_done_callback(self._spawning.discard)

    async def _spawn(self) -> None:
        _transport.sent("Docker", self._argv)
        process = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            start_new_session=os.name == "posix",
        )
        if self._closed:
            await _stop_client(process)
        else:
            self._ready.append(process)

    def _take(self) -> asyncio.subprocess.Process:
        while self._ready:
            process = self._ready.pop(0)
            if process.returncode is None:
                self.fill()
                return process
        self.fill()
        raise ChannelUnavailable("no pre-started exec is ready")

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None,
        stdin_data: bytes | None,
    ) -> ChannelOutput:
        """Run *argv* (``[shell, "-c", command]``) on a pre-started exec;
        :class:`ChannelUnavailable` when none is ready or the step is not
        one the pool can start (another shell, a ``-w`` folder)."""
        if (
            self._closed
            or cwd is not None
            or len(argv) != 3
            or list(argv[:2]) != [self.shell, "-c"]
        ):
            raise ChannelUnavailable("not a step for the pre-started execs")
        process = self._take()
        assert process.stdin is not None
        command = argv[2].encode()
        flag = b"0" if stdin_data is None else b"1"
        try:
            try:
                process.stdin.write(
                    b"%d %s\n" % (len(command), flag)
                    + command
                    + b"OK\n"
                    + (stdin_data or b"")
                )
                await process.stdin.drain()
                process.stdin.close()
            except (ConnectionError, RuntimeError) as exc:
                await _stop_client(process)
                if stdin_data:
                    # The client may have passed on the command and "OK"
                    # and died with the stdin bytes still pending: the step
                    # may have started (review R3-M1's class).
                    raise ChannelLost(
                        f"the pre-started exec stopped while its input was sent: {exc}"
                    ) from exc
                # Cut off before the frame ended, with "OK" last: the
                # reader exits 125, nothing ran.
                raise ChannelUnavailable(str(exc)) from exc
            chunks: list[tuple[int, bytes]] = []

            async def pump(stream: asyncio.StreamReader | None, fd: int) -> None:
                assert stream is not None
                while data := await stream.read(65536):
                    chunks.append((fd, data))

            await asyncio.gather(pump(process.stdout, 1), pump(process.stderr, 2))
            return ChannelOutput(await process.wait(), chunks)
        except BaseException:
            # A timeout or Ctrl-C, as for any docker exec client.
            await _stop_client(process)
            raise

    async def close(self) -> None:
        self._closed = True
        for task in list(self._spawning):
            await asyncio.wait([task])
        ready, self._ready = self._ready, []
        for process in ready:
            await _stop_client(process)


async def _stop_client(process: asyncio.subprocess.Process) -> None:
    """Kill a docker client's process group and reap it (an unused
    pre-started exec's reader then sees EOF and runs nothing)."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.name == "posix" and process.pid > 1:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=CLOSE_GRACE_SEC)
