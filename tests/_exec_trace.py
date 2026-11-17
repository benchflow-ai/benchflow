"""Which root processes started in a container while a trace was open
(TRUSTED-RUNTIME.md §7, Done-when 1: every root process BenchFlow starts
after the agent is bfx or one of the sealed route's programs).

Two host-side sources, both read-only toward the traced container:

- **The daemon's exec records.** Every ``docker exec`` (BenchFlow's root
  steps, the exec channel's helper) appears in ``docker events`` as
  ``exec_create``; the Engine API's ``GET /exec/{id}/json`` then gives its
  user and argv exactly. Nothing is missed, however short-lived: the trace
  subscribes to the container's events before it opens and reads them live
  (``docker events --since`` read back at the end replays only the
  daemon's last 256 events, of every container, so a busy daemon pushed the
  traced execs out). The sampler's own start and exit, in the same stream,
  bracket the trace; a stream that misses either, or an exec's create,
  fails the trace instead of passing it.
- **A ``/proc`` sampler** for what those execs start in turn: a separate
  container that shares only the traced container's PID namespace
  (``--pid container:<id>``), with no network, no capabilities, a read-only
  root and no new privileges, lists every process whose real, effective or
  saved UID is 0 and that was not there when the trace opened, and each
  new program a process runs in place. It skips the container runtime's
  own ``runc init`` step of each exec (its parent is outside the
  namespace). It polls (``interval``), so a descendant that lives shorter
  than one interval can be missed; the exec records still name its parent.

Usage::

    with ExecTrace(container) as trace:
        ...  # the trial's root steps
    assert disallowed(trace.root_starts(), {"/.bfx/bfx", ...}) == []
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

SAMPLER_IMAGE = os.environ.get("BENCHFLOW_TEST_TRACE_IMAGE", "python:3.12-slim")
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=exec-trace")
PREFIX = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "exectrace-")

# Runs in the sampler container: one JSON line per root process that starts
# after the first pass (pid, start time, uids, argv), until stdin closes.
_SAMPLER = r"""
import json, os, sys, threading, time
interval = float(sys.argv[1])
me = os.getpid()
done = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), done.set()), daemon=True).start()

def table():
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            with open(f"/proc/{name}/status", "rb") as f:
                status = f.read().decode(errors="replace")
            with open(f"/proc/{name}/cmdline", "rb") as f:
                cmdline = f.read()
        except OSError:
            continue
        fields = stat[stat.rfind(b")") + 2 :].split()
        uids = next(
            (line.split()[1:4] for line in status.splitlines() if line.startswith("Uid:")),
            [],
        )
        argv = tuple(a.decode(errors="replace") for a in cmdline.split(b"\0") if a)
        # A process that runs a new program keeps its pid and start time:
        # the program is part of the key, so each program counts.
        out[(int(name), int(fields[19]), argv)] = (
            int(fields[1]),
            [int(u) for u in uids],
            fields[0].decode(),
        )
    return out

seen = set(table())
procs = {(pid, start) for pid, start, _ in seen}
print(json.dumps({"ready": True}), flush=True)
while not done.is_set():
    for key, (ppid, uids, state) in table().items():
        if key in seen:
            continue
        pid, start, argv = key
        # An empty argv on a live process is read mid-exec: read it again
        # next pass. On one that has exited it is news only when the process
        # was never seen running (its program is then unknown).
        if not argv and (state not in ("Z", "X") or (pid, start) in procs):
            continue
        seen.add(key)
        procs.add((pid, start))
        # The container runtime's own step for each docker exec, which then
        # becomes the exec's program (named by the daemon's exec record).
        if ppid == 0 and argv == ("runc", "init"):
            continue
        if 0 in uids:
            print(json.dumps({"pid": pid, "start": start, "ppid": ppid,
                              "uids": uids, "argv": list(argv)}), flush=True)
    time.sleep(interval)
"""


@dataclass(frozen=True)
class Start:
    """One root process start: from the daemon's exec record (``exec``) or
    seen by the sampler (``sampled``)."""

    source: str
    argv: tuple[str, ...]
    user: str
    pid: int | None = None
    ppid: int | None = None


def _docker_socket() -> str:
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        return host[len("unix://") :]
    return "/var/run/docker.sock"


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str) -> None:
        super().__init__("localhost", timeout=30)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(self._path)


def exec_record(exec_id: str) -> dict:
    """The Engine API's record of one exec (``GET /exec/{id}/json``)."""
    connection = _UnixConnection(_docker_socket())
    try:
        connection.request("GET", f"/exec/{exec_id}/json")
        reply = connection.getresponse()
        body = reply.read()
        if reply.status != 200:
            raise RuntimeError(f"exec {exec_id}: {reply.status} {body[:200]!r}")
        return json.loads(body)
    finally:
        connection.close()


def _is_root(user: str, default_user: str) -> bool:
    name = (user or default_user or "").split(":")[0]
    return name in ("", "0", "root")


class ExecTrace:
    """Records the root process starts in *container* while open."""

    def __init__(
        self, container: str, *, interval: float = 0.005, fence: float = 60.0
    ) -> None:
        self.container = container
        self.interval = interval
        self.fence = fence
        self.sampled: list[dict] = []
        self.exec_ids: list[str] = []
        self.events: list[dict] = []
        self._threads: list[threading.Thread] = []
        self._sampler_name = PREFIX + "sampler-" + uuid.uuid4().hex[:8]
        self._sampler_started = threading.Event()
        self._sampler_gone = threading.Event()
        self._stream_ended = threading.Event()

    def __enter__(self) -> ExecTrace:
        since = f"{time.time():.6f}"
        inspected = subprocess.run(
            ["docker", "inspect", "--format",
             "{{.Id}}\t{{json .ExecIDs}}\t{{.Config.User}}", self.container],
            capture_output=True,
            text=True,
            check=True,
        )  # fmt: skip
        self.container_id, running, self.default_user = inspected.stdout.rstrip(
            "\n"
        ).split("\t", 2)
        # Execs already running may end inside the trace: their records
        # come from before it.
        self._running_before = set(json.loads(running) or [])
        # The replay from *since* covers the moment before the subscription
        # is live; the sampler's start, waited for below, shows that it is.
        self._events = subprocess.Popen(
            [
                "docker", "events", "--since", since,
                "--filter", f"container={self.container_id}",
                "--filter", f"container={self._sampler_name}",
                "--format", "{{json .}}",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )  # fmt: skip
        events = threading.Thread(target=self._read_events, daemon=True)
        events.start()
        self._threads.append(events)
        key, _, value = LABEL.partition("=")
        self._sampler = subprocess.Popen(
            [
                "docker", "run", "--rm", "-i", "--name", self._sampler_name,
                "--label", f"{key}={value}",
                "--pid", f"container:{self.container_id}",
                "--network", "none", "--cap-drop", "ALL", "--read-only",
                "--security-opt", "no-new-privileges:true", "--user", "0:0",
                SAMPLER_IMAGE, "python3", "-I", "-c", _SAMPLER, str(self.interval),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )  # fmt: skip
        assert self._sampler.stdout is not None
        first = self._sampler.stdout.readline()
        if json.loads(first or "{}").get("ready") is not True:
            self._sampler.kill()
            self._stop_events()
            raise RuntimeError(f"the sampler did not start: {first!r}")
        if not self._wait_for(self._sampler_started):
            self._stop_sampler()
            self._stop_events()
            raise RuntimeError(
                "the event stream never showed the sampler's start: "
                "the trace could miss exec records"
            )
        reader = threading.Thread(target=self._read_samples, daemon=True)
        reader.start()
        self._threads.append(reader)
        return self

    def _read_samples(self) -> None:
        assert self._sampler.stdout is not None
        for line in self._sampler.stdout:
            self.sampled.append(json.loads(line))

    def _read_events(self) -> None:
        assert self._events.stdout is not None
        for line in self._events.stdout:
            event = json.loads(line)
            self.events.append(event)
            attributes = (event.get("Actor") or {}).get("Attributes") or {}
            if attributes.get("name") == self._sampler_name:
                if event.get("Action") == "start":
                    self._sampler_started.set()
                elif event.get("Action") in ("die", "destroy"):
                    self._sampler_gone.set()
        self._stream_ended.set()

    def _wait_for(self, seen: threading.Event) -> bool:
        """Whether the event stream shows *seen* within the fence; False as
        soon as the stream ends without it."""
        until = time.monotonic() + self.fence
        while not seen.wait(0.05):
            if self._stream_ended.is_set() or time.monotonic() > until:
                return seen.is_set()
        return True

    def _stop_sampler(self) -> None:
        assert self._sampler.stdin is not None
        self._sampler.stdin.close()
        try:
            self._sampler.wait(timeout=30)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", self._sampler_name], capture_output=True
            )
            self._sampler.wait(timeout=30)

    def _stop_events(self) -> None:
        self._events.terminate()
        try:
            self._events.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._events.kill()
            self._events.wait(timeout=30)

    def __exit__(self, *exc) -> None:
        # Let a last pass see what the final steps started.
        time.sleep(max(self.interval * 4, 0.05))
        self._stop_sampler()
        # The sampler exits after every traced step returned, so the stream
        # has delivered their exec records once it shows that exit.
        complete = self._wait_for(self._sampler_gone)
        self._stop_events()
        for thread in self._threads:
            thread.join(timeout=10)
        created, used = [], set()
        for event in self.events:
            actor = event.get("Actor") or {}
            exec_id = (actor.get("Attributes") or {}).get("execID")
            if actor.get("ID") != self.container_id or not exec_id:
                continue
            action = str(event.get("Action") or "").partition(":")[0]
            if action == "exec_create":
                created.append(exec_id)
            elif action in ("exec_start", "exec_die"):
                used.add(exec_id)
        self.exec_ids = list(dict.fromkeys(created))
        if exc[0] is not None:
            return
        if not complete:
            raise RuntimeError(
                "the event stream never showed the sampler's exit: "
                "exec records may be missing"
            )
        unrecorded = used - set(created) - self._running_before
        if unrecorded:
            raise RuntimeError(
                f"execs started without an exec_create record: {sorted(unrecorded)}"
            )

    def root_starts(self) -> list[Start]:
        """Every root start the trace saw: each root exec from the daemon's
        records, then each root process the sampler saw (an exec's own
        process usually appears in both; the exec's pid is the host's, so
        the two are not merged)."""
        starts: list[Start] = []
        for exec_id in self.exec_ids:
            record = exec_record(exec_id)
            config = record.get("ProcessConfig") or {}
            user = str(config.get("user") or "")
            if not _is_root(user, self.default_user):
                continue
            argv = (
                str(config.get("entrypoint") or ""),
                *map(str, config.get("arguments") or []),
            )
            starts.append(Start("exec", argv, user or self.default_user or "root"))
        for sample in self.sampled:
            uids = sample["uids"]
            user = "root" if uids and uids[1] == 0 else f"uids {uids}"
            starts.append(
                Start(
                    "sampled",
                    tuple(sample["argv"]),
                    user,
                    sample["pid"],
                    sample["ppid"],
                )
            )
        return starts


def program(start: Start) -> str:
    """argv[0], or "" when the sampler saw no program (the process had
    already exited)."""
    return start.argv[0] if start.argv else ""


def disallowed(starts: Iterable[Start], allowed: Iterable[str]) -> list[Start]:
    """The starts whose program (argv[0]) is not one of *allowed*. A sampled
    process with an empty argv (a kernel thread, or one that exited between
    reads) is reported too."""
    names = set(allowed)
    return [s for s in starts if program(s) not in names]
