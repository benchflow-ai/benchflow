"""Opt-in local runtime. Fresh forked CLI workers, no state shared by trials.

Only the parent imports are shared. It never runs a task or mutates a job.
Requests transfer stdio descriptors so Click/Rich, log files and result writers
take the normal paths. Before acceptance a failed attachment can fall back;
after acceptance it must never replay a possibly-running job.
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import fcntl
import hashlib
import importlib
import io
import json
import os
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, NoReturn, cast

from benchflow._daemon_wire import (
    connect,
    listen,
    peer_uid,
    private_dir,
    receive,
    receive_fds,
    recording_randomness,
    send,
    send_fds,
)

# Signals a terminal or a harness sends the CLI: an attached CLI passes them
# to its worker, which is not in the caller's process group.
FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def identity() -> str:
    """Do not attach a CLI from another installation or an edited source tree.

    BenchFlow's own files by content (sources and package data: templates,
    compose files, allowlists); every other import location by its entries,
    so an installed or upgraded distribution (its dist-info name carries the
    version) changes it too, and by the contents of what runs at interpreter
    start there (`*.pth`, sitecustomize, usercustomize).
    """
    digest = hashlib.sha256(os.fsencode(sys.executable))
    for path in sorted(Path(__file__).parent.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        digest.update(os.fsencode(path))
        digest.update(path.read_bytes())
    # Only the installation's own locations: sys.path[0] is the script's or
    # the current folder, which differs between a daemon and its callers
    # (PYTHONPATH entries are compared as start settings).
    prefixes = tuple(
        os.path.join(os.path.abspath(root), "")
        for root in {sys.prefix, sys.base_prefix, sys.exec_prefix}
    )
    # Python prepends one entry (the script's folder, or the current one)
    # unless safe-path mode is on.
    for entry in sys.path[0 if sys.flags.safe_path else 1 :]:
        if not os.path.join(os.path.abspath(entry or "."), "").startswith(prefixes):
            continue
        with contextlib.suppress(OSError):
            # Bytecode caches appear as anything first imports; not the install.
            names = sorted(
                name for name in os.listdir(entry or ".") if name != "__pycache__"
            )
            digest.update(os.fsencode(entry) + b"\0" + "\0".join(names).encode())
            for name in names:
                if name.endswith(".pth") or name in _START_HOOKS:
                    with contextlib.suppress(OSError):
                        digest.update(Path(entry or ".", name).read_bytes())
    return digest.hexdigest()


# Modules the interpreter imports at start when they are on an install path.
_START_HOOKS = ("sitecustomize.py", "usercustomize.py")


# Every signal whose disposition a process inherits across exec.
_INHERITED_SIGNALS = tuple(
    signum
    for signum in signal.valid_signals()
    if signum not in (signal.SIGKILL, signal.SIGSTOP)
)


def _signals() -> dict[str, list[int]]:
    """The caller's ignored signals and its blocked set: a cold CLI inherits
    both (nohup, background jobs in a script, a harness ignoring SIGUSR1) and
    a worker must have the same."""
    return {
        "ignored": sorted(
            int(signum)
            for signum in _INHERITED_SIGNALS
            if signal.getsignal(signum) == signal.SIG_IGN
        ),
        "mask": sorted(
            int(signum) for signum in signal.pthread_sigmask(signal.SIG_BLOCK, [])
        ),
    }


def _apply_signals(value: dict[str, Any]) -> None:
    ignored = set(value["signals"]["ignored"])
    for signum in _INHERITED_SIGNALS:
        if signum in ignored:
            signal.signal(signum, signal.SIG_IGN)
        elif signum == signal.SIGINT:
            # Python's own, unless SIGINT came in ignored (as above).
            signal.signal(signum, signal.default_int_handler)
        elif signum in FORWARDED_SIGNALS or signal.getsignal(signum) == signal.SIG_IGN:
            # The runtime's own disposition, not the caller's.
            signal.signal(signum, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, value["signals"]["mask"])


def _lsm_labels() -> dict[str, str | None]:
    """The Linux security module labels of this process (AppArmor, SELinux,
    Smack): `attr/current` and each `attr/<lsm>/current`. A module that is
    not active answers EINVAL and is left out; any other failure is None
    (unreadable: the call then runs cold). So is a listing of
    `attr` that fails, or an entry whose kind cannot be read: which labels
    exist is then unknown (review r4 M3)."""
    import errno as _errno

    labels: dict[str, str | None] = {}
    candidates = ["current"]
    try:
        entries = sorted(os.listdir("/proc/self/attr"))
    except OSError:
        entries = []
        labels["(listing)"] = None
    for name in entries:
        try:
            if stat.S_ISDIR(os.stat(f"/proc/self/attr/{name}").st_mode):
                candidates.append(f"{name}/current")
        except OSError:
            labels[f"{name}/current"] = None
    for name in candidates:
        try:
            labels[name] = Path(f"/proc/self/attr/{name}").read_text().strip("\0\n ")
        except OSError as exc:
            if exc.errno not in (_errno.EINVAL, _errno.ENOENT, _errno.EOPNOTSUPP):
                labels[name] = None
    return labels


def _readable(read: Any) -> Any:
    """*read*'s answer, or None when the kernel refuses it: unreadable is
    never assumed equal on both sides."""
    try:
        return read()
    except OSError:
        return None


def _process() -> dict[str, Any]:
    """Process attributes a fork keeps from the runtime: groups, limits, CPU
    affinity, cgroup and priority. A call whose own differ runs cold, and so
    does one where any of them cannot be read (None, on either side)."""
    import resource

    limits = {
        name: _readable(
            lambda name=name: list(resource.getrlimit(getattr(resource, name)))
        )
        for name in sorted(dir(resource))
        if name.startswith("RLIMIT_")
    }
    cgroup = _readable(Path("/proc/self/cgroup").read_text)
    # Confinement a cold CLI keeps and a fork of the runtime would not:
    # no_new_privs, seccomp (which Landlock needs, or CAP_SYS_ADMIN, both
    # visible here), capabilities, and every namespace.
    status: dict[str, str] = {}
    with contextlib.suppress(OSError):
        for line in Path("/proc/self/status").read_text().splitlines():
            name, _, rest = line.partition(":")
            status[name] = rest.strip()
    confinement = {
        name: status.get(name)
        for name in (
            "NoNewPrivs",
            "Seccomp",
            "Seccomp_filters",
            "CapInh",
            "CapPrm",
            "CapEff",
            "CapBnd",
            "CapAmb",
        )
    }
    namespaces: dict[str, str | None] = {}
    try:
        kinds = sorted(os.listdir("/proc/self/ns"))
    except OSError:
        kinds = []
    for name in kinds or ["(unreadable)"]:
        try:
            namespaces[name] = os.readlink(f"/proc/self/ns/{name}")
        except OSError:
            namespaces[name] = None
    return {
        "confinement": confinement,
        "namespaces": namespaces,
        "lsm": _lsm_labels(),
        "groups": _readable(lambda: sorted(os.getgroups())),
        # getgid() and getegid() cannot fail.
        "gid": os.getgid(),
        "egid": os.getegid(),
        "rlimits": limits,
        "affinity": _readable(lambda: sorted(os.sched_getaffinity(0))),
        "cgroup": cgroup,
        "priority": _readable(lambda: os.getpriority(os.PRIO_PROCESS, 0)),
    }


def _interpreter() -> str:
    """Interpreter settings fixed at start, which a fork keeps from its parent.

    UTF-8 mode counts through sys.flags: the runtime is started with the
    starting CLI's own `-X utf8` (see start()).
    """
    options = sorted(item for item in sys._xoptions.items() if item[0] != "utf8")
    return repr((tuple(sys.flags), sys.warnoptions, options))


def _start_settings(env: dict[str, str]) -> dict[str, str]:
    """Environment the interpreter reads at its own start, not through os.environ:
    import paths, I/O encoding, hash seed, locale and time zone."""
    return {
        key: value
        for key, value in env.items()
        if key.startswith(("PYTHON", "LC_")) or key in ("LANG", "TZ")
    }


def _stdio() -> list[list[Any]]:
    """How the CLI's interpreter opened its standard streams."""
    # A stand-in stream (a test runner's) lacks some; the TextIOWrapper defaults.
    return [
        [
            getattr(stream, "encoding", None) or "utf-8",
            getattr(stream, "errors", None) or "strict",
            bool(getattr(stream, "line_buffering", False)),
            bool(getattr(stream, "write_through", False)),
        ]
        for stream in (sys.stdin, sys.stdout, sys.stderr)
    ]


def socket_path() -> Path:
    configured = os.environ.get("BENCHFLOW_DAEMON_DIR")
    root = (
        Path(configured)
        if configured
        else Path.home() / ".cache" / "benchflow" / "daemon"
    )
    return root / "runtime.sock"


def request(value: dict[str, Any], timeout: float = 5) -> Any:
    with connect(socket_path()) as conn:
        conn.settimeout(timeout)
        send(conn, value)
        return receive(conn)


def _private_log(path: Path) -> Any:
    """A log file only this user can read (it may name task paths)."""
    return os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600),
        "ab",
    )


def attachable(argv: list[str]) -> bool:
    """Whether this CLI call may run in the local runtime.

    Only runs, and only without a terminal: job control, prompts and a
    terminal's colours belong to the caller's foreground process, so an
    interactive call keeps the cold path.
    """
    if os.environ.get("BENCHFLOW_DAEMON", "").lower() not in {"1", "on", "auto"}:
        return False
    if os.environ.get("_BENCHFLOW_DAEMON_WORKER") == "1":
        return False
    words = [word for word in argv if not word.startswith("-")]
    if not (words[:1] == ["run"] or words[:2] in (["eval", "run"], ["eval", "create"])):
        return False
    return not any(os.isatty(fd) for fd in (0, 1, 2))


def try_attach(argv: list[str]) -> int | None:
    """None means definitely not accepted; any accepted failure returns 1."""
    if not attachable(argv):
        return None
    accepted = False
    worker: int | None = None
    forwarded: list[int] = []
    previous: dict[int, Any] = {}
    try:
        with connect(socket_path()) as conn:
            send(
                conn,
                {
                    "action": "run",
                    "identity": identity(),
                    "interpreter": _interpreter(),
                    "stdio": _stdio(),
                    "env": dict(os.environ),
                    "cwd": os.getcwd(),
                    "argv0": sys.argv[0],
                    "argv": argv,
                    "umask": _umask(),
                    "signals": _signals(),
                    "process": _process(),
                },
            )
            # A prepared call checks itself first (its cold build included).
            conn.settimeout(600)
            if receive(conn) != {"ready": True}:
                return None
            conn.settimeout(5)
            sys.stdout.flush()
            sys.stderr.flush()
            send_fds(conn, [0, 1, 2])
            # From this point the daemon may already have launched the job.
            # An EOF is an interrupted call, not permission to run it twice.
            accepted = True
            conn.settimeout(None)
            worker = int(receive(conn)["pid"])

            def forward(signum: int, _frame: Any) -> None:
                forwarded.append(signum)
                with contextlib.suppress(ProcessLookupError):
                    os.kill(worker, signum)

            # An ignored signal stays ignored here, as in a cold CLI (the
            # worker ignores it too).
            for signum in FORWARDED_SIGNALS:
                if signal.getsignal(signum) != signal.SIG_IGN:
                    previous[signum] = signal.signal(signum, forward)
            response = receive(conn)
            return int(response["exit"])
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        if not accepted:
            return None
        ended = _worker_exit(worker) if worker is not None else None
        if ended is None and forwarded:
            ended = -forwarded[-1]
        if ended is not None and ended < 0:
            # The worker ended by a signal (one passed on, or an OOM kill):
            # end the same way, as the cold CLI would have.
            # SIGKILL and SIGSTOP have no handler to reset.
            with contextlib.suppress(OSError, ValueError):
                signal.signal(-ended, signal.SIG_DFL)
            os.kill(os.getpid(), -ended)
        if ended is not None:
            return ended
        print(
            "Local runtime disconnected after accepting the run; not replaying it. Inspect the job before resuming.",
            file=sys.stderr,
        )
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _worker_exit(pid: int) -> int | None:
    """How the runtime saw a worker end (negative: by that signal), or None."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError, EOFError, ValueError, KeyError):
            found = request({"action": "ended", "pid": pid})["exit"]
            if found is not None:
                return int(found)
        time.sleep(0.1)
    return None


def _umask() -> int:
    value = os.umask(0o077)
    os.umask(value)
    return value


def _preload() -> dict[str, str | None]:
    reads: dict[str, str | None] = {}
    original_env = dict(os.environ)
    environ_type = cast("Any", type(os.environ))
    original = environ_type.__getitem__
    original_set = environ_type.__setitem__
    written: list[str] = []

    def getitem(self: Any, key: str) -> str:
        reads[key] = original_env.get(key)
        return original(self, key)

    def setitem(self: Any, key: str, value: str) -> None:
        written.append(key)
        original_set(self, key, value)

    environ_type.__getitem__ = getitem
    environ_type.__setitem__ = setitem
    try:
        with recording_randomness() as drew:
            for module in (
                "benchflow.evaluation",
                "benchflow.rollout",
                "benchflow.sandbox.docker",
                "benchflow.eval_plan",
                "benchflow.cli.main",
            ):
                importlib.import_module(module)
    finally:
        environ_type.__getitem__ = original
        environ_type.__setitem__ = original_set
        os.environ.clear()
        os.environ.update(original_env)
    # A worker gets its caller's environment in place of this one, so a value
    # an import wrote here would be missing there (a cold CLI's import would
    # have written it): none may be written.
    if written:
        raise RuntimeError(
            "Runtime preload wrote the environment ("
            + ", ".join(sorted(set(written)))
            + "); refusing to fork."
        )
    # A value drawn at import would be the same in every worker.
    undeclared = sorted(drew - _REDRAWN_AT_FORK)
    if undeclared:
        raise RuntimeError(
            "Runtime preload drew randomness at import ("
            + ", ".join(undeclared)
            + "); refusing to fork."
        )
    if threading.active_count() != 1:
        raise RuntimeError("Runtime preload started threads; refusing to fork.")
    return reads


# Modules whose import-time random values are drawn again in every fork
# (os.register_at_fork): lockdown's pytest-guard marker secret.
_REDRAWN_AT_FORK = frozenset({"benchflow.sandbox.lockdown"})


def _admission_mismatch(
    value: dict[str, Any],
    signature: str,
    interpreter: str,
    start_env: dict[str, str],
    imports: dict[str, str | None],
    process: dict[str, Any] | None = None,
) -> str | None:
    """Why this call cannot run in a fork of this parent, or None."""
    if value["identity"] != signature:
        return "installation"
    if process is not None:
        theirs = value["process"]
        # No attribute that cannot be read, on either side, is assumed
        # equal: confinement, namespaces and LSM labels, and B-4's set
        # (review r4 L1: an unreadable cgroup on both sides attached).
        for key in sorted(set(process) | set(theirs)):
            for side in (process.get(key), theirs.get(key)):
                if isinstance(side, dict):
                    unreadable = sorted(
                        name for name, found in side.items() if found is None
                    )
                    if unreadable or not side:
                        return f"process attributes unreadable: {key} {', '.join(unreadable)}".rstrip()
                elif side is None:
                    return f"process attributes unreadable: {key}"
        differ = []
        for key in sorted(process):
            if theirs.get(key) == process[key]:
                continue
            if isinstance(process[key], dict) and isinstance(theirs.get(key), dict):
                names = sorted(
                    name
                    for name in set(process[key]) | set(theirs[key])
                    if process[key].get(name) != theirs[key].get(name)
                )
                differ.append(f"{key} ({', '.join(names)})")
            else:
                differ.append(key)
        if differ:
            return "process attributes: " + "; ".join(differ)
    if value["interpreter"] != interpreter:
        return "interpreter flags"
    if _start_settings(value["env"]) != start_env:
        return "interpreter start settings"
    for key, expected in imports.items():
        if value["env"].get(key) != expected:
            return "import-time setting " + key
    return None


def _reopen_stdio(config: list[list[Any]]) -> None:
    """Standard streams as the calling CLI's interpreter opened them."""
    streams = []
    for fd, (encoding, errors, line_buffering, write_through) in enumerate(config):
        binary = open(fd, "rb" if fd == 0 else "wb", closefd=False)  # noqa: SIM115
        streams.append(
            io.TextIOWrapper(
                binary,
                encoding=encoding,
                errors=errors,
                line_buffering=line_buffering,
                write_through=write_through,
            )
        )
    sys.stdin = sys.__stdin__ = streams[0]
    sys.stdout = sys.__stdout__ = streams[1]
    sys.stderr = sys.__stderr__ = streams[2]


def _watch_caller(conn: socket.socket) -> None:
    """End the worker when its caller is gone (killed, or its pipe closed):
    nothing would read its output or wait for its exit any more."""
    with contextlib.suppress(OSError):
        if not conn.recv(1):
            os.kill(os.getpid(), signal.SIGTERM)


def _enter_child(value: dict[str, Any]) -> None:
    """What every worker does first: the caller's environment, folder, umask
    and argv, and the runtime's state that must belong to this process."""
    os.environ.clear()
    os.environ.update(value["env"])
    os.environ["_BENCHFLOW_DAEMON_WORKER"] = "1"
    # This registry is empty in the pre-imported parent. Its exit cleanup
    # must belong to this new CLI process, not retain the parent's pid.
    from benchflow.providers import litellm_runtime

    litellm_runtime._HOST_PROXY_OWNER_PID = os.getpid()
    os.chdir(value["cwd"])
    os.umask(value["umask"])
    _apply_signals(value)
    sys.argv = [value["argv0"], *value["argv"]]


def _run_main() -> int:
    try:
        from benchflow.cli.main import main

        main()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def _finish(code: int, caller: socket.socket | None, note: socket.socket) -> NoReturn:
    # os._exit must not bypass the existing proxy-exit cleanup.
    atexit._run_exitfuncs()
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    from benchflow import _park

    with contextlib.suppress(OSError):
        send(note, {"exit": code, "parkable": _park.parkable()})
    if caller is not None:
        with contextlib.suppress(OSError):
            send(caller, {"exit": code})
        caller.close()
    os._exit(code)


def _worker(
    conn: socket.socket, note: socket.socket, fds: list[int], value: dict[str, Any]
) -> NoReturn:
    conn.settimeout(None)
    _enter_child(value)
    for target, fd in enumerate(fds):
        os.dup2(fd, target)
        os.close(fd)
    _reopen_stdio(value["stdio"])
    from benchflow import _park

    _park.begin_run()
    code = 1
    try:
        send(conn, {"pid": os.getpid()})
        threading.Thread(target=_watch_caller, args=(conn,), daemon=True).start()
        code = _run_main()
    finally:
        _finish(code, conn, note)


def _parked_worker(
    channel: socket.socket,
    note: socket.socket,
    spool: tuple[Path, Path],
    value: dict[str, Any],
) -> NoReturn:
    """Run the caller's command again, parked before its trial (benchflow._park)."""
    _enter_child(value)
    stdin = os.open(os.devnull, os.O_RDONLY)
    os.dup2(stdin, 0)
    os.close(stdin)
    for target, path in zip((1, 2), spool, strict=True):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        os.dup2(fd, target)
        os.close(fd)
    _reopen_stdio(value["stdio"])
    from benchflow import _park

    _park.begin_parked(channel, spool)
    code = 1
    try:
        code = _run_main()
    finally:
        # Never admitted: the call ended before its trial (it could not be
        # prepared), and its caller never existed.
        _finish(code, _park.caller(), note)


def call_key(value: dict[str, Any]) -> str:
    """What two calls must share for one to take the other's prepared run."""
    fields = (
        "identity",
        "interpreter",
        "stdio",
        "env",
        "cwd",
        "argv0",
        "argv",
        "umask",
        "signals",
    )
    return hashlib.sha256(
        json.dumps([value[field] for field in fields], sort_keys=True).encode()
    ).hexdigest()


def _log(message: str) -> None:
    """A line in the runtime's private daemon.log."""
    print(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        message,
        file=sys.stderr,
        flush=True,
    )


def _proc_start(pid: int) -> int | None:
    from benchflow._proxy_zygote import _start_time

    return _start_time(pid)


def _docker_env(env: dict[str, str]) -> dict[str, str]:
    """What names the caller's Docker daemon (its client settings)."""
    return {
        key: value
        for key, value in env.items()
        if key.startswith("DOCKER_") or key in ("PATH", "HOME")
    }


def reclaim(made: dict[str, str], docker_env: dict[str, str]) -> None:
    """Remove what a killed prepared call made, by the names it reported: its
    Compose project's containers and networks, and its job folder while that
    still carries the prepared marker and holds no result."""
    from benchflow._park import PREPARED_MARKER

    project = made.get("project")
    if project:
        label = f"label=com.docker.compose.project={project}"
        for kind in ("container", "network"):
            listed = subprocess.run(
                [
                    "docker",
                    kind,
                    "ls",
                    "-aq" if kind == "container" else "-q",
                    "--filter",
                    label,
                ],
                env=docker_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            names = listed.stdout.split() if listed.returncode == 0 else []
            if names:
                remove = [
                    "docker",
                    kind,
                    "rm",
                    *(["-f"] if kind == "container" else []),
                    *names,
                ]
                subprocess.run(remove, env=docker_env, capture_output=True, timeout=120)
                _log(f"removed a killed prepared call's {kind}s ({project})")
    job = made.get("job")
    if (
        job
        and (Path(job) / PREPARED_MARKER).exists()
        and not any(Path(job).rglob("result.json"))
    ):
        shutil.rmtree(job, ignore_errors=True)
        _log(f"removed a killed prepared call's job folder {job}")
        parent = made.get("jobs_dir_made")
        if parent:
            with contextlib.suppress(OSError):
                Path(parent).rmdir()


class _Child:
    """A worker: an attached call ("run") or a prepared one ("parked")."""

    def __init__(
        self, pid: int, kind: str, note: socket.socket, value: dict[str, Any]
    ) -> None:
        self.pid, self.kind, self.note, self.value = pid, kind, note, value
        self.key = call_key(value)
        self.channel: socket.socket | None = None
        self.spool: tuple[Path, Path] | None = None
        self.ready_at: float | None = None
        self.since = time.monotonic()
        self.result: dict[str, Any] | None = None
        self.discarded = False
        # What a prepared call reported it made (job folder, Compose project).
        self.made: dict[str, str] = {}
        # A caller waiting while this prepared call checks itself (the
        # rebuild takes seconds): its connection and request.
        self.pending: tuple[socket.socket, dict[str, Any]] | None = None
        # It said it removed what it made (else it was killed: reclaim).
        self.reported_removal = False

    def sockets(self) -> list[socket.socket]:
        return [sock for sock in (self.note, self.channel) if sock is not None]


class _Runtime:
    """The parent's bookkeeping: workers, prepared calls, limits."""

    def __init__(self, root: Path, prepare: int, ttl: float) -> None:
        self.root, self.prepare, self.ttl = root, prepare, ttl
        self.children: dict[int, _Child] = {}
        self.accepted_calls = 0
        self.prepared_calls = 0
        self.pending: _Child | None = None
        self.exits: dict[int, int] = {}
        self.helpers: list[subprocess.Popen[bytes]] = []
        # Callers whose prepared call ended while it checked itself.
        self.orphaned: list[tuple[socket.socket, dict[str, Any]]] = []
        self.state = root / "prepared.json"

    def parked(self) -> list[_Child]:
        return [child for child in self.children.values() if child.kind == "parked"]

    def running(self) -> list[_Child]:
        return [child for child in self.children.values() if child.kind == "run"]

    def inherited(self) -> list[socket.socket]:
        return [sock for child in self.children.values() for sock in child.sockets()]

    def discard(self, child: _Child) -> None:
        if child.channel is not None and not child.discarded:
            child.discarded = True
            with contextlib.suppress(OSError):
                send(child.channel, {"action": "discard"})

    def reap(self) -> list[_Child]:
        """Collect ended workers; returns the attached calls that ended."""
        ended = []
        for pid, child in list(self.children.items()):
            found, raw = os.waitpid(pid, os.WNOHANG)
            if not found:
                continue
            del self.children[pid]
            # How it ended, for a caller whose connection broke (an OOM kill
            # must read as one, 137 in a shell, as a cold CLI's would).
            self.exits[pid] = os.waitstatus_to_exitcode(raw)
            while len(self.exits) > 256:
                self.exits.pop(next(iter(self.exits)))
            if child.kind == "parked":
                # A prepared call removes what it made and says so; one that
                # was killed cannot, so the runtime does, by the names it
                # reported (never for one that cleaned up: no Docker calls
                # in this loop then).
                if not child.reported_removal and child.made:
                    self.reclaim_later(child.made, _docker_env(child.value["env"]))
                self.forget(child)
                if child.pending is not None:
                    self.orphaned.append(child.pending)
                    child.pending = None
            with contextlib.suppress(OSError, EOFError, ValueError):
                child.note.settimeout(0.5)
                child.result = receive(child.note)
            for sock in child.sockets():
                sock.close()
            if child.spool is not None:
                for spooled in child.spool:
                    spooled.unlink(missing_ok=True)
            if child.kind == "run" or child.result is not None:
                ended.append(child)
        return ended

    def reclaim_later(self, made: dict[str, str], docker_env: dict[str, str]) -> None:
        """Reclaim in a helper process: its Docker calls never hold up the
        runtime's loop (callers wait on it with a 5 s timeout)."""
        with _private_log(self.root / "daemon.log") as log:
            self.helpers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "benchflow._daemon",
                        "--reclaim",
                        json.dumps([made, docker_env]),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    cwd=self.root,
                )
            )

    def record(self, child: _Child) -> None:
        """Keep a prepared call's names on disk, for a runtime killed with it."""
        self._write(
            {
                **self._read(),
                str(child.pid): {
                    "start": _proc_start(child.pid),
                    "spool": [str(path) for path in child.spool or ()],
                    "made": child.made,
                    "docker_env": _docker_env(child.value["env"]),
                },
            }
        )

    def forget(self, child: _Child) -> None:
        state = self._read()
        if state.pop(str(child.pid), None) is not None:
            self._write(state)

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.state.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.state.with_suffix(".tmp")
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle)
        temporary.replace(self.state)

    def reclaim_dead(self) -> None:
        """At start: what prepared calls of a killed runtime left, by name."""
        state = self._read()
        for pid, entry in list(state.items()):
            if (
                _proc_start(int(pid)) == entry.get("start")
                and entry.get("start") is not None
            ):
                continue  # still alive: it removes its own when it sees we are gone
            for spooled in entry.get("spool", []):
                Path(spooled).unlink(missing_ok=True)
            reclaim(entry.get("made", {}), entry.get("docker_env", {}))
            state.pop(pid)
        self._write(state)

    def expire(self) -> None:
        now = time.monotonic()
        for child in self.parked():
            if child.pending is not None:
                continue
            idle = child.ready_at is not None and now - child.ready_at > self.ttl
            stuck = child.ready_at is None and now - child.since > max(self.ttl, 600)
            if idle or stuck:
                self.discard(child)

    def want(self, ended: _Child) -> None:
        """Prepare the call that just ended well again, if it can be.

        The newest call wins: when N are prepared already, the oldest is
        removed, and this one is prepared once its sandbox is gone (removals
        count against N until their workers end).
        """
        result = ended.result or {}
        if not (
            self.prepare > 0 and result.get("exit") == 0 and result.get("parkable")
        ):
            return
        live = [child for child in self.parked() if not child.discarded]
        if any(child.key == ended.key for child in live):
            return
        self.pending = ended
        live.sort(key=lambda child: child.since)
        while len(live) >= self.prepare:
            self.discard(live.pop(0))

    def next_to_prepare(self) -> _Child | None:
        if self.pending is None or len(self.parked()) >= self.prepare:
            return None
        pending, self.pending = self.pending, None
        return pending

    def candidate(self, value: dict[str, Any]) -> _Child | None:
        """A prepared call for exactly this call, not yet checked again."""
        key = call_key(value)
        for child in self.parked():
            if (
                child.key == key
                and child.ready_at is not None
                and child.channel is not None
                and not child.discarded
                and child.pending is None
            ):
                return child
        return None


class _Zygotes:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, tuple[subprocess.Popen[bytes], int, Path]] = {}

    def get(self, env: dict[str, str], *, wait: bool = True) -> Path | None:
        """The socket of the parent for this environment, started if need be;
        None (the proxy starts cold) when it cannot be had."""
        from benchflow._proxy_zygote import dotenv_off, parent_setting

        baseline = {key: value for key, value in env.items() if parent_setting(key)}
        # A parent reads nothing but its environment and installed files:
        # with python-dotenv on, LiteLLM's import would read a .env found
        # from the parent's own folders, which a cold proxy's are not.
        if not dotenv_off(baseline):
            return None
        encoded = json.dumps(baseline, sort_keys=True).encode()
        key = hashlib.sha256(encoded).hexdigest()[:20]
        entry = self.entries.get(key)
        if entry is None:
            # Bounded resident memory, without evicting a parent with live proxies.
            if len(self.entries) >= 2:
                return None
            entry = self._start(key, baseline)
        process, _, path = entry
        if not wait:
            return None
        # Its preload takes seconds: a first trial waits for it rather than
        # start cold beside it.
        started = time.monotonic()
        while process.poll() is None and time.monotonic() - started < 60:
            if path.exists():
                return path
            time.sleep(0.02)
        return None

    def _start(
        self, key: str, baseline: dict[str, str]
    ) -> tuple[subprocess.Popen[bytes], int, Path]:
        path = self.root / (key + ".sock")
        read_fd, write_fd = os.pipe()
        with _private_log(self.root / (key + ".log")) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchflow._proxy_zygote",
                    str(path),
                    str(read_fd),
                ],
                env=baseline,
                cwd=self.root,
                pass_fds=(read_fd,),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        os.close(read_fd)
        self.entries[key] = (process, write_fd, path)
        return self.entries[key]

    def close(self) -> None:
        for process, fd, _ in self.entries.values():
            os.close(fd)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(10)
            if process.poll() is None:
                process.terminate()
                process.wait(10)


def serve(prepare: int = 0, ttl: float = 300.0) -> None:
    path = socket_path()
    private_dir(path.parent)
    lock_fd = os.open(
        path.parent / "runtime.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    # A runtime still shutting down (its socket already gone) holds this
    # until its prepared calls and proxy parents are gone: wait for it.
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    if path.exists():
        try:
            with connect(path):
                raise RuntimeError("A local runtime is already listening.")
        except OSError as exc:
            if exc.errno != errno.ECONNREFUSED:
                raise
            # The lock excludes another managed daemon's start. connect()
            # already checked the stale object's type, uid and permissions.
            path.unlink()
    # Like the bench script, never the folder the runtime was started from:
    # a module there would shadow one every attached call imports.
    if not sys.flags.safe_path and sys.path:
        sys.path[0] = str(Path(sys.executable).parent)
    listener = listen(path)
    signature = identity()
    interpreter = _interpreter()
    start_env = _start_settings(dict(os.environ))
    imports = _preload()
    process = _process()
    zygotes = _Zygotes(path.parent)
    # The proxy parent for this runtime's own environment, which a caller
    # started from the same shell shares, preloads while nothing waits.
    from benchflow.providers.litellm_runtime import _proxy_process_env

    zygotes.get(_proxy_process_env(dict(os.environ), str(path.parent)), wait=False)
    runtime = _Runtime(path.parent, prepare, ttl)
    runtime.reclaim_dead()
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ)
    # No new calls once set; the runtime ends when its running calls (whose
    # host proxies its proxy parents serve) and prepared calls are gone.
    stopping = False
    listening = True

    def stop(_signum: int, _frame: Any) -> None:
        # A signal (a service manager's stop) closes like `bench daemon stop`
        # but never interrupts a running call: it waits for them.
        nonlocal stopping
        stopping = True

    def close_listener() -> None:
        nonlocal listening
        if listening:
            listening = False
            # Seen by `bench daemon status` and `start` until this ends.
            _mark_stopping(path.parent, len(runtime.running()))
            selector.unregister(listener)
            listener.close()
            path.unlink(missing_ok=True)
            # Prepared calls remove what they made (one checking itself for a
            # waiting caller finishes that first).
            for child in runtime.parked():
                if child.pending is None:
                    runtime.discard(child)

    def fork() -> tuple[int, socket.socket]:
        """Fork a worker; in the child, close everything only the parent holds."""
        note, child_note = socket.socketpair()
        # Nothing buffered here may reach the worker's output.
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            note.close()
            listener.close()
            for sock in runtime.inherited():
                sock.close()
            # Workers must not keep the zygotes' lifetime pipes alive.
            for _, fd, _ in zygotes.entries.values():
                os.close(fd)
            os.close(lock_fd)
            selector.close()
            return 0, child_note
        child_note.close()
        return pid, note

    def prepare_next(ended: _Child) -> None:
        channel, child_channel = socket.socketpair()
        token = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        spool = (path.parent / f"park-{token}.out", path.parent / f"park-{token}.err")
        pid, note = fork()
        if pid == 0:
            channel.close()
            _parked_worker(child_channel, note, spool, ended.value)
        child_channel.close()
        child = _Child(pid, "parked", note, ended.value)
        child.channel, child.spool = channel, spool
        runtime.children[pid] = child
        runtime.record(child)
        selector.register(channel, selectors.EVENT_READ, child)

    def answer(conn: socket.socket) -> None:
        nonlocal stopping
        fds: list[int] = []
        held = False
        try:
            conn.settimeout(5)
            if peer_uid(conn) != os.getuid():
                return
            value = receive(conn)
            action = value["action"]
            if action == "status":
                send(
                    conn,
                    {
                        "pid": os.getpid(),
                        "workers": len(runtime.running()),
                        "accepted_calls": runtime.accepted_calls,
                        "prepared_calls": runtime.prepared_calls,
                        "parked": sum(c.ready_at is not None for c in runtime.parked()),
                        "preparing": sum(c.ready_at is None for c in runtime.parked()),
                        "prepare": runtime.prepare,
                        "prepare_ttl": runtime.ttl,
                        "identity": signature,
                        "interpreter": interpreter,
                        "start_settings": start_env,
                        "parked_pids": [c.pid for c in runtime.parked()],
                    },
                )
            elif action == "stop":
                # Refuse to kill a live job through a management call;
                # prepared calls remove their sandboxes and folders.
                if not runtime.running():
                    stopping = True
                send(conn, {"stopping": stopping, "workers": len(runtime.running())})
            elif action == "ended":
                # How a worker ended, for its caller (see _Runtime.reap).
                send(conn, {"exit": runtime.exits.get(int(value["pid"]))})
            elif action == "zygote":
                result = zygotes.get(value["env"])
                send(conn, {"path": str(result) if result else None})
            elif action == "run":
                mismatch = _admission_mismatch(
                    value, signature, interpreter, start_env, imports, process
                )
                if mismatch or threading.active_count() != 1:
                    # The call runs cold; the reason is kept in daemon.log.
                    _log(f"call refused (it runs cold): {mismatch}")
                    send(conn, {"ready": False, "reason": mismatch})
                    return
                prepared = runtime.candidate(value)
                if prepared is not None and prepared.channel is not None:
                    # It checks itself (inputs, images, the cold build) in its
                    # own process; the reply completes this call, below.
                    send(prepared.channel, {"action": "validate"})
                    prepared.pending = (conn, value)
                    held = True
                    return
                admit(conn, value, None)
                held = True  # admit() closed it
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            pass
        finally:
            for fd in fds:
                os.close(fd)
            if not held:
                conn.close()

    def admit(
        conn: socket.socket, value: dict[str, Any], prepared: _Child | None
    ) -> None:
        """Accept a call: hand it to a prepared call that checked itself, or
        fork a fresh worker. Closes the caller's connection here."""
        fds: list[int] = []
        try:
            send(conn, {"ready": True})
            fds = receive_fds(conn, 3)
            if prepared is not None and prepared.channel is not None:
                send(prepared.channel, {"action": "admit"})
                send_fds(prepared.channel, [*fds, conn.fileno()])
                selector.unregister(prepared.channel)
                prepared.channel.close()
                prepared.channel = None
                prepared.kind = "run"
                # Taken: no longer the runtime's to reclaim, as a cold call's
                # sandbox is not (L-A1).
                runtime.forget(prepared)
                runtime.accepted_calls += 1
                runtime.prepared_calls += 1
                return
            pid, note = fork()
            if pid == 0:
                _worker(conn, note, fds, value)
            runtime.children[pid] = _Child(pid, "run", note, value)
            runtime.accepted_calls += 1
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            pass
        finally:
            for fd in fds:
                os.close(fd)
            conn.close()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        parked_deadline = None
        draining = -1
        while True:
            for ended in runtime.reap():
                if not stopping:
                    runtime.want(ended)
            while runtime.orphaned:
                orphan_conn, orphan_value = runtime.orphaned.pop()
                admit(orphan_conn, orphan_value, None)
            runtime.helpers = [
                helper for helper in runtime.helpers if helper.poll() is None
            ]
            if stopping:
                close_listener()
                if len(runtime.running()) != draining:
                    draining = len(runtime.running())
                    _mark_stopping(path.parent, draining)
                parked_deadline = parked_deadline or time.monotonic() + 600
                if not runtime.running() and (
                    not runtime.parked() or time.monotonic() > parked_deadline
                ):
                    break
            elif (pending := runtime.next_to_prepare()) is not None:
                prepare_next(pending)
            runtime.expire()
            for key, _ in selector.select(0.1):
                if key.fileobj is listener:
                    answer(listener.accept()[0])
                    continue
                # A prepared call's notes: what it made, that its sandbox is
                # ready, or why it removed itself.
                child = key.data
                try:
                    note = receive(child.channel)
                except (OSError, EOFError, ValueError):
                    selector.unregister(key.fileobj)
                    if child.pending is not None:
                        # It ended while checking: the caller runs fresh.
                        waiting, child.pending = child.pending, None
                        child.discarded = True
                        admit(waiting[0], waiting[1], None)
                    continue
                if note.get("parked") is True:
                    child.ready_at = time.monotonic()
                if "valid" in note and child.pending is not None:
                    waiting, child.pending = child.pending, None
                    timing = f"checks {note.get('checks_s')} s, rebuild {note.get('rebuild_s')} s"
                    if note["valid"] is True:
                        _log(f"prepared call taken ({timing})")
                        admit(waiting[0], waiting[1], child)
                    else:
                        _log(
                            f"prepared call not taken (it is removed): {note.get('reason')} ({timing})"
                        )
                        child.ready_at = None
                        child.discarded = True
                        admit(waiting[0], waiting[1], None)
                    continue
                if "discarded" in note:
                    child.discarded = True
                    child.reported_removal = True
                    _log(f"prepared call removed: {note['discarded']}")
                for name in ("job", "jobs_dir_made", "project"):
                    if note.get(name):
                        child.made[name] = note[name]
                        runtime.record(child)
    finally:
        # New calls start cold from here on (also after an error here).
        close_listener()
        # Proxy parents keep serving the proxies of calls still alive after
        # this (benchflow._proxy_zygote: lifetime end), so closing them is safe.
        zygotes.close()
        selector.close()
        (path.parent / "runtime.stopping").unlink(missing_ok=True)
        os.close(lock_fd)


def _mark_stopping(root: Path, running: int) -> None:
    temporary = root / "runtime.stopping.tmp"
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "w") as handle:
        json.dump(
            {"pid": os.getpid(), "start": _proc_start(os.getpid()), "running": running},
            handle,
        )
    temporary.replace(root / "runtime.stopping")


def stopping_runtime() -> dict[str, Any] | None:
    """This folder's runtime while it stops (it waits for its running calls:
    `running` of them are left), or None."""
    try:
        found = json.loads((socket_path().parent / "runtime.stopping").read_text())
    except (OSError, ValueError):
        return None
    pid = found.get("pid")
    if isinstance(pid, int) and _proc_start(pid) == found.get("start") is not None:
        return found
    return None


class RuntimeMismatch(RuntimeError):
    """A runtime is running here that this start would not have started."""


def _running_differs(state: dict[str, Any], prepare: int, ttl: float) -> str | None:
    if state.get("identity") != identity():
        return "it belongs to another installation or source tree"
    if state.get("interpreter") != _interpreter() or state.get(
        "start_settings"
    ) != _start_settings(dict(os.environ)):
        return "it was started with other interpreter settings"
    if (state.get("prepare"), state.get("prepare_ttl")) != (prepare, float(ttl)):
        return (
            f"it runs with --prepare {state.get('prepare')} "
            f"--prepare-ttl {state.get('prepare_ttl')}"
        )
    return None


def start(prepare: int = 0, ttl: float = 300.0) -> None:
    path = socket_path()
    private_dir(path.parent)
    try:
        state = request({"action": "status"})
    except (OSError, EOFError):
        state = None
    if state is None and stopping_runtime() is not None:
        # One stopping (`bench daemon stop` just now, or a signal) waits for
        # its running calls: start after it, or say so.
        deadline = time.monotonic() + 120
        while stopping_runtime() is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        if stopping_runtime() is not None:
            raise RuntimeMismatch(
                f"A local runtime in {path.parent} is stopping; it waits for its running calls."
            )
    if state is not None:
        differs = _running_differs(state, prepare, ttl)
        if differs:
            raise RuntimeMismatch(
                f"A local runtime is already running in {path.parent}, but {differs}."
            )
        return
    with _private_log(path.parent / "daemon.log") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                # In a C locale an interpreter runs in UTF-8 mode and exports
                # LC_CTYPE=C.UTF-8, which would start this child without it.
                "-X",
                f"utf8={sys.flags.utf8_mode}",
                "-m",
                "benchflow._daemon",
                "--prepare",
                str(prepare),
                "--prepare-ttl",
                str(ttl),
            ],
            # Its own folder, never the caller's: see serve() on sys.path.
            cwd=path.parent,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    started = time.monotonic()
    while process.poll() is None and time.monotonic() - started < 60:
        try:
            request({"action": "status"})
            return
        except (OSError, EOFError):
            time.sleep(0.05)
    raise RuntimeError("Local runtime failed to start; see its private daemon.log.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="benchflow._daemon")
    parser.add_argument("--prepare", type=int, default=0)
    parser.add_argument("--prepare-ttl", type=float, default=300.0)
    parser.add_argument("--reclaim", default=None)
    options = parser.parse_args()
    if options.reclaim is not None:
        made, docker_env = json.loads(options.reclaim)
        reclaim(made, docker_env)
    else:
        serve(options.prepare, options.prepare_ttl)
