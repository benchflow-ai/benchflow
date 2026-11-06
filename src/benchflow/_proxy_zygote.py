"""Single-threaded LiteLLM parent: only configuration-free imports precede fork.

The runtime daemon owns one parent per exact non-trial environment. A proxy
child gets its own runtime files and the existing sitecustomize/config path.
No callback, guard, provider patch or trial configuration is loaded here.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import json
import os
import secrets
import selectors
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchflow._daemon_wire import (
    connect,
    listen,
    peer_uid,
    receive,
    recording_randomness,
    send,
)

# These vary for every trial; none may influence the pre-imported modules.
TRIAL_ENV = frozenset(
    {
        "PYTHONPATH",
        "LITELLM_MASTER_KEY",
        "BENCHFLOW_LITELLM_LOG_PATH",
        "BENCHFLOW_LITELLM_ROUTE_PROVIDER",
        "BENCHFLOW_LITELLM_UPSTREAM_MODEL",
        "BENCHFLOW_LITELLM_SERVED_MODELS",
        "BENCHFLOW_LITELLM_CHAT_BRIDGE_MODELS",
        "BENCHFLOW_LITELLM_VERTEX_NATIVE",
        "BENCHFLOW_SKILL_CATALOG_GATE_AGENT",
        "BENCHFLOW_REQUIRED_SKILL_NAMES_JSON",
    }
)


# What a proxy start imports whatever its route and config: the server and
# CLI, and what their startup imports lazily (the router's redis client,
# OpenTelemetry, uvicorn's loop, protocol and lifespan modules). Measured on
# a forked child: 178 modules came after the fork without these. Provider
# modules (litellm.llms.*) depend on the route and stay with the child.
PRELOAD = (
    "litellm.proxy.proxy_server",
    "litellm.proxy.proxy_cli",
    "uvicorn",
)
# Imported when installed (an extra or an optional speedup may be absent;
# whatever is missing the child imports as a cold proxy would).
PRELOAD_IF_INSTALLED = (
    "litellm.router",
    "litellm._redis",
    "redis.asyncio",
    "redis.cluster",
    "redis.sentinel",
    "opentelemetry.trace",
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "ctypes",
    "uvloop",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "websockets.legacy.server",
)


def _trial_setting(key: str) -> bool:
    """Names left out of a parent's environment: they vary per trial."""
    return key in TRIAL_ENV or key.startswith(("BENCHFLOW_", "_BENCHFLOW_"))


# Set by LiteLLM's own import over any value it is given (TIKTOKEN_CACHE_DIR:
# its bundled tokenizers, or CUSTOM_TIKTOKEN_CACHE_DIR, which it reads). A
# CLI process has it once its own LiteLLM import ran and lacks it before, so
# it keys no parent; a child replays the import's write (IMPORT_WRITES).
IMPORT_SET = frozenset({"TIKTOKEN_CACHE_DIR"})


def parent_setting(key: str) -> bool:
    """Names a proxy parent's environment (and its key) is made of."""
    return not _trial_setting(key) and key not in IMPORT_SET


@dataclass(frozen=True)
class Scan:
    """What one whole-environment scan's result depends on.

    ``inputs`` names them: an exact name, or ``*`` and a suffix, matched in
    any case (pydantic-settings and urllib both fold case). ``cache`` is the
    ``module:function`` whose cached result a proxy child clears so that it
    computes it again from its own environment, as a cold start does.
    """

    inputs: str
    cache: str | None = None

    def matches(self, key: str) -> bool:
        if not self.inputs:
            return False
        if self.inputs.startswith("*"):
            return key.upper().endswith(self.inputs[1:])
        return key.upper() == self.inputs


# Whole-environment scans the preload is known to make, by the function that
# makes them. A scan not listed here refuses the parent: a per-name audit
# cannot see what a scan's result depends on, and the parent's environment
# lacks every trial setting, which a cold proxy's scan would see.
KNOWN_SCANS: dict[tuple[str, str], Scan] = {
    # LiteLLM's OTel v2 switch: a pydantic-settings model with one field,
    # cached for the process.
    ("litellm.integrations.otel.model.config", "is_otel_v2_enabled"): Scan(
        "LITELLM_OTEL_V2", "litellm.integrations.otel.model.config:is_otel_v2_enabled"
    ),
    # The standard library's proxy settings (http_proxy, NO_PROXY and the
    # rest); REQUEST_METHOD, which it also reads, is an ordinary read.
    ("urllib.request", "getproxies_environment"): Scan("*_PROXY"),
}
# Frames between a scan's maker and os.environ: the mapping machinery and the
# settings library, never where a scan's meaning lives.
_SCAN_PLUMBING = ("os", "_collections_abc", "collections", "typing", "pydantic")


def _scan_origin() -> tuple[str, str]:
    frame = sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.split(".", 1)[0] not in _SCAN_PLUMBING and not module.startswith(
            "pydantic_settings"
        ):
            return module, frame.f_code.co_name
        frame = frame.f_back
    return "", ""


def _scanned(env: Mapping[str, str], scan: Scan) -> list[list[str]]:
    """The names and values *scan* reads in *env*."""
    return sorted([key, value] for key, value in env.items() if scan.matches(key))


# BenchFlow's proxy reads no .env file, cold or attached: every value it
# needs is passed in its environment, and that environment turns
# python-dotenv off. LiteLLM's import-time
# `load_dotenv()` calls (litellm/__init__.py, litellm/proxy/proxy_cli.py)
# then return before they search for a file, so the working folder and the
# folders above LiteLLM's files play no part in what a proxy imports.
DOTENV_OFF = {"PYTHON_DOTENV_DISABLED": "1"}


def dotenv_off(env: Mapping[str, str]) -> bool:
    """Whether *env* turns python-dotenv off, as BenchFlow's proxy env does."""
    return all(env.get(key) == value for key, value in DOTENV_OFF.items())


@contextlib.contextmanager
def _refusing_dotenv() -> Any:
    """Yields a list that gets an entry for each `.env` python-dotenv parses
    while the preload imports. With python-dotenv off there is none; one
    (an import calling `dotenv_values()`, which the switch does not cover)
    refuses the parent, and proxies start cold."""
    import dotenv.main as dotenv_main

    parsed: list[str] = []
    dotenv_class = cast("Any", dotenv_main.DotEnv)
    original = dotenv_class.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        frame = sys._getframe(1)
        while frame.f_back is not None and frame.f_globals.get(
            "__name__", ""
        ).startswith("dotenv"):
            frame = frame.f_back
        parsed.append(frame.f_globals.get("__name__", ""))
        original(self, *args, **kwargs)

    dotenv_class.__init__ = init
    try:
        yield parsed
    finally:
        dotenv_class.__init__ = original


def preload() -> tuple[dict[str, str | None], dict[str, list[list[str]]]]:
    """Audit every environment lookup and scan, including transitive imports.

    Recording absence matters: an unset credential at import is still a read.
    This is an admission check, not a guessed list of LiteLLM's global settings.
    Returns each name read with the value the imports saw (None: unset), and
    each known scan's inputs, as the imports saw them.
    """
    if not dotenv_off(os.environ):
        raise RuntimeError(
            "python-dotenv is not off in the proxy parent's environment."
        )
    reads: dict[str, str | None] = {}
    scans: set[tuple[str, str]] = set()
    scanning = 0
    environ_type = cast("Any", type(os.environ))
    original_get = environ_type.__getitem__
    original_iter = environ_type.__iter__
    original_set = environ_type.__setitem__
    original_del = environ_type.__delitem__
    written: dict[str, str | None] = {}

    def getitem(self: Any, key: str) -> str:
        if key in written:  # the imports' own value, made again in a child
            return original_get(self, key)
        try:
            value = original_get(self, key)
        except KeyError:
            reads.setdefault(key, None)
            raise
        # A scan's reads (during its iteration, or after it from the same
        # function, as `dict.update(os.environ)` does) count through
        # KNOWN_SCANS only.
        if not scanning and not (scans and _scan_origin() in scans):
            reads.setdefault(key, value)
        return value

    def setitem(self: Any, key: str, value: str) -> None:
        written[key] = value
        original_set(self, key, value)

    def delitem(self: Any, key: str) -> None:
        written[key] = None
        original_del(self, key)

    def iterate(self: Any) -> Any:
        nonlocal scanning
        scans.add(_scan_origin())
        scanning += 1
        try:
            yield from original_iter(self)
        finally:
            scanning -= 1

    environ_type.__getitem__ = getitem
    environ_type.__iter__ = iterate
    environ_type.__setitem__ = setitem
    environ_type.__delitem__ = delitem
    # LiteLLM's proxy modules add this process's working folder and its
    # "../.." to the import path; a child's comes from its own runtime
    # folder, as a cold proxy's does (its sitecustomize puts it back too).
    import_path = list(sys.path)
    try:
        with recording_randomness() as drew, _refusing_dotenv() as parsed:
            for module in PRELOAD:
                importlib.import_module(module)
            for module in PRELOAD_IF_INSTALLED:
                with contextlib.suppress(ImportError):
                    importlib.import_module(module)
    finally:
        sys.path[:] = import_path
        environ_type.__getitem__ = original_get
        environ_type.__iter__ = original_iter
        environ_type.__setitem__ = original_set
        environ_type.__delitem__ = original_del
    if parsed:
        raise RuntimeError(
            f"{parsed[0]} parsed a .env while python-dotenv was off; "
            "proxies start cold."
        )
    undeclared = sorted(drew - IMPORT_RANDOMNESS.keys())
    if undeclared:
        raise RuntimeError(
            "Unaudited import-time randomness in proxy preload: "
            + ", ".join(undeclared)
        )
    unknown = sorted(".".join(origin) for origin in scans if origin not in KNOWN_SCANS)
    if unknown:
        raise RuntimeError(
            "Unaudited environment scan during proxy preload: " + ", ".join(unknown)
        )
    scanned = {
        KNOWN_SCANS[origin].inputs: _scanned(os.environ, KNOWN_SCANS[origin])
        for origin in scans
        if KNOWN_SCANS[origin].inputs
    }
    trial = {key for key in [*reads, *written] if _trial_setting(key)}
    trial |= {
        key for seen in scanned.values() for key, _ in seen if _trial_setting(key)
    }
    trial |= {
        name
        for name in TRIAL_ENV
        if any(KNOWN_SCANS[origin].matches(name) for origin in scans)
    }
    if trial:
        raise RuntimeError(
            "Trial environment read during proxy preload: " + ", ".join(sorted(trial))
        )
    if threading.active_count() != 1:
        raise RuntimeError("Proxy preload started threads; refusing to fork.")
    IMPORT_WRITES.clear()
    IMPORT_WRITES.update(written)
    return reads, scanned


def _redraw_guardrail_token() -> None:
    guardrail = sys.modules.get("litellm.integrations.custom_guardrail")
    if guardrail is not None:
        cast("Any", guardrail)._PRE_CALL_EXECUTED_TOKEN = secrets.token_hex(16)


def _redraw_multiprocessing_authkey() -> None:
    process = sys.modules.get("multiprocessing.process")
    if process is not None:
        cast("Any", process).current_process().authkey = os.urandom(32)


# Modules the preload saw drawing OS randomness at import, and what a child
# does about each: a secret is drawn again (a cold proxy draws its own), as
# lockdown's marker secret is for BenchFlow's workers. A module drawing at
# import that is not listed refuses the parent (trials start their proxy cold).
IMPORT_RANDOMNESS: dict[str, Any] = {
    # LiteLLM's per-process marker against forged guardrail pre-call flags.
    "litellm.integrations.custom_guardrail": _redraw_guardrail_token,
    # The process's multiprocessing authkey.
    "multiprocessing.process": _redraw_multiprocessing_authkey,
    # Context key names (uuid4 suffixes): identifiers that must stay as the
    # keys already made, not secrets.
    "opentelemetry.context": None,
}


def _redraw_import_randomness() -> None:
    for redraw in IMPORT_RANDOMNESS.values():
        if redraw is not None:
            redraw()


# What the preloaded imports wrote to the environment (None: removed). A
# cold proxy's own import writes the same, from the same inputs (its reads,
# checked at every launch), over the environment it was given: LiteLLM sets
# TIKTOKEN_CACHE_DIR to its bundled tokenizers this way. A child replays it.
IMPORT_WRITES: dict[str, str | None] = {}


def _replay_import_writes() -> None:
    for key, value in IMPORT_WRITES.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _import_time_mismatch(
    reads: dict[str, str | None],
    env: dict[str, str],
    scanned: dict[str, list[list[str]]] | None = None,
) -> list[str]:
    """Names whose value at import differs from this launch's environment.

    The daemon keys parents by every non-trial value, so this is empty by
    construction; checked again here because a fork must never run with a
    module state a cold start of this exact environment would not have.
    """
    changed = {key for key, value in reads.items() if env.get(key) != value}
    for inputs, seen in (scanned or {}).items():
        if _scanned(env, Scan(inputs)) != seen:
            changed.add(inputs)
    return sorted(changed)


def _reset_scans() -> None:
    """Clear each known scan's cached result in a proxy child (see KNOWN_SCANS)."""
    for scan in KNOWN_SCANS.values():
        if scan.cache is None:
            continue
        module, name = scan.cache.split(":")
        cached = getattr(sys.modules.get(module), name, None)
        if cached is not None:
            cached.cache_clear()


def _child(
    request: dict[str, Any], listener: socket.socket, conn: socket.socket
) -> None:
    listener.close()
    conn.close()
    os.environ.clear()
    os.environ.update(request["env"])
    _replay_import_writes()
    _reset_scans()
    _redraw_import_randomness()
    runtime = Path(request["cwd"])
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    for target, name in ((1, "stdout.log"), (2, "stderr.log")):
        fd = os.open(runtime / name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, target)
        os.close(fd)
    fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd, 0)
    os.close(fd)
    try:
        # The exact cold patches run only now, after the trial env is set.
        # A sitecustomize the parent's interpreter start imported (one on
        # its own path) must not stand in for the trial's: a cold start
        # finds the trial's first, on PYTHONPATH.
        sys.modules.pop("sitecustomize", None)
        importlib.import_module("sitecustomize")
        from litellm.proxy.proxy_cli import run_server

        run_server.main(args=request["args"], standalone_mode=True)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    # As a cold proxy's interpreter exit does: LiteLLM's exit handler flushes
    # queued success and failure callbacks (BenchFlow's usage records among
    # them) after a trial's log wait gave up.
    with contextlib.suppress(BaseException):
        atexit._run_exitfuncs()
    with contextlib.suppress(BaseException):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def serve(path: Path, lifetime: int) -> None:
    reads, scanned = preload()
    # An audit contains names only, never credential values.
    audit = os.open(
        path.with_suffix(".audit.json"),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(audit, "w") as handle:
        json.dump({"reads": sorted(reads), "scans": sorted(scanned)}, handle)
    listener = listen(path)
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ)
    selector.register(lifetime, selectors.EVENT_READ)
    children: dict[int, int | None] = {}
    draining = False
    # Each proxy's owner (the CLI process that launched it): its pid and its
    # start time, so a reused pid is never taken for it.
    owners: dict[int, tuple[int, int]] = {}
    try:
        while True:
            for pid, status in list(children.items()):
                if status is None:
                    found, raw = os.waitpid(pid, os.WNOHANG)
                    if found:
                        children[pid] = os.waitstatus_to_exitcode(raw)
            # An owner that exited takes its proxies with it, as a cold
            # proxy's exit clean-up would stop them.
            for pid, owner in list(owners.items()):
                if _start_time(owner[0]) != owner[1]:
                    if children[pid] is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGTERM)
                    del owners[pid]
            if draining and not any(children[pid] is None for pid in owners):
                return
            for key, _ in selector.select(0.02):
                if key.fileobj == lifetime:
                    # The runtime is gone (stopped, killed or crashed). Its
                    # calls may still be running: their proxies keep serving
                    # until each owner is gone; nothing new is launched.
                    selector.unregister(lifetime)
                    os.close(lifetime)
                    lifetime = -1
                    draining = True
                    continue
                conn, _ = listener.accept()
                # One bad request never takes down other trials' proxies.
                with conn, contextlib.suppress(Exception):
                    _answer(
                        conn,
                        (reads, scanned),
                        children,
                        owners,
                        listener,
                        selector,
                        lifetime,
                        draining,
                    )
    finally:
        for pid, status in children.items():
            if status is None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGTERM)
        listener.close()
        selector.close()
        path.unlink(missing_ok=True)


def _start_time(pid: int) -> int | None:
    """A process's start time (clock ticks after boot), or None once it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat_line = handle.read()
    except OSError:
        return None
    # The name (field 2) may hold spaces and parentheses; start time is field 22.
    return int(stat_line[stat_line.rindex(b")") + 2 :].split()[19])


def _launch_refusal(
    audit: tuple[dict[str, str | None], dict[str, list[list[str]]]],
    env: dict[str, str],
    draining: bool = False,
) -> str | None:
    """Why a proxy for *env* cannot be forked from this parent (it then starts
    cold), or None."""
    mismatch = _import_time_mismatch(audit[0], env, audit[1])
    if mismatch:
        return "import-time settings differ: " + ", ".join(mismatch)
    if draining:
        return "the local runtime has stopped"
    if not dotenv_off(env):
        return "python-dotenv is not off in this proxy's environment"
    if threading.active_count() != 1:
        return "proxy parent is not single-threaded"
    return None


def _answer(
    conn: socket.socket,
    audit: tuple[dict[str, str | None], dict[str, list[list[str]]]],
    children: dict[int, int | None],
    owners: dict[int, tuple[int, int]],
    listener: socket.socket,
    selector: selectors.BaseSelector,
    lifetime: int,
    draining: bool = False,
) -> None:
    conn.settimeout(5)
    if peer_uid(conn) != os.getuid():
        return
    caller = struct.unpack(
        "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )[0]
    request = receive(conn)
    action = request["action"]
    if action == "launch":
        # A refusal comes before any fork: the caller then starts the
        # proxy cold, as it would without the local runtime.
        refusal = _launch_refusal(audit, request["env"], draining)
        if refusal:
            send(conn, {"refused": refusal})
            return
        started = _start_time(caller)
        if started is None:
            send(conn, {"refused": "the launching process cannot be watched"})
            return
        pid = os.fork()
        if pid == 0:
            if lifetime >= 0:
                os.close(lifetime)
            selector.close()
            _child(request, listener, conn)
        children[pid] = None
        owners[pid] = (caller, started)
        send(conn, {"pid": pid})
        return
    # Only the process that launched a proxy may watch or stop it: one
    # trial's CLI process never reaches another trial's proxy.
    pid = request["pid"]
    if owners.get(pid, (None, None))[0] != caller:
        send(conn, {"status": 1} if action == "poll" else {})
        return
    if action == "poll":
        send(conn, {"status": children[pid]})
    elif action == "signal":
        if children[pid] is None:
            os.kill(pid, signal.SIGKILL if request["kill"] else signal.SIGTERM)
        send(conn, {})


class ProxyProcess:
    """The Popen subset used by HostLiteLLMProcess, with parent-owned reaping."""

    def __init__(self, path: Path, pid: int) -> None:
        self.path, self.pid = path, pid
        self.returncode: int | None = None
        # Known while the proxy runs: if its parent is killed, this process
        # watches and stops it itself (a reused pid is not it).
        self.started = _start_time(pid)

    def _call(self, action: str, **fields: Any) -> dict[str, Any]:
        with connect(self.path) as conn:
            send(conn, {"action": action, "pid": self.pid, **fields})
            return receive(conn)

    def _alive(self) -> bool:
        return self.started is not None and _start_time(self.pid) == self.started

    def poll(self) -> int | None:
        try:
            self.returncode = self._call("poll")["status"]
        except (OSError, EOFError):
            # The parent is gone: the proxy (now init's child) may still run.
            self.returncode = None if self._alive() else 1
        return self.returncode

    def _signal(self, kill: bool) -> None:
        try:
            self._call("signal", kill=kill)
        except (OSError, EOFError):
            # As Popen's: a proxy already gone is no error.
            if self._alive():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(self.pid, signal.SIGKILL if kill else signal.SIGTERM)

    def terminate(self) -> None:
        self._signal(False)

    def kill(self) -> None:
        self._signal(True)

    def wait(self, timeout: float | None = None) -> int:
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired("proxy zygote", timeout)
            time.sleep(0.01)
        assert self.returncode is not None
        return self.returncode


if __name__ == "__main__":
    serve(Path(sys.argv[1]), int(sys.argv[2]))
