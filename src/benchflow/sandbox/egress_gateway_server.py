"""The egress gateway: a task's denylist for sandboxes that may reach one address. Stdlib only.

A Modal sandbox (gVisor) has no netfilter, so BenchFlow's uid firewall cannot
hold its agent to the loopback egress proxy. What Modal does enforce, outside
the sandbox and for root too, is a create-time ``outbound_cidr_allowlist``. A
sandbox the gateway holds may reach this gateway's address and nothing else,
and BenchFlow's own egress proxy runs here instead of in the sandbox
(:mod:`benchflow.sandbox.egress_gateway` is the controller's side).

The operator runs it on a host with a fixed public IPv4 address and one open
TCP port. It needs ``python3`` (3.10 or newer) and ``openssl``, nothing else:

    python3 -I egress_gateway_server.py serve --state-dir D --listen 0.0.0.0:PORT --public IPv4:PORT

(with BenchFlow installed: ``python -m benchflow.sandbox.egress_gateway_server``).
``status --state-dir D`` prints the registered rollouts.

Per registered rollout it runs BenchFlow's egress proxy twice on loopback,
from the files the controller sent (the same files the in-sandbox proxy gets,
so the gateway runs the controller's proxy version): ``open`` with an empty
policy and ``deny`` with the task's. The rollout's phase says which one a new
connection reaches. Every connection to the public port starts with a
challenge: the gateway sends ``BFEG/1 <nonce>`` and the sandbox's relay
answers ``<rollout> <hmac>``, the HMAC-SHA256 of ``BFEG/1|<nonce>|<rollout>``
under the rollout's token. A wrong or missing answer is closed before any
byte reaches a proxy. Switching to ``deny`` cuts every connection opened in
``open``; switching to ``open`` cuts nothing.

Control: one JSON request per connection on the unix socket
``D/control.sock`` (mode 600, ``D`` 700): ``hello``, ``register``, ``deny``,
``open``, ``log``, ``deregister``, ``status``. A controller on another host
reaches it through ``ssh -L``. Each rollout's token, CA, policy and phase are
kept under ``D``, so a restarted gateway brings its rollouts back in their
phase; a rollout never deregistered is dropped at its time to live.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL = "BFEG/1"
#: Seconds a connection has to answer the challenge.
HANDSHAKE_TIMEOUT = 10.0
#: Connections allowed to wait in the challenge at once (a scan cannot exhaust threads).
MAX_PENDING = 256
#: Longest challenge answer line accepted (rollout id, space, 64 hex digits).
MAX_LINE = 512
#: Longest control request accepted (the proxy files travel in ``register``).
MAX_CONTROL = 16 * 1024 * 1024
#: The longest a rollout may live without being deregistered.
MAX_TTL_SEC = 2 * 86400
DEFAULT_TTL_SEC = 86400 + 3600
#: Seconds a proxy has to answer its health check, per try.
BACKEND_START_TIMEOUT = 20.0
#: Tries a proxy gets to become healthy, each on a fresh port.
BACKEND_START_TRIES = 3
#: Timeout of each half of a health check (the connect, then the read).
HEALTH_PROBE_TIMEOUT = 2.0
HEALTH_POLL_SEC = 0.1
#: Seconds a stopping proxy is waited for after SIGTERM, and again after SIGKILL.
STOP_WAIT_SEC = 5.0
REAP_INTERVAL = 15.0
#: Most bytes of one log returned by one ``log`` request.
MAX_LOG = 8 * 1024 * 1024
PHASES = ("open", "deny")
#: The logs ``log`` reads by offset: each phase's blocked and allowed log.
LOGS = ("open", "open_allowed", "deny", "deny_allowed")
#: A rollout's id, also its folder's name: never only dots.
ROLLOUT_ID = re.compile(r"(?!\.+\Z)[A-Za-z0-9._-]{1,128}")
TOKEN = re.compile(r"[0-9a-f]{64}")
#: Files a registration must carry, beside ``certs/<host>.pem`` leaf certificates
#: (``benchflow.sandbox.egress_denylist._runtime_files``).
REQUIRED_FILES = ("policy.json", "proxy.py", "request_guard.py", "ca.crt", "ca.key")
_CERT_FILE = re.compile(r"certs/[A-Za-z0-9._-]+\.pem")
#: Keys a policy here may carry. Allow mode (its DNS filter cannot hold:
#: the sandbox's resolver is the provider's), the native transports and a
#: model gateway port (it would name this host's loopback) are refused.
POLICY_KEYS = frozenset(
    {
        "blocked_urls",
        "blocked_hosts",
        "diagnostic_secrets",
        "model_gateway_port",
        "native_claude_model_only",
        "native_codex_model_only",
    }
)
OPEN_POLICY: dict[str, Any] = {"blocked_urls": [], "blocked_hosts": []}


class ControlError(Exception):
    """A control request the gateway refuses; its message is the answer's ``error``."""


def answer(token: str, nonce: str, rollout: str) -> str:
    """The challenge answer: HMAC-SHA256 of ``PROTOCOL|nonce|rollout`` under the token."""
    message = f"{PROTOCOL}|{nonce}|{rollout}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def parse_address(value: str) -> tuple[str, int]:
    """``host:port`` as a pair; ValueError otherwise."""
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"expected host:port, got {value!r}")
    return host, int(port)


def check_policy(policy: Any) -> dict[str, Any]:
    """*policy* (a ``policy.json`` object) when the gateway can hold it; ControlError otherwise."""
    if not isinstance(policy, dict):
        raise ControlError("policy must be a JSON object")
    extra = sorted(set(policy) - POLICY_KEYS)
    if extra:
        raise ControlError(
            f"policy keys {extra} are not held here (allowlists and their "
            "controller additions are refused)"
        )
    for key in ("blocked_urls", "blocked_hosts", "diagnostic_secrets"):
        value = policy.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ControlError(f"policy {key} must be a list of strings")
    if policy.get("model_gateway_port") is not None:
        raise ControlError(
            "a model gateway port names the sandbox's loopback; the relay serves it"
        )
    for key in ("native_claude_model_only", "native_codex_model_only"):
        if policy.get(key, False) is not False:
            raise ControlError(f"policy {key} is not held here")
    return policy


def register_max_sec() -> float:
    """The longest ``register`` can take: its open proxy's start, every try run out."""
    per_try = (
        BACKEND_START_TIMEOUT
        + 2 * HEALTH_PROBE_TIMEOUT
        + HEALTH_POLL_SEC
        + 2 * STOP_WAIT_SEC
    )
    return BACKEND_START_TRIES * per_try


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def backend_healthy(port: int, timeout: float = HEALTH_PROBE_TIMEOUT) -> bool:
    """Whether the proxy on ``127.0.0.1:port`` answers its ``/healthz`` with 200."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as probe:
            probe.sendall(
                b"GET /healthz HTTP/1.1\r\nHost: healthz\r\nConnection: close\r\n\r\n"
            )
            head = probe.recv(64)
    except OSError:
        return False
    return head.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200"))


def _bound(stderr: Path, offset: int, port: int) -> bool:
    """Whether *stderr* past *offset* has the proxy's line for a bind of ``port``."""
    line = f"egress proxy listening on 127.0.0.1:{port}".encode()
    try:
        with stderr.open("rb") as err:
            err.seek(offset)
            return line in err.read().splitlines()
    except OSError:
        return False


def _read_line(sock: socket.socket, limit: int) -> tuple[str | None, bytes]:
    r"""One ``\n``-terminated line shorter than *limit* (without it) and the
    bytes read after it; (None, b'') if none. The bound is the line's alone:
    the relay sends its first request bytes in the answer's write."""
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(min(65536, limit + 1 - len(data)))
        if not chunk:
            return None, b""
        data += chunk
        if len(data) > limit and b"\n" not in data[:limit]:
            return None, b""
    line, _, rest = data.partition(b"\n")
    try:
        return line.decode("utf-8"), rest
    except UnicodeDecodeError:
        return None, b""


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def read_log(path: Path, offset: int, limit: int = MAX_LOG) -> tuple[str, int]:
    """The whole lines of *path* from byte *offset* (at most *limit* bytes) and
    the offset after them."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = min(max(0, offset), size)
            handle.seek(start)
            data = handle.read(limit)
    except OSError:
        return "", max(0, offset)
    if b"\n" not in data:
        return "", start
    data = data[: data.rindex(b"\n") + 1]
    return data.decode("utf-8", "replace"), start + len(data)


class _Conn:
    """One spliced connection: the sandbox's side and the proxy's."""

    def __init__(self, phase: str, client: socket.socket, upstream: socket.socket):
        self.phase = phase
        self.client = client
        self.upstream = upstream

    def cut(self) -> None:
        for sock in (self.client, self.upstream):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def splice(client: socket.socket, upstream: socket.socket) -> None:
    """Copy bytes both ways until both directions end; then close both sockets."""
    back = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    back.start()
    _pump(client, upstream)
    back.join()
    for sock in (client, upstream):
        with contextlib.suppress(OSError):
            sock.close()


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=STOP_WAIT_SEC)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=STOP_WAIT_SEC)


@dataclass
class Backend:
    """One copy of BenchFlow's egress proxy, a child process on a loopback port."""

    name: str
    directory: Path
    python: str
    openssl: str
    port: int | None = None
    process: subprocess.Popen[bytes] | None = None

    def argv(self, port: int) -> list[str]:
        """The proxy's command line on loopback ``port``."""
        d = self.directory
        return [
            self.python,
            "-I",
            str(d / "proxy.py"),
            "--port",
            str(port),
            "--policy",
            str(d / f"{self.name}.policy.json"),
            "--cert-dir",
            str(d / f"certs-{self.name}"),
            "--ca-cert",
            str(d / "ca.crt"),
            "--ca-key",
            str(d / "ca.key"),
            "--openssl",
            self.openssl,
            "--log",
            str(d / f"{self.name}.jsonl"),
            "--allowed-log",
            str(d / f"{self.name}_allowed.jsonl"),
        ]

    def launch(self, policy: dict[str, Any]) -> None:
        """Write the policy and start the proxy; RuntimeError when it never becomes healthy."""
        self.stop()
        _write_private(
            self.directory / f"{self.name}.policy.json", json.dumps(policy).encode()
        )
        (self.directory / f"certs-{self.name}").mkdir(mode=0o700, exist_ok=True)
        stderr = self.directory / f"{self.name}.stderr"
        last = ""
        for _ in range(BACKEND_START_TRIES):
            port = _free_port()
            with stderr.open("ab") as err:
                offset = err.seek(0, os.SEEK_END)
                process = subprocess.Popen(
                    self.argv(port),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                    start_new_session=True,
                )
            deadline = time.monotonic() + BACKEND_START_TIMEOUT
            while time.monotonic() < deadline and process.poll() is None:
                # Every BenchFlow proxy answers /healthz alike: the port is
                # this child's only once it says it bound it.
                if _bound(stderr, offset, port) and backend_healthy(port):
                    self.port, self.process = port, process
                    return
                time.sleep(HEALTH_POLL_SEC)
            last = f"exit {process.poll()}"
            _stop_process(process)
        raise RuntimeError(f"the {self.name} proxy did not become healthy ({last})")

    def alive(self) -> bool:
        """Whether the proxy process is running."""
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        """Stop the proxy process, if any."""
        if self.process is not None:
            _stop_process(self.process)
        self.process, self.port = None, None


@dataclass
class Rollout:
    """A registered rollout: its secret, its proxies, its phase and its live connections."""

    rid: str
    token: str = field(repr=False)
    directory: Path
    expires_at: float
    phase: str = "open"
    deny_policy: dict[str, Any] | None = None
    backends: dict[str, Backend] = field(default_factory=dict)
    conns: set[_Conn] = field(default_factory=set)
    counters: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False

    def count(self, name: str, n: int = 1) -> None:
        """Add ``n`` to the rollout's counter ``name``."""
        self.counters[name] = self.counters.get(name, 0) + n

    def save(self) -> None:
        """Persist the rollout (mode 600) so a restarted gateway brings it back."""
        record = {
            "rollout": self.rid,
            "token": self.token,
            "expires_at": self.expires_at,
            "phase": self.phase,
            "deny_policy": self.deny_policy,
            "counters": self.counters,
            "pids": {
                name: backend.process.pid
                for name, backend in self.backends.items()
                if backend.process is not None
            },
        }
        _write_private(self.directory / "rollout.json", json.dumps(record).encode())


def policy_sha256(policy: dict[str, Any] | None) -> str | None:
    """The sha256 of *policy*'s canonical JSON."""
    if policy is None:
        return None
    text = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class Gateway:
    """The public listener, the control socket and the per-rollout proxies."""

    def __init__(
        self,
        state_dir: Path,
        listen: tuple[str, int],
        public: str,
        *,
        python: str | None = None,
        openssl: str | None = None,
        clock: Any = time.time,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.listen = listen
        self.public = public
        self.python = python or sys.executable
        found = openssl or shutil.which("openssl")
        if not found:
            raise RuntimeError(
                "the gateway needs openssl (the proxy mints certificates)"
            )
        self.openssl = found
        self.clock = clock
        self.started_at = round(clock(), 3)
        self._rollouts: dict[str, Rollout] = {}
        self._lock = threading.Lock()
        self._pending = threading.BoundedSemaphore(MAX_PENDING)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._front: socket.socket | None = None
        self._control: socket.socket | None = None
        self.counters: dict[str, int] = {}

    # -- lifecycle -----------------------------------------------------------

    @property
    def control_path(self) -> Path:
        return self.state_dir / "control.sock"

    @property
    def rollouts_dir(self) -> Path:
        return self.state_dir / "rollouts"

    @property
    def address(self) -> tuple[str, int]:
        """The bound public listener (its port when ``listen`` named port 0)."""
        assert self._front is not None
        host, port = self._front.getsockname()[:2]
        return str(host), int(port)

    def open(self) -> None:
        """Bind both sockets, bring back persisted rollouts, start the threads."""
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.rollouts_dir.mkdir(exist_ok=True, mode=0o700)
        front = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        front.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        front.bind(self.listen)
        front.listen(512)
        front.settimeout(1.0)
        self._front = front
        with contextlib.suppress(FileNotFoundError):
            self.control_path.unlink()
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            control.bind(str(self.control_path))
        finally:
            os.umask(old)
        control.listen(64)
        control.settimeout(1.0)
        self._control = control
        self._load()
        for target in (self._accept_front, self._accept_control, self._reap):
            thread = threading.Thread(target=target, daemon=True, name=target.__name__)
            thread.start()
            self._threads.append(thread)
        host, port = self.address
        self._event("started", listen=f"{host}:{port}", public=self.public)

    def close(self) -> None:
        """Close both sockets and stop every proxy (persisted state stays)."""
        self._stop.set()
        for sock in (self._front, self._control):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        for thread in self._threads:
            thread.join(timeout=5)
        with self._lock:
            rollouts = list(self._rollouts.values())
        for rollout in rollouts:
            with rollout.lock:
                for conn in list(rollout.conns):
                    conn.cut()
                for backend in rollout.backends.values():
                    backend.stop()
        with contextlib.suppress(FileNotFoundError):
            self.control_path.unlink()
        self._event("stopped")

    def serve_forever(self) -> None:
        """Run until SIGTERM or SIGINT."""
        self.open()
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        try:
            while not self._stop.wait(1.0):
                pass
        finally:
            self.close()

    def _event(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"ts": round(self.clock(), 3), "event": kind, **fields})
        with contextlib.suppress(OSError):
            fd = os.open(
                self.state_dir / "gateway.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(fd, "a") as handle:
                handle.write(line + "\n")

    def _count(self, name: str) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + 1

    # -- persisted rollouts --------------------------------------------------

    def _load(self) -> None:
        for path in sorted(self.rollouts_dir.glob("*/rollout.json")):
            try:
                data = json.loads(path.read_text())
                rollout = Rollout(
                    rid=data["rollout"],
                    token=data["token"],
                    directory=path.parent,
                    expires_at=float(data["expires_at"]),
                    phase=data["phase"] if data["phase"] in PHASES else "deny",
                    deny_policy=data.get("deny_policy"),
                    counters=dict(data.get("counters") or {}),
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._event("load_failed", path=str(path), error=str(exc)[:200])
                continue
            for pid in (data.get("pids") or {}).values():
                _kill_stale(int(pid), path.parent)
            if rollout.expires_at <= self.clock():
                shutil.rmtree(rollout.directory, ignore_errors=True)
                self._event("expired", rollout=rollout.rid, at="load")
                continue
            for name in PHASES:
                policy = OPEN_POLICY if name == "open" else rollout.deny_policy
                if policy is None:
                    continue
                backend = Backend(name, rollout.directory, self.python, self.openssl)
                rollout.backends[name] = backend
                try:
                    backend.launch(policy)
                except RuntimeError as exc:
                    self._event("proxy_failed", rollout=rollout.rid, error=str(exc))
            rollout.save()
            with self._lock:
                self._rollouts[rollout.rid] = rollout
            self._event("restored", rollout=rollout.rid, phase=rollout.phase)

    # -- the public listener -------------------------------------------------

    def _accept_front(self) -> None:
        assert self._front is not None
        while not self._stop.is_set():
            try:
                sock, _ = self._front.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            if not self._pending.acquire(blocking=False):
                self._count("refused_busy")
                with contextlib.suppress(OSError):
                    sock.close()
                continue
            threading.Thread(
                target=self._serve_front, args=(sock,), daemon=True
            ).start()

    def _serve_front(self, sock: socket.socket) -> None:
        released = False
        try:
            conn = self._handshake(sock)
            self._pending.release()
            released = True
            if conn is None:
                with contextlib.suppress(OSError):
                    sock.close()
                return
            rollout, link, rest = conn
            try:
                if rest:
                    link.upstream.sendall(rest)
                splice(link.client, link.upstream)
            except OSError:
                link.cut()
            finally:
                with rollout.lock:
                    rollout.conns.discard(link)
        finally:
            if not released:
                self._pending.release()

    def _handshake(self, sock: socket.socket) -> tuple[Rollout, _Conn, bytes] | None:
        """The challenge, then the connection to the rollout's phase proxy; None to refuse."""
        try:
            sock.settimeout(HANDSHAKE_TIMEOUT)
            nonce = secrets.token_hex(16)
            sock.sendall(f"{PROTOCOL} {nonce}\n".encode())
            line, rest = _read_line(sock, MAX_LINE)
        except OSError:
            self._count("handshake_dropped")
            return None
        rid, _, mac = (line or "").partition(" ")
        with self._lock:
            rollout = self._rollouts.get(rid)
        if rollout is None or not hmac.compare_digest(
            mac.encode(), answer(rollout.token, nonce, rid).encode()
        ):
            self._count("handshake_refused")
            return None
        with rollout.lock:
            backend = rollout.backends.get(rollout.phase)
            # A proxy that exited keeps its port until the reaper restarts it,
            # and another listener (another rollout's open proxy) may bind it.
            if (
                rollout.closed
                or backend is None
                or backend.port is None
                or not backend.alive()
            ):
                rollout.count("refused_no_proxy")
                return None
            try:
                upstream = socket.create_connection(
                    ("127.0.0.1", backend.port), timeout=10
                )
            except OSError:
                rollout.count("refused_proxy_down")
                return None
            for each in (sock, upstream):
                each.settimeout(None)
                each.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            link = _Conn(rollout.phase, sock, upstream)
            rollout.conns.add(link)
            rollout.count(f"conn_{rollout.phase}")
        return rollout, link, rest

    # -- control -------------------------------------------------------------

    def _accept_control(self) -> None:
        assert self._control is not None
        while not self._stop.is_set():
            try:
                sock, _ = self._control.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(
                target=self._serve_control, args=(sock,), daemon=True
            ).start()

    def _serve_control(self, sock: socket.socket) -> None:
        with sock:
            try:
                sock.settimeout(register_max_sec() + 60)
                line, _ = _read_line(sock, MAX_CONTROL)
                request = json.loads(line or "null")
                if not isinstance(request, dict):
                    raise ControlError("expected one JSON object per line")
                reply = self.handle(request)
            except ControlError as exc:
                reply = {"ok": False, "error": str(exc)}
            except (ValueError, OSError) as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            with contextlib.suppress(OSError):
                sock.sendall(json.dumps(reply).encode() + b"\n")

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one control request; ControlError for a refused one."""
        command = request.get("command")
        handler = {
            "hello": self._cmd_hello,
            "register": self._cmd_register,
            "deny": self._cmd_deny,
            "open": self._cmd_open,
            "log": self._cmd_log,
            "deregister": self._cmd_deregister,
            "status": self._cmd_status,
        }.get(command if isinstance(command, str) else "")
        if handler is None:
            raise ControlError(f"unknown command {command!r}")
        return handler(request)

    def _rollout(self, request: dict[str, Any]) -> Rollout:
        rid = request.get("rollout")
        with self._lock:
            rollout = self._rollouts.get(rid) if isinstance(rid, str) else None
        if rollout is None:
            raise ControlError(f"no rollout {rid!r}")
        return rollout

    def _cmd_hello(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            n = len(self._rollouts)
        return {
            "ok": True,
            "protocol": PROTOCOL,
            "public": self.public,
            "started_at": self.started_at,
            "rollouts": n,
        }

    def _cmd_register(self, request: dict[str, Any]) -> dict[str, Any]:
        rid, token = request.get("rollout"), request.get("token")
        if not isinstance(rid, str) or not ROLLOUT_ID.fullmatch(rid):
            raise ControlError(
                "rollout must match [A-Za-z0-9._-]{1,128}, not only dots"
            )
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            raise ControlError("token must be 64 lowercase hex digits")
        ttl = request.get("ttl_sec", DEFAULT_TTL_SEC)
        if type(ttl) is not int or not 0 < ttl <= MAX_TTL_SEC:
            raise ControlError(f"ttl_sec must be an integer in (0, {MAX_TTL_SEC}]")
        files = request.get("files")
        if not isinstance(files, dict) or any(n not in files for n in REQUIRED_FILES):
            raise ControlError(f"files must hold {', '.join(REQUIRED_FILES)}")
        decoded: dict[str, bytes] = {}
        for name, value in files.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ControlError("files maps names to base64 strings")
            if name not in REQUIRED_FILES and not _CERT_FILE.fullmatch(name):
                raise ControlError(f"unexpected file {name!r}")
            decoded[name] = base64.b64decode(value, validate=True)
        try:
            check_policy(json.loads(decoded["policy.json"]))
        except ValueError as exc:
            raise ControlError(f"policy.json is not JSON: {exc}") from exc
        with self._lock:
            existing = self._rollouts.get(rid)
        if existing is not None:
            if hmac.compare_digest(existing.token, token):
                return {"ok": True, "phase": existing.phase, "existing": True}
            raise ControlError(f"rollout {rid!r} is registered with another token")
        directory = self.rollouts_dir / rid
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(mode=0o700)
        for name, data in decoded.items():
            target = directory / name.replace("certs/", "certs-deny/", 1)
            _write_private(target, data)
        rollout = Rollout(rid, token, directory, self.clock() + ttl)
        backend = Backend("open", directory, self.python, self.openssl)
        rollout.backends["open"] = backend
        try:
            backend.launch(OPEN_POLICY)
        except RuntimeError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ControlError(str(exc)) from exc
        rollout.save()
        with self._lock:
            self._rollouts[rid] = rollout
        self._event("registered", rollout=rid, ttl_sec=ttl)
        return {"ok": True, "phase": rollout.phase}

    def _cmd_deny(self, request: dict[str, Any]) -> dict[str, Any]:
        rollout = self._rollout(request)
        policy = check_policy(request.get("policy"))
        with rollout.lock:
            backend = rollout.backends.get("deny")
            if policy != rollout.deny_policy or backend is None or not backend.alive():
                backend = backend or Backend(
                    "deny", rollout.directory, self.python, self.openssl
                )
                rollout.backends["deny"] = backend
                # Deny first: a proxy that fails to start leaves the rollout
                # in deny with no proxy, which refuses every connection.
                rollout.phase, rollout.deny_policy = "deny", policy
                try:
                    backend.launch(policy)
                except RuntimeError as exc:
                    rollout.save()
                    raise ControlError(str(exc)) from exc
            rollout.phase = "deny"
            cut = [conn for conn in rollout.conns if conn.phase != "deny"]
            for conn in cut:
                conn.cut()
            rollout.count("to_deny")
            rollout.save()
        self._event("phase", rollout=rollout.rid, phase="deny", cut=len(cut))
        return {
            "ok": True,
            "phase": "deny",
            "cut": len(cut),
            "policy_sha256": policy_sha256(policy),
        }

    def _cmd_open(self, request: dict[str, Any]) -> dict[str, Any]:
        rollout = self._rollout(request)
        with rollout.lock:
            rollout.phase = "open"
            rollout.count("to_open")
            rollout.save()
        self._event("phase", rollout=rollout.rid, phase="open")
        return {"ok": True, "phase": "open"}

    def _cmd_log(self, request: dict[str, Any]) -> dict[str, Any]:
        rollout = self._rollout(request)
        offsets = request.get("offsets") or {}
        if not isinstance(offsets, dict):
            raise ControlError("offsets must map log names to byte offsets")
        logs = {}
        for name in LOGS:
            offset = offsets.get(name, 0)
            if type(offset) is not int or offset < 0:
                raise ControlError(f"offset of {name} must be a non-negative integer")
            text, end = read_log(rollout.directory / f"{name}.jsonl", offset, MAX_LOG)
            logs[name] = {"text": text, "offset": end}
        with rollout.lock:
            counters = dict(rollout.counters)
            phase = rollout.phase
        return {"ok": True, "phase": phase, "logs": logs, "counters": counters}

    def _cmd_deregister(self, request: dict[str, Any]) -> dict[str, Any]:
        rollout = self._rollout(request)
        counters = self._drop(rollout)
        self._event("deregistered", rollout=rollout.rid, counters=counters)
        return {"ok": True, "counters": counters}

    def _drop(self, rollout: Rollout) -> dict[str, int]:
        """Stop the rollout's proxies and remove its folder (its CA key with it)."""
        with self._lock:
            self._rollouts.pop(rollout.rid, None)
        with rollout.lock:
            rollout.closed = True
            for conn in list(rollout.conns):
                conn.cut()
            for backend in rollout.backends.values():
                backend.stop()
            counters = dict(rollout.counters)
        shutil.rmtree(rollout.directory, ignore_errors=True)
        return counters

    def _cmd_status(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rollouts = list(self._rollouts.values())
            counters = dict(self.counters)
        rows = []
        for rollout in rollouts:
            with rollout.lock:
                rows.append(
                    {
                        "rollout": rollout.rid,
                        "phase": rollout.phase,
                        "expires_in_sec": round(rollout.expires_at - self.clock()),
                        "proxies": {n: b.alive() for n, b in rollout.backends.items()},
                        "connections": len(rollout.conns),
                        "counters": dict(rollout.counters),
                    }
                )
        return {
            "ok": True,
            "public": self.public,
            "started_at": self.started_at,
            "rollouts": rows,
            "counters": counters,
        }

    # -- housekeeping --------------------------------------------------------

    def _reap(self) -> None:
        while not self._stop.wait(REAP_INTERVAL):
            self.reap_once()

    def reap_once(self) -> None:
        """Drop expired rollouts and restart any proxy that died."""
        with self._lock:
            rollouts = list(self._rollouts.values())
        now = self.clock()
        for rollout in rollouts:
            if rollout.expires_at <= now:
                counters = self._drop(rollout)
                self._event("expired", rollout=rollout.rid, counters=counters)
                continue
            with rollout.lock:
                for name, backend in rollout.backends.items():
                    policy = OPEN_POLICY if name == "open" else rollout.deny_policy
                    if backend.alive() or policy is None:
                        continue
                    try:
                        backend.launch(policy)
                        self._event("proxy_restarted", rollout=rollout.rid, proxy=name)
                    except RuntimeError as exc:
                        self._event(
                            "proxy_failed",
                            rollout=rollout.rid,
                            proxy=name,
                            error=str(exc),
                        )
                rollout.save()


def _kill_stale(pid: int, directory: Path) -> None:
    """Stop a proxy a previous gateway left running for this rollout (its argv names the folder)."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return
    if any(str(directory).encode() in part for part in argv):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def request(
    control: str | Path, payload: dict[str, Any], timeout: float = 60.0
) -> dict[str, Any]:
    """Send one control request to the gateway at ``control`` and return its answer."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(control))
        sock.sendall(json.dumps(payload).encode() + b"\n")
        line, _ = _read_line(sock, MAX_CONTROL + MAX_LOG * len(LOGS) * 2)
    if line is None:
        raise ConnectionError(f"the gateway at {control} sent no answer")
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise ConnectionError(f"the gateway at {control} sent {line[:200]!r}")
    return reply


def main(argv: list[str] | None = None) -> int:
    """``serve`` the gateway, or print its ``status``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the gateway until SIGTERM")
    serve.add_argument("--state-dir", type=Path, required=True)
    serve.add_argument(
        "--listen", required=True, help="host:port to bind (the public port)"
    )
    serve.add_argument("--public", required=True, help="IPv4:port sandboxes connect to")
    serve.add_argument(
        "--python", help="interpreter for the proxies (default: this one)"
    )
    serve.add_argument("--openssl", help="openssl executable (default: from PATH)")
    status = sub.add_parser("status", help="print the gateway's rollouts")
    status.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        reply = request(args.state_dir / "control.sock", {"command": "status"})
        print(json.dumps(reply, indent=1))
        return 0
    parse_address(args.public)
    gateway = Gateway(
        args.state_dir,
        parse_address(args.listen),
        args.public,
        python=args.python,
        openssl=args.openssl,
    )
    gateway.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
