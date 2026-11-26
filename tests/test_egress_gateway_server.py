"""The egress gateway (benchflow.sandbox.egress_gateway_server), over real loopback sockets.

Guards the Modal denylist the egress gateway holds (PR #1253, which adds
``BENCHFLOW_EGRESS_GATEWAY``): every connection is challenged before a byte
reaches a proxy, a connection reaches its rollout's phase proxy (BenchFlow's
own egress proxy, run here), ``deny`` cuts what ``open`` let in, and a
rollout outlives a gateway restart and ends at its time to live.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from benchflow.sandbox import egress_gateway_server as server
from benchflow.sandbox.egress_denylist import EgressDenylist, _runtime_files

BLOCKED = b"GET http://blocked.test/ HTTP/1.1\r\nHost: blocked.test\r\n\r\n"
HEALTH = b"GET /healthz HTTP/1.1\r\nHost: healthz\r\n\r\n"
TOKEN = "ab" * 32


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def state_dir():
    # A unix socket path must stay short (104 bytes on macOS).
    folder = Path(tempfile.mkdtemp(prefix="bfeg-"))
    yield folder / "state"
    shutil.rmtree(folder, ignore_errors=True)


def _gateway(state_dir: Path, clock: Clock | None = None) -> server.Gateway:
    gateway = server.Gateway(
        state_dir,
        ("127.0.0.1", 0),
        "127.0.0.1:1",
        openssl=shutil.which("openssl") or "/usr/bin/false",
        clock=clock or Clock(),
    )
    gateway.open()
    gateway.public = f"127.0.0.1:{gateway.address[1]}"
    return gateway


@pytest.fixture
def gateway(state_dir):
    gateway = _gateway(state_dir)
    yield gateway
    gateway.close()


def _files(policy: EgressDenylist | None = None) -> dict[str, str]:
    files = _runtime_files(
        policy or EgressDenylist(("paper.test/abs/1",), ("blocked.test",)), None
    )
    return {k: base64.b64encode(v).decode() for k, v in files.items()}


def _policy(**extra) -> dict:
    return {
        "blocked_urls": ["paper.test/abs/1"],
        "blocked_hosts": ["blocked.test"],
        "model_gateway_port": None,
        "native_claude_model_only": False,
        "diagnostic_secrets": [],
        **extra,
    }


def _control(gateway: server.Gateway, **payload) -> dict:
    return server.request(gateway.control_path, payload, timeout=120)


def _register(gateway: server.Gateway, rid: str = "trial-1", **extra) -> dict:
    reply = _control(
        gateway,
        command="register",
        rollout=rid,
        token=TOKEN,
        files=extra.pop("files", None) or _files(),
        **extra,
    )
    assert reply["ok"], reply
    return reply


def _challenge(gateway: server.Gateway) -> tuple[socket.socket, str]:
    """A new connection to the public port and the nonce it was sent."""
    sock = socket.create_connection(gateway.address, timeout=10)
    line = b""
    while not line.endswith(b"\n"):
        line += sock.recv(1)
    proto, nonce = line.decode().split()
    assert proto == server.PROTOCOL
    return sock, nonce


def _connect(gateway: server.Gateway, rid: str = "trial-1", mac: str | None = None):
    """A relay's side of the challenge; the socket, ready for the request."""
    sock, nonce = _challenge(gateway)
    answer = server.answer(TOKEN, nonce, rid) if mac is None else mac
    sock.sendall(f"{rid} {answer}\n".encode())
    return sock


def _wait_connections(gateway: server.Gateway, count: int) -> None:
    """Until the one registered rollout holds *count* connections: the
    gateway's thread answers the challenge after the relay's write returns."""
    deadline = time.monotonic() + 10
    while True:
        (row,) = _control(gateway, command="status")["rollouts"]
        if row["connections"] >= count or time.monotonic() > deadline:
            assert row["connections"] == count, row
            return
        time.sleep(0.02)


def _counters(gateway: server.Gateway) -> dict[str, int]:
    """The one registered rollout's counters."""
    (row,) = _control(gateway, command="status")["rollouts"]
    return row["counters"]


def _ask(sock: socket.socket, request: bytes) -> bytes:
    sock.sendall(request)
    data = b""
    try:
        while chunk := sock.recv(4096):
            data += chunk
    except OSError:
        pass
    sock.close()
    return data


def test_a_wrong_or_missing_answer_reaches_no_proxy(gateway, monkeypatch):
    """A connection that fails the challenge is closed before a byte of it
    reaches a proxy: the proxies see no connection at all.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    _register(gateway)
    wrong = _connect(gateway, mac="0" * 64)
    assert _ask(wrong, HEALTH) == b""
    other = _connect(gateway, rid="trial-2")
    assert _ask(other, HEALTH) == b""
    monkeypatch.setattr(server, "HANDSHAKE_TIMEOUT", 0.5)
    silent = socket.create_connection(gateway.address, timeout=10)
    assert silent.recv(64).startswith(b"BFEG/1 ")
    assert silent.recv(64) == b""
    silent.close()
    status = _control(gateway, command="status")
    assert status["counters"]["handshake_refused"] == 2
    (row,) = status["rollouts"]
    assert row["counters"] == {}


def test_a_challenge_answered_a_byte_at_a_time_ends_at_the_handshake_timeout(
    gateway, monkeypatch
):
    """The whole challenge ends at ``HANDSHAKE_TIMEOUT``: a client that sends
    one byte at a time is closed then, its slot freed.

    Guards the review of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``): the
    timeout bounded each read, so such a client held one of the
    ``MAX_PENDING`` slots, shared by every rollout, for up to ``MAX_LINE``
    timeouts."""
    monkeypatch.setattr(server, "HANDSHAKE_TIMEOUT", 1.0)
    sock, _ = _challenge(gateway)
    sock.settimeout(0.25)
    begun = time.monotonic()
    ended = None
    while ended is None and time.monotonic() - begun < 15:
        try:
            sock.sendall(b"a")
            if sock.recv(1) == b"":
                ended = time.monotonic() - begun
        except TimeoutError:
            continue
        except OSError:
            ended = time.monotonic() - begun
    sock.close()
    assert ended is not None and ended < 6, ended
    counters = _control(gateway, command="status")["counters"]
    assert (
        counters.get("handshake_refused", 0) + counters.get("handshake_dropped", 0) == 1
    ), counters


def test_a_thread_that_cannot_start_refuses_that_connection_alone(gateway, monkeypatch):
    """When the host can start no thread for a connection, that connection
    is closed and the gateway keeps accepting: the next one is served.

    Guards the review of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``): the
    error ended the public listener's accept loop, and held its challenge
    slot, so every rollout's later connections hung while ``hello`` still
    answered."""
    _register(gateway)
    real_start = threading.Thread.start
    failed = []

    def start(self):
        if getattr(self, "_target", None) == gateway._serve_front and not failed:
            failed.append(self)
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    first = socket.create_connection(gateway.address, timeout=10)
    with contextlib.suppress(ConnectionResetError):
        assert first.recv(64) == b""
    first.close()
    assert _ask(_connect(gateway), HEALTH).startswith(b"HTTP/1.1 200")
    assert failed
    assert _control(gateway, command="status")["counters"]["refused_no_thread"] == 1
    assert all(thread.is_alive() for thread in gateway._threads)


def test_an_answer_is_good_for_its_own_connection_only(gateway):
    """An answer is bound to the nonce its connection was sent: replayed on
    the next connection (by a host that captured it on the way), it is
    refused and reaches no proxy.

    Guards the review round of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)
    (mutant M5, a constant nonce, survived the first round's tests)."""
    _register(gateway)
    first, nonce = _challenge(gateway)
    captured = f"trial-1 {server.answer(TOKEN, nonce, 'trial-1')}\n".encode()
    first.sendall(captured)
    assert _ask(first, HEALTH).startswith(b"HTTP/1.1 200")
    replay, _ = _challenge(gateway)
    replay.sendall(captured)
    assert _ask(replay, HEALTH) == b""
    status = _control(gateway, command="status")
    assert status["counters"]["handshake_refused"] == 1
    (row,) = status["rollouts"]
    assert row["counters"] == {"conn_open": 1}


def test_a_long_request_sent_with_the_answer_reaches_the_proxy(gateway):
    """The relay sends its answer and the client's first bytes in one write:
    a request longer than ``MAX_LINE`` behind a short answer reaches the
    proxy. An answer line that long is still refused.

    Guards the third review round of PR #1253 (adds
    ``BENCHFLOW_EGRESS_GATEWAY``): ``_read_line`` refused any read past
    ``MAX_LINE``, however short the line in it."""
    _register(gateway)
    pad = b"".join(b"X-Pad-%d: %s\r\n" % (i, b"a" * 60) for i in range(10))
    request = HEALTH[:-2] + pad + b"\r\n"
    assert len(request) > server.MAX_LINE
    sock, nonce = _challenge(gateway)
    answer = f"trial-1 {server.answer(TOKEN, nonce, 'trial-1')}\n".encode()
    assert _ask(sock, answer + request).startswith(b"HTTP/1.1 200")
    long, _ = _challenge(gateway)
    assert _ask(long, b"x" * server.MAX_LINE + b"\n" + HEALTH) == b""


def test_a_dead_deny_proxy_s_port_is_never_dialled(state_dir, monkeypatch):
    """In ``deny``, a connection whose deny proxy has exited is refused, even
    once another listener (such as another rollout's open proxy) has bound
    the dead proxy's port before the reaper restarts it.

    Guards the review round of PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``):
    ``_handshake`` dialled ``Backend.port`` without checking that the proxy
    still ran, and the denied rollout's connection reached the new listener."""
    monkeypatch.setattr(server, "REAP_INTERVAL", 3600.0)
    gateway = _gateway(state_dir)
    try:
        _register(gateway)
        _control(gateway, command="deny", rollout="trial-1", policy=_policy())
        backend = gateway._rollouts["trial-1"].backends["deny"]
        port, process = backend.port, backend.process
        assert port is not None and process is not None
        process.kill()
        process.wait(timeout=10)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
            listener.listen(8)
            client = _connect(gateway)
            client.sendall(BLOCKED)
            deadline = time.monotonic() + 10
            while not {"refused_no_proxy", "conn_deny"} & set(
                counters := _counters(gateway)
            ):
                assert time.monotonic() < deadline, counters
                time.sleep(0.05)
            listener.settimeout(0.5)
            try:
                stolen, _ = listener.accept()
            except TimeoutError:
                received = b""
            else:
                with stolen:
                    stolen.settimeout(5)
                    received = stolen.recv(4096)
        assert received == b""
        assert counters.get("conn_deny") is None
        assert counters["refused_no_proxy"] == 1
        # Closed with the request unread: an end of file or a reset, no reply.
        client.settimeout(10)
        with client, contextlib.suppress(ConnectionResetError):
            assert client.recv(64) == b""
    finally:
        gateway.close()


@contextlib.contextmanager
def _stand_in(port: int = 0):
    """Another proxy's stand-in on ``127.0.0.1:port``: answers ``/healthz``
    with 200 and keeps every other request. Yields its port and what it kept."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)
    listener.settimeout(0.2)
    received: list[bytes] = []
    done = threading.Event()

    def serve() -> None:
        while not done.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                with contextlib.suppress(OSError):
                    data = conn.recv(4096)
                    if data.startswith(b"GET /healthz"):
                        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                    else:
                        received.append(data)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1], received
    finally:
        done.set()
        thread.join(timeout=5)
        listener.close()


def test_a_launch_takes_only_the_port_its_own_proxy_bound(gateway, monkeypatch):
    """A proxy launched on a port another listener already holds (two
    launches given the same port by bind(0)) is not taken for healthy on that
    listener's ``/healthz``: the launch moves to a new port, and the denied
    request never reaches the other listener. The same holds when the reaper
    relaunches a dead proxy on that proxy's own old port, now another
    listener's: the dead proxy's line for that port, already in the stderr,
    does not count, since a launch reads only what follows its offset.

    Guards the second review round of PR #1253 (adds
    ``BENCHFLOW_EGRESS_GATEWAY``): ``Backend.launch`` took the first ``/healthz``
    200 on the port, and ``deny`` recorded the other listener's port."""
    _register(gateway)
    ports: list[int] = []
    free_port = server._free_port
    monkeypatch.setattr(
        server, "_free_port", lambda: ports.pop() if ports else free_port()
    )
    with _stand_in() as (taken, received):
        ports.append(taken)
        reply = _control(gateway, command="deny", rollout="trial-1", policy=_policy())
        assert reply["ok"], reply
        refused = _ask(_connect(gateway), BLOCKED)
        assert received == []
        assert refused.startswith(b"HTTP/1.1 403")
        backend = gateway._rollouts["trial-1"].backends["deny"]
        assert backend.port != taken and backend.alive()
    # The reaper's relaunch, given the port the dead deny proxy itself bound.
    old, process = backend.port, backend.process
    assert old is not None and process is not None
    process.kill()
    process.wait(timeout=10)
    stderr = (backend.directory / "deny.stderr").read_text()
    assert f"egress proxy listening on 127.0.0.1:{old}" in stderr
    with _stand_in(old) as (_, received):
        ports.append(old)
        gateway.reap_once()
        refused = _ask(_connect(gateway), BLOCKED)
        assert received == []
        assert refused.startswith(b"HTTP/1.1 403")
        assert backend.port != old and backend.alive()


def test_a_connection_reaches_the_phase_proxy(gateway):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    _register(gateway)
    assert _ask(_connect(gateway), HEALTH).startswith(b"HTTP/1.1 200")
    # open: no policy, so the proxy tries the host (which does not resolve).
    assert _ask(_connect(gateway), BLOCKED).startswith(b"HTTP/1.1 502")
    reply = _control(gateway, command="deny", rollout="trial-1", policy=_policy())
    assert reply["phase"] == "deny" and len(reply["policy_sha256"]) == 64
    refused = _ask(_connect(gateway), BLOCKED)
    assert refused.startswith(b"HTTP/1.1 403") and b"X-BenchFlow-Blocked: 1" in refused
    assert _ask(_connect(gateway), HEALTH).startswith(b"HTTP/1.1 200")
    _control(gateway, command="open", rollout="trial-1")
    assert _ask(_connect(gateway), BLOCKED).startswith(b"HTTP/1.1 502")


def test_deny_cuts_what_open_let_in_and_open_cuts_nothing(gateway):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    _register(gateway)
    early = _connect(gateway)
    _wait_connections(gateway, 1)
    reply = _control(gateway, command="deny", rollout="trial-1", policy=_policy())
    assert reply["cut"] == 1
    early.settimeout(10)
    assert early.recv(64) == b""
    held = _connect(gateway)
    _control(gateway, command="open", rollout="trial-1")
    held.settimeout(0.5)
    with pytest.raises(TimeoutError):
        held.recv(64)
    assert _ask(held, HEALTH).startswith(b"HTTP/1.1 200")


def test_a_restarted_gateway_brings_its_rollouts_back_in_their_phase(state_dir):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    clock = Clock()
    first = _gateway(state_dir, clock)
    try:
        _register(first)
        _control(first, command="deny", rollout="trial-1", policy=_policy())
    finally:
        first.close()
    second = _gateway(state_dir, clock)
    try:
        (row,) = _control(second, command="status")["rollouts"]
        assert row["rollout"] == "trial-1" and row["phase"] == "deny"
        assert row["proxies"] == {"open": True, "deny": True}
        assert _ask(_connect(second), BLOCKED).startswith(b"HTTP/1.1 403")
    finally:
        second.close()


def test_a_rollout_ends_at_its_time_to_live_and_at_deregister(state_dir):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    clock = Clock()
    gateway = _gateway(state_dir, clock)
    try:
        _register(gateway, "trial-1", ttl_sec=60)
        _register(gateway, "trial-2")
        folder = gateway.rollouts_dir / "trial-1"
        assert (folder / "ca.key").is_file()
        clock.now += 61
        gateway.reap_once()
        assert not folder.exists()
        rows = _control(gateway, command="status")["rollouts"]
        assert [r["rollout"] for r in rows] == ["trial-2"]
        assert _control(gateway, command="deregister", rollout="trial-2")["ok"]
        assert not (gateway.rollouts_dir / "trial-2").exists()
        assert _control(gateway, command="status")["rollouts"] == []
        assert _ask(_connect(gateway, "trial-2"), HEALTH) == b""
    finally:
        gateway.close()


def test_a_rollout_deregistered_during_a_reap_stays_stopped(state_dir, monkeypatch):
    """A rollout deregistered after the reaper's snapshot, before the reaper
    takes its lock and before its folder is removed, is not relaunched: no
    registry entry, reap or shutdown would reach that proxy again.

    Guards the third review round of PR #1253 (adds
    ``BENCHFLOW_EGRESS_GATEWAY``): ``reap_once`` restarted a dead proxy
    without checking ``Rollout.closed``."""
    monkeypatch.setattr(server, "REAP_INTERVAL", 3600.0)
    clock = Clock()
    gateway = _gateway(state_dir, clock)
    backend = None
    try:
        _register(gateway)
        rollout = gateway._rollouts["trial-1"]
        backend = rollout.backends["open"]
        assert backend.process is not None
        backend.process.kill()
        backend.process.wait(timeout=10)

        def deregistered_in_the_gap() -> float:
            # reap_once reads the clock once, after its snapshot.
            gateway.clock = clock
            with monkeypatch.context() as patch:
                patch.setattr(server.shutil, "rmtree", lambda *a, **k: None)
                gateway._drop(rollout)
            return clock()

        gateway.clock = deregistered_in_the_gap
        gateway.reap_once()
        assert rollout.closed and "trial-1" not in gateway._rollouts
        assert not backend.alive()
    finally:
        if backend is not None:
            backend.stop()
        gateway.close()


@pytest.mark.parametrize("command", ["deny", "open"])
def test_a_phase_change_after_deregistration_is_refused(
    state_dir, monkeypatch, command
):
    """A deny or open that looked its rollout up before a deregistration took
    the rollout's lock is refused, and no proxy is launched for it.

    Guards the third review round of PR #1253 (adds
    ``BENCHFLOW_EGRESS_GATEWAY``): ``_cmd_deny`` and ``_cmd_open`` did not
    check ``Rollout.closed``, as ``reap_once`` now does."""
    gateway = _gateway(state_dir)
    rollout = None
    try:
        _register(gateway)
        rollout = gateway._rollouts["trial-1"]
        with monkeypatch.context() as patch:
            patch.setattr(server.shutil, "rmtree", lambda *a, **k: None)
            gateway._drop(rollout)
        # The handler looked the rollout up before the drop took its lock.
        monkeypatch.setattr(gateway, "_rollout", lambda request: rollout)
        reply = _control(gateway, command=command, rollout="trial-1", policy=_policy())
        assert not reply["ok"], reply
        assert not any(backend.alive() for backend in rollout.backends.values())
    finally:
        if rollout is not None:
            for backend in rollout.backends.values():
                backend.stop()
        gateway.close()


@pytest.mark.parametrize(
    "policy",
    [
        EgressDenylist((), (), allowed_hosts=("pypi.org",)),
        EgressDenylist((), (), native_claude_model_only=True),
        EgressDenylist((), (), native_codex_model_only=True),
    ],
    ids=["allowlist", "native-claude", "native-codex"],
)
def test_register_refuses_a_policy_the_gateway_cannot_hold(gateway, policy):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    reply = server.request(
        gateway.control_path,
        {
            "command": "register",
            "rollout": "trial-1",
            "token": TOKEN,
            "files": _files(policy),
        },
        timeout=120,
    )
    assert reply["ok"] is False
    assert not (gateway.rollouts_dir / "trial-1").exists()


@pytest.mark.parametrize(
    "extra",
    [
        {"model_gateway_port": 4000},
        {"allowed_hosts": ["pypi.org"]},
        {"native_claude_model_only": True},
        {"native_codex_model_only": True},
        {"blocked_hosts": "blocked.test"},
    ],
    ids=["gateway-port", "allowlist", "native-claude", "native-codex", "not-a-list"],
)
def test_deny_refuses_a_policy_the_gateway_cannot_hold(gateway, extra):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    _register(gateway)
    reply = server.request(
        gateway.control_path,
        {"command": "deny", "rollout": "trial-1", "policy": _policy(**extra)},
    )
    assert reply["ok"] is False
    (row,) = _control(gateway, command="status")["rollouts"]
    assert row["phase"] == "open"


def test_log_reads_each_proxy_log_from_an_offset(gateway):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    _register(gateway)
    _control(gateway, command="deny", rollout="trial-1", policy=_policy())
    _ask(_connect(gateway), BLOCKED)
    first = _control(gateway, command="log", rollout="trial-1")
    (line,) = first["logs"]["deny"]["text"].splitlines()
    assert json.loads(line)["action"] == "blocked"
    offsets = {name: log["offset"] for name, log in first["logs"].items()}
    _ask(_connect(gateway), BLOCKED)
    second = _control(gateway, command="log", rollout="trial-1", offsets=offsets)
    assert len(second["logs"]["deny"]["text"].splitlines()) == 1
    assert second["logs"]["deny"]["offset"] > offsets["deny"]


def test_the_gateway_needs_nothing_outside_the_standard_library():
    """It runs on an operator's host with no BenchFlow installed.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    tree = ast.parse(Path(server.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
