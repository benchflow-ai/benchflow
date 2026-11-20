"""The egress gateway (benchflow.sandbox.egress_gateway_server), over real loopback sockets.

Guards the Modal denylist the egress gateway holds (the PR that adds
``BENCHFLOW_EGRESS_GATEWAY``): every connection is challenged before a byte
reaches a proxy, a connection reaches its rollout's phase proxy (BenchFlow's
own egress proxy, run here), ``deny`` cuts what ``open`` let in, and a
rollout outlives a gateway restart and ends at its time to live.
"""

from __future__ import annotations

import ast
import base64
import json
import shutil
import socket
import sys
import tempfile
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


def _connect(gateway: server.Gateway, rid: str = "trial-1", mac: str | None = None):
    """A relay's side of the challenge; the socket, ready for the request."""
    sock = socket.create_connection(gateway.address, timeout=10)
    line = b""
    while not line.endswith(b"\n"):
        line += sock.recv(1)
    proto, nonce = line.decode().split()
    assert proto == server.PROTOCOL
    answer = server.answer(TOKEN, nonce, rid) if mac is None else mac
    sock.sendall(f"{rid} {answer}\n".encode())
    return sock


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
    reaches a proxy: the proxies see no connection at all."""
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


def test_a_connection_reaches_the_phase_proxy(gateway):
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
    _register(gateway)
    early = _connect(gateway)
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
    _register(gateway)
    reply = server.request(
        gateway.control_path,
        {"command": "deny", "rollout": "trial-1", "policy": _policy(**extra)},
    )
    assert reply["ok"] is False
    (row,) = _control(gateway, command="status")["rollouts"]
    assert row["phase"] == "open"


def test_log_reads_each_proxy_log_from_an_offset(gateway):
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
    """It runs on an operator's host with no BenchFlow installed."""
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
