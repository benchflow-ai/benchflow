"""Root's relay to the egress gateway (benchflow.sandbox._egress_gateway_relay), run as
the sandbox runs it (``python3 -I relay.py config.json``) against a stand-in gateway.

Guards the Modal denylist the egress gateway holds (PR #1253, which adds
``BENCHFLOW_EGRESS_GATEWAY``): the relay answers the challenge without sending
the token, forwards everything else byte for byte, closes on a bad challenge,
and serves a loopback CONNECT itself, the wire shape Gemini's Undici client
uses for the in-sandbox model gateway (tests/test_egress_model_gateway.py).
"""

from __future__ import annotations

import ast
import http.client
import http.server
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from benchflow.sandbox import _egress_gateway_relay as relay

TOKEN = "cd" * 32
SOURCE = Path(relay.__file__)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class StandInGateway:
    """Sends ``challenge``, records what each connection sent, answers ``reply``."""

    def __init__(self, challenge: bytes | None = None, reply: bytes = b"") -> None:
        self.challenge = challenge
        self.reply = reply
        self.received: list[bytes] = []
        self.nonces: list[str] = []
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                sock, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(sock,), daemon=True).start()

    def _one(self, sock: socket.socket) -> None:
        with sock:
            nonce = f"n{len(self.nonces)}"
            self.nonces.append(nonce)
            sock.sendall(self.challenge or f"BFEG/1 {nonce}\n".encode())
            sock.settimeout(5)
            data = b""
            try:
                while b"\r\n\r\n" not in data:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
            except OSError:
                pass
            self.received.append(data)
            if self.reply:
                sock.sendall(self.reply)

    def close(self) -> None:
        self.listener.close()


@pytest.fixture
def run_relay(tmp_path):
    started: list[subprocess.Popen] = []

    def run(gateway, rollout: str = "trial-1") -> int:
        port = _free_port()
        config = tmp_path / f"relay-{port}.json"
        config.write_text(
            json.dumps(
                {
                    "gateway": f"127.0.0.1:{gateway.port}",
                    "rollout": rollout,
                    "token": TOKEN,
                    "port": port,
                }
            )
        )
        process = subprocess.Popen(
            [sys.executable, "-I", str(SOURCE), str(config)],
            stderr=subprocess.PIPE,
        )
        started.append(process)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return port
            except OSError:
                time.sleep(0.05)
        raise AssertionError("the relay did not listen")

    yield run
    for process in started:
        process.kill()
        process.wait()


def _exchange(port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        data = b""
        try:
            while chunk := sock.recv(4096):
                data += chunk
        except OSError:
            pass
        return data


def test_the_challenge_is_answered_and_the_bytes_forwarded_as_sent(run_relay):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    gateway = StandInGateway(reply=b"HTTP/1.1 204 No Content\r\n\r\n")
    try:
        port = run_relay(gateway)
        request = b"GET http://example.org/x HTTP/1.1\r\nHost: example.org\r\n\r\n"
        assert _exchange(port, request) == b"HTTP/1.1 204 No Content\r\n\r\n"
        (sent,) = gateway.received
        answer, _, rest = sent.partition(b"\n")
        assert answer.decode() == f"trial-1 {relay.answer(TOKEN, 'n0', 'trial-1')}"
        assert rest == request
        assert TOKEN.encode() not in sent
    finally:
        gateway.close()


@pytest.mark.parametrize(
    "challenge", [b"FPEG/1 n0\n", b"BFEG/1\n", b"BFEG/1 a b\n", b""]
)
def test_a_bad_challenge_closes_the_client(run_relay, challenge):
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    gateway = StandInGateway(challenge=challenge or b"x" * 300)
    try:
        port = run_relay(gateway)
        assert _exchange(port, b"GET /healthz HTTP/1.1\r\nHost: h\r\n\r\n") == b""
        assert all(TOKEN.encode() not in sent for sent in gateway.received)
    finally:
        gateway.close()


@pytest.fixture
def model_gateway():
    class Gateway(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"model gateway response"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_loopback_connect_is_served_in_the_sandbox(run_relay, model_gateway, host):
    """Undici's CONNECT to the in-sandbox model gateway, through HTTPS_PROXY,
    never goes to the gateway (there 127.0.0.1 is the gateway host's own).

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    gateway = StandInGateway()
    try:
        port = run_relay(gateway)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.set_tunnel(host, model_gateway)
        conn.request("GET", "/model")
        response = conn.getresponse()
        assert response.status == 200
        assert response.read() == b"model gateway response"
        conn.close()
        assert gateway.received == []
    finally:
        gateway.close()


def test_other_connects_go_to_the_gateway(run_relay):
    """A CONNECT to the relay's own port, a privileged loopback port or a
    public host is the gateway's to judge.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    gateway = StandInGateway(reply=b"HTTP/1.1 403 Forbidden\r\n\r\n")
    try:
        port = run_relay(gateway)
        for target in (f"127.0.0.1:{port}", "127.0.0.1:22", "example.org:443"):
            request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()
            assert _exchange(port, request).startswith(b"HTTP/1.1 403")
        assert len(gateway.received) == 3
    finally:
        gateway.close()


def test_the_relay_and_the_gateway_speak_the_same_protocol(run_relay):
    """End to end on loopback: the relay, a real gateway and BenchFlow's
    proxy behind it, in the deny phase.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    import base64
    import shutil
    import tempfile
    from types import SimpleNamespace

    from benchflow.sandbox import egress_gateway_server as server
    from benchflow.sandbox.egress_denylist import EgressDenylist, _runtime_files

    folder = Path(tempfile.mkdtemp(prefix="bfeg-"))
    gateway = server.Gateway(
        folder / "state",
        ("127.0.0.1", 0),
        "127.0.0.1:1",
        openssl=shutil.which("openssl") or "/usr/bin/false",
    )
    gateway.open()
    try:
        files = _runtime_files(EgressDenylist((), ("blocked.test",)), None)
        policy = json.loads(files["policy.json"])
        for payload in (
            {
                "command": "register",
                "rollout": "trial-1",
                "token": TOKEN,
                "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
            },
            {"command": "deny", "rollout": "trial-1", "policy": policy},
        ):
            assert server.request(gateway.control_path, payload, timeout=120)["ok"]
        port = run_relay(SimpleNamespace(port=gateway.address[1]))
        health = _exchange(port, b"GET /healthz HTTP/1.1\r\nHost: h\r\n\r\n")
        assert health.startswith(b"HTTP/1.1 200")
        refused = _exchange(
            port, b"CONNECT blocked.test:443 HTTP/1.1\r\nHost: blocked.test:443\r\n\r\n"
        )
        assert (
            refused.startswith(b"HTTP/1.1 403") and b"X-BenchFlow-Blocked: 1" in refused
        )
        other = run_relay(SimpleNamespace(port=gateway.address[1]), rollout="trial-2")
        assert _exchange(other, b"GET /healthz HTTP/1.1\r\nHost: h\r\n\r\n") == b""
    finally:
        gateway.close()
        shutil.rmtree(folder, ignore_errors=True)


def test_loopback_port_reads_only_a_loopback_connect():
    """Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    assert relay.loopback_port(b"CONNECT 127.0.0.1:4000 HTTP/1.1\r\n", 18628) == 4000
    assert relay.loopback_port(b"CONNECT localhost:4000 HTTP/1.0\r\n", 18628) == 4000
    for head in (
        b"CONNECT 127.0.0.1:18628 HTTP/1.1\r\n",
        b"CONNECT 127.0.0.1:80 HTTP/1.1\r\n",
        b"CONNECT 127.0.0.2:4000 HTTP/1.1\r\n",
        b"CONNECT 127.0.0.1.example.org:4000 HTTP/1.1\r\n",
        b"GET http://127.0.0.1:4000/ HTTP/1.1\r\n",
        b"CONNECT 127.0.0.1:70000 HTTP/1.1\r\n",
    ):
        assert relay.loopback_port(head, 18628) is None, head


def test_the_relay_runs_on_python_38_with_the_standard_library_alone():
    """Task images bring their own python3; the relay is uploaded as a file.

    Guards PR #1253 (adds ``BENCHFLOW_EGRESS_GATEWAY``)."""
    text = SOURCE.read_text()
    tree = ast.parse(text, feature_version=(3, 8))
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
