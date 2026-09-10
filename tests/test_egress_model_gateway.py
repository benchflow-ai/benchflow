"""Guard the model-gateway proxy regression introduced by PR #1113.

Gemini 0.42 uses Undici ProxyAgent, which CONNECTs even to the HTTP model
endpoint and ignores NO_PROXY. Exercise that wire protocol with real sockets.
"""

from __future__ import annotations

import http.client
import http.server
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy
from benchflow.sandbox.egress_denylist import EgressDenylist, start_egress_denylist


@pytest.fixture
def gateway_stack(tmp_path: Path):
    class Gateway(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"model gateway response"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    gateway = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    other = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    log = tmp_path / "blocked.jsonl"
    server = proxy.serve(
        0,
        proxy.Policy(["paper.test/abs/2401.12345"], [], gateway.server_port),
        proxy.CertStore(str(tmp_path)),
        proxy.Log(str(log)),
    )
    servers = (gateway, other, server)
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1], gateway.server_port, other.server_port, log
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()


@pytest.mark.parametrize("tunnel", [True, False])
def test_only_controller_gateway_is_reachable(gateway_stack, tunnel):
    """Guards PR #1113 without weakening rejection of other private destinations."""
    proxy_port, gateway_port, other_port, log = gateway_stack
    for host, port, allowed in (
        ("127.0.0.1", gateway_port, True),
        ("127.0.0.1", other_port, False),
        ("localhost", other_port, False),
        ("169.254.169.254", 80, False),
    ):
        conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
        try:
            if tunnel:
                conn.set_tunnel(host, port)
                if not allowed:
                    with pytest.raises(OSError, match="403"):
                        conn.request("GET", "/model")
                    continue
                conn.request("GET", "/model")
            else:
                conn.request("GET", f"http://{host}:{port}/model")
            response = conn.getresponse()
            assert response.status == (200 if allowed else 403)
            if allowed:
                assert response.read() == b"model gateway response"
        finally:
            conn.close()

    conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
    try:
        conn.request("GET", "http://paper.test/abs/2401.12345v2")
        response = conn.getresponse()
        assert response.status == 403
        assert response.getheader("X-BenchFlow-Blocked") == "1"
    finally:
        conn.close()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 4
    assert {e["rule"] for e in events} == {
        "ip-literal",
        "private-address",
        "url:paper.test/abs/2401.12345",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:12345",
        "http://localhost:12345",
        "http://127.0.0.2:12345",
        "http://127.0.0.1:80",
        "http://127.0.0.1",
        "https://127.0.0.1:12345",
        "http://user@127.0.0.1:12345",
        "http://127.0.0.1:12345/path",
        "http://127.0.0.1:12345?endpoint=x",
    ],
)
async def test_invalid_runtime_gateway_fails_before_upload(url):
    """Guards the PR #1113 repair's controller/runtime boundary."""
    env = MagicMock(upload_file=AsyncMock(), exec=AsyncMock())
    with pytest.raises(ValueError, match="model gateway"):
        await start_egress_denylist(
            env, "agent", EgressDenylist((), ()), model_gateway_url=url
        )
    env.upload_file.assert_not_called()
    env.exec.assert_not_called()


async def test_gateway_is_written_to_root_owned_policy():
    """Guards PR #1113 by carrying the actual runtime port across the sandbox boundary."""
    policies = []

    async def upload(local, remote, **kwargs):
        if remote.endswith("/policy.json"):
            assert kwargs["mode"] == "600"
            policies.append(proxy.Policy.load(local))

    env = MagicMock()
    env.upload_file = AsyncMock(side_effect=upload)
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    env.exec_transient = env.exec
    await start_egress_denylist(
        env,
        "agent",
        EgressDenylist(("paper.test/abs/1",), ()),
        model_gateway_url="http://127.0.0.1:12345",
    )
    assert len(policies) == 1
    assert policies[0].host_rule("127.0.0.1", 12345) is None
    assert policies[0].host_rule("127.0.0.1", 12346) == "ip-literal"
