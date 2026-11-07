"""Native Claude no-web proxy: request bodies may not enable Anthropic-executed tools.

Guards the fix for the model-only transport added with the native Claude OAuth reviewer transport, whose proxy
admitted any POST /v1/messages body. Server tools (web_search, web_fetch, ...),
MCP connector servers and URL image/document sources make Anthropic fetch the
web on the sandbox's behalf, so a no-web task could regain web access and
exfiltrate through them. Everything here runs a real proxy against a local
TLS origin standing in for api.anthropic.com; nothing contacts Anthropic.
"""

from __future__ import annotations

import http.client
import http.server
import json
import socket
import ssl
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import certificate_material

ORIGIN = "api.anthropic.com"
PATH = "/v1/messages?beta=true"
MESSAGE = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
CUSTOM_TOOL = {
    "name": "Bash",
    "description": "run a command",
    "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
}


def request_body(**fields) -> bytes:
    return json.dumps(
        {"model": "claude-test", "max_tokens": 16, "messages": [MESSAGE], **fields}
    ).encode()


class _Model(http.server.BaseHTTPRequestHandler):
    """Records forwarded requests and streams two SSE events."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.server.received.append((dict(self.headers), self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"event: message_start\ndata: {}\n\n")
        self.wfile.flush()
        self.server.release.wait(10)
        self.wfile.write(b"event: message_stop\ndata: {}\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def native(tmp_path: Path, monkeypatch):
    upstream_ca = certificate_material((ORIGIN,))
    proxy_ca = certificate_material((ORIGIN,))
    (tmp_path / "upstream-ca.crt").write_bytes(upstream_ca["ca.crt"])
    (tmp_path / "proxy-ca.crt").write_bytes(proxy_ca["ca.crt"])
    (tmp_path / "upstream.pem").write_bytes(upstream_ca[f"{ORIGIN}.pem"])
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / f"{ORIGIN}.pem").write_bytes(proxy_ca[f"{ORIGIN}.pem"])

    origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(tmp_path / "upstream.pem"))
    origin.socket = context.wrap_socket(origin.socket, server_side=True)
    origin.received = []
    origin.release = threading.Event()
    origin.release.set()
    connects = []

    def fake_connect_upstream(host, port, *, model_gateway_port=None):
        assert (host, port) == (ORIGIN, 443)
        connects.append(host)
        return socket.create_connection(("127.0.0.1", origin.server_address[1]))

    monkeypatch.setattr(proxy_mod, "_connect_upstream", fake_connect_upstream)
    monkeypatch.setattr(proxy_mod, "_resolve", lambda host, port: ["160.79.104.10"])
    log = tmp_path / "blocked.jsonl"
    server = proxy_mod.serve(
        0,
        proxy_mod.Policy([], [], native_claude_model_only=True),
        proxy_mod.CertStore(str(certs)),
        proxy_mod.Log(str(log)),
        upstream_ca=str(tmp_path / "upstream-ca.crt"),
    )
    for s in (origin, server):
        threading.Thread(
            target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
    client_ctx = ssl.create_default_context(cafile=str(tmp_path / "proxy-ca.crt"))

    def post(body, headers=None):
        conn = http.client.HTTPSConnection(
            "127.0.0.1", server.server_address[1], context=client_ctx, timeout=10
        )
        conn.set_tunnel(ORIGIN, 443)
        conn.request(
            "POST",
            PATH,
            body=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        return conn.getresponse()

    yield SimpleNamespace(
        post=post,
        received=origin.received,
        release=origin.release,
        connects=connects,
        log=log,
    )
    for s in (server, origin):
        s.shutdown()
        s.server_close()


def test_client_tool_request_is_forwarded_byte_for_byte(native):
    body = request_body(tools=[CUSTOM_TOOL, {**CUSTOM_TOOL, "type": "custom"}])
    response = native.post(body)
    assert response.status == 200
    assert response.read().count(b"event:") == 2
    headers, forwarded = native.received[0]
    assert forwarded == body
    assert headers["Content-Length"] == str(len(body))


def test_chunked_request_is_forwarded_with_its_length(native):
    body = request_body(tools=[CUSTOM_TOOL])
    response = native.post(iter([body[:7], body[7:]]))
    assert response.status == 200
    response.read()
    headers, forwarded = native.received[0]
    assert forwarded == body
    assert "Transfer-Encoding" not in headers


def test_ordinary_agent_history_is_not_mistaken_for_a_fetch(native):
    """Tool inputs are model data, and inline image/text sources are local."""
    history = [
        MESSAGE,
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t",
                    "name": "Bash",
                    "input": {"source": {"type": "url", "url": "https://x.test"}},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "notes",
                            },
                        },
                    ],
                }
            ],
        },
    ]
    response = native.post(request_body(messages=history, tools=[CUSTOM_TOOL]))
    assert response.status == 200
    response.read()
    assert len(native.received) == 1


def test_model_response_still_streams(native):
    native.release.clear()
    response = native.post(request_body())
    assert response.status == 200
    # The origin holds the second event until the first reaches the client.
    assert response.readline() == b"event: message_start\n"
    native.release.set()
    assert b"event: message_stop" in response.read()


@pytest.mark.parametrize(
    "body",
    [
        request_body(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
        request_body(
            tools=[CUSTOM_TOOL, {"type": "web_fetch_20250910", "name": "web_fetch"}]
        ),
        request_body(
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}]
        ),
        request_body(tools=[{"type": "advisor_20260301", "name": "advisor"}]),
        request_body(tools=[{"type": "mcp_toolset", "mcp_server_name": "x"}]),
        request_body(tools=[{**CUSTOM_TOOL, "type": None}]),
        request_body(tools={"web": {"type": "web_search_20250305"}}),
        request_body(tools=["web_search"]),
        request_body(
            mcp_servers=[{"type": "url", "url": "https://x.test/mcp", "name": "x"}]
        ),
        request_body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x.test/?q=1"},
                        }
                    ],
                }
            ]
        ),
        request_body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {"type": "url", "url": "https://x.test"},
                                }
                            ],
                        }
                    ],
                }
            ]
        ),
        request_body(
            system=[
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://x.test"},
                            }
                        ],
                    },
                }
            ]
        ),
        # Last-key-wins here, first-key-wins elsewhere: never guess which.
        b'{"tools":[{"type":"web_search_20250305","name":"web_search"}],"tools":[]}',
        b"not json",
        b"[]",
        b"",
        b'{"tools": [], "x": "\xff"}',
    ],
    ids=[
        "web-search",
        "web-fetch-beside-custom",
        "code-execution",
        "advisor",
        "mcp-toolset",
        "null-type",
        "tools-object",
        "tool-string",
        "mcp-servers",
        "image-url",
        "nested-document-url",
        "content-document-url",
        "duplicate-tools-key",
        "not-json",
        "json-array",
        "empty",
        "invalid-utf8",
    ],
)
def test_request_enabling_anthropic_side_fetch_is_refused(native, body):
    response = native.post(body)
    assert response.status == 403
    assert response.headers["X-BenchFlow-Blocked"] == "1"
    assert native.connects == []
    entry = json.loads(native.log.read_text().splitlines()[-1])
    assert entry["url"] == "native-model-only"
    assert entry["rule"].startswith("native-model-body")


def test_encoded_body_is_refused(native):
    response = native.post(request_body(), {"Content-Encoding": "gzip"})
    assert response.status == 403
    assert native.connects == []


@pytest.mark.parametrize("chunked", [False, True])
def test_oversized_body_is_refused_before_upstream(native, monkeypatch, chunked):
    monkeypatch.setattr(proxy_mod, "MODEL_BODY_LIMIT", 64)
    body = request_body(system="x" * 100)
    response = native.post(iter([body]) if chunked else body)
    assert response.status == 403
    assert native.connects == []
    assert json.loads(native.log.read_text())["rule"] == "native-model-body-size"


def test_declared_oversize_is_refused_without_reading_it(native, monkeypatch):
    monkeypatch.setattr(proxy_mod, "MODEL_BODY_LIMIT", 64)
    response = native.post(b"{}", {"Content-Length": str(10**9)})
    assert response.status == 403
    assert native.connects == []


def test_expect_continue_client_is_inspected_too(native):
    body = request_body(tools=[{"type": "web_search_20250305", "name": "web_search"}])
    response = native.post(body, {"Expect": "100-continue"})
    assert response.status == 403
    assert native.connects == []
