"""A real LiteLLM proxy, guarded as benchflow runs it for no-web and denylist rollouts.

Guards no-web and denylist egress (PR #1113) against the model gateway: the
agent can call it directly, and a provider-side fetch or search reaches pages
no sandbox rule sees. The guard wraps LiteLLM's app at import time, so this
also catches LiteLLM upgrades that move or rename it. Only loopback servers
are involved.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import yaml

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.sandbox.egress_denylist import EgressGuard

pytest.importorskip("litellm.proxy.proxy_server")

_MASTER_KEY = "sk-benchflow-test"
_PAPER = "https://arxiv.org/pdf/2401.12345"
_GEMINI = f"/gemini/v1beta/models/gemini-x:generateContent?key={_MASTER_KEY}"


class _Provider(http.server.BaseHTTPRequestHandler):
    """Answers chat, Responses and GenerateContent calls; records every request it gets."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.paths.append(self.path)
        if "generateContent" in self.path:
            reply = {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "ok"}]}}
                ]
            }
        elif self.path.endswith("/responses"):
            reply = {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "fake",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "ok", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        else:
            reply = {
                "id": "chat_1",
                "object": "chat.completion",
                "created": 1,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    provider.paths = []
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    provider_url = f"http://127.0.0.1:{provider.server_address[1]}"
    runtime_dir = tmp_path_factory.mktemp("litellm")
    config = {
        "model_list": [
            {
                "model_name": "fake",
                "litellm_params": {
                    "model": "openai/fake",
                    "api_base": f"{provider_url}/v1",
                    "api_key": "upstream",
                },
            }
        ],
        "general_settings": {"master_key": _MASTER_KEY},
        "litellm_settings": {
            "callbacks": ["benchflow_litellm_callback.proxy_handler_instance"],
            "drop_params": True,
            "use_chat_completions_url_for_anthropic_messages": True,
        },
    }
    block_log = runtime_dir / "blocked.jsonl"
    block_log.write_text("")
    guard = EgressGuard(block_log=str(block_log))
    for name, source in runtime_mod._runtime_modules(guard).items():
        (runtime_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (runtime_dir / name).write_text(source)
    config_path = runtime_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OPENAI", "ANTHROPIC", "GEMINI", "AZURE", "BENCHFLOW"))
        },
        "PYTHONPATH": str(runtime_dir),
        "GEMINI_API_KEY": "upstream",
        "GEMINI_API_BASE": provider_url,
        **runtime_mod._PROXY_DOCS_DISABLE_ENV,
    }
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            runtime_mod._host_litellm_executable(),
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=runtime_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while True:
        try:
            urllib.request.urlopen(f"{base}/health/liveliness", timeout=2).close()
            break
        except OSError:
            if process.poll() is not None or time.monotonic() > deadline:
                process.kill()
                pytest.fail("LiteLLM proxy did not start")
            time.sleep(0.5)
    yield SimpleNamespace(
        base=base, port=port, paths=provider.paths, block_log=block_log
    )
    process.terminate()
    process.wait(timeout=30)
    provider.shutdown()
    provider.server_close()


def _call(gateway, path: str, body: dict | None = None) -> tuple[int, str, dict]:
    request = urllib.request.Request(
        gateway.base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_MASTER_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), dict(error.headers)


def _user(content) -> list[dict]:
    return [{"role": "user", "content": content}]


@pytest.mark.parametrize(
    ("path", "body", "rule"),
    [
        (
            "/v1/responses",
            {"model": "fake", "input": "hi", "tools": [{"type": "web_search"}]},
            "hosted-fetch:tool:web_search",
        ),
        (
            "/v1/responses",
            {
                "model": "fake",
                "input": _user([{"type": "input_file", "file_url": _PAPER}]),
            },
            "hosted-fetch:input:input_file",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake",
                "messages": _user(
                    [{"type": "image_url", "image_url": {"url": f"{_PAPER}.png"}}]
                ),
            },
            "hosted-fetch:input:image_url",
        ),
        (
            "/v1/messages",
            {
                "model": "fake",
                "max_tokens": 8,
                "messages": _user(
                    [{"type": "document", "source": {"type": "url", "url": _PAPER}}]
                ),
            },
            "hosted-fetch:input:document",
        ),
        (
            _GEMINI,
            {
                "contents": [{"parts": [{"text": _PAPER}]}],
                "tools": [{"url_context": {}}],
            },
            "hosted-fetch:tool:url_context",
        ),
    ],
    ids=[
        "responses-search",
        "responses-file-url",
        "chat-image-url",
        "messages-url",
        "gemini",
    ],
)
def test_hosted_fetch_never_reaches_the_provider(gateway, path, body, rule):
    before = len(gateway.paths)

    status, text, _headers = _call(gateway, path, body)

    assert status == 403
    assert rule in text
    assert len(gateway.paths) == before
    logged = [json.loads(line) for line in gateway.block_log.read_text().splitlines()]
    assert logged[-1]["rule"] == rule


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/responses", {"model": "fake", "input": "hi"}),
        ("/v1/chat/completions", {"model": "fake", "messages": _user("hi")}),
        ("/v1/messages", {"model": "fake", "max_tokens": 8, "messages": _user("hi")}),
        (_GEMINI, {"contents": [{"parts": [{"text": "hi"}]}]}),
    ],
    ids=["responses", "chat", "messages", "gemini"],
)
def test_ordinary_requests_still_reach_the_provider(gateway, path, body):
    before = len(gateway.paths)

    status, _text, headers = _call(gateway, path, body)

    assert status == 200
    assert headers["x-benchflow-egress-guard"] == "1"
    assert len(gateway.paths) == before + 1


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/anthropic/v1/messages/batches",
            {"requests": [{"params": {"tools": [{"type": "web_fetch_20250910"}]}}]},
        ),
        (
            f"/gemini/v1beta/models/gemini-x:batchGenerateContent?key={_MASTER_KEY}",
            {"batch": {"input_config": {"requests": {"requests": []}}}},
        ),
        ("/v1/files", {"purpose": "batch"}),
        ("/openai/v1/responses", {"model": "fake", "input": "hi"}),
    ],
    ids=["anthropic-batches", "gemini-batch", "files", "openai-pass-through"],
)
def test_routes_beyond_model_calls_are_refused(gateway, path, body):
    before = len(gateway.paths)

    status, text, headers = _call(gateway, path, body)

    assert (status, headers["x-benchflow-blocked"]) == (403, "1")
    assert "gateway-route" in text
    assert len(gateway.paths) == before


def test_extra_body_cannot_swap_the_model(gateway):
    body = {"model": "fake", "input": "hi", "extra_body": {"model": "search-preview"}}

    status, text, _headers = _call(gateway, "/v1/responses", body)

    assert status == 403
    assert "hosted-fetch:extra_body" in text


def test_websocket_sessions_are_refused(gateway):
    with socket.create_connection(("127.0.0.1", gateway.port), timeout=10) as sock:
        sock.sendall(
            (
                "GET /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                f"Authorization: Bearer {_MASTER_KEY}\r\n\r\n"
            ).encode()
        )
        status_line = sock.recv(4096).split(b"\r\n", 1)[0]

    assert b" 403 " in status_line + b" "
