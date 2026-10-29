"""The model gateway guard: in a no-web or denylist run, LiteLLM serves the agent model calls only.

Guards no-web and denylist egress (PR #1113) against the model gateway, which
the agent can call directly: provider pass-through, file and batch routes,
WebSocket sessions, ``extra_body`` and hosted-fetch requests all let the
provider reach the web outside every sandbox rule.
"""

from __future__ import annotations

import gzip
import json

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from benchflow.providers import litellm_egress_guard as guard_mod


async def _echo(scope, receive, send):
    """Stands in for LiteLLM: answers with the path and the body it received."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    payload = json.dumps({"path": scope["path"], "body": body.decode()}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": payload})


@pytest.fixture
def gateway(tmp_path):
    block_log = tmp_path / "blocked.jsonl"
    block_log.write_text("")
    client = TestClient(
        guard_mod.ModelEndpointGuard(_echo, block_log=str(block_log)),
        base_url="http://127.0.0.1:4000",
    )
    client.block_log = block_log
    return client


def _logged(gateway) -> list[dict]:
    return [json.loads(line) for line in gateway.block_log.read_text().splitlines()]


ORDINARY = {"model": "m", "input": "hi", "tools": [{"type": "function", "name": "sh"}]}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/responses"),
        ("POST", "/responses"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("POST", "/gemini/v1beta/models/gemini-x:streamGenerateContent"),
        ("POST", "/gemini/v1beta/models/gemini-x:countTokens"),
        ("GET", "/v1/models"),
        ("GET", "/health/liveliness"),
    ],
)
def test_model_routes_are_served_and_marked(gateway, method, path):
    response = gateway.request(
        method, path, json=ORDINARY if method == "POST" else None
    )

    assert response.status_code == 200
    assert response.headers["x-benchflow-egress-guard"] == "1"
    if method == "POST":
        assert json.loads(response.json()["body"]) == ORDINARY
    assert _logged(gateway) == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/files"),
        ("GET", "/v1/batches"),
        ("POST", "/v1/messages/batches"),
        ("POST", "/anthropic/v1/messages"),
        ("POST", "/openai/v1/responses"),
        ("POST", "/azure/openai/deployments/m/chat/completions"),
        ("POST", "/gemini/v1beta/models/gemini-x:batchGenerateContent"),
        ("POST", "/gemini/upload/v1beta/files"),
        ("POST", "/v1/realtime"),
        ("DELETE", "/v1/responses"),
    ],
)
def test_every_other_route_is_refused_and_logged(gateway, method, path):
    response = gateway.request(method, path, json=ORDINARY)

    assert response.status_code == 403
    assert response.headers["x-benchflow-blocked"] == "1"
    assert "gateway-route" in response.json()["error"]["message"]
    entry = _logged(gateway)[-1]
    assert (entry["action"], entry["method"], entry["rule"]) == (
        "blocked",
        method,
        "gateway-route",
    )
    assert entry["url"] == f"http://127.0.0.1:4000{path}"


@pytest.mark.parametrize(
    ("path", "body", "rule"),
    [
        (
            "/gemini/v1beta/models/gemini-x:generateContent",
            {"contents": [], "tools": [{"url_context": {}}]},
            "hosted-fetch:tool:url_context",
        ),
        (
            "/v1/responses",
            {
                "model": "m",
                "input": "hi",
                "extra_body": {"model": "gpt-4o-search-preview"},
            },
            "hosted-fetch:extra_body",
        ),
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [], "plugins": [{"id": "web"}]},
            "hosted-fetch:plugins",
        ),
        (
            # A client-chosen type must not hide the request's own input.
            "/v1/responses",
            {
                "type": "tool_use",
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_url": "https://x.test/p.pdf"}
                        ],
                    }
                ],
            },
            "hosted-fetch:input:input_file",
        ),
        (
            # LiteLLM turns a URL file_id into a URL source the provider fetches.
            "/v1/responses",
            {
                "model": "m",
                "input": [{"type": "input_file", "file_id": "https://x.test/p.pdf"}],
            },
            "hosted-fetch:input:input_file",
        ),
    ],
)
def test_hosted_fetch_requests_never_reach_litellm(gateway, path, body, rule):
    response = gateway.post(path, json=body)

    assert response.status_code == 403
    assert rule in response.json()["error"]["message"]
    assert _logged(gateway)[-1]["rule"] == rule


def test_content_coded_bodies_are_read_or_refused(gateway):
    hosted = json.dumps({"model": "m", "tools": [{"type": "web_search"}]}).encode()
    coded = gateway.post(
        "/v1/responses",
        content=gzip.compress(hosted),
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    unknown = gateway.post(
        "/v1/responses",
        content=b"\x28\xb5\x2f\xfd",
        headers={"content-encoding": "zstd", "content-type": "application/json"},
    )

    assert coded.status_code == 403
    assert [entry["rule"] for entry in _logged(gateway)] == [
        "hosted-fetch:tool:web_search",
        "uninspectable-body",
    ]
    assert unknown.status_code == 403


@pytest.mark.parametrize(
    "content_type",
    ["application/x-www-form-urlencoded", "multipart/form-data; boundary=x"],
)
def test_form_bodies_are_refused(gateway, content_type):
    """LiteLLM reads a form body field by field, not as the JSON the guard judges."""
    response = gateway.post(
        "/v1/responses",
        content=b"model=m&input=hi",
        headers={"content-type": content_type},
    )

    assert response.status_code == 403
    assert _logged(gateway)[-1]["rule"] == "uninspectable-body"


def test_oversized_body_is_refused(gateway, monkeypatch):
    monkeypatch.setattr(guard_mod, "BODY_INSPECT_LIMIT", 16)

    response = gateway.post("/v1/responses", json=ORDINARY)

    assert response.status_code == 403
    assert _logged(gateway)[-1]["rule"] == "oversized-body"


def test_websocket_sessions_are_refused(gateway):
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        gateway.websocket_connect("/v1/responses"),
    ):
        pass

    assert closed.value.code == 1008
    assert _logged(gateway)[-1]["rule"] == "gateway-websocket"


def test_refusal_stands_without_a_block_log():
    """A no-web run has no denylist proxy, so the guard has nowhere to log."""
    client = TestClient(guard_mod.ModelEndpointGuard(_echo, block_log=None))

    assert client.post("/v1/files", json={}).status_code == 403


def test_install_turns_off_litellm_image_downloads(monkeypatch):
    """LiteLLM's own downloads leave the sandbox outside the egress proxy."""
    monkeypatch.delenv("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", raising=False)
    finders = list(guard_mod.sys.meta_path)
    try:
        guard_mod.install(block_log=None)
        assert guard_mod.os.environ["MAX_IMAGE_URL_DOWNLOAD_SIZE_MB"] == "0"
        assert isinstance(guard_mod.sys.meta_path[0], guard_mod._GuardOnImport)
    finally:
        guard_mod.sys.meta_path[:] = finders
