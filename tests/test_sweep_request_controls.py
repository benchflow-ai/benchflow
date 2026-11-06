"""Guards the fix at 363d815b: client data is never routing."""

import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request, WebSocket

from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)
from benchflow.providers.litellm_runtime import (
    _sitecustomize_source,
    _write_runtime_files,
)


def _guarded(app, monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", SimpleNamespace(app=app)
    )
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["allowed", "gemini/gemini-2.5-flash"]'
    )
    config = litellm_proxy_config(
        resolve_litellm_route("gemini-2.5-flash", {"GEMINI_API_KEY": "dummy"}),
        master_key="dummy",
    )
    _write_runtime_files(tmp_path, config=config)
    monkeypatch.syspath_prepend(str(tmp_path))
    exec(_sitecustomize_source(config), {})
    return app


@pytest.fixture
def guarded_app(monkeypatch, tmp_path):
    app = FastAPI()

    @app.api_route("/{rest:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def upstream(request: Request):
        return {"forwarded": True}

    return _guarded(app, monkeypatch, tmp_path)


@pytest.fixture
def echo_app(monkeypatch, tmp_path):
    app = FastAPI()

    @app.api_route("/{rest:path}", methods=["POST"])
    async def upstream(request: Request):
        return {"headers": dict(request.headers)}

    return _guarded(app, monkeypatch, tmp_path)


@pytest.mark.asyncio
async def test_codex_subagent_label_reaches_proxy_callbacks(echo_app):
    """Guards the fix against 75ecbe8e2's header allowlist: it dropped
    x-openai-subagent, the Codex client's sub-agent label, so a proxy callback
    that pins sub-agents (FP's patch 21) saw every request as the main agent's.
    Forwarding and LiteLLM control headers are still dropped."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={
                "x-openai-subagent": "review",
                "x-litellm-timeout": "1",
                "x-forwarded-host": "dummy-routing",
                "openai-organization": "dummy-org",
                "api-key": "dummy-key",
            },
            json={"model": "allowed", "input": "hello"},
        )
    assert response.status_code == 200, response.text
    headers = response.json()["headers"]
    assert headers["x-openai-subagent"] == "review"
    for dropped in (
        "x-litellm-timeout",
        "x-forwarded-host",
        "openai-organization",
        "api-key",
    ):
        assert dropped not in headers


# All routing/credential namespaces are outside the wire-format allowlist,
# including new/unknown spellings and nested carriers LiteLLM may consume.
CONTROLS = [
    "api_base",
    "base_url",
    "api_key",
    "custom_llm_provider",
    "api_version",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "vertex_credentials",
    "organization",
    "router_settings",
    "model_list",
    "fallbacks",
    "litellm_params",
    "proxy_server_request",
    "extra_headers",
    "headers",
    "deployment_id",
    "credential_name",
    "future_routing_switch",
]


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/responses",
        "/responses",
        "/v1/messages",
    ],
)
@pytest.mark.parametrize("key", CONTROLS)
@pytest.mark.asyncio
async def test_client_routing_controls_are_refused(guarded_app, path, key):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            path, json={"model": "allowed", key: "dummy-routing-value"}
        )
    assert response.status_code == 400
    assert "dummy-routing-value" not in response.text


@pytest.mark.parametrize("carrier", ["extra_body", "metadata"])
@pytest.mark.asyncio
async def test_opaque_carriers_cannot_supply_routing(guarded_app, carrier):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "allowed", carrier: {"api_base": "http://dummy-routing"}},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_function_schema_data_is_not_mistaken_for_controls(guarded_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "allowed",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "api_key",
                            "parameters": {
                                "type": "object",
                                "properties": {"api_base": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_pinned_gateway_refuses_override_before_upstream_receives_key(tmp_path):
    """Exercise the installed LiteLLM HTTP server with dummy upstream credentials."""
    from benchflow.providers import litellm_runtime as runtime
    from tests.fixtures.mock_token_logprobs_server import start_server

    upstream = start_server()
    listener = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": upstream.base_url + "/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "dummy-upstream",
        "OPENAI_API_KEY": "dummy-upstream",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    route = resolve_litellm_route("vllm/mock-policy", env)
    proc = None
    try:
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="sweep",
            agent_name="codex-acp",
        )
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"Authorization": "Bearer dummy-gateway"}
            body = {
                "model": route.model_alias,
                "messages": [{"role": "user", "content": "dummy prompt"}],
            }
            control = await client.post(
                proc.base_url + "/v1/chat/completions", headers=headers, json=body
            )
            assert control.status_code == 200, control.text
            assert len(upstream.requests) == 1
            response = await client.post(
                proc.base_url + "/v1/chat/completions",
                headers=headers,
                json={**body, "api_base": listener.base_url + "/v1"},
            )
            assert response.status_code == 400, response.text
            assert not listener.requests
    finally:
        if proc:
            await proc.stop()
        upstream.shutdown()
        listener.shutdown()


@pytest.mark.asyncio
async def test_request_inspection_is_bounded(guarded_app, monkeypatch):
    guard = sys.modules.get("benchflow_litellm_request_guard")
    if guard is not None:
        monkeypatch.setattr(guard, "MAX_REQUEST_BYTES", 1024)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "allowed",
                "messages": [{"role": "user", "content": "a" * 2048}],
            },
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_native_responses_stream_options_are_protocol_data(guarded_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "allowed",
                "input": "hello",
                "stream": True,
                "stream_options": {"include_obfuscation": False},
            },
        )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "path", ["/v1/responses", "/responses", "/openai/v1/realtime", "/vertex_ai/live"]
)
def test_websocket_routes_cannot_bypass_http_protocol_policy(guarded_app, path):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    @guarded_app.websocket("/{rest:path}")
    async def upstream(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("forwarded")
        await websocket.close()

    with (
        TestClient(guarded_app) as client,
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(path) as websocket,
    ):
        assert websocket.receive_text() == "forwarded"


@pytest.mark.asyncio
async def test_codex_client_metadata_is_opaque_protocol_data(guarded_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "allowed",
                "input": "hello",
                "client_metadata": {
                    "api_base": "data only",
                    "x-codex-turn-metadata": "dummy",
                },
            },
        )
    assert response.status_code == 200, response.text
