"""Guard the provider regressions reproduced on candidate ac3bedaa1."""

import copy

import httpx
import pytest

from benchflow.agents.env import resolve_provider_env
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.providers.litellm_request_guard import SingleRouteGuard


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")

    async def downstream(scope, receive, send):
        request = await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": request["body"]})

    return SingleRouteGuard(downstream)


def _messages_body(output_config):
    # Minimal shape of the pinned ACP 0.81.2 / CLI 2.1.280 capture. The raw
    # capture remains in release evidence rather than in the repository.
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Inspect the local file."},
            {
                "role": "system",
                "content": "Working directory: /app",
                "output_config": output_config,
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", None])
@pytest.mark.parametrize(
    "path", ["/v1/messages?beta=true", "/v1/messages/count_tokens"]
)
async def test_message_effort_survives_restricted_boundary(boundary, effort, path):
    """Guards the ac3bedaa1 rejection of the pinned client's message effort."""
    body = _messages_body({"effort": effort})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200
    assert response.json() == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_config",
    [
        None,
        "max",
        [],
        {"effort": {}},
        {"effort": ["max"]},
        {"effort": True},
        {"effort": "future"},
        {"effort": "max", "tools": [{"type": "web_search_20250305"}]},
        {"effort": "max", "api_base": "https://dummy.invalid"},
        {"format": {"type": "url", "url": "https://dummy.invalid"}},
    ],
)
async def test_message_effort_does_not_admit_opaque_carriers(boundary, output_config):
    """Guards ac3bedaa1's closed envelope while adding its effort exception."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post("/v1/messages", json=_messages_body(output_config))
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_effort_keeps_remote_content_and_hosted_tools_blocked(boundary):
    """Guards ac3bedaa1's network controls with the pinned effort envelope."""
    body = _messages_body({"effort": "max"})
    remote = copy.deepcopy(body)
    remote["messages"][0]["content"] = [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://dummy.invalid/image.png"},
        }
    ]
    hosted = dict(body, tools=[{"type": "web_search_20250305", "name": "web_search"}])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        for candidate in (remote, hosted):
            response = await client.post("/v1/messages", json=candidate)
            assert response.status_code == 400
        # The extension belongs to Messages, not Chat Completions.
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400


@pytest.mark.parametrize("model", ["mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro"])
def test_mimo_gateway_uses_completions_endpoint_for_bare_and_prefixed_ids(model):
    """Guards ac3bedaa1's bare MiMo OpenAI-to-Anthropic endpoint mismatch."""
    env = {
        "XIAOMI_API_KEY": "dummy-native",
        "XIAOMI_BASE_URL": "http://127.0.0.1:1/v1",
    }
    resolve_provider_env(env, model, "claude-agent-acp")
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "http://127.0.0.1:1/anthropic"
    route = resolve_litellm_route(model, env)
    assert route.provider_name == "xiaomi"
    assert route.upstream_model == "openai/mimo-v2.6-pro"
    assert route.litellm_params["api_base"] == "http://127.0.0.1:1/v1"
    assert route.litellm_params["api_key"] == "os.environ/XIAOMI_API_KEY"
    assert route.required_env == ("XIAOMI_API_KEY",)


@pytest.mark.parametrize("model", ["mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro"])
def test_mimo_explicit_proxy_route_stays_authoritative(model):
    """Guards ac3bedaa1's explicit routing contract during bare-ID resolution."""
    route = resolve_litellm_route(
        model,
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "https://proxy.example.test/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "dummy-explicit",
            "XIAOMI_API_KEY": "dummy-native",
            "XIAOMI_BASE_URL": "https://native.example.test/v1",
        },
    )
    assert route.litellm_params["api_base"] == "https://proxy.example.test/v1"
    assert route.litellm_params["api_key"] == "os.environ/BENCHFLOW_PROVIDER_API_KEY"
    assert route.required_env == ("BENCHFLOW_PROVIDER_API_KEY",)


def test_explicit_openai_model_is_not_reinterpreted_as_bare_provider():
    """Guards ac3bedaa1's explicit provider selection against family lookup."""
    route = resolve_litellm_route("openai/mimo-v2.6-pro", {"OPENAI_API_KEY": "dummy"})
    assert route.upstream_model == "openai/mimo-v2.6-pro"
    assert route.provider_name == "openai"
    assert route.litellm_params["api_base"] == "https://api.openai.com/v1"
    assert route.required_env == ("OPENAI_API_KEY",)
