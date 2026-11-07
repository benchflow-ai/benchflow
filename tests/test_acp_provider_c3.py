"""Guards the fix at f4324120."""

import httpx
import pytest
from fastapi import FastAPI, Request

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_request_guard import SingleRouteGuard


@pytest.fixture
def boundary(monkeypatch):
    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {
            "path": request.url.path,
            "body": await request.json(),
            "content_type": request.headers.get("content-type"),
        }

    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["allowed", "gemini/gemini-2.5-flash"]'
    )
    app.add_middleware(SingleRouteGuard)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,tool",
    [
        ("/v1/responses", {"type": "web_search_preview"}),
        ("/responses", {"type": "web_search"}),
        ("/v1/responses", {"type": "web_search_preview_2025_03_11"}),
        ("/v1/responses", {"type": "mcp", "server_url": "https://dummy"}),
        ("/v1/responses", {"type": "future_hosted_tool"}),
        (
            "/v1/responses",
            {"type": "namespace", "name": "n", "tools": [{"type": "web_search"}]},
        ),
        ("/v1/messages", {"type": "web_search_20250305", "name": "web_search"}),
        ("/v1/messages", {"type": "web_fetch_20250910", "name": "web_fetch"}),
        ("/v1/chat/completions", {"type": "web_search"}),
        (
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            {"googleSearch": {}},
        ),
        (
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            {"google_search_retrieval": {}},
        ),
        ("/gemini/v1beta/models/gemini-2.5-flash:generateContent", {"urlContext": {}}),
    ],
)
async def test_restricted_gateway_rejects_hosted_tools(
    boundary, monkeypatch, path, tool
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json={"tools": [tool]})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_unrestricted_gateway_preserves_hosted_search(boundary, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "0")
    tool = {"type": "web_search_preview"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post("/v1/responses", json={"tools": [tool]})
    assert response.status_code == 200
    assert response.json()["body"]["tools"] == [tool]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,tool",
    [
        ("/v1/responses", {"type": "custom", "name": "apply_patch"}),
        ("/v1/responses", {"type": "tool_search", "execution": "client"}),
        (
            "/responses",
            {
                "type": "namespace",
                "name": "n",
                "tools": [
                    {
                        "type": "function",
                        "name": "web_search",
                        "parameters": {"description": "web_search"},
                    }
                ],
            },
        ),
        ("/v1/messages", {"name": "web_fetch", "input_schema": {"type": "object"}}),
        (
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            {"functionDeclarations": [{"name": "googleSearch"}]},
        ),
    ],
)
async def test_restricted_gateway_preserves_agent_executed_tools(
    boundary, monkeypatch, path, tool
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json={"tools": [tool]})
    assert response.status_code == 200
    assert response.json()["body"]["tools"] == [tool]


@pytest.mark.asyncio
async def test_nested_native_count_request_obeys_web_policy(boundary, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/gemini/v1beta/models/gemini-2.5-flash:countTokens",
            json={"generateContentRequest": {"tools": [{"googleSearch": {}}]}},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "remote-docker"])
async def test_web_policy_is_controller_owned_and_changes_reuse(
    monkeypatch, environment
):
    from tests.test_litellm_runtime import FakeLiteLLMServer

    observed = []

    async def start(**kwargs):
        observed.append(kwargs["agent_env"]["BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"])
        return FakeLiteLLMServer("http://dummy", kwargs["route"])

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    current = None
    for disallow in [False, True, True, False]:
        _, current = await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            model="openai/gpt-4.1",
            agent_env={
                "OPENAI_API_KEY": "dummy",
                "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS": "0",
            },
            runtime=current,
            environment=environment,
            sandbox=object(),
            disallow_web_tools=disallow,
        )
    assert observed == ["0", "1", "0"]
