"""Guards the fix at f4324120."""

import httpx
import pytest
from fastapi import FastAPI, Request

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
    "path", ["/messages", "/v1/messages", "/v1/messages/count_tokens"]
)
async def test_claude_beta_and_user_metadata_are_protocol_data(boundary, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            path + "?beta=true",
            json={
                "model": "allowed",
                "messages": [],
                "metadata": {"user_id": "dummy-session"},
            },
        )
    assert response.status_code == 200
    assert response.json()["body"]["metadata"] == {"user_id": "dummy-session"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"api_base": "http://dummy"},
        {"user_id": {"api_key": "dummy"}},
        {"user_id": "ok", "headers": {}},
        "dummy",
    ],
)
async def test_claude_metadata_cannot_be_a_gateway_control(boundary, metadata):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post("/v1/messages", json={"metadata": metadata})
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "beta=false",
        "beta=true&beta=true",
        "beta=true&api_base=http://dummy",
        "key=dummy",
    ],
)
async def test_claude_query_does_not_admit_gateway_controls(boundary, query):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post("/v1/messages?" + query, json={"messages": []})
    assert response.status_code == 400
