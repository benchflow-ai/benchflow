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
    "path",
    [
        "/v1/responses",
        "/responses",
        "/v1/chat/completions",
        "/v1/messages",
        "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
    ],
)
@pytest.mark.parametrize(
    "media",
    [
        "application/x-www-form-urlencoded",
        "multipart/form-data; boundary=dummy",
        "text/plain",
        "",
    ],
)
async def test_only_json_parser_can_receive_validated_body(boundary, path, media):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            path,
            content=b'{"input":"&model=allowed&api_base=http://dummy&x="}',
            headers={"content-type": media},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_json_is_replayed_canonically(boundary):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/responses",
            content=b'{"input":"ok"}',
            headers={"content-type": "Application/JSON; charset=utf-8"},
        )
    assert response.status_code == 200
    assert response.json()["content_type"] == "application/json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [
            ("content-type", "application/json"),
            ("content-type", "application/x-www-form-urlencoded"),
        ],
        [("content-type", "application/json"), ("content-type", "application/json")],
        [("content-type", "application/json, application/x-www-form-urlencoded")],
    ],
)
async def test_ambiguous_media_cannot_choose_a_second_parser(boundary, headers):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/v1/responses", content=b'{"input":"ok"}', headers=headers
        )
    assert response.status_code == 400
