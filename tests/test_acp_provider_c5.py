"""Guards the fix at f4324120."""

import json

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
@pytest.mark.parametrize("version", ["v1", "v1beta"])
@pytest.mark.parametrize(
    "operation", ["generateContent", "streamGenerateContent", "countTokens"]
)
async def test_native_vertex_uses_private_controller_project(
    boundary, monkeypatch, version, operation
):
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["vertex_ai/gemini-2.5-flash"]'
    )
    route = runtime.resolve_litellm_route(
        "google-vertex/gemini-2.5-flash",
        {
            "GOOGLE_CLOUD_PROJECT": "dummy-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
    )
    for k, v in runtime._native_passthrough_env(route, {}).items():
        monkeypatch.setenv(k, v)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            f"/gemini/{version}/models/gemini-2.5-flash:{operation}",
            json={"contents": []},
        )
    assert response.status_code == 200
    assert (
        response.json()["path"]
        == f"/vertex_ai/v1/projects/dummy-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:{operation}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project,location",
    [("", "global"), ("../dummy", "global"), ("dummy", "x/y"), ("dummy", "")],
)
async def test_native_vertex_refuses_unconfigured_resource(
    boundary, monkeypatch, project, location
):
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["vertex_ai/gemini-2.5-flash"]'
    )
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_VERTEX_NATIVE",
        json.dumps(
            {"model": "gemini-2.5-flash", "project": project, "location": location}
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": []},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["generateContent", "streamGenerateContent", "countTokens"]
)
@pytest.mark.parametrize("authentication", ["authorization", "query", "x-goog-api-key"])
async def test_pinned_vertex_handler_uses_proxy_adc(
    boundary, monkeypatch, operation, authentication
):
    """Exercise pinned vendor routing/auth with dummy ADC and no cloud I/O."""
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as vendor
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
        PassthroughEndpointRouter,
    )

    # The handler must consume controller config, then generate upstream auth;
    # forwarding the agent's dummy gateway header would fail this assertion.
    env = runtime._native_passthrough_env(
        runtime.resolve_litellm_route(
            "google-vertex/gemini-2.5-flash",
            {
                "GOOGLE_CLOUD_PROJECT": "dummy-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            },
        ),
        {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/dummy-private-adc.json"},
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["vertex_ai/gemini-2.5-flash"]'
    )
    router = PassthroughEndpointRouter()
    router.set_default_vertex_config()
    monkeypatch.setattr(vendor, "passthrough_endpoint_router", router)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    observed = {}

    async def auth(**kwargs):
        assert kwargs["api_key"] == "Bearer dummy-gateway"
        return UserAPIKeyAuth()

    async def token(self, **kwargs):
        observed["credentials"] = kwargs["credentials"]
        return "dummy-upstream-access", "dummy-project"

    def token_url(self, **kwargs):
        return "dummy-upstream-access", "unused"

    def forwarding(**kwargs):
        observed.update(kwargs)

        async def response(*args):
            return {"ok": True}

        return response

    monkeypatch.setattr(vendor, "user_api_key_auth", auth)
    monkeypatch.setattr(vendor.VertexBase, "_ensure_access_token_async", token)
    monkeypatch.setattr(vendor.VertexBase, "_get_token_and_url", token_url)
    monkeypatch.setattr(vendor, "create_pass_through_route", forwarding)
    app = FastAPI()

    @app.post("/vertex_ai/{endpoint:path}")
    async def vertex(endpoint: str, request: Request):
        from fastapi import Response

        return await vendor.vertex_proxy_route(
            endpoint, request, Response(), UserAPIKeyAuth()
        )

    app.add_middleware(SingleRouteGuard)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dummy"
    ) as client:
        path = f"/gemini/v1beta/models/gemini-2.5-flash:{operation}"
        headers = {authentication: "dummy-gateway"}
        if authentication == "query":
            path += "?key=dummy-gateway"
            headers = {}
        elif authentication == "authorization":
            headers = {"authorization": "Bearer dummy-gateway"}
        response = await client.post(
            path,
            headers=headers,
            json={"contents": []},
        )
    assert response.status_code == 200, response.text
    assert observed["credentials"] == "/tmp/dummy-private-adc.json"
    assert observed["custom_headers"]["Authorization"] == "Bearer dummy-upstream-access"
    assert (
        f"/v1/projects/dummy-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:{operation}"
        in observed["target"]
    )
    assert observed["is_streaming_request"] == (operation == "streamGenerateContent")
