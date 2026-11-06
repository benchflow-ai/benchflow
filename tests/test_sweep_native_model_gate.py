"""Guards the fix at 363d815b: native paths carry model policy."""

import httpx
import pytest

from tests.test_sweep_request_controls import guarded_app as _guarded_app

guarded_app = _guarded_app


@pytest.mark.parametrize(
    "path",
    [
        "/gemini/v1beta/models/gemini-2.5-pro:generateContent",
        "/gemini/v1/models/gemini-2.5-pro:streamGenerateContent",
        "/gemini/v1beta/models/gemini-2.5-flash:delete",
        "/gemini/v1beta/models",
        "/gemini/v1beta/files",
        "/gemini/v1beta/models/%2567emini-2.5-flash:generateContent",
        "/v1beta/models/gemini-2.5-pro:generateContent",
        "/vertex_ai/projects/dummy/locations/global/models/other:generateContent",
        "/openai/v1/responses",
        "/anthropic/v1/messages",
    ],
)
@pytest.mark.asyncio
async def test_native_paths_outside_configured_model_are_refused(guarded_app, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json={"contents": []})
    assert response.status_code == 400


@pytest.mark.parametrize("version", ["v1", "v1beta"])
@pytest.mark.parametrize(
    "operation", ["generateContent", "streamGenerateContent", "countTokens"]
)
@pytest.mark.asyncio
async def test_configured_native_model_and_operations_pass(
    guarded_app, version, operation
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=guarded_app), base_url="http://dummy"
    ) as client:
        response = await client.post(
            f"/gemini/{version}/models/gemini-2.5-flash:{operation}?alt=sse",
            json={"contents": []},
        )
    assert response.status_code == 200
