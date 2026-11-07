"""Remote content admitted by 93ac0317 at 585a8aea."""

import copy

import httpx
import pytest

from tests.test_acp_provider_c7 import boundary as boundary

INLINE = "data:image/png;base64,ZA=="
REMOTE = [
    "https://dummy.invalid/marker",
    "http://dummy.invalid/marker",
    "//dummy.invalid/marker",
    "HTTPS://dummy.invalid/marker",
    " https://dummy.invalid/marker",
    "gs://dummy/marker",
    "s3://dummy/marker",
    "file:///tmp/dummy-marker",
    "ftp://dummy.invalid/marker",
    "data:text/uri-list,https://dummy.invalid/marker",
    "data:image/png;url=https://dummy.invalid/marker",
]
PATHS = [
    "/responses",
    "/v1/responses",
    "/chat/completions",
    "/v1/chat/completions",
    "/messages",
    "/v1/messages",
    "/v1/messages/count_tokens",
] + [
    f"/gemini/{version}/models/gemini-2.5-flash:{operation}"
    for version in ("v1", "v1beta")
    for operation in ("generateContent", "streamGenerateContent", "countTokens")
]


def envelopes(path, url):
    if "responses" in path:
        for part in (
            {"type": "input_image", "image_url": url},
            {"type": "input_file", "file_url": url},
        ):
            yield {"input": [{"role": "user", "content": [part]}]}
            for kind in ("function_call_output", "custom_tool_call_output"):
                yield {"input": [{"type": kind, "call_id": "dummy", "output": [part]}]}
        yield {
            "input": [
                {
                    "type": "computer_call_output",
                    "call_id": "dummy",
                    "output": {"type": "computer_screenshot", "image_url": url},
                }
            ]
        }
    elif "chat/completions" in path:
        for role in ("user", "tool"):
            for part in (
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "file", "file": {"file_data": url}},
                {"type": "video_url", "video_url": {"url": url}},
            ):
                yield {"messages": [{"role": role, "content": [part]}]}
    elif "messages" in path:
        for kind in ("image", "document"):
            part = {"type": kind, "source": {"type": "url", "url": url}}
            yield {"messages": [{"role": "user", "content": [part]}]}
            yield {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "dummy",
                                "content": [part],
                            }
                        ],
                    }
                ]
            }
            yield {"system": [part]}
    else:
        for spelling, uri in (("fileData", "fileUri"), ("file_data", "file_uri")):
            part = {spelling: {uri: url, "mimeType": "image/png"}}
            for nested in (False, True):
                content = {"contents": [{"role": "user", "parts": [part]}]}
                yield {"generateContentRequest": content} if nested else content
            yield {"systemInstruction": {"parts": [part]}}
            yield {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "dummy",
                                    "response": {},
                                    "parts": [part],
                                }
                            }
                        ],
                    }
                ]
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("url", REMOTE)
async def test_all_remote_content_envelopes_refuse(boundary, monkeypatch, path, url):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        for body in envelopes(path, url):
            response = await client.post(path, json=body)
            assert response.status_code == 400, (path, body, response.text)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_inline_and_literal_data_survive(boundary, monkeypatch, path):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    literal = REMOTE[0]
    if "responses" in path:
        body = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": INLINE},
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,ZA==",
                            "filename": "dummy.pdf",
                        },
                        {"type": "input_text", "text": literal},
                    ],
                }
            ]
        }
    elif "chat/completions" in path:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": INLINE}},
                        {
                            "type": "file",
                            "file": {"file_data": "ZA==", "filename": "dummy.pdf"},
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "ZA==", "format": "wav"},
                        },
                        {"type": "text", "text": literal},
                    ],
                }
            ]
        }
    elif "messages" in path:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "dummy",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "ZA==",
                                    },
                                },
                                {"type": "text", "text": literal},
                            ],
                        }
                    ],
                }
            ]
        }
    else:
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": "ZA=="}},
                        {
                            "functionResponse": {
                                "name": "dummy",
                                "response": {"fileUri": literal},
                            }
                        },
                        {"text": literal},
                    ],
                }
            ]
        }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["body"] == body


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_unrestricted_remote_content_is_preserved(boundary, monkeypatch, path):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "0")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    body = copy.deepcopy(next(envelopes(path, REMOTE[0])))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["body"] == body


@pytest.mark.asyncio
async def test_pinned_gateway_never_forwards_remote_content():
    """Real pinned LiteLLM HTTP boundary, dummy loopback upstream only."""
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    upstream = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": upstream.base_url + "/v1",
        "OPENAI_API_KEY": "dummy-upstream",
        "BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    route = resolve_litellm_route("openai/gpt-4.1", env)
    proc = None
    try:
        proc = await runtime._start_host_litellm(
            route=route,
            master_key="dummy-gateway",
            agent_env=env,
            environment="local",
            session_id="c9",
            agent_name="codex-acp",
        )
        async with httpx.AsyncClient(timeout=30) as client:
            for path in PATHS[:7]:
                for body in envelopes(path, REMOTE[0]):
                    response = await client.post(
                        proc.base_url + path,
                        headers={"Authorization": "Bearer dummy-gateway"},
                        json={"model": route.model_alias, **body},
                    )
                    assert response.status_code == 400, response.text
            assert not upstream.requests
            control = await client.post(
                proc.base_url + "/v1/responses",
                headers={"Authorization": "Bearer dummy-gateway"},
                json={"model": route.model_alias, "input": "dummy prompt"},
            )
            assert control.status_code == 200, control.text
            assert len(upstream.requests) == 1
    finally:
        if proc is not None:
            await proc.stop()
        upstream.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "variant", ["unknown", "reference", "malformed", "extra_source"]
)
async def test_uninspected_content_forms_fail_closed(
    boundary, monkeypatch, path, variant
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    if "responses" in path:
        parts = {
            "unknown": {"type": "future_media", "url": REMOTE[0]},
            "reference": {"type": "input_image", "file_id": "dummy-file"},
            "malformed": {"type": "input_image", "image_url": {"url": INLINE}},
            "extra_source": {
                "type": "input_image",
                "image_url": INLINE,
                "url": REMOTE[0],
            },
        }
        body = {"input": [{"role": "user", "content": [parts[variant]]}]}
    elif "chat/completions" in path:
        parts = {
            "unknown": {"type": "future_media", "url": REMOTE[0]},
            "reference": {"type": "file", "file": {"file_id": "dummy-file"}},
            "malformed": {"type": "image_url", "image_url": [INLINE]},
            "extra_source": {
                "type": "image_url",
                "image_url": {"url": INLINE, "file_url": REMOTE[0]},
            },
        }
        body = {"messages": [{"role": "tool", "content": [parts[variant]]}]}
    elif "messages" in path:
        sources = {
            "unknown": {"type": "future_source", "url": REMOTE[0]},
            "reference": {"type": "file", "file_id": "dummy-file"},
            "malformed": {"type": "base64", "data": {"url": REMOTE[0]}},
            "extra_source": {"type": "base64", "data": "ZA==", "url": REMOTE[0]},
        }
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "document", "source": sources[variant]}],
                }
            ]
        }
    else:
        parts = {
            "unknown": {"futureMedia": {"url": REMOTE[0]}},
            "reference": {"fileData": {"fileUri": "dummy-file"}},
            "malformed": {"inlineData": {"data": ["ZA=="]}},
            "extra_source": {"inlineData": {"data": "ZA==", "fileUri": REMOTE[0]}},
        }
        body = {"contents": [{"parts": [parts[variant]]}]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_nested_inline_tool_outputs_and_sources_survive(
    boundary, monkeypatch, path
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    if "responses" in path:
        body = {
            "input": [
                {
                    "type": "computer_call_output",
                    "call_id": "dummy",
                    "output": {"type": "computer_screenshot", "image_url": INLINE},
                }
            ]
        }
    elif "chat/completions" in path:
        body = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "dummy",
                    "content": [{"type": "image_url", "image_url": {"url": INLINE}}],
                }
            ]
        }
    elif "messages" in path:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "dummy",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "content",
                                        "content": [
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": "image/png",
                                                    "data": "ZA==",
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    else:
        inner = {
            "contents": [
                {
                    "parts": [
                        {
                            "function_response": {
                                "name": "dummy",
                                "response": {"url": REMOTE[0]},
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": "ZA==",
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        body = {"generateContentRequest": inner}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["body"] == body


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS[7:])
@pytest.mark.parametrize(
    "inner",
    [
        {"cachedContent": "cachedContents/dummy"},
        {
            "generateContentRequest": {
                "contents": [{"parts": [{"fileData": {"fileUri": REMOTE[0]}}]}]
            }
        },
    ],
)
async def test_native_opaque_and_nested_envelopes_refuse(
    boundary, monkeypatch, path, inner
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        for body in (inner, {"generateContentRequest": inner}):
            response = await client.post(path, json=body)
            assert response.status_code == 400
