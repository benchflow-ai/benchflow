"""Text/content confusion left open by cd73d4d3."""

import copy
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from benchflow.providers import litellm_runtime as runtime
from tests.test_litellm_hardening import _FakeSandbox


@pytest_asyncio.fixture(params=["host", "sandbox"])
async def boundary(request, tmp_path):
    # The integration base has no guard; exercise its actual unguarded route
    # too, rather than failing the origin check on a missing module import.
    if request.param == "host":
        runtime._write_runtime_files(tmp_path, config={})
        guard = tmp_path / "benchflow_litellm_request_guard.py"
        source = guard.read_text() if guard.exists() else None
    else:
        sandbox = _FakeSandbox()
        paths = await runtime._upload_runtime_files_to_sandbox(
            sandbox, runtime_dir="/tmp/dummy-runtime", config={}
        )
        source = sandbox.uploaded.get(paths.get("request_guard", ""))
    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {"body": await request.json()}

    if source is not None:
        namespace = {}
        exec(compile(source, "request_guard.py", "exec"), namespace)
        app.add_middleware(namespace["SingleRouteGuard"])
    return app


async def post(boundary, path, body):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=boundary), base_url="http://dummy"
    ) as client:
        payload = body if path.startswith("/gemini/") else {"model": "allowed", **body}
        return await client.post(path, json=payload)


BAD_TEXT = [
    [{"type": "image_url", "image_url": {"url": "http://dummy.invalid/marker"}}],
    {"type": "input_image", "image_url": "https://dummy.invalid/marker"},
    [{"type": "input_file", "file_url": "https://dummy.invalid/marker"}],
    [{"type": "text", "text": "ordinary nested text"}],
    [],
    {},
    True,
    1,
    1.5,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
@pytest.mark.parametrize("instructions", BAD_TEXT)
async def test_instructions_cannot_be_reinterpreted_as_content(
    boundary, monkeypatch, path, instructions
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    response = await post(
        boundary, path, {"input": "describe", "instructions": instructions}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
@pytest.mark.parametrize(
    "instructions", [None, "", "Describe http://dummy.invalid/marker as text."]
)
async def test_literal_instructions_survive(boundary, monkeypatch, path, instructions):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    response = await post(
        boundary, path, {"input": "describe", "instructions": instructions}
    )
    assert response.status_code == 200
    assert response.json()["body"]["instructions"] == instructions


def textual_carriers():
    # Text fields that translators can copy into another protocol's content.
    for path in ("/responses", "/v1/responses"):
        for kind, field in (
            ("input_text", "text"),
            ("output_text", "text"),
            ("refusal", "refusal"),
            ("reasoning_text", "text"),
            ("summary_text", "text"),
        ):
            for carrier in (
                "message",
                "function_call_output",
                "custom_tool_call_output",
                "reasoning",
            ):
                slot = (
                    "output"
                    if carrier.endswith("_output")
                    else "summary"
                    if carrier == "reasoning"
                    else "content"
                )
                yield (
                    path,
                    {
                        "input": [
                            {
                                "type": carrier,
                                "role": "user",
                                slot: [{"type": kind, field: "text"}],
                            }
                        ]
                    }
                    if carrier == "message"
                    else {
                        "input": [
                            {"type": carrier, slot: [{"type": kind, field: "text"}]}
                        ]
                    },
                    ("input", 0, slot, 0, field),
                )
        for kind, field in (
            ("function_call", "arguments"),
            ("custom_tool_call", "input"),
            ("reasoning", "encrypted_content"),
        ):
            yield path, {"input": [{"type": kind, field: "text"}]}, ("input", 0, field)
    for path in (
        "/chat/completions",
        "/v1/chat/completions",
        "/messages",
        "/v1/messages",
        "/v1/messages/count_tokens",
    ):
        for role in ("user", "system", "assistant", "tool"):
            yield (
                path,
                {
                    "messages": [
                        {"role": role, "content": [{"type": "text", "text": "text"}]}
                    ]
                },
                ("messages", 0, "content", 0, "text"),
            )
        for field in ("refusal", "reasoning_content"):
            yield (
                path,
                {"messages": [{"role": "assistant", field: "text"}]},
                ("messages", 0, field),
            )
        if "messages" in path:
            for field in ("text", "thinking", "signature"):
                part = {
                    "type": "text" if field == "text" else "thinking",
                    field: "text",
                }
                yield path, {"system": [part]}, ("system", 0, field)
                yield (
                    path,
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "tool_result", "content": [part]}],
                            }
                        ]
                    },
                    ("messages", 0, "content", 0, "content", 0, field),
                )
    for version in ("v1", "v1beta"):
        for operation in ("generateContent", "streamGenerateContent", "countTokens"):
            path = f"/gemini/{version}/models/gemini-2.5-flash:{operation}"
            for body, keys in (
                (
                    {"contents": [{"parts": [{"text": "text"}]}]},
                    ("contents", 0, "parts", 0, "text"),
                ),
                (
                    {"systemInstruction": {"parts": [{"text": "text"}]}},
                    ("systemInstruction", "parts", 0, "text"),
                ),
                (
                    {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "functionResponse": {
                                            "name": "dummy",
                                            "response": {},
                                            "parts": [{"text": "text"}],
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    ("contents", 0, "parts", 0, "functionResponse", "parts", 0, "text"),
                ),
            ):
                yield path, body, keys
                yield (
                    path,
                    {"generateContentRequest": body},
                    ("generateContentRequest", *keys),
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body,keys", list(textual_carriers()))
@pytest.mark.parametrize("structured", [False, True])
async def test_textual_content_leaves_are_scalar(
    boundary, monkeypatch, path, body, keys, structured
):
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", '["gemini/gemini-2.5-flash"]')
    body = copy.deepcopy(body)
    leaf = body
    for key in keys[:-1]:
        leaf = leaf[key]
    leaf[keys[-1]] = BAD_TEXT[0] if structured else "http://dummy.invalid/literal"
    response = await post(boundary, path, body)
    assert response.status_code == (400 if structured else 200), response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
async def test_pinned_gemma_translation_never_fetches_instructions(
    boundary, monkeypatch, path
):
    from litellm.litellm_core_utils.prompt_templates import factory
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    fetched = []

    def dummy_fetch(url, **kwargs):
        fetched.append(url)
        return "data:image/png;base64,ZA=="

    monkeypatch.setattr(factory, "convert_url_to_base64", dummy_fetch)
    monkeypatch.setenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", "1")
    body = {"input": "describe", "instructions": BAD_TEXT[0]}
    response = await post(boundary, path, body)
    if response.status_code == 200:
        admitted = response.json()["body"]
        messages = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                admitted["input"], admitted
            )
        )
        _gemini_convert_messages_with_history(
            cast(Any, messages), model="gemma-3-27b-it", custom_llm_provider="gemini"
        )
    assert fetched == [], f"privileged fetches: {fetched}"
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
async def test_unrestricted_instructions_keep_wire_shape(boundary, monkeypatch, path):
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    response = await post(boundary, path, {"instructions": BAD_TEXT[0]})
    assert response.status_code == 200
    assert response.json()["body"]["instructions"] == BAD_TEXT[0]
