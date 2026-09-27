"""Token ids and logprobs captured at the LiteLLM gateway (opt-in).

With ``BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`` the gateway requests sampled-token
logprobs, and token ids on routes that accept ``return_token_ids``, for chat
completions, Anthropic Messages and Responses calls, and stores them per call in
``llm_trajectory.jsonl`` as ``metadata.token_capture``
(``benchflow.token_capture.v1``). Fields a provider cannot return carry an
explicit ``unavailable`` reason. Before this, only chat-completions calls asked
for logprobs (#926), nothing asked for token ids, and LiteLLM's stream assembly
dropped both from streamed calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import benchflow.providers.litellm_runtime as runtime_mod
import benchflow.providers.litellm_token_capture_patch as patch
from benchflow.providers.litellm_config import LiteLLMRoute, resolve_litellm_route
from benchflow.providers.litellm_logging import (
    callback_module_source,
    trajectory_from_litellm_callback_log,
)
from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_SCHEMA_VERSION,
    build_token_capture,
    strip_captured_token_ids,
)
from tests.fixtures.mock_token_logprobs_server import (
    REPLY,
    logprob,
    start_server,
    token_id,
)

_CAPTURE_ENV = "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS"
_REPLY_IDS = [token_id(c) for c in REPLY]
_REPLY_LOGPROBS = [logprob(i) for i in range(len(REPLY))]


def _prompt_ids(text: str) -> list[int]:
    return [ord(c) for c in text]


# --------------------------------------------------------------------------- #
# End to end: real LiteLLM proxy (BenchFlow config, callback and patches)      #
# in front of a mock vLLM-compatible server.                                  #
# --------------------------------------------------------------------------- #

_CASES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "chat": ("/v1/chat/completions", "chat", {}),
    "chat_stream": ("/v1/chat/completions", "chat", {"stream": True}),
    "messages": ("/v1/messages", "messages", {"max_tokens": 16}),
    "messages_stream": (
        "/v1/messages",
        "messages",
        {"max_tokens": 16, "stream": True},
    ),
    "responses": ("/v1/responses", "responses", {}),
    "responses_bridge": ("/v1/responses", "bridge", {}),
    "responses_bridge_stream": ("/v1/responses", "bridge", {"stream": True}),
}


def _marker(case: str) -> str:
    """Unique prompt text per case; also the mock's prompt token ids."""
    return f"case-{case}"


def _request_body(case: str, alias: str) -> tuple[str, dict[str, str], dict]:
    path, kind, extra = _CASES[case]
    headers = {"Authorization": "Bearer sk-master"}
    if kind in {"chat", "messages"}:
        body = {
            "model": alias,
            "messages": [{"role": "user", "content": _marker(case)}],
        }
        if kind == "messages":
            headers["anthropic-version"] = "2023-06-01"
    else:
        model = f"{alias}-responses-bridge" if kind == "bridge" else alias
        body = {"model": model, "input": _marker(case)}
    return path, headers, {**body, **extra}


async def _drive_gateway(tmp_path: Path, route_model: str, cases: list[str]):
    server = start_server()
    env = {
        "BENCHFLOW_PROVIDER_BASE_URL": f"{server.base_url}/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "sk-mock",
        "OPENAI_API_KEY": "sk-mock",
        "ANTHROPIC_API_KEY": "sk-mock",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        _CAPTURE_ENV: "1",
    }
    if route_model.startswith("anthropic/"):
        route = LiteLLMRoute(
            requested_model=route_model,
            model_alias="benchflow-mock-claude",
            upstream_model=route_model,
            provider_name="native",
            litellm_params={
                "model": route_model,
                "api_base": server.base_url,
                "api_key": "os.environ/ANTHROPIC_API_KEY",
            },
        )
    else:
        route = resolve_litellm_route(route_model, env)
    proc = await runtime_mod._start_host_litellm(
        route=route,
        master_key="sk-master",
        agent_env=env,
        environment="local",
        session_id="s",
        agent_name="token-capture-test",
    )
    artifact = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    try:
        proc.start_live_capture(artifact)
        async with httpx.AsyncClient(timeout=60) as client:
            for case in cases:
                path, headers, body = _request_body(case, route.model_alias)
                response = await client.post(
                    proc.base_url + path, headers=headers, json=body
                )
                assert response.status_code == 200, (case, response.text)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.log_path.exists() and len(
                proc.log_path.read_text().splitlines()
            ) >= len(cases):
                break
            await asyncio.sleep(0.2)
    finally:
        await proc.stop()
        server.shutdown()
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    by_case = {}
    for record in records:
        request = json.dumps(record["request"]["body"])
        matches = [c for c in cases if f'"{_marker(c)}"' in request]
        assert len(matches) == 1, request
        by_case[matches[0]] = record
    return by_case, server.requests


@pytest.fixture(scope="module")
def vllm_gateway(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vllm-gateway")
    return asyncio.run(_drive_gateway(tmp, "vllm/mock-policy", list(_CASES)))


@pytest.mark.parametrize(
    "case",
    [
        "chat",
        "chat_stream",
        "messages",
        "messages_stream",
        "responses_bridge",
        "responses_bridge_stream",
    ],
)
def test_gateway_records_token_ids_and_logprobs(vllm_gateway, case):
    """Guards the gateway token-capture prototype: every call that reaches a
    vLLM-compatible chat backend (chat, Anthropic Messages and Responses via
    LiteLLM's bridges, streamed or not) records prompt ids, completion ids and
    logprobs in ``llm_trajectory.jsonl``. Streamed calls failed before because
    LiteLLM's stream assembly dropped these fields."""
    records, _ = vllm_gateway
    record = records[case]
    capture = record["metadata"]["token_capture"]

    assert capture["schema_version"] == TOKEN_CAPTURE_SCHEMA_VERSION
    assert capture["wire"] == "openai-chat"
    assert capture["requested"]["token_ids"] is True
    assert capture["unavailable"] == {}
    assert capture["prompt_token_ids"] == _prompt_ids(_marker(case))
    [completion] = capture["completions"]
    assert completion["token_ids"] == _REPLY_IDS
    assert completion["tokens"] == list(REPLY)
    assert completion["logprobs"] == _REPLY_LOGPROBS
    # Token ids are stored once, in token_capture, not again in the raw body.
    assert "prompt_token_ids" not in json.dumps(record["response"]["body"])


def test_gateway_native_responses_records_logprobs_and_no_ids_reason(vllm_gateway):
    """Guards the gateway token-capture prototype: a native Responses call asks
    for ``message.output_text.logprobs`` and records why token ids are absent."""
    records, upstream = vllm_gateway
    capture = records["responses"]["metadata"]["token_capture"]

    assert capture["wire"] == "openai-responses"
    assert capture["completions"][0]["logprobs"] == _REPLY_LOGPROBS
    assert capture["prompt_token_ids"] is None
    assert capture["unavailable"]["completion_token_ids"]["reason"] == (
        "provider_api_unsupported"
    )
    responses_calls = [r for r in upstream if r["path"].endswith("/responses")]
    assert responses_calls[0]["body"]["include"] == ["message.output_text.logprobs"]


def test_gateway_requests_reach_the_server_with_capture_fields(vllm_gateway):
    _, upstream = vllm_gateway
    chat_calls = [
        r["body"] for r in upstream if r["path"].endswith("/chat/completions")
    ]

    assert len(chat_calls) == 6
    assert all(body["logprobs"] is True for body in chat_calls)
    assert all(body["return_token_ids"] is True for body in chat_calls)


def test_gateway_anthropic_upstream_records_explicit_unavailable(tmp_path):
    """Guards the gateway token-capture prototype: an Anthropic Messages upstream
    has no logprobs or token ids, so each field is marked unavailable with a
    reason instead of being silently absent."""
    records, upstream = asyncio.run(
        _drive_gateway(tmp_path, "anthropic/mock-claude", ["messages"])
    )
    capture = records["messages"]["metadata"]["token_capture"]

    assert capture["wire"] == "anthropic-messages"
    assert {v["reason"] for v in capture["unavailable"].values()} == {
        "provider_api_unsupported"
    }
    assert set(capture["unavailable"]) == {
        "prompt_token_ids",
        "completion_token_ids",
        "logprobs",
    }
    assert "logprobs" not in upstream[0]["body"]


# --------------------------------------------------------------------------- #
# Request side: the proxy pre-call hook                                        #
# --------------------------------------------------------------------------- #


def _logger():
    namespace: dict[str, Any] = {}
    exec(callback_module_source(), namespace)
    return namespace["BenchFlowLiteLLMLogger"]()


def _route_env(monkeypatch, provider: str, upstream: str) -> None:
    monkeypatch.setenv(_CAPTURE_ENV, "1")
    monkeypatch.setenv("BENCHFLOW_LITELLM_ROUTE_PROVIDER", provider)
    monkeypatch.setenv("BENCHFLOW_LITELLM_UPSTREAM_MODEL", upstream)


def _hook(data: dict, call_type: str) -> dict | None:
    return asyncio.run(_logger().async_pre_call_hook(None, None, data, call_type))


def test_hook_is_inert_without_opt_in(monkeypatch):
    monkeypatch.delenv(_CAPTURE_ENV, raising=False)
    monkeypatch.setenv("BENCHFLOW_LITELLM_ROUTE_PROVIDER", "vllm")
    monkeypatch.setenv("BENCHFLOW_LITELLM_UPSTREAM_MODEL", "openai/qwen")

    for call_type, data in (
        ("acompletion", {"messages": [{"role": "user", "content": "hi"}]}),
        ("anthropic_messages", {"messages": [{"role": "user", "content": "hi"}]}),
        ("aresponses", {"model": "m", "input": "hi"}),
    ):
        assert _hook(data, call_type) is None


def test_hook_merges_token_ids_into_existing_extra_body(monkeypatch):
    _route_env(monkeypatch, "vllm", "openai/qwen")
    data = {
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }

    cleaned = _hook(data, "acompletion")

    assert cleaned is not None
    assert cleaned["logprobs"] is True
    assert cleaned["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
    }
    assert data["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


@pytest.mark.parametrize(
    ("provider", "setting", "expected"),
    [
        ("vllm", None, True),
        ("vllm", "0", False),
        ("openai", None, False),
        ("sglang-like", "1", True),
    ],
)
def test_hook_token_id_request_follows_route_and_override(
    monkeypatch, provider, setting, expected
):
    _route_env(monkeypatch, provider, "openai/qwen")
    if setting is None:
        monkeypatch.delenv("BENCHFLOW_CAPTURE_TOKEN_IDS", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_CAPTURE_TOKEN_IDS", setting)

    cleaned = _hook({"messages": [{"role": "user", "content": "hi"}]}, "completion")

    assert cleaned is not None
    assert cleaned["logprobs"] is True
    assert ("extra_body" in cleaned) is expected


def test_hook_leaves_native_anthropic_messages_untouched(monkeypatch):
    """Anthropic and Bedrock reject unknown fields and have no logprobs."""
    _route_env(monkeypatch, "aws-bedrock", "bedrock/us.anthropic.claude-fable-5-1")

    assert (
        _hook({"messages": [{"role": "user", "content": "hi"}]}, "anthropic_messages")
        is None
    )


def test_hook_native_responses_asks_for_output_text_logprobs(monkeypatch):
    _route_env(monkeypatch, "openai", "openai/gpt-5.5")
    monkeypatch.setenv("BENCHFLOW_CAPTURE_TOP_LOGPROBS", "2")
    data = {
        "model": "benchflow-openai-gpt-5.5",
        "input": "hi",
        "include": ["reasoning.encrypted_content"],
    }

    cleaned = _hook(data, "aresponses")

    assert cleaned is not None
    assert cleaned["include"] == [
        "reasoning.encrypted_content",
        "message.output_text.logprobs",
    ]
    assert cleaned["top_logprobs"] == 2
    assert "logprobs" not in cleaned


# --------------------------------------------------------------------------- #
# Normalization: build_token_capture                                          #
# --------------------------------------------------------------------------- #


def _record(response: Any, *, plan: dict | None = None, **extra: Any) -> dict:
    return {
        "event": "success",
        "token_capture": {
            "enabled": True,
            "wire": "openai-chat",
            "provider": "vllm",
            "request": "chat",
            "logprobs": True,
            "top_logprobs": None,
            "token_ids": True,
            **(plan or {}),
        },
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": {}},
        "response": response,
        "start_time": "2026-01-01T10:00:00",
        "end_time": "2026-01-01T10:00:01",
        **extra,
    }


def _logprob_content(tokens: str) -> dict:
    return {
        "content": [
            {
                "token": t,
                "logprob": logprob(i),
                "bytes": list(t.encode()),
                "top_logprobs": [],
            }
            for i, t in enumerate(tokens)
        ]
    }


def test_sglang_choice_level_token_ids_are_captured():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "logprobs": _logprob_content(REPLY),
                "provider_specific_fields": {
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": _REPLY_IDS,
                },
            }
        ]
    }

    capture = build_token_capture(_record(response))

    assert capture is not None
    assert capture["prompt_token_ids"] == [1, 2]
    assert capture["completions"][0]["token_ids"] == _REPLY_IDS
    assert capture["unavailable"] == {}
    stripped = strip_captured_token_ids(response)
    assert stripped["choices"][0]["provider_specific_fields"] == {}
    assert stripped["choices"][0]["logprobs"] == response["choices"][0]["logprobs"]


def test_unrequested_and_unreturned_fields_carry_distinct_reasons():
    response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}}]
    }

    not_requested = build_token_capture(
        _record(response, plan={"provider": "openai", "token_ids": False})
    )
    assert not_requested is not None
    reasons = {k: v["reason"] for k, v in not_requested["unavailable"].items()}
    assert reasons == {
        "prompt_token_ids": "not_requested",
        "completion_token_ids": "not_requested",
        "logprobs": "not_returned",
    }
    assert (
        "BENCHFLOW_CAPTURE_TOKEN_IDS=1"
        in (not_requested["unavailable"]["completion_token_ids"]["detail"])
    )


def test_failed_call_and_disabled_capture():
    failed = build_token_capture({**_record(None), "event": "failure"})
    assert failed is not None
    assert {v["reason"] for v in failed["unavailable"].values()} == {"request_failed"}

    disabled = _record({"choices": []})
    del disabled["token_capture"]
    assert build_token_capture(disabled) is None
    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(disabled), session_id="s", agent_name="a"
    )
    assert "token_capture" not in trajectory.exchanges[0].metadata


def test_callback_record_carries_plan_and_stream_tokens(monkeypatch):
    _route_env(monkeypatch, "vllm", "openai/qwen")
    now = datetime.now()
    stash = {"chunks": 2, "prompt_token_ids": [5], "choices": {"0": {"token_ids": [7]}}}

    record = _logger()._base_record(
        {
            "model": "qwen",
            "call_type": "acompletion",
            "messages": [{"role": "user", "content": "hi"}],
            "optional_params": {
                "logprobs": True,
                "extra_body": {"return_token_ids": True},
            },
            patch.STREAM_TOKENS_KEY: stash,
        },
        now,
        now,
    )

    assert record["token_capture"]["wire"] == "openai-chat"
    assert record["token_capture"]["token_ids"] is True
    assert record["stream_tokens"] == stash
    assert record["request"]["body"]["extra_body"] == {"return_token_ids": True}


# --------------------------------------------------------------------------- #
# Streaming: the proxy patch                                                   #
# --------------------------------------------------------------------------- #


def test_stream_chunks_accumulate_ids_and_logprobs_from_sdk_objects():
    """Guards the gateway token-capture prototype against LiteLLM's stream
    assembly dropping per-chunk ``token_ids``/``logprobs`` and the first
    chunk's top-level ``prompt_token_ids`` (vLLM streaming shape)."""
    from openai.types.chat import ChatCompletionChunk

    details: dict[str, Any] = {}
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "prompt_token_ids": [104, 105],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        },
        *(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "token_ids": [token_id(char)],
                        "logprobs": {
                            "content": [
                                {
                                    "token": char,
                                    "logprob": logprob(i),
                                    "bytes": None,
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    }
                ],
            }
            for i, char in enumerate(REPLY)
        ),
    ]
    for chunk in chunks:
        patch.record_stream_chunk(details, ChatCompletionChunk.model_validate(chunk))

    stash = details[patch.STREAM_TOKENS_KEY]
    assert stash["prompt_token_ids"] == [104, 105]
    assert stash["choices"]["0"]["token_ids"] == _REPLY_IDS
    assert [e["logprob"] for e in stash["choices"]["0"]["logprobs"]] == _REPLY_LOGPROBS


@pytest.mark.parametrize("enabled", [False, True])
def test_stream_patch_records_chunks_only_with_opt_in(monkeypatch, enabled):
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    assert getattr(
        CustomStreamWrapper.chunk_creator, "__benchflow_token_capture_patch__", False
    )
    if enabled:
        monkeypatch.setenv(_CAPTURE_ENV, "1")
    else:
        monkeypatch.delenv(_CAPTURE_ENV, raising=False)
    details: dict[str, Any] = {}
    wrapper = SimpleNamespace(logging_obj=SimpleNamespace(model_call_details=details))

    # The fake wrapper makes LiteLLM's own chunk handling fail after the patch
    # has run; only the patch's side effect matters here.
    with contextlib.suppress(Exception):
        CustomStreamWrapper.chunk_creator(
            wrapper, {"choices": [{"index": 0, "token_ids": [7]}]}
        )

    assert (patch.STREAM_TOKENS_KEY in details) is enabled


def test_runtime_files_load_token_capture_patch(tmp_path):
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})

    sitecustomize = (tmp_path / "sitecustomize.py").read_text()
    assert "import benchflow_litellm_token_capture_patch" in sitecustomize
    assert (tmp_path / "benchflow_litellm_token_capture_patch.py").is_file()
    assert runtime_mod._route_env(
        resolve_litellm_route(
            "vllm/qwen",
            {
                "BENCHFLOW_PROVIDER_BASE_URL": "http://x/v1",
                "BENCHFLOW_PROVIDER_API_KEY": "k",
            },
        )
    ) == {
        "BENCHFLOW_LITELLM_ROUTE_PROVIDER": "vllm",
        "BENCHFLOW_LITELLM_UPSTREAM_MODEL": "openai/qwen",
    }


def test_mock_server_matches_vllm_field_names():
    """The fixture mirrors vLLM's OpenAI server fields (checked against
    vllm-project/vllm main, entrypoints/openai/chat_completion/protocol.py)."""
    from tests.fixtures.mock_token_logprobs_server import chat_completion

    body = chat_completion(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "return_token_ids": True,
        }
    )

    assert body["prompt_token_ids"] == [104, 105]
    assert body["choices"][0]["token_ids"] == _REPLY_IDS
