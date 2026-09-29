"""SGLang token ids captured at the LiteLLM gateway (``sglang/`` route).

SGLang's OpenAI-compatible server returns exact prompt and sampled token ids
through its response-level ``sglext`` extension when the request sets
``return_input_ids_in_sglext`` / ``return_output_ids_in_sglext``: in the body
of a non-streamed call, and in one final ``choices: []`` chunk of a streamed
call. Its older ``return_token_ids`` flag is refused on streamed chat. Before
the ``sglang`` provider existed a SGLang server could only be reached as a
generic OpenAI route, so the gateway recorded logprobs but no token ids.
"""

from __future__ import annotations

import asyncio

import pytest

import benchflow.providers.litellm_token_capture_patch as patch
from benchflow.agents.providers import find_provider
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.trajectories.token_capture import build_token_capture
from tests.fixtures.mock_token_logprobs_server import REPLY, logprob, token_id
from tests.test_gateway_token_capture import (
    _drive_gateway,
    _hook,
    _logprob_content,
    _marker,
    _record,
    _route_env,
)

_REPLY_IDS = [token_id(c) for c in REPLY]
_SGLANG_FLAGS = {
    "return_input_ids_in_sglext": True,
    "return_output_ids_in_sglext": True,
}
_SGLANG_PLAN = {"provider": "sglang", "token_id_style": "sglang"}


def test_sglang_is_a_registered_self_hosted_provider():
    found = find_provider("sglang/mock-policy")
    assert found is not None
    name, cfg = found
    assert name == "sglang"
    assert cfg.base_url == ""
    route = resolve_litellm_route(
        "sglang/mock-policy",
        {
            "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:30000/v1",
            "BENCHFLOW_PROVIDER_API_KEY": "sk-mock",
        },
    )
    assert route.provider_name == "sglang"
    assert route.upstream_model == "openai/mock-policy"
    assert route.litellm_params["api_base"] == "http://127.0.0.1:30000/v1"


def test_hook_asks_sglang_for_sglext_ids_not_return_token_ids(monkeypatch):
    _route_env(monkeypatch, "sglang", "openai/qwen")
    monkeypatch.delenv("BENCHFLOW_CAPTURE_TOKEN_IDS", raising=False)

    for call_type in ("acompletion", "anthropic_messages"):
        cleaned = _hook(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            call_type,
        )
        assert cleaned is not None
        assert cleaned["logprobs"] is True
        assert cleaned["extra_body"] == _SGLANG_FLAGS


def test_forcing_token_ids_on_sglang_route_still_uses_sglext(monkeypatch):
    _route_env(monkeypatch, "sglang", "openai/qwen")
    monkeypatch.setenv("BENCHFLOW_CAPTURE_TOKEN_IDS", "1")

    cleaned = _hook({"messages": [{"role": "user", "content": "hi"}]}, "completion")

    assert cleaned is not None
    assert "return_token_ids" not in cleaned["extra_body"]
    assert cleaned["extra_body"] == _SGLANG_FLAGS


def test_sglext_body_ids_are_captured():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "logprobs": _logprob_content(REPLY),
            }
        ],
        "sglext": {"input_ids": [5, 6, 7], "output_ids": [_REPLY_IDS]},
    }
    record = _record(response, plan=_SGLANG_PLAN)
    record["request"]["body"] = dict(_SGLANG_FLAGS)

    capture = build_token_capture(record)

    assert capture is not None
    assert capture["requested"]["token_ids"] is True
    assert capture["prompt_token_ids"] == [5, 6, 7]
    assert capture["completions"][0]["token_ids"] == _REPLY_IDS
    assert capture["completions"][0]["logprobs"] == [
        logprob(i) for i in range(len(REPLY))
    ]
    assert capture["unavailable"] == {}


def test_streamed_sglext_chunk_is_accumulated():
    details: dict = {}
    for chunk in (
        {"choices": [{"index": 0, "delta": {"content": "o"}}]},
        {"choices": [], "sglext": {"input_ids": [1, 2], "output_ids": [[9, 8]]}},
    ):
        patch.record_stream_chunk(details, chunk)

    stash = details[patch.STREAM_TOKENS_KEY]
    assert stash["prompt_token_ids"] == [1, 2]
    assert stash["choices"]["0"]["token_ids"] == [9, 8]


def test_sglang_server_ignoring_the_flags_is_not_returned():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "logprobs": _logprob_content(REPLY),
            }
        ]
    }
    record = _record(response, plan=_SGLANG_PLAN)
    record["request"]["body"] = dict(_SGLANG_FLAGS)

    capture = build_token_capture(record)

    assert capture is not None
    assert capture["unavailable"]["prompt_token_ids"]["reason"] == "not_returned"
    assert capture["unavailable"]["completion_token_ids"]["reason"] == "not_returned"


@pytest.fixture(scope="module")
def sglang_gateway(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sglang-gateway")
    cases = ["chat", "chat_stream", "messages", "messages_stream"]
    return asyncio.run(
        _drive_gateway(tmp, "sglang/mock-policy", cases, flavor="sglang")
    )


@pytest.mark.parametrize("case", ["chat", "chat_stream", "messages", "messages_stream"])
def test_gateway_records_sglang_ids_streamed_and_not(sglang_gateway, case):
    """End to end through the real LiteLLM proxy: every chat-backed call to a
    SGLang-shaped server records prompt ids, sampled ids and logprobs."""
    records, upstream = sglang_gateway
    capture = records[case]["metadata"]["token_capture"]

    assert capture["provider"] == "sglang"
    assert capture["unavailable"] == {}
    assert capture["prompt_token_ids"] == [ord(c) for c in _marker(case)]
    [completion] = capture["completions"]
    assert completion["token_ids"] == _REPLY_IDS
    assert completion["logprobs"] == [logprob(i) for i in range(len(REPLY))]
    assert all(
        r["body"].get("return_input_ids_in_sglext") is True
        and "return_token_ids" not in r["body"]
        for r in upstream
    )


def test_sglext_only_chunk_is_recognised_but_usage_chunk_is_not():
    """Guards the SGLang streaming fix: LiteLLM 1.91's Anthropic Messages stream
    adapter raised IndexError on SGLang's final ``choices: []`` sglext chunk, so
    a streamed agent call to SGLang wrote no gateway record. The patch drops
    that chunk after reading its ids; usage-only chunks still pass through."""
    assert patch.is_sglext_only_chunk({"choices": [], "sglext": {"input_ids": [1]}})
    assert not patch.is_sglext_only_chunk(
        {"choices": [], "usage": {"prompt_tokens": 1}, "sglext": {}}
    )
    assert not patch.is_sglext_only_chunk({"choices": [], "usage": {"x": 1}})
    assert not patch.is_sglext_only_chunk({"choices": [{"index": 0}]})
