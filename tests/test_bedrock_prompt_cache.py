"""Regression coverage for #1135: Bedrock prompt-cache breakpoints.

Claude Code sends mid-conversation ``role: "system"`` messages and puts its
moving cache breakpoint on the newest one. LiteLLM 1.91.0's Bedrock Invoke
transform hoisted those messages (and the breakpoint) into the top-level
``system`` field, so no breakpoint was left in ``messages`` and every turn
re-billed the transcript as uncached input. These tests drive BenchFlow's
route through LiteLLM's real ``/v1/messages`` handler and capture the body it
would send to Bedrock; no network calls are made.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchflow.providers.litellm_bedrock_patch import (
    BEDROCK_MID_CONVERSATION_SYSTEM_RE,
    _relocate_trailing_system_breakpoint,
)
from benchflow.providers.litellm_config import (
    litellm_proxy_config,
    resolve_litellm_route,
)

_MODEL = "aws-bedrock/us.anthropic.claude-fable-5-1"
_EPHEMERAL = {"type": "ephemeral"}
_REMINDER = "<context>synthetic reminder</context>"


def _claude_code_body(turn: int) -> dict:
    """Request shape Claude Code sends to a BenchFlow proxy (synthetic values)."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix the failing test."}],
        },
        {
            "role": "system",
            "content": "# Environment\nPrimary working directory: /app",
            "output_config": {"effort": "max"},
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "a.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "print(1)"}
            ],
        },
    ]
    reminder = [
        {"type": "text", "text": _REMINDER, "cache_control": dict(_EPHEMERAL)},
        {"type": "text", "text": "Request every independent item in one response."},
    ]
    if turn == 1:
        messages.append({"role": "system", "content": reminder})
    else:
        messages += [
            {"role": "system", "content": _REMINDER},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Read",
                        "input": {"file_path": "b.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "print(2)",
                    }
                ],
            },
            {"role": "system", "content": reminder},
        ]
    return {
        "max_tokens": 1024,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "max"},
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=0.0.0; cc_entrypoint=sdk-cli;",
            },
            {
                "type": "text",
                "text": "You are a Claude agent.",
                "cache_control": dict(_EPHEMERAL),
            },
            {
                "type": "text",
                "text": "Long system prompt.",
                "cache_control": dict(_EPHEMERAL),
            },
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "messages": messages,
    }


def _breakpoints(body: dict) -> list[tuple[str, int, int]]:
    found = [
        ("system", i, -1)
        for i, b in enumerate(body.get("system") or [])
        if "cache_control" in b
    ]
    for i, message in enumerate(body["messages"]):
        content = message["content"]
        if isinstance(content, list):
            found += [
                (message["role"], i, j)
                for j, b in enumerate(content)
                if "cache_control" in b
            ]
    return found


async def _captured_bedrock_body(monkeypatch, body: dict) -> tuple[httpx.Request, dict]:
    """Send ``body`` through BenchFlow's route and LiteLLM's real handler."""
    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    import benchflow.providers.litellm_bedrock_patch  # noqa: F401 - applies the proxy patch

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-key")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-2")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    route = resolve_litellm_route(
        _MODEL, {"AWS_BEARER_TOKEN_BEDROCK": "x", "AWS_REGION": "us-east-2"}
    )
    settings = litellm_proxy_config(route, master_key="sk-test")["litellm_settings"]
    assert isinstance(settings, dict)
    monkeypatch.setattr(litellm, "drop_params", settings["drop_params"])
    monkeypatch.setattr(
        litellm,
        "use_chat_completions_url_for_anthropic_messages",
        settings["use_chat_completions_url_for_anthropic_messages"],
    )

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "us.anthropic.claude-fable-5-1",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await litellm.anthropic_messages(
        **body, **route.litellm_params, client=client, stream=False
    )
    assert len(captured) == 1
    return captured[0], json.loads(captured[0].content)


@pytest.mark.parametrize("turn", [1, 2])
async def test_bedrock_invoke_body_keeps_a_messages_cache_breakpoint(monkeypatch, turn):
    """Guards #1135: the outgoing Bedrock body must keep a breakpoint in
    ``messages`` on the user turn before the trailing system message."""
    body = _claude_code_body(turn)
    request, sent = await _captured_bedrock_body(monkeypatch, body)

    assert request.url.path == "/model/us.anthropic.claude-fable-5-1/invoke"
    # System messages stay where Claude Code put them.
    assert [m["role"] for m in sent["messages"]] == [
        m["role"] for m in body["messages"]
    ]
    assert all(set(m) == {"role", "content"} for m in sent["messages"])
    last_user = len(sent["messages"]) - 2
    assert sent["messages"][last_user]["role"] == "user"
    assert "cache_control" in sent["messages"][last_user]["content"][-1]
    assert all("cache_control" not in b for b in sent["messages"][-1]["content"])
    # Same number of breakpoints as the client sent, never more than four.
    assert len(_breakpoints(sent)) == len(_breakpoints(body)) <= 4
    assert not any(
        b["text"].startswith("x-anthropic-billing-header:") for b in sent["system"]
    )


def test_relocation_moves_breakpoint_from_string_user_turn() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "r",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
    ]

    out = _relocate_trailing_system_breakpoint(messages)

    assert out[0]["content"] == [
        {
            "type": "text",
            "text": "hello",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert out[1]["content"] == [{"type": "text", "text": "r"}]
    # The caller's request objects are not mutated.
    assert messages[0]["content"] == "hello"
    assert "cache_control" in messages[1]["content"][0]


@pytest.mark.parametrize(
    "messages",
    [
        # Nothing trails the user turn.
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "u", "cache_control": _EPHEMERAL}],
            }
        ],
        # A trailing system message after an assistant turn has no user target.
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            {
                "role": "system",
                "content": [{"type": "text", "text": "r", "cache_control": _EPHEMERAL}],
            },
        ],
        # The trailing system message carries no breakpoint.
        [{"role": "user", "content": "u"}, {"role": "system", "content": "r"}],
    ],
)
def test_relocation_leaves_other_shapes_unchanged(messages) -> None:
    assert _relocate_trailing_system_breakpoint(messages) is messages


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("us.anthropic.claude-fable-5-1", True),
        ("global.anthropic.claude-fable-5", True),
        ("us.anthropic.claude-opus-4-8", True),
        ("us.anthropic.claude-opus-5-5", True),
        ("us.anthropic.claude-sonnet-5", False),
        ("us.anthropic.claude-opus-4-7", False),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", False),
    ],
)
def test_mid_conversation_system_model_gate(model, supported) -> None:
    assert bool(BEDROCK_MID_CONVERSATION_SYSTEM_RE.search(model)) is supported
