"""The #1135 Bedrock Invoke patch under the pinned LiteLLM's call signature.

LiteLLM 1.103 normalizes ``role: "system"`` messages in
``_normalize_system_role_messages``, called with the routed model as a
``model=`` keyword: it hoists the leading run into ``system``, keeps a
mid-conversation one in place on the models its map flags, and turns it into a
user turn on the others. BenchFlow's wrapper runs that first. These tests call
the patched method the way 1.103's ``transform_anthropic_messages_request``
does; no network calls are made.
"""

from __future__ import annotations

from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)

import benchflow.providers.litellm_bedrock_patch  # noqa: F401 - applies the proxy patch


def _request(model: str) -> dict:
    return {
        "model": model,
        "system": [{"type": "text", "text": "You are a Claude agent."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Fix it."}]},
            {
                "role": "system",
                "content": "<context>reminder</context>",
                "output_config": {"effort": "high"},
            },
        ],
    }


def _normalize(request: dict, model: str) -> None:
    config = AmazonAnthropicClaudeMessagesConfig()
    method = config._normalize_system_role_messages
    assert getattr(method, "__benchflow_bedrock_patch__", False)
    method(request, model=model)


def test_mid_conversation_model_keeps_system_message_in_place() -> None:
    """Kept in place, with only its content: Bedrock refuses per-message
    fields such as ``output_config``."""
    model = "us.anthropic.claude-fable-5-1"
    request = _request(model)
    _normalize(request, model)
    assert [m["role"] for m in request["messages"]] == ["user", "system"]
    assert request["messages"][1] == {
        "role": "system",
        "content": "<context>reminder</context>",
    }
    assert [b["text"] for b in request["system"]] == ["You are a Claude agent."]


def test_a_model_litellm_flags_but_bedrock_was_not_seen_to_accept_gets_a_user_turn() -> (
    None
):
    """LiteLLM's map flags Sonnet 5 for mid-conversation system messages; BenchFlow
    keeps one in place only where Bedrock was seen to accept it, so it becomes
    a user turn in place, as on any unflagged model."""
    model = "us.anthropic.claude-sonnet-5"
    request = _request(model)
    _normalize(request, model)
    assert "system" not in [m["role"] for m in request["messages"]]
    assert [b["text"] for b in request["system"]] == ["You are a Claude agent."]


def test_other_model_gets_litellms_user_turn_with_its_arguments() -> None:
    """A model without mid-conversation system support gets LiteLLM's own
    in-place user turn; the top-level system prompt is left as it was."""
    model = "us.anthropic.claude-sonnet-4-6"
    request = _request(model)
    _normalize(request, model)
    assert "system" not in [m["role"] for m in request["messages"]]
    assert "<context>reminder</context>" in repr(request["messages"])
    assert [b["text"] for b in request["system"]] == ["You are a Claude agent."]


def test_a_leading_system_message_is_hoisted_on_every_model() -> None:
    model = "us.anthropic.claude-fable-5-1"
    request = _request(model)
    request["messages"].insert(0, {"role": "system", "content": "Be brief."})
    _normalize(request, model)
    assert [m["role"] for m in request["messages"]] == ["user", "system"]
    assert [b["text"] for b in request["system"]] == [
        "You are a Claude agent.",
        "Be brief.",
    ]
