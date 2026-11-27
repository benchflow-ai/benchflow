"""A denylist agent's native Claude Code request reaches api.anthropic.com.

Claude Code (2.1.280 to 2.1.321 at least) adds ``diagnostics:
{"previous_message_id": ...}`` to each main-thread request it sends to the
first-party API (the cache-diagnosis-2026-04-07 beta), never to a custom
ANTHROPIC_BASE_URL. ``BODY`` has the top-level fields and value shapes of the
first main-turn request Claude Code 2.1.280 sent on a subscription login
(captured with OTEL_LOG_RAW_API_BODIES; text shortened, one tool kept,
``betas`` dropped: the SDK sends it as the anthropic-beta header).

Guards the fix from PR #1255 (issue #1254) against the refusal that came with
the denylist's model-body check in PR #1209.
"""

import json

from benchflow.providers import litellm_request_guard as guard
from benchflow.sandbox import _egress_denylist_proxy as proxy

BODY = {
    "model": "claude-haiku-5-5",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Solve the task."}]},
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "<total_tokens>15000000 tokens left</total_tokens>",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
    ],
    "system": [
        {"type": "text", "text": "You are a Claude agent."},
        {
            "type": "text",
            "text": "Use the tools.",
            "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"},
        },
    ],
    "tools": [
        {
            "name": "Bash",
            "description": "Run a command.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ],
    "metadata": {"user_id": "user_0"},
    "max_tokens": 128000,
    "thinking": {"type": "adaptive", "display": "updates"},
    "context_management": {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    },
    "output_config": {"effort": "low"},
    "diagnostics": {"previous_message_id": None},
    "stream": True,
}
POLICY = proxy.Policy(["https://paper.test/protected"], [])


def _body(**changes):
    return json.dumps({**BODY, **changes}).encode()


def test_the_denylist_admits_claude_codes_main_turn_request():
    """Guards PR #1255: the main-turn body with ``diagnostics`` (null or a
    string id) passes the denylist on both Messages paths."""
    for diagnostics in (
        {"previous_message_id": None},
        {"previous_message_id": "msg_0"},
    ):
        body = _body(diagnostics=diagnostics)
        assert guard.outbound_refusal("POST", "/v1/messages", body) is None
        for path in ("/v1/messages", "/v1/messages?beta=true"):
            assert proxy.deny_body_rule(POLICY, body, path) is None


def test_diagnostics_carries_only_the_previous_message_id():
    """Guards PR #1255: any other ``diagnostics`` shape is still refused, and
    the reason names it."""
    for diagnostics in (
        {"previous_message_id": None, "url": "https://paper.test/protected"},
        {"previous_message_id": 1},
        "msg_0",
    ):
        why = guard.outbound_refusal(
            "POST", "/v1/messages", _body(diagnostics=diagnostics)
        )
        assert why is not None and "diagnostics" in why, diagnostics


def test_a_refused_messages_body_is_logged_with_the_messages_reason():
    """Guards PR #1255: a body both schemas refuse is logged with the Messages
    reason, not only the chat-completions one. Since PR #1259 a body sent to
    /v1/messages is judged as Messages alone; without a path both reasons are
    still named."""
    body = _body(diagnostics={"previous_message_id": 1})
    assert proxy.deny_body_rule(POLICY, body, "/v1/messages") == (
        "deny-model-body: diagnostics previous_message_id is not a string"
    )
    why = proxy.deny_body_rule(POLICY, body, "")
    assert why is not None and why.startswith("deny-model-body: as Messages: "), why
    assert "diagnostics" in why.split(";", 1)[0], why


def test_hosted_search_in_the_same_request_is_still_refused():
    """Guards PR #1255: admitting ``diagnostics`` lets no hosted tool through."""
    body = _body(tools=[{"type": "web_search_20250305", "name": "web_search"}])
    assert proxy.deny_body_rule(POLICY, body, "/v1/messages?beta=true")
