"""The denylist judges a model request body by the protocol its path names.

``deny_body_rule`` used to pick the schema from the body's keys. A body with
``max_tokens``, ``messages`` and ``model`` was tried as Messages, then as chat
completions; any other model body, a Messages body without ``max_tokens``
included, was judged as chat completions only. Claude Code sends
``/v1/messages/count_tokens`` such a body, so its ``system`` field got a 403,
and a content block whose ``source`` the provider fetches (``"type": "url"``)
went out on either Messages path. Now a path under ``/v1/messages`` is judged
by the Messages schema, and no model request may name a fetched source.

Guards the fix from PR #1259 against the key-based schema choice that came
with the denylist's model-body check in PR #1209.
"""

# Re-export the real socket fixture for this module.
# ruff: noqa: F811
from __future__ import annotations

import json

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy
from tests.test_egress_claude_code_diagnostics import BODY
from tests.test_egress_connect_authority import (  # noqa: F401
    _read_response,
    _tunnel,
    authority_stack,
)

POLICY = proxy.Policy(["https://paper.test/protected"], [])
MESSAGES = "/v1/messages?beta=true"
COUNT_TOKENS = "/v1/messages/count_tokens?beta=true"
# A count_tokens request: the main turn's model, conversation, system and
# tools, and no max_tokens.
COUNT = {key: BODY[key] for key in ("model", "messages", "system", "tools")}
SOURCE_REFUSED = "deny-model-body: a content source is fetched by the provider"
URL_SOURCES = {
    "document": {
        "type": "document",
        "source": {"type": "url", "url": "https://paper.test/protected"},
    },
    "image": {
        "type": "image",
        "source": {"type": "url", "url": "https://example.test/figure.png"},
    },
}


def _rule(body: dict, path: str) -> str | None:
    return proxy.deny_body_rule(POLICY, json.dumps(body).encode(), path)


def _asking(body: dict, block: dict) -> dict:
    """``body`` with one more user turn that carries ``block``."""
    turn = {"role": "user", "content": [block]}
    return {**body, "messages": [*body["messages"], turn]}


@pytest.mark.parametrize("block", URL_SOURCES.values(), ids=URL_SOURCES.keys())
@pytest.mark.parametrize(
    ("body", "path"),
    [
        (COUNT, COUNT_TOKENS),
        (COUNT, "/v1/messages/count_tokens"),
        (BODY, MESSAGES),
        (BODY, "/v1/messages"),
    ],
    ids=["count_tokens", "count_tokens-no-query", "messages", "messages-no-query"],
)
def test_a_url_source_is_refused_on_either_messages_path(body, path, block):
    """Guards PR #1259: a count_tokens body was judged as chat completions,
    which reads no content block, and a full Messages body passed the Messages
    schema, which does not either."""
    assert _rule(_asking(body, block), path) == SOURCE_REFUSED


def test_a_url_source_in_system_or_a_tool_result_is_refused():
    """Guards PR #1259: a fetched source is found wherever a content block can
    carry one."""
    document = URL_SOURCES["document"]
    result = {"type": "tool_result", "tool_use_id": "t", "content": [document]}
    assert _rule({**COUNT, "system": [document]}, COUNT_TOKENS) == SOURCE_REFUSED
    assert _rule(_asking(BODY, result), MESSAGES) == SOURCE_REFUSED


@pytest.mark.parametrize(
    "path",
    [
        "/V1/%6Dessages/count_tokens",
        "/v1//x/../messages/count_tokens",
        "/v1/messages/count_tokens/",
        "/anthropic/v1/messages/count_tokens",
        "/v1/messages/count_tokens%0a",
    ],
    ids=["encoded", "dot-segments", "trailing-slash", "prefixed", "trailing-newline"],
)
def test_the_path_is_compared_in_its_decoded_form(path):
    """Guards PR #1259: an encoded or prefixed Messages path is a Messages path."""
    assert _rule(_asking(COUNT, URL_SOURCES["document"]), path) == SOURCE_REFUSED


_SEARCH = {"type": "web_search_20250305", "name": "web_search"}
SERVER_TOOLS = {
    "web_search": (
        {"tools": [_SEARCH]},
        "tool type 'web_search_20250305' is not a client-executed function",
    ),
    "web_fetch": (
        {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
        "tool type 'web_fetch_20250910' is not a client-executed function",
    ),
    "code_execution": (
        {"tools": [{"type": "code_execution_20250825", "name": "code_execution"}]},
        "tool type 'code_execution_20250825' is not a client-executed function",
    ),
    "mcp_servers": (
        {
            "mcp_servers": [
                {"type": "url", "url": "https://example.test/mcp", "name": "m"}
            ]
        },
        "unsupported request field(s) mcp_servers",
    ),
    "server-tool_choice": (
        {"tool_choice": {"type": "web_search", "name": "web_search"}},
        "tool_choice type is not a client-executed function",
    ),
}


@pytest.mark.parametrize(
    ("change", "reason"), SERVER_TOOLS.values(), ids=SERVER_TOOLS.keys()
)
@pytest.mark.parametrize(
    ("body", "path"),
    [(COUNT, COUNT_TOKENS), (BODY, MESSAGES)],
    ids=["count_tokens", "messages"],
)
def test_each_server_tool_is_refused_with_the_messages_reason(
    body, path, change, reason
):
    """Guards PR #1259: a server tool, an MCP server or a server tool_choice is
    refused on either Messages path, and the reason is the Messages schema's
    own, not one naming the chat-completions fields."""
    assert _rule({**body, **change}, path) == f"deny-model-body: {reason}"


_CHAT = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]}
ADMITTED = {
    "messages": (BODY, MESSAGES),
    "messages-no-query": (BODY, "/v1/messages"),
    "count_tokens": (COUNT, COUNT_TOKENS),
    "count_tokens-direct-endpoint": (COUNT, "/anthropic/v1/messages/count_tokens"),
    "count_tokens-base64-image": (
        _asking(
            COUNT,
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
            },
        ),
        COUNT_TOKENS,
    ),
    "batch": (
        {"requests": [{"custom_id": "a", "params": BODY}]},
        "/v1/messages/batches",
    ),
    "chat": ({**_CHAT, "max_tokens": 5}, "/v1/chat/completions"),
    "chat-function-tool": (
        {**_CHAT, "tools": [{"type": "function", "function": {"name": "f"}}]},
        "/v1/chat/completions",
    ),
    "responses": ({"model": "gpt-x", "input": "hi"}, "/v1/responses"),
    "gemini": (
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        "/v1beta/models/gemini-x:generateContent",
    ),
}


@pytest.mark.parametrize(("body", "path"), ADMITTED.values(), ids=ADMITTED.keys())
def test_ordinary_model_requests_are_admitted(body, path):
    """Guards PR #1259: Claude Code's Messages and count_tokens bodies, and
    ordinary chat, Responses and Gemini bodies, still leave."""
    assert _rule(body, path) is None


def test_the_key_based_guess_misjudged_a_count_tokens_body():
    """Guards PR #1259 against the key-based schema choice from PR #1209: a
    Messages body without max_tokens was judged as chat completions. Without a
    path the guess is all there is, and it still refuses this body; on its
    real path the body is a Messages request and leaves."""
    assert _rule(COUNT, "") == "deny-model-body: unsupported request field(s) system"
    assert _rule(COUNT, COUNT_TOKENS) is None
    # A fetched source is refused whatever schema the body is read as.
    assert _rule(_asking(BODY, URL_SOURCES["document"]), "") == SOURCE_REFUSED


def _post(path: str, body: dict) -> bytes:
    data = json.dumps(body).encode()
    return (
        f"POST {path} HTTP/1.1\r\nHost: carrier.test\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(data)}\r\n\r\n"
    ).encode() + data


def test_a_refused_count_tokens_request_is_a_logged_403(authority_stack):
    """Guards PR #1259: the refusal reaches the block log with its reason, and
    nothing reaches the origin; the same request without the source does."""
    refused = _asking(COUNT, URL_SOURCES["document"])
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(COUNT_TOKENS, refused))
        status, headers, _ = _read_response(tls)
    assert status == 403
    assert headers.get("x-benchflow-blocked") == "1"
    assert authority_stack.requests == []
    events = [json.loads(line) for line in authority_stack.log.read_text().splitlines()]
    assert [(e["action"], e["url"], e["rule"]) for e in events] == [
        ("blocked", "https://carrier.test/v1/messages/count_tokens", SOURCE_REFUSED)
    ]
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(COUNT_TOKENS, COUNT))
        status, _, body = _read_response(tls)
    assert status == 200
    assert json.loads(json.loads(body)["body"]) == COUNT
