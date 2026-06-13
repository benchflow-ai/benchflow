from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime

import pytest

from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)


def _load_callback_logger():
    """Exec the embedded LiteLLM callback module (the real production source)
    in an isolated namespace, stubbing the ``litellm`` import so we can drive
    the callback against synthetic ``model_call_details`` without a live proxy.
    """
    fake = types.ModuleType("litellm")
    fake.completion_cost = lambda **_kwargs: 0.0
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class _CustomLogger:  # minimal CustomLogger base
        pass

    custom_logger.CustomLogger = _CustomLogger
    integrations.custom_logger = custom_logger
    saved = {
        name: sys.modules.get(name)
        for name in (
            "litellm",
            "litellm.integrations",
            "litellm.integrations.custom_logger",
        )
    }
    sys.modules["litellm"] = fake
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger
    try:
        namespace: dict[str, object] = {}
        exec(compile(callback_module_source(), "<callback>", "exec"), namespace)
        return namespace["proxy_handler_instance"]
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class _FakeResponse:
    def __init__(self, payload: dict, cost: float | None = None):
        self._payload = payload
        self._hidden_params = {"response_cost": cost} if cost is not None else {}

    def model_dump(self, mode=None):
        del mode
        return self._payload


_START = datetime(2026, 6, 4, 10, 0, 0)
_END = datetime(2026, 6, 4, 10, 0, 1)


def _cross_protocol_details():
    """OpenAI-protocol agent call routed to a Gemini GenerateContent backend."""
    return {
        "model": "benchflow-gemini-3-1-flash-lite",
        "messages": [{"role": "user", "content": "hi"}],
        "call_type": "acompletion",
        "litellm_params": {
            "model": "gemini/gemini-3.1-flash-lite",
            "api_base": "https://gw.example.test/v1beta/models/gemini-3.1-flash-lite",
        },
        "optional_params": {},
        "additional_args": {
            "complete_input_dict": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
            },
            "api_base": "https://gw.example.test/v1beta/models/gemini-3.1-flash-lite",
        },
        "original_response": json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            }
        ),
    }


def _real_proxy_success_details():
    """The REAL kwargs LiteLLM hands the success callback for an OpenAI-protocol
    agent routed to a Gemini GenerateContent backend through the proxy.

    Captured from a live probe of the running gateway: at success time
    ``litellm_params['model']`` is ``None`` and ``kwargs['model']`` is the BARE
    gateway alias (no ``gemini/`` prefix), so the upstream protocol is no longer
    derivable from the model name. The authoritative signal is
    ``litellm_params['api_base']`` — the true ``:generateContent`` resource URL.
    ``additional_args`` carries the translated native body but NO ``api_base``,
    and ``original_response`` is the raw provider JSON *string*.
    """
    return {
        "model": "gemini-3.1-flash-lite",  # bare gateway alias, no provider prefix
        "messages": [{"role": "user", "content": "ping"}],
        "call_type": "acompletion",
        "litellm_params": {
            "model": None,  # None at success — cannot classify from this
            "api_base": (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-3.1-flash-lite:generateContent"
            ),
        },
        "optional_params": {"stream": False},
        "additional_args": {
            # The LiteLLM-translated native GenerateContent body. Crucially,
            # additional_args does NOT carry 'api_base' on the real success path.
            "complete_input_dict": {
                "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                "generationConfig": {},
            },
        },
        "original_response": json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "pong"}]}}],
                "modelVersion": "gemini-3.1-flash-lite",
                "responseId": "abc123",
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 9,
                },
            }
        ),
    }


def _same_protocol_details():
    """OpenAI-protocol agent call routed to an OpenAI-compatible backend."""
    return {
        "model": "benchflow-gpt",
        "messages": [{"role": "user", "content": "hi"}],
        "call_type": "acompletion",
        "litellm_params": {
            "model": "openai/gpt-4.1-mini",
            "api_base": "https://api.openai.com/v1",
        },
        "optional_params": {},
        "additional_args": {
            "complete_input_dict": {"messages": [{"role": "user", "content": "hi"}]},
            "api_base": "https://api.openai.com/v1",
        },
    }


def test_cross_protocol_record_carries_provider_facing_view():
    logger = _load_callback_logger()
    record = logger._base_record(_cross_protocol_details(), _START, _END)

    # Agent-facing OpenAI view is preserved unchanged.
    assert record["request"]["path"] == "/v1/chat/completions"
    assert record["provider_model"] == "gemini/gemini-3.1-flash-lite"

    upstream = record["upstream"]
    assert upstream["protocol"] == "generate_content"
    assert upstream["request"]["path"] == ":generateContent"
    assert upstream["request"]["url"] == (
        "https://gw.example.test/v1beta/models/gemini-3.1-flash-lite"
    )
    # The translated GenerateContent payload (LiteLLM complete_input_dict).
    assert upstream["request"]["body"] == {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }


def test_cross_protocol_streaming_uses_stream_generate_content_path():
    details = _cross_protocol_details()
    details["optional_params"] = {"stream": True}
    logger = _load_callback_logger()

    record = logger._base_record(details, _START, _END)
    assert record["upstream"]["request"]["path"] == ":streamGenerateContent"


def test_success_event_attaches_raw_provider_response(tmp_path, monkeypatch):
    logger = _load_callback_logger()
    log_path = tmp_path / "llm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    response = _FakeResponse(
        {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        cost=0.0001,
    )
    asyncio.run(
        logger.async_log_success_event(
            _cross_protocol_details(), response, _START, _END
        )
    )

    record = json.loads(log_path.read_text().strip())
    # Agent-facing OpenAI response preserved.
    assert record["response"]["choices"][0]["message"]["content"] == "hello"
    # Provider-facing block carries the RAW GenerateContent response.
    upstream = record["upstream"]
    assert upstream["request"]["path"] == ":generateContent"
    assert upstream["response"]["candidates"][0]["content"]["parts"][0]["text"] == (
        "hello"
    )
    assert upstream["response"]["usageMetadata"]["totalTokenCount"] == 7


def test_real_proxy_success_kwargs_emit_upstream_block(tmp_path, monkeypatch):
    """Regression for the live-path bug: the cross-protocol upstream block must
    fire on the REAL proxy success kwargs, where ``litellm_params['model']`` is
    ``None`` and ``kwargs['model']`` is the bare alias (no ``gemini/`` prefix).

    The old detector classified the bare alias as ``openai_chat`` (no ``/``),
    saw it equal the agent-facing protocol, and emitted NO upstream block —
    silently dropping the faithful capture on the only path that matters. The
    fix derives the protocol from ``litellm_params['api_base']`` (the true
    ``:generateContent`` URL). This drives the SHIPPED success handler end to end
    and asserts the persisted (redacted) ``to_jsonl`` artifact gains the block.
    """
    logger = _load_callback_logger()
    log_path = tmp_path / "llm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    # The agent-facing OpenAI response the proxy returns to the caller.
    response = _FakeResponse(
        {
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
        },
        cost=3.5e-06,
    )
    asyncio.run(
        logger.async_log_success_event(
            _real_proxy_success_details(), response, _START, _END
        )
    )

    raw = log_path.read_text().strip()
    written = json.loads(raw)
    # Agent-facing OpenAI view preserved (the bug must not regress this).
    assert written["request"]["path"] == "/v1/chat/completions"
    assert written["response"]["choices"][0]["message"]["content"] == "pong"

    # The cross-protocol upstream block now fires on the real-shaped kwargs.
    upstream = written["upstream"]
    assert upstream["protocol"] == "generate_content"
    assert ":generateContent" in upstream["request"]["path"]
    assert upstream["request"]["url"].endswith(":generateContent")
    # Body is the translated native GenerateContent payload.
    assert "contents" in upstream["request"]["body"]
    assert "generationConfig" in upstream["request"]["body"]
    # Raw provider response carries the native GenerateContent fields.
    assert "candidates" in upstream["response"]
    assert "usageMetadata" in upstream["response"]
    assert upstream["response"]["usageMetadata"]["totalTokenCount"] == 9

    # And it survives the importer into the persisted, redacted to_jsonl artifact
    # — the only file actually shipped downstream.
    trajectory = trajectory_from_litellm_callback_log(
        raw, session_id="session", agent_name="gemini-agent"
    )
    persisted = json.loads(trajectory.to_jsonl(redact_keys=True).splitlines()[0])
    pup = persisted["upstream"]
    assert pup["protocol"] == "generate_content"
    assert ":generateContent" in pup["request"]["path"]
    assert "contents" in pup["request"]["body"]
    assert "generationConfig" in pup["request"]["body"]
    assert "candidates" in pup["response"]
    assert "usageMetadata" in pup["response"]


def test_real_proxy_success_kwargs_redact_planted_secret(tmp_path, monkeypatch):
    """A live-shaped secret planted in the real upstream URL query AND raw
    provider response must be scrubbed by the persisted ``to_jsonl`` chokepoint."""
    secret = "AIzaSy" + "C" * 30  # fake Google API key shape
    details = _real_proxy_success_details()
    # Mirror the real wire: the GenerateContent URL carries ?key=<secret>.
    details["litellm_params"]["api_base"] += f"?key={secret}"
    details["original_response"] = json.dumps(
        {
            "candidates": [{"content": {"parts": [{"text": "pong"}]}}],
            "apiKey": secret,
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 1,
                "totalTokenCount": 9,
            },
        }
    )

    logger = _load_callback_logger()
    log_path = tmp_path / "llm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))
    response = _FakeResponse(
        {"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
        cost=0.0,
    )
    asyncio.run(logger.async_log_success_event(details, response, _START, _END))

    raw = log_path.read_text().strip()
    trajectory = trajectory_from_litellm_callback_log(
        raw, session_id="session", agent_name="gemini-agent"
    )
    out = trajectory.to_jsonl(redact_keys=True)
    assert secret not in out
    assert "***REDACTED***" in out
    # Non-secret payload still survives redaction.
    assert "generateContent" in out
    assert "contents" in out


def test_same_protocol_record_is_byte_unchanged(tmp_path, monkeypatch):
    logger = _load_callback_logger()

    # _base_record must add no provider-facing fields for a same-protocol call.
    record = logger._base_record(_same_protocol_details(), _START, _END)
    assert "upstream" not in record

    # And the persisted success record must not gain an upstream block either.
    log_path = tmp_path / "llm.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))
    response = _FakeResponse(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        cost=0.0,
    )
    asyncio.run(
        logger.async_log_success_event(_same_protocol_details(), response, _START, _END)
    )
    written = json.loads(log_path.read_text().strip())
    assert "upstream" not in written


def test_anthropic_messages_agent_to_gemini_is_cross_protocol():
    """Cross-protocol detection is symmetric: an anthropic-messages agent call
    landing on a gemini GenerateContent backend also records the upstream view."""
    details = _cross_protocol_details()
    details["call_type"] = "anthropic_messages"
    logger = _load_callback_logger()

    record = logger._base_record(details, _START, _END)
    assert record["request"]["path"] == "/v1/messages"  # agent-facing preserved
    assert record["upstream"]["protocol"] == "generate_content"


def test_anthropic_messages_agent_to_anthropic_backend_is_same_protocol():
    """An anthropic-messages agent on an anthropic backend is same-protocol —
    no provider-facing block (byte-unchanged)."""
    details = {
        "model": "benchflow-claude",
        "messages": [{"role": "user", "content": "hi"}],
        "call_type": "anthropic_messages",
        "litellm_params": {
            "model": "anthropic/claude-haiku-4-5",
            "api_base": "https://api.anthropic.com",
        },
        "optional_params": {},
        "additional_args": {
            "complete_input_dict": {"messages": []},
            "api_base": "https://api.anthropic.com",
        },
    }
    logger = _load_callback_logger()

    record = logger._base_record(details, _START, _END)
    assert "upstream" not in record


@pytest.mark.parametrize(
    ("provider_model", "expected"),
    [
        ("gemini/gemini-3.1-flash-lite", "generate_content"),
        ("vertex_ai/gemini-2.5-pro", "generate_content"),
        ("anthropic/claude-haiku-4-5", "anthropic_messages"),
        ("openai/gpt-4.1-mini", "openai_chat"),
        ("azure/gpt-5.5", "openai_chat"),
    ],
)
def test_upstream_wire_protocol_classification(provider_model, expected):
    """The embedded callback classifies upstream wire protocols by model prefix
    so cross-protocol detection only fires when protocols truly differ."""
    fake = types.ModuleType("litellm")
    fake.completion_cost = lambda **_kwargs: 0.0
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = type("CustomLogger", (), {})
    integrations.custom_logger = custom_logger
    saved = {
        name: sys.modules.get(name)
        for name in (
            "litellm",
            "litellm.integrations",
            "litellm.integrations.custom_logger",
        )
    }
    sys.modules["litellm"] = fake
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger
    try:
        namespace: dict[str, object] = {}
        exec(compile(callback_module_source(), "<callback>", "exec"), namespace)
        assert namespace["_upstream_wire_protocol"](provider_model) == expected
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_callback_module_source_exposes_proxy_handler_instance():
    source = callback_module_source()

    assert "class BenchFlowLiteLLMLogger" in source
    assert "proxy_handler_instance = BenchFlowLiteLLMLogger()" in source


def test_litellm_callback_jsonl_imports_usage_and_cost():
    record = {
        "event": "success",
        "request_model": "benchflow-claude-haiku-4-5",
        "provider_model": "anthropic/claude-haiku-4-5-20251001",
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "body": {
                "model": "benchflow-claude-haiku-4-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        },
        "response": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 1,
            },
        },
        "response_cost": 0.00042,
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
        "duration_ms": 1000,
    }
    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="claude-agent-acp",
    )

    assert len(trajectory.exchanges) == 1
    assert trajectory.total_input_tokens == 13
    assert trajectory.total_output_tokens == 3
    assert trajectory.total_cache_read_tokens == 2
    assert trajectory.total_cache_creation_tokens == 1
    assert trajectory.total_cost_usd == 0.00042

    usage = extract_usage_from_trajectory(
        trajectory,
        fallback_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert usage["usage_source"] == "provider_response"
    assert usage["n_input_tokens"] == 13
    assert usage["n_output_tokens"] == 3
    assert usage["total_tokens"] == 16
    assert usage["cost_usd"] == 0.00042


def _cross_protocol_callback_record(secret: str) -> dict:
    """A success callback record as written by the embedded logger for a
    cross-protocol (OpenAI-agent -> GenerateContent backend) call, carrying the
    provider-facing ``upstream`` block with a secret-shaped value planted in the
    real upstream URL and the raw provider response.
    """
    return {
        "event": "success",
        "request_model": "benchflow-gemini-3-1-flash-lite",
        "provider_model": "gemini/gemini-3.1-flash-lite",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "benchflow-gemini-3-1-flash-lite",
                "messages": [{"role": "user", "content": "hi"}],
            },
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        "upstream": {
            "protocol": "generate_content",
            "request": {
                "method": "POST",
                "path": ":generateContent",
                "url": (
                    "https://gw.example.test/v1beta/models/"
                    f"gemini-3.1-flash-lite?key={secret}"
                ),
                "body": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            },
            "response": {
                "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "apiKey": secret,
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            },
        },
        "response_cost": 0.0001,
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
        "duration_ms": 1000,
    }


def test_upstream_block_reaches_persisted_trajectory_to_jsonl():
    """Blocker 1: the captured provider-facing ``upstream`` block must survive
    the importer into the round-tripped Trajectory and the persisted
    ``to_jsonl()`` artifact — the only file actually written downstream."""
    record = _cross_protocol_callback_record(secret="unused")
    record["upstream"]["request"]["url"] = (
        "https://gw.example.test/v1beta/models/gemini-3.1-flash-lite"
    )
    record["upstream"]["response"].pop("apiKey")

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="gemini-agent",
    )

    # The trajectory model now carries the provider-facing block.
    upstream = trajectory.exchanges[0].upstream
    assert upstream is not None
    assert upstream["protocol"] == "generate_content"
    assert upstream["request"]["path"] == ":generateContent"
    assert upstream["request"]["url"] == (
        "https://gw.example.test/v1beta/models/gemini-3.1-flash-lite"
    )
    assert upstream["request"]["body"] == {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }
    assert (
        upstream["response"]["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    )

    # And it is present in the only persisted artifact: to_jsonl() output.
    out = trajectory.to_jsonl(redact_keys=True)
    line = json.loads(out.splitlines()[0])
    persisted = line["upstream"]
    assert persisted["request"]["path"] == ":generateContent"
    assert persisted["request"]["body"]["contents"][0]["parts"][0]["text"] == "hi"
    assert "generateContent" in out
    assert "contents" in out


def test_upstream_block_is_redacted_on_persisted_path():
    """Blocker 2: a secret planted in the new fields (upstream URL / raw
    provider response) must be scrubbed by ``to_jsonl(redact_keys=True)`` on the
    retained artifact — not merely by the raw callback file's ephemeral lifetime."""
    secret = "AIzaSy" + "B" * 30  # fake Google API key shape
    record = _cross_protocol_callback_record(secret=secret)

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="gemini-agent",
    )

    # Sanity: the secret really is in the in-memory (unredacted) block.
    upstream = trajectory.exchanges[0].upstream
    assert secret in json.dumps(upstream)

    # Redacted export must not leak the secret anywhere (URL query or response).
    out = trajectory.to_jsonl(redact_keys=True)
    assert secret not in out
    assert "***REDACTED***" in out
    # The non-secret payload still survives redaction.
    assert "generateContent" in out
    assert "contents" in out


def test_same_protocol_to_jsonl_has_no_upstream_field():
    """Blocker 1/2 corollary: a same-protocol record (no upstream block) stays
    byte-identical — to_jsonl() must not introduce an ``upstream`` field."""
    record = {
        "event": "success",
        "request_model": "benchflow-gpt",
        "provider_model": "openai/gpt-4.1-mini",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {"model": "benchflow-gpt", "messages": []},
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
        "duration_ms": 1000,
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="gpt-agent",
    )

    assert trajectory.exchanges[0].upstream is None
    out = trajectory.to_jsonl(redact_keys=True)
    assert "upstream" not in out


def test_callback_log_preserves_bedrock_reasoning_effort_in_request_body():
    record = {
        "event": "success",
        "request_model": "benchflow-bedrock",
        "provider_model": "bedrock/us.anthropic.claude-opus-4-8",
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "us.anthropic.claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "max",
            },
        },
        "response": {
            "model": "bedrock/us.anthropic.claude-opus-4-8",
            "usage": {"inputTokens": 10, "outputTokens": 3},
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="openhands",
    )

    assert trajectory.exchanges[0].request.body["reasoning_effort"] == "max"


def test_litellm_failure_records_become_error_exchanges():
    record = {
        "event": "failure",
        "request_model": "benchflow-gpt",
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": {}},
        "error": {"type": "AuthenticationError", "message": "bad key"},
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:00",
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="codex-acp",
    )

    assert trajectory.exchanges[0].response.status_code == 500
    assert trajectory.exchanges[0].response.body["error"]["message"] == "bad key"
    usage = extract_usage_from_trajectory(trajectory, fallback_model="openai/gpt-4")
    assert usage["usage_source"] == "unavailable"
