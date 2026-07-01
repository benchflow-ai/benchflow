from __future__ import annotations

import json

from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)


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


def test_callback_source_labels_responses_call_type_as_v1_responses():
    # A Responses-API call (call_type responses/aresponses) must be recorded with
    # the /v1/responses wire path and keep its `input`, so codex/Responses turns
    # land in llm_trajectory.jsonl labelled correctly (not as /v1/chat/completions).
    from datetime import datetime

    namespace: dict = {}
    exec(callback_module_source(), namespace)  # exercise the generated proxy module

    path_for = namespace["_request_path_for_call_type"]
    assert path_for("responses") == "/v1/responses"
    assert path_for("aresponses") == "/v1/responses"
    assert path_for("_aresponses_websocket") == "/v1/responses"
    assert path_for("anthropic_messages") == "/v1/messages"
    assert path_for("acompletion") == "/v1/chat/completions"
    assert path_for(None) == "/v1/chat/completions"

    logger = namespace["BenchFlowLiteLLMLogger"]()
    record = logger._base_record(
        {
            "call_type": "aresponses",
            "model": "benchflow-openai-gpt-5.4-mini",
            "input": [{"role": "user", "content": "hi"}],
        },
        datetime(2026, 6, 4, 10, 0, 0),
        datetime(2026, 6, 4, 10, 0, 1),
    )
    assert record["request"]["path"] == "/v1/responses"
    assert record["input_shape"]["has_input"] is True
    assert record["request"]["body"]["input"]


def test_responses_api_record_becomes_exchange_with_usage():
    # A /v1/responses success record (OpenAI Responses shape: `input` + a usage
    # block with input_tokens/output_tokens) parses into a trajectory exchange
    # with recoverable provider token usage — same as chat/completions + messages.
    record = {
        "event": "success",
        "request_model": "benchflow-openai-gpt-5.4-mini",
        "provider_model": "openai/gpt-5.4-mini",
        "call_type": "aresponses",
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "body": {"model": "benchflow-openai-gpt-5.4-mini", "input": "hi"},
        },
        "response": {
            "model": "gpt-5.4-mini",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "pong"}],
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        },
        "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        "response_cost": 0.0001,
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:01",
        "duration_ms": 1000,
    }
    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record), session_id="session", agent_name="codex-acp"
    )
    assert len(trajectory.exchanges) == 1
    assert trajectory.total_input_tokens == 12
    assert trajectory.total_output_tokens == 4

    usage = extract_usage_from_trajectory(
        trajectory, fallback_model="openai/gpt-5.4-mini"
    )
    assert usage["usage_source"] == "provider_response"
    assert usage["total_tokens"] == 16
