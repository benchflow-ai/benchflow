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


def test_context_length_failure_imports_as_permanent_rejected_request():
    """Guards issue #830: context-window failures must not look like 500s."""
    record = {
        "event": "failure",
        "request_model": "benchflow-qwen",
        "request": {"method": "POST", "path": "/v1/chat/completions", "body": {}},
        "error": {
            "type": "NoneType",
            "message": "None",
            "traceback": (
                "litellm.exceptions.BadRequestError: OpenAIException - Requested "
                "token count exceeds the model's maximum context length of "
                "16384 tokens."
            ),
        },
        "start_time": "2026-06-04T10:00:00",
        "end_time": "2026-06-04T10:00:00",
    }

    trajectory = trajectory_from_litellm_callback_log(
        json.dumps(record),
        session_id="session",
        agent_name="pi-acp",
    )

    assert trajectory.exchanges[0].response.status_code == 400
