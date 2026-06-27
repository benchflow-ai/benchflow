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
        "verifiers_step": {
            "prompt": [{"role": "user", "content": "hi"}],
            "completion": [{"role": "assistant", "content": "hello"}],
            "response": {"model": "anthropic/claude-haiku-4-5-20251001"},
            "tokens": None,
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "",
            "extras": {"source": "litellm_callback"},
        },
        "verifiers_tool_defs": [{"type": "function", "function": {"name": "bash"}}],
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
    assert trajectory.exchanges[0].verifiers_step is not None
    assert trajectory.exchanges[0].verifiers_step["completion"] == [
        {"role": "assistant", "content": "hello"}
    ]
    assert trajectory.exchanges[0].verifiers_tool_defs == [
        {"type": "function", "function": {"name": "bash"}}
    ]

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


def _callback_namespace() -> dict:
    """Exec the embedded callback module source so its helpers/classes can be
    exercised directly — the source ships as a string (runs inside the proxy
    process) and cannot be imported, so exec is the only faithful seam."""
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace


_CONTEXT_CAUSE = (
    "litellm.ContextWindowExceededError: OpenAIException - Requested token "
    "count exceeds the model's maximum context length of 16384 tokens. You "
    "requested a total of 17964 tokens: 1580 input + 16384 completion."
)


def test_failure_detail_prefers_exception_when_response_none():
    """Guards issue #830 fix#2: when litellm fires the failure hook with
    response_obj=None, the real cause in kwargs['exception'] must drive
    error.type/message — not the literal 'None'/'NoneType'."""
    _failure_detail = _callback_namespace()["_failure_detail"]
    detail = _failure_detail(None, ValueError(_CONTEXT_CAUSE))
    assert type(detail).__name__ == "ValueError"
    assert "16384 tokens" in str(detail)


def test_failure_detail_uses_response_when_present():
    """No behavior change on the existing path: a non-None response_obj wins."""
    _failure_detail = _callback_namespace()["_failure_detail"]
    assert _failure_detail("boom", ValueError("ignored")) == "boom"


def test_failure_detail_none_when_both_missing():
    """Graceful degradation: no response and no exception stays the old 'None'."""
    _failure_detail = _callback_namespace()["_failure_detail"]
    assert _failure_detail(None, None) is None


def test_failure_traceback_falls_back_to_exception_without_active_exc():
    """Greptile P2 / #830: when no exception is active (format_exc() is the
    'NoneType: None' sentinel) but we recovered the cause from kwargs['exception'],
    the traceback formats that exception so it doesn't go blank under a meaningful
    error.message."""
    _failure_traceback = _callback_namespace()["_failure_traceback"]
    # Called OUTSIDE any except block → traceback.format_exc() == 'NoneType: None\n'.
    tb = _failure_traceback(ValueError(_CONTEXT_CAUSE))
    assert "ValueError" in tb
    assert "16384 tokens" in tb
    assert "NoneType: None" not in tb


def test_failure_traceback_uses_active_exception():
    """When an exception IS active, format_exc() (the real stack) is used as-is."""
    _failure_traceback = _callback_namespace()["_failure_traceback"]
    try:
        raise RuntimeError("active boom")
    except RuntimeError as exc:
        tb = _failure_traceback(exc)
    assert "RuntimeError" in tb
    assert "active boom" in tb
    assert "Traceback (most recent call last)" in tb


def test_failure_traceback_non_exception_detail_keeps_sentinel():
    """No active exception and a non-exception detail (both-None path) keeps the
    old 'NoneType: None' behavior — no spurious formatting."""
    _failure_traceback = _callback_namespace()["_failure_traceback"]
    assert _failure_traceback(None).strip() == "NoneType: None"


async def test_failure_event_records_exception_cause_when_response_none(
    tmp_path, monkeypatch
):
    """End-to-end through the real write path: a context-window reject
    (response_obj=None, cause in kwargs['exception']) lands a USABLE
    error.message in the callback record, not 'None' (issue #830 fix#2)."""
    from datetime import datetime

    namespace = _callback_namespace()
    logger = namespace["BenchFlowLiteLLMLogger"]()
    log_path = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))

    now = datetime.now()
    await logger.async_log_failure_event(
        {"model": "benchflow-qwen", "exception": ValueError(_CONTEXT_CAUSE)},
        None,
        now,
        now,
    )

    record = json.loads(log_path.read_text().splitlines()[-1])
    assert record["event"] == "failure"
    assert record["error"]["type"] == "ValueError"
    assert "16384 tokens" in record["error"]["message"]
    assert record["error"]["message"] != "None"
