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


def _instantiate_callback_logger():
    """Exec the proxy-side callback module source and return a logger instance."""
    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace["BenchFlowLiteLLMLogger"]()


def test_callback_records_bf_agent_attribution_from_request_metadata():
    """A multi-agent run tags each LLM call with ``bf.*`` fields in request
    ``metadata``; the proxy callback must record them (under
    ``request.body['bf']``) so one shared proxy log can be split into an unmixed
    agent tree. Today the callback keeps only ``model_group`` and drops the rest."""
    from datetime import datetime

    logger = _instantiate_callback_logger()
    record = logger._base_record(
        kwargs={
            "model": "benchflow-deepseek-v4-pro",
            "messages": [{"role": "user", "content": "side effects?"}],
            "metadata": {
                "model_group": "benchflow-deepseek-v4-pro",  # litellm-internal, untouched
                "bf.agent_id": "answer",
                "bf.agent_name": "answer",
                "bf.span_kind": "chat",
                "bf.parent_agent_id": "supervisor",
                "bf.run_id": "answer#1",
                "bf.session_id": "medical-run-1",
            },
        },
        start_time=datetime(2026, 6, 29, 10, 0, 0),
        end_time=datetime(2026, 6, 29, 10, 0, 1),
    )

    assert record["request"]["body"]["bf"] == {
        "agent_id": "answer",
        "agent_name": "answer",
        "span_kind": "chat",
        "parent_agent_id": "supervisor",
        "run_id": "answer#1",
        "session_id": "medical-run-1",
    }


def test_callback_records_extended_bf_vocabulary_without_code_change():
    """The callback captures ANY ``bf.*`` key generically, so the richer
    attribution dimensions from the adapter proposal (#847) — role, scene,
    turn_index, team_id, framework, framework_node_id, trace_id — flow through
    with no code change. One shared ``bf.*`` vocabulary across PRs, not two."""
    from datetime import datetime

    logger = _instantiate_callback_logger()
    record = logger._base_record(
        kwargs={
            "model": "m",
            "messages": [],
            "metadata": {
                "bf.agent_id": "planner",
                "bf.role": "planner",
                "bf.scene": "review",
                "bf.turn_index": 2,
                "bf.team_id": "core",
                "bf.framework": "langgraph",
                "bf.framework_node_id": "plan_node",
                "bf.trace_id": "t-abc",
            },
        },
        start_time=datetime(2026, 6, 29, 10, 0, 0),
        end_time=datetime(2026, 6, 29, 10, 0, 1),
    )

    bf = record["request"]["body"]["bf"]
    assert bf["role"] == "planner"
    assert bf["scene"] == "review"
    assert bf["turn_index"] == 2
    assert bf["team_id"] == "core"
    assert bf["framework"] == "langgraph"
    assert bf["framework_node_id"] == "plan_node"
    assert bf["trace_id"] == "t-abc"


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
