"""EN-645: a Responses API answer's tokens reach result.json's counters.

X4 cells ``rel-harvey-lab-harness-acp-mimo-hello-default-run`` (trial
hello__67110789, 12,030 provider tokens) and its rc twin (hello__5f869b29)
made two real Responses calls each through the model proxy. result.json,
summary.json, the metrics and the job export held null counters (0.9.x kept
the provider total in ``usage_details`` only) and ``trials_without_usage``
was 0. LiteLLM hands the callbacks the answer with its usage swapped for chat
completions' ``Usage``, which the answer's own model cannot dump: the logged
body kept ``{"total_tokens": ...}`` alone. The answers below are the
provider's bytes from X4's independent upstream recorder.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow._utils.evaluation_results import usage_summary
from benchflow.metrics import BenchmarkMetrics, TaskMetrics
from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)

# The two answers' usage blocks as MiMo sent them (X4 evidence/outside/
# rc-harvey-lab-harness-acp-mimo-hello-default-run/*.response.raw).
_X4_ANSWERS: list[dict[str, Any]] = [
    {
        "id": "resp_cb38ae71-fb5c-4e2a-a064-7e827e87f067",
        "object": "response",
        "created_at": 1794698211,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": "mimo-v2.6-flash",
        "metadata": None,
        "output": [
            {
                "id": "fc_5c91c88c47c44af4b57eb520f6ed27e3",
                "type": "function_call",
                "call_id": "call_a3c5e126cf7e417a833c9bb3",
                "name": "bash",
                "arguments": '{"command": "printf \'ok\\\\n\' > /app/answer.txt"}',
                "status": "completed",
            }
        ],
        "output_text": "",
        "usage": {
            "input_tokens": 5943,
            "input_tokens_details": {},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 17},
            "total_tokens": 5996,
        },
    },
    {
        "id": "resp_d99eec81-2ef1-45ec-81c5-44aba90c3b62",
        "object": "response",
        "created_at": 1794698215,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": "mimo-v2.6-flash",
        "metadata": None,
        "output": [
            {
                "id": "msg_71c10472d4284090bc16039395397477",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Task complete.", "annotations": []}
                ],
            }
        ],
        "output_text": "Task complete.",
        "usage": {
            "input_tokens": 6005,
            "input_tokens_details": {"cached_tokens": 5888},
            "output_tokens": 38,
            "output_tokens_details": {"reasoning_tokens": 17},
            "total_tokens": 6043,
        },
    },
]


def _as_litellm_hands_it_to_callbacks(answer: dict[str, Any]) -> Any:
    """The answer object LiteLLM's success handler passes to the callbacks:
    its own ``Logging._transform_usage_objects`` on the parsed answer."""
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.llms.openai import ResponsesAPIResponse

    parsed = ResponsesAPIResponse(**answer)
    stub = SimpleNamespace(model_call_details={})
    return Logging._transform_usage_objects(stub, parsed)


async def _log_x4_calls(tmp_path, monkeypatch) -> str:
    namespace: dict[str, Any] = {}
    exec(callback_module_source(), namespace)
    logger = namespace["BenchFlowLiteLLMLogger"]()
    log_path = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log_path))
    for answer in _X4_ANSWERS:
        now = datetime.now()
        await logger.async_log_success_event(
            {
                "model": "mimo-v2.6-flash",
                "call_type": "aresponses",
                "input": [{"role": "user", "content": "write ok"}],
            },
            _as_litellm_hands_it_to_callbacks(answer),
            now,
            now,
        )
    return log_path.read_text()


@pytest.mark.asyncio
async def test_x4_harvey_responses_usage_reaches_the_counters(tmp_path, monkeypatch):
    log = await _log_x4_calls(tmp_path, monkeypatch)
    for line in log.splitlines():
        assert json.loads(line)["response"]["usage"]["completion_tokens"] > 0

    trajectory = trajectory_from_litellm_callback_log(
        log, session_id="hello__5f869b29", agent_name="harvey-lab-harness"
    )
    usage = extract_usage_from_trajectory(trajectory)

    # Uncached input: 5943 + (6005 - 5888); X4's inclusive input is 11948.
    assert usage["n_input_tokens"] == 6060
    assert usage["n_cache_read_tokens"] == 5888
    assert usage["n_output_tokens"] == 91
    assert usage["total_tokens"] == 12039
    # #261 records cache-write details in usage_details; no reader is strict about it, per F2b-FXX4U 11-15.
    assert set(usage) == {
        "n_input_tokens",
        "n_output_tokens",
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "n_cache_write_5m_tokens",
        "n_cache_write_1h_tokens",
        "total_tokens",
        "usage_details",
        "cost_usd",
        "usage_source",
        "price_source",
    }
    assert set(usage["usage_details"]) == {"cache_writes"}
    assert usage["usage_details"]["cache_writes"] == {
        "tokens": None,
        "named_tokens": 0,
        "inside_input": True,
        "models": ["mimo-v2.6-flash"],
        "priced_models": [],
        "cost_floor": False,
    }


def _x4_stored_record(total: int) -> dict[str, Any]:
    # A callback line as the unfixed proxy wrote it: the usage cut to its total.
    return {
        "event": "success",
        "call_type": "aresponses",
        "request_model": "mimo-v2.6-flash",
        "request": {"method": "POST", "path": "/v1/responses", "body": {}},
        "response": {"object": "response", "usage": {"total_tokens": total}},
        "usage": {"total_tokens": total},
        "response_cost": 0.0004,
        "start_time": "2026-11-14T23:16:48",
        "end_time": "2026-11-14T23:16:52",
    }


def test_a_total_without_its_split_stays_on_record():
    log = "\n".join(json.dumps(_x4_stored_record(n)) for n in (5996, 6043))
    trajectory = trajectory_from_litellm_callback_log(
        log, session_id="hello__5f869b29", agent_name="harvey-lab-harness"
    )

    usage = extract_usage_from_trajectory(trajectory)

    for field in (
        "n_input_tokens",
        "n_output_tokens",
        "n_cache_read_tokens",
        "n_cache_creation_tokens",
        "total_tokens",
    ):
        assert usage[field] is None, field
    assert usage["usage_details"] == {
        "provider_total_tokens": 12039,
        "split": "unreported",
    }


# result.json's agent_result for hello__67110789 as the unfixed 0.9.x wrote it.
_X4_NULL_COUNTERS = {
    "n_input_tokens": None,
    "n_output_tokens": None,
    "n_cache_read_tokens": None,
    "n_cache_write_tokens": None,
    "n_cache_creation_tokens": None,
    "total_tokens": None,
    "cost_usd": 0.0000801528,
    "usage_source": "proxy",
    "usage_source_legacy": "provider_response",
    "cost_source": "list_price_estimate",
    "no_usage_reason": None,
    "usage_details": {"provider_total_tokens": 12030, "split": "unreported"},
}

_COUNTED = {
    **_X4_NULL_COUNTERS,
    "n_input_tokens": 6060,
    "n_output_tokens": 91,
    "n_cache_read_tokens": 5888,
    "total_tokens": 12039,
    "usage_details": None,
}


def test_a_trial_with_null_counters_is_a_trial_without_usage():
    summary = usage_summary(
        {
            "hello__67110789": {"agent_result": _X4_NULL_COUNTERS},
            "counted": {"agent_result": _COUNTED},
        }
    )

    assert summary["trials_without_usage"] == 1
    assert summary["usage_sources"] == {"proxy": 1}
    assert summary["telemetry_coverage"] == 0.5
    assert summary["total_tokens"] == 12039
    # Its cost is still known and still counted.
    assert summary["trials_without_cost"] == 0
    assert summary["total_cost_usd"] == pytest.approx(2 * 0.0000801528)


def test_metrics_coverage_leaves_out_null_counters():
    metrics = BenchmarkMetrics(
        benchmark="x4",
        agent="harvey-lab-harness",
        model="mimo-v2.6-flash",
        tasks=[
            TaskMetrics(
                task_name="hello",
                reward=1.0,
                usage_source="proxy",
                cost_usd=0.0000801528,
            ),
            TaskMetrics(
                task_name="hello-2",
                reward=1.0,
                total_tokens=12039,
                usage_source="proxy",
                cost_usd=0.0000801528,
            ),
        ],
    )

    assert metrics.telemetry_coverage == 0.5
