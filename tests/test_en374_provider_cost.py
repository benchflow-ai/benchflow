"""A provider's bill is the cost; list price only fills in (EN-374, 0.9.3).

The lead's rule (2026-11-13): when the provider billed a call, its figure is
the cost, with ``cost_source: provider``; a list price is only the fallback,
labelled as one. On rc2 a LiteLLM -> OpenRouter run, streamed, recorded
OpenRouter's bill as cost_usd (it equals the sum of usage.cost in all 54
frontier-design-benchmark PR #17 trials) but labelled it ``litellm``, and the
per-call record had no cost. 0.9.2 takes the same path. Now: a trial every
call of which was billed costs the bill (``price_source: openrouter``), each
call's bill is in its record, and a trial some calls of which carried no bill
costs the billed part plus a list-price estimate of the rest
(``list_price_estimate``), with ``cost_usd_provider`` and
``n_calls_cost_estimated`` beside it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

MODEL = "openrouter/moonshotai/kimi-k3"


def _callback():
    from benchflow.providers.litellm_logging import callback_module_source

    namespace: dict = {}
    exec(callback_module_source(), namespace)
    return namespace


def _streamed_openrouter_response(cost: float | None):
    """A streamed OpenRouter answer as LiteLLM assembles it: the chunks, the
    last carrying usage (with OpenRouter's usage.cost), built into one
    response, whose hidden params then get the cost as LiteLLM's stream
    wrapper sets them."""
    import litellm
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    usage = {"prompt_tokens": 400, "completion_tokens": 20, "total_tokens": 420}
    if cost is not None:
        usage["cost"] = cost
    chunks = [
        litellm.ModelResponseStream(
            id="gen-1",
            created=1,
            model="moonshotai/kimi-k3",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"role": "assistant", "content": "ok"}}],
        ),
        litellm.ModelResponseStream(
            id="gen-1",
            created=1,
            model="moonshotai/kimi-k3",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=usage,
        ),
    ]
    response = litellm.stream_chunk_builder(chunks)
    CustomStreamWrapper._propagate_usage_cost_to_hidden_params(response, "openrouter")
    return response


def _kwargs() -> dict:
    return {"model": MODEL, "litellm_params": {"custom_llm_provider": "openrouter"}}


def test_a_streamed_openrouter_answer_carries_its_bill():
    billed = _callback()["_provider_billed_cost"]
    assert billed(_kwargs(), _streamed_openrouter_response(0.00731)) == pytest.approx(
        0.00731
    )
    assert billed(_kwargs(), _streamed_openrouter_response(None)) is None
    assert (
        billed(
            {"model": "anthropic/claude-opus-5-5"}, _streamed_openrouter_response(1.0)
        )
        is None
    )


def _record(litellm_cost: float | None, billed: float | None) -> str:
    return json.dumps(
        {
            "event": "success",
            "request": {"body": {"model": MODEL, "stream": True}},
            "response": {
                "model": "moonshotai/kimi-k3",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 20,
                    "total_tokens": 420,
                    **({"cost": billed} if billed is not None else {}),
                },
            },
            "response_cost": litellm_cost,
            "provider_cost": billed,
        }
    )


def _trajectory(*lines: str):
    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log

    return trajectory_from_litellm_callback_log(
        "\n".join(lines), session_id="s", agent_name="opencode"
    )


def test_every_call_billed_costs_the_bill_and_each_call_keeps_its_own():
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory

    trajectory = _trajectory(_record(0.0073, 0.00731), _record(0.0050, 0.00502))
    usage = extract_usage_from_trajectory(trajectory)
    assert usage["cost_usd"] == pytest.approx(0.00731 + 0.00502)
    assert usage["cost_source"] == "provider"
    assert usage["price_source"] == "openrouter"
    assert "cost_usd_provider" not in usage
    assert [ex.metadata["provider_cost_usd"] for ex in trajectory.exchanges] == [
        0.00731,
        0.00502,
    ]


def test_calls_without_a_bill_are_estimated_beside_the_billed_part(tmp_path: Path):
    """Two calls billed, one not: the cost is the bill plus LiteLLM's list
    price of the third, an estimate; result.json says so."""
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory
    from benchflow.rollout import _build_rollout_result

    trajectory = _trajectory(
        _record(0.0073, 0.00731), _record(0.0050, 0.00502), _record(0.0040, None)
    )
    usage = extract_usage_from_trajectory(trajectory)
    assert usage["cost_usd"] == pytest.approx(0.00731 + 0.00502 + 0.0040)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["cost_usd_provider"] == pytest.approx(0.00731 + 0.00502)
    assert usage["n_calls_cost_estimated"] == 1

    _build_rollout_result(
        tmp_path,
        task_name="task",
        rollout_name="task__1",
        agent="opencode",
        agent_name="opencode",
        model=MODEL,
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[{"type": "agent_message", "text": "done"}],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime(2026, 11, 13, 6, 0),
        timing={},
        **usage,
    )
    agent = json.loads((tmp_path / "result.json").read_text())["agent_result"]
    assert agent["cost_source"] == "list_price_estimate"
    assert agent["cost_usd_provider"] == pytest.approx(0.00731 + 0.00502)
    assert agent["n_calls_cost_estimated"] == 1
    assert agent["usage_source"] == "proxy"


def test_a_call_without_a_bill_or_a_list_price_leaves_the_cost_unknown():
    from benchflow.providers.litellm_logging import extract_usage_from_trajectory

    usage = extract_usage_from_trajectory(
        _trajectory(_record(0.0073, 0.00731), _record(None, None))
    )
    assert usage["cost_usd"] is None
    assert usage["cost_usd_provider"] == pytest.approx(0.00731)
    assert usage["n_calls_cost_estimated"] == 1
