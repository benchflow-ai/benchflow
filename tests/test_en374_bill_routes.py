"""A bill is a bill only on a route that bills; OpenRouter's cache writes are
counted; a trial's estimated calls are never dropped (review of PR #122).

F2b-R122, 2026-11-13: M4 (per-call records called LiteLLM's own list price
a provider bill on every proxied route), M5 (OpenRouter's cache writes stayed
inside the uncached input), M6 (the bill lookup dropped another route's
estimated calls and labelled the trial ``provider``), L1 (a mixed proxy and
login trial said 0 calls were estimated). The rulings: one token split
(input is uncached input only; OpenRouter's ``prompt_tokens`` includes the
cached tokens, and the cache writes when it reports them inside it), and a
mixed cost is the billed part plus the list-price estimate of the unbilled
calls, ``provider`` only when every call was billed.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchflow.providers.litellm_logging import (
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)
from benchflow.rollout._cost import apply_list_price, resolve_provider_bills
from benchflow.trajectories.model_calls import model_calls
from benchflow.trajectories.types import _exchange_token_usage


def _record(model, usage, cost, *, rid, provider_cost=None, keyed=True, group=None):
    """One line of the proxy callback's log (``provider_cost`` is None on a
    route that is not OpenRouter's, or an answer that carried no bill)."""
    record = {
        "event": "success",
        "start_time": "2026-11-13T00:00:00",
        "end_time": "2026-11-13T00:00:01",
        "provider_model": model,
        "request_model": group or model,
        "model_group": group or model,
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {"model": group or model, "messages": []},
        },
        "response": {
            "id": rid,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": usage,
        },
        "response_cost": cost,
    }
    if keyed:
        record["provider_cost"] = provider_cost
    return record


def _trajectory(records):
    return trajectory_from_litellm_callback_log(
        "\n".join(json.dumps(r) for r in records),
        session_id="s",
        agent_name="opencode",
    )


def _calls(tmp_path, records):
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        _trajectory(records).to_jsonl()
    )
    return model_calls(trial)


USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}


@pytest.mark.parametrize(
    ("model", "keyed"),
    [
        ("openai/gpt-5.5", True),
        ("anthropic/claude-sonnet-5-5", True),
        # A record from before the callback wrote provider_cost.
        ("openai/gpt-5.5", False),
    ],
    ids=["openai", "anthropic-api-key", "older-record"],
)
def test_litellm_s_own_usage_cost_is_no_bill_per_call(tmp_path, model, keyed):
    """M4: the Docker tier's eval/hello-pass shape (claude-agent-acp through
    LiteLLM to a provider that is not OpenRouter): LiteLLM puts its list
    price in ``usage.cost``. The per-call record and result.json both call it
    a list-price estimate."""
    records = [
        _record(model, {**USAGE, "cost": 0.0065}, 0.0065, rid="r1", keyed=keyed),
        _record(model, {**USAGE, "cost": 0.0065}, 0.0065, rid="r2", keyed=keyed),
    ]
    calls = _calls(tmp_path, records)
    assert [c["cost_source"] for c in calls] == ["list_price_estimate"] * 2
    assert [c["cost_usd"] for c in calls] == [0.0065, 0.0065]
    trial = extract_usage_from_trajectory(_trajectory(records))
    assert trial["cost_usd"] == pytest.approx(0.013)
    assert trial.get("cost_source") in (None, "list_price_estimate")
    assert trial["price_source"] == "litellm"


def test_an_openrouter_bill_is_a_bill_per_call(tmp_path):
    model = "openrouter/moonshotai/kimi-k3"
    records = [
        _record(
            model, {**USAGE, "cost": 0.0141}, 0.0141, rid="g1", provider_cost=0.0141
        )
    ]
    calls = _calls(tmp_path, records)
    assert [(c["cost_usd"], c["cost_source"]) for c in calls] == [(0.0141, "provider")]
    trial = extract_usage_from_trajectory(_trajectory(records))
    assert (trial["cost_usd"], trial["cost_source"]) == (0.0141, "provider")


# OpenRouter's documented usage object (https://openrouter.ai/docs/use-cases/
# usage-accounting, read 2026-11-13): "prompt_tokens": 194, "completion_tokens":
# 2, "total_tokens": 196, "prompt_tokens_details": {"cached_tokens": 0,
# "cache_write_tokens": 100}. 194 + 2 = 196: the 100 tokens written to the
# cache are inside prompt_tokens, as the cached tokens read are
# (https://openrouter.ai/docs/features/prompt-caching: "prompt_tokens": 10339
# with "cached_tokens": 10318). R122's probe: prompt 10,000, cached 6,000,
# written 3,000 is 1,000 uncached input and 3,000 cache writes.
OPENROUTER_ANSWER = {
    "id": "gen-123",
    "object": "chat.completion",
    "model": "anthropic/claude-sonnet-5-5",
    "created": 1,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": 10000,
        "completion_tokens": 100,
        "total_tokens": 10100,
        "prompt_tokens_details": {"cached_tokens": 6000, "cache_write_tokens": 3000},
        "completion_tokens_details": {"reasoning_tokens": 0},
        "cost": 0.0123,
    },
}


def _litellm_copy(answer):
    """The answer as the pinned LiteLLM serializes it (its ModelResponse)."""
    import litellm
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )

    response = convert_to_model_response_object(
        response_object=json.loads(json.dumps(answer)),
        model_response_object=litellm.ModelResponse(),
    )
    return json.loads(json.dumps(response.model_dump(), default=str))


def _litellm_streamed(answer):
    """*answer* streamed as OpenRouter streams it (the text, then a last
    chunk with the usage), put together by the pinned LiteLLM as its success
    callback gets it (``stream_chunk_builder``)."""
    import litellm
    from litellm.main import stream_chunk_builder

    head = {key: answer[key] for key in ("id", "model", "created")}
    chunks = [
        {
            **head,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "o"}}],
        },
        {
            **head,
            "choices": [
                {"index": 0, "delta": {"content": "k"}, "finish_reason": "stop"}
            ],
        },
        {**head, "choices": [], "usage": answer["usage"]},
    ]
    return stream_chunk_builder(
        chunks=[
            litellm.ModelResponseStream(**json.loads(json.dumps(c))) for c in chunks
        ]
    )


def _responses_bridge(chat):
    """What LiteLLM's Responses API bridge answers for the chat completion
    *chat* (a ModelResponse); a streamed answer's ``response.completed`` is
    built from the put-together stream the same way."""
    import litellm
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    if isinstance(chat, dict):
        chat = litellm.ModelResponse(**chat)
    response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="hi", responses_api_request={}, chat_completion_response=chat
    )
    return json.loads(json.dumps(response.model_dump(), default=str))


def _dumped(response):
    return json.loads(json.dumps(response.model_dump(), default=str))


# Every OpenRouter route goes through LiteLLM; a Responses API client's
# through its bridge, which moves the writes to input_tokens_details.
SHAPES = {
    "openrouter": lambda: OPENROUTER_ANSWER,
    "litellm-copy": lambda: _litellm_copy(OPENROUTER_ANSWER),
    "streamed": lambda: _dumped(_litellm_streamed(OPENROUTER_ANSWER)),
    "responses-bridge": lambda: _responses_bridge(_litellm_copy(OPENROUTER_ANSWER)),
    "responses-bridge-streamed": lambda: _responses_bridge(
        _litellm_streamed(OPENROUTER_ANSWER)
    ),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_openrouter_cache_writes_leave_the_uncached_input(tmp_path, shape):
    body = SHAPES[shape]()
    model = "openrouter/anthropic/claude-sonnet-5-5"
    record = _record(model, body["usage"], 0.0123, rid="gen-123", provider_cost=0.0123)
    record["response"] = body
    if shape.startswith("responses"):
        assert body["usage"]["input_tokens_details"]["cache_write_tokens"] == 3000
        record["request"] = {
            "method": "POST",
            "path": "/v1/responses",
            "body": {"model": model, "input": "hi"},
        }
    trajectory = _trajectory([record])
    usage = _exchange_token_usage(trajectory.exchanges[0])
    assert usage.input_tokens == 1000
    assert usage.cache_read_tokens == 6000
    assert usage.cache_creation_tokens == 3000
    trial = extract_usage_from_trajectory(trajectory)
    assert trial["n_input_tokens"] == 1000
    assert trial["n_cache_creation_tokens"] == 3000
    assert trial["total_tokens"] == 10100
    calls = _calls(tmp_path, [record])
    assert calls[0]["cache_write_tokens"] == 3000
    assert calls[0]["cache_read_tokens"] == 6000


class _NoBill:
    """OpenRouter's record of an aborted stream: $0, no tokens."""

    status_code = 200

    def json(self):
        return {
            "data": {
                "total_cost": 0.0,
                "native_tokens_prompt": 0,
                "native_tokens_completion": 0,
            }
        }


class _Client:
    async def get(self, url, params=None, headers=None, timeout=None):
        return _NoBill()


def _resolved(records):
    metrics = extract_usage_from_trajectory(_trajectory(records))
    rollout = SimpleNamespace(
        _usage_metrics=dict(metrics),
        _agent_env={"OPENROUTER_API_KEY": "sk-or-v1-FAKEFAKEFAKE"},
    )
    asyncio.run(resolve_provider_bills(rollout, client=_Client()))
    return metrics, rollout._usage_metrics


KIMI = "openrouter/moonshotai/kimi-k3"
KIMI_GROUP = "runtime-openrouter-moonshotai-kimi-k3"
ONE_K = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
ABORTED = {"prompt_tokens": 2000, "completion_tokens": 0, "total_tokens": 2000}


def test_the_bill_lookup_keeps_another_route_s_estimate():
    """M6, R122's probe: an OpenRouter call billed $0.01, an aborted
    OpenRouter stream (estimate $0.004, its record says $0) and a call on
    another route at LiteLLM's $0.0065. The cost is $0.01 + $0 + $0.0065,
    a list-price estimate with one call estimated."""
    records = [
        _record(
            KIMI,
            {**ONE_K, "cost": 0.01},
            0.01,
            rid="gen-1",
            provider_cost=0.01,
            group=KIMI_GROUP,
        ),
        _record(KIMI, ABORTED, 0.004, rid="gen-2", group=KIMI_GROUP),
        _record(
            "openai/gpt-5.5",
            {**ONE_K, "cost": 0.0065},
            0.0065,
            rid="resp-3",
            group="runtime-openai-gpt-5.5",
        ),
    ]
    before, after = _resolved(records)
    assert before["cost_usd"] == pytest.approx(0.0205)
    assert before["n_calls_cost_estimated"] == 2
    assert after["cost_usd"] == pytest.approx(0.0165)
    assert after["cost_source"] == "list_price_estimate"
    assert after["cost_usd_provider"] == pytest.approx(0.01)
    assert after["n_calls_cost_estimated"] == 1


def test_the_bill_lookup_with_no_other_route_is_a_provider_bill():
    records = [
        _record(
            KIMI,
            {**ONE_K, "cost": 0.01},
            0.01,
            rid="gen-1",
            provider_cost=0.01,
            group=KIMI_GROUP,
        ),
        _record(KIMI, ABORTED, 0.004, rid="gen-2", group=KIMI_GROUP),
    ]
    _before, after = _resolved(records)
    assert after["cost_usd"] == pytest.approx(0.01)
    assert after["cost_source"] == "provider"
    assert after["n_calls_cost_estimated"] == 0


def test_the_bill_lookup_keeps_another_route_s_estimate_when_nothing_was_billed():
    """No call carried a bill before the lookup: the other route's call is
    still an estimate after it."""
    records = [
        _record(KIMI, ABORTED, 0.004, rid="gen-2", group=KIMI_GROUP),
        _record(
            "openai/gpt-5.5",
            {**ONE_K, "cost": 0.0065},
            0.0065,
            rid="resp-3",
            group="runtime-openai-gpt-5.5",
        ),
    ]
    _before, after = _resolved(records)
    assert after["cost_usd"] == pytest.approx(0.0065)
    assert after["cost_source"] == "list_price_estimate"
    assert after["n_calls_cost_estimated"] == 1


def test_a_mixed_proxy_and_login_trial_does_not_know_its_estimated_calls():
    """L1: an OpenRouter bill of $0.25 for the proxied role and a login's
    1,000/100 tokens priced at list price. How many of the login's calls
    were estimated is unknown: null, not the proxied part's 0."""
    provider = {
        "n_input_tokens": 500,
        "n_output_tokens": 50,
        "cost_usd": 0.25,
        "usage_source": "provider_response",
        "price_source": "openrouter",
        "cost_source": "provider",
        "cost_usd_provider": 0.25,
        "n_calls_cost_estimated": 0,
    }
    combined = {
        **provider,
        "n_input_tokens": 1500,
        "n_output_tokens": 150,
        "total_tokens": 1650,
        "usage_details": {
            "provider_response": provider,
            "agent_native_acp": {"n_input_tokens": 1000, "n_output_tokens": 100},
        },
        "cost_usd": None,
        "price_source": None,
        "cost_source": None,
        "price_table_date": None,
    }
    rollout = SimpleNamespace(
        _usage_metrics=combined,
        _config=SimpleNamespace(
            primary_agent="claude-agent-acp", harness="acp", model="claude-opus-5-5"
        ),
        _proxy_reported_models=["openrouter/anthropic/claude-opus-5-5"],
        _session_reported_models=[],
        _native_usage_slices={
            ("claude-opus-5-5", True): {"input_tokens": 1000, "output_tokens": 100}
        },
    )
    apply_list_price(rollout)
    usage = rollout._usage_metrics
    assert usage["cost_usd"] == pytest.approx(0.25 + 1000 * 4e-6 + 100 * 20e-6)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["cost_usd_provider"] == 0.25
    assert usage["n_calls_cost_estimated"] is None
