"""One pricing step: every trial with tokens gets a cost.

A Codex trial on a ChatGPT login, a Claude trial whose session log could not
be read, an agent's own count: each has tokens and, before that change, no cost.
frontier-design-benchmark PR #17's Codex rows got their cost_usd from a
script outside BenchFlow (gpt-6-astra, 44,881 uncached + 1,064,960 cached
input and 44,285 output tokens: $3.728 at the 2026-11-04 snapshot). Now
teardown prices them at list prices, from BenchFlow's dated snapshot, else
LiteLLM's table, and labels every cost: ``cost_source`` ``provider`` (a
provider's billed figure, an OpenRouter response's ``usage.cost``) or
``list_price_estimate``, with ``price_table_date``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _rollout(metrics: dict, **fields) -> SimpleNamespace:
    base = {
        "_usage_metrics": metrics,
        "_config": SimpleNamespace(
            primary_agent="codex-acp", harness="native", model="openai/gpt-6-astra"
        ),
        "_native_usage_slices": {},
        "_usage_slices": None,
        "_session_reported_models": [],
        "_proxy_reported_models": [],
    }
    base.update(fields)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "anthropic/claude-opus-5-5",
        "claude-opus-5-5[1m]",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "gpt-6-astra",
        "openai/gpt-6.1-sol",
        "xiaomi/mimo-v2.6-flash",
    ],
)
def test_current_models_have_dated_list_prices(model):
    from benchflow._utils.pricing import SNAPSHOT_TABLE, list_rates

    rates = list_rates(model)
    assert rates is not None
    assert rates.table == SNAPSHOT_TABLE
    assert rates.table_date == "2026-11-04"
    assert rates.input > 0 and rates.output > rates.input


def test_a_model_no_table_has_stays_unpriced():
    from benchflow._utils.pricing import TokenSlice, list_rates, price_slices

    assert list_rates("acme/never-released-9") is None
    assert price_slices([TokenSlice(model="acme/never-released-9", input=10)]) is None
    assert price_slices([]) is None


def test_a_codex_login_trial_on_the_native_harness_is_priced():
    """PR #17's gpt-6-astra alpha-sh trial (1,109,841 input of which 1,064,960
    cached, 44,285 output): BenchFlow now prices it as the script outside it
    did. Codex's native harness takes the cache out of its input at the
    source, so the end-of-turn slice holds 44,881 uncached tokens."""
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {
            "n_input_tokens": 44881,
            "n_output_tokens": 44285,
            "n_cache_read_tokens": 1064960,
            "n_cache_creation_tokens": 0,
            "total_tokens": 1154126,
            "cost_usd": None,
            "usage_source": "agent_native_acp",
            "price_source": None,
        },
        _native_usage_slices={
            ("openai/gpt-6-astra", True): {
                "input_tokens": 44881,
                "output_tokens": 44285,
                "cached_read_tokens": 1064960,
                "cached_write_tokens": 0,
            }
        },
    )
    apply_list_price(rollout)
    usage = rollout._usage_metrics
    assert usage["cost_usd"] == pytest.approx(3.72802)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_source"] == "list_price"
    assert usage["price_table_date"] == "2026-11-04"
    assert usage["usage_details"]["cost_estimate"]["models"] == {
        "openai/gpt-6-astra": pytest.approx(3.72802)
    }


def test_an_end_of_turn_report_is_priced_without_the_cache_in_its_input():
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {"total_tokens": 10, "cost_usd": None, "usage_source": "agent_native_acp"},
        _config=SimpleNamespace(
            primary_agent="codex-acp", harness="acp", model="gpt-6.1-sol"
        ),
        _native_usage_slices={
            ("gpt-6.1-sol", True): {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cached_read_tokens": 4000,
                "cached_write_tokens": 0,
            }
        },
    )
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] == pytest.approx(
        1000 * 2e-6 + 100 * 10e-6 + 4000 * 0.1e-6
    )


def test_an_unnamed_model_is_priced_as_its_log_names_it():
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {"total_tokens": 10, "cost_usd": None, "usage_source": "agent_native_acp"},
        _config=SimpleNamespace(
            primary_agent="claude-agent-acp", harness="acp", model=None
        ),
        _session_reported_models=["claude-sonnet-5-5"],
        _native_usage_slices={
            ("", True): {
                "input_tokens": 10,
                "output_tokens": 10,
                "cached_read_tokens": 0,
                "cached_write_tokens": 0,
            }
        },
    )
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] == pytest.approx(10 * 2e-6 + 10 * 10e-6)


def test_a_mixed_trial_costs_its_proxied_and_login_parts_together():
    """Before a trial with a proxied role and a login role recorded
    cost None. Now: the gateway's price of its part plus the login's tokens
    at list prices."""
    from benchflow.rollout._cost import apply_list_price

    provider = {
        "n_input_tokens": 500,
        "n_output_tokens": 50,
        "cost_usd": 0.25,
        "usage_source": "provider_response",
        "price_source": "litellm",
    }
    rollout = _rollout(
        {
            "n_input_tokens": 1500,
            "n_output_tokens": 150,
            "total_tokens": 1650,
            "cost_usd": None,
            "price_source": None,
            "usage_source": "provider_response",
            "usage_details": {
                "provider_response": provider,
                "agent_native_acp": {"n_input_tokens": 1000, "n_output_tokens": 100},
            },
        },
        _config=SimpleNamespace(
            primary_agent="claude-agent-acp", harness="acp", model="claude-opus-5-5"
        ),
        _proxy_reported_models=["claude-opus-5-5"],
        _native_usage_slices={
            ("anthropic/claude-opus-5-5", False): {
                "input_tokens": 500,
                "output_tokens": 50,
                "cached_read_tokens": 0,
                "cached_write_tokens": 0,
            },
            ("claude-opus-5-5", True): {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cached_read_tokens": 0,
                "cached_write_tokens": 0,
            },
        },
    )
    apply_list_price(rollout)
    usage = rollout._usage_metrics
    login = 1000 * 4e-6 + 100 * 20e-6
    assert usage["cost_usd"] == pytest.approx(0.25 + login)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_table_date"] == "2026-11-04"
    assert usage["usage_details"]["cost_estimate"]["provider_response_usd"] == 0.25


def test_a_mixed_trial_whose_proxied_part_has_no_price_stays_unpriced():
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {
            "total_tokens": 10,
            "cost_usd": None,
            "usage_source": "provider_response",
            "usage_details": {"provider_response": {"cost_usd": None}},
        },
        _native_usage_slices={
            ("claude-opus-5-5", True): {"input_tokens": 10, "output_tokens": 1}
        },
    )
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] is None


@pytest.mark.parametrize(
    ("price_source", "model", "cost_source", "date"),
    [
        ("litellm", "xiaomi/mimo-v2.6-flash", "list_price_estimate", "2026-11-04"),
        ("litellm", "openrouter/moonshotai/kimi-k3", "list_price_estimate", None),
        ("provider", "openrouter/moonshotai/kimi-k3", "provider", None),
    ],
)
def test_a_cost_someone_else_set_gets_its_label(price_source, model, cost_source, date):
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {
            "total_tokens": 10,
            "cost_usd": 0.5,
            "usage_source": "provider_response",
            "price_source": price_source,
        },
        _config=SimpleNamespace(primary_agent="opencode", harness="acp", model=model),
    )
    apply_list_price(rollout)
    usage = rollout._usage_metrics
    assert usage["cost_usd"] == 0.5
    assert usage["cost_source"] == cost_source
    assert usage["price_table_date"] == date


def test_no_tokens_no_cost():
    from benchflow.rollout._cost import apply_list_price
    from benchflow.usage_tracking import usage_unavailable

    rollout = _rollout(usage_unavailable())
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] is None
    assert rollout._usage_metrics.get("cost_source") is None


def _record(
    cost: float | None,
    billed: float | None,
    *,
    stream: bool = False,
    generation: str | None = None,
) -> str:
    """One proxy callback line for an OpenRouter call: LiteLLM's
    response_cost beside the response OpenRouter sent, whose usage carries
    its own bill (``usage.cost``; a streamed response carries it in its
    final chunk, which LiteLLM's assembled response keeps)."""
    usage: dict = {
        "prompt_tokens": 47355,
        "completion_tokens": 448,
        "total_tokens": 47803,
    }
    if billed is not None:
        usage.update(cost=billed, is_byok=False)
    return json.dumps(
        {
            "event": "success",
            "provider_model": "openrouter/google/gemini-4-argon",
            "request": {
                "body": {"model": "openrouter/google/gemini-4-argon", "stream": stream}
            },
            "response": {
                "id": generation or "gen-billed",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": usage,
            },
            "response_cost": cost,
        }
    )


def _proxy_usage(lines: list[str]) -> dict:
    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    return extract_usage_from_trajectory(
        trajectory_from_litellm_callback_log(
            "\n".join(lines), session_id="s", agent_name="opencode"
        )
    )


def test_an_openrouter_bill_is_the_cost():
    """frontier-design-benchmark PR #17 (rc2, OpenCode through the proxy to
    OpenRouter, every request streamed): cost_usd equalled the sum of
    OpenRouter's usage.cost, but was labelled price_source litellm, so a
    reader replaced it with list price. The responses carried the bill: the
    cost is the provider's, cost_source provider, priced by openrouter."""
    usage = _proxy_usage(
        [_record(0.01, 0.012, stream=True), _record(0.01, 0.013, stream=True)]
    )
    assert usage["cost_usd"] == pytest.approx(0.025)
    assert usage["cost_source"] == "provider"
    assert usage["price_source"] == "openrouter"
    assert usage["cost_usd_provider"] == pytest.approx(0.025)
    assert usage["n_calls_cost_estimated"] == 0


@pytest.mark.parametrize("callback", ["current", "older"])
def test_litellm_s_own_usage_cost_is_no_bill(callback):
    """LiteLLM's proxy puts its own list-price ``usage.cost`` in the answers
    it returns (the Docker tier's codex-acp record through the fake: 0.0065
    for 1,000 + 50 tokens of gpt-5.5, LiteLLM's price). Off OpenRouter it is
    the gateway's estimate, never a provider's bill, whether the callback
    recorded ``provider_cost`` (None) or predates it."""
    record = {
        "event": "success",
        "provider_model": "openai/gpt-5.5",
        "request": {"body": {"model": "openai/gpt-5.5"}},
        "response": {
            "id": "resp_1",
            "object": "response",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "total_tokens": 1050,
                "cost": 0.0065,
            },
        },
        "response_cost": 0.0065,
    }
    if callback == "current":
        record["provider_cost"] = None
    usage = _proxy_usage([json.dumps(record), json.dumps(record)])
    assert usage["cost_usd"] == pytest.approx(0.013)
    assert usage["price_source"] == "litellm"
    assert "cost_usd_provider" not in usage
    assert usage.get("cost_source") != "provider"


# rc2's OpenRouter lines (FDB PR #17, OpenCode on kimi-k3; their usage.cost
# matched OpenRouter's /generation bill): LiteLLM logged the upstream model
# without its prefix, and the route is in BenchFlow's group name. rc3's MiMo
# line: its usage.cost is LiteLLM's list price.
RC2_OPENROUTER = {
    "request_model": "runtime-openrouter-moonshotai-kimi-k3",
    "provider_model": "moonshotai/kimi-k3",
    "model_group": "runtime-openrouter-moonshotai-kimi-k3",
    "body_model": "runtime-openrouter-moonshotai-kimi-k3",
    "usage": {
        "prompt_tokens": 21877,
        "completion_tokens": 312,
        "total_tokens": 22189,
        "cost": 0.0141,
    },
}
RC3_MIMO = {
    "request_model": "mimo-v2.6-flash",
    "provider_model": "mimo-v2.6-flash",
    "model_group": "runtime-xiaomi-mimo-v2.6-flash",
    "body_model": "mimo-v2.6-flash",
    "usage": {
        "prompt_tokens": 11041,
        "completion_tokens": 150,
        "total_tokens": 11191,
        "prompt_tokens_details": {"cached_tokens": 6656},
        "cost": 0.0006745368,
    },
}


def _callback_line(shape: dict) -> dict:
    """The proxy callback's line, as a callback before provider_cost wrote it."""
    return {
        "event": "success",
        "request_model": shape["request_model"],
        "provider_model": shape["provider_model"],
        "model_group": shape["model_group"],
        "request": {"body": {"model": shape["body_model"]}},
        "response": {"id": "gen-1", "usage": dict(shape["usage"])},
        "response_cost": shape["usage"]["cost"],
    }


def _trajectory_record(shape: dict) -> dict:
    """The same call in llm_trajectory.jsonl: the model fields under metadata."""
    return {
        "metadata": {
            "request_model": shape["request_model"],
            "provider_model": shape["provider_model"],
            "model_group": shape["model_group"],
            "response_cost": shape["usage"]["cost"],
        },
        "request": {"body": {"model": shape["body_model"]}},
        "response": {"body": {"usage": dict(shape["usage"])}},
    }


@pytest.mark.parametrize(
    ("shape", "billed"),
    [(RC2_OPENROUTER, 0.0141), (RC3_MIMO, None)],
    ids=["rc2-openrouter", "rc3-mimo"],
)
def test_the_route_is_read_from_benchflow_s_group_name(shape, billed):
    from benchflow.providers.litellm_logging import billed_cost, route_provider

    line = _callback_line(shape)
    assert billed_cost(line["response"], line) == billed
    record = _trajectory_record(shape)
    assert billed_cost(record["response"]["body"], record) == billed
    expected = "openrouter" if billed else "xiaomi"
    assert route_provider(line) == route_provider(record) == expected

    usage = _proxy_usage([json.dumps(line), json.dumps(line)])
    if billed:
        assert usage["cost_source"] == "provider"
        assert usage["price_source"] == "openrouter"
        assert usage["cost_usd_provider"] == pytest.approx(2 * billed)
    else:
        assert "cost_usd_provider" not in usage
        assert usage["price_source"] == "litellm"


@pytest.mark.parametrize(
    ("fields", "route"),
    [
        # OpenRouter's model prefix.
        ({"provider_model": "openrouter/moonshotai/kimi-k3"}, "openrouter"),
        # 0.10's group, the upstream model without a prefix (rc2).
        (
            {
                "model_group": "runtime-openrouter-moonshotai-kimi-k3",
                "provider_model": "moonshotai/kimi-k3",
            },
            "openrouter",
        ),
        # 0.9.x's group, its model id openai/<vendor>/<model>.
        (
            {
                "model_group": "benchflow-openrouter-moonshotai-kimi-k3",
                "provider_model": "openai/moonshotai/kimi-k3",
            },
            "openrouter",
        ),
        # 0.9.x's model id with OpenRouter's endpoint logged.
        (
            {
                "provider_model": "openai/moonshotai/kimi-k3",
                "litellm_params": {"api_base": "https://openrouter.ai/api/v1"},
            },
            "openrouter",
        ),
        # MiMo on either line, and another route: no bill.
        (
            {
                "model_group": "runtime-xiaomi-mimo-v2.6-flash",
                "provider_model": "mimo-v2.6-flash",
            },
            "xiaomi",
        ),
        (
            {
                "model_group": "benchflow-xiaomi-mimo-v2.6-flash",
                "provider_model": "openai/mimo-v2.6-flash",
                "litellm_params": {"api_base": "https://api.xiaomimimo.com/v1"},
            },
            "xiaomi",
        ),
        ({"provider_model": "anthropic/claude-opus-5-5"}, "anthropic"),
    ],
    ids=[
        "openrouter-prefix",
        "0.10-group",
        "0.9-group",
        "0.9-api-base",
        "mimo-0.10",
        "mimo-0.9",
        "anthropic",
    ],
)
def test_every_line_s_openrouter_route_is_a_bill(fields, route):
    """The backfill reads jobs from 0.9.x and 0.10 alike: OpenRouter's
    usage.cost is a bill whichever way the line named the route; any other
    route's usage.cost is LiteLLM's list price."""
    from benchflow.providers.litellm_logging import billed_cost, route_provider

    record = {**fields, "response": {"usage": {"cost": 0.0141}}}
    assert route_provider(record) == route
    expected = 0.0141 if route == "openrouter" else None
    assert billed_cost(record["response"], record) == expected
    # The same call as a trajectory record: its fields under metadata.
    nested = {"metadata": dict(fields)}
    assert billed_cost({"usage": {"cost": 0.0141}}, nested) == expected


def test_the_callback_knows_openrouter_by_its_group_name():
    from benchflow.providers.litellm_logging import callback_module_source

    namespace: dict = {}
    exec(callback_module_source(), namespace)
    kwargs = {
        "model": "moonshotai/kimi-k3",
        "litellm_params": {
            "metadata": {"model_group": "runtime-openrouter-moonshotai-kimi-k3"}
        },
    }
    assert (
        namespace["_provider_billed_cost"](kwargs, {"usage": {"cost": 0.0141}})
        == 0.0141
    )
    kwargs["litellm_params"]["metadata"]["model_group"] = (
        "runtime-xiaomi-mimo-v2.6-flash"
    )
    assert (
        namespace["_provider_billed_cost"](kwargs, {"usage": {"cost": 0.0141}}) is None
    )
    # 0.9.x's route: OpenRouter's endpoint.
    kwargs["litellm_params"]["api_base"] = "https://openrouter.ai/api/v1"
    assert (
        namespace["_provider_billed_cost"](kwargs, {"usage": {"cost": 0.0141}})
        == 0.0141
    )


def test_three_of_thirty_four_calls_without_a_bill_are_estimated():
    """The PR #17 Argon trial: 3 of 34 responses (status 200) carried no
    usage.cost, and OpenRouter's bill equals the other 31. The cost is the
    billed part plus a list-price estimate for the 3, labelled an estimate,
    with the billed part and the estimated calls beside it."""
    billed = [_record(0.07, 0.0723, stream=True) for _ in range(31)]
    unbilled = [_record(0.0701, None, stream=True) for _ in range(3)]
    usage = _proxy_usage(billed[:15] + unbilled + billed[15:])
    assert usage["cost_usd_provider"] == pytest.approx(31 * 0.0723)
    assert usage["n_calls_cost_estimated"] == 3
    assert usage["cost_usd"] == pytest.approx(31 * 0.0723 + 3 * 0.0701)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["price_source"] == "litellm"


def test_a_call_with_no_bill_and_no_price_leaves_the_cost_unknown():
    usage = _proxy_usage([_record(None, 0.012), _record(None, None)])
    assert usage["cost_usd"] is None
    assert usage["cost_usd_provider"] == pytest.approx(0.012)
    assert usage["n_calls_cost_estimated"] == 1


def test_each_model_call_record_names_its_cost_source(tmp_path):
    """trajectory/model-calls.jsonl had cost_usd null on every call though
    every response carried usage.cost: each call now has its cost and its
    cost_source."""
    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
    from benchflow.trajectories.model_calls import _proxy_calls

    trajectory = trajectory_from_litellm_callback_log(
        "\n".join([_record(0.07, 0.0723, stream=True), _record(0.0701, None)]),
        session_id="s",
        agent_name="opencode",
    )
    log = tmp_path / "llm_trajectory.jsonl"
    log.write_text(
        "".join(
            json.dumps(ex.model_dump(mode="json")) + "\n" for ex in trajectory.exchanges
        )
    )
    calls = _proxy_calls(log)
    assert [c["cost_usd"] for c in calls] == [
        pytest.approx(0.0723),
        pytest.approx(0.0701),
    ]
    assert [c["cost_source"] for c in calls] == ["provider", "list_price_estimate"]


def test_billed_parts_sum_in_the_summary():
    from benchflow._utils.evaluation_results import usage_summary

    def trial(cost, billed, estimated, source):
        return {
            "rewards": {"reward": 1.0},
            "agent_result": {
                "usage_source": "proxy",
                "n_input_tokens": 10,
                "n_output_tokens": 1,
                "total_tokens": 11,
                "cost_usd": cost,
                "cost_source": source,
                "cost_usd_provider": billed,
                "n_calls_cost_estimated": estimated,
            },
        }

    summary = usage_summary(
        {
            "a": trial(2.453161, 2.453161, 0, "provider"),
            "b": trial(2.45, 2.24, 3, "list_price_estimate"),
        }
    )
    assert summary["total_cost_usd"] == pytest.approx(2.453161 + 2.45)
    assert summary["total_cost_usd_provider"] == pytest.approx(2.453161 + 2.24)
    assert summary["n_calls_cost_estimated"] == 3
    assert summary["cost_sources"] == {"list_price_estimate": 1, "provider": 1}


@pytest.mark.parametrize(
    ("kwargs", "response", "expected"),
    [
        (
            {"model": "openrouter/moonshotai/kimi-k3"},
            {"usage": {"cost": 0.0042}},
            0.0042,
        ),
        (
            {"model": "kimi", "litellm_params": {"custom_llm_provider": "openrouter"}},
            SimpleNamespace(usage=SimpleNamespace(cost=0.5)),
            0.5,
        ),
        ({"model": "anthropic/claude-opus-5-5"}, {"usage": {"cost": 1.0}}, None),
        ({"model": "openrouter/x"}, {"usage": {"cost": True}}, None),
        ({"model": "openrouter/x"}, {"usage": {}}, None),
    ],
)
def test_only_an_openrouter_response_s_cost_is_a_billed_figure(
    kwargs, response, expected
):
    """The proxy's callback (the module BenchFlow writes beside LiteLLM's
    config) records the billed figure beside LiteLLM's response_cost."""
    from benchflow.providers.litellm_logging import callback_module_source

    namespace: dict = {}
    exec(callback_module_source(), namespace)
    assert namespace["_provider_billed_cost"](kwargs, response) == expected


def _argon_with_three_aborted() -> dict:
    """The Argon trial's proxy usage: 31 billed calls, then the 3 calls of
    one aborted request whose streams carried no usage.cost."""
    billed = [_record(0.07, 0.0723, stream=True) for _ in range(31)]
    aborted = [
        _record(0.0701, None, stream=True, generation=f"gen-aborted-{n}")
        for n in range(3)
    ]
    return _proxy_usage(billed + aborted)


async def _resolve(usage: dict, handler) -> tuple[dict, list]:
    import httpx

    from benchflow.rollout._cost import resolve_provider_bills

    seen: list = []

    def recording(request):
        seen.append(request)
        return handler(request)

    rollout = SimpleNamespace(
        _usage_metrics=usage, _agent_env={"OPENROUTER_API_KEY": "sk-or-v1-test-0000"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(recording)) as client:
        await resolve_provider_bills(rollout, client=client)
    return rollout._usage_metrics, seen


def _generation(cost: float, prompt: int, completion: int):
    import httpx

    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": request.url.params["id"],
                    "total_cost": cost,
                    "native_tokens_prompt": prompt,
                    "native_tokens_completion": completion,
                    "native_tokens_cached": 0,
                }
            },
        )

    return handler


@pytest.fixture
def _no_lookup_delay(monkeypatch):
    from benchflow.rollout import _cost

    monkeypatch.setattr(_cost, "BILL_LOOKUP_DELAY_SEC", 0.0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_lookup_delay")
async def test_an_aborted_stream_s_bill_is_looked_up():
    """OpenRouter keeps a generation record for each call: one whose stream
    carried no usage.cost is looked up by its id, and its bill is the
    provider's (cost_source provider, no estimated call)."""
    usage, seen = await _resolve(
        _argon_with_three_aborted(), _generation(0.0702, 47355, 448)
    )
    assert usage["cost_source"] == "provider"
    assert usage["price_source"] == "openrouter"
    assert usage["cost_usd"] == pytest.approx(31 * 0.0723 + 3 * 0.0702)
    assert usage["n_calls_cost_estimated"] == 0
    assert [r.url.params["id"] for r in seen] == [f"gen-aborted-{n}" for n in range(3)]
    # The key travels in the header, never in the URL.
    assert all(r.headers["authorization"] == "Bearer sk-or-v1-test-0000" for r in seen)
    assert all("sk-or" not in str(r.url) for r in seen)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_lookup_delay")
async def test_an_aborted_stream_billed_nothing_costs_nothing():
    """The PR #17 Argon trial: OpenRouter's records show the 3 aborted calls
    billed $0 with 0 tokens. That cost is known: the trial costs its 31
    billed calls, labelled provider, and the 3 calls' tokens are the
    record's 0."""
    before = _argon_with_three_aborted()
    usage, _ = await _resolve(dict(before), _generation(0.0, 0, 0))
    assert usage["cost_usd"] == pytest.approx(31 * 0.0723)
    assert usage["cost_source"] == "provider"
    assert usage["n_calls_cost_estimated"] == 0
    assert usage["n_output_tokens"] == before["n_output_tokens"] - 3 * 448


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_lookup_delay")
@pytest.mark.parametrize("failure", ["404", "timeout"])
async def test_a_failed_lookup_keeps_the_estimate(failure):
    """No record (404 after a few tries) or no answer (a timeout): the call
    keeps its list-price estimate, labelled as such and counted."""
    import httpx

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(404, json={"error": {"message": "Generation not found"}})

    usage, seen = await _resolve(_argon_with_three_aborted(), handler)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["n_calls_cost_estimated"] == 3
    assert usage["cost_usd_provider"] == pytest.approx(31 * 0.0723)
    assert usage["cost_usd"] == pytest.approx(31 * 0.0723 + 3 * 0.0701)
    assert len(seen) == 3 * 3  # each id tried three times: the record can lag


def test_an_in_place_branch_child_prices_its_own_turns(tmp_path, monkeypatch):
    """The Docker tier's in-place branch: a child drives the parent's
    rollout, and its list price counted the parent's turns too (0.01 against
    its own 0.005). A child's end-of-turn counts are its own, and the fork's
    restore gives the parent its own back."""
    from benchflow.branch_result import scope_child_result_state
    from benchflow.rollout._cost import apply_list_price
    from benchflow.rollout_branch import _LinearState
    from tests.test_en374_salvage import OPUS, OPUS_RATES
    from tests.test_en374_salvage import _rollout as _real_rollout

    rollout, _ = _real_rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=tmp_path
    )
    rollout._session_reported_models = [OPUS]
    rollout._record_native_usage_slice(
        "claude-agent-acp", {"input_tokens": 3000, "output_tokens": 150}
    )
    parent_slices = dict(rollout._native_usage_slices)
    saved = _LinearState.capture(rollout)

    scope_child_result_state(rollout)
    assert rollout._native_usage_slices == {}
    rollout._record_native_usage_slice(
        "claude-agent-acp", {"input_tokens": 1000, "output_tokens": 50}
    )
    rollout._usage_metrics = {
        **rollout._usage_metrics,
        "n_input_tokens": 1000,
        "n_output_tokens": 50,
        "total_tokens": 1050,
        "usage_source": "agent_native_acp",
        "cost_usd": None,
    }
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] == pytest.approx(
        1000 * OPUS_RATES[0] + 50 * OPUS_RATES[1]
    )

    saved.restore_onto(rollout)
    assert rollout._native_usage_slices == parent_slices
