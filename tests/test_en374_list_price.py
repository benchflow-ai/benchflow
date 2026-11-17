"""One pricing step: every trial with tokens gets a cost (EN-374).

A Codex trial on a ChatGPT login, a Claude trial whose session log could not
be read, an agent's own count: each has tokens and, before EN-374, no cost.
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
    """PR #17's gpt-6-astra alpha-sh trial: Codex's native turns report
    OpenAI's input_tokens, cached ones included, which BenchFlow's split now
    leaves out at the source (44,881 uncached). BenchFlow prices it as the
    script outside it did."""
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


def test_a_codex_acp_report_leaves_the_cache_out_of_its_input():
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
    """Before EN-374 a trial with a proxied role and a login role recorded
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


def test_a_proxied_trial_the_gateway_could_not_price_is_not_priced_by_name():
    """The gateway counted every call and priced none of them (a private
    deployment of an open model): a list price for the bare model name would
    be its vendor's API, not this deployment's. The cost stays unknown, as
    on 0.10 (dc5d552c0)."""
    from benchflow.rollout._cost import apply_list_price

    rollout = _rollout(
        {
            "total_tokens": 11,
            "cost_usd": None,
            "usage_source": "provider_response",
            "usage_details": {},
        },
        _native_usage_slices={
            ("claude-opus-5-5", False): {"input_tokens": 10, "output_tokens": 1}
        },
    )
    apply_list_price(rollout)
    assert rollout._usage_metrics["cost_usd"] is None
    assert rollout._usage_metrics.get("cost_source") is None


@pytest.mark.parametrize(
    ("price_source", "model", "cost_source", "date"),
    [
        ("litellm", "xiaomi/mimo-v2.6-flash", "list_price_estimate", "2026-11-04"),
        ("litellm", "openrouter/moonshotai/kimi-k3", "list_price_estimate", None),
        # A snapshot model through OpenRouter got no snapshot rate: no date.
        (
            "litellm",
            "openrouter/anthropic/claude-opus-5-5",
            "list_price_estimate",
            None,
        ),
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


def _record(cost: float | None, billed: float | None) -> str:
    return json.dumps(
        {
            "event": "success",
            "request": {"body": {"model": "openrouter/moonshotai/kimi-k3"}},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 2,
                    "total_tokens": 42,
                },
            },
            "response_cost": cost,
            "provider_cost": billed,
        }
    )


def test_an_openrouter_billed_cost_is_the_provider_s():
    """Every answered call carries OpenRouter's own usage.cost: the trial's
    cost is their sum, labelled provider; one call without it falls back to
    LiteLLM's figure, a list-price estimate."""
    from benchflow.providers.litellm_logging import (
        extract_usage_from_trajectory,
        trajectory_from_litellm_callback_log,
    )

    billed = trajectory_from_litellm_callback_log(
        "\n".join([_record(0.01, 0.012), _record(0.01, 0.013)]),
        session_id="s",
        agent_name="opencode",
    )
    usage = extract_usage_from_trajectory(billed)
    assert usage["cost_usd"] == pytest.approx(0.025)
    assert usage["cost_source"] == "provider"
    assert usage["price_source"] == "openrouter"

    partly = trajectory_from_litellm_callback_log(
        "\n".join([_record(0.01, 0.012), _record(0.01, None)]),
        session_id="s",
        agent_name="opencode",
    )
    usage = extract_usage_from_trajectory(partly)
    # The lead's rule (EN-374): the billed part plus a list-price estimate of
    # the call without a bill, labelled an estimate.
    assert usage["cost_usd"] == pytest.approx(0.012 + 0.01)
    assert usage["cost_source"] == "list_price_estimate"
    assert usage["cost_usd_provider"] == pytest.approx(0.012)
    assert usage["n_calls_cost_estimated"] == 1


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


def test_an_in_place_branch_child_prices_its_own_turns(tmp_path, monkeypatch):
    """The Docker tier's in-place branch: a child drives the parent's
    rollout, and the parent's list price counted its children's turns too
    (0.01 against its own 0.005). A child's end-of-turn counts are its own,
    and the fork's restore gives the parent its own back (as 0.10's
    68eb47459)."""
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
