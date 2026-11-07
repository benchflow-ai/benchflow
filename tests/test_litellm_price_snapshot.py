"""Guards the pinned pricing and its stale-audit extension: the
gateway prices current first-party models at the vendors' published rates
(read 2026-11-04), cache reads included, and no reseller inherits them."""

from datetime import date
from typing import Any, cast

import pytest

from benchflow.providers.litellm_config import (
    MODEL_COST_PER_TOKEN,
    litellm_proxy_config,
    resolve_litellm_route,
)
from benchflow.providers.pricing_snapshot import PRICES, VALID_THROUGH

ENV = {"XIAOMI_BASE_URL": "https://api.xiaomimimo.com/v1"}

# Model as a user types it, then input, output and cache-read USD per token,
# each read on the vendor's pricing page on 2026-11-04.
VENDOR_RATES = [
    ("anthropic/claude-opus-5-5", 4e-6, 20e-6, 0.2e-6),
    ("anthropic/claude-sonnet-5-5", 2e-6, 10e-6, 0.1e-6),
    ("claude-sonnet-5", 2e-6, 10e-6, 0.2e-6),
    ("claude-haiku-5-5", 0.1e-6, 0.5e-6, 0.01e-6),
    ("anthropic/claude-fable-5-1", 10e-6, 50e-6, 0.25e-6),
    ("openai/gpt-6-astra", 10e-6, 50e-6, 1e-6),
    ("openai/gpt-6.1-sol", 2e-6, 10e-6, 0.1e-6),
    ("openai/gpt-6.1-terra", 1e-6, 5e-6, 0.05e-6),
    ("gpt-6-luna", 0.1e-6, 0.5e-6, 0.01e-6),
    ("openai/gpt-5.6-sol", 4e-6, 20e-6, 0.4e-6),
    ("gemini/gemini-4-argon", 2e-6, 10e-6, 0.1e-6),
    ("gemini/gemini-3.8-flash", 0.75e-6, 3.75e-6, 0.075e-6),
    ("gemini/gemini-3.5-flash-lite", 0.3e-6, 2.5e-6, 0.03e-6),
    ("deepseek/deepseek-flash", 0.3e-6, 1.2e-6, 0.006e-6),
    ("deepseek/deepseek-v4-flash", 0.3e-6, 1.2e-6, 0.006e-6),
    ("deepseek/deepseek-v4-pro", 1.32e-6, 3.96e-6, 0.044e-6),
    ("xiaomi/mimo-v2.6-pro", 0.435e-6, 0.87e-6, 0.0036e-6),
    ("xiaomi/mimo-v2.6-flash", 0.14e-6, 0.28e-6, 0.0028e-6),
]


def _entries(model: str) -> list[dict[str, Any]]:
    route = resolve_litellm_route(model, ENV)
    config = litellm_proxy_config(route, master_key="local-test")
    return cast(list[dict[str, Any]], config["model_list"])


@pytest.mark.parametrize("model,input_rate,output_rate,cache_rate", VENDOR_RATES)
def test_price_snapshot_reaches_every_alias(model, input_rate, output_rate, cache_rate):
    for entry in _entries(model):
        for where in (entry["litellm_params"], entry["model_info"]):
            assert where["input_cost_per_token"] == input_rate
            assert where["output_cost_per_token"] == output_rate
            assert where["cache_read_input_token_cost"] == cache_rate


def test_every_snapshot_entry_has_a_rate_test():
    tested = {m.rsplit("/", 1)[-1] for m, *_ in VENDOR_RATES}
    assert {model for _, model in PRICES} == tested


@pytest.mark.parametrize(
    "model",
    [
        "vllm/mimo-v2.6-pro",
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/anthropic/claude-opus-5-5",
    ],
)
def test_provider_price_does_not_apply_to_another_host(model):
    """A private server or a reseller of a vendor's model is not priced as the
    vendor. Also guards the removal of the unverified $0.28 / $0.42 DeepSeek V4
    placeholder, which every route whose model contained deepseek-v4 took."""
    env = {**ENV, "BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8000/v1"}
    route = resolve_litellm_route(model, env)
    config = litellm_proxy_config(route, master_key="local-test")
    entry = cast(list[dict[str, Any]], config["model_list"])[0]
    assert "input_cost_per_token" not in entry["litellm_params"]
    assert "model_info" not in entry
    assert MODEL_COST_PER_TOKEN == {}


def _cost(model: str, prompt: int, cached: int, completion: int) -> float:
    import litellm

    route = resolve_litellm_route(model, ENV)
    entry = _entries(model)[0]
    entry["litellm_params"]["api_key"] = "local-test"
    router = litellm.Router(model_list=[entry])
    deployments = router.get_model_list()
    assert deployments is not None
    response = litellm.ModelResponse(
        model=route.upstream_model,
        usage=litellm.Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details={"cached_tokens": cached},
        ),
    )
    return litellm.completion_cost(
        completion_response=response,
        custom_pricing=True,
        router_model_id=deployments[0]["model_info"]["id"],
        custom_llm_provider=route.upstream_model.split("/", 1)[0],
    )


@pytest.mark.parametrize("model,input_rate,output_rate,cache_rate", VENDOR_RATES)
def test_litellm_prices_cached_usage_with_the_deployment_snapshot(
    model, input_rate, output_rate, cache_rate
):
    """An 80% cache-hit call (1000 prompt tokens, 800 cached, 100 out) costs
    the vendor's rates in LiteLLM's own calculation."""
    expected = 200 * input_rate + 800 * cache_rate + 100 * output_rate
    assert _cost(model, 1000, 800, 100) == pytest.approx(expected)


def test_haiku_5_5_over_100k_tokens_costs_its_higher_rates_at_the_gateway():
    """LiteLLM's router drops the tier keys from litellm_params (the call cost
    $0.0155 instead of $0.0775); model_info keeps them."""
    assert _cost("claude-haiku-5-5", 150_000, 0, 1000) == pytest.approx(
        150_000 * 0.5e-6 + 1000 * 2.5e-6
    )
    assert _cost("claude-haiku-5-5", 150_000, 120_000, 1000) == pytest.approx(
        30_000 * 0.5e-6 + 120_000 * 0.05e-6 + 1000 * 2.5e-6
    )


def test_no_snapshot_rate_is_past_the_vendors_announced_change():
    """Google doubles Gemini 4 Argon's rates on 2026-12-01 and Gemini 3.8
    Flash's on 2027-01-01: after that the snapshot needs a dated refresh."""
    for key, last_day in VALID_THROUGH.items():
        assert key in PRICES
        assert date.today() <= date.fromisoformat(last_day), (
            f"{key} rates ended {last_day}; read the vendor's pricing page "
            "and update pricing_snapshot.PRICES, VALID_THROUGH and this date"
        )
