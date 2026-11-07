"""Standard API list prices for current first-party models (USD per token).

The gateway reads LiteLLM's installed price table (LITELLM_LOCAL_MODEL_COST_MAP).
The pinned LiteLLM 1.103.3 table has half of these models, at these rates, and
lacks the other half (Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5, GPT-6.1 Sol
and Terra, GPT-6 Luna, Gemini 4 Argon, MiMo v2.6 Pro and Flash): a route to
one of those would record no cost. Every rate below was read on 2026-11-04 on
the vendor's own pricing page and matches LiteLLM main's table at commit
574f0659 (2026-11-04):

- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
  Cache writes are the 5-minute and 1-hour rates. Haiku 5.5 charges more for
  a prompt over 100,000 tokens (cache reads and writes count toward it).
- OpenAI: https://developers.openai.com/api/docs/pricing, the Standard tier
  (cached input, 30-minute cache write). OpenAI also prices prompts over 272K
  tokens higher; that tier is not in this snapshot, so such a request is
  under-counted.
- Google: https://ai.google.dev/gemini-api/docs/pricing, the paid tier.
  Gemini 4 Argon's rates double on 2026-12-01 and Gemini 3.8 Flash's on
  2027-01-01 (VALID_THROUGH).
- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing, at peak rates.
  Off-peak hours bill half, so a DeepSeek cost is an upper bound.
  ``deepseek-v4-flash`` is a retired name that DeepSeek bills as Flash.
- MiMo: https://platform.xiaomimimo.com/static/docs/price/pay-as-you-go.md

These are reproducible estimates, not subscription spend or negotiated rates.
Refresh them by hand, with the date and sources, rather than downloading a
mutable table at proxy startup.
"""

from __future__ import annotations

# Only exact vendor/model identities qualify. In particular, a private vLLM
# deployment or a reseller of one of these models does not inherit vendor rates.
_HAIKU_5_5_OVER_100K = {
    "input_cost_per_token_above_100k_tokens": 0.5e-6,
    "output_cost_per_token_above_100k_tokens": 2.5e-6,
    "cache_read_input_token_cost_above_100k_tokens": 0.05e-6,
    "cache_creation_input_token_cost_above_100k_tokens": 0.625e-6,
    "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-6,
}


def _anthropic(
    inp: float, out: float, read: float, write_5m: float, write_1h: float
) -> dict[str, float]:
    return {
        "input_cost_per_token": inp,
        "output_cost_per_token": out,
        "cache_read_input_token_cost": read,
        "cache_creation_input_token_cost": write_5m,
        "cache_creation_input_token_cost_above_1hr": write_1h,
    }


def _rates(
    inp: float, out: float, read: float, write: float | None = None
) -> dict[str, float]:
    rates = {
        "input_cost_per_token": inp,
        "output_cost_per_token": out,
        "cache_read_input_token_cost": read,
    }
    if write is not None:
        rates["cache_creation_input_token_cost"] = write
    return rates


PRICES: dict[tuple[str, str], dict[str, float]] = {
    ("anthropic", "claude-fable-5-1"): _anthropic(
        10e-6, 50e-6, 0.25e-6, 12.5e-6, 20e-6
    ),
    ("anthropic", "claude-opus-5-5"): _anthropic(4e-6, 20e-6, 0.2e-6, 5e-6, 8e-6),
    ("anthropic", "claude-sonnet-5-5"): _anthropic(2e-6, 10e-6, 0.1e-6, 2.5e-6, 4e-6),
    ("anthropic", "claude-sonnet-5"): _anthropic(2e-6, 10e-6, 0.2e-6, 2.5e-6, 4e-6),
    ("anthropic", "claude-haiku-5-5"): {
        **_anthropic(0.1e-6, 0.5e-6, 0.01e-6, 0.125e-6, 0.2e-6),
        **_HAIKU_5_5_OVER_100K,
    },
    ("openai", "gpt-6-astra"): _rates(10e-6, 50e-6, 1e-6, 12.5e-6),
    ("openai", "gpt-6.1-sol"): _rates(2e-6, 10e-6, 0.1e-6, 2.5e-6),
    ("openai", "gpt-6.1-terra"): _rates(1e-6, 5e-6, 0.05e-6, 1.25e-6),
    ("openai", "gpt-6-luna"): _rates(0.1e-6, 0.5e-6, 0.01e-6, 0.125e-6),
    ("openai", "gpt-5.6-sol"): _rates(4e-6, 20e-6, 0.4e-6, 5e-6),
    ("gemini", "gemini-4-argon"): _rates(2e-6, 10e-6, 0.1e-6),
    ("gemini", "gemini-3.8-flash"): _rates(0.75e-6, 3.75e-6, 0.075e-6),
    ("gemini", "gemini-3.5-flash-lite"): _rates(0.3e-6, 2.5e-6, 0.03e-6),
    ("deepseek", "deepseek-flash"): _rates(0.3e-6, 1.2e-6, 0.006e-6),
    ("deepseek", "deepseek-v4-flash"): _rates(0.3e-6, 1.2e-6, 0.006e-6),
    ("deepseek", "deepseek-v4-pro"): _rates(1.32e-6, 3.96e-6, 0.044e-6),
    ("xiaomi", "mimo-v2.6-pro"): _rates(0.435e-6, 0.87e-6, 0.0036e-6),
    ("xiaomi", "mimo-v2.6-flash"): _rates(0.14e-6, 0.28e-6, 0.0028e-6),
}

# Rates the vendor has announced will change: refresh the entry by this date.
VALID_THROUGH: dict[tuple[str, str], str] = {
    ("gemini", "gemini-4-argon"): "2026-11-30",
    ("gemini", "gemini-3.8-flash"): "2026-12-31",
}

# An ``openai/`` or ``deepseek/`` upstream also names OpenAI-compatible
# endpoints of other hosts, so those families are priced only on the
# vendor's own routes.
_FIRST_PARTY_PROVIDERS = {
    "openai": {"openai", "native"},
    "deepseek": {"deepseek"},
}


def model_prices(provider: str, upstream: str) -> dict[str, float]:
    """Get the snapshot for a first-party route, leaving unknowns unpriced."""
    family, _, model = upstream.partition("/")
    if provider == "xiaomi":
        family = provider
    elif family in _FIRST_PARTY_PROVIDERS:
        if provider not in _FIRST_PARTY_PROVIDERS[family]:
            return {}
    elif family not in {"anthropic", "gemini"}:
        return {}
    return dict(PRICES.get((family, model), {}))


def anthropic_prices(model: str) -> dict[str, float] | None:
    """The snapshot for a bare Claude model id (Claude Code's session logs)."""
    entry = PRICES.get(("anthropic", model))
    return dict(entry) if entry else None
