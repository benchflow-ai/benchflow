"""Official list prices per model and a post-hoc cost function.

BenchFlow records *usage* (input / output / cache-read / cache-creation tokens)
for every run mode and computes *cost* afterwards from this table, so a wrong or
missing price never changes what was recorded and can always be re-applied.

Prices are USD per million tokens (MTok). ``PRICING_VERSION`` stamps every cost
figure derived from :data:`LIST_PRICES` so a later price change is traceable.

Sources for the list table (checked 2026-09-07):
- Anthropic first-party API list prices, cached model table 2026-06-24
  (input / output). Cache read = 0.1 x input except Claude Fable 5.1 = $0.25/MTok;
  cache write = 1.25 x input for the 5-minute TTL, 2 x input for the 1-hour TTL.
- Claude Code's own ``modelUsage.costUSD`` (``costBasis: "list"``) on 2026-09-07
  reproduces these numbers for ``claude-fable-5-1`` and ``claude-haiku-4-5``.

:data:`LITELLM_OBSERVED_FP_2026_09` is NOT a list price. It is the effective
price table LiteLLM applied in the FrontierPhysics 2026-08-30..09-04 run, fitted
exactly (residual < 1e-14 relative) from the 538 recorded ``result.json``
files. It exists so ``bench cost --pricing litellm-observed-fp-2026-09`` can
reproduce the recorded ``total_cost_usd`` bit-for-bit, which is the regression
check that the aggregation is right. Its cache-write price is 0.25 x input,
i.e. one fifth of the official 1.25 x, which is the source of the gap between
recorded cost and the real bill.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

PRICING_VERSION = "anthropic-list-2026-09-07"


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for one model."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    @classmethod
    def anthropic(
        cls,
        input: float,
        output: float,
        *,
        cache_read: float | None = None,
        cache_write_5m: float | None = None,
        cache_write_1h: float | None = None,
    ) -> ModelPrice:
        """Anthropic default ratios: read 0.1x, write 1.25x (5m) / 2x (1h)."""
        return cls(
            input=input,
            output=output,
            cache_read=input * 0.1 if cache_read is None else cache_read,
            cache_write_5m=input * 1.25 if cache_write_5m is None else cache_write_5m,
            cache_write_1h=input * 2.0 if cache_write_1h is None else cache_write_1h,
        )


@dataclass(frozen=True)
class PriceTable:
    version: str
    prices: Mapping[str, ModelPrice]
    note: str = ""

    def get(self, model: str) -> ModelPrice | None:
        key = canonical_model(model)
        return self.prices.get(key) if key else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "note": self.note,
            "unit": "USD per 1M tokens",
            "prices": {k: asdict(v) for k, v in self.prices.items()},
        }


LIST_PRICES = PriceTable(
    version=PRICING_VERSION,
    note="Anthropic first-party API list prices; cache read 0.1x (Fable 5.1: $0.25/MTok), "
    "cache write 1.25x (5m TTL) / 2x (1h TTL).",
    prices={
        "claude-fable-5-1": ModelPrice.anthropic(10.0, 50.0, cache_read=0.25),
        "claude-mythos-5-1": ModelPrice.anthropic(10.0, 50.0, cache_read=0.25),
        "claude-fable-5": ModelPrice.anthropic(10.0, 50.0),
        "claude-opus-5": ModelPrice.anthropic(5.0, 25.0),
        "claude-opus-4-8": ModelPrice.anthropic(5.0, 25.0),
        "claude-opus-4-7": ModelPrice.anthropic(5.0, 25.0),
        "claude-opus-4-6": ModelPrice.anthropic(5.0, 25.0),
        "claude-sonnet-5": ModelPrice.anthropic(2.0, 10.0),
        "claude-sonnet-4-6": ModelPrice.anthropic(3.0, 15.0),
        "claude-haiku-4-5": ModelPrice.anthropic(1.0, 5.0),
    },
)

LITELLM_OBSERVED_FP_2026_09 = PriceTable(
    version="litellm-observed-fp-2026-09",
    note="Effective prices LiteLLM applied in the FrontierPhysics 2026-08-30..09-04 run, "
    "fitted from 538 result.json files (exact). Cache write is 0.25x input, NOT the "
    "official 1.25x. For reproduction of recorded cost only.",
    prices={
        "claude-fable-5": ModelPrice(10.0, 50.0, 1.0, 2.5, 2.5),
        "claude-fable-5-1": ModelPrice(10.0, 50.0, 0.25, 2.5, 2.5),
        "claude-opus-5": ModelPrice(5.0, 25.0, 0.5, 1.25, 1.25),
    },
)

PRICE_TABLES: dict[str, PriceTable] = {
    "list": LIST_PRICES,
    LIST_PRICES.version: LIST_PRICES,
    LITELLM_OBSERVED_FP_2026_09.version: LITELLM_OBSERVED_FP_2026_09,
}


def price_table(name: str | None) -> PriceTable:
    if not name:
        return LIST_PRICES
    try:
        return PRICE_TABLES[name]
    except KeyError:
        known = ", ".join(sorted(PRICE_TABLES))
        raise KeyError(f"unknown price table {name!r}; known: {known}") from None


_PROVIDER_PREFIX_RE = re.compile(
    r"^(?:(?:anthropic|openrouter|bedrock|bedrock_converse|vertex_ai|azure_ai|databricks|"
    r"deepinfra|perplexity)/)+",
)
_REGION_PREFIX_RE = re.compile(r"^(?:us|eu|global|jp|au|apac|us-gov)\.")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
_BEDROCK_VERSION_RE = re.compile(r"-v\d+(?::\d+)?$")
_CONTEXT_SUFFIX_RE = re.compile(r"\[\w+\]$")


def canonical_model(model: str | None) -> str | None:
    """Reduce a provider-specific model id to the list-price key.

    Handles LiteLLM route prefixes (``anthropic/…``), Bedrock ids
    (``us.anthropic.claude-opus-5-v1:0``), Claude Code's context suffix
    (``claude-opus-5[1m]``), dated snapshots (``claude-haiku-4-5-20251001``) and
    dotted versions (``claude-fable-5.1``). Returns ``None`` for empty input.
    """
    if not model:
        return None
    m = model.strip().lower()
    m = _PROVIDER_PREFIX_RE.sub("", m)
    m = _REGION_PREFIX_RE.sub("", m)
    m = m.removeprefix("anthropic.")
    m = m.removeprefix("databricks-")
    m = _CONTEXT_SUFFIX_RE.sub("", m)
    m = _BEDROCK_VERSION_RE.sub("", m)
    m = _DATE_SUFFIX_RE.sub("", m)
    m = m.replace(".", "-")
    return m or None


@dataclass(frozen=True)
class UsageCounts:
    """Token counts in the Anthropic (additive) convention.

    ``input_tokens`` is the UNCACHED input; cache reads and writes are separate
    additive components. This is the shape of ``response.usage`` from the
    Anthropic API and of Claude Code's telemetry. BenchFlow's normalized
    ``n_input_tokens`` (cache-inclusive) must be converted with
    :func:`from_inclusive_input` before pricing.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_creation_1h_tokens: int = 0  # subset of cache_creation_tokens

    def __add__(self, other: UsageCounts) -> UsageCounts:
        return UsageCounts(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
            self.cache_creation_1h_tokens + other.cache_creation_1h_tokens,
        )

    @property
    def total_input_inclusive(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    @classmethod
    def from_inclusive_input(
        cls,
        *,
        input_tokens_inclusive: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        cache_creation_1h_tokens: int = 0,
    ) -> UsageCounts:
        """Build from BenchFlow's cache-inclusive ``n_input_tokens`` convention."""
        uncached = input_tokens_inclusive - cache_read_tokens - cache_creation_tokens
        return cls(
            input_tokens=max(int(uncached), 0),
            output_tokens=int(output_tokens),
            cache_read_tokens=int(cache_read_tokens),
            cache_creation_tokens=int(cache_creation_tokens),
            cache_creation_1h_tokens=int(cache_creation_1h_tokens),
        )

    @classmethod
    def from_anthropic_usage(cls, usage: Mapping[str, Any]) -> UsageCounts:
        """From an Anthropic Messages ``usage`` block (also Claude Code transcripts)."""

        def _i(*keys: str) -> int:
            for k in keys:
                v = usage.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        continue
            return 0

        cc = usage.get("cache_creation")
        one_hour = 0
        if isinstance(cc, Mapping):
            try:
                one_hour = int(cc.get("ephemeral_1h_input_tokens") or 0)
            except (TypeError, ValueError):
                one_hour = 0
        return cls(
            input_tokens=_i("input_tokens", "inputTokens"),
            output_tokens=_i("output_tokens", "outputTokens"),
            cache_read_tokens=_i(
                "cache_read_input_tokens", "cache_read_tokens", "cacheReadInputTokens"
            ),
            cache_creation_tokens=_i(
                "cache_creation_input_tokens",
                "cache_creation_tokens",
                "cacheCreationInputTokens",
            ),
            cache_creation_1h_tokens=one_hour,
        )


@dataclass(frozen=True)
class CostBreakdown:
    model: str
    canonical_model: str
    pricing_version: str
    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float

    @property
    def total_usd(self) -> float:
        return (
            self.input_usd
            + self.output_usd
            + self.cache_read_usd
            + self.cache_write_usd
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_usd"] = self.total_usd
        return d


class UnknownModelPrice(KeyError):
    """Raised when a model id has no entry in the selected price table."""


def compute_cost(
    model: str,
    usage: UsageCounts,
    table: PriceTable = LIST_PRICES,
) -> CostBreakdown:
    """Price one usage record. Raises :class:`UnknownModelPrice` if unpriced."""
    price = table.get(model)
    key = canonical_model(model)
    if price is None or key is None:
        raise UnknownModelPrice(
            f"no price for model {model!r} (canonical {key!r}) in table {table.version}"
        )
    per = 1e-6
    write_5m = usage.cache_creation_tokens - usage.cache_creation_1h_tokens
    return CostBreakdown(
        model=model,
        canonical_model=key,
        pricing_version=table.version,
        input_usd=usage.input_tokens * price.input * per,
        output_usd=usage.output_tokens * price.output * per,
        cache_read_usd=usage.cache_read_tokens * price.cache_read * per,
        cache_write_usd=(
            max(write_5m, 0) * price.cache_write_5m
            + usage.cache_creation_1h_tokens * price.cache_write_1h
        )
        * per,
    )
