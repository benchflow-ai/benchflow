"""Tokens at list prices: the one pricing step every unpriced trial goes through.

BenchFlow's model gateway prices the calls it proxies. Every other count of a
trial's tokens -- a subscription login's end-of-turn report, a count salvaged
from the agent's own stream or session log -- has no price until it is priced
here. A model's rates come from BenchFlow's dated price snapshot
(:mod:`benchflow.providers.pricing_snapshot`), else LiteLLM's bundled table;
a model in neither is left unpriced. The figure is a list-price estimate:
nothing here sees what the provider billed, and a subscription is billed by
its plan.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Anthropic's server-side web search: $10 per 1,000 searches.
WEB_SEARCH_USD = 0.01
SNAPSHOT_TABLE = "benchflow-snapshot"
LITELLM_TABLE = "litellm"


@dataclass(frozen=True)
class Rates:
    """USD per token for one model, and the table they came from."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    cache_write_1h: float
    table: str
    table_date: str | None


@dataclass
class TokenSlice:
    """Tokens one model used, in the terms a list price takes.

    ``input`` leaves out the cache reads and writes (Anthropic's
    ``input_tokens``); ``cache_write_1h`` is the part of ``cache_write``
    written for an hour. ``prompt`` is one request's prompt (input plus
    cache), which picks a long-prompt tier; 0 for a slice that sums many
    requests, which then pays the base tier.
    """

    model: str
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0
    web_searches: int = 0
    prompt: int = 0


@dataclass
class Priced:
    """What :func:`price_slices` found: the USD, per model, and its tables."""

    usd: float
    models: dict[str, float] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    table_date: str | None = None


def _number(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _tier_suffix(entry: dict[str, Any], prompt_tokens: int) -> str:
    """The key suffix of the highest long-prompt tier ``prompt_tokens`` is over."""
    over = [
        int(m.group(1))
        for key in entry
        if (m := re.fullmatch(r"input_cost_per_token_above_(\d+)k_tokens", key))
        and prompt_tokens > int(m.group(1)) * 1000
    ]
    return f"_above_{max(over)}k_tokens" if over else ""


def _candidates(model: str) -> tuple[list[str], list[str]]:
    """(snapshot names, LiteLLM table names) to try for ``model``, in order.

    ``claude-opus-5-5[1m]`` is priced as ``claude-opus-5-5`` (the 1M-context
    variant's own rates are not in either table), a dated id as its undated
    family, and a routed id (``anthropic/...``, ``openai/...``) by its last
    part too.
    """
    base = model.split("[", 1)[0].strip()
    tail = base.rsplit("/", 1)[-1]
    undated = re.sub(r"-\d{8}$", "", tail)
    snapshot = list(dict.fromkeys([tail, undated]))
    table = list(
        dict.fromkeys(
            [
                model,
                base,
                tail,
                undated,
                f"anthropic/{tail}",
                f"anthropic/{undated}",
                f"openai/{tail}",
                f"gemini/{tail}",
            ]
        )
    )
    return snapshot, table


def list_rates(model: str, prompt_tokens: int = 0) -> Rates | None:
    """USD per token for ``model`` (input, output, cache read, cache write),
    from BenchFlow's dated snapshot, else LiteLLM's bundled table; None when
    neither has the model. ``prompt_tokens`` picks a long-prompt tier."""
    from benchflow._utils.session_cost import _price_table
    from benchflow.providers.pricing_snapshot import (
        SNAPSHOT_DATE,
        bare_model_prices,
    )

    if not model:
        return None
    snapshot_names, table_names = _candidates(model)
    entry: dict[str, Any] | None = None
    table, table_date = LITELLM_TABLE, None
    for name in snapshot_names:
        entry = bare_model_prices(name)
        if entry:
            table, table_date = SNAPSHOT_TABLE, SNAPSHOT_DATE
            break
    if not entry:
        litellm_table = _price_table()
        for name in table_names:
            candidate = litellm_table.get(name)
            if isinstance(candidate, dict) and "input_cost_per_token" in candidate:
                entry = candidate
                break
    if not entry or "input_cost_per_token" not in entry:
        return None
    tier = _tier_suffix(entry, prompt_tokens)

    def rate(key: str) -> Any:
        return entry.get(key + tier, entry.get(key))

    input_rate = _number(rate("input_cost_per_token"))
    write = rate("cache_creation_input_token_cost")
    # A table with no cache-write rate (OpenAI, Gemini) bills a written
    # token as input: never price it at 0.
    write_rate = _number(write) if write is not None else input_rate
    hour = rate("cache_creation_input_token_cost_above_1hr")
    read = rate("cache_read_input_token_cost")
    return Rates(
        input=input_rate,
        output=_number(rate("output_cost_per_token")),
        cache_read=_number(read) if read is not None else input_rate,
        cache_write=write_rate,
        cache_write_1h=_number(hour) if hour is not None else write_rate,
        table=table,
        table_date=table_date,
    )


def price_slices(slices: list[TokenSlice]) -> Priced | None:
    """The USD of ``slices`` at list prices; None when a slice's model has no
    list price (a partial figure would read as the whole cost)."""
    if not slices:
        return None
    by_model: dict[str, float] = defaultdict(float)
    tables: dict[str, str] = {}
    dates: set[str | None] = set()
    for piece in slices:
        rates = list_rates(piece.model, piece.prompt)
        if rates is None:
            return None
        minutes = max(piece.cache_write - piece.cache_write_1h, 0)
        by_model[piece.model] += (
            max(piece.input, 0) * rates.input
            + max(piece.output, 0) * rates.output
            + max(piece.cache_read, 0) * rates.cache_read
            + minutes * rates.cache_write
            + max(piece.cache_write_1h, 0) * rates.cache_write_1h
            + max(piece.web_searches, 0) * WEB_SEARCH_USD
        )
        tables[piece.model] = rates.table
        dates.add(rates.table_date)
    usd = round(sum(by_model.values()), 10)
    # One date only when every slice came from the dated snapshot.
    table_date = next(iter(dates)) if len(dates) == 1 else None
    return Priced(
        usd=usd,
        models={model: round(v, 10) for model, v in by_model.items()},
        tables=tables,
        table_date=table_date,
    )


def anthropic_slice(model: str, usage: dict[str, Any]) -> TokenSlice:
    """One Anthropic-shaped ``usage`` (a response's, or a turn's total) as a
    slice: ``input_tokens`` leaves the cache out, as Anthropic counts it."""

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    written = usage.get("cache_creation")
    hour = (
        count_from(written, "ephemeral_1h_input_tokens")
        if isinstance(written, dict)
        else 0
    )
    server = usage.get("server_tool_use")
    searches = (
        count_from(server, "web_search_requests") if isinstance(server, dict) else 0
    )
    fresh = count("input_tokens")
    read = count("cache_read_input_tokens")
    write = count("cache_creation_input_tokens")
    return TokenSlice(
        model=model,
        input=fresh,
        output=count("output_tokens"),
        cache_read=read,
        cache_write=write,
        cache_write_1h=min(hour, write),
        web_searches=searches,
        prompt=fresh + read + write,
    )


def count_from(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
