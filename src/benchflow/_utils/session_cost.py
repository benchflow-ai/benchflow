"""What a Claude Code rollout cost, from Claude Code's own session log.

With a Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``), or any run whose
model calls bypass BenchFlow's gateway, Claude Code calls the API itself:
BenchFlow counts its tokens (over ACP) but has no price for them. Claude Code
keeps a session log, ``<config dir>/projects/<cwd>/<session>.jsonl``, with each
subagent's log in ``<session>/subagents/``:

- ``cost-state`` lines hold Claude Code's running totals: ``totalCostUSD`` and
  ``modelUsage`` (``{model: {inputTokens, outputTokens, cacheReadInputTokens,
  cacheCreationInputTokens, costUSD}}``), the main session's, its subagents'
  and Claude Code's own side calls' (a title, a compaction summary, a web
  fetch's digest), which no ``assistant`` line logs. They are written at turn
  boundaries, so the last one can predate the session's last turn.
- ``assistant`` lines hold each API response's ``message.usage``, repeated on
  every line of a response that has several content blocks.

:func:`estimate_claude_code_cost` prices a folder of such logs. When each
session's last ``cost-state`` line counts at least the tokens of every
response in the log, of each kind, it prices Claude Code's own counts: a
Claude model at Claude Code's own figure (``claude-code-cost``), any other
model at BenchFlow's list price for it (``claude-code-tokens-at-list-price``).
Claude Code knows only Claude models' prices; a model behind another
provider's Anthropic-compatible endpoint (MiMo, DeepSeek, GLM, Kimi) gets
Claude Code's default rate, for mimo-v2.6-pro $4 / $20 / $0.20 per million
input, output and cache-read tokens against MiMo's $0.435 / $0.87 / $0.0036,
18 to 20 times its price (benchflow-ai/benchflow#1240). Otherwise the
responses' usage is priced at list prices (``usage-at-list-price``). A model
no list price covers leaves the figure unknown, never Claude Code's default
rate. Either way the figure is an estimate: the provider bills a subscription
differently, and nothing here sees the bill.
"""

from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchflow._utils.reachable_write import ReachablePath

SOURCE = "claude-code-session-log"
CLAUDE_CODE_COST = "claude-code-cost"
#: Claude Code's own per-model counts, at BenchFlow's list price for each model
#: Claude Code does not price (and Claude Code's own figure for a Claude model).
CLAUDE_CODE_TOKENS = "claude-code-tokens-at-list-price"
LIST_PRICE = "usage-at-list-price"
# Anthropic's server-side web search: $10 per 1,000 searches.
WEB_SEARCH_USD = 0.01

_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_STATE_KEYS = (
    "inputTokens",
    "outputTokens",
    "cacheReadInputTokens",
    "cacheCreationInputTokens",
)


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def claude_code_prices(model: str) -> bool:
    """Whether Claude Code's own cost for ``model`` is at that model's rate.

    Claude Code prices Claude models only. Pointed at another provider's
    Anthropic-compatible endpoint, it costs that provider's model at its
    default rate (benchflow-ai/benchflow#1240), so its figure is no price of
    the model's calls.
    """
    from benchflow.agents.claude_models import claude_alias_models

    name = model.split("[", 1)[0].strip().lower()
    # anthropic/claude-..., a Bedrock id (us.anthropic.claude-...-v1:0), or
    # one of Claude Code's own aliases (sonnet, opus[1m], opusplan).
    return (
        re.search(r"(^|[./])claude-", name) is not None
        or claude_alias_models(model) is not None
    )


def stated_counts(states: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Claude Code's own counts per model, summed over each session's last
    ``cost-state`` line: ``input`` (without the cache), ``output``,
    ``cache_read``, ``cache_write`` and ``web_searches``."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(
            ("input", "output", "cache_read", "cache_write", "web_searches"), 0
        )
    )
    for state in states.values():
        for model, usage in (state.get("modelUsage") or {}).items():
            if not isinstance(usage, dict):
                continue
            held = counts[str(model)]
            held["input"] += _int(usage.get("inputTokens"))
            held["output"] += _int(usage.get("outputTokens"))
            held["cache_read"] += _int(usage.get("cacheReadInputTokens"))
            held["cache_write"] += _int(usage.get("cacheCreationInputTokens"))
            held["web_searches"] += _int(usage.get("webSearchRequests"))
    return dict(counts)


def covers(states: dict[str, dict], responses: dict[str, list[tuple]]) -> bool:
    """Whether each session's last ``cost-state`` counts at least its logged
    responses, for every kind of token: a state written before the
    session's last turn counts less of some kind and does not."""
    if not states:
        return False
    for session in set(states) | set(responses):
        state = states.get(session)
        if state is None:
            return False
        usage = [
            u for u in (state.get("modelUsage") or {}).values() if isinstance(u, dict)
        ]
        logged = responses.get(session, [])
        for state_key, key in zip(_STATE_KEYS, _TOKEN_KEYS, strict=True):
            if sum(_int(u.get(state_key)) for u in usage) < sum(
                _int(u.get(key)) for _, u in logged
            ):
                return False
    return True


def _read(
    logs: list[Path | ReachablePath],
) -> tuple[dict[str, dict], dict[str, list[tuple]]]:
    """Each session's last cost-state line, and its responses (model, usage)
    once per response (see :func:`read_scoped`)."""
    states, responses, _subagents = read_scoped(logs)
    return states, responses


def read_scoped(
    logs: list[Path | ReachablePath],
) -> tuple[dict[str, dict], dict[str, list[tuple]], list[dict[str, Any]]]:
    """Each session's last cost-state line, its responses (model, usage)
    once per response, and the usages of the responses a subagent's log
    (``<session>/subagents/``) holds.

    A response is one (message id, request id), however many lines and
    session files log it: it counts once, in the session that logged it
    first, with the usage of its line with the most output (a line per
    content block, each with the output so far).
    """
    states: dict[str, dict] = {}
    responses: dict[str, dict[tuple, tuple]] = defaultdict(dict)
    first_logged: dict[tuple, str] = {}
    in_subagent: dict[tuple, bool] = {}
    for path in logs:
        subagent = path.parent.name == "subagents"
        session = path.parent.parent.name if subagent else path.stem
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines()):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "cost-state" and not subagent:
                states[session] = entry
            message = entry.get("message")
            if entry.get("type") != "assistant" or not isinstance(message, dict):
                continue
            usage, model = message.get("usage"), message.get("model")
            if isinstance(usage, dict) and model and model != "<synthetic>":
                # One response spans several lines (one per content block),
                # all with its message id; a line without one stands alone.
                response = message.get("id") or f"{path}:{number}"
                key = (response, entry.get("requestId"))
                owner = first_logged.setdefault(key, session)
                in_subagent.setdefault(key, subagent)
                held = responses[owner].get(key)
                if held is None or _int(usage.get("output_tokens")) >= _int(
                    held[1].get("output_tokens")
                ):
                    responses[owner][key] = (str(model), usage)
    subagent_usages = [
        usage
        for held in responses.values()
        for key, (_model, usage) in held.items()
        if in_subagent.get(key)
    ]
    return states, {s: list(r.values()) for s, r in responses.items()}, subagent_usages


@functools.lru_cache(maxsize=1)
def _price_table() -> dict[str, Any]:
    """LiteLLM's bundled model price table, read without importing LiteLLM."""
    import importlib.util

    spec = importlib.util.find_spec("litellm")
    for folder in (spec.submodule_search_locations or []) if spec else []:
        path = Path(folder) / "model_prices_and_context_window_backup.json"
        try:
            table = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(table, dict):
            return table
    return {}


def _tier_suffix(entry: dict[str, Any], prompt_tokens: int) -> str:
    """The key suffix of the highest long-prompt tier ``prompt_tokens`` is over."""
    over = [
        int(m.group(1))
        for key in entry
        if (m := re.fullmatch(r"input_cost_per_token_above_(\d+)k_tokens", key))
        and prompt_tokens > int(m.group(1)) * 1000
    ]
    return f"_above_{max(over)}k_tokens" if over else ""


def list_price(model: str, prompt_tokens: int = 0) -> dict[str, float] | None:
    """USD per token for ``model`` (input, output, cache read, cache write).

    From BenchFlow's dated snapshot for current Claude models, else LiteLLM's
    table; None when neither has the model. ``prompt_tokens`` (input plus
    cache reads and writes, as Anthropic counts a prompt) picks a long-prompt
    tier, such as Haiku 5.5's over 100,000 tokens.
    """
    # Here, not at import: the providers package imports the gateway runtime.
    from benchflow.providers.pricing_snapshot import anthropic_prices

    table = _price_table()
    base = model.split("[", 1)[0]
    undated = re.sub(r"-\d{8}$", "", base)
    for name in (model, base, f"anthropic/{base}", undated, f"anthropic/{undated}"):
        entry = anthropic_prices(name.removeprefix("anthropic/")) or table.get(name)
        if not isinstance(entry, dict) or "input_cost_per_token" not in entry:
            continue
        tier = _tier_suffix(entry, prompt_tokens)

        def rate(key: str, entry: dict[str, Any] = entry, tier: str = tier) -> Any:
            return entry.get(key + tier, entry.get(key))

        price = {
            "input": _number(rate("input_cost_per_token")),
            "output": _number(rate("output_cost_per_token")),
            "cache_read": _number(rate("cache_read_input_token_cost")),
            "cache_write": _number(rate("cache_creation_input_token_cost")),
        }
        hour = rate("cache_creation_input_token_cost_above_1hr")
        price["cache_write_1h"] = (
            _number(hour) if hour is not None else price["cache_write"]
        )
        return price
    return None


def _unpriced(models: list[str]) -> list[str]:
    from benchflow._utils.pricing import list_rates

    return sorted({m for m in models if list_rates(m) is None})


def _at_list_price(
    responses: list[tuple],
) -> tuple[float | None, dict[str, float], dict[str, str]]:
    """The responses' usage at list prices (BenchFlow's pricing step,
    :mod:`benchflow._utils.pricing`): (USD, USD per model, table per model);
    the USD is None when a response's model has no list price."""
    from benchflow._utils.pricing import anthropic_slice, price_slices

    priced = price_slices([anthropic_slice(model, usage) for model, usage in responses])
    if priced is None:
        return None, {}, {}
    return priced.usd, dict(priced.models), dict(priced.tables)


def _stated_at_list_price(
    counts: dict[str, dict[str, int]],
) -> tuple[float | None, dict[str, float], dict[str, str]]:
    """Claude Code's own counts of each model at list prices (as
    :func:`_at_list_price`). A count sums many requests, so it pays the base
    tier; it has no 5-minute/1-hour split, so its cache writes pay the
    5-minute rate."""
    from benchflow._utils.pricing import TokenSlice, price_slices

    priced = price_slices(
        [
            TokenSlice(
                model=model,
                input=c["input"],
                output=c["output"],
                cache_read=c["cache_read"],
                cache_write=c["cache_write"],
                web_searches=c["web_searches"],
            )
            for model, c in counts.items()
        ]
    )
    if priced is None:
        return None, {}, {}
    return priced.usd, dict(priced.models), dict(priced.tables)


def estimate_claude_code_cost(root: Path | ReachablePath) -> dict[str, Any] | None:
    """The USD of the Claude Code sessions logged under ``root``.

    Returns ``{"usd", "source", "method", "sessions", "responses", "models",
    "context_1m"}``: ``method`` is ``claude-code-cost`` (Claude Code's own
    totals), ``claude-code-tokens-at-list-price`` (Claude Code's own counts,
    a model it does not price at BenchFlow's list price) or
    ``usage-at-list-price`` (the logged responses); ``models`` is USD per
    model; ``context_1m`` is True when a model ran its 1M-context variant (a
    ``[1m]`` suffix), which is priced higher. When Claude Code's own figure
    is set aside, ``claude_code_usd`` keeps it, and ``repriced_models``
    names the models priced instead.

    ``usd`` is None, with ``method`` None and ``unpriced_models`` naming the
    models, when a model has no list price and Claude Code's own figure for
    it is not at its rate (or does not cover the log). Returns None when
    the logs hold no response (and Claude Code's own totals say nothing was
    spent).
    """
    logs = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    states, responses = _read(logs)
    if not responses and not any(
        _number(state.get("totalCostUSD")) for state in states.values()
    ):
        # A cost-state line with no response and no cost (Claude Code wrote
        # its totals before any call answered): nothing was measured, so the
        # cost is unknown, not $0.
        return None
    stated: dict[str, float] = defaultdict(float)
    for state in states.values():
        for model, usage in (state.get("modelUsage") or {}).items():
            if isinstance(usage, dict):
                stated[str(model)] += _number(usage.get("costUSD"))
    logged_models = [model for rs in responses.values() for model, _ in rs]
    context_1m = any(model.endswith("[1m]") for model in stated) or any(
        model.endswith("[1m]") for model in logged_models
    )
    base = {
        "source": SOURCE,
        "sessions": len(set(states) | set(responses)),
        "responses": sum(len(r) for r in responses.values()),
        "context_1m": context_1m,
    }
    claude_code_usd = round(
        sum(_number(s.get("totalCostUSD")) for s in states.values()), 10
    )
    foreign = sorted(
        model
        for model in set(stated) | set(logged_models)
        if not claude_code_prices(model)
    )
    counts = stated_counts(states)
    if covers(states, responses) and all(model in counts for model in foreign):
        if not foreign:
            return {
                **base,
                "usd": claude_code_usd,
                "method": CLAUDE_CODE_COST,
                "models": dict(stated),
            }
        # Claude Code's own counts are the trial's most complete (its side
        # calls included); its figure for these models is at its default
        # rate, not theirs (benchflow-ai/benchflow#1240).
        usd, models, _tables = _stated_at_list_price(
            {model: counts[model] for model in foreign}
        )
        if usd is None:
            return {
                **base,
                "usd": None,
                "method": None,
                "models": {},
                "unpriced_models": _unpriced(foreign),
                "claude_code_usd": claude_code_usd,
            }
        own = {m: v for m, v in stated.items() if claude_code_prices(m)}
        return {
            **base,
            "usd": round(usd + sum(own.values()), 10),
            "method": CLAUDE_CODE_TOKENS,
            "models": {**own, **models},
            "repriced_models": foreign,
            "claude_code_usd": claude_code_usd,
        }
    if not responses:
        return None
    usd, models, _tables = _at_list_price([r for rs in responses.values() for r in rs])
    if usd is None:
        return {
            **base,
            "usd": None,
            "method": None,
            "models": {},
            "unpriced_models": _unpriced(logged_models),
        }
    return {**base, "usd": round(usd, 10), "method": LIST_PRICE, "models": models}
