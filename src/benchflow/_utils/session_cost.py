"""What a Claude Code rollout cost, from Claude Code's own session log.

With a Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``), or any run whose
model calls bypass BenchFlow's gateway, Claude Code calls the API itself:
BenchFlow counts its tokens (over ACP) but has no price for them. Claude Code
keeps a session log, ``<config dir>/projects/<cwd>/<session>.jsonl``, with each
subagent's log in ``<session>/subagents/``:

- ``cost-state`` lines hold Claude Code's running totals: ``totalCostUSD`` and
  ``modelUsage`` (``{model: {inputTokens, outputTokens, cacheReadInputTokens,
  cacheCreationInputTokens, costUSD}}``). They are written at turn
  boundaries, so the last one can predate the session's last turn.
- ``assistant`` lines hold each API response's ``message.usage``, repeated on
  every line of a response that has several content blocks.

:func:`estimate_claude_code_cost` prices a folder of such logs: Claude Code's
own ``totalCostUSD`` when each session's last ``cost-state`` line counts at
least the tokens of every response in the log (``claude-code-cost``),
otherwise the responses' usage at list prices from LiteLLM's price table
(``usage-at-list-price``). Either way the figure is an estimate: the provider
bills a subscription differently, and nothing here sees the bill.
"""

from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SOURCE = "claude-code-session-log"
CLAUDE_CODE_COST = "claude-code-cost"
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


def _read(logs: list[Path]) -> tuple[dict[str, dict], dict[str, list[tuple]]]:
    """Each session's last cost-state line, and its responses (model, usage)
    once per response."""
    states: dict[str, dict] = {}
    responses: dict[str, dict[tuple, tuple]] = defaultdict(dict)
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
                responses[session][key] = (str(model), usage)
    return states, {s: list(r.values()) for s, r in responses.items()}


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


def list_price(model: str) -> dict[str, float] | None:
    """USD per token for ``model`` (input, output, cache read, cache write),
    from LiteLLM's table; None when the model is not in it."""
    table = _price_table()
    base = model.split("[", 1)[0]
    undated = re.sub(r"-\d{8}$", "", base)
    for name in (model, base, f"anthropic/{base}", undated, f"anthropic/{undated}"):
        entry = table.get(name)
        if not isinstance(entry, dict) or "input_cost_per_token" not in entry:
            continue
        price = {
            "input": _number(entry.get("input_cost_per_token")),
            "output": _number(entry.get("output_cost_per_token")),
            "cache_read": _number(entry.get("cache_read_input_token_cost")),
            "cache_write": _number(entry.get("cache_creation_input_token_cost")),
        }
        hour = entry.get("cache_creation_input_token_cost_above_1hr")
        price["cache_write_1h"] = (
            _number(hour) if hour is not None else price["cache_write"]
        )
        return price
    return None


def _at_list_price(responses: list[tuple]) -> tuple[float | None, dict[str, float]]:
    by_model: dict[str, float] = defaultdict(float)
    for model, usage in responses:
        price = list_price(model)
        if price is None:
            return None, {}
        written = usage.get("cache_creation") or {}
        hour = _int(written.get("ephemeral_1h_input_tokens"))
        minutes = max(_int(usage.get("cache_creation_input_tokens")) - hour, 0)
        searches = _int((usage.get("server_tool_use") or {}).get("web_search_requests"))
        by_model[model] += (
            _int(usage.get("input_tokens")) * price["input"]
            + _int(usage.get("output_tokens")) * price["output"]
            + _int(usage.get("cache_read_input_tokens")) * price["cache_read"]
            + minutes * price["cache_write"]
            + hour * price["cache_write_1h"]
            + searches * WEB_SEARCH_USD
        )
    return sum(by_model.values()), dict(by_model)


def estimate_claude_code_cost(root: Path) -> dict[str, Any] | None:
    """The USD of the Claude Code sessions logged under ``root``, or None.

    Returns ``{"usd", "source", "method", "sessions", "responses", "models",
    "context_1m"}``: ``method`` is ``claude-code-cost`` (Claude Code's own
    totals) or ``usage-at-list-price``; ``models`` is USD per model;
    ``context_1m`` is True when a model ran its 1M-context variant (a
    ``[1m]`` suffix), which is priced higher. None when the logs hold no
    response, or a response's model has no list price and Claude Code's own
    totals do not cover the log.
    """
    logs = sorted(Path(root).rglob("*.jsonl")) if Path(root).is_dir() else []
    states, responses = _read(logs)
    if not responses and not states:
        return None
    stated: dict[str, float] = defaultdict(float)
    for state in states.values():
        for model, usage in (state.get("modelUsage") or {}).items():
            if isinstance(usage, dict):
                stated[str(model)] += _number(usage.get("costUSD"))
    context_1m = any(model.endswith("[1m]") for model in stated) or any(
        model.endswith("[1m]") for rs in responses.values() for model, _ in rs
    )

    def covered(session: str) -> bool:
        """Whether the session's last cost-state counts all its responses."""
        state = states.get(session)
        if state is None:
            return False
        counted = sum(
            _int(usage.get(key))
            for usage in (state.get("modelUsage") or {}).values()
            if isinstance(usage, dict)
            for key in _STATE_KEYS
        )
        logged = sum(
            _int(usage.get(key))
            for _, usage in responses.get(session, [])
            for key in _TOKEN_KEYS
        )
        return counted >= logged

    base = {
        "source": SOURCE,
        "sessions": len(set(states) | set(responses)),
        "responses": sum(len(r) for r in responses.values()),
        "context_1m": context_1m,
    }
    if states and all(covered(s) for s in set(states) | set(responses)):
        usd = sum(_number(s.get("totalCostUSD")) for s in states.values())
        return {
            **base,
            "usd": round(usd, 10),
            "method": CLAUDE_CODE_COST,
            "models": dict(stated),
        }
    usd, models = _at_list_price([r for rs in responses.values() for r in rs])
    if usd is None or not responses:
        return None
    return {**base, "usd": round(usd, 10), "method": LIST_PRICE, "models": models}
