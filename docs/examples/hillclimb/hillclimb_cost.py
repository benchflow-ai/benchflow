"""What each rollout of the hill-climb demo cost, from Claude Code's session log.

With a Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``) Claude Code calls the
Anthropic API itself rather than through BenchFlow's model proxy, so a trial's
``result.json`` has its tokens (from ACP) but no USD, and a ``bf.Budget``'s
``max_cost_usd`` cannot count it. Claude Code keeps its own session log,
``~/.claude/projects/<cwd>/<session>.jsonl``, with each subagent's log in
``<session>/subagents/``:

- ``cost-state`` lines hold Claude Code's running totals: ``totalCostUSD`` and
  ``modelUsage`` (``{model: {inputTokens, outputTokens, cacheReadInputTokens,
  cacheCreationInputTokens, webSearchRequests, costUSD}}``; a ``[1m]`` suffix on
  the model marks the 1M-context variant). They are written at turn
  boundaries, so the last one can predate the session's last turn.
- ``assistant`` lines hold each API response's ``message.usage``, repeated on
  every line of a response that has several content blocks.

The demo puts the session log into every trial folder, at
``artifacts/claude-sessions/`` (:data:`CAPTURE` for the evaluations; a
declared artifact in the optimizer's task), and prices a trial from it: Claude
Code's own ``totalCostUSD`` when a ``cost-state`` line counts at least the
tokens of every response in the log, otherwise the responses' usage at
:data:`LIST_PRICES`. A trial without a session log keeps BenchFlow's own
``cost_usd`` (an API key, priced by BenchFlow's model proxy) and is unknown
otherwise. Every figure says which of these it came from.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

SESSIONS = "claude-sessions"  # under the trial's artifacts/

# A root setup command for the evaluations (EvaluationConfig.config_override;
# it runs before the agent is installed). BenchFlow collects everything under
# /logs/artifacts into the trial folder, and copies /root/.claude into the
# sandbox user's home when it creates that user, so Claude Code's
# ~/.claude/projects becomes a link into /logs/artifacts. It never fails the
# trial: without it the trial only loses its session log.
CAPTURE = (
    "( mkdir -p /logs/artifacts/claude-sessions /root/.claude"
    " && chmod 1777 /logs/artifacts/claude-sessions"
    " && { [ -e /root/.claude/projects ] || [ -L /root/.claude/projects ]"
    " || ln -s /logs/artifacts/claude-sessions /root/.claude/projects; } ) || true"
)

# USD per million tokens: input, output, cache read, cache write (5 minutes),
# cache write (1 hour); Anthropic's list prices. On a real Claude Code session
# log these reproduce Claude Code's own costUSD for claude-opus-5-5 within 0.3%.
LIST_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25, 2.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00, 8.00),
}
WEB_SEARCH_USD = 0.01

COST_STATE = "claude-code-cost-state"
LIST_PRICE = "claude-code-usage-at-list-price"
BENCHFLOW = "benchflow"
UNKNOWN = "unknown"

# Credentials a session log could echo (an agent that prints its environment);
# they are replaced before the demo keeps or reads the log.
SECRET_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


def session_logs(trial_dir: Path | str | None) -> list[Path]:
    root = Path(trial_dir) / "artifacts" / SESSIONS if trial_dir else None
    return sorted(root.rglob("*.jsonl")) if root and root.is_dir() else []


def scrub(trial_dir: Path | str | None, extra: Iterable[str] = ()) -> int:
    """Replace any credential from the environment, and the ``extra`` ones (a
    pool's tokens), in the trial's session logs; return how many files changed."""
    found = [os.environ.get(n, "") for n in SECRET_ENV] + list(extra)
    secrets = [v for v in found if len(v) >= 8]
    changed = 0
    for path in session_logs(trial_dir) if secrets else []:
        text = path.read_text(errors="replace")
        clean = text
        for secret in secrets:
            clean = clean.replace(secret, "[redacted]")
        if clean != text:
            path.write_text(clean)
            changed += 1
    return changed


def _price(model: str) -> tuple[float, ...] | None:
    base = model.split("[")[0]
    for name, price in LIST_PRICES.items():
        if base == name or base.startswith(name + "-"):
            return price
    return None


def _tokens(usage: dict) -> int:
    return sum(
        int(usage.get(k) or 0)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def _read(logs: list[Path]) -> tuple[dict[str, dict], dict[str, list[tuple]]]:
    """The last cost-state line of each session, and each session's responses
    (model, usage), once per response."""
    states: dict[str, dict] = {}
    responses: dict[str, dict[tuple, tuple]] = defaultdict(dict)
    for path in logs:
        session = (
            path.parent.parent.name if path.parent.name == "subagents" else path.stem
        )
        for line in path.read_text(errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "cost-state" and path.parent.name != "subagents":
                states[session] = entry
            message = entry.get("message")
            if entry.get("type") != "assistant" or not isinstance(message, dict):
                continue
            usage, model = message.get("usage"), message.get("model")
            if isinstance(usage, dict) and model and model != "<synthetic>":
                key = (message.get("id"), entry.get("requestId"))
                responses[session][key] = (model, usage)
    return states, {s: list(r.values()) for s, r in responses.items()}


def _list_price(responses: list[tuple]) -> tuple[float | None, dict[str, float]]:
    by_model: dict[str, float] = defaultdict(float)
    for model, usage in responses:
        price = _price(model)
        if price is None:
            return None, {}
        write = usage.get("cache_creation") or {}
        hour = int(write.get("ephemeral_1h_input_tokens") or 0)
        minutes = int(usage.get("cache_creation_input_tokens") or 0) - hour
        searches = (usage.get("server_tool_use") or {}).get("web_search_requests") or 0
        by_model[model] += (
            int(usage.get("input_tokens") or 0) * price[0]
            + int(usage.get("output_tokens") or 0) * price[1]
            + int(usage.get("cache_read_input_tokens") or 0) * price[2]
            + max(minutes, 0) * price[3]
            + hour * price[4]
        ) / 1e6 + searches * WEB_SEARCH_USD
    return sum(by_model.values()), dict(by_model)


def trial_cost(trial_dir: Path | str | None, benchflow_usd: float | None) -> dict:
    """``{"usd", "source", "models", "context_1m"}`` for one trial or rollout."""
    if benchflow_usd is not None:
        return {
            "usd": benchflow_usd,
            "source": BENCHFLOW,
            "models": {},
            "context_1m": None,
        }
    states, responses = _read(session_logs(trial_dir))
    if not states and not responses:
        return {"usd": None, "source": UNKNOWN, "models": {}, "context_1m": None}
    stated: dict[str, float] = Counter()  # Claude Code's costUSD per model
    for state in states.values():
        for model, usage in (state.get("modelUsage") or {}).items():
            stated[model] += float(usage.get("costUSD") or 0.0)
    context_1m = any(model.endswith("[1m]") for model in stated) or None

    def covers(session: str) -> bool:
        """Whether the session's last cost-state counts all its responses."""
        state = states.get(session)
        if state is None:
            return False
        stated_tokens = sum(
            int(usage.get(k) or 0)
            for usage in (state.get("modelUsage") or {}).values()
            for k in (
                "inputTokens",
                "outputTokens",
                "cacheReadInputTokens",
                "cacheCreationInputTokens",
            )
        )
        return stated_tokens >= sum(_tokens(u) for _, u in responses.get(session, []))

    if all(covers(session) for session in set(states) | set(responses)):
        usd = sum(float(s.get("totalCostUSD") or 0.0) for s in states.values())
        return {
            "usd": usd,
            "source": COST_STATE,
            "models": dict(stated),
            "context_1m": context_1m,
        }
    usd, models = _list_price([r for rs in responses.values() for r in rs])
    return {
        "usd": usd,
        "source": LIST_PRICE if usd is not None else UNKNOWN,
        "models": models,
        "context_1m": context_1m,
    }


def summary(costs: list[dict]) -> dict:
    """Totals over trials: known USD, trials per source, and the source overall."""
    sources = Counter(c["source"] for c in costs)
    known = [c["usd"] for c in costs if c["usd"] is not None]
    kinds = sorted(k for k in sources if k != UNKNOWN)
    return {
        "usd": sum(known) if known else None,
        "sources": dict(sorted(sources.items())),
        "source": kinds[0] if len(kinds) == 1 else "mixed" if kinds else UNKNOWN,
        "context_1m": any(c.get("context_1m") for c in costs),
    }
