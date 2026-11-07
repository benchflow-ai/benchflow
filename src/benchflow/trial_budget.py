"""Per-trial token and turn budgets, enforced while the trial runs.

A trial budget caps each trial on one measure or both:

- ``tokens``: the tokens of every model call the trial's agent makes, counted
  as ``total_tokens`` counts them: the input, cached input included, plus the
  output.
- ``turns``: the model calls themselves. One turn is one call BenchFlow's
  model proxy forwards for the trial, whoever in the agent made it: its own
  loop, a subagent, or a side call such as a session title.

BenchFlow's model proxy enforces both (its callback, in
``providers/litellm_logging.py``): every call of the trial's agent goes
through it, subagents' calls among them. Before it forwards a call it compares
what the trial has used with the budget. Once a measure is spent it refuses
the call with HTTP 400 (``trial_budget_spent: ...``) and records the refusal in
its log. A call's tokens are known only when it ends, so the call that spends
the token budget completes, and so does any call already under way beside it:
a trial can pass its token budget by those calls, never by a call that starts
after.

The rollout reads that record from the proxy's log while the trial runs and
stops the agent. The verifier then grades what the agent left, as after a
timeout: the trial counts 0 unless that work passes, and it is never retried.
``result.json``'s ``trial_budget`` records the budget, what the trial used,
what its subagents used where the agent's own session log says, and which
measure stopped it.

A run sets a budget for every trial (``--token-budget``, ``--turn-budget``, a
run config's ``token_budget`` and ``turn_budget``); a task sets its own in
``[agent]`` (``token_budget``, ``turn_budget``). A measure the run sets
replaces the task's. An agent whose calls do not go through the proxy (a
subscription login, a direct endpoint, an agent that calls its own model)
leaves nothing to enforce the budget with, so its trial is refused before the
agent starts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchflow.diagnostics import AgentModelNotOfferedError

__all__ = [
    "MEASURES",
    "SPENT_RULE",
    "STOP_TEXT",
    "TrialBudget",
    "TrialBudgetSpentError",
    "TrialBudgetUnenforceableError",
    "budget_refusal",
    "effective_budget",
    "stop_text",
    "task_budget",
    "unenforceable_reason",
]

MEASURES = ("tokens", "turns")
#: The proxy's environment: the budget, and what earlier proxies of the same
#: trial already counted (a scene role on another route starts a new proxy).
TOKEN_BUDGET_ENV = "BENCHFLOW_TRIAL_TOKEN_BUDGET"
TURN_BUDGET_ENV = "BENCHFLOW_TRIAL_TURN_BUDGET"
USED_ENV = "BENCHFLOW_TRIAL_BUDGET_USED"
#: The rule the proxy's refusal record names, and the words its 400 opens with.
SPENT_RULE = "trial-budget"
#: How a trial its budget stopped records its error (category ``trial_budget``).
STOP_TEXT = "trial budget reached"
_UNIT = {"tokens": ("token", "tokens"), "turns": ("turn", "turns")}


def _positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive whole number, not {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be a positive whole number, not {value!r}")
        value = int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("_", "")
        if not text.isdecimal():
            raise ValueError(f"{name} must be a positive whole number, not {value!r}")
        value = int(text)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive whole number, not {value!r}")
    return value


def _count(n: int, measure: str) -> str:
    one, many = _UNIT[measure]
    return f"{n:,} {one if n == 1 else many}"


@dataclass(frozen=True)
class TrialBudget:
    """A per-trial budget: ``tokens``, ``turns`` or both (see the module).

    >>> TrialBudget(tokens=200_000, turns=50).text()
    '200,000 tokens and 50 turns'
    """

    tokens: int | None = None
    turns: int | None = None

    def __post_init__(self) -> None:
        for measure in MEASURES:
            object.__setattr__(
                self,
                measure,
                _positive_int(f"{measure} budget", getattr(self, measure)),
            )
        if self.tokens is None and self.turns is None:
            raise ValueError(
                "a trial budget needs a token budget, a turn budget or both"
            )

    @classmethod
    def coerce(
        cls, value: TrialBudget | Mapping[str, Any] | None
    ) -> TrialBudget | None:
        """A budget from itself, a mapping (:meth:`to_dict`'s shape) or None;
        a mapping with neither measure set is no budget."""
        if value is None or isinstance(value, TrialBudget):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"trial budget must be a mapping of tokens and turns, not "
                f"{type(value).__name__}"
            )
        unknown = sorted(str(key) for key in value if key not in MEASURES)
        if unknown:
            raise ValueError(
                f"unknown trial budget key(s) {', '.join(unknown)}; use tokens, turns"
            )
        if all(value.get(measure) is None for measure in MEASURES):
            return None
        return cls(tokens=value.get("tokens"), turns=value.get("turns"))

    def to_dict(self) -> dict[str, int | None]:
        return {"tokens": self.tokens, "turns": self.turns}

    def text(self) -> str:
        """The budget in words: ``200,000 tokens and 50 turns``."""
        return " and ".join(
            _count(value, measure)
            for measure in MEASURES
            if (value := getattr(self, measure)) is not None
        )

    def proxy_env(self, used: Mapping[str, int] | None = None) -> dict[str, str]:
        """The variables the trial's model proxy reads its budget from, and
        what earlier proxies of the trial counted (``used``)."""
        env = {
            name: str(value)
            for name, value in (
                (TOKEN_BUDGET_ENV, self.tokens),
                (TURN_BUDGET_ENV, self.turns),
            )
            if value is not None
        }
        carried = {
            measure: int(used.get(measure) or 0)
            for measure in MEASURES
            if used and used.get(measure)
        }
        if carried:
            env[USED_ENV] = json.dumps(carried, separators=(",", ":"), sort_keys=True)
        return env


def task_budget(agent: Any) -> TrialBudget | None:
    """The budget a task's ``[agent]`` section sets, or None."""
    tokens = getattr(agent, "token_budget", None)
    turns = getattr(agent, "turn_budget", None)
    if tokens is None and turns is None:
        return None
    return TrialBudget(tokens=tokens, turns=turns)


def effective_budget(
    run: TrialBudget | None, task: TrialBudget | None
) -> tuple[TrialBudget | None, dict[str, str]]:
    """The budget a trial runs under and where each measure came from
    (``run`` or ``task``): a measure the run sets replaces the task's."""
    values: dict[str, int] = {}
    sources: dict[str, str] = {}
    for measure in MEASURES:
        for source, budget in (("run", run), ("task", task)):
            value = getattr(budget, measure, None) if budget is not None else None
            if value is not None:
                values[measure], sources[measure] = value, source
                break
    if not values:
        return None, {}
    return TrialBudget(**values), sources


def unenforceable_reason(
    agent: str, model: str | None, agent_env: Mapping[str, str] | None
) -> str | None:
    """Why a budget could not be enforced on *agent* running *model*: its
    model calls would go around BenchFlow's model proxy, as planning reads
    its route (the rollout checks the proxy it really started). None when
    they go through the proxy, or the agent makes no model call of its own
    (a control run, an operator script)."""
    from benchflow.agents.env import (
        inherit_agent_auth_env,
        uses_direct_anthropic_endpoint,
        uses_native_subscription_auth,
    )
    from benchflow.agents.registry import agent_routes_own_model, is_scripted_agent

    if is_scripted_agent(agent):
        return None
    if agent_routes_own_model(agent):
        return 'it calls its own model endpoint (model_route = "agent")'
    if not model:
        return "the run names no model for the proxy to route"
    env = inherit_agent_auth_env(agent, model, dict(agent_env or {}))
    if uses_direct_anthropic_endpoint(agent, model, env):
        return "the run gives it an endpoint and token of its own"
    if uses_native_subscription_auth(agent, model, env):
        return "it signs in with a subscription login, which goes around the proxy"
    return None


def budget_refusal(refused: Any) -> dict[str, Any] | None:
    """The first record of a call the proxy refused for the trial's budget,
    from a trajectory's ``refused_requests`` metadata, with how many such
    calls it refused (``refused_calls``); None when it refused none."""
    hits = [
        entry
        for entry in refused or []
        if isinstance(entry, Mapping) and entry.get("rule") == SPENT_RULE
    ]
    if not hits:
        return None
    first = dict(hits[0])
    first["refused_calls"] = len(hits)
    return first


def stop_text(record: Mapping[str, Any]) -> str:
    """The error of a trial its budget stopped, from the proxy's refusal record."""
    measure = str(record.get("budget") or "tokens")
    limits = record.get("limits")
    limits = limits if isinstance(limits, Mapping) else {}
    used = record.get("used")
    used = used if isinstance(used, Mapping) else {}
    limit = limits.get(measure)
    spent = (
        f"the {measure[:-1]} budget of {limit:,} is spent"
        if isinstance(limit, int)
        else f"the {measure[:-1]} budget is spent"
    )
    counts = ", ".join(
        _count(int(used[m]), m) for m in MEASURES if isinstance(used.get(m), int)
    )
    return (
        f"{STOP_TEXT}: {spent}"
        + (
            f" ({counts} used when the model proxy refused the next call)"
            if counts
            else ""
        )
        + "; the agent was stopped and the verifier graded what it left"
    )


class TrialBudgetSpentError(Exception):
    """The rollout stopped the agent because the proxy refused a call for the
    trial's budget; ``record`` is the proxy's refusal record."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self.record = dict(record)
        super().__init__(stop_text(record))


class TrialBudgetUnenforceableError(AgentModelNotOfferedError):
    """A trial has a budget, but its agent's model calls do not go through
    BenchFlow's model proxy, which alone counts them. Raised before the
    prompt: an unscored ``agent_budget`` integration failure, not retried."""

    cause = "agent_budget"
    evidence_source = "the trial's model route"

    def __init__(self, agent: str, budget: TrialBudget, why: str) -> None:
        self.agent = agent
        self.model = ""
        self.offered = ()
        self.budget = budget
        RuntimeError.__init__(
            self,
            f"agent integration failure [agent_budget]: the trial budget "
            f"({budget.text()}) cannot be enforced: {agent} calls its model "
            f"around BenchFlow's model proxy ({why}), and the proxy is what "
            "counts a trial's tokens and turns",
        )
