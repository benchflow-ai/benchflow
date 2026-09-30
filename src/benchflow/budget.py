"""A hard per-job budget: stop a job when it has spent enough.

``Budget`` caps one evaluation job on any of four measures. A *rollout* is one
run of a task: a trial's first attempt, or a retry of it.

- ``max_cost_usd``: USD summed over finished rollouts that reported a cost
  (provider- or price-table-derived, see usage tracking; an estimate from the
  agent's own session log counts too). Rollouts without a USD figure are
  counted in ``usd_unknown_trials`` and add nothing, so this cap is only as
  complete as the USD data.
- ``max_sandbox_seconds``: trial wall-clock seconds (from the trial's start,
  sandbox creation included, to its end, retries included), summed over
  finished trials *and* the elapsed time of running ones. Always known.
- ``max_tokens``: total tokens over finished rollouts. Known whenever the
  agent's usage reaches BenchFlow (``tokens_unknown_trials`` counts the
  others).
- ``max_rollouts``: rollouts started, retries included. Always known.

When a spend cap (USD, sandbox-seconds, tokens) is reached (spent >= cap) the
job stops starting rollouts and cancels the running ones: their rollouts are
cancelled, which runs sandbox cleanup and writes no ``result.json``. At the
rollout cap no rollout starts, and running ones finish. Neither kind is a
failure: they are left out of the totals and listed in ``summary.json``'s
``budget`` block (``cancelled``, ``not_started``, ``reason``, ``caps``,
``spent``), and a resume runs them, counting what the job already spent.

Tokens and USD are only known when a rollout finishes, so those two caps are
enforced between rollout starts: once rollouts have finished, a new one waits
while the running ones, at the job's mean spend per finished rollout, would
reach the cap. A job therefore passes a USD or token cap by about one
rollout's spend, except for the rollouts that start before any has finished
(the first ``concurrency`` of them), which start without an estimate.

>>> Budget(max_tokens=1_000).exceeded({"tokens": 1_000}) is not None
True
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Budget", "BudgetGuard"]

_CAPS = ("max_cost_usd", "max_sandbox_seconds", "max_tokens", "max_rollouts")


@dataclass(frozen=True)
class Budget:
    """Caps for one job; set at least one (see the module docstring).

    >>> Budget(max_cost_usd=20, max_rollouts=100)
    Budget(max_cost_usd=20.0, max_sandbox_seconds=None, max_tokens=None, max_rollouts=100)
    """

    max_cost_usd: float | None = None
    max_sandbox_seconds: float | None = None
    max_tokens: int | None = None
    max_rollouts: int | None = None

    def __post_init__(self) -> None:
        if all(getattr(self, name) is None for name in _CAPS):
            raise ValueError(
                "Budget needs at least one of max_cost_usd, max_sandbox_seconds, "
                "max_tokens, max_rollouts"
            )
        for name in ("max_cost_usd", "max_sandbox_seconds"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive number, got {value}")
                object.__setattr__(self, name, value)
        for name in ("max_tokens", "max_rollouts"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            object.__setattr__(self, name, int(value))

    @classmethod
    def coerce(cls, value: Budget | Mapping[str, Any] | None) -> Budget | None:
        """A Budget from itself, a mapping (``to_dict``'s shape) or None."""
        if value is None or isinstance(value, Budget):
            return value
        if isinstance(value, Mapping):
            if all(value.get(k) is None for k in _CAPS):
                return None
            return cls(**{k: value.get(k) for k in _CAPS})
        raise TypeError(
            f"budget must be a Budget or a mapping, got {type(value).__name__}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The caps as a mapping (``summary.json``'s ``budget.caps``)."""
        return asdict(self)

    def exceeded(self, spent: Mapping[str, Any]) -> str | None:
        """The reason a spend cap (USD, sandbox-seconds, tokens) is reached by
        ``spent``, or None. The rollout cap limits starts instead: see
        :class:`BudgetGuard`."""
        checks = (
            ("cost_usd", self.max_cost_usd, "USD", "${:.4f}"),
            ("sandbox_seconds", self.max_sandbox_seconds, "sandbox-seconds", "{:.1f}"),
            ("tokens", self.max_tokens, "tokens", "{:,}"),
        )
        for key, cap, label, fmt in checks:
            value = spent.get(key) or 0
            if cap is not None and value >= cap:
                return (
                    f"budget reached: {fmt.format(value)} {label} spent of a "
                    f"{fmt.format(cap)} cap"
                )
        return None


def _verifier_sandbox_seconds(timing: Any) -> float:
    """A separate verifier sandbox runs beside the agent's: both are billed."""
    value = (
        timing.get("verifier_sandbox_total") if isinstance(timing, Mapping) else None
    )
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, float(value))
    return 0.0


def _recorded_verifier_sandbox_seconds(result: Any) -> float:
    rollout_dir = getattr(result, "rollout_dir", None)
    if rollout_dir is None:
        return 0.0
    try:
        timing = json.loads((Path(rollout_dir) / "timing.json").read_text())
    except (OSError, ValueError):
        return 0.0
    return _verifier_sandbox_seconds(timing)


@dataclass
class BudgetGuard:
    """Tracks one job's spend against a :class:`Budget` while it runs.

    The Evaluation calls :meth:`admit` (or :meth:`start`) before a trial,
    :meth:`retry` before each retry of it, :meth:`attempt_done` after every
    rollout and :meth:`finish` when the trial is over.
    """

    budget: Budget
    cost_usd: float = 0.0
    sandbox_seconds: float = 0.0
    tokens: int = 0
    # Rollouts started (on resume: every attempt the job folder records).
    rollouts: int = 0
    # Finished rollouts that reported no USD / no tokens.
    usd_unknown_trials: int = 0
    tokens_unknown_trials: int = 0
    reason: str | None = None
    cancelled: list[str] = field(default_factory=list)
    not_started: list[str] = field(default_factory=list)
    _running: dict[str, float] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    _cancel_requested: set[str] = field(default_factory=set)
    _warned_usd: bool = False
    # A spend cap was reached: running rollouts are cancelled too.
    _cancelling: bool = False
    # Finished rollouts that reported USD / tokens (the means' denominators).
    _usd_rollouts: int = 0
    _token_rollouts: int = 0
    # Trials whose rollouts reported through attempt_done.
    _reported: set[str] = field(default_factory=set)
    _changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def stopped(self) -> bool:
        """Whether no new rollout may start (a cap was reached)."""
        return self.reason is not None

    def seed(self, results: Iterable[Mapping[str, Any]], interrupted: int = 0) -> None:
        """Count what the job already spent: every attempt's ``result.json``
        (a resumed job's finished and retried rollouts) and ``interrupted``
        attempts that wrote none (killed, or cancelled for the budget)."""
        for r in results:
            agent = r.get("agent_result") or {}
            timing = r.get("timing") or {}
            seconds = timing.get("total") if isinstance(timing, Mapping) else None
            self.rollouts += 1
            self._add(
                cost=agent.get("cost_usd", r.get("cost_usd")),
                tokens=agent.get("total_tokens", r.get("total_tokens")),
                seconds=(seconds if isinstance(seconds, int | float) else 0.0)
                + _verifier_sandbox_seconds(timing),
            )
        self.rollouts += max(0, interrupted)
        self.check()

    def _add(self, *, cost: Any, tokens: Any, seconds: float) -> None:
        self.sandbox_seconds += max(0.0, float(seconds))
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            self.tokens += tokens
            self._token_rollouts += 1
        else:
            self.tokens_unknown_trials += 1
        if (
            isinstance(cost, int | float)
            and not isinstance(cost, bool)
            and math.isfinite(cost)
        ):
            self.cost_usd += float(cost)
            self._usd_rollouts += 1
        else:
            self.usd_unknown_trials += 1
            if self.budget.max_cost_usd is not None and not self._warned_usd:
                self._warned_usd = True
                logger.warning(
                    "A rollout finished without a USD cost; --max-cost-usd counts "
                    "only rollouts that report one (see summary.json budget.spent)."
                )

    def spent(self, *, now: float | None = None) -> dict[str, Any]:
        """What the job has spent so far (``summary.json``'s ``budget.spent``)."""
        now = time.monotonic() if now is None else now
        running = sum(now - t for t in self._running.values())
        return {
            "cost_usd": round(self.cost_usd, 10),
            "sandbox_seconds": round(self.sandbox_seconds + running, 3),
            "tokens": self.tokens,
            "rollouts": self.rollouts,
            "usd_unknown_trials": self.usd_unknown_trials,
            "tokens_unknown_trials": self.tokens_unknown_trials,
        }

    def _rollout_cap_reached(self) -> bool:
        """Whether the rollout cap refuses another start (and says so)."""
        cap = self.budget.max_rollouts
        if cap is None or self.rollouts < cap:
            return False
        if self.reason is None:
            self.reason = (
                f"budget reached: {self.rollouts:,} rollouts started of a {cap:,} cap"
            )
            logger.warning(
                "%s; stopping: no new rollouts, %d running trial(s) finish",
                self.reason,
                len(self._running),
            )
        return True

    def start(self, name: str, task: asyncio.Task[Any] | None = None) -> bool:
        """Mark ``name`` running; False (and recorded) when no rollout may start."""
        if self.stopped or self._rollout_cap_reached():
            self.not_started.append(name)
            return False
        self._running[name] = time.monotonic()
        self.rollouts += 1
        if task is not None:
            self._tasks[name] = task
        return True

    def attach(self, name: str, task: asyncio.Task[Any]) -> None:
        """Register the task running trial ``name``, so the budget can cancel it."""
        if name in self._running:
            self._tasks[name] = task

    def projected(self) -> str | None:
        """Why another trial should wait: the running ones, at the mean spend
        per finished rollout so far, would reach the USD or token cap with it.
        None when it may start (or nothing has finished to estimate from)."""
        starting = len(self._running) + 1
        checks = (
            (
                self.budget.max_cost_usd,
                self.cost_usd,
                self._usd_rollouts,
                "USD",
                "${:.4f}",
            ),
            (
                self.budget.max_tokens,
                self.tokens,
                self._token_rollouts,
                "tokens",
                "{:,.0f}",
            ),
        )
        for cap, spent, finished, label, fmt in checks:
            if cap is None or not finished:
                continue
            estimate = spent + starting * spent / finished
            if estimate >= cap:
                return (
                    f"{fmt.format(estimate)} {label} expected with {starting} more "
                    f"rollout(s) at the mean so far, of a {fmt.format(cap)} cap"
                )
        return None

    async def admit(self, name: str, task: asyncio.Task[Any] | None = None) -> bool:
        """:meth:`start`, once the budget allows it.

        While trials are running and :meth:`projected` says their expected
        spend would reach a cap, wait for one of them to finish, then decide
        again. With nothing running, a trial starts whenever no cap is reached.
        """
        while not self.stopped and self._running and self.projected() is not None:
            changed = self._changed
            await changed.wait()
        return self.start(name, task)

    def retry(self, name: str) -> bool:
        """Whether trial ``name`` may start another rollout (a retry); counts it."""
        if self.stopped or self._rollout_cap_reached():
            return False
        self.rollouts += 1
        return True

    def attempt_done(self, name: str, result: Any) -> None:
        """Count one finished rollout of trial ``name``: its USD and tokens,
        and a separate verifier sandbox's seconds."""
        self._reported.add(name)
        self._count(result)
        self.check(exclude=name)
        self._notify()

    def _count(self, result: Any) -> None:
        self._add(
            cost=getattr(result, "cost_usd", None),
            tokens=getattr(result, "total_tokens", None),
            seconds=_recorded_verifier_sandbox_seconds(result),
        )

    def finish(self, name: str, result: Any = None) -> None:
        """Trial ``name`` is over: count its wall-clock seconds (every attempt),
        and ``result``'s spend unless its rollouts reported through
        :meth:`attempt_done`."""
        started = self._running.pop(name, None)
        self._tasks.pop(name, None)
        self.sandbox_seconds += 0.0 if started is None else time.monotonic() - started
        if result is not None and name not in self._reported:
            self._count(result)
        self._reported.discard(name)
        self.check()
        self._notify()

    def was_cancelled(self, name: str) -> bool:
        """Record a running trial cancelled for the budget (True), or not ours."""
        started = self._running.pop(name, None)
        self._tasks.pop(name, None)
        self._reported.discard(name)
        self._notify()
        if name not in self._cancel_requested:
            return False
        if started is not None:
            self.sandbox_seconds += time.monotonic() - started
        self.cancelled.append(name)
        return True

    def _notify(self) -> None:
        """Wake trials waiting in :meth:`admit`."""
        self._changed.set()
        self._changed = asyncio.Event()

    def check(self, *, exclude: str | None = None) -> str | None:
        """Stop the job (cancelling running trials, except ``exclude``) once a
        spend cap is reached."""
        if not self._cancelling:
            reason = self.budget.exceeded(self.spent())
            if reason is not None:
                self.reason = reason
                self._cancelling = True
                logger.warning(
                    "%s; stopping: no new rollouts, %d running trial(s) cancelled",
                    reason,
                    len([n for n in self._running if n != exclude]),
                )
                self._notify()
        if self._cancelling:
            for name, task in list(self._tasks.items()):
                if name == exclude or name in self._cancel_requested or task.done():
                    continue
                self._cancel_requested.add(name)
                task.cancel(self.reason)
        return self.reason

    async def watch(self, interval: float = 0.2) -> None:
        """Re-check running trials' sandbox-seconds until cancelled."""
        while True:
            await asyncio.sleep(interval)
            if self._running:
                self.check()

    def summary(self) -> dict[str, Any]:
        """The ``budget`` block of summary.json."""
        return {
            "caps": self.budget.to_dict(),
            "spent": self.spent(),
            "stopped": self.stopped,
            "reason": self.reason,
            "cancelled": sorted(self.cancelled),
            "not_started": sorted(self.not_started),
        }


def guard_for(budget: Budget | None) -> BudgetGuard | None:
    return None if budget is None else BudgetGuard(budget)
