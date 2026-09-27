"""A hard per-job budget: stop a job when it has spent enough.

``Budget`` caps one evaluation job on any of three measures:

- ``max_cost_usd``: USD summed over finished trials that reported a cost
  (provider- or price-table-derived, see usage tracking). Trials without a
  USD figure are counted in ``usd_unknown_trials`` and add nothing, so this
  cap is only as complete as the USD data.
- ``max_sandbox_seconds``: trial wall-clock seconds (from the trial's start,
  sandbox creation included, to its end, retries included), summed over
  finished trials *and* the elapsed time of running ones. Always known.
- ``max_tokens``: total tokens over finished trials. Known whenever the agent's
  usage reaches BenchFlow (``tokens_unknown_trials`` counts the others).

When a cap is reached (spent >= cap) the job stops launching trials and
cancels the running ones: their rollouts are cancelled, which runs sandbox
cleanup and writes no ``result.json``. Neither kind is a failure: they are
left out of the totals and listed in ``summary.json``'s ``budget`` block
(``cancelled``, ``not_started``, ``reason``, ``caps``, ``spent``), and a
resume runs them, counting what the job already spent. Because tokens and USD
arrive when a trial finishes, a job can overshoot those two caps by what the
trials in flight spend.

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


@dataclass(frozen=True)
class Budget:
    """Caps for one job; set at least one (see the module docstring)."""

    max_cost_usd: float | None = None
    max_sandbox_seconds: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        caps = (self.max_cost_usd, self.max_sandbox_seconds, self.max_tokens)
        if all(c is None for c in caps):
            raise ValueError(
                "Budget needs at least one of max_cost_usd, max_sandbox_seconds, "
                "max_tokens"
            )
        for name in ("max_cost_usd", "max_sandbox_seconds"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive number, got {value}")
                object.__setattr__(self, name, value)
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or int(self.max_tokens) <= 0:
                raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
            object.__setattr__(self, "max_tokens", int(self.max_tokens))

    @classmethod
    def coerce(cls, value: Budget | Mapping[str, Any] | None) -> Budget | None:
        """A Budget from itself, a mapping (``to_dict``'s shape) or None."""
        if value is None or isinstance(value, Budget):
            return value
        if isinstance(value, Mapping):
            if all(
                value.get(k) is None
                for k in ("max_cost_usd", "max_sandbox_seconds", "max_tokens")
            ):
                return None
            return cls(
                max_cost_usd=value.get("max_cost_usd"),
                max_sandbox_seconds=value.get("max_sandbox_seconds"),
                max_tokens=value.get("max_tokens"),
            )
        raise TypeError(
            f"budget must be a Budget or a mapping, got {type(value).__name__}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def exceeded(self, spent: Mapping[str, Any]) -> str | None:
        """The reason a cap is reached by ``spent``, or None."""
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
    """Tracks one job's spend against a :class:`Budget` while it runs."""

    budget: Budget
    cost_usd: float = 0.0
    sandbox_seconds: float = 0.0
    tokens: int = 0
    usd_unknown_trials: int = 0
    tokens_unknown_trials: int = 0
    reason: str | None = None
    cancelled: list[str] = field(default_factory=list)
    not_started: list[str] = field(default_factory=list)
    _running: dict[str, float] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    _cancel_requested: set[str] = field(default_factory=set)
    _warned_usd: bool = False

    @property
    def stopped(self) -> bool:
        return self.reason is not None

    def seed(self, results: Iterable[Mapping[str, Any]]) -> None:
        """Count what finished trials (e.g. reused on resume) already spent."""
        for r in results:
            agent = r.get("agent_result") or {}
            timing = r.get("timing") or {}
            seconds = timing.get("total") if isinstance(timing, Mapping) else None
            self._add(
                cost=agent.get("cost_usd", r.get("cost_usd")),
                tokens=agent.get("total_tokens", r.get("total_tokens")),
                seconds=(seconds if isinstance(seconds, int | float) else 0.0)
                + _verifier_sandbox_seconds(timing),
            )
        self.check()

    def _add(self, *, cost: Any, tokens: Any, seconds: float) -> None:
        self.sandbox_seconds += max(0.0, float(seconds))
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            self.tokens += tokens
        else:
            self.tokens_unknown_trials += 1
        if (
            isinstance(cost, int | float)
            and not isinstance(cost, bool)
            and math.isfinite(cost)
        ):
            self.cost_usd += float(cost)
        else:
            self.usd_unknown_trials += 1
            if self.budget.max_cost_usd is not None and not self._warned_usd:
                self._warned_usd = True
                logger.warning(
                    "A trial finished without a USD cost; --max-cost-usd counts only "
                    "trials that report one (see summary.json budget.spent)."
                )

    def spent(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        running = sum(now - t for t in self._running.values())
        return {
            "cost_usd": round(self.cost_usd, 10),
            "sandbox_seconds": round(self.sandbox_seconds + running, 3),
            "tokens": self.tokens,
            "usd_unknown_trials": self.usd_unknown_trials,
            "tokens_unknown_trials": self.tokens_unknown_trials,
        }

    def start(self, name: str, task: asyncio.Task[Any] | None = None) -> bool:
        """Mark ``name`` running; False (and recorded) when the job has stopped."""
        if self.stopped:
            self.not_started.append(name)
            return False
        self._running[name] = time.monotonic()
        if task is not None:
            self._tasks[name] = task
        return True

    def finish(self, name: str, result: Any) -> None:
        started = self._running.pop(name, None)
        self._tasks.pop(name, None)
        self._add(
            cost=getattr(result, "cost_usd", None),
            tokens=getattr(result, "total_tokens", None),
            seconds=(0.0 if started is None else time.monotonic() - started)
            + _recorded_verifier_sandbox_seconds(result),
        )
        self.check()

    def was_cancelled(self, name: str) -> bool:
        """Record a running trial cancelled for the budget (True), or not ours."""
        started = self._running.pop(name, None)
        self._tasks.pop(name, None)
        if name not in self._cancel_requested:
            return False
        if started is not None:
            self.sandbox_seconds += time.monotonic() - started
        self.cancelled.append(name)
        return True

    def check(self) -> str | None:
        """Stop the job (cancelling running trials) once a cap is reached."""
        if self.reason is None:
            reason = self.budget.exceeded(self.spent())
            if reason is not None:
                self.reason = reason
                logger.warning(
                    "%s; stopping: no new trials, %d running trial(s) cancelled",
                    reason,
                    len(self._running),
                )
        if self.reason is not None:
            for name, task in list(self._tasks.items()):
                if name not in self._cancel_requested and not task.done():
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
