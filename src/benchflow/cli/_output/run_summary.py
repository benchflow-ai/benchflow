"""The final screen of a run, and its ``--json`` document.

The default screen is one row per trial (outcome, reward, cost, tokens,
time), one footer (score, model cost, tokens, wall time) and the results
folder, in plain aligned text, so it reads the same on a TTY, in a pipe and
in a CI log. The figures follow the run's own records:

- the score cell is ``passed/total`` with the unscored and not-run trials
  named beside it (the benchflow.score.v1 denominator, unchanged);
- a cost is the provider's bill (``$0.0108``), a list-price estimate
  (``~$0.0108``), a floor when a trial counted only some of its calls or
  reported none (``≥~$0.0108 · 3/4 trials``), unknown (``—``), or ``$0`` for
  a control that called no model, its verifier's LLM judge included; a
  missing figure is never a 0;
- a trial's cost and tokens are its agent's and its judge's together
  (result.json keeps them apart: ``agent_result`` and ``judge_usage``);
- tokens are the known sum, ``—`` when nothing counted them;
- when the job paid for more attempts than the trials it kept (a resumed
  job's stopped attempt, a retry), one more line gives summary.json's
  ``spend_all_attempts``, a floor (``≥``) when an attempt's usage was
  salvaged from its logs or is unknown.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._utils.aware_time import aware

SCHEMA = "benchflow.run-result.v1"
#: Agents that call no model: their spend is $0, not unknown.
CONTROL_AGENTS = frozenset({"oracle", "nop"})
#: Trial rows printed before "… N more".
MAX_ROWS = 20
_OUTCOME_LABEL = {
    "passed": "passed",
    "failed": "failed",
    "errored": "unscored",
    "verifier_errored": "verifier error",
    "partial": "partial",
}
_DASH = "—"


@dataclass(frozen=True)
class TrialRow:
    """One trial of the final screen."""

    task: str
    trial: str
    outcome: str
    reward: float | None
    cost_usd: float | None
    cost_estimated: bool
    cost_partial: bool
    tokens: int | None
    seconds: float | None
    error_category: str | None
    #: A control that called no model, no judge either: $0, no tokens.
    model_free: bool = False
    judge_calls: int = 0
    judge_cost_usd: float | None = None
    judge_tokens: int | None = None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def _seconds(trial: Any) -> float | None:
    start, end = getattr(trial, "started_at", None), getattr(trial, "finished_at", None)
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max(0.0, (aware(end) - aware(start)).total_seconds())
    return None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _spend(trial: Any, control: bool) -> dict[str, Any]:
    """A trial's model spend: its agent's (none for a control) and its
    verifier's LLM judge's (``judge_usage``) as one cost and token count."""
    judge = getattr(trial, "judge_usage", None)
    judge = judge if isinstance(judge, dict) else {}
    calls = _count(judge.get("calls")) or 0
    # (usd, estimated, partial, tokens) of each part that spent.
    parts: list[tuple[float | None, bool, bool, int | None]] = []
    if not control:
        parts.append(
            (
                _number(getattr(trial, "cost_usd", None)),
                getattr(trial, "cost_source", None) != "provider",
                getattr(trial, "usage_incomplete", None) is True,
                _count(getattr(trial, "total_tokens", None)),
            )
        )
    judge_usd = _number(judge.get("cost_usd")) if calls else None
    judge_tokens = _count(judge.get("total_tokens")) if calls else None
    if calls:
        parts.append(
            (
                judge_usd,
                judge.get("cost_source") != "provider",
                bool(judge.get("calls_without_usage")),
                judge_tokens,
            )
        )
    known = [part for part in parts if part[0] is not None]
    tokens = [part[3] for part in parts if part[3] is not None]
    return {
        "model_free": control and not calls,
        "cost_usd": math.fsum(part[0] or 0.0 for part in known) if known else None,
        "cost_estimated": any(part[1] for part in known),
        # A part whose cost is unknown beside one that is known: a floor.
        "cost_partial": any(part[2] for part in parts) or len(known) < len(parts),
        "tokens": sum(tokens) if tokens else None,
        "judge_calls": calls,
        "judge_cost_usd": judge_usd,
        "judge_tokens": judge_tokens,
    }


def trial_rows(result: Any) -> list[TrialRow]:
    """The run's trials, in task order, as the final screen shows them."""
    rows = []
    control = is_control(result)
    results = getattr(result, "results", None) or {}
    for name in sorted(results):
        trial = results[name]
        outcome = str(getattr(trial, "score_outcome", "") or "")
        reward = _number(getattr(trial, "reward", None))
        if outcome == "failed" and reward is not None and 0.0 < reward < 1.0:
            # Partial credit is its own state, never read as a plain fail.
            outcome = "partial"
        rows.append(
            TrialRow(
                task=str(getattr(trial, "task_name", "") or name),
                trial=str(getattr(trial, "rollout_name", "") or name),
                outcome=_OUTCOME_LABEL.get(outcome, outcome or "unscored"),
                reward=reward,
                seconds=_seconds(trial),
                error_category=getattr(trial, "error_category", None),
                **_spend(trial, control),
            )
        )
    return rows


def is_control(result: Any) -> bool:
    agent = str(getattr(getattr(result, "config", None), "agent", "") or "")
    return agent.lower() in CONTROL_AGENTS


def _usd(value: float) -> str:
    return f"${value:.2f}" if value >= 1 else f"${value:.4f}"


def cost_cell(rows: Sequence[TrialRow]) -> str:
    """The model cost of ``rows`` (see the module docstring)."""
    if rows and all(row.model_free for row in rows):
        return "$0"
    priced = [row for row in rows if row.model_free or row.cost_usd is not None]
    if not priced:
        return _DASH
    text = (
        "~" if any(row.cost_estimated for row in priced if not row.model_free) else ""
    ) + _usd(math.fsum(row.cost_usd or 0.0 for row in priced))
    if len(priced) < len(rows):
        return f"≥{text} · {len(priced)}/{len(rows)} trials"
    if any(row.cost_partial for row in priced):
        return f"≥{text}"
    return text


def tokens_cell(rows: Sequence[TrialRow]) -> str:
    # A model-free control trial used no tokens; it is no unknown count.
    spent = [row for row in rows if not row.model_free]
    known = [row.tokens for row in spent if row.tokens is not None]
    if not known:
        return _DASH
    text = f"{sum(known):,}"
    return f"≥{text}" if len(known) < len(spent) else text


def every_attempt_spend(job_dir: Path | None) -> dict[str, Any] | None:
    """summary.json's ``spend_all_attempts`` for ``job_dir``, or None."""
    import json

    if job_dir is None:
        return None
    try:
        summary = json.loads((Path(job_dir) / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    block = summary.get("spend_all_attempts") if isinstance(summary, dict) else None
    return block if isinstance(block, dict) else None


def _attempt_floor(block: dict[str, Any]) -> bool:
    return any(
        (_count(block.get(key)) or 0) > 0
        for key in (
            "attempts_usage_salvaged",
            "attempts_cost_incomplete",
            "attempts_without_cost",
            "stopped_attempts_without_usage",
            "stopped_attempts_without_cost",
        )
    )


def attempts_beyond_kept(
    block: dict[str, Any] | None, rows: Sequence[TrialRow]
) -> bool:
    """Whether the job paid for attempts its kept trials don't show: a
    stopped attempt (Ctrl-C, a killed run, a budget) or a retry."""
    if block is None:
        return False
    attempts = (_count(block.get("attempts")) or 0) + (
        _count(block.get("stopped_attempts")) or 0
    )
    return attempts > len(rows) or (_count(block.get("stopped_attempts")) or 0) > 0


def every_attempt_line(block: dict[str, Any], rows: Sequence[TrialRow]) -> str:
    """``Every attempt: ≥~$0.0080, ≥34,488 tokens (2 attempts, 1 stopped, 1
    counted from its logs)``."""
    usd = _number(block.get("cost_usd"))
    tokens = _count(block.get("tokens"))
    floor = "≥" if _attempt_floor(block) else ""
    stopped = _count(block.get("stopped_attempts")) or 0
    salvaged = _count(block.get("attempts_usage_salvaged")) or 0
    estimated = stopped > 0 or any(
        row.cost_estimated for row in rows if not row.model_free
    )
    cost = (
        _DASH
        if usd is None
        else "$0"
        if usd == 0 and not floor
        else floor + ("~" if estimated else "") + _usd(usd)
    )
    count = f"{tokens:,} tokens" if tokens is not None else "tokens unknown"
    if tokens is not None and floor:
        count = "≥" + count
    attempts = (_count(block.get("attempts")) or 0) + stopped
    detail = [f"{attempts} attempts"]
    if stopped:
        detail.append(f"{stopped} stopped")
    if salvaged:
        detail.append(
            f"{salvaged} counted from {'its' if salvaged == 1 else 'their'} logs"
        )
    return f"Every attempt: {cost}, {count} ({', '.join(detail)})"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return _DASH
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _reward(value: float | None) -> str:
    return _DASH if value is None else f"{value:.3g}"


def score_cell(result: Any) -> str:
    """``passed/total passed`` with the unscored and not-run trials beside it."""
    total = int(getattr(result, "total", 0) or 0)
    passed = int(getattr(result, "passed", 0) or 0)
    unscored = int(getattr(result, "errored", 0) or 0) + int(
        getattr(result, "verifier_errored", 0) or 0
    )
    not_run = int(getattr(result, "not_run", 0) or 0)
    parts = [f"{passed}/{total} passed"]
    if unscored:
        parts.append(f"{unscored} unscored")
    if not_run:
        parts.append(f"{not_run} not run")
    mean = _number(getattr(result, "mean_reward", None))
    rewards = [row.reward for row in trial_rows(result) if row.reward is not None]
    if mean is not None and any(r not in (0.0, 1.0) for r in rewards):
        parts.append(f"mean reward {mean:.2f}")
    return ", ".join(parts)


def _table(
    header: Sequence[str], rows: Iterable[Sequence[str]], right: set[int]
) -> list[str]:
    body = [list(header), *[list(row) for row in rows]]
    widths = [max(len(row[i]) for row in body) for i in range(len(header))]
    lines = []
    for row in body:
        cells = [
            cell.rjust(widths[i]) if i in right else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
    return lines


def summary_lines(
    result: Any, job_dir: Path | None, *, width: int = 120, max_rows: int = MAX_ROWS
) -> list[str]:
    """The default final screen as plain lines (no markup)."""
    rows = trial_rows(result)
    lines: list[str] = []
    reused = int(getattr(result, "reused", 0) or 0)
    if reused:
        ran = int(getattr(result, "ran", 0) or 0)
        lines.append(
            f"Resumed: {reused} finished trial(s) kept, {ran} ran now"
            if ran
            else f"Resumed: all {reused} trial(s) were finished, so nothing ran"
        )
    shown = rows[:max_rows]
    task_width = max(8, width - 60)
    trial_lines = _table(
        ("Task", "Outcome", "Reward", "Cost", "Tokens", "Time"),
        [
            (
                row.task
                if len(row.task) <= task_width
                else row.task[: task_width - 1] + "…",
                row.outcome
                + (
                    f" ({row.error_category})"
                    if row.outcome == "unscored" and row.error_category
                    else ""
                ),
                _reward(row.reward),
                cost_cell([row]),
                tokens_cell([row]),
                duration(row.seconds),
            )
            for row in shown
        ],
        right={2, 4, 5},
    )
    lines.extend(trial_lines)
    if len(rows) > len(shown):
        lines.append(f"… {len(rows) - len(shown)} more trials")
    lines.append("")
    wall = duration(_number(getattr(result, "elapsed_sec", None)))
    if reused:
        wall += " (this run)"
    lines.extend(
        _table(
            ("Score", "Model cost", "Tokens", "Wall time"),
            [
                (
                    score_cell(result),
                    cost_cell(rows),
                    tokens_cell(rows),
                    wall,
                )
            ],
            right=set(),
        )
    )
    spend = every_attempt_spend(job_dir)
    if spend is not None and attempts_beyond_kept(spend, rows):
        lines.append(every_attempt_line(spend, rows))
    if job_dir is not None:
        lines.append(f"Results: {job_dir}/")
    return lines


def summary_document(result: Any, job_dir: Path | None) -> dict[str, Any]:
    """The ``--json`` document: one per run, every figure null when unknown."""
    rows = trial_rows(result)
    spent = [row for row in rows if not row.model_free]
    priced = [row.cost_usd for row in rows if row.cost_usd is not None]
    known_tokens = [row.tokens for row in rows if row.tokens is not None]
    judged = [row for row in rows if row.judge_calls]
    config = getattr(result, "config", None)
    if rows and not spent:
        # A control whose verifier called no LLM judge either.
        cost: dict[str, Any] = {
            "usd": 0.0,
            "source": "no_model",
            "trials_priced": len(rows),
        }
    else:
        cost = {
            "usd": math.fsum(priced) if priced else None,
            "source": None
            if not priced
            else "estimate"
            if any(row.cost_estimated for row in spent if row.cost_usd is not None)
            else "provider",
            "trials_priced": len(priced) + len(rows) - len(spent),
        }
    cost["partial"] = bool(priced) and (
        len(priced) < len(spent) or any(row.cost_partial for row in spent)
    )
    # The judges' part of the cost and tokens above.
    judge_usd = [r.judge_cost_usd for r in judged if r.judge_cost_usd is not None]
    judge_tokens = [r.judge_tokens for r in judged if r.judge_tokens is not None]
    cost["judge_calls"] = sum(row.judge_calls for row in judged)
    cost["judge_cost_usd"] = math.fsum(judge_usd) if judge_usd else None
    cost["judge_tokens"] = sum(judge_tokens) if judge_tokens else None
    attempt_block = every_attempt_spend(job_dir)
    return {
        "schema": SCHEMA,
        "job_name": getattr(result, "job_name", None),
        "results_dir": str(job_dir) if job_dir is not None else None,
        "agent": getattr(config, "agent", None),
        "model": getattr(config, "model", None),
        "counts": {
            "trials": int(getattr(result, "total", 0) or 0),
            "passed": int(getattr(result, "passed", 0) or 0),
            "failed": int(getattr(result, "failed", 0) or 0),
            "unscored": int(getattr(result, "errored", 0) or 0)
            + int(getattr(result, "verifier_errored", 0) or 0),
            "not_run": int(getattr(result, "not_run", 0) or 0),
            "reused": int(getattr(result, "reused", 0) or 0),
        },
        "score": _number(getattr(result, "score", None)),
        "score_rule": "benchflow.score.v1",
        "mean_reward": _number(getattr(result, "mean_reward", None)),
        "cost": cost,
        "tokens": sum(known_tokens) if known_tokens else None,
        # Every attempt the job paid for, retries and stopped attempts
        # included (summary.json's spend_all_attempts); null when unknown.
        "spend_all_attempts": None
        if attempt_block is None
        else {
            "usd": _number(attempt_block.get("cost_usd")),
            "tokens": _count(attempt_block.get("tokens")),
            "judge_usd": _number(attempt_block.get("judge_cost_usd")),
            "attempts": (_count(attempt_block.get("attempts")) or 0)
            + (_count(attempt_block.get("stopped_attempts")) or 0),
            "stopped": _count(attempt_block.get("stopped_attempts")) or 0,
            "salvaged": _count(attempt_block.get("attempts_usage_salvaged")) or 0,
            "partial": _attempt_floor(attempt_block),
        },
        "wall_seconds": _number(getattr(result, "elapsed_sec", None)),
        "trials": [
            {
                "task": row.task,
                "trial": row.trial,
                "outcome": row.outcome,
                "reward": row.reward,
                "error_category": row.error_category,
                "cost_usd": 0.0 if row.model_free else row.cost_usd,
                "cost_estimated": None
                if row.model_free or row.cost_usd is None
                else row.cost_estimated,
                "usage_incomplete": row.cost_partial,
                "tokens": row.tokens,
                "judge_calls": row.judge_calls,
                "judge_cost_usd": row.judge_cost_usd,
                "judge_tokens": row.judge_tokens,
                "seconds": row.seconds,
            }
            for row in rows
        ],
    }
