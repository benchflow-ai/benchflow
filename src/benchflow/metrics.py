"""Metrics collection and aggregation for benchmark runs.

Computes pass rates, tool usage stats, timing, and error breakdowns
from trial results.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._scoring import (
    classify_error,
    classify_verifier_error,
    pass_rate,
    pass_rate_excl_errors,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for a single task."""

    task_name: str
    reward: float | None = None
    n_tool_calls: int = 0
    n_prompts: int = 0
    error: str | None = None
    verifier_error: str | None = None
    duration_sec: float = 0.0
    agent_name: str = ""

    @property
    def passed(self) -> bool:
        return self.reward == 1.0

    @property
    def failed(self) -> bool:
        return self.reward is not None and self.reward != 1.0

    @property
    def errored(self) -> bool:
        return self.reward is None and self.error is not None

    @property
    def verifier_errored(self) -> bool:
        """True when task failed due to verifier error (not agent error)."""
        return self.reward is None and self.verifier_error is not None


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""

    benchmark: str
    agent: str
    model: str
    tasks: list[TaskMetrics] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tasks if t.passed)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tasks if t.failed)

    @property
    def errored(self) -> int:
        return sum(1 for t in self.tasks if t.errored)

    @property
    def verifier_errored(self) -> int:
        """Count of tasks that failed due to verifier error."""
        return sum(1 for t in self.tasks if t.verifier_errored)

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)

    @property
    def avg_tool_calls(self) -> float:
        """Average tool calls per completed task."""
        completed = [t for t in self.tasks if not t.errored and not t.verifier_errored]
        return (
            sum(t.n_tool_calls for t in completed) / len(completed)
            if completed
            else 0.0
        )

    @property
    def avg_duration(self) -> float:
        """Average duration per completed task (seconds)."""
        completed = [
            t
            for t in self.tasks
            if not t.errored and not t.verifier_errored and t.duration_sec > 0
        ]
        return (
            sum(t.duration_sec for t in completed) / len(completed)
            if completed
            else 0.0
        )

    @property
    def error_breakdown(self) -> dict[str, int]:
        """Categorize errors."""
        breakdown: dict[str, int] = {}
        for t in self.tasks:
            if not t.errored:
                continue
            category = classify_error(t.error)
            if category:
                breakdown[category] = breakdown.get(category, 0) + 1
        return breakdown

    @property
    def verifier_error_breakdown(self) -> dict[str, int]:
        """Categorize verifier errors."""
        breakdown: dict[str, int] = {}
        for t in self.tasks:
            if not t.verifier_errored:
                continue
            category = classify_verifier_error(t.verifier_error)
            if category:
                breakdown[category] = breakdown.get(category, 0) + 1
        return breakdown

    def summary(self) -> dict[str, Any]:
        """Export as summary dict."""
        return {
            "benchmark": self.benchmark,
            "agent": self.agent,
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "verifier_errored": self.verifier_errored,
            "score": f"{self.score:.1%}",
            "score_excl_errors": f"{self.score_excl_errors:.1%}",
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "avg_duration_sec": round(self.avg_duration, 1),
            "error_breakdown": self.error_breakdown,
            "verifier_error_breakdown": self.verifier_error_breakdown,
            "passed_tasks": sorted(t.task_name for t in self.tasks if t.passed),
            "failed_tasks": sorted(t.task_name for t in self.tasks if t.failed),
            "errored_tasks": sorted(t.task_name for t in self.tasks if t.errored),
            "verifier_errored_tasks": sorted(
                t.task_name for t in self.tasks if t.verifier_errored
            ),
        }


def collect_metrics(
    results_dir: str | Path,
    benchmark: str = "",
    agent: str = "",
    model: str = "",
) -> BenchmarkMetrics:
    """Collect metrics from a results directory.

    Reads all result.json files, picks the best result per task
    (rewards > no rewards, higher reward preferred).
    """
    results_dir = Path(results_dir)
    best: dict[str, dict] = {}

    for rfile in results_dir.rglob("result.json"):
        try:
            r = json.loads(rfile.read_text())
            task = r["task_name"]
            if (
                task not in best
                or (r.get("rewards") is not None and best[task].get("rewards") is None)
                or (
                    r.get("rewards")
                    and best[task].get("rewards")
                    and r["rewards"].get("reward", 0)
                    > best[task]["rewards"].get("reward", 0)
                )
            ):
                best[task] = r
        except Exception as e:
            logger.debug(f"Skipping corrupt result file {rfile}: {e}")

    tasks = []
    for task_name, r in sorted(best.items()):
        reward = r.get("rewards", {}).get("reward") if r.get("rewards") else None
        # Calculate duration
        duration = 0.0
        try:
            started = datetime.fromisoformat(r["started_at"])
            finished = datetime.fromisoformat(r["finished_at"])
            duration = (finished - started).total_seconds()
        except (KeyError, ValueError):
            logger.debug("Could not compute duration for task %s", task_name)

        tasks.append(
            TaskMetrics(
                task_name=task_name,
                reward=reward,
                n_tool_calls=r.get("n_tool_calls", 0),
                n_prompts=r.get("n_prompts", 0),
                error=r.get("error"),
                verifier_error=r.get("verifier_error"),
                duration_sec=duration,
                agent_name=r.get("agent_name", ""),
            )
        )

    return BenchmarkMetrics(
        benchmark=benchmark,
        agent=agent,
        model=model,
        tasks=tasks,
    )
