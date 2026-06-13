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

from benchflow._utils.reward_events import memory_score_from_result
from benchflow._utils.scoring import (
    classify_error,
    classify_score_outcome,
    classify_verifier_error,
    pass_rate,
    pass_rate_excl_errors,
)
from benchflow.trajectories.metrics import result_skill_invocations
from benchflow.usage_tracking import is_trusted_usage_source

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for a single task."""

    task_name: str
    reward: float | None = None
    n_tool_calls: int = 0
    n_skill_invocations: int = 0
    n_prompts: int = 0
    error: str | None = None
    verifier_error: str | None = None
    duration_sec: float = 0.0
    agent_name: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    usage_source: str = "unavailable"
    memory_score: float | None = None

    @property
    def outcome(self) -> str:
        return self.score_outcome

    @property
    def score_outcome(self) -> str:
        """Terminal score bucket via the shared classifier (see _utils.scoring).

        Guarantees passed/failed/errored/verifier_errored are disjoint and
        exhaustive — the same classification used by ``EvaluationResult``.
        """
        return classify_score_outcome(self._result_shape)

    @property
    def _result_shape(self) -> dict[str, Any]:
        return {
            "rewards": {"reward": self.reward} if self.reward is not None else None,
            "error": self.error,
            "verifier_error": self.verifier_error,
        }

    @property
    def passed(self) -> bool:
        return self.score_outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.score_outcome == "failed"

    @property
    def errored(self) -> bool:
        return self.score_outcome == "errored"

    @property
    def verifier_errored(self) -> bool:
        """Backward-compatible alias for verifier-error evidence."""
        return self.has_verifier_error_evidence

    @property
    def has_verifier_error_evidence(self) -> bool:
        """True when the task carries verifier-error evidence."""
        return self.verifier_error is not None

    @property
    def score_verifier_errored(self) -> bool:
        """True when the disjoint score bucket is verifier_errored."""
        return self.score_outcome == "verifier_errored"

    @property
    def completed(self) -> bool:
        """True when task reached a terminal non-error reward state."""
        return self.score_outcome in ("passed", "failed")


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
        """Backward-compatible alias for score-verifier-error count."""
        return self.score_verifier_errored

    @property
    def score_verifier_errored(self) -> int:
        """Count of tasks whose disjoint score bucket is verifier_errored."""
        return sum(1 for t in self.tasks if t.score_verifier_errored)

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
        completed = [t for t in self.tasks if t.completed]
        return (
            sum(t.n_tool_calls for t in completed) / len(completed)
            if completed
            else 0.0
        )

    @property
    def avg_skill_invocations(self) -> float:
        """Average structured skill invocations per completed task."""
        completed = [t for t in self.tasks if t.completed]
        return (
            sum(t.n_skill_invocations for t in completed) / len(completed)
            if completed
            else 0.0
        )

    @property
    def avg_duration(self) -> float:
        """Average duration per completed task (seconds)."""
        completed = [t for t in self.tasks if t.completed and t.duration_sec > 0]
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
    def telemetry_tasks(self) -> list[TaskMetrics]:
        """Completed tasks with provider telemetry."""
        return [
            t
            for t in self.tasks
            if t.completed and is_trusted_usage_source(t.usage_source)
        ]

    @property
    def telemetry_completed_tasks(self) -> list[TaskMetrics]:
        """Completed tasks included in telemetry coverage denominator."""
        return [t for t in self.tasks if t.completed]

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens or 0 for t in self.telemetry_tasks)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens or 0 for t in self.telemetry_tasks)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(t.cache_read_tokens or 0 for t in self.telemetry_tasks)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(t.cache_creation_tokens or 0 for t in self.telemetry_tasks)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens or 0 for t in self.telemetry_tasks)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(t.cost_usd or 0.0 for t in self.telemetry_tasks), 10)

    @property
    def avg_cost_per_trial_usd(self) -> float | None:
        if not self.telemetry_tasks:
            return None
        return round(self.total_cost_usd / len(self.telemetry_tasks), 10)

    @property
    def telemetry_coverage(self) -> float:
        completed = self.telemetry_completed_tasks
        if not completed:
            return 0.0
        return len(self.telemetry_tasks) / len(completed)

    @property
    def memory_scores(self) -> dict[str, float]:
        return {
            t.task_name: t.memory_score
            for t in self.tasks
            if t.memory_score is not None
        }

    @property
    def memory_score(self) -> float | None:
        scores = list(self.memory_scores.values())
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def memory_summary(self) -> dict[str, Any]:
        avg = self.memory_score
        return {
            "scored": len(self.memory_scores),
            "avg_score": avg,
            "score": f"{avg:.1%}" if avg is not None else None,
        }

    @property
    def verifier_error_breakdown(self) -> dict[str, int]:
        """Categorize verifier errors."""
        breakdown: dict[str, int] = {}
        for t in self.tasks:
            if not t.has_verifier_error_evidence:
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
            "avg_skill_invocations": round(self.avg_skill_invocations, 1),
            "avg_duration_sec": round(self.avg_duration, 1),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_trial_usd": self.avg_cost_per_trial_usd,
            "telemetry_coverage": self.telemetry_coverage,
            "memory_score": self.memory_score,
            "memory_score_coverage": (
                len(self.memory_scores) / self.total if self.total else 0.0
            ),
            "memory": self.memory_summary,
            "memory_scores": self.memory_scores,
            "error_breakdown": self.error_breakdown,
            "verifier_error_breakdown": self.verifier_error_breakdown,
            "passed_tasks": sorted(t.task_name for t in self.tasks if t.passed),
            "failed_tasks": sorted(t.task_name for t in self.tasks if t.failed),
            "errored_tasks": sorted(t.task_name for t in self.tasks if t.errored),
            "verifier_errored_tasks": sorted(
                t.task_name for t in self.tasks if t.score_verifier_errored
            ),
        }


def _safe_reward(rewards: dict) -> float:
    """Extract reward value from a rewards dict, defaulting to 0 if None/missing.

    Prevents TypeError when comparing reward values where one is None
    (e.g. rewards={"reward": None, "rubric": [...]}).
    """
    val = rewards.get("reward")
    return val if isinstance(val, (int, float)) else 0.0


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

    for rfile in sorted(results_dir.rglob("result.json")):
        try:
            r = json.loads(rfile.read_text())
            task = r["task_name"]
            if (
                task not in best
                or (r.get("rewards") is not None and best[task].get("rewards") is None)
                or (
                    r.get("rewards")
                    and best[task].get("rewards")
                    and _safe_reward(r["rewards"]) > _safe_reward(best[task]["rewards"])
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
                n_skill_invocations=result_skill_invocations(r),
                n_prompts=r.get("n_prompts", 0),
                error=r.get("error"),
                verifier_error=r.get("verifier_error"),
                duration_sec=duration,
                agent_name=r.get("agent_name", ""),
                input_tokens=(r.get("agent_result") or {}).get(
                    "n_input_tokens", r.get("n_input_tokens")
                ),
                output_tokens=(r.get("agent_result") or {}).get(
                    "n_output_tokens", r.get("n_output_tokens")
                ),
                cache_read_tokens=(r.get("agent_result") or {}).get(
                    "n_cache_read_tokens", r.get("n_cache_read_tokens")
                ),
                cache_creation_tokens=(r.get("agent_result") or {}).get(
                    "n_cache_creation_tokens", r.get("n_cache_creation_tokens")
                ),
                total_tokens=(r.get("agent_result") or {}).get(
                    "total_tokens", r.get("total_tokens")
                ),
                cost_usd=(r.get("agent_result") or {}).get(
                    "cost_usd", r.get("cost_usd")
                ),
                usage_source=(r.get("agent_result") or {}).get(
                    "usage_source", r.get("usage_source", "unavailable")
                ),
                memory_score=memory_score_from_result(r),
            )
        )

    return BenchmarkMetrics(
        benchmark=benchmark,
        agent=agent,
        model=model,
        tasks=tasks,
    )
