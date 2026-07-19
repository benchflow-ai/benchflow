"""Paired lift reports for BenchFlow eval job directories."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA_GROUP_FIELDS = ("difficulty_level", "reward_mode_initial")
_ERROR_FIELDS = ("error", "verifier_error", "export_error")
_UNHEALTHY_REASONS = ("unhealthy", "partial_trajectory")


@dataclass(frozen=True)
class LiftRollout:
    task_id: str
    rollout_dir: Path
    reward: float | None
    metadata: dict[str, Any]
    excluded_reason: str | None

    @property
    def healthy(self) -> bool:
        return self.excluded_reason is None

    @property
    def passed(self) -> bool:
        return self.reward == 1.0


@dataclass(frozen=True)
class LiftPair:
    task_id: str
    baseline: LiftRollout
    trained: LiftRollout

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self.baseline.metadata, **self.trained.metadata}


def build_lift_report(
    baseline_jobs_dir: Path,
    trained_jobs_dir: Path,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    allow_duplicate_first_by_path: bool = False,
) -> dict[str, Any]:
    """Compare two BenchFlow job directories by task id."""

    baseline_rollouts = _load_rollouts(baseline_jobs_dir)
    trained_rollouts = _load_rollouts(trained_jobs_dir)
    baseline_selected, baseline_duplicate_count = _select_one_healthy_per_task(
        baseline_rollouts,
        side="baseline",
        allow_duplicate_first_by_path=allow_duplicate_first_by_path,
    )
    trained_selected, trained_duplicate_count = _select_one_healthy_per_task(
        trained_rollouts,
        side="trained",
        allow_duplicate_first_by_path=allow_duplicate_first_by_path,
    )

    paired_task_ids = sorted(set(baseline_selected) & set(trained_selected))
    pairs = [
        LiftPair(
            task_id=task_id,
            baseline=baseline_selected[task_id],
            trained=trained_selected[task_id],
        )
        for task_id in paired_task_ids
    ]

    metrics = _metrics_for_pairs(
        pairs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_metadata = _by_metadata(
        pairs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": str(baseline_jobs_dir),
        "trained": str(trained_jobs_dir),
        "coverage": {
            "baseline": _coverage_summary(
                baseline_rollouts,
                duplicate_healthy_rollouts=baseline_duplicate_count,
            ),
            "trained": _coverage_summary(
                trained_rollouts,
                duplicate_healthy_rollouts=trained_duplicate_count,
            ),
        },
        "pairing": {
            "paired_tasks_count": len(pairs),
            "paired_tasks": paired_task_ids,
            "baseline_only_healthy_tasks": sorted(
                set(baseline_selected) - set(trained_selected)
            ),
            "trained_only_healthy_tasks": sorted(
                set(trained_selected) - set(baseline_selected)
            ),
        },
        "metrics": metrics,
        "by_metadata": by_metadata,
        "pairs": [_pair_payload(pair) for pair in pairs],
        "limitations": _limitations(
            baseline_duplicate_count=baseline_duplicate_count,
            trained_duplicate_count=trained_duplicate_count,
            has_metadata=bool(by_metadata),
        ),
    }


def write_lift_report(
    *,
    baseline_jobs_dir: Path,
    trained_jobs_dir: Path,
    markdown_path: Path,
    json_path: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    allow_duplicate_first_by_path: bool = False,
) -> dict[str, Any]:
    report = build_lift_report(
        baseline_jobs_dir,
        trained_jobs_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        allow_duplicate_first_by_path=allow_duplicate_first_by_path,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_lift_markdown(report))
    return report


def render_lift_markdown(report: Mapping[str, Any]) -> str:
    metrics = _as_mapping(report.get("metrics"))
    pairing = _as_mapping(report.get("pairing"))
    coverage = _as_mapping(report.get("coverage"))
    baseline_coverage = _as_mapping(coverage.get("baseline"))
    trained_coverage = _as_mapping(coverage.get("trained"))
    lines = [
        "# Paired Eval Lift Report",
        "",
        f"- Baseline: `{report.get('baseline')}`",
        f"- Trained: `{report.get('trained')}`",
        f"- Paired healthy tasks: {pairing.get('paired_tasks_count', 0)}",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Trained | Delta | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            "| Pass rate | "
            f"{_fmt_rate(metrics.get('pass_rate_base'))} | "
            f"{_fmt_rate(metrics.get('pass_rate_trained'))} | "
            f"{_fmt_signed_rate(metrics.get('pass_rate_delta'))} | "
            f"{_fmt_ci(_as_mapping(_as_mapping(metrics.get('ci')).get('pass_rate_delta')), rate=True)} |"
        ),
        (
            "| Mean reward | "
            f"{_fmt_float(metrics.get('mean_reward_base'))} | "
            f"{_fmt_float(metrics.get('mean_reward_trained'))} | "
            f"{_fmt_signed_float(metrics.get('mean_reward_delta'))} | "
            f"{_fmt_ci(_as_mapping(_as_mapping(metrics.get('ci')).get('mean_reward_delta')), rate=False)} |"
        ),
        "",
        "## Coverage",
        "",
        "| Side | Rollouts | Healthy rollouts | Error rollouts | Unscored rollouts | Unhealthy rollouts | Healthy tasks | Healthy task coverage | Duplicate healthy rollouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _coverage_row("Baseline", baseline_coverage),
        _coverage_row("Trained", trained_coverage),
        "",
        "## Pairing",
        "",
        f"- Baseline-only healthy tasks: {len(pairing.get('baseline_only_healthy_tasks', []))}",
        f"- Trained-only healthy tasks: {len(pairing.get('trained_only_healthy_tasks', []))}",
    ]
    by_metadata = _as_mapping(report.get("by_metadata"))
    if by_metadata:
        lines.extend(["", "## Metadata Breakdowns", ""])
        for field, raw_rows in by_metadata.items():
            rows = _as_mapping(raw_rows)
            lines.extend(
                [
                    f"### {field}",
                    "",
                    "| Value | N | Pass rate base | Pass rate trained | Pass delta | Mean reward delta |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for value, raw_metrics in rows.items():
                row_metrics = _as_mapping(raw_metrics)
                lines.append(
                    "| "
                    f"{value} | "
                    f"{row_metrics.get('paired_count', 0)} | "
                    f"{_fmt_rate(row_metrics.get('pass_rate_base'))} | "
                    f"{_fmt_rate(row_metrics.get('pass_rate_trained'))} | "
                    f"{_fmt_signed_rate(row_metrics.get('pass_rate_delta'))} | "
                    f"{_fmt_signed_float(row_metrics.get('mean_reward_delta'))} |"
                )
            lines.append("")
    pairs = report.get("pairs")
    if isinstance(pairs, list) and pairs:
        lines.extend(
            [
                "## Task Pairs",
                "",
                "| Task | Base reward | Trained reward | Reward delta | Base pass | Trained pass |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in pairs[:50]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                f"{row.get('task_id')} | "
                f"{_fmt_float(row.get('reward_base'))} | "
                f"{_fmt_float(row.get('reward_trained'))} | "
                f"{_fmt_signed_float(row.get('reward_delta'))} | "
                f"{_fmt_bool(row.get('passed_base'))} | "
                f"{_fmt_bool(row.get('passed_trained'))} |"
            )
        if len(pairs) > 50:
            lines.append(f"| ... {len(pairs) - 50} more task pairs omitted | | | | | |")
    limitations = report.get("limitations")
    if isinstance(limitations, list) and limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines).rstrip() + "\n"


def _load_rollouts(job_dir: Path) -> list[LiftRollout]:
    rollouts = []
    for rollout_dir in _iter_rollouts(job_dir):
        result = _read_json(rollout_dir / "result.json")
        if result is None:
            continue
        reward = _reward(result)
        rollouts.append(
            LiftRollout(
                task_id=_task_id(result, rollout_dir),
                rollout_dir=rollout_dir,
                reward=reward,
                metadata=_extract_metadata(result, rollout_dir),
                excluded_reason=_excluded_reason(result, reward),
            )
        )
    return sorted(rollouts, key=lambda row: (row.task_id, str(row.rollout_dir)))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _iter_rollouts(job_dir: Path) -> list[Path]:
    if (job_dir / "result.json").is_file():
        return [job_dir]
    roots = [job_dir]
    shard_root = _worker_shards_root(job_dir)
    if shard_root is not None:
        roots.append(shard_root)
    return sorted(
        {
            path.parent
            for root in roots
            if root.is_dir()
            for path in root.rglob("result.json")
        }
    )


def _worker_shards_root(job_dir: Path) -> Path | None:
    for candidate in (job_dir / "worker-shards", job_dir.parent / "worker-shards"):
        if (candidate / "plan.json").is_file():
            return candidate
    return None


def _reward(result: Mapping[str, Any]) -> float | None:
    rewards = result.get("rewards")
    if isinstance(rewards, Mapping):
        value = rewards.get("reward")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    value = result.get("reward")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _task_id(result: Mapping[str, Any], rollout_dir: Path) -> str:
    direct = result.get("task_name") or result.get("task_id")
    if direct:
        return str(direct)
    task = result.get("task")
    if isinstance(task, Mapping) and task.get("id"):
        return str(task["id"])
    return rollout_dir.name


def _extract_metadata(result: Mapping[str, Any], rollout_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    config = _read_json(rollout_dir / "config.json") or {}
    for source in (
        config.get("metadata"),
        result.get("metadata"),
        result.get("task_metadata"),
    ):
        if isinstance(source, Mapping):
            metadata.update(dict(source))
    for field in METADATA_GROUP_FIELDS:
        if field in result:
            metadata[field] = result[field]
    return metadata


def _excluded_reason(result: Mapping[str, Any], reward: float | None) -> str | None:
    if result.get("healthy") is False:
        return "unhealthy"
    if result.get("partial_trajectory") is True:
        return "partial_trajectory"
    if reward is not None:
        return None
    for field in _ERROR_FIELDS:
        if result.get(field):
            return field
    if reward is None:
        return "unscored"
    return None


def _select_one_healthy_per_task(
    rollouts: Sequence[LiftRollout],
    *,
    side: str,
    allow_duplicate_first_by_path: bool,
) -> tuple[dict[str, LiftRollout], int]:
    selected: dict[str, LiftRollout] = {}
    duplicate_paths: dict[str, list[Path]] = {}
    duplicates = 0
    for rollout in rollouts:
        if not rollout.healthy:
            continue
        if rollout.task_id in selected:
            duplicates += 1
            duplicate_paths.setdefault(
                rollout.task_id, [selected[rollout.task_id].rollout_dir]
            )
            duplicate_paths[rollout.task_id].append(rollout.rollout_dir)
            continue
        selected[rollout.task_id] = rollout
    if duplicate_paths and not allow_duplicate_first_by_path:
        raise ValueError(_duplicate_rollouts_message(side, duplicate_paths))
    return selected, duplicates


def _duplicate_rollouts_message(
    side: str, duplicate_paths: Mapping[str, list[Path]]
) -> str:
    task_summaries = []
    for task_id, paths in sorted(duplicate_paths.items()):
        joined_paths = ", ".join(str(path) for path in paths)
        task_summaries.append(f"{task_id}: {joined_paths}")
    return (
        f"Duplicate healthy rollouts found on {side} side. "
        "Canonicalize the job directory or pass --allow-duplicate-first-by-path "
        f"to keep the first rollout by path order. Duplicates: {'; '.join(task_summaries)}"
    )


def _coverage_summary(
    rollouts: Sequence[LiftRollout], *, duplicate_healthy_rollouts: int
) -> dict[str, Any]:
    reasons = Counter(row.excluded_reason for row in rollouts if row.excluded_reason)
    healthy = [row for row in rollouts if row.healthy]
    total_tasks = len({row.task_id for row in rollouts})
    healthy_tasks = len({row.task_id for row in healthy})
    error_rollouts = sum(reasons.get(field, 0) for field in _ERROR_FIELDS)
    unscored_rollouts = reasons.get("unscored", 0)
    unhealthy_rollouts = sum(reasons.get(field, 0) for field in _UNHEALTHY_REASONS)
    return {
        "total_rollouts": len(rollouts),
        "healthy_rollouts": len(healthy),
        "excluded_rollouts": sum(reasons.values()),
        "error_rollouts": error_rollouts,
        "unscored_rollouts": unscored_rollouts,
        "unhealthy_rollouts": unhealthy_rollouts,
        "excluded_reason_counts": dict(sorted(reasons.items())),
        "total_tasks": total_tasks,
        "healthy_tasks": healthy_tasks,
        "healthy_rollout_coverage": _ratio(len(healthy), len(rollouts)),
        "healthy_task_coverage": _ratio(healthy_tasks, total_tasks),
        "duplicate_healthy_rollouts": duplicate_healthy_rollouts,
    }


def _metrics_for_pairs(
    pairs: Sequence[LiftPair],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rewards_base = [_required_reward(pair.baseline) for pair in pairs]
    rewards_trained = [_required_reward(pair.trained) for pair in pairs]
    passes_base = [1.0 if pair.baseline.passed else 0.0 for pair in pairs]
    passes_trained = [1.0 if pair.trained.passed else 0.0 for pair in pairs]
    pass_deltas = [
        trained - base
        for base, trained in zip(passes_base, passes_trained, strict=True)
    ]
    reward_deltas = [
        float(trained) - float(base)
        for base, trained in zip(rewards_base, rewards_trained, strict=True)
    ]
    n = len(pairs)
    pass_rate_base = _mean(passes_base)
    pass_rate_trained = _mean(passes_trained)
    mean_reward_base = _mean_float_values(rewards_base)
    mean_reward_trained = _mean_float_values(rewards_trained)
    return {
        "paired_count": n,
        "pass_rate_base": pass_rate_base,
        "pass_rate_trained": pass_rate_trained,
        "pass_rate_delta": _delta(pass_rate_trained, pass_rate_base),
        "mean_reward_base": mean_reward_base,
        "mean_reward_trained": mean_reward_trained,
        "mean_reward_delta": _delta(mean_reward_trained, mean_reward_base),
        "ci": {
            "pass_rate_delta": _bootstrap_mean_ci(
                pass_deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "mean_reward_delta": _bootstrap_mean_ci(
                reward_deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
        },
    }


def _by_metadata(
    pairs: Sequence[LiftPair],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[LiftPair]]] = {}
    for pair in pairs:
        for field in METADATA_GROUP_FIELDS:
            value = pair.metadata.get(field)
            if isinstance(value, str | int | float | bool):
                grouped.setdefault(field, {}).setdefault(str(value), []).append(pair)
    return {
        field: {
            value: _metrics_for_pairs(
                rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for value, rows in sorted(values.items())
        }
        for field, values in sorted(grouped.items())
    }


def _pair_payload(pair: LiftPair) -> dict[str, Any]:
    base_reward = _required_reward(pair.baseline)
    trained_reward = _required_reward(pair.trained)
    return {
        "task_id": pair.task_id,
        "baseline_rollout_dir": str(pair.baseline.rollout_dir),
        "trained_rollout_dir": str(pair.trained.rollout_dir),
        "reward_base": base_reward,
        "reward_trained": trained_reward,
        "reward_delta": trained_reward - base_reward,
        "passed_base": pair.baseline.passed,
        "passed_trained": pair.trained.passed,
        "metadata": pair.metadata,
    }


def _limitations(
    *,
    baseline_duplicate_count: int,
    trained_duplicate_count: int,
    has_metadata: bool,
) -> list[str]:
    limitations = [
        "Only tasks with healthy, scored rollouts on both sides are included in paired lift metrics.",
        "Pass rate follows BenchFlow scoring: reward == 1.0 is pass; other numeric rewards are failures.",
    ]
    if baseline_duplicate_count or trained_duplicate_count:
        limitations.append(
            "Multiple healthy rollouts for the same task were explicitly allowed and de-duplicated by deterministic path order."
        )
    if not has_metadata:
        limitations.append(
            "No configured metadata breakdown fields were present in the paired result.json/config.json records."
        )
    return limitations


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean_float_values(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _required_reward(rollout: LiftRollout) -> float:
    if rollout.reward is None:
        raise ValueError(f"paired rollout is missing reward: {rollout.rollout_dir}")
    return float(rollout.reward)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not values:
        return None
    if len(values) == 1 or samples <= 0:
        value = float(values[0])
        return {
            "low": value,
            "high": value,
            "confidence": 0.95,
            "method": "paired_nonparametric_bootstrap",
            "samples": 0,
            "seed": seed,
        }
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return {
        "low": means[int(0.025 * (samples - 1))],
        "high": means[int(0.975 * (samples - 1))],
        "confidence": 0.95,
        "method": "paired_nonparametric_bootstrap",
        "samples": samples,
        "seed": seed,
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coverage_row(label: str, coverage: Mapping[str, Any]) -> str:
    return (
        f"| {label} | "
        f"{coverage.get('total_rollouts', 0)} | "
        f"{coverage.get('healthy_rollouts', 0)} | "
        f"{coverage.get('error_rollouts', 0)} | "
        f"{coverage.get('unscored_rollouts', 0)} | "
        f"{coverage.get('unhealthy_rollouts', 0)} | "
        f"{coverage.get('healthy_tasks', 0)} | "
        f"{_fmt_rate(coverage.get('healthy_task_coverage'))} | "
        f"{coverage.get('duplicate_healthy_rollouts', 0)} |"
    )


def _fmt_rate(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):.1%}"


def _fmt_signed_rate(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):+.1%}"


def _fmt_float(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):.3f}"


def _fmt_signed_float(value: Any) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{float(value):+.3f}"


def _fmt_bool(value: Any) -> str:
    return "yes" if value is True else "no" if value is False else "n/a"


def _fmt_ci(ci: Mapping[str, Any], *, rate: bool) -> str:
    low = ci.get("low")
    high = ci.get("high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        return "n/a"
    if rate:
        return f"[{float(low):+.1%}, {float(high):+.1%}]"
    return f"[{float(low):+.3f}, {float(high):+.3f}]"
