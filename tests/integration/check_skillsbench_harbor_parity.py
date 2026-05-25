#!/usr/bin/env python3
"""Compare BenchFlow SkillsBench results against a pinned Harbor baseline.

The Harbor baseline is intentionally supplied as a local path. The public
``benchflow-ai/skillsbench-trajectories`` repository is large, so this checker
does not clone or refresh it as a side effect of a release evidence run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import classify_result

DEFAULT_HARBOR_BASELINE_REF = "2d86fe82f6a06f7c7b3a22a3ae90d554d0e9655c"
PIN_FILE = ".benchflow-harbor-baseline-ref"
BENCHFLOW_REQUIRED = {"task_name", "rewards", "error", "verifier_error"}
HARBOR_REQUIRED = {"task_name", "config", "agent_info", "verifier_result"}
OUTCOMES = ("passed", "failed", "errored", "verifier_errored")


@dataclass(frozen=True)
class NormalizedTrajectory:
    path: Path
    format: str
    steps: int
    tool_calls: int


@dataclass(frozen=True)
class NormalizedResult:
    source: str
    path: Path
    task_name: str
    reward: float | None
    outcome: str
    agent: str | None
    model: str | None
    environment: str | None
    trajectory: NormalizedTrajectory | None


@dataclass(frozen=True)
class ResultSet:
    source: str
    results: list[NormalizedResult]
    issues: list[str]


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _jsonl_records(path: Path) -> list[Any]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _trajectory_candidates(result_path: Path) -> list[Path]:
    parent = result_path.parent
    candidates = [
        parent / "trajectory" / "acp_trajectory.jsonl",
        parent / "agent" / "trajectory.json",
        parent / "trajectory.json",
    ]
    agent_dir = parent / "agent"
    if agent_dir.exists():
        candidates.extend(sorted(agent_dir.glob("*.trajectory.json")))
    return candidates


def _load_trajectory(result_path: Path) -> NormalizedTrajectory | None:
    path = next(
        (
            candidate
            for candidate in _trajectory_candidates(result_path)
            if candidate.exists()
        ),
        None,
    )
    if path is None:
        return None

    if path.suffix == ".jsonl":
        records = _jsonl_records(path)
        tool_calls = sum(
            1
            for record in records
            if isinstance(record, dict) and record.get("type") == "tool_call"
        )
        return NormalizedTrajectory(path, "acp-jsonl", len(records), tool_calls)

    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"trajectory JSON is not an object: {path}")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"trajectory JSON has no steps list: {path}")
    tool_calls = 0
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("tool_calls"), list):
            tool_calls += len(step["tool_calls"])
    schema = _string(data.get("schema_version")) or "json"
    return NormalizedTrajectory(path, schema, len(steps), tool_calls)


def _find_result_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("result.json") if path.is_file())


def _task_name_from_path(path: Path) -> str:
    return path.parent.name.rsplit("__", 1)[0]


def _path_may_match_task(path: Path, tasks: set[str]) -> bool:
    path_task = _task_name_from_path(path)
    return any(task == path_task or task.startswith(path_task) for task in tasks)


def _normalize_benchflow_result(path: Path, data: dict[str, Any]) -> NormalizedResult:
    missing = BENCHFLOW_REQUIRED - set(data)
    if missing:
        raise ValueError(f"missing BenchFlow field(s): {sorted(missing)}")
    task_name = _string(data.get("task_name"))
    if task_name is None:
        raise ValueError("BenchFlow task_name must be a non-empty string")
    rewards = data.get("rewards")
    reward = _number(rewards.get("reward")) if isinstance(rewards, dict) else None
    error = _string(data.get("error"))
    verifier_error = _string(data.get("verifier_error"))
    return NormalizedResult(
        source="benchflow",
        path=path,
        task_name=task_name,
        reward=reward,
        outcome=classify_result(
            reward=reward,
            error=error,
            verifier_error=verifier_error,
        ),
        agent=_string(data.get("agent")),
        model=_string(data.get("model")),
        environment=None,
        trajectory=_load_trajectory(path),
    )


def _harbor_agent_name(agent_info: Any, config_agent: Any) -> str | None:
    if isinstance(agent_info, dict):
        return _string(agent_info.get("name"))
    if isinstance(config_agent, dict):
        return _string(config_agent.get("name"))
    return None


def _harbor_model_name(agent_info: Any, config_agent: Any) -> str | None:
    model_info = agent_info.get("model_info") if isinstance(agent_info, dict) else None
    if isinstance(model_info, dict):
        return _string(model_info.get("name"))
    if isinstance(config_agent, dict):
        return _string(config_agent.get("model_name"))
    return None


def _normalize_harbor_result(path: Path, data: dict[str, Any]) -> NormalizedResult:
    missing = HARBOR_REQUIRED - set(data)
    if missing:
        raise ValueError(f"missing Harbor field(s): {sorted(missing)}")
    task_name = _string(data.get("task_name"))
    if task_name is None:
        raise ValueError("Harbor task_name must be a non-empty string")

    verifier_result = data.get("verifier_result")
    rewards = (
        verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    )
    reward = _number(rewards.get("reward")) if isinstance(rewards, dict) else None
    exception_info = data.get("exception_info")
    error = json.dumps(exception_info, sort_keys=True) if exception_info else None

    agent_info = data.get("agent_info")
    config = data.get("config")
    config_agent = config.get("agent") if isinstance(config, dict) else None
    config_env = config.get("environment") if isinstance(config, dict) else None

    return NormalizedResult(
        source="harbor",
        path=path,
        task_name=task_name,
        reward=reward,
        outcome=classify_result(
            reward=reward,
            error=error,
            verifier_error=None,
        ),
        agent=_harbor_agent_name(agent_info, config_agent),
        model=_harbor_model_name(agent_info, config_agent),
        environment=(
            _string(config_env.get("type")) if isinstance(config_env, dict) else None
        ),
        trajectory=_load_trajectory(path),
    )


def load_result_set(root: Path, *, source: str, tasks: set[str] | None) -> ResultSet:
    normalizer = (
        _normalize_benchflow_result
        if source == "benchflow"
        else _normalize_harbor_result
    )
    results: list[NormalizedResult] = []
    issues: list[str] = []
    for path in _find_result_files(root):
        if tasks is not None and not _path_may_match_task(path, tasks):
            continue
        try:
            data = _load_json(path)
            if not isinstance(data, dict):
                raise ValueError("result.json is not a JSON object")
            result = normalizer(path, data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"{source}: {path}: {exc}")
            continue
        if tasks is None or result.task_name in tasks:
            results.append(result)
    if not results:
        issues.append(f"{source}: no matching result.json files under {root}")
    return ResultSet(source, results, issues)


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _baseline_pin_issues(root: Path, expected_ref: str) -> list[str]:
    if not expected_ref:
        return []
    head = _git_head(root)
    if head is not None:
        if head != expected_ref:
            return [
                "harbor: baseline git HEAD "
                f"{head} does not match pinned ref {expected_ref}"
            ]
        return []

    pin_path = root / PIN_FILE
    if not pin_path.exists():
        return [
            "harbor: baseline root is not a git checkout and has no "
            f"{PIN_FILE} pin file"
        ]
    actual = pin_path.read_text().strip()
    if actual != expected_ref:
        return [
            f"harbor: {PIN_FILE} has {actual}, expected pinned ref {expected_ref}"
        ]
    return []


def _by_task(results: list[NormalizedResult]) -> dict[str, list[NormalizedResult]]:
    grouped: dict[str, list[NormalizedResult]] = {}
    for result in results:
        grouped.setdefault(result.task_name, []).append(result)
    return grouped


def _distribution(results: list[NormalizedResult]) -> dict[str, Any]:
    total = len(results)
    counts = Counter(result.outcome for result in results)
    rewards = [result.reward if result.reward is not None else 0.0 for result in results]
    mean_reward = sum(rewards) / total if total else 0.0
    return {
        "total": total,
        "counts": {outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
        "rates": {
            outcome: counts.get(outcome, 0) / total if total else 0.0
            for outcome in OUTCOMES
        },
        "mean_reward": mean_reward,
    }


def _format_distribution(label: str, distribution: dict[str, Any]) -> str:
    counts = distribution["counts"]
    return (
        f"{label}: total={distribution['total']} "
        f"passed={counts['passed']} failed={counts['failed']} "
        f"errored={counts['errored']} "
        f"verifier_errored={counts['verifier_errored']} "
        f"mean_reward={distribution['mean_reward']:.3f}"
    )


def compare_result_sets(
    benchflow: ResultSet,
    harbor: ResultSet,
    *,
    tasks: set[str] | None,
    require_trajectories: bool,
    max_outcome_rate_delta: float,
    max_mean_reward_delta: float,
    max_task_reward_delta: float,
) -> list[str]:
    issues = [*benchflow.issues, *harbor.issues]
    benchflow_by_task = _by_task(benchflow.results)
    harbor_by_task = _by_task(harbor.results)
    selected_tasks = sorted(tasks or benchflow_by_task)

    for task in selected_tasks:
        if task not in benchflow_by_task:
            issues.append(f"benchflow: missing task result for {task}")
        if task not in harbor_by_task:
            issues.append(f"harbor: missing baseline result for {task}")

    comparable_tasks = [
        task
        for task in selected_tasks
        if task in benchflow_by_task and task in harbor_by_task
    ]
    if not comparable_tasks:
        issues.append("no overlapping SkillsBench tasks to compare")
        return issues

    for task in comparable_tasks:
        benchflow_task_results = benchflow_by_task[task]
        harbor_task_results = harbor_by_task[task]

        if require_trajectories:
            for result in [*benchflow_task_results, *harbor_task_results]:
                if result.trajectory is None:
                    issues.append(
                        f"{result.source}: {task}: missing trajectory for "
                        f"{result.path}"
                    )

        harbor_outcomes = {result.outcome for result in harbor_task_results}
        for result in benchflow_task_results:
            if result.outcome not in harbor_outcomes:
                issues.append(
                    f"benchflow: {task}: outcome {result.outcome!r} is not present "
                    f"in Harbor baseline outcomes {sorted(harbor_outcomes)}"
                )

        harbor_rewards = [
            result.reward for result in harbor_task_results if result.reward is not None
        ]
        if harbor_rewards:
            lower = min(harbor_rewards) - max_task_reward_delta
            upper = max(harbor_rewards) + max_task_reward_delta
            for result in benchflow_task_results:
                if result.reward is not None and not lower <= result.reward <= upper:
                    issues.append(
                        f"benchflow: {task}: reward {result.reward:g} outside "
                        f"Harbor baseline range {min(harbor_rewards):g}..{max(harbor_rewards):g}"
                    )

    benchflow_compared = [
        result for task in comparable_tasks for result in benchflow_by_task[task]
    ]
    harbor_compared = [
        result for task in comparable_tasks for result in harbor_by_task[task]
    ]
    benchflow_distribution = _distribution(benchflow_compared)
    harbor_distribution = _distribution(harbor_compared)

    for outcome in OUTCOMES:
        delta = abs(
            benchflow_distribution["rates"][outcome]
            - harbor_distribution["rates"][outcome]
        )
        if delta > max_outcome_rate_delta:
            issues.append(
                f"outcome rate drift for {outcome}: {delta:.3f} "
                f"> {max_outcome_rate_delta:.3f}"
            )

    mean_delta = abs(
        benchflow_distribution["mean_reward"] - harbor_distribution["mean_reward"]
    )
    if mean_delta > max_mean_reward_delta:
        issues.append(
            f"mean reward drift: {mean_delta:.3f} > {max_mean_reward_delta:.3f}"
        )

    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check SkillsBench result parity against a pinned Harbor baseline."
    )
    parser.add_argument("--benchflow-root", type=Path, required=True)
    parser.add_argument("--harbor-baseline-root", type=Path, required=True)
    parser.add_argument(
        "--harbor-baseline-ref",
        default=DEFAULT_HARBOR_BASELINE_REF,
        help="Pinned benchflow-ai/skillsbench-trajectories ref.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="SkillsBench task name to compare. May be repeated.",
    )
    parser.add_argument(
        "--no-require-trajectories",
        action="store_true",
        help="Do not require trajectory artifacts for compared results.",
    )
    parser.add_argument(
        "--max-outcome-rate-delta",
        type=float,
        default=0.25,
        help="Allowed absolute delta per normalized outcome rate.",
    )
    parser.add_argument(
        "--max-mean-reward-delta",
        type=float,
        default=0.25,
        help="Allowed absolute delta in mean reward over compared rows.",
    )
    parser.add_argument(
        "--max-task-reward-delta",
        type=float,
        default=0.0,
        help="Allowed per-task reward movement outside Harbor observed range.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = set(args.task) if args.task else None
    benchflow_root = args.benchflow_root.resolve()
    harbor_root = args.harbor_baseline_root.resolve()

    pin_issues = _baseline_pin_issues(harbor_root, args.harbor_baseline_ref)
    benchflow = load_result_set(benchflow_root, source="benchflow", tasks=tasks)
    effective_tasks = tasks or {result.task_name for result in benchflow.results}
    harbor = load_result_set(harbor_root, source="harbor", tasks=effective_tasks)
    issues = [
        *pin_issues,
        *compare_result_sets(
            benchflow,
            harbor,
            tasks=effective_tasks,
            require_trajectories=not args.no_require_trajectories,
            max_outcome_rate_delta=args.max_outcome_rate_delta,
            max_mean_reward_delta=args.max_mean_reward_delta,
            max_task_reward_delta=args.max_task_reward_delta,
        ),
    ]

    print("SkillsBench Harbor parity")
    print("-" * 80)
    print(f"BenchFlow root: {benchflow_root}")
    print(f"Harbor baseline root: {harbor_root}")
    print(f"Harbor baseline ref: {args.harbor_baseline_ref}")
    print(
        "Tasks: "
        f"{', '.join(sorted(effective_tasks)) if effective_tasks else '(none)'}"
    )

    if benchflow.results:
        print(_format_distribution("BenchFlow", _distribution(benchflow.results)))
    if harbor.results:
        print(_format_distribution("Harbor", _distribution(harbor.results)))

    if issues:
        print("FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
