"""Summarize saved ASI-Bench jobs using the public local scoring contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NoReturn


class RunnerNotIntegratedError(RuntimeError):
    """Raised for the unused standalone execution command."""


def run(tasks_dir: Path) -> NoReturn:
    """Direct callers to BenchFlow's native execution command."""
    raise RunnerNotIntegratedError(
        "Run materialized ASI-Bench tasks with `bench eval run --tasks-dir "
        f"{tasks_dir}`; this adapter provides conversion and summarization."
    )


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _score_from_reward(path: Path) -> tuple[str, str, float, str, str]:
    data = _read_object(path)
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing ASI-Bench metadata in {path}")
    if (
        metadata.get("benchmark") != "ASI-Bench"
        or metadata.get("seed") != 31415
        or metadata.get("official") is not False
    ):
        raise ValueError(f"Expected non-official ASI-Bench seed31415 reward: {path}")
    task_id = metadata.get("task_id")
    instance_id = metadata.get("instance_id")
    level = metadata.get("prompt_level")
    if (
        not isinstance(task_id, str)
        or not task_id
        or instance_id != f"{task_id}__seed31415"
        or not isinstance(level, str)
        or level not in {"B1", "B2", "B3", "B4"}
    ):
        raise ValueError(f"Invalid task, instance, or prompt level in {path}")
    asi_revision = metadata.get("asi_revision")
    hf_revision = metadata.get("hf_revision")
    if not isinstance(asi_revision, str) or not isinstance(hf_revision, str):
        raise ValueError(f"Missing source revisions in {path}")
    values = (
        data.get("reward"),
        metadata.get("final_score"),
        metadata.get("max_score"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"Missing or non-finite score in {path}")
    reward, final_score, max_score = values
    if (
        max_score <= 0
        or not 0 <= reward <= 1
        or not math.isclose(reward, final_score / max_score, abs_tol=1e-9)
    ):
        raise ValueError(f"Reward disagrees with ASI-Bench score in {path}")
    return task_id, level, reward * 100, asi_revision, hf_revision


def summarize(jobs_dir: Path) -> dict:
    """Aggregate verified rewards; never turn missing scores into zeroes."""
    if not jobs_dir.is_dir():
        raise ValueError(f"Jobs directory does not exist: {jobs_dir}")
    jobs = sorted(path for path in jobs_dir.iterdir() if path.is_dir())
    if not jobs:
        raise ValueError(f"No jobs found in {jobs_dir}")

    by_task: dict[str, list[tuple[str, float]]] = {}
    by_level: dict[str, list[float]] = {}
    scores: list[float] = []
    revisions: set[tuple[str, str]] = set()
    scorer_error_count = 0
    unscored_count = 0
    for job in jobs:
        verifier = job / "verifier"
        reward_file = verifier / "reward.json"
        error_file = verifier / "asi_error.json"
        result_file = job / "result.json"
        result = _read_object(result_file) if result_file.is_file() else None
        verifier_error = result.get("verifier_error") if result else None
        if reward_file.exists() and (error_file.exists() or verifier_error):
            raise ValueError(f"Job has both reward and evaluator error: {job}")
        if reward_file.exists():
            if result is None:
                raise ValueError(f"Reward has no persisted result.json: {job}")
            task_id, level, score, asi_revision, hf_revision = _score_from_reward(
                reward_file
            )
            revisions.add((asi_revision, hf_revision))
            by_task.setdefault(task_id, []).append((level, score))
            by_level.setdefault(level, []).append(score)
            scores.append(score)
        elif error_file.exists() or verifier_error:
            if (
                error_file.exists()
                and _read_object(error_file).get("scorer_internal_error") is not True
            ):
                raise ValueError(
                    f"Evaluator error lacks scorer_internal_error: {error_file}"
                )
            scorer_error_count += 1
        else:
            unscored_count += 1

    if len(revisions) > 1:
        raise ValueError("Cannot aggregate jobs from different ASI/HF source revisions")

    task_summaries = []
    for task_id, task_scores in sorted(by_task.items()):
        level_scores: dict[str, list[float]] = {}
        for level, score in task_scores:
            level_scores.setdefault(level, []).append(score)
        task_summaries.append(
            {
                "task_id": task_id,
                "scored_instances": len(task_scores),
                "mean_score": sum(score for _, score in task_scores) / len(task_scores),
                "scores_by_level": {
                    level: sum(values) / len(values)
                    for level, values in sorted(level_scores.items())
                },
            }
        )
    asi_revision, hf_revision = next(iter(revisions)) if revisions else (None, None)
    return {
        "schema_version": 1,
        "scope": "seed31415 local, non-official",
        "seed": 31415,
        "official": False,
        "asi_revision": asi_revision,
        "hf_revision": hf_revision,
        "job_count": len(jobs),
        "scored_instance_count": len(scores),
        "scorer_error_count": scorer_error_count,
        "unscored_count": unscored_count,
        "overall_mean_score": sum(scores) / len(scores) if scores else None,
        "by_task": task_summaries,
        "by_prompt_level": {
            level: sum(values) / len(values)
            for level, values in sorted(by_level.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize ASI-Bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--tasks-dir", type=Path, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("jobs_dir", type=Path)

    args = parser.parse_args()
    if args.command == "run":
        run(args.tasks_dir)
    else:
        print(json.dumps(summarize(args.jobs_dir), indent=2))


if __name__ == "__main__":
    main()
