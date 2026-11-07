"""``bench eval metrics jobs/`` must not let one agent's pass hide another's fail.

``collect_metrics`` keeps the best result per task so a retried task counts
once. The key was the task name alone, so over a jobs folder that held an
``oracle`` job and a ``nop`` control job on the same tasks, every ``nop``
failure was replaced by the oracle's pass, and the table reported every task
passed. ``bf.load_job`` already keeps one trial per task, agent and model.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.metrics import collect_metrics


def _trial(job: Path, name: str, task: str, agent: str, reward: float) -> None:
    trial = job / name
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "agent_name": agent,
                "model": None,
                "rewards": {"reward": reward},
                "error": None,
                "verifier_error": None,
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:00:10",
            }
        )
    )


def test_two_agents_on_the_same_tasks_are_both_counted(tmp_path: Path):
    jobs = tmp_path / "jobs"
    for task in ("alpha", "beta"):
        _trial(jobs / "oracle-run", f"{task}__o", task, "oracle", 1.0)
        _trial(jobs / "nop-run", f"{task}__n", task, "nop", 0.0)

    metrics = collect_metrics(jobs)

    assert metrics.total == 4
    assert metrics.passed == 2
    assert metrics.failed == 2


def test_a_retried_task_of_one_agent_still_counts_once(tmp_path: Path):
    jobs = tmp_path / "jobs"
    _trial(jobs / "run", "alpha__1", "alpha", "oracle", 0.0)
    _trial(jobs / "run", "alpha__2", "alpha", "oracle", 1.0)
    # Retries happen inside an Evaluation job, which records evaluation.json.
    (jobs / "run" / "evaluation.json").write_text("{}")

    metrics = collect_metrics(jobs)

    assert (metrics.total, metrics.passed) == (1, 1)
