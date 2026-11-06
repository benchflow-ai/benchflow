"""A task edited since its result is run again on resume, never reused (DF-13).

Dogfood p2-task-author (2026-10-18): ``bench run tasks/my-task`` once, edit
the task (``[:3]`` to ``[:4]`` in the verifier, "3 most frequent" to "4" in
the prompt), ``bench run`` again without ``--fresh``: the job resumed and
printed the earlier 1/1, although the trial's config.json records
``task_digest`` and the digest had changed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

from benchflow._utils.result_paths import SET_ASIDE_DIR
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult


def _tasks(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "tasks"
    for name in ("task-0", "task-1"):
        (tasks_dir / name).mkdir(parents=True, exist_ok=True)
        (tasks_dir / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _job(tmp_path: Path, tasks_dir: Path, job_name: str | None) -> Evaluation:
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="oracle", concurrency=1, retry=RetryConfig(max_retries=0)
        ),
        job_name=job_name,
    )

    async def fake_run(task_path, _cfg):
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": rollout_dir.name,
                    "rewards": {"reward": 1.0},
                    "error": None,
                    "verifier_error": None,
                    "agent": "oracle",
                }
            )
        )
        # What a real trial records (Evaluation._run_single_task).
        (rollout_dir / "config.json").write_text(
            json.dumps({"agent": "oracle", "task_digest": task_digest(task_path)})
        )
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 1.0},
            agent="oracle",
            rollout_dir=rollout_dir,
        )

    job._run_single_task = AsyncMock(side_effect=fake_run)
    return job


def _edit(task: Path) -> tuple[str, str]:
    before = task_digest(task)
    (task / "instruction.md").write_text("Report the 4 most frequent words.\n")
    return before, task_digest(task)


def test_an_edited_task_runs_again_on_an_unnamed_resume(tmp_path, caplog) -> None:
    tasks_dir = _tasks(tmp_path)
    first = asyncio.run(_job(tmp_path, tasks_dir, None).run())
    assert first.ran == 2
    before, after = _edit(tasks_dir / "task-0")

    job = _job(tmp_path, tasks_dir, None)
    with caplog.at_level(logging.WARNING):
        second = asyncio.run(job.run())

    # The same job, with only the edited task run again.
    jobs = [p for p in (tmp_path / "jobs").iterdir() if p.is_dir()]
    assert jobs == [job.job_dir]
    assert (second.reused, second.ran) == (1, 1)
    ran = [call.args[0].name for call in job._run_single_task.call_args_list]
    assert ran == ["task-0"]
    assert (
        f"task 'task-0' at task_digest {before[:19]} (this run: {after[:19]}; "
        "the task's files changed since)"
    ) in caplog.text
    set_aside = list((job.job_dir / SET_ASIDE_DIR).iterdir())
    assert len(set_aside) == 1
    assert (set_aside[0] / "task-0__abcd1234").is_dir()


def test_an_edited_task_runs_again_on_a_named_resume(tmp_path) -> None:
    tasks_dir = _tasks(tmp_path)
    asyncio.run(_job(tmp_path, tasks_dir, "named").run())
    _edit(tasks_dir / "task-1")

    job = _job(tmp_path, tasks_dir, "named")
    second = asyncio.run(job.run())

    assert (second.reused, second.ran) == (1, 1)
    ran = [call.args[0].name for call in job._run_single_task.call_args_list]
    assert ran == ["task-1"]


def test_an_unchanged_task_is_still_reused(tmp_path) -> None:
    tasks_dir = _tasks(tmp_path)
    asyncio.run(_job(tmp_path, tasks_dir, "named").run())

    job = _job(tmp_path, tasks_dir, "named")
    second = asyncio.run(job.run())

    assert (second.reused, second.ran) == (2, 0)
    assert not (job.job_dir / SET_ASIDE_DIR).exists()
