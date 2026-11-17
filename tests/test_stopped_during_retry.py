"""A run stopped during a retry leaves its task interrupted, with its spend.

Guards the fix made on branch ``audit-final-attempt-2-selector`` after
the independent review of PR #138 (2026-11-13). With the attempt records, an
attempt the run retried is never the trial's result. A run stopped (Ctrl-C,
SIGTERM, a crash) after it retried an attempt and before it kept one left a
task whose every finished attempt was retried, and the readers dropped it:
``summary.json`` said ``not_started`` for a task that had run and spent,
``bf.load_job`` had no trial and no cost for it, and the report lost it.

A task with any attempt folder is interrupted, never not started, and the
attempts the run retried stay reachable (``Job.retried``) and in the cost line.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow as bf
from benchflow._utils.attempt_decision import RETRIED, read_decision, record_decision
from benchflow._utils.result_paths import load_task_results
from benchflow.evaluation import Evaluation, RetryConfig
from benchflow.models import RolloutResult
from tests.test_dogfood_rc_results import _config, _invoke, _write_task

IDLE = "Agent idle for 600s (no output); killed"


def _retried_attempt(job_dir: Path, task: str, name: str) -> dict:
    """An attempt that went idle with a reward and a spend, as its folder."""
    trial = job_dir / name
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"agent": "claude-agent-acp", "task_path": f"/tasks/{task}"})
    )
    payload = {
        "task_name": task,
        "rollout_name": name,
        "agent": "claude-agent-acp",
        "model": "m",
        "rewards": {"reward": 1.0},
        "error": IDLE,
        "cost_usd": 0.5,
        "started_at": "2026-11-12T10:00:00",
        "finished_at": "2026-11-12T10:20:00",
    }
    (trial / "result.json").write_text(json.dumps(payload))
    return payload


def _stopped_during_a_retry(tmp_path: Path) -> Evaluation:
    """t0 finishes, t1's first attempt is retried and the job is stopped while
    the retry waits (before it has a folder), t2 never starts."""
    tasks = tmp_path / "tasks"
    for name in ("t0", "t1", "t2"):
        _write_task(tasks / name)
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=_config(concurrency=2, retry=RetryConfig(max_retries=2, min_wait_sec=0)),
    )
    retrying = asyncio.Event()
    calls: dict[str, int] = {}

    async def fake_run(task_path, _cfg):
        name = task_path.name
        calls[name] = calls.get(name, 0) + 1
        if name == "t2":
            await asyncio.Event().wait()  # waits for a slot: never starts
        if name == "t1" and calls[name] == 1:
            payload = _retried_attempt(job.job_dir, "t1", "t1__aaaa0001")
            return RolloutResult(
                task_name="t1",
                rollout_name="t1__aaaa0001",
                error=IDLE,
                rewards=payload["rewards"],
                rollout_dir=job.job_dir / "t1__aaaa0001",
            )
        if name == "t1":
            retrying.set()
            await asyncio.Event().wait()  # the retry: runs until the job stops
        trial = job.job_dir / "t0__abcd1234"
        trial.mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps({"agent": "claude-agent-acp", "task_path": str(task_path)})
        )
        payload = {
            "task_name": "t0",
            "rollout_name": trial.name,
            "agent": "claude-agent-acp",
            "model": "m",
            "rewards": {"reward": 1.0},
        }
        (trial / "result.json").write_text(json.dumps(payload))
        return RolloutResult(task_name="t0", rewards={"reward": 1.0}, rollout_dir=trial)

    job._run_single_task = AsyncMock(side_effect=fake_run)

    async def no_sweep():
        return None

    job._sweep_docker = no_sweep

    async def main():
        runner = asyncio.create_task(job.run())
        await retrying.wait()
        while not (job.job_dir / "t0__abcd1234" / "result.json").is_file():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(main())
    return job


def test_the_run_summary_says_the_retried_task_is_interrupted(tmp_path):
    job = _stopped_during_a_retry(tmp_path)

    assert read_decision(job.job_dir / "t1__aaaa0001")["decision"] == RETRIED
    summary = json.loads((job.job_dir / "summary.json").read_text())
    assert summary["stopped"] is True
    assert summary["interrupted_tasks"] == ["t1"]
    assert (summary["interrupted"], summary["not_started"]) == (1, 1)
    assert (summary["planned"], summary["ran"], summary["not_run"]) == (3, 1, 2)


def test_every_reader_keeps_the_task_and_its_spend(tmp_path):
    job_dir = _stopped_during_a_retry(tmp_path).job_dir

    job = bf.load_job(job_dir)

    assert [t.task_name for t in job.trials] == ["t0"]
    assert job.unfinished() == (["t1"], ["t2"])
    assert [(t.task_name, t.cost_usd) for t in job.retried] == [("t1", 0.5)]
    assert "Cost: $0.5000 over 1 rollout(s)" in job.to_markdown()
    # The export names the tasks unfinished() does, by their attempt folders.
    exported = job.to_json_dict()["interrupted"]
    assert [(e["task_name"], Path(e["path"]).name) for e in exported] == [
        ("t1", "t1__aaaa0001")
    ]
    # A resume runs it again: no reader counts the retried attempt's reward.
    assert set(load_task_results(job_dir)) == {"t0"}
    metrics = _invoke("eval", "metrics", str(job_dir))
    assert "3 tasks, 1 ran, 1 interrupted, 1 not started" in metrics.output, (
        metrics.output
    )
    from benchflow.report import _trials

    rows, missing, _ = _trials([job_dir])
    assert [row["task"] for row in rows] == ["t0"]
    assert missing == ["t1", "t2"]


def test_a_job_whose_only_attempt_was_retried_still_loads(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": ["t1"], "config": {}})
    )
    _retried_attempt(job_dir, "t1", "t1__aaaa0001")
    record_decision(job_dir / "t1__aaaa0001", RETRIED, task="t1", attempt=1)

    job = bf.load_job(job_dir)

    assert job.trials == []
    assert job.unfinished() == (["t1"], [])
    assert [t.path.name for t in job.retried] == ["t1__aaaa0001"]
    assert "Cost: $0.5000 over 1 rollout(s)" in job.to_markdown()
    # Every attempt as a trial (re-review of PR #138): the attempt is a
    # trial, its task is still interrupted, and its spend is counted once.
    every = bf.load_job(job_dir, attempts="all")
    assert [t.path.name for t in every.trials] == ["t1__aaaa0001"]
    assert every.unfinished() == (["t1"], [])
    assert "Cost: $0.5000 over 1 rollout(s)" in every.to_markdown()


def test_a_job_whose_only_result_cannot_be_read_still_loads(tmp_path):
    """Re-review of PR #138: such a job raised "no trial ... under"; its task
    has an attempt folder, so it is interrupted, never not started."""
    job_dir = tmp_path / "job"
    trial = job_dir / "t1__aaaa0001"
    trial.mkdir(parents=True)
    (job_dir / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": ["t1", "t2"], "config": {}})
    )
    (trial / "config.json").write_text(
        json.dumps({"agent": "claude-agent-acp", "task_path": "/tasks/t1"})
    )
    (trial / "result.json").write_text('{"task_name": "t1", "rewa')

    job = bf.load_job(job_dir)

    assert job.trials == [] and job.unreadable == [trial]
    assert job.unfinished() == (["t1"], ["t2"])
    assert [e["task_name"] for e in job.to_json_dict()["interrupted"]] == ["t1"]


def test_the_exports_interrupted_list_drops_a_task_a_resume_finished(tmp_path):
    """The lead's ruling on PR #138: the export's ``interrupted`` list is
    equal to ``Job.unfinished()``. A finished task never reads as interrupted."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": ["t1", "t2"], "config": {}})
    )
    for name, task in (("t1__cut00001", "t1"), ("t2__cut00002", "t2")):
        cut = job_dir / name  # started, no result: a kill
        cut.mkdir()
        (cut / "config.json").write_text(
            json.dumps({"agent": "claude-agent-acp", "task_path": f"/tasks/{task}"})
        )
    done = job_dir / "t1__done0003"  # a resume ran t1 again, and it finished
    done.mkdir()
    (done / "config.json").write_text(
        json.dumps({"agent": "claude-agent-acp", "task_path": "/tasks/t1"})
    )
    (done / "result.json").write_text(
        json.dumps(
            {
                "task_name": "t1",
                "rollout_name": done.name,
                "agent": "claude-agent-acp",
                "model": "m",
                "rewards": {"reward": 1.0},
            }
        )
    )

    job = bf.load_job(job_dir)

    assert job.unfinished() == (["t2"], [])
    exported = job.to_json_dict()["interrupted"]
    assert [(e["task_name"], Path(e["path"]).name) for e in exported] == [
        ("t2", "t2__cut00002")
    ]
