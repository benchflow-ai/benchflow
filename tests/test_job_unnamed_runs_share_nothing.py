"""Guards the fix against the job lock (_utils/job_lock.py) refusing unnamed runs.

Per-task runners launched together into one jobs dir without a job name (FP's
campaigns) each picked the latest job, or the same new timestamp, and every
run but the first died with ``JobLockedError``; 0.7.11 had no lock and ran
them all. An unnamed run now takes a new job of its own when its job is
held, two runs never share one, and a named job is still refused.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import benchflow.evaluation as evaluation_module
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult

TASKS = ("qho", "sho", "lho")


def _tasks(root: Path) -> Path:
    tasks_dir = root / "tasks"
    for name in TASKS:
        (tasks_dir / name).mkdir(parents=True, exist_ok=True)
        (tasks_dir / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _evaluation(root: Path, task: str, job_name: str | None = None) -> Evaluation:
    config = EvaluationConfig(
        agent="oracle",
        concurrency=1,
        retry=RetryConfig(max_retries=0),
        include_tasks={task},
    )
    return Evaluation(
        tasks_dir=_tasks(root),
        jobs_dir=root / "jobs",
        config=config,
        job_name=job_name,
        preflight=False,
    )


def _take(evaluation: Evaluation):
    """run()'s steps before any sandbox: settle the job name, take its lock."""
    evaluation._settle_auto_job_name()
    return evaluation._acquire_job_lock()


def test_runs_started_in_the_same_second_get_their_own_jobs(tmp_path, caplog):
    first, second = _evaluation(tmp_path, "qho"), _evaluation(tmp_path, "sho")
    # Both picked the same new timestamp, as runners started in one second do.
    stamp = "2026-10-26__00-50-58"
    for run in (first, second):
        run._job_name = run._auto_job_name = stamp
    held = _take(first)
    try:
        with caplog.at_level("WARNING"):
            other = _take(second)
        try:
            assert first._job_name == stamp
            assert second._job_name != stamp
            assert second._auto_job_name == second._job_name
            assert (tmp_path / "jobs" / second._job_name).is_dir()
            assert f"Starting a new job {second._job_name}" in caplog.text
        finally:
            other.release()
    finally:
        held.release()


def test_a_runner_launched_beside_a_live_sibling_starts_a_new_job(tmp_path):
    sibling = _evaluation(tmp_path, "lho", job_name="2026-10-20__12-00-00")
    held = _take(sibling)
    try:
        late = _evaluation(tmp_path, "sho")
        # The sibling's job is the latest one and has no result yet.
        assert late._job_name == "2026-10-20__12-00-00"
        lock = _take(late)
        try:
            assert late._job_name not in ("2026-10-20__12-00-00", None)
        finally:
            lock.release()
    finally:
        held.release()


def test_a_named_job_is_still_refused_while_it_runs(tmp_path):
    holder = _evaluation(tmp_path, "lho", job_name="pass-1")
    held = _take(holder)
    try:
        again = _evaluation(tmp_path, "sho", job_name="pass-1")
        with pytest.raises(RuntimeError, match="already running"):
            _take(again)
        assert again._job_name == "pass-1"
    finally:
        held.release()


def test_new_job_names_are_reserved_by_their_folders(tmp_path, monkeypatch):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 26, 0, 50, 58)

    monkeypatch.setattr(evaluation_module, "datetime", _Clock)
    jobs = tmp_path / "jobs"
    names = [evaluation_module._reserve_new_job(jobs) for _ in range(3)]
    assert names == [
        "2026-10-26__00-50-58",
        "2026-10-26__00-50-58-2",
        "2026-10-26__00-50-58-3",
    ]
    assert all((jobs / name).is_dir() for name in names)


def _runner(root, task, start, names, finish):
    start.wait(30)
    evaluation = _evaluation(Path(root), task)
    lock = _take(evaluation)
    names.put(evaluation._job_name)
    finish.wait(30)
    lock.release()


def test_runners_started_together_never_share_or_die(tmp_path):
    ctx = multiprocessing.get_context("fork")
    count = 8
    start, finish = ctx.Barrier(count + 1), ctx.Event()
    names = ctx.Queue()
    _tasks(tmp_path)
    processes = [
        ctx.Process(
            target=_runner,
            args=(str(tmp_path), TASKS[i % len(TASKS)], start, names, finish),
        )
        for i in range(count)
    ]
    for process in processes:
        process.start()
    try:
        start.wait(30)
        taken = [names.get(timeout=60) for _ in range(count)]
    finally:
        finish.set()
        for process in processes:
            process.join(30)
    assert [process.exitcode for process in processes] == [0] * count
    assert len(set(taken)) == count


@pytest.mark.asyncio
async def test_two_unnamed_runs_finish_side_by_side(tmp_path):
    def fake_runner(job: Evaluation, gate: asyncio.Event | None = None):
        async def run(task_path, _cfg):
            if gate is not None:
                await gate.wait()
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
            return RolloutResult(
                task_name=task_path.name,
                rollout_name=rollout_dir.name,
                rewards={"reward": 1.0},
                agent="oracle",
                rollout_dir=rollout_dir,
            )

        return run

    first, second = _evaluation(tmp_path, "qho"), _evaluation(tmp_path, "sho")
    gate = asyncio.Event()
    first._run_single_task = AsyncMock(side_effect=fake_runner(first, gate))
    second._run_single_task = AsyncMock(side_effect=fake_runner(second))
    running = asyncio.create_task(first.run())
    for _ in range(200):
        if (first._jobs_dir / first._job_name / ".evaluation.lock").exists():
            break
        await asyncio.sleep(0.01)
    done = await second.run()
    assert done.passed == 1
    gate.set()
    assert (await running).passed == 1
    assert first._job_name != second._job_name
    held = {
        run._job_name: sorted(
            p.name.split("__")[0]
            for p in (run._jobs_dir / run._job_name).iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        for run in (first, second)
    }
    assert held == {first._job_name: ["qho"], second._job_name: ["sho"]}
