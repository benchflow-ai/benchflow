"""Batch ergonomics on Evaluation.

A Python caller could not watch results arrive except through the
``on_result`` callback, could only resume a job by re-creating the exact
``Evaluation(tasks_dir, jobs_dir, config, job_name)`` it started with, and had
to hand-roll CSV/JSONL export. ``Evaluation.stream()``, ``run_sync()``,
``Evaluation.resume(job_dir)`` (backed by an ``evaluation.json`` written when a
job starts, with agent_env values left out) and ``to_records``/``to_csv``/
``to_jsonl`` on ``EvaluationResult`` close those gaps.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    RetryConfig,
)
from benchflow.models import RolloutResult

SECRET = "sk-test-not-a-real-key-1234567890"


class _ProcessDied(BaseException):
    """Simulates the process dying mid-batch (not caught as an Exception)."""


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _job(tmp_path: Path, n: int = 3, **cfg) -> Evaluation:
    config = EvaluationConfig(
        agent="oracle", concurrency=1, retry=RetryConfig(max_retries=0), **cfg
    )
    return Evaluation(
        tasks_dir=_tasks(tmp_path, n), jobs_dir=tmp_path / "jobs", config=config
    )


def _fake_runner(job: Evaluation, *, delay: float = 0.0, fail: set[str] = frozenset()):
    async def fake_run(task_path, _cfg):
        await asyncio.sleep(delay)
        if task_path.name in fail:
            raise _ProcessDied
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_name": task_path.name,
            "rollout_name": rollout_dir.name,
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
            "agent": "oracle",
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 1.0},
            agent="oracle",
            rollout_dir=rollout_dir,
        )

    return fake_run


@pytest.mark.asyncio
async def test_stream_yields_results_as_they_finish(tmp_path: Path) -> None:
    job = _job(tmp_path)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    names = []
    async with contextlib.aclosing(job.stream()) as stream:
        async for name, result in stream:
            assert isinstance(result, RolloutResult)
            names.append(name)
    assert sorted(names) == ["task-0", "task-1", "task-2"]
    assert job.result is not None and job.result.passed == 3


def test_run_sync(tmp_path: Path) -> None:
    job = _job(tmp_path, n=2)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    assert job.run_sync().passed == 2


@pytest.mark.asyncio
async def test_job_start_records_evaluation_json_without_secrets(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, n=1, agent_env={"OPENAI_API_KEY": SECRET}, model="m")
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    await job.run()
    record = job._jobs_dir / job._job_name / "evaluation.json"
    text = record.read_text()
    assert SECRET not in text
    data = json.loads(text)
    assert data["tasks_dir"] == str(job._tasks_dir)
    assert data["config"]["model"] == "m"
    assert data["config"]["agent_env_keys"] == ["OPENAI_API_KEY"]


@pytest.mark.asyncio
async def test_resume_from_job_dir_runs_only_the_rest(tmp_path: Path) -> None:
    job = _job(tmp_path, n=3, model="m")
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job, fail={"task-2"}))
    first = await job.run()  # task-2 dies before writing its result.json
    assert first.errored == 1
    job_dir = job._jobs_dir / job._job_name

    resumed = Evaluation.resume(job_dir)
    assert resumed._tasks_dir == job._tasks_dir
    assert resumed._config.model == "m" and resumed._config.agent == "oracle"
    assert resumed._job_name == job_dir.name
    resumed._run_single_task = AsyncMock(side_effect=_fake_runner(resumed))
    result = await resumed.run()

    ran = [c.args[0].name for c in resumed._run_single_task.await_args_list]
    assert ran == ["task-2"]
    assert result.total == 3 and result.passed == 3


def test_resume_warns_about_agent_env_it_cannot_restore(tmp_path: Path, caplog) -> None:
    job = _job(tmp_path, n=1, agent_env={"OPENAI_API_KEY": SECRET})
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    job.run_sync()
    job_dir = job._jobs_dir / job._job_name
    with caplog.at_level("WARNING"):
        Evaluation.resume(job_dir)
    assert "OPENAI_API_KEY" in caplog.text
    restored = Evaluation.resume(job_dir, agent_env={"OPENAI_API_KEY": SECRET})
    assert restored._config.agent_env == {"OPENAI_API_KEY": SECRET}


def test_resume_without_a_record_names_the_fix(tmp_path: Path) -> None:
    (tmp_path / "old-job").mkdir()
    with pytest.raises(FileNotFoundError, match="tasks_dir"):
        Evaluation.resume(tmp_path / "old-job")


def test_evaluation_result_exports(tmp_path: Path) -> None:
    job = _job(tmp_path, n=2)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    result = job.run_sync()
    records = result.to_records()
    assert [r["task_name"] for r in records] == ["task-0", "task-1"]
    rows = list(csv.DictReader(result.to_csv(tmp_path / "out.csv").open()))
    assert len(rows) == 2 and rows[0]["passed"] == "True"
    lines = result.to_jsonl(tmp_path / "out.jsonl").read_text().splitlines()
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_a_second_run_of_a_live_job_is_refused(tmp_path: Path) -> None:
    """Regression test: resuming a job whose first process was still running made
    both processes run the remaining tasks into the same job directory."""
    job = _job(tmp_path, n=2)
    gate = asyncio.Event()
    runner = _fake_runner(job)

    async def slow(task_path, cfg):
        await gate.wait()
        return await runner(task_path, cfg)

    job._run_single_task = AsyncMock(side_effect=slow)
    first = asyncio.create_task(job.run())
    job_dir = job._jobs_dir / job._job_name
    for _ in range(100):
        if (job_dir / ".evaluation.lock").exists():
            break
        await asyncio.sleep(0.01)

    second = Evaluation.resume(job_dir)
    second._run_single_task = AsyncMock(side_effect=runner)
    with pytest.raises(RuntimeError, match="already running"):
        await second.run()
    assert second._run_single_task.await_count == 0

    gate.set()
    assert (await first).passed == 2
    assert not (job_dir / ".evaluation.lock").exists()


def test_a_stale_lock_from_a_dead_process_is_taken_over(tmp_path: Path, caplog) -> None:
    job = _job(tmp_path, n=1)
    job_dir = job._jobs_dir / job._job_name
    job_dir.mkdir(parents=True)
    import socket

    (job_dir / ".evaluation.lock").write_text(
        json.dumps({"pid": 999_999_999, "host": socket.gethostname()})
    )
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))
    with caplog.at_level("WARNING"):
        assert job.run_sync().passed == 1
    assert "stale" in caplog.text


def test_a_refused_empty_selection_leaves_no_job_dir(tmp_path: Path) -> None:
    from benchflow.evaluation import EmptyTaskSelectionError

    job = _job(tmp_path, n=1, include_tasks={"nope"})
    with pytest.raises(EmptyTaskSelectionError):
        job.run_sync()
    assert not (job._jobs_dir / job._job_name).exists()


@pytest.mark.asyncio
async def test_cancelling_a_stream_consumer_stops_the_job(tmp_path: Path) -> None:
    """Ctrl-C cancels asyncio.run's main task; when that task is iterating
    evaluation.stream(), the job must stop and release its lock, not run the
    remaining tasks."""
    job = _job(tmp_path, n=4)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job, delay=0.05))
    names: list[str] = []

    async def consume() -> None:
        async with contextlib.aclosing(job.stream()) as stream:
            async for name, _ in stream:
                names.append(name)

    consumer = asyncio.create_task(consume())
    while not names:
        await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.sleep(0.3)
    assert job._run_single_task.await_count <= 2
    assert not (job._jobs_dir / job._job_name / ".evaluation.lock").exists()


async def _lock_evidence(sandbox) -> None:
    """A pre-agent hook (it would lock a folder before the agent starts)."""


def test_pre_agent_hooks_reach_every_rollout_of_an_evaluation(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """``EvaluationConfig(pre_agent_hooks=[...])`` runs them in every rollout.

    An Evaluation had no hooks (only ``RolloutConfig`` did), so a job that
    needed to prepare each sandbox had to give up retries, resume and the
    summary and loop over ``bf.run`` itself. Hooks cannot be stored: config
    files refuse them, evaluation.json names them, a resume asks for them.
    """
    from benchflow.eval_sharding import run_sharded_evaluation

    seen = []

    async def fake_create(config):
        seen.append(config)

        class FakeRollout:
            async def run(self):
                return RolloutResult(
                    task_name=config.task_path.name, rewards={"reward": 1.0}
                )

        return FakeRollout()

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_create)
    config = EvaluationConfig(agent="oracle", pre_agent_hooks=[_lock_evidence])
    job = Evaluation(
        _tasks(tmp_path, 2), tmp_path / "jobs", config=config, preflight=False
    )
    asyncio.run(job.run())

    assert [c.pre_agent_hooks for c in seen] == [[_lock_evidence], [_lock_evidence]]
    record = json.loads((job.job_dir / "evaluation.json").read_text())
    assert record["config"]["pre_agent_hooks"] == [f"{__name__}._lock_evidence"]
    with pytest.raises(ValueError, match="pre_agent_hooks"):
        job.to_dict()
    with caplog.at_level("WARNING"):
        resumed = Evaluation.resume(job.job_dir)
    assert "pass pre_agent_hooks" in caplog.text
    assert resumed._config.pre_agent_hooks is None
    again = Evaluation.resume(job.job_dir, pre_agent_hooks=[_lock_evidence])
    assert again._config.pre_agent_hooks == [_lock_evidence]
    with pytest.raises(ValueError, match="worker"):
        asyncio.run(
            run_sharded_evaluation(
                tasks_dir=_tasks(tmp_path / "w", 1),
                jobs_dir=tmp_path / "wjobs",
                config=config,
                worker_concurrency=1,
                worker_retries=0,
                worker_start_stagger_sec=0.0,
            )
        )


def test_a_missing_tasks_folder_says_what_to_pass(tmp_path: Path) -> None:
    """``Evaluation(tasks_dir=<missing>)`` failed with a bare ``[Errno 2]``."""
    job = Evaluation(
        tmp_path / "nope",
        tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle"),
        preflight=False,
    )
    with pytest.raises(FileNotFoundError, match=r"Tasks directory not found.*task\.md"):
        job.run_sync()
