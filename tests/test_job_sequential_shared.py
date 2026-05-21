"""Tests for the ``sequential-shared`` Job mode — continual learning wiring.

A continual-learning Job runs in ``sequential-shared`` mode: Rollouts run in
order over a persistent, generation-versioned LearnerStore. The default
``parallel-independent`` mode is unchanged — these tests pin the new mode and
prove the old one still runs concurrently.
"""

import asyncio

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.learner_store import LearnerStore
from benchflow.models import RunResult


def _make_job(tmp_path, n_tasks: int, *, job_mode: str, concurrency: int = 4):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for i in range(n_tasks):
        (tasks_dir / f"task-{i}").mkdir()
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    cfg = EvaluationConfig(
        concurrency=concurrency,
        retry=RetryConfig(max_retries=0),
        job_mode=job_mode,
    )
    return Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)


# --- config ---


def test_default_job_mode_is_parallel_independent():
    assert EvaluationConfig().job_mode == "parallel-independent"


def test_unknown_job_mode_rejected():
    with pytest.raises(ValueError, match="job_mode"):
        EvaluationConfig(job_mode="bogus")


def test_native_yaml_parses_job_mode(tmp_path):
    yaml_path = tmp_path / "job.yaml"
    yaml_path.write_text(
        "tasks_dir: tasks\njobs_dir: jobs\njob_mode: sequential-shared\n"
    )
    job = Evaluation.from_yaml(yaml_path)
    assert job._config.job_mode == "sequential-shared"


# --- sequential-shared: ordering ---


@pytest.mark.asyncio
async def test_sequential_shared_runs_tasks_in_order(tmp_path):
    """sequential-shared must run rollouts strictly one-at-a-time, in order —
    never overlapping, regardless of the concurrency setting."""
    job = _make_job(tmp_path, n_tasks=4, job_mode="sequential-shared", concurrency=8)

    order: list[str] = []
    in_flight = 0
    max_in_flight = 0

    async def fake_run(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        task_dir = args[0]
        order.append(task_dir.name)
        await asyncio.sleep(0)
        in_flight -= 1
        return RunResult(task_name=task_dir.name, rewards={"reward": 1.0})

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()

    assert order == ["task-0", "task-1", "task-2", "task-3"]
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_parallel_independent_still_overlaps(tmp_path):
    """Regression guard: the default mode must still run concurrently."""
    concurrency = 3
    job = _make_job(
        tmp_path, n_tasks=6, job_mode="parallel-independent", concurrency=concurrency
    )

    in_flight = 0
    max_in_flight = 0
    enough = asyncio.Event()

    async def fake_run(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight >= concurrency:
            enough.set()
        await enough.wait()
        in_flight -= 1
        return RunResult(task_name=args[0].name, rewards={"reward": 1.0})

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()
    assert max_in_flight == concurrency


# --- sequential-shared: the learner store ---


@pytest.mark.asyncio
async def test_sequential_shared_commits_a_generation_per_rollout(tmp_path):
    """Each rollout stamps a generation on the shared learner store."""
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")

    async def fake_run(*args, **kwargs):
        return RunResult(task_name=args[0].name, rewards={"reward": 1.0})

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()

    assert isinstance(job.learner_store, LearnerStore)
    # 3 improving/flat rollouts => 3 generations.
    assert job.learner_store.generation == 3


@pytest.mark.asyncio
async def test_sequential_shared_records_reward_as_learning_curve(tmp_path):
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    rewards = iter([0.2, 0.6, 1.0])

    async def fake_run(*args, **kwargs):
        return RunResult(task_name=args[0].name, rewards={"reward": next(rewards)})

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()
    assert job.learner_store.learning_curve() == [0.2, 0.6, 1.0]


@pytest.mark.asyncio
async def test_sequential_shared_reverts_a_regression(tmp_path):
    """A rollout whose reward regresses against the best generation so far is
    rolled back — the store stays at the better generation."""
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    rewards = iter([0.8, 0.3, 0.9])

    async def fake_run(*args, **kwargs):
        return RunResult(task_name=args[0].name, rewards={"reward": next(rewards)})

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()

    # gen1 = 0.8 (kept), gen2 = 0.3 (regression, reverted), gen3 = 0.9 (kept).
    assert job.learner_store.learning_curve() == [0.8, 0.9]
    assert job.learner_store.generation == 2


@pytest.mark.asyncio
async def test_sequential_shared_errored_rollout_does_not_commit(tmp_path):
    """An errored rollout (no reward) must not stamp a generation."""
    job = _make_job(tmp_path, n_tasks=2, job_mode="sequential-shared")
    results = iter(
        [
            RunResult(task_name="task-0", error="boom"),
            RunResult(task_name="task-1", rewards={"reward": 1.0}),
        ]
    )

    async def fake_run(*args, **kwargs):
        return next(results)

    job._run_task = fake_run  # type: ignore[method-assign]

    await job.run()
    # Only the scored rollout committed.
    assert job.learner_store.generation == 1


@pytest.mark.asyncio
async def test_sequential_shared_still_aggregates_results(tmp_path):
    """sequential-shared must still produce a normal EvaluationResult."""
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    rewards = iter([1.0, 0.0, 1.0])

    async def fake_run(*args, **kwargs):
        return RunResult(task_name=args[0].name, rewards={"reward": next(rewards)})

    job._run_task = fake_run  # type: ignore[method-assign]

    result = await job.run()
    assert result.total == 3
    assert result.passed == 2
    assert result.failed == 1
