"""Resume tests for the ``sequential-shared`` Job mode — guards issue #394.

A continual-learning Job persists its :class:`LearnerStore` under the job
directory and restores it on resume, so rollout N+1 inherits the (memory +
skills) state earlier rollouts evolved. A resume with completed rollouts but
no snapshot must fail closed — silently mixing old result rows with a fresh
empty store would invalidate the learning curve without a hard failure.
"""

from __future__ import annotations

import json

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.learner_store import LearnerState
from benchflow.models import RunResult


def _make_job(tmp_path, n_tasks: int, *, jobs_dir=None, job_name=None):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for i in range(n_tasks):
        td = tasks_dir / f"task-{i}"
        td.mkdir(exist_ok=True)
        (td / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    cfg = EvaluationConfig(
        concurrency=1,
        retry=RetryConfig(max_retries=0),
        job_mode="sequential-shared",
    )
    return Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir or tmp_path / "jobs",
        config=cfg,
        job_name=job_name,
    )


@pytest.mark.asyncio
async def test_sequential_shared_persists_learner_store_to_disk(tmp_path):
    """Every commit/revert must write the LearnerStore snapshot under the
    job directory — a resume needs it to continue the learning curve (#394)."""
    job = _make_job(tmp_path, n_tasks=2)
    rewards = iter([0.5, 0.9])

    async def fake_run(*args, **kwargs):
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": next(rewards)},
            evolved_skills={"k": "v"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    snapshot = job._jobs_dir / job._job_name / "learner_store.json"
    assert snapshot.is_file(), "learner_store.json must be persisted under job dir"
    data = json.loads(snapshot.read_text())
    assert data["generation"] == 2
    assert [g["metric"] for g in data["history"] if g["number"] != 0] == [0.5, 0.9]


@pytest.mark.asyncio
async def test_resume_restores_learner_store_from_disk(tmp_path):
    """A second Evaluation pointed at the same job dir must load the
    persisted store, not start from generation 0 (the bug #394 reports)."""
    jobs_dir = tmp_path / "jobs"
    job_name = "rollout-job"

    # First run: 1 task, stamps generation 1.
    job1 = _make_job(tmp_path, n_tasks=2, jobs_dir=jobs_dir, job_name=job_name)
    # Pretend task-0 already completed by writing a result.json *and* the
    # learner_store snapshot directly — that is the resume-from-crash shape.
    job1.learner_store.commit(  # type: ignore[union-attr]
        LearnerState(skills={"seed": "from-rollout-0"}), metric=0.8
    )
    job1._save_learner_store()

    rollout_dir = jobs_dir / job_name / "task-0__abc"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-0",
                "rollout_name": "task-0__abc",
                "rewards": {"reward": 0.8},
                "timing": {},
            }
        )
    )

    # Second run resumes: __init__ must load the persisted store.
    job2 = _make_job(tmp_path, n_tasks=2, jobs_dir=jobs_dir, job_name=job_name)
    assert job2.learner_store is not None
    assert job2.learner_store.generation == 1, (
        "resumed store must inherit prior generation, not restart at 0"
    )
    assert job2.learner_store.current().skills == {"seed": "from-rollout-0"}

    # The remaining task sees the inherited skills as `before_state.skills`.
    seen_before: list[dict] = []

    async def fake_run(*args, **kwargs):
        from benchflow.learner_skills import capture_skills

        seen_before.append(capture_skills(job2._learner_skills_dir))
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills={"seed": "from-rollout-0", "fresh": "rollout-1"},
        )

    job2._run_task = fake_run  # type: ignore[method-assign]
    await job2.run()

    assert seen_before == [{"seed": "from-rollout-0"}], (
        "the remaining rollout must inherit the persisted skill set"
    )
    assert job2.learner_store.generation == 2
    assert job2.learner_store.learning_curve() == [0.8, 1.0]


@pytest.mark.asyncio
async def test_resume_without_snapshot_but_completed_tasks_fails_closed(tmp_path):
    """The bug #394 reports: completed rollouts exist but the LearnerStore
    snapshot is missing. Must fail closed rather than silently aggregate old
    result rows with a fresh empty store."""
    jobs_dir = tmp_path / "jobs"
    job_name = "broken-resume"

    # Simulate an old job: result.json exists but no learner_store.json.
    rollout_dir = jobs_dir / job_name / "task-0__abc"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-0",
                "rollout_name": "task-0__abc",
                "rewards": {"reward": 0.5},
                "timing": {},
            }
        )
    )

    job = _make_job(tmp_path, n_tasks=2, jobs_dir=jobs_dir, job_name=job_name)

    async def should_never_run(*args, **kwargs):
        raise AssertionError("a corrupt-resume job must abort before running tasks")

    job._run_task = should_never_run  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no persisted LearnerStore"):
        await job.run()


@pytest.mark.asyncio
async def test_fresh_job_with_no_completed_tasks_does_not_fail(tmp_path):
    """A fresh sequential-shared job (no completed rollouts) must still run
    even though no snapshot exists — that is the normal first-run case."""
    job = _make_job(tmp_path, n_tasks=2)

    async def fake_run(*args, **kwargs):
        return RunResult(task_name=args[0].name, rewards={"reward": 1.0})

    job._run_task = fake_run  # type: ignore[method-assign]
    result = await job.run()
    assert result.total == 2
    assert job.learner_store.generation == 2


@pytest.mark.asyncio
async def test_summary_includes_learner_store_provenance(tmp_path):
    """summary.json must record the final generation and learning curve so a
    resumed run can be audited end-to-end (the third bullet in #394)."""
    job = _make_job(tmp_path, n_tasks=2)
    rewards = iter([0.4, 0.9])

    async def fake_run(*args, **kwargs):
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": next(rewards)},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    summary = json.loads((job._jobs_dir / "summary.json").read_text())
    assert "learner_store" in summary
    assert summary["learner_store"]["generation"] == 2
    assert summary["learner_store"]["learning_curve"] == [0.4, 0.9]
    # The path is recorded so the auditor knows where the snapshot lives.
    assert summary["learner_store"]["snapshot_path"].endswith("learner_store.json")


@pytest.mark.asyncio
async def test_result_artifact_records_generation_provenance(tmp_path):
    """Each rollout's result.json gets a learner_generation block — which
    generation it inherited from and which it produced — so a resume can be
    audited per-rollout (the third bullet in #394)."""
    job = _make_job(tmp_path, n_tasks=2)
    rewards = iter([0.4, 0.9])

    async def fake_run(*args, **kwargs):
        rollout_name = f"{args[0].name}__abc"
        rollout_dir = job._jobs_dir / job._job_name / rollout_name
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": args[0].name,
                    "rollout_name": rollout_name,
                    "rewards": {"reward": next(rewards)},
                    "timing": {},
                }
            )
        )
        return RunResult(
            task_name=args[0].name,
            rollout_name=rollout_name,
            rewards={"reward": 0.4 if args[0].name == "task-0" else 0.9},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    artifact_0 = json.loads(
        (job._jobs_dir / job._job_name / "task-0__abc" / "result.json").read_text()
    )
    artifact_1 = json.loads(
        (job._jobs_dir / job._job_name / "task-1__abc" / "result.json").read_text()
    )

    # Rollout 0 inherited from gen 0 (empty store) and produced gen 1.
    assert artifact_0["learner_generation"] == {
        "inherited_from": 0,
        "produced": 1,
        "committed": True,
    }
    # Rollout 1 inherited from gen 1 and produced gen 2.
    assert artifact_1["learner_generation"] == {
        "inherited_from": 1,
        "produced": 2,
        "committed": True,
    }


@pytest.mark.asyncio
async def test_regressed_rollout_records_committed_false(tmp_path):
    """A reverted (regressed) rollout records ``committed=False`` and no
    produced generation so the audit trail does not lie about the curve."""
    job = _make_job(tmp_path, n_tasks=2)
    rewards = iter([0.9, 0.2])  # second regresses

    async def fake_run(*args, **kwargs):
        rollout_name = f"{args[0].name}__abc"
        rollout_dir = job._jobs_dir / job._job_name / rollout_name
        rollout_dir.mkdir(parents=True, exist_ok=True)
        reward = next(rewards)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": args[0].name,
                    "rollout_name": rollout_name,
                    "rewards": {"reward": reward},
                    "timing": {},
                }
            )
        )
        return RunResult(
            task_name=args[0].name,
            rollout_name=rollout_name,
            rewards={"reward": reward},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    regressed = json.loads(
        (job._jobs_dir / job._job_name / "task-1__abc" / "result.json").read_text()
    )
    assert regressed["learner_generation"] == {
        "inherited_from": 1,
        "produced": None,
        "committed": False,
    }


def test_corrupt_snapshot_fails_closed_at_init(tmp_path):
    """A corrupt learner_store.json must raise at ``__init__`` time — silently
    starting fresh would reintroduce the bug #394."""
    jobs_dir = tmp_path / "jobs"
    job_name = "with-bad-snap"
    (jobs_dir / job_name).mkdir(parents=True)
    (jobs_dir / job_name / "learner_store.json").write_text("{not valid json")

    with pytest.raises(RuntimeError, match="Could not load persisted LearnerStore"):
        _make_job(tmp_path, n_tasks=1, jobs_dir=jobs_dir, job_name=job_name)


def test_parallel_independent_does_not_persist_learner_store(tmp_path):
    """parallel-independent jobs have no LearnerStore — there is nothing to
    persist, and no snapshot should appear under the job dir."""
    cfg = EvaluationConfig(job_mode="parallel-independent")
    job = Evaluation(
        tasks_dir=tmp_path / "t",
        jobs_dir=tmp_path / "jobs",
        config=cfg,
    )
    assert job.learner_store is None
