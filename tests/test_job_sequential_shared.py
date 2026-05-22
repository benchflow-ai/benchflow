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
async def test_sequential_shared_propagates_cancellation(tmp_path):
    """Guards v0.5-idle-timeout@219906c against swallowing cancellation."""
    job = _make_job(tmp_path, n_tasks=1, job_mode="sequential-shared")

    async def fake_run(*args, **kwargs):
        raise asyncio.CancelledError

    job._run_task = fake_run  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await job.run()


@pytest.mark.asyncio
async def test_parallel_independent_still_overlaps(tmp_path):
    """Regression guard: the default mode must still run concurrently.

    The fakes block on ``enough`` until ``concurrency`` rollouts overlap.
    If that overlap regresses (rollouts run one-at-a-time) the event is
    never set — the whole job would hang. ``asyncio.wait_for`` converts
    that hang into a fast, clear failure instead of a CI deadlock.
    """
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

    try:
        await asyncio.wait_for(job.run(), timeout=10)
    except TimeoutError:
        pytest.fail(
            "parallel-independent rollouts did not overlap — the job hung "
            f"waiting for {concurrency} concurrent rollouts that never "
            "started. Concurrency regressed to sequential."
        )
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


# --- sequential-shared: the memory/skills producer (capability 5) ---


@pytest.mark.asyncio
async def test_sequential_shared_injects_evolved_skills_into_next_rollout(tmp_path):
    """Rollout N+1 starts from the skill set rollout N evolved.

    The producer materializes the LearnerStore's current skills into the
    rollout's skills_dir. A rollout that evolves a skill must see that skill
    injected on the *next* rollout — the data path that closes capability 5.
    """
    from benchflow.learner_skills import capture_skills

    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    seen_injected: list[dict] = []

    async def fake_run(*args, **kwargs):
        # READ side: whatever the store materialized is the injected skills_dir.
        injected = capture_skills(job._learner_skills_dir)
        seen_injected.append(injected)
        # CAPTURE side: the agent evolved one new skill this rollout.
        n = len(seen_injected)
        evolved = dict(injected)
        evolved[f"skill-{n}"] = f"body-{n}"
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills=evolved,
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    # Rollout 1 started empty; rollout 2 saw skill-1; rollout 3 saw 1 and 2.
    assert seen_injected[0] == {}
    assert set(seen_injected[1]) == {"skill-1"}
    assert set(seen_injected[2]) == {"skill-1", "skill-2"}
    # The store's final generation carries every evolved skill.
    assert set(job.learner_store.current().skills) == {
        "skill-1",
        "skill-2",
        "skill-3",
    }


@pytest.mark.asyncio
async def test_sequential_shared_records_memory_delta_on_a_node(tmp_path):
    """Each rollout records before/after skills on a tree node, so the
    Memory-space scorer has its writer."""
    from benchflow.rewards.memory_scorer import MEMORY_STATE_KEY, MemoryScorer

    job = _make_job(tmp_path, n_tasks=2, job_mode="sequential-shared")

    async def fake_run(*args, **kwargs):
        n = len(job.learner_nodes) + 1
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills={f"skill-{n}": f"body-{n}"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    # One node per rollout, each carrying a memory_delta.
    assert len(job.learner_nodes) == 2
    first = job.learner_nodes[0].state[MEMORY_STATE_KEY]
    assert first["before"] == {}
    assert first["after"] == {"skill-1": "body-1"}
    # No `expected` answer-key is recorded — it must not be derived from the
    # agent's own diff (that would make the scorer a tautology).
    assert "expected" not in first

    # The recorded delta is scorable by the MemoryScorer end-to-end. With no
    # fixture the scorer grades activity: a non-empty skill change scores 1.0.
    event = await MemoryScorer().score(job.learner_nodes[0])
    assert event.space == "memory"
    assert event.reward == 1.0


@pytest.mark.asyncio
async def test_sequential_shared_threads_task_expected_skills_to_memory_delta(tmp_path):
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d.

    The Memory-space scorer must grade against a task-authored fixture, not
    derive an answer key from the agent's own diff.
    """
    from benchflow.rewards.memory_scorer import MEMORY_STATE_KEY, MemoryScorer

    job = _make_job(tmp_path, n_tasks=1, job_mode="sequential-shared")
    task_toml = job._tasks_dir / "task-0" / "task.toml"
    task_toml.write_text(
        'version = "1.0"\n'
        "[verifier]\ntimeout_sec = 60\n"
        '[verifier.memory]\nexpected_skills = ["skill-1"]\n'
        "[agent]\ntimeout_sec = 60\n"
        "[environment]\n"
    )

    async def fake_run(*args, **kwargs):
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills={"skill-1": "body-1", "spurious": "body-2"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    delta = job.learner_nodes[0].state[MEMORY_STATE_KEY]
    assert delta["expected"] == ["skill-1"]

    event = await MemoryScorer().score(job.learner_nodes[0])
    assert event.reward == 0.5


@pytest.mark.asyncio
async def test_sequential_shared_surfaces_memory_scores_in_summary_and_artifacts(
    tmp_path,
):
    """Guards OPEN-3 on v0.5-integration@ffef85d.

    Memory-space scoring is additive: it must surface as its own metric without
    changing output reward pass/fail semantics or leaking hidden expected skills.
    """
    import json

    job = _make_job(tmp_path, n_tasks=1, job_mode="sequential-shared")
    task_toml = job._tasks_dir / "task-0" / "task.toml"
    task_toml.write_text(
        'version = "1.0"\n'
        "[verifier]\ntimeout_sec = 60\n"
        '[verifier.memory]\nexpected_skills = ["skill-1"]\n'
        "[agent]\ntimeout_sec = 60\n"
        "[environment]\n"
    )

    async def fake_run(*args, **kwargs):
        rollout_name = f"{args[0].name}__abc123"
        rollout_dir = job._jobs_dir / job._job_name / rollout_name
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": args[0].name,
                    "rollout_name": rollout_name,
                    "rewards": {"reward": 1.0},
                    "timing": {},
                }
            )
        )
        (rollout_dir / "rewards.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-05-22T00:00:00",
                    "type": "terminal",
                    "source": "verifier",
                    "value": 1.0,
                    "tag": "reward",
                    "step_index": None,
                    "meta": {},
                }
            )
            + "\n"
        )
        return RunResult(
            task_name=args[0].name,
            rollout_name=rollout_name,
            rewards={"reward": 1.0},
            evolved_skills={"skill-1": "body-1", "spurious": "body-2"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]

    result = await job.run()

    assert result.score == 1.0
    assert result.memory_score == pytest.approx(0.5)
    assert result.memory_scores == {"task-0": pytest.approx(0.5)}

    summary = json.loads((job._jobs_dir / "summary.json").read_text())
    assert summary["score"] == "100.0%"
    assert summary["memory"] == {
        "scored": 1,
        "avg_score": 0.5,
        "score": "50.0%",
    }
    assert summary["memory_score"] == 0.5
    assert summary["memory_score_coverage"] == 1.0
    assert summary["memory_scores"] == {"task-0": 0.5}

    artifact = json.loads(
        (job._jobs_dir / job._job_name / "task-0__abc123" / "result.json").read_text()
    )
    assert artifact["memory_score"] == 0.5
    assert artifact["reward_events"] == [
        {
            "type": "terminal",
            "reward": 0.5,
            "source": "memory",
            "space": "memory",
            "granularity": "terminal",
        }
    ]
    serialized = json.dumps(artifact)
    assert "expected_skills" not in serialized
    assert '"expected"' not in serialized

    reward_lines = [
        json.loads(line)
        for line in (job._jobs_dir / job._job_name / "task-0__abc123" / "rewards.jsonl")
        .read_text()
        .splitlines()
    ]
    assert reward_lines[-1] == {
        "ts": "2026-05-22T00:00:00",
        "type": "terminal",
        "source": "memory",
        "value": 0.5,
        "tag": "memory",
        "step_index": None,
        "meta": {"space": "memory", "granularity": "terminal"},
    }
    serialized_rewards = json.dumps(reward_lines)
    assert "expected_skills" not in serialized_rewards
    assert '"expected"' not in serialized_rewards


@pytest.mark.asyncio
async def test_sequential_shared_evolved_skills_survive_a_revert(tmp_path):
    """A regressing rollout's evolved skills are discarded — the store stays
    at the better generation's skill set."""
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    plan = iter(
        [
            (0.8, {"good": "v1"}),  # gen 1 kept
            (0.3, {"good": "v1", "bad": "v1"}),  # regression, reverted
            (0.9, {"good": "v2"}),  # gen 2 kept
        ]
    )

    async def fake_run(*args, **kwargs):
        reward, evolved = next(plan)
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": reward},
            evolved_skills=evolved,
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    assert job.learner_store.learning_curve() == [0.8, 0.9]
    # The reverted "bad" skill never stuck; the final skill set is gen 2's.
    assert job.learner_store.current().skills == {"good": "v2"}


@pytest.mark.asyncio
async def test_sequential_shared_errored_rollout_records_delta_no_commit(tmp_path):
    """An errored rollout records its (empty) delta but commits no generation."""
    job = _make_job(tmp_path, n_tasks=2, job_mode="sequential-shared")
    results = iter(
        [
            RunResult(task_name="task-0", error="boom"),
            RunResult(
                task_name="task-1",
                rewards={"reward": 1.0},
                evolved_skills={"s": "v1"},
            ),
        ]
    )

    async def fake_run(*args, **kwargs):
        return next(results)

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    # Both rollouts recorded a node; only the scored one committed.
    assert len(job.learner_nodes) == 2
    assert job.learner_store.generation == 1


@pytest.mark.asyncio
async def test_run_task_raising_is_caught_in_sequential_shared(tmp_path):
    """A _run_task that raises must not abort the whole sequential job —
    it is caught, recorded as an errored result, and the run continues."""
    job = _make_job(tmp_path, n_tasks=3, job_mode="sequential-shared")
    calls = {"n": 0}

    async def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("rollout exploded")
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills={f"s{calls['n']}": "v1"},
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    result = await job.run()

    # All three tasks accounted for; the middle one is an error.
    assert result.total == 3
    assert result.errored == 1
    assert result.passed == 2
    # The two scored rollouts each stamped a generation; the crash did not.
    assert job.learner_store.generation == 2
    # The crashed rollout is skipped before _commit_learner_generation, so it
    # records no node — only the two completed rollouts do.
    assert len(job.learner_nodes) == 2
