"""Regression tests for #389 follow-up: export-error channel + commit gating.

Two material issues found in PR-#389 review:

1. Channel mix: routing a skill-export failure through ``Rollout._error`` runs
   ``classify_error("Skill export failed: ... connection lost")`` through the
   "looks_like_infra" heuristic and mis-tags the rollout as agent
   ``infra_failure``. That contaminates agent-error dashboards with a
   cleanup-phase event the agent never caused. Fix: dedicated
   ``_export_error`` / ``RolloutResult.export_error`` sibling channel.

2. Learner-store poisoning: ``_commit_learner_generation`` ran unconditionally
   even when export failed but the verifier already produced rewards.
   It would commit an empty/partial skill delta as the next generation. Fix:
   short-circuit on ``result.export_error``, and have
   ``evolved_skills_for_result`` return ``{}`` instead of re-reading the
   half-written export dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.learner_memory import evolved_skills_for_result
from benchflow._utils.scoring import classify_error
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RunResult

# --- Issue 1: classify_error must not mis-tag export-time infra failures ---


def test_classify_error_on_export_failure_message_would_say_infra():
    """Document the bug PR-389 fix introduced if export rides ``error``.

    The fix lifts the export message onto a sibling channel exactly because
    ``classify_error`` cannot distinguish agent infra failures from
    cleanup-phase export infra failures by string alone.
    """
    # A real-shape message PR-389 was routing through self._error.
    msg = "Skill export failed: sandbox connection lost while pulling /app/skills"
    # If this were on result.error, dashboards would group it with the
    # agent's own infra crashes. That's the contamination #389 follow-up
    # exists to prevent — the fix keeps result.error = None and puts the
    # message on result.export_error instead.
    assert classify_error(msg) == "infra_failure"


def test_rollout_result_export_error_does_not_feed_classify_error():
    """A RolloutResult with only export_error set looks clean to classify_error."""
    r = RunResult(
        task_name="t",
        rewards={"reward": 1.0},
        export_error="Skill export failed: sandbox connection lost",
    )
    # The agent error channel stays clean, so dashboards see no infra bump.
    assert r.error is None
    assert classify_error(r.error) is None
    # But the run is not "success" — the export failure is observable.
    assert r.success is False


def test_rollout_result_success_requires_no_export_error():
    """``RolloutResult.success`` must also gate on ``export_error``."""
    ok = RunResult(task_name="t", rewards={"reward": 1.0})
    assert ok.success is True

    failed = RunResult(
        task_name="t",
        rewards={"reward": 1.0},
        export_error="Skill export failed: download failed",
    )
    assert failed.success is False


# --- Issue 2: evolved_skills_for_result must short-circuit on export_error ---


def test_evolved_skills_for_result_returns_empty_when_export_failed(tmp_path: Path):
    """Refuse to re-read the half-written export dir on a failed export.

    Otherwise the next generation would commit whatever partial state the
    sandbox managed to write before it died — exactly the poisoning the
    export-error channel exists to prevent.
    """
    # Simulate a half-written export dir on disk (would normally be picked
    # up by capture_skills if we fell through).
    export_dir = tmp_path / "exports"
    skill_dir = export_dir / "half-written"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: half-written\n---\n# WIP\n")

    result = RunResult(
        task_name="t",
        rewards={"reward": 1.0},
        evolved_skills=None,  # Rollout set this to None on export failure.
        export_error="Skill export failed: download failed",
    )

    # Even with files on disk and rewards populated, we return {}.
    assert evolved_skills_for_result(result, export_dir) == {}


def test_evolved_skills_for_result_prefers_result_payload_when_no_error(
    tmp_path: Path,
):
    """Happy path: still prefer the in-memory payload, ignore the disk fallback."""
    result = RunResult(
        task_name="t",
        rewards={"reward": 1.0},
        evolved_skills={"a": "# A\n"},
    )
    assert evolved_skills_for_result(result, tmp_path / "exports") == {"a": "# A\n"}


# --- Issue 2: _commit_learner_generation must skip on export_error ---


def _make_job(tmp_path: Path, n_tasks: int):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for i in range(n_tasks):
        (tasks_dir / f"task-{i}").mkdir()
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    cfg = EvaluationConfig(
        concurrency=1,
        retry=RetryConfig(max_retries=0),
        job_mode="sequential-shared",
    )
    return Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)


@pytest.mark.asyncio
async def test_export_failure_with_rewards_does_not_commit_generation(
    tmp_path: Path,
):
    """The exact poisoning scenario: verifier produced rewards, but export failed.

    Pre-fix, ``_commit_learner_generation`` would happily stamp a new
    generation with an empty/partial skill delta because ``result.rewards``
    was populated. The fix gates the entire commit path on
    ``result.export_error is None``.
    """
    job = _make_job(tmp_path, n_tasks=2)
    results = iter(
        [
            # Rewards populated AND evolved_skills cleared by the rollout
            # AND export_error set — the realistic failure shape.
            RunResult(
                task_name="task-0",
                rewards={"reward": 1.0},
                evolved_skills=None,
                export_error="Skill export failed: connection lost",
            ),
            RunResult(
                task_name="task-1",
                rewards={"reward": 1.0},
                evolved_skills={"good": "v1"},
            ),
        ]
    )

    async def fake_run(*args, **kwargs):
        return next(results)

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    # Only the clean rollout stamped a generation; the export-failed one
    # left the store untouched even though it had rewards.
    assert job.learner_store.generation == 1
    assert job.learner_store.current().skills == {"good": "v1"}


@pytest.mark.asyncio
async def test_export_failure_records_no_memory_node(tmp_path: Path):
    """Skipping the commit also skips the Memory-space node attachment.

    The Memory-space scorer reads ``memory_delta`` off the rollout's tree
    node. A poisoned delta would still score as activity, so the entire
    commit path — node attach included — must short-circuit.
    """
    job = _make_job(tmp_path, n_tasks=1)

    async def fake_run(*args, **kwargs):
        return RunResult(
            task_name=args[0].name,
            rewards={"reward": 1.0},
            evolved_skills=None,
            export_error="Skill export failed: connection lost",
        )

    job._run_task = fake_run  # type: ignore[method-assign]
    await job.run()

    assert job.learner_store.generation == 0
    # No learner node attached for the export-failed rollout.
    assert job.learner_nodes == []
