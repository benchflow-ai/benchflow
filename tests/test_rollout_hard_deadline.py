"""Host-side hard deadline per rollout attempt.

Every phase inside a rollout has its own timeout, but an await stuck BELOW
that instrumentation (a Daytona PTY kill on a dead websocket, a wedged session
exec in the post-verify export path) used to freeze the whole job: one hung
bike-rebalance rollout wedged a 25-task eval for 11+ hours after its verifier
had already finished (2026-08-07). ``Rollout.run()`` now enforces a computed
hard deadline around the lifecycle (covering every caller — Evaluation,
Runtime, bf.run, continue_run, acceptance) and converts a trip into a normal
infra-retryable error result.

The enforcement deliberately avoids a bare ``asyncio.wait_for``: cancelling
the lifecycle runs its ``finally: cleanup()``, and when the teardown is
itself the wedged path, ``wait_for`` would block past its own deadline
waiting for that cleanup — so cancellation gets its own bounded grace before
the task is abandoned outright.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from benchflow._utils.scoring import INFRA_ERROR
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.rollout import Rollout, RolloutConfig, _deadline
from benchflow.rollout._deadline import hard_deadline_sec


def _write_task(task_dir: Path) -> None:
    # A minimal *valid* legacy task (task.toml + instruction.md): the deadline
    # tests must exercise the computed-budget path, not the unreadable-task
    # fallback.
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do the task.\n")


class TestDeadlineComputation:
    def test_env_disables(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        for raw in ("off", "none", "0", "0.0", "-5"):
            monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raw)
            assert hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t")) is None

    def test_env_numeric_overrides(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "123.5")
        assert hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t")) == 123.5

    def test_computed_covers_all_phase_budgets(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        deadline = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t"))
        # agent 60 + verifier 60 + build/install defaults + fixed margin: the
        # backstop must strictly dominate the sum of the declared phase budgets.
        assert deadline is not None
        assert deadline > 60 + 60 + 1800
        # ... and stay below the unreadable-task fallback: this proves the
        # computed path ran (a fallback value would also satisfy the bounds
        # above, silently masking a budget-read regression).
        assert deadline < _deadline._FALLBACK_SEC

    def test_caller_timeout_override_dominates_task_budget(self, tmp_path, monkeypatch):
        """RolloutConfig.timeout (Runtime's wall-clock seam, #378) is the
        enforced agent budget — the backstop must be derived from it, not the
        smaller task default, or it would fire during a legitimate long run."""
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        base = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "t"))
        raised = hard_deadline_sec(
            RolloutConfig(task_path=tmp_path / "t", timeout=7200)
        )
        assert base is not None and raised is not None
        assert raised >= base + 7200 - 60

    def test_user_loop_rounds_dominate(self, tmp_path, monkeypatch):
        """A user-loop run legitimately spends max_user_rounds x (agent +
        soft-verify); the backstop must cover it, not fire mid-loop."""
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        cfg = RolloutConfig(task_path=tmp_path / "t", max_user_rounds=10)
        cfg.user = SimpleNamespace()  # any non-None user engages the loop
        deadline = hard_deadline_sec(cfg)
        assert deadline is not None
        assert deadline >= 10 * (60 + 60) + 1800

    def test_unreadable_task_falls_back_conservative(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        deadline = hard_deadline_sec(RolloutConfig(task_path=tmp_path / "missing"))
        assert deadline is not None
        assert deadline >= 3600


def _wedged_rollout(task_dir: Path, *, cleanup_wedged: bool = False) -> Rollout:
    """A real Rollout whose lifecycle hangs; optionally its cancellation
    cleanup hangs too (the re-wedge case a bare wait_for cannot break)."""
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=task_dir)

    async def _lifecycle():
        try:
            await asyncio.sleep(3600)  # wedged transport await
        finally:
            if cleanup_wedged:
                # Mirrors run()'s `finally: await self.cleanup()` hanging on a
                # dead connection: swallow the cancel and keep blocking.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(3600)

    rollout._run_lifecycle = _lifecycle
    return rollout


@pytest.mark.asyncio
async def test_wedged_lifecycle_becomes_infra_error(tmp_path, monkeypatch):
    """A lifecycle that never returns must yield an error result, not a hang,
    and the standard retry policy must treat it as retryable."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    rollout = _wedged_rollout(task_dir)
    result = await asyncio.wait_for(rollout.run(), timeout=15)

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR
    assert RetryConfig().should_retry(result.error, category=result.error_category)


@pytest.mark.asyncio
async def test_wedged_cleanup_cannot_re_wedge(tmp_path, monkeypatch):
    """Even when the cancellation-triggered cleanup also hangs (the observed
    incident shape: teardown wedged on a dead websocket), the deadline must
    still surface a result — a bare wait_for would block here forever."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.5)

    rollout = _wedged_rollout(task_dir, cleanup_wedged=True)
    result = await asyncio.wait_for(rollout.run(), timeout=15)

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR


@pytest.mark.asyncio
async def test_caller_cancellation_is_bounded(tmp_path, monkeypatch):
    """Cancelling run() from outside (job shutdown) must forward the cancel to
    the lifecycle and not wait unbounded for a wedged teardown."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "60")
    monkeypatch.setattr(_deadline, "ABANDON_CLEANUP_BOUND_SEC", 0.5)

    rollout = _wedged_rollout(task_dir, cleanup_wedged=True)
    run_task = asyncio.create_task(rollout.run())
    await asyncio.sleep(0.05)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=15)


@pytest.mark.asyncio
async def test_evaluation_path_surfaces_infra_error(tmp_path, monkeypatch):
    """End-to-end through Evaluation._run_single_task: the deadline inside
    Rollout.run() protects the eval job without any caller-side wrap."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="wedge-run",
    )

    with patch(
        "benchflow.rollout.Rollout.create",
        AsyncMock(return_value=_wedged_rollout(task_dir)),
    ):
        result = await asyncio.wait_for(
            job._run_single_task(task_dir, job._config), timeout=15
        )

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR
