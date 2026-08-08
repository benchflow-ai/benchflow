"""Host-side hard deadline per rollout attempt.

Every phase inside a rollout has its own timeout, but an await stuck BELOW that
instrumentation (a Daytona PTY kill on a dead websocket, a wedged session exec
in the post-verify export path) used to freeze the whole job: one hung
bike-rebalance rollout wedged a 25-task eval for 11+ hours after its verifier
had already finished (2026-08-07). ``_run_single_task`` now wraps
``rollout.run()`` in a computed hard deadline and converts a trip into a
normal infra-retryable error result.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow._utils.scoring import INFRA_ERROR
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    RetryConfig,
    _rollout_hard_deadline_sec,
)


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )


class TestDeadlineComputation:
    def test_env_disables(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        for raw in ("off", "none", "0"):
            monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raw)
            assert _rollout_hard_deadline_sec(tmp_path / "t", EvaluationConfig()) is None

    def test_env_numeric_overrides(self, tmp_path, monkeypatch):
        _write_task(tmp_path / "t")
        monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "123.5")
        assert _rollout_hard_deadline_sec(tmp_path / "t", EvaluationConfig()) == 123.5

    def test_computed_covers_all_phase_budgets(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        _write_task(tmp_path / "t")
        deadline = _rollout_hard_deadline_sec(tmp_path / "t", EvaluationConfig())
        # agent 60 + verifier 60 + build/install defaults + fixed margin: the
        # backstop must strictly dominate the sum of the declared phase budgets.
        assert deadline is not None
        assert deadline > 60 + 60 + 1800
        assert deadline < 24 * 3600

    def test_unreadable_task_falls_back_conservative(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", raising=False)
        deadline = _rollout_hard_deadline_sec(tmp_path / "missing", EvaluationConfig())
        assert deadline is not None
        assert deadline >= 3600


@pytest.mark.asyncio
async def test_wedged_rollout_becomes_infra_error(tmp_path, monkeypatch):
    """A rollout whose run() never returns must yield an error result, not a hang."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="wedge-run",
    )

    wedged = AsyncMock()

    async def _hang() -> None:
        await asyncio.sleep(3600)

    wedged.run = _hang
    wedged.cleanup = AsyncMock()

    with patch(
        "benchflow.rollout.Rollout.create", AsyncMock(return_value=wedged)
    ):
        result = await asyncio.wait_for(
            job._run_single_task(task_dir, job._config), timeout=15
        )

    assert result.error is not None
    assert "hard deadline" in result.error
    assert result.error_category == INFRA_ERROR
    wedged.cleanup.assert_awaited()


@pytest.mark.asyncio
async def test_wedged_cleanup_cannot_re_wedge(tmp_path, monkeypatch):
    """Even a cleanup that also hangs must not stall the error path."""
    task_dir = tmp_path / "wedge-task"
    _write_task(task_dir)
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.5")

    job = Evaluation(
        tasks_dir=task_dir,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(retry=RetryConfig(max_retries=0)),
        job_name="wedge-run-2",
    )

    wedged = AsyncMock()

    async def _hang() -> None:
        await asyncio.sleep(3600)

    wedged.run = _hang
    wedged.cleanup = _hang

    with (
        patch("benchflow.rollout.Rollout.create", AsyncMock(return_value=wedged)),
        patch("benchflow.evaluation._CLEANUP_BOUND_SEC", 0.5),
    ):
        result = await asyncio.wait_for(
            job._run_single_task(task_dir, job._config), timeout=15
        )

    assert result.error is not None
    assert "hard deadline" in result.error
