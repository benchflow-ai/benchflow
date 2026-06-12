"""Tests for batch continuation discovery and scheduling."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow.continue_run.batch import (
    continue_batch,
    discover_timeout_run_folders,
    summarize_batch,
)
from benchflow.continue_run.orchestrator import ContinueResult

from ._helpers import completion, exchange, write_run_folder


def test_discover_timeout_run_folders_filters_non_timeouts(tmp_path):
    """Guards PR #648 follow-up: batch mode must only pick unfinished runs."""
    timeout = write_run_folder(
        tmp_path / "timeout",
        exchanges=[exchange(completion(content="a"))],
        error_category="timeout",
    )
    write_run_folder(
        tmp_path / "agent-error",
        exchanges=[exchange(completion(content="a"))],
        error_category="agent_error",
    )

    assert discover_timeout_run_folders(tmp_path) == [timeout]


@pytest.mark.asyncio
async def test_continue_batch_runs_with_bounded_concurrency(tmp_path):
    """Guards PR #648 follow-up rolling scheduler for large Daytona batches."""
    folders = [
        write_run_folder(
            tmp_path / f"run-{idx}",
            exchanges=[exchange(completion(content="a"))],
            error_category="timeout",
        )
        for idx in range(3)
    ]
    active = 0
    max_active = 0

    async def runner(folder: Path, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return ContinueResult(
            rollout_dir=folder / "continued",
            rewards={"reward": 1.0},
            error=None,
            n_recorded=1,
            n_live=1,
            divergences=0,
        )

    results = await continue_batch(
        folders,
        concurrency=2,
        tasks_dir=None,
        model=None,
        timeout=None,
        output_dir=None,
        runner=runner,
    )

    assert [result.ok for result in results] == [True, True, True]
    assert max_active <= 2


@pytest.mark.asyncio
async def test_continue_batch_marks_agent_error_as_failed(tmp_path):
    """Guards PR #648 follow-up: batch progress must not hide failed artifacts."""
    folder = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        error_category="timeout",
    )

    async def runner(folder: Path, **kwargs):
        return ContinueResult(
            rollout_dir=folder / "continued",
            rewards=None,
            error="Failed to create session",
            n_recorded=1,
            n_live=0,
            divergences=0,
        )

    results = await continue_batch(
        [folder],
        concurrency=1,
        tasks_dir=None,
        model=None,
        timeout=None,
        output_dir=None,
        runner=runner,
    )

    summary = summarize_batch(results)
    assert results[0].ok is False
    assert summary["failed"] == 1
    assert summary["errors"][0]["output"].endswith("/continued")
