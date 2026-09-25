"""Tests for batch continuation discovery and scheduling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.continue_run.batch import (
    continue_batch,
    discover_timeout_run_folders,
    summarize_batch,
)
from benchflow.continue_run.orchestrator import ContinueResult
from benchflow.continue_run.run_folder import load_run_folder

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
    assert results[0].skipped is False
    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"][0]["output"].endswith("/continued")


# ── #1083 step 1(c): one unsupported run must not sink the batch ──────────────


@pytest.mark.asyncio
async def test_mixed_batch_skips_unsupported_and_continues_the_rest(tmp_path):
    """A batch of mostly-openhands runs continues; the odd ones out are skips."""
    supported = [
        write_run_folder(
            tmp_path / f"oh-{idx}",
            exchanges=[exchange(completion(content="a"))],
            error_category="timeout",
        )
        for idx in range(3)
    ]
    unsupported = write_run_folder(
        tmp_path / "acp",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
        error_category="timeout",
    )
    unrecorded = write_run_folder(
        tmp_path / "subscription",
        exchanges=[exchange(completion(content="a"))],
        error_category="timeout",
    )
    (unrecorded / "trajectory" / "llm_trajectory.jsonl").unlink()

    folders = [*supported, unsupported, unrecorded]

    async def runner(folder: Path, **kwargs):
        # Real runners load the folder first; an unsupported one raises here.
        load_run_folder(folder, require_timeout=kwargs.get("require_timeout", True))
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

    by_folder = {result.folder: result for result in results}
    # the supported runs all completed — the batch was not aborted
    assert [by_folder[folder].ok for folder in supported] == [True, True, True]

    skipped = [result for result in results if result.skipped]
    assert {result.folder for result in skipped} == {unsupported, unrecorded}
    assert all(result.ok is False for result in skipped)
    reasons = {result.folder: result.reason_code for result in skipped}
    assert reasons[unsupported] == "unsupported_agent"
    assert reasons[unrecorded] == "no_llm_recording"

    summary = summarize_batch(results)
    assert summary["total"] == 5
    assert summary["succeeded"] == 3
    # skips are reported separately — they are not run failures
    assert summary["failed"] == 0
    assert summary["skipped"] == 2
    assert summary["errors"] == []
    skip_rows = {row["folder"]: row for row in summary["skips"]}
    assert skip_rows[str(unsupported)]["reason"] == "unsupported_agent"
    assert skip_rows[str(unsupported)]["recoverable_in_principle"] is True
    assert "claude-agent-acp" in skip_rows[str(unsupported)]["detail"]
    assert skip_rows[str(unrecorded)]["reason"] == "no_llm_recording"
    assert skip_rows[str(unrecorded)]["recoverable_in_principle"] is False


@pytest.mark.asyncio
async def test_batch_of_many_survives_a_minority_of_unsupported_runs(tmp_path):
    """The #1083 shape: 190 of 200 supported must still be continued."""
    total, unsupported_count = 20, 2
    folders = [
        write_run_folder(
            tmp_path / f"run-{idx}",
            exchanges=[exchange(completion(content="a"))],
            agent="openhands" if idx >= unsupported_count else "claude-agent-acp",
            error_category="timeout",
        )
        for idx in range(total)
    ]

    async def runner(folder: Path, **kwargs):
        load_run_folder(folder)
        return ContinueResult(
            rollout_dir=folder / "continued",
            rewards=None,
            error=None,
            n_recorded=1,
            n_live=1,
            divergences=0,
        )

    summary = summarize_batch(
        await continue_batch(
            folders,
            concurrency=8,
            tasks_dir=None,
            model=None,
            timeout=None,
            output_dir=None,
            runner=runner,
        )
    )
    assert summary["succeeded"] == total - unsupported_count
    assert summary["skipped"] == unsupported_count
    assert summary["failed"] == 0


def test_discovery_reports_unsupported_timeout_candidates(tmp_path):
    """Discovery must not swallow the runs it drops for being unsupported."""
    supported = write_run_folder(
        tmp_path / "oh",
        exchanges=[exchange(completion(content="a"))],
        error_category="timeout",
    )
    unsupported = write_run_folder(
        tmp_path / "acp",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
        error_category="timeout",
    )
    # A *finished* run from another agent was never a continue candidate and
    # must stay quiet — otherwise a mixed jobs dir drowns the operator in noise.
    write_run_folder(
        tmp_path / "acp-done",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
        error_category=None,
    )

    seen: list[tuple[Path, str]] = []
    folders = discover_timeout_run_folders(
        tmp_path, on_skip=lambda folder, exc: seen.append((folder, exc.reason_code))
    )

    assert folders == [supported]
    assert seen == [(unsupported, "unsupported_agent")]


def test_batch_cli_reports_skips_and_exits_zero(tmp_path):
    """`continue-batch` over an all-unsupported tree is not a batch failure."""
    write_run_folder(
        tmp_path / "acp",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
        error_category="timeout",
    )

    res = CliRunner().invoke(app, ["eval", "continue-batch", str(tmp_path)])

    assert res.exit_code == 0, res.output
    assert "Skipping 1 timeout run(s)" in res.output
    payload = json.loads(res.output[res.output.index("{") :])
    assert payload["skipped"] == 1
    assert payload["failed"] == 0
    assert payload["skips"][0]["reason"] == "unsupported_agent"
    assert "claude-agent-acp" in payload["skips"][0]["detail"]


def test_single_run_continue_still_fails_loudly(tmp_path):
    """An explicitly named unsupported run must keep failing — not be skipped."""
    folder = write_run_folder(
        tmp_path / "acp",
        exchanges=[exchange(completion(content="a"))],
        agent="claude-agent-acp",
        error_category="timeout",
    )

    res = CliRunner().invoke(app, ["eval", "continue", str(folder)])

    assert res.exit_code == 1
    assert "claude-agent-acp" in res.output
    assert "openhands" in res.output
