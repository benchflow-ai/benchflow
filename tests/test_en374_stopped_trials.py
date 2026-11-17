"""A stopped or failed trial keeps what it spent, in its own folder.


X2's live repro (rc3, 2026-11-13): Codex ACP interrupted with SIGINT exited
130 and tore its sandbox down, but left no result.json; summary.json said 0
while Codex's raw log held 12,960 tokens. A cancelled trial (Ctrl-C, SIGTERM,
a budget stop) now writes its result.json with the tokens and cost cleanup
counted, marked ``interrupted`` so readers still see an interrupted trial and
a resume runs it again; its budget receipt and the job's ``budget.spent``
carry them too.

X1's: a lost sandbox made a new error trial id; the original trial
folder, which held the captured usage, was orphaned. The trial keeps its
folder and id, with its error and its salvaged tokens.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.test_en374_salvage import (
    KILLED_MID_TURN,
    KILLED_TOKENS,
    _rollout,
    _stream_log,
)

# X2's Ctrl-C trial: one Codex call before the SIGINT, 12,960 tokens.
CTRL_C_CALL = {
    "input_tokens": 12894,
    "cached_input_tokens": 11264,
    "cache_write_input_tokens": 0,
    "output_tokens": 66,
    "reasoning_output_tokens": 0,
    "total_tokens": 12960,
}


def _stopped_after_cleanup(rollout):
    """A lifecycle that ran its cleanup (as the real one's finally does),
    then was cancelled by Ctrl-C."""

    async def run_with_scoring():
        await rollout._cleanup_before_teardown()
        raise asyncio.CancelledError

    return run_with_scoring


@pytest.mark.asyncio
async def test_ctrl_c_writes_the_trial_s_spend(tmp_path, monkeypatch):
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_paths import (
        interrupted_trial_results,
        iter_task_result_paths,
        load_task_results,
    )
    from benchflow.budget import Budget, BudgetGuard
    from benchflow.jobs import _interrupted
    from tests.test_codex_acp_tokens import _codex_log

    home = tmp_path / "sandbox-home"
    _codex_log(home, [CTRL_C_CALL])
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    (trial / "config.json").write_text(json.dumps({"agent": "codex-acp"}))
    monkeypatch.setattr(rollout, "_run_with_scoring", _stopped_after_cleanup(rollout))
    guard = BudgetGuard(Budget(max_tokens=10**9))

    with guard.track_trial("task"), pytest.raises(asyncio.CancelledError):
        await rollout.run()

    data = json.loads((trial / "result.json").read_text())
    assert data["interrupted"] is True
    assert data["error"].startswith("Interrupted:")
    assert data["rewards"] is None
    agent = data["agent_result"]
    assert agent["total_tokens"] == 12960
    assert agent["usage_source"] == "session_log"
    usd = (12894 - 11264) * 2e-6 + 11264 * 0.1e-6 + 66 * 10e-6
    assert agent["cost_usd"] == pytest.approx(usd)
    assert agent["cost_source"] == "list_price_estimate"
    receipt = json.loads((trial / "budget-cancelled.json").read_text())
    assert receipt["reason"] == "interrupted"
    assert receipt["agent_result"]["total_tokens"] == 12960
    assert receipt["agent_result"]["cost_usd"] == pytest.approx(usd)
    spent = guard.spent()
    assert spent["tokens"] == 12960
    assert spent["cost_usd"] == pytest.approx(usd)

    # Only the result: no training records for the job's exports to gather.
    assert not (trial / "results.jsonl").exists()
    assert not (trial / "trainer").exists()
    assert not (trial / "result-interrupted.json").exists()
    # Readers still see an interrupted trial, and a resume runs it again.
    job = trial.parent
    assert trial / "result.json" not in iter_task_result_paths(job)
    assert load_task_results(job, include_terminal_errors=True) == {}
    assert trial in _interrupted(job)
    # summary.json's usage counts what it spent.
    stopped = interrupted_trial_results(job)
    assert list(stopped) == [trial.name]
    summary = usage_summary(stopped)
    assert summary["total_tokens"] == 12960
    assert summary["total_cost_usd"] == pytest.approx(usd)


@pytest.mark.asyncio
async def test_a_resumed_budget_counts_the_stopped_trial_s_receipt(
    tmp_path, monkeypatch
):
    """A resumed job seeds its budget from the stopped trial's receipt: its
    tokens and USD count once."""
    from benchflow.budget import Budget, BudgetGuard
    from benchflow.evaluation import _recorded_attempts
    from tests.test_codex_acp_tokens import _codex_log

    home = tmp_path / "sandbox-home"
    _codex_log(home, [CTRL_C_CALL])
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="codex-acp", harness="acp", home=home
    )
    rollout._usage_report_missing = True
    (trial / "config.json").write_text(json.dumps({"agent": "codex-acp"}))
    monkeypatch.setattr(rollout, "_run_with_scoring", _stopped_after_cleanup(rollout))
    with pytest.raises(asyncio.CancelledError):
        await rollout.run()

    attempts, without_receipt = _recorded_attempts(trial.parent)
    guard = BudgetGuard(Budget(max_tokens=10**9))
    guard.seed(attempts, interrupted=without_receipt)
    assert guard.spent()["tokens"] == 12960


@pytest.mark.asyncio
async def test_a_failed_run_keeps_its_own_trial_id(tmp_path, monkeypatch):
    """the run raised after its sandbox was lost (its teardown could
    not remove what it staged). The trial's result goes into its own folder,
    with the tokens and cost the host-side stream log holds; no new trial id."""
    rollout, trial = _rollout(
        tmp_path,
        monkeypatch,
        agent="claude-agent-acp",
        harness="native",
        home=tmp_path / "gone",
    )
    _stream_log(trial, [KILLED_MID_TURN])
    rollout._usage_report_missing = True
    rollout._env = None
    await rollout._teardown_finalize_usage()

    result = rollout.result_after_failure(RuntimeError("container is not running"))

    assert result is not None
    assert result.rollout_name == trial.name
    data = json.loads((trial / "result.json").read_text())
    assert data["rollout_name"] == trial.name
    assert data["error"].startswith("Unexpected: ")
    agent = data["agent_result"]
    assert agent["total_tokens"] == KILLED_TOKENS["total_tokens"]
    assert agent["usage_source"] == "stream_log"
    assert agent["cost_source"] == "list_price_estimate"
    assert list(trial.parent.rglob("result.json")) == [trial / "result.json"]
