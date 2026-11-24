"""An attempt that ended with no result.json and no receipt still counts.

Guards 0.10.0rc6's fix: a host reset mid-turn left the attempt's folder with
its config.json and the model proxy's live log (each finished call's
provider answer, usage included) but no result.json or cancellation
receipt. ``evaluation._recorded_attempts`` only counted such folders for a
budget's rollout count, so summary.json's ``spend_all_attempts`` left the
attempt out (a dogfood run: 1 attempt, 22,947 tokens, beside a saved HTTP-200
answer of 11,407 tokens) and nothing warned. The attempt is now counted
from its logs, as a stopped attempt whose cost is a floor, or, with no log,
named in ``stopped_attempts_without_usage``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from benchflow import Budget
from benchflow.budget import CANCELLED_BUDGET_RECORD, BudgetGuard
from benchflow.evaluation import _recorded_attempts, _spend_all_attempts

MODEL = "deepseek/deepseek-flash"
FINISHED = {"total_tokens": 22_947, "cost_usd": 0.003982884}


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _proxy_row(prompt: int, output: int, figure: float) -> dict:
    """One line of the proxy's live llm_trajectory.jsonl: a DeepSeek answer
    with its usage, and the gateway's list-price figure for it."""
    return {
        "duration_ms": 4000.0,
        "metadata": {
            "request_model": MODEL,
            "provider_model": "deepseek-flash",
            "call_type": "anthropic_messages",
            "call_purpose": "agent",
            "response_cost": figure,
        },
        "request": {
            "timestamp": "2026-11-17T08:00:10",
            "body": {"model": MODEL, "max_tokens": 32000},
        },
        "response": {
            "status_code": 200,
            "timestamp": "2026-11-17T08:00:14",
            "body": {
                "model": "deepseek-flash",
                "choices": [{"finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "total_tokens": prompt + output,
                },
            },
        },
    }


def _attempt(job: Path, name: str, *, rows: list[dict] | None = None) -> Path:
    folder = job / name
    _write(
        folder / "config.json",
        {"agent": "claude-agent-acp", "model": MODEL, "task_path": "/tasks/state"},
    )
    if rows is not None:
        log = folder / "trajectory" / "llm_trajectory.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return folder


def _finished(job: Path, name: str) -> Path:
    folder = _attempt(job, name)
    _write(
        folder / "result.json",
        {
            "task_name": "state",
            "agent": "claude-agent-acp",
            "model": MODEL,
            "rewards": {"reward": 1.0},
            "agent_result": {**FINISHED, "usage_source": "proxy"},
            "finished_at": "2026-11-17T08:20:00+00:00",
        },
    )
    return folder


@pytest.fixture
def job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "J-host"
    _finished(job, "state__f30185ef")
    return job


def test_a_host_reset_attempt_is_counted_from_its_proxy_log(job, caplog):
    """S1: two finished calls in the stopped attempt's live log."""
    _attempt(
        job,
        "state__fe029a02",
        rows=[_proxy_row(10_000, 1_000, 0.002), _proxy_row(10_000, 1_000, 0.002)],
    )
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["attempts"] == 1
    assert spend["stopped_attempts"] == 1
    assert spend["attempts_usage_salvaged"] == 1
    assert spend["stopped_attempts_without_usage"] == 0
    assert spend["tokens"] == FINISHED["total_tokens"] + 22_000
    assert spend["cost_usd"] == pytest.approx(FINISHED["cost_usd"] + 0.004, abs=1e-9)
    # A call in flight at the reset has no line: the total is a floor.
    assert spend["attempts_cost_incomplete"] == 1
    assert "counted from the logs they left" in caplog.text


def test_an_attempt_with_no_log_is_named_not_dropped(job):
    """S2: nothing on disk says what it spent; it is a stopped attempt with
    no usage, never absent."""
    _attempt(job, "state__fe029a02")
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["attempts"] == 1
    assert spend["stopped_attempts"] == 1
    assert spend["stopped_attempts_without_usage"] == 1
    assert spend["attempts_usage_salvaged"] == 0
    assert spend["tokens"] == FINISHED["total_tokens"]


def test_a_receipt_is_counted_once_and_its_log_is_not_read(job):
    """S3: a stopped attempt that wrote its receipt is counted by it."""
    folder = _attempt(job, "state__fe029a02", rows=[_proxy_row(10_000, 1_000, 0.002)])
    _write(
        folder / CANCELLED_BUDGET_RECORD,
        {
            "version": 1,
            "timing": {"total": 30.0, "verifier_sandbox_total": 0.0},
            "agent_result": {"total_tokens": 500, "cost_usd": 0.001},
        },
    )
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["stopped_attempts"] == 1
    assert spend["attempts_usage_salvaged"] == 0
    assert spend["tokens"] == FINISHED["total_tokens"] + 500


def test_folders_inside_a_trial_are_not_attempts(job):
    """S4: a branch child's folder in a finished trial keeps its own log;
    it is part of that trial, not an attempt of the job."""
    trial = job / "state__f30185ef"
    _attempt(
        trial / "branches" / "f1" / "children",
        "n1",
        rows=[_proxy_row(10_000, 1_000, 0.002)],
    )
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["stopped_attempts"] == 0
    assert spend["attempts_usage_salvaged"] == 0
    assert spend["tokens"] == FINISHED["total_tokens"]


def test_a_resumed_budget_counts_the_salvaged_spend_once(job):
    """S5: the budget a resume seeds counts the attempt once, with what it
    spent, not as a rollout with no spend beside it."""
    _attempt(job, "state__fe029a02", rows=[_proxy_row(10_000, 1_000, 0.002)])
    attempts, unrecorded = _recorded_attempts(job)
    assert unrecorded == 0
    guard = BudgetGuard(Budget(max_cost_usd=1.0))
    guard.seed(attempts, interrupted=unrecorded)
    assert guard.rollouts == 2
    assert guard.cost_usd == pytest.approx(FINISHED["cost_usd"] + 0.002, abs=1e-9)


def test_an_interrupted_trial_s_branch_child_is_not_salvaged(job):
    """Review of #383: salvage reads only attempts directly below the job. An
    interrupted trial whose branch child is unfinished too: the trial's own
    proxy log counts, the child's (inside it) does not."""
    trial = _attempt(job, "state__fe029a02", rows=[_proxy_row(10_000, 1_000, 0.002)])
    _attempt(
        trial / "branches" / "f1" / "children",
        "n7",
        rows=[_proxy_row(50_000, 5_000, 0.01)],
    )
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["attempts_usage_salvaged"] == 1
    assert spend["tokens"] == FINISHED["total_tokens"] + 11_000
