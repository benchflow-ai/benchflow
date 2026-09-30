"""The end-of-run summary splits trials by outcome and cause, with a next step.

Guards the dx/errors fix for summaries that said only ``✗ Score: 0/1 (0.0%),
errors=1`` (the usage-limit smoke) or ``errors=0 verifier-errors=1`` (the
unscorable powerlifting-coef-calc oracle): nothing said why, whose fault,
what to do, what it cost or how long it took.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import benchflow.cli._shared as shared
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.failures import cause_of
from benchflow.models import RolloutResult

LIMIT = (
    "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): 7-day "
    "window, resets 2026-10-03 19:00 UTC (You've hit your weekly limit · resets "
    "Oct 3, 7pm (UTC))"
)
PLUGIN = (
    "verifier crashed: PluginGuardLoadError: pytest plugin pytest-json-ctrf 0.3.5 "
    "was installed after the agent stopped, by the verifier into /root/.venv, "
    "where the agent could write; ..."
)


def _trial(job: Path, name: str, **fields) -> RolloutResult:
    folder = job / f"{name}__abcd1234"
    folder.mkdir(parents=True)
    payload = {"task_name": name, "rollout_name": folder.name, **fields}
    (folder / "result.json").write_text(json.dumps(payload))
    return RolloutResult.from_dict(payload, rollout_dir=folder)


@pytest.fixture
def job(tmp_path: Path) -> tuple[Path, EvaluationResult]:
    job = tmp_path / "jobs" / "run"
    tasks = tmp_path / "tasks"
    for name in ("hello", "powerlifting", "quiet", "fine"):
        (tasks / name).mkdir(parents=True)
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps({"tasks_dir": str(tasks)}))
    (job / "summary.json").write_text(
        json.dumps({"usage_limit": {"not_started": ["t5", "t6"]}})
    )
    results = {
        "fine": _trial(
            job, "fine", rewards={"reward": 1.0}, cost_usd=0.25, total_tokens=1000
        ),
        "hello": _trial(
            job,
            "hello",
            error=LIMIT,
            error_category="usage_limit",
            usage_limit_info={
                "login": "CLAUDE_CODE_OAUTH_TOKEN (environment)",
                "window": "7-day",
                "resets_at": "2026-10-03T19:00:00+00:00",
                "detail": "You've hit your weekly limit · resets Oct 3, 7pm (UTC)",
            },
        ),
        "powerlifting": _trial(
            job,
            "powerlifting",
            verifier_error=PLUGIN,
            verifier_error_category="verifier_failure",
        ),
        "quiet": _trial(
            job,
            "quiet",
            error="Agent idle for 600s with no new tool call, message, or thought",
            error_category="idle_timeout",
            cost_usd=0.5,
        ),
    }
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=4,
        passed=1,
        errored=2,
        verifier_errored=1,
        elapsed_sec=130.2,
        job_dir=job,
        results=results,
    )
    return job, result


def _render(monkeypatch, result, job_dir) -> str:
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(result, job_dir)
    return rec.file.getvalue()


def test_outcomes_by_cause_with_whose_fault_and_what_next(job, monkeypatch, tmp_path):
    job_dir, result = job
    out = _render(monkeypatch, result, job_dir)
    assert "Outcomes: 1 scored (1 passed, 0 failed), 1 unscored, 2 errored" in out
    assert (
        "1 unscored: verifier plugin trust: test.sh installs a pytest plugin where "
        "the agent could write (task problem): powerlifting"
    ) in out
    assert f"next: `bench tasks check {tmp_path / 'tasks' / 'powerlifting'}`" in out
    assert (
        "1 errored: usage limit on login CLAUDE_CODE_OAUTH_TOKEN (environment), "
        "7-day window, resets 2026-10-03 19:00 UTC (setup problem): hello"
    ) in out
    assert f"then `bench eval resume {job_dir}`" in out
    assert (
        "1 errored: the agent went silent and the idle watchdog stopped it "
        "(agent problem): quiet"
    ) in out
    assert "2 not started: the login's usage limit stopped the job" in out
    assert "$0.75, 1,000 tokens (2 of 4 trials reported no cost)" in out
    assert "Time:      2m 10s" in out
    # The existing lines are still there.
    assert "Score: 1/4" in out and f"View:      bench eval view {job_dir}" in out


def test_no_cost_says_so(job, monkeypatch):
    job_dir, result = job
    for trial in result.results.values():
        trial.cost_usd = None
    out = _render(monkeypatch, result, job_dir)
    assert "Cost:      not reported in USD (subscription logins" in out
    assert "report tokens only), 1,000 tokens" in out
    for trial in result.results.values():
        trial.total_tokens = 0
    out = _render(monkeypatch, result, job_dir)
    assert "Cost:      none: no model tokens were used" in out


def test_a_scored_trial_has_no_cause(job):
    _, result = job
    assert cause_of(result.results["fine"]) is None
    cause = cause_of(result.results["hello"])
    assert cause is not None and cause.fault == "setup"


def test_results_without_trials_print_no_outcomes(monkeypatch):
    from types import SimpleNamespace

    out = _render(
        monkeypatch,
        SimpleNamespace(passed=0, total=1, errored=1, verifier_errored=0, score=0.0),
        None,
    )
    assert "Outcomes:" not in out and "Score: 0/1" in out
