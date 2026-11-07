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
    assert (
        "Outcomes: 1 scored (1 passed, 0 failed), 1 unscored (1 verifier error), "
        "2 errored"
    ) in out
    assert (
        "1 unscored: verifier plugin trust: test.sh installs a pytest plugin where "
        "the agent could write (task problem): powerlifting"
    ) in out
    assert f"next: `bench check {tmp_path / 'tasks' / 'powerlifting'}`" in out
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
    assert "Score: 1/4" in out and f"View:      bench view {job_dir}" in out


def test_the_summary_names_the_explain_id_of_each_cause(job, monkeypatch):
    """Round PY8's D2: each cause shown has a `bench explain` page, and the
    summary says which to read (Rust's ids where the meaning is the same)."""
    job_dir, result = job
    out = _render(monkeypatch, result, job_dir)
    assert "`bench explain verifier_plugin_trust` says more (also R213, R203)" in out
    assert out.index("says more") > out.index("idle watchdog")
    assert shared.explain_pointer([]) is None
    assert shared.explain_pointer(["a", "b", "a", "c", "d"]) == (
        "`bench explain a` says more (also b, c, ...)"
    )


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


def test_no_tokens_is_none_only_when_every_trial_reported_usage(job, monkeypatch):
    """Review finding: one trial reporting 0 tokens and the rest reporting
    nothing read "none: no model tokens were used"."""
    job_dir, result = job
    for trial in result.results.values():
        trial.cost_usd = trial.total_tokens = None
    result.results["hello"].total_tokens = 0
    out = _render(monkeypatch, result, job_dir)
    assert "no model tokens were used" not in out
    assert (
        "Cost:      not reported (3 of 4 trials reported no usage); the others "
        "used no model tokens"
    ) in out


def test_the_summary_reads_only_the_files_it_reports_from(job, monkeypatch):
    """Review finding: the summary read the job's evaluation.json once per
    trial and every trial's result.json, passed ones included."""
    job_dir, result = job
    reads: list[str] = []
    read_text = Path.read_text

    def counting(self, *args, **kwargs):
        reads.append(self.parent.name if self.name == "result.json" else self.name)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    monkeypatch.setattr(shared, "console", Console(file=io.StringIO(), width=240))
    shared._report_outcomes(result, job_dir)
    assert reads.count("evaluation.json") == 1
    assert "fine__abcd1234" not in reads


def test_a_trial_with_no_verdict_and_no_error_is_errored_on_both_lines(
    tmp_path, monkeypatch
):
    """Review finding: the Score line counts a trial with no reward and no
    error as an error; the Outcomes line called it unscored."""
    job_dir = tmp_path / "jobs" / "run"
    job_dir.mkdir(parents=True)
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        job_dir=job_dir,
        results={"cut": _trial(job_dir, "cut")},
    )
    assert result.results["cut"].score_outcome == "errored"
    out = _render(monkeypatch, result, job_dir)
    assert "errors=1" in out
    assert (
        "Outcomes: 0 scored (0 passed, 0 failed), 0 unscored, 1 errored (1 with no "
        "verdict)"
    ) in out
    assert "1 errored: no verdict was recorded" in out


def test_a_trial_whose_cause_is_unknown_keeps_the_score_line_s_bucket(
    tmp_path, monkeypatch
):
    """Second review finding: a trial the Score line counts as errored but
    whose cause cannot be read (an unreadable or missing result.json) was
    counted a scored failure, so the two lines disagreed. It now lands in
    the same bucket, only without an explanation under it."""
    import benchflow.failures as failures

    job_dir = tmp_path / "jobs" / "run"
    job_dir.mkdir(parents=True)
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        job_dir=job_dir,
        results={"cut": _trial(job_dir, "cut")},
    )
    monkeypatch.setattr(failures, "cause_of", lambda *a, **k: None)
    out = _render(monkeypatch, result, job_dir)
    assert (
        "Outcomes: 0 scored (0 passed, 0 failed), 0 unscored, 1 errored (1 with no "
        "verdict)"
    ) in out
    assert "next:" not in out  # no cause to explain, so no group under the line


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


def test_failures_the_verifier_scored_zero_are_told_apart(tmp_path, monkeypatch):
    """A benchmark's lesson (seen in a 0.10.0rc1 acceptance run): a test script that wrote
    no reward scored 0 as a failure, and only result.json's warnings said so.
    Such a record (rc1 wrote it) is still told apart from a wrong answer."""
    from benchflow._utils.scoring import score_summary_fields

    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    zero = "Verifier scored 0: No reward file found at <reachable:verifier/reward.txt>"
    results = {
        "no-reward": _trial(job, "no-reward", rewards={"reward": 0.0}, warnings=[zero]),
        "wrong": _trial(job, "wrong", rewards={"reward": 0.0}),
        "right": _trial(job, "right", rewards={"reward": 1.0}),
    }
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_outcomes(SimpleResult(results), job)
    text = rec.file.getvalue()
    assert "Outcomes: 3 scored (1 passed, 2 failed)" in text
    assert (
        "1 of the failed: the verifier scored them 0 on its own evidence, not on "
        "an answer (a failure, not an error): no-reward" in text
    )
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    assert score_summary_fields(rows)["verifier_scored_zero"] == 1


def test_grading_timeouts_are_named_apart(tmp_path, monkeypatch):
    """A benchmark's lesson: tests still running at the verifier's
    time limit score 0, and a slow grading machine does that too, so the
    summary names them and says how to rerun only the verifier."""
    from benchflow._utils.scoring import score_summary_fields

    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    zero = "Verifier scored 0: verifier timed out after 600.0s"
    results = {
        "slow": _trial(
            job,
            "slow",
            rewards={"reward": 0.0},
            warnings=[zero],
            verifier_timeout_info={"timeout_budget_sec": 600.0, "scored": True},
        ),
        "right": _trial(job, "right", rewards={"reward": 1.0}),
    }
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_outcomes(SimpleResult(results), job)
    text = rec.file.getvalue()
    assert (
        "1 of the failed: the tests were still running at the verifier's time "
        "limit, so they scored 0 (slow code, or a slow grading machine): slow"
    ) in text
    assert "bench regrade <job dir>" in text
    assert "on its own evidence" not in text
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    summary = score_summary_fields(rows)
    assert summary["grading_timeouts"] == 1
    assert summary["verifier_scored_zero"] == 1


def test_no_usable_reward_is_a_verifier_error_that_counts_as_a_failure(
    tmp_path, monkeypatch
):
    """A final verify with no usable reward is now the verifier's
    error, named as such; the error-excluding score still counts it failed."""
    from benchflow._utils.scoring import NO_USABLE_REWARD, score_summary_fields

    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    error = (
        NO_USABLE_REWARD + "Failed to parse rewards from text file verifier/reward.txt"
    )
    results = {
        "no-reward": _trial(job, "no-reward", rewards=None, verifier_error=error),
        "right": _trial(job, "right", rewards={"reward": 1.0}),
    }
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_outcomes(SimpleResult(results), job)
    text = rec.file.getvalue()
    assert (
        "Outcomes: 1 scored (1 passed, 0 failed), 1 unscored (1 refused), 0 errored"
        in text
    )
    assert "the verifier wrote no usable reward (a failure in every score)" in text
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    summary = score_summary_fields(rows)
    assert summary["unusable_rewards"] == 1
    assert summary["score_excl_errors_ratio"] == 0.5


def test_a_reviewed_trial_with_no_usable_reward_still_counts_as_a_failure():
    """A rubric-reviewed trial's review refuses a trial whose verifier left no
    reward, and its scoring block keeps the error as ``ValueError: ...``
    (review/automatic.py): the trial is still a failure in every score."""
    from benchflow._utils.scoring import (
        NO_USABLE_REWARD,
        VERIFIER_FAILED,
        classify_verifier_error,
        is_unusable_reward,
        score_summary_fields,
    )

    error = "ValueError: " + NO_USABLE_REWARD + "verifier/reward.txt is empty"
    reviewed = {
        "rewards": None,
        "verifier_error": error,
        "scoring": {"status": "error", "error": error},
    }
    assert is_unusable_reward(error)
    assert classify_verifier_error(error) == VERIFIER_FAILED
    summary = score_summary_fields([reviewed, {"rewards": {"reward": 1.0}}])
    assert (summary["unusable_rewards"], summary["score_excl_errors_ratio"]) == (1, 0.5)


class SimpleResult:
    def __init__(self, results):
        self.results = results
