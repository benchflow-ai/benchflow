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
    assert (
        "$0.75, 1,000 tokens (3 of 4 trials reported no usage; 2 of 4 trials "
        "reported no cost)"
    ) in out
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
        trial.usage_source = "proxy"  # a counted zero
    out = _render(monkeypatch, result, job_dir)
    assert "Cost:      none: no model tokens were used" in out


def test_a_zero_nothing_counted_is_no_usage_not_no_model_tokens(tmp_path, monkeypatch):
    """EN-374, X2 on 2026-11-13: 0.9.2 recorded a native Claude Code turn cut
    off at its 15 s cap (one tool call, 27,017 tokens in Claude Code's session
    log) as 0 tokens from an unavailable source, and this line said "none: no
    model tokens were used". A zero nothing counted is no usage."""
    job_dir = tmp_path / "jobs" / "run"
    job_dir.mkdir(parents=True)
    cut = _trial(
        job_dir,
        "timeout",
        error="Agent timed out after 15s",
        agent_result={
            "n_tool_calls": 1,
            "n_input_tokens": 0,
            "n_output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": None,
            "usage_source": "unavailable",
        },
    )
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        job_dir=job_dir,
        results={"timeout": cut},
    )
    out = _render(monkeypatch, result, job_dir)
    assert "no model tokens" not in out
    assert "Cost:      not reported" in out


def test_a_known_cost_does_not_hide_unknown_tokens(job, monkeypatch):
    """EN-645 (X4 cell *-harvey-lab-harness-acp-mimo-hello-default-run): every
    trial priced, none with tokens, and the Cost line read "$0.00" alone."""
    job_dir, result = job
    for trial in result.results.values():
        trial.cost_usd = 0.0008903664
        trial.total_tokens = None
    out = _render(monkeypatch, result, job_dir)
    assert "Cost:      $0.00 (4 of 4 trials reported no usage)" in out


def test_no_tokens_is_none_only_when_every_trial_reported_usage(job, monkeypatch):
    """Review finding: one trial reporting 0 tokens and the rest reporting
    nothing read "none: no model tokens were used"."""
    job_dir, result = job
    for trial in result.results.values():
        trial.cost_usd = trial.total_tokens = None
    result.results["hello"].total_tokens = 0
    result.results["hello"].usage_source = "proxy"  # a counted zero
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
    assert "Outcomes: 0 scored (0 passed, 0 failed), 0 unscored, 1 errored" in out
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
    assert "Outcomes: 0 scored (0 passed, 0 failed), 0 unscored, 1 errored" in out
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


NO_REWARD = (
    "verifier wrote no usable reward: No reward file found at "
    "<reachable:verifier/reward.txt> or <reachable:verifier/reward.json>"
)


def test_no_usable_reward_is_named_a_failure_in_every_score(tmp_path, monkeypatch):
    """The outcome lines name a verifier that wrote no usable reward, which
    every score counts as a failure, not a crashed verifier (EN-376, as
    0.10's 2aa35a977)."""
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    results = {
        "no-reward": _trial(
            job,
            "no-reward",
            rewards=None,
            verifier_error=NO_REWARD,
            verifier_error_category="verifier_failure",
        ),
        "right": _trial(job, "right", rewards={"reward": 1.0}),
    }
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_outcomes(
        EvaluationResult(job_name="run", config=EvaluationConfig(), results=results),
        job,
    )
    text = rec.file.getvalue()
    assert "Outcomes: 1 scored (1 passed, 0 failed), 1 unscored, 0 errored" in text
    assert (
        "1 unscored: the verifier wrote no usable reward (a failure in every "
        "score) (task problem): no-reward"
    ) in text
    assert cause_of(results["no-reward"]).key == "verifier_no_usable_reward"


def test_bench_run_scores_a_missing_reward_as_summary_json_does(tmp_path, monkeypatch):
    """`bench run` (the oracle) over a pass, a fail and a verifier that wrote
    no reward prints the scored-only figure summary.json's score_excl_errors
    holds: 1/3 (33.3%), not 1/2 (R093 M2; 0.10's rule)."""
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from benchflow.cli.main import app

    rewards = {
        "pass-task": {"reward": 1.0},
        "fail-task": {"reward": 0.0},
        "noreward-task": None,
    }
    tasks = tmp_path / "tasks"
    for name in rewards:
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )

    async def create(config):
        name = config.task_path.name
        rollout = AsyncMock()
        rollout.run = AsyncMock(
            return_value=RolloutResult(
                task_name=name,
                rewards=rewards[name],
                verifier_error=None if rewards[name] else NO_REWARD,
                verifier_error_category=None if rewards[name] else "verifier_failure",
            )
        )
        return rollout

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    jobs = tmp_path / "jobs"
    run = CliRunner().invoke(
        app, ["run", str(tasks), "--sandbox", "docker", "--jobs-dir", str(jobs)]
    )
    assert run.exit_code == 1, run.output  # a verifier error
    (summary_file,) = jobs.glob("*/summary.json")
    summary = json.loads(summary_file.read_text())
    assert (summary["unusable_rewards"], summary["score_excl_errors"]) == (1, "33.3%")
    shown = " ".join(run.output.split())
    assert "scored only: 1/3 (33.3%)" in shown, shown
    assert "the verifier wrote no usable reward (a failure in every score)" in shown


def test_codex_trials_say_their_tokens_are_generating_calls_only(job, monkeypatch):
    """EN-648 (X4 cells *-codex-acp-acp-codex-wire-*-run): a Codex trial
    counted from its session log leaves out its generate=false warm-up, and
    the Cost line says so instead of printing a bare total."""
    from benchflow._utils.result_metadata import codex_warmup

    job_dir, result = job
    for trial in result.results.values():
        trial.usage_details = {"warmup": codex_warmup()}
    out = _render(monkeypatch, result, job_dir)
    assert (
        "4 trial(s) count generating calls only: Codex's generate=false warm-up "
        "is not observed or priced, so the total is a floor"
    ) in out


def test_a_cache_write_floor_is_worded_by_its_own_cause(job, monkeypatch):
    """RUSAGE (#261 L2, merge level): with #259 in, the closing line counted
    only warm-ups, so a GPT-6 trial whose writes went unreported on a route
    that prices them showed no floor at all."""
    job_dir, result = job
    writes = {
        "tokens": None,
        "named_tokens": 0,
        "inside_input": True,
        "models": ["openai/gpt-6.1-sol"],
        "priced_models": ["openai/gpt-6.1-sol"],
        "cost_floor": True,
    }
    for trial in result.results.values():
        trial.usage_details = {"cache_writes": writes}
    out = _render(monkeypatch, result, job_dir)
    assert (
        "2 trial(s) left out cache writes their answers did not report, on a "
        "route that prices them, so the total is a floor"
    ) in out
    assert "Codex" not in out
