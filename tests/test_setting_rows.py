"""Rows keyed by what ran: route, harness version, BenchFlow version, effort sent.

An evaluation audit found rows that mixed routes, CLI versions or BenchFlow
versions under one label: one model's row held cells from two providers, two
Claude Code releases and two BenchFlow releases. Every reader pooled them
into one figure. ``benchflow.setting_groups`` keys each trial by what its
records say ran, and a job whose trials differ reads as mixed settings.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import benchflow as bf
from benchflow.setting_groups import setting_rows
from tests import _run_records as rr

LABEL = "anthropic/claude-opus-5"
SERVED = "claude-opus-5-20260801"
DIRECT, BEDROCK = "anthropic/claude-opus-5", "bedrock/us.anthropic.claude-opus-5-v1:0"


def _calls(route: str, effort: str | None = "high", n: int = 3) -> list[dict]:
    return [
        rr.anthropic_call(model=LABEL, served=SERVED, route=route, effort=effort)
        for _ in range(n)
    ]


def _resumed_job(root: Path) -> Path:
    """One job: three tasks run by BenchFlow 0.10.0 with Claude Code 2.1.257
    on the direct route, then resumed by 0.10.1, whose Claude Code is
    2.1.260, on Bedrock. One trial of each run errored before its first
    model call."""
    job = rr.evaluation_job(root / "job", [f"t{i}" for i in range(6)], model=LABEL)
    first = rr.receipt(cli=(rr.CLAUDE_CLI, "2.1.257"))
    second = rr.receipt(cli=(rr.CLAUDE_CLI, "2.1.260"))
    for i, (route, harness, version) in enumerate(
        [
            (DIRECT, first, "0.10.0"),
            (DIRECT, first, "0.10.0"),
            (None, first, "0.10.0"),
            (BEDROCK, second, "0.10.1"),
            (BEDROCK, second, "0.10.1"),
            (None, second, "0.10.1"),
        ]
    ):
        rr.trial(
            job,
            f"t{i}",
            i,
            reward=None if route is None else float(i % 2),
            model=LABEL,
            effort="high",
            calls=_calls(route) if route else None,
            harness_versions=harness,
            benchflow_version=version,
            error="sandbox lost before the agent started" if route is None else None,
            error_category="infra_failure" if route is None else None,
        )
    return job


def test_two_routes_and_two_cli_versions_under_one_label_are_two_rows(tmp_path):
    job = bf.load_job(_resumed_job(tmp_path))
    assert {t.model for t in job.trials} == {LABEL}  # one label for all six
    settings = setting_rows(job.trials)
    rows = settings.rows
    assert [(r.setting.route, r.setting.benchflow_version) for r in rows] == [
        (DIRECT, "0.10.0"),
        (BEDROCK, "0.10.1"),
    ]
    assert [r.setting.harness_versions for r in rows] == [
        (f"{rr.CLAUDE_ADAPTER}@0.88.0", f"{rr.CLAUDE_CLI}@2.1.257"),
        (f"{rr.CLAUDE_ADAPTER}@0.88.0", f"{rr.CLAUDE_CLI}@2.1.260"),
    ]
    # The trial that errored before its first call stays in its launch's
    # row, where it counts 0: neither row loses a failure.
    for row in rows:
        assert len(row.trials) == 3
        assert row.setting.model == SERVED and row.setting.model_source == "served"
        figures = row.score()
        assert (figures.n, figures.outcomes["errored"]) == (3, 1)
    assert settings.mixed
    assert settings.differ() == [
        ("harness version", ["claude-agent-acp 0.88.0 + claude-code 2.1.257",
                             "claude-agent-acp 0.88.0 + claude-code 2.1.260"]),
        ("route", [DIRECT, BEDROCK]),
        ("BenchFlow version", ["0.10.0", "0.10.1"]),
    ]  # fmt: skip
    warning = settings.warning() or ""
    assert warning.startswith("mixed settings: these trials ran 2 settings")
    assert f"route ({DIRECT}, {BEDROCK})" in warning
    assert "claude-code 2.1.257" in warning and "claude-code 2.1.260" in warning
    labels = [r.setting.label() for r in rows]
    assert labels[1] == (
        f"claude-agent-acp (acp; claude-agent-acp 0.88.0, claude-code 2.1.260) on "
        f"{SERVED} via {BEDROCK}, effort high, BenchFlow 0.10.1"
    )


def test_a_trial_whose_launch_took_two_routes_is_not_guessed(tmp_path):
    """One launch whose trials' calls went two ways: a trial with no call of
    its own cannot be put in either route's row, so it is a row of its own
    and the warning names the unknown route."""
    job = rr.evaluation_job(tmp_path / "job", ["a", "b", "c"])
    rr.trial(job, "a", 1, reward=1.0, model=LABEL, effort="high", calls=_calls(DIRECT))
    rr.trial(job, "b", 2, reward=0.0, model=LABEL, effort="high", calls=_calls(BEDROCK))
    rr.trial(job, "c", 3, reward=None, model=LABEL, effort="high", error="lost")
    settings = setting_rows(bf.load_job(job).trials)
    assert {r.setting.route for r in settings.rows} == {DIRECT, BEDROCK, None}
    differ = dict(settings.differ())
    assert sorted(differ["route"]) == sorted([DIRECT, BEDROCK, "unknown"])
    assert "route (unknown, " in (settings.warning() or "")


def test_one_setting_is_one_row_and_not_mixed(tmp_path):
    job = rr.evaluation_job(tmp_path / "job", ["a", "b"])
    for i, task in enumerate(("a", "b")):
        rr.trial(
            job, task, i, reward=1.0, model=LABEL, effort="high", calls=_calls(DIRECT)
        )
    settings = setting_rows(bf.load_job(job).trials)
    assert len(settings.rows) == 1 and not settings.mixed
    assert settings.warning() is None and settings.differ() == []
    assert settings.rows[0].effort_checked == 2


def test_the_effort_the_calls_carried_decides_and_is_marked(tmp_path):
    """A run labelled max whose calls carried high is the setting high
    (requested max), kept apart from a run that asked for high."""
    root = tmp_path / "jobs"
    asked_max = rr.evaluation_job(root / "max" / "job", ["a", "b"])
    asked_high = rr.evaluation_job(root / "high" / "job", ["a", "b"])
    for job, label in ((asked_max, "max"), (asked_high, "high")):
        for i, task in enumerate(("a", "b")):
            rr.trial(
                job, task, i, reward=1.0, model=LABEL, effort=label,
                calls=_calls(DIRECT, effort="high"),
            )  # fmt: skip
    rows = setting_rows(bf.load_job(root).trials).rows
    assert [(r.setting.effort, r.setting.effort_mismatch) for r in rows] == [
        ("high", False),
        ("high", True),
    ]
    assert rows[1].setting.effort_requested == "max"
    assert (
        rows[1]
        .setting.label()
        .endswith("effort high (requested max), BenchFlow 0.10.0")
    )


def test_calls_that_carried_no_effort_are_not_sent_not_unknown(tmp_path):
    job = rr.evaluation_job(tmp_path / "job", ["a"])
    rr.trial(
        job,
        "a",
        1,
        reward=1.0,
        model=LABEL,
        effort="high",
        calls=_calls(DIRECT, effort=None),
    )
    (row,) = setting_rows(bf.load_job(job).trials).rows
    assert (row.setting.effort, row.setting.effort_mismatch) == ("not sent", True)
    assert "effort not sent (requested high)" in row.setting.label()


def test_an_effort_nothing_records_is_its_own_row(tmp_path):
    """A trial with no recorded effort and no calls that show one is never
    merged into a row whose effort is known."""
    root = tmp_path / "jobs"
    rr.trial(
        root / "known",
        "t",
        0,
        reward=1.0,
        model=LABEL,
        effort="high",
        calls=_calls(DIRECT),
    )
    rr.trial(root / "older", "t", 0, reward=0.0, model=LABEL, effort=None)
    rows = setting_rows(bf.load_job(root).trials).rows
    assert sorted(r.setting.effort for r in rows) == ["high", "unknown"]
    # With no call, its model is the label it was given, marked as such.
    unknown = next(r for r in rows if r.setting.effort == "unknown")
    assert unknown.setting.model_source == "requested"
    assert "on anthropic/claude-opus-5 (requested)" in unknown.setting.label()


def test_a_restarted_job_reads_as_mixed(tmp_path):
    """summary.json's restarted_after_change (bench run --restart-changed)
    makes the job mixed settings, and the warning says what was set aside."""
    job = rr.evaluation_job(tmp_path / "job", ["a"])
    rr.trial(job, "a", 1, reward=1.0, model=LABEL, effort="high", calls=_calls(DIRECT))
    block = {
        "at": "2026-11-05T10:00:00+00:00",
        "changed": ["model: claude-opus-4-8 -> anthropic/claude-opus-5"],
        "set_aside": ".set-aside/2026-11-05T10-00-00Z",
        "trials": 2,
        "finished_trials": 2,
    }
    (job / "summary.json").write_text(json.dumps({"restarted_after_change": block}))
    loaded = bf.load_job(job)
    from benchflow.setting_groups import job_summaries

    settings = setting_rows(loaded.trials, summaries=job_summaries(job, loaded.trials))
    assert len(settings.rows) == 1 and settings.mixed
    assert settings.warning() == (
        "mixed settings: the job was restarted after a change (model: "
        "claude-opus-4-8 -> anthropic/claude-opus-5): 2 earlier trials (2 "
        "finished) ran with the old settings and are set aside in "
        ".set-aside/2026-11-05T10-00-00Z, not counted"
    )
    assert settings.to_dict()["restarted_after_change"] == [block]


def test_two_versions_of_one_task_are_two_task_sets(tmp_path):
    """The same task at two digests in two jobs: the rows differ in task set,
    and the warning names it (mixing task versions in one score)."""
    root = tmp_path / "jobs"
    for name, digest in (("old", "sha256:aaa"), ("new", "sha256:bbb")):
        job = rr.evaluation_job(root / name, ["a", "b"])
        rr.trial(job, "a", 1, reward=1.0, model=LABEL, effort="high",
                 calls=_calls(DIRECT), task_digest=digest)  # fmt: skip
        rr.trial(job, "b", 2, reward=0.0, model=LABEL, effort="high",
                 calls=_calls(DIRECT), task_digest="sha256:same")  # fmt: skip
    settings = setting_rows(bf.load_job(root).trials)
    assert len(settings.rows) == 2
    (name, values) = settings.differ()[0]
    assert name == "task set" and all(v.startswith("(task versions ") for v in values)


def test_a_run_writes_its_settings_rows_into_summary_json(tmp_path, monkeypatch):
    """summary.json's scoring block holds one row per setting with the
    rule's figures; a job whose trials all agree has one row, not mixed."""
    from tests.test_eval_runner_results_dx import _fake_rollouts, _invoke, _write_task

    tasks = tmp_path / "tasks"
    for name in ("a", "b", "c"):
        _write_task(tasks / name)
    _fake_rollouts(monkeypatch)
    jobs = tmp_path / "jobs"
    run = _invoke(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--sandbox", "docker", "--jobs-dir", str(jobs),
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    (summary_path,) = jobs.glob("*/summary.json")
    scoring = json.loads(summary_path.read_text())["scoring"]
    settings = scoring["settings"]
    assert settings["mixed"] is False and settings["warning"] is None
    (row,) = settings["rows"]
    assert row["agent"] == "oracle" and row["trials"] == 3
    assert row["scoring"]["n"] == scoring["n"] == 3


def test_the_resumed_job_s_summary_says_mixed_settings(tmp_path):
    from benchflow._utils.scoring import score_summary_fields
    from benchflow.setting_groups import summary_settings

    job = _resumed_job(tmp_path)
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    summary = score_summary_fields(rows)
    settings = summary_settings(job, [summary])
    assert settings["mixed"] is True
    assert [r["route"] for r in settings["rows"]] == [DIRECT, BEDROCK]
    assert [r["scoring"]["n"] for r in settings["rows"]] == [3, 3]
    assert sum(r["scoring"]["n"] for r in settings["rows"]) == summary["scoring"]["n"]
    assert settings["warning"].startswith("mixed settings: these trials ran 2 settings")


def test_a_restart_after_a_change_reads_as_mixed_in_the_job_s_records(tmp_path):
    """A named job resumed with another model and restart_changed=True sets
    its earlier trials aside: summary.json, report.md and bench eval metrics
    say mixed settings and how many trials are not counted, and keep the
    headline over the trials that ran now, which agree."""
    from benchflow.cli.main import app
    from tests.test_resume_refuses_changed_settings import _resumed
    from tests.test_resume_settings_match import _first_job, _run

    old = _first_job(tmp_path, job_name="named")
    _run(_resumed(old, model="m2", restart_changed=True))
    summary = json.loads((old / "summary.json").read_text())
    settings = summary["scoring"]["settings"]
    assert (settings["mixed"], settings["split"]) == (True, False)
    assert settings["restarted_after_change"] == [summary["restarted_after_change"]]
    assert settings["warning"].startswith(
        "mixed settings: the job was restarted after a change (model='m1' (this "
        "run: 'm2')): 2 earlier trials (2 finished) ran with the old settings"
    )
    assert (
        "Warning: mixed settings: the job was restarted"
        in (old / "report.md").read_text()
    )
    out = CliRunner().invoke(app, ["eval", "metrics", str(old), "--json"])
    doc = json.loads(out.stdout)
    assert doc["score_ratio"] == summary["score_ratio"]  # one setting now
    assert doc["scoring"]["settings"]["mixed"] is True
