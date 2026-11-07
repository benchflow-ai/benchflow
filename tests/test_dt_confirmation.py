"""Guard CONF-DT1-1..5 against the omissions at b00ae784 (PY8)."""

import copy

import pytest

from tests.test_dt_codex_round import _expected, _ledger, _r3_rounds, _summary
from tests.test_dt_compare import rec, row
from tests.test_dt_normalise import materialise
from tools.dt import compare, normalise, run


def test_accepted_outcome_changes_do_not_hide_scoring_regressions():
    before, after = [], []
    for name in ("net-allowlist", "net-ip-allowlist"):
        b, a = _r3_rounds(name, after_reward=1.0)
        for r in b:
            r["record"]["summary"] = _summary(mean_reward=None)
        for r in a:
            r["record"]["summary"] = _summary(mean_reward=0.25)
        before += b
        after += a
    findings = compare.compare(before, after, mode="before-after")
    verdict = compare.judge(
        findings, mode="before-after", **_ledger(), expected=_expected(), full=False
    )
    assert not verdict.ok
    assert any(
        f.cls == "summary_change" and any("0.25" in d for d in f.detail)
        for f in findings
    )


@pytest.mark.parametrize(
    "scoring",
    ["score_ratio", "pass_at_k.1", "solve_rates.solve_rate", "mean_reward_scored_only"],
)
def test_scoring_regressions_survive_subsuming_classes(scoring):
    before = rec(
        agent="nop", outcome="failed", reward=0.0, summary=_summary(**{scoring: 0.0})
    )
    after = rec(agent="nop", summary=_summary(**{scoring: 0.25}))
    findings = compare.compare(
        [row(before, agent="nop")], [row(after, agent="nop")], mode="before-after"
    )
    assert {f.cls for f in findings} >= {"nop_above_zero", "summary_change"}


def test_losing_a_summary_is_an_artifact_change_even_when_the_reward_changes():
    before = rec(outcome="unscored", reward=None)
    after = rec(summary=None)
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert any(f.cls == "key_removed" and "summary.json" in f.detail for f in findings)


def test_a_refusal_does_not_hide_a_removed_scoring_field():
    before = rec(summary=_summary())
    summary = _summary()
    summary["keys"].remove("mean_reward")
    summary["scoring"].pop("mean_reward")
    after = rec(outcome="refused_at_load", reward=None, summary=summary)
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert any(
        f.cls == "key_removed" and "summary.json:mean_reward" in f.detail
        for f in findings
    )


def test_removing_a_null_scoring_leaf_is_not_silence():
    before = rec(summary=_summary(**{"pass_at_k.1": None}))
    after = copy.deepcopy(before)
    after["summary"]["scoring"].pop("pass_at_k.1")
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert any(
        f.cls == "summary_change" and "pass_at_k.1 None -> <missing>" in f.detail
        for f in findings
    )


@pytest.mark.parametrize("wrong_mean", [0.25, 1.01])
def test_an_intended_summary_transition_needs_its_own_exact_entry(wrong_mean):
    before, after = [], []
    for name in ("net-allowlist", "net-ip-allowlist"):
        b, a = _r3_rounds(name, after_reward=1.0)
        for r in b:
            r["record"]["summary"] = _summary(mean_reward=None)
        for r in a:
            r["record"]["summary"] = _summary(mean_reward=1.0)
        before += b
        after += a
    expected = [
        *_expected(),
        {
            "id": "exact-summary-transition",
            "target": "R3",
            "item": "R3 B10",
            "mode": "before-after",
            "cells": ["net-allowlist/oracle", "net-ip-allowlist/oracle"],
            "class": "summary_change",
            "fields": ["mean_reward None -> 1.0"],
            "exact_fields": True,
        },
    ]
    findings = compare.compare(before, after, mode="before-after")
    assert compare.judge(
        findings,
        mode="before-after",
        **_ledger(),
        expected=expected,
        full=False,
        rounds=3,
    ).ok
    for r in after:
        r["record"]["summary"]["scoring"]["mean_reward"] = wrong_mean
    findings = compare.compare(before, after, mode="before-after")
    assert not compare.judge(
        findings,
        mode="before-after",
        **_ledger(),
        expected=expected,
        full=False,
        rounds=3,
    ).ok


@pytest.mark.parametrize("content", [None, "null", "[]", "1"])
def test_a_completed_job_requires_an_object_summary(tmp_path, content):
    job = materialise("py-0.8__hello-split", tmp_path / "fixture")
    summary = job / "summary.json"
    if content is None:
        summary.unlink()
    else:
        summary.write_text(content)
    with pytest.raises(normalise.NormaliseError, match=r"summary\.json"):
        normalise.normalise(job)
    # The run adapter must also fail closed, rather than emit an intact trial.
    cell = tmp_path / "cell"
    cell.mkdir()
    job.parent.rename(cell / "jobs")
    (cell / "run.log").write_text("")
    result = {"dir": str(cell), "cell": {"name": "hello-split", "agent": "oracle"}}
    assert run.read_cell(result, kind="python") is None
    assert "summary.json" in result["error"]
    assert result["record"] is None


def test_refused_at_load_still_needs_no_summary():
    """Guards b00ae784's refusal support with complete exports (CONF3-DT1-1)."""
    refused = normalise.refused_at_load_record(
        "t",
        "oracle",
        "Skipping task 't': t: BenchFlow cannot honor these task.md draft 2 fields yet, so it refuses the task\n  - steps: unsupported\n",
    )
    assert refused is not None
    assert compare.compare([row(refused)], [row(refused)], mode="before-after") == []
    completed = rec(summary=_summary(score_excl_errors_ratio=1.0))
    for before, after, cls in (
        (completed, refused, "newly_refused"),
        (refused, completed, "newly_accepted"),
    ):
        findings = compare.compare([row(before)], [row(after)], mode="before-after")
        assert {f.cls for f in findings} == {cls, "summary_change"}
        scoring = next(f for f in findings if f.cls == "summary_change")
        assert {d.split(" ", 1)[0] for d in scoring.detail} == {
            "score_ratio",
            "score_excl_errors_ratio",
            "mean_reward",
        }


def test_reader_verifier_ownership_is_compared():
    record = rec(
        outcome="unscored",
        reward=None,
        error_category=None,
        verifier_error_category="verifier_infra",
    )
    export = {
        "task_name": "t",
        "score_outcome": "verifier_errored",
        "reward": None,
        "error_category": None,
        "verifier_error_category": "verifier_infra",
    }
    before = normalise.reader_projection({"trials": [export]})
    export["verifier_error_category"] = "verifier_failure"
    after = normalise.reader_projection({"trials": [export]})
    findings = compare.compare(
        [row(record, reader=before)], [row(record, reader=after)], mode="before-after"
    )
    assert [(f.cls, f.arm) for f in findings] == [("reader_drift", "b")]
    assert "verifier_error_category" in findings[0].detail[0]


@pytest.mark.parametrize("different", [False, True])
def test_duplicate_reader_rows_cannot_overwrite_one_another(different):
    export = {"task_name": "t", "score_outcome": "passed", "reward": 1.0}
    duplicate = copy.deepcopy(export)
    if different:
        duplicate.update(score_outcome="failed", reward=0.0)
    with pytest.raises(ValueError, match=r"duplicate.*t"):
        normalise.reader_projection({"trials": [export, duplicate]})


def test_mutant_e_has_a_matched_script_capable_base():
    base = run.arm_spec("base-e")
    mutant = run.arm_spec("mut-e")
    assert base["commit"] == mutant["commit"]
    assert base["commit"] != run.arm_spec("base")["commit"]
    assert "patch" not in base
    assert mutant["patch"] == "mutants/e-keep-verifier-logs.patch"
