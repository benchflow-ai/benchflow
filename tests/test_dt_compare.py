"""The differential-test comparer: one test per class, on synthetic records.

Round PY8, target DT1 (tools/dt/compare.py). Before/after classes compare
two builds of one engine; cross-engine classes compare Python with Rust.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tools.dt import compare, normalise


def rec(**over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": normalise.SCHEMA,
        "engine": "python",
        "task": "t",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": {
            "reward_text": "1",
            "reward_json": None,
            "stdout_sha256": "s",
            "stderr_sha256": None,
            "ctrf": None,
        },
        "counts": {"n_prompts": 1, "n_tool_calls": 0},
        "prompt": {"n": 1, "raw_sha256": "raw", "stripped_sha256": "stripped"},
        "timing": {"setup": 2.0, "agent": 0.2, "verify": 3.0, "total": 12.0},
        "task_digest": "sha256:d",
        "skill_mode": "no-skill",
        "summary": {
            "counts": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errored": 0,
                "verifier_errored": 0,
                "unscored": 0,
            },
            "raw": {"error_categories": {}, "verifier_error_categories": {}},
            "families": {},
        },
    }
    for key, value in over.items():
        record[key] = value
    return record


RAW = {
    "result.json": {
        "sha256": "x",
        "keys": ["error", "rewards.reward", "agent_result.cost_usd"],
        "values": {"error": None, "rewards.reward": 1.0, "agent_result.cost_usd": None},
    },
    "timing.json": {
        "sha256": "y",
        "keys": ["environment_setup", "total", "verifier"],
        "values": None,
    },
}


def row(
    record, *, name="t", agent="oracle", rnd=1, raw=None, reader=None, tree="tree1"
):
    return {
        "cell": {"name": name, "tree": tree, "agent": agent, "round": rnd},
        "record": record,
        "raw": copy.deepcopy(raw if raw is not None else RAW),
        "reader": reader,
        "error": None,
    }


def classes(before, after, mode="before-after", **kw) -> dict[str, set[str]]:
    findings = compare.compare(before, after, mode=mode, **kw)
    out: dict[str, set[str]] = {}
    for f in findings:
        out.setdefault(f.cell, set()).add(f.cls)
    return out


def test_identical_arms_give_no_finding():
    assert classes([row(rec())], [row(rec())]) == {}


def test_regression():
    after = rec(outcome="failed", reward=0.0)
    assert classes([row(rec())], [row(after)]) == {
        "t/oracle": {"regression", "reward_change"}
    }


def test_fix():
    before = rec(outcome="failed", reward=0.0)
    assert classes([row(before)], [row(rec())]) == {
        "t/oracle": {"fix", "reward_change"}
    }


REFUSAL = {"family": "refusal", "code": "R401", "paths": ["verifier.collect"]}


def test_newly_refused_stands_alone():
    after = rec(
        outcome="errored", reward=None, family="refusal", code="R401", refusal=REFUSAL
    )
    assert classes([row(rec())], [row(after)]) == {"t/oracle": {"newly_refused"}}


def test_newly_accepted_stands_alone():
    before = rec(
        outcome="errored", reward=None, family="refusal", code="R401", refusal=REFUSAL
    )
    after = rec(outcome="failed", reward=0.0)
    assert classes([row(before)], [row(after)]) == {"t/oracle": {"newly_accepted"}}


def test_refused_at_load_counts_as_refused():
    before = rec(outcome="refused_at_load", reward=None, refusal=REFUSAL)
    assert classes([row(before)], [row(rec())]) == {"t/oracle": {"newly_accepted"}}


def test_outcome_flip():
    before = rec(outcome="unscored", reward=None)
    after = rec(outcome="failed", reward=1.5)
    assert classes([row(before)], [row(after)]) == {
        "t/oracle": {"outcome_flip", "reward_change"}
    }


def test_reward_change_alone():
    before = rec(outcome="failed", reward=0.75)
    after = rec(outcome="failed", reward=0.5)
    assert classes([row(before)], [row(after)]) == {"t/oracle": {"reward_change"}}


def test_verifier_output_change():
    after = rec()
    after["verifier"] = {**after["verifier"], "stdout_sha256": "other"}
    found = compare.compare([row(rec())], [row(after)], mode="before-after")
    assert [(f.cls, f.detail) for f in found] == [
        ("verifier_output_change", ["stdout_sha256"])
    ]


def test_field_change_names_the_field():
    after = rec(prompt={"n": 1, "raw_sha256": "raw2", "stripped_sha256": "stripped"})
    found = compare.compare([row(rec())], [row(after)], mode="before-after")
    assert [(f.cls, f.detail) for f in found] == [
        ("field_change", ["prompt.raw_sha256"])
    ]


def _raw_with(key: str | None = None, drop: str | None = None) -> dict:
    raw = copy.deepcopy(RAW)
    if key:
        raw["result.json"]["keys"].append(key)
        raw["result.json"]["values"][key] = 1
    if drop:
        raw["result.json"]["keys"].remove(drop)
        raw["result.json"]["values"].pop(drop)
    return raw


def test_key_added_and_removed_from_the_raw_key_set():
    """The raw diff runs beside the normalised one: a field the mapping table
    does not read still shows when it appears or goes."""
    found = classes([row(rec())], [row(rec(), raw=_raw_with(key="brand_new"))])
    assert found == {"t/oracle": {"key_added"}}
    found = classes(
        [row(rec())], [row(rec(), raw=_raw_with(drop="agent_result.cost_usd"))]
    )
    assert found == {"t/oracle": {"key_removed"}}


def test_an_allowlisted_key_is_not_a_finding():
    found = classes(
        [row(rec())],
        [row(rec(), raw=_raw_with(key="brand_new"))],
        allow_keys=["result.json:brand_*"],
    )
    assert found == {}


def test_a_raw_value_the_record_does_not_read_still_shows():
    raw = copy.deepcopy(RAW)
    raw["result.json"]["values"]["agent_result.cost_usd"] = 0.5
    found = compare.compare([row(rec())], [row(rec(), raw=raw)], mode="before-after")
    assert [(f.cls, f.detail) for f in found] == [
        ("raw_value_change", ["result.json:agent_result.cost_usd"])
    ]


def test_timing_keys_are_diffed_raw_too():
    raw = copy.deepcopy(RAW)
    raw["timing.json"]["keys"].append("agent_execution")
    assert classes([row(rec())], [row(rec(), raw=raw)]) == {"t/oracle": {"key_added"}}


def test_summary_change_without_a_record_change():
    """A classify_result change leaves result.json alone (it stores no
    outcome), but summary.json's counts move."""
    before = rec(outcome="failed", reward=0.75)
    before["summary"]["counts"].update(passed=0, failed=1)
    after = copy.deepcopy(before)
    after["summary"]["counts"].update(passed=1, failed=0)
    assert classes([row(before)], [row(after)]) == {"t/oracle": {"summary_change"}}


def test_reader_drift_when_the_arms_own_reader_disagrees():
    """bench eval inspect --json (src/benchflow/cli/results.py:95) against
    the stdlib reader, per arm."""
    record = rec(outcome="failed", reward=0.75)
    honest = {"t": {"outcome": "failed", "reward": 0.75, "error_category": None}}
    drifted = {"t": {"outcome": "passed", "reward": 0.75, "error_category": None}}
    found = compare.compare(
        [row(record, reader=honest)], [row(record, reader=drifted)], mode="before-after"
    )
    assert [(f.cls, f.arm) for f in found] == [("reader_drift", "b")]


def test_nop_above_zero_stands_alone():
    before = rec(agent="nop", outcome="failed", reward=0.0)
    after = rec(
        agent="nop",
        outcome="passed",
        reward=1.0,
        prompt={"n": 1, "raw_sha256": "z", "stripped_sha256": "z"},
    )
    found = classes([row(before, agent="nop")], [row(after, agent="nop")])
    assert found == {"t/nop": {"nop_above_zero"}}


def test_a_nop_that_scored_before_is_not_nop_above_zero():
    """A task whose verifier writes the same reward whatever the agent did
    (reward-fraction) scores the nop in both arms: no finding."""
    nop = rec(agent="nop", outcome="failed", reward=0.75)
    assert classes([row(nop, agent="nop")], [row(nop, agent="nop")]) == {}


def test_probe_above_nop():
    nop = rec(agent="nop", outcome="failed", reward=0.0)
    probe_before = rec(agent="probe:forge-reward", outcome="failed", reward=0.0)
    probe_after = rec(agent="probe:forge-reward", outcome="passed", reward=1.0)
    before = [row(nop, agent="nop"), row(probe_before, agent="probe:forge-reward")]
    after = [row(nop, agent="nop"), row(probe_after, agent="probe:forge-reward")]
    found = classes(before, after)
    assert "probe_above_nop" in found["t/probe:forge-reward"]


def test_timing_shift_on_the_phase_that_moved_only():
    before = [row(rec(), rnd=r) for r in (1, 2, 3)]
    after = []
    for r, extra in zip((1, 2, 3), (2.0, 2.1, 1.9), strict=True):
        timed = rec()
        timed["timing"] = {
            **timed["timing"],
            "verify": 3.0 + extra,
            "total": 12.0 + extra,
        }
        after.append(row(timed, rnd=r))
    found = compare.compare(before, after, mode="before-after")
    assert [(f.cls, f.detail[0].split()[0]) for f in found] == [
        ("timing_shift", "verify")
    ]


def test_timing_noise_is_not_a_shift():
    before = [row(rec(), rnd=r) for r in (1, 2, 3)]
    after = []
    for r, extra in zip((1, 2, 3), (1.2, -0.3, 0.1), strict=True):
        timed = rec()
        timed["timing"] = {**timed["timing"], "verify": 3.0 + extra}
        after.append(row(timed, rnd=r))
    assert compare.compare(before, after, mode="before-after") == []


def test_cells_pair_on_name_and_tree_never_on_digest():
    """Different task_digest, same tree: compared (field_change); same name,
    different tree: refused as unreadable."""
    found = classes([row(rec())], [row(rec(task_digest="sha256:other"))])
    assert found == {"t/oracle": {"field_change"}}
    found = classes([row(rec())], [row(rec(), tree="tree2")])
    assert found == {"t/oracle": {compare.UNREADABLE}}


def test_an_unreadable_cell_is_never_matched():
    bad = row(None)
    bad["error"] = "RefusalFormatError: refusal line does not match"
    verdict = compare.judge(
        compare.compare([row(rec())], [bad], mode="before-after"),
        mode="before-after",
        accepted=[
            {
                "id": "everything",
                "mode": "any",
                "cells": ["*"],
                "class": compare.UNREADABLE,
                "cite": "x",
                "spec_side": "n/a",
                "owner": "DT1",
            }
        ],
    )
    assert not verdict.ok
    assert any("unreadable" in f for f in verdict.failures)


# --- cross-engine -------------------------------------------------------------


def rust(**over: Any) -> dict[str, Any]:
    return rec(engine="rust", skill_mode=None, **over)


def test_python_refuses_rust_runs():
    py = rec(outcome="refused_at_load", reward=None, refusal=REFUSAL, family="refusal")
    found = classes([row(py)], [row(rust())], mode="cross-engine")
    assert found == {"t/oracle": {"python_refuses_rust_runs"}}


def test_rust_refuses_python_runs():
    rs = rust(
        outcome="errored", reward=None, refusal=REFUSAL, family="refusal", code="R401"
    )
    found = classes([row(rec())], [row(rs)], mode="cross-engine")
    assert found == {"t/oracle": {"rust_refuses_python_runs"}}


def test_score_disagree():
    found = classes(
        [row(rec())], [row(rust(outcome="failed", reward=0.0))], mode="cross-engine"
    )
    assert found == {"t/oracle": {"score_disagree"}}


def test_cross_engine_ignores_what_one_engine_does_not_record():
    """skill_mode (DROP for Rust), token counts and timing are not compared."""
    rs = rust()
    rs["counts"] = {"n_prompts": 1, "n_tool_calls": 0, "n_input_tokens": 0}
    rs["timing"] = {"setup": 0.1, "agent": 0.1, "verify": 0.1, "total": 1.0}
    assert classes([row(rec())], [row(rs)], mode="cross-engine") == {}


SPEC = [
    {
        "id": "E-reward-out-of-range",
        "task": "t",
        "agent": "oracle",
        "outcome": "unscored",
        "code": "R302",
        "source": "integ/p1:examples/README.md:55",
    },
]


def test_spec_mismatch_names_the_side_that_misses_the_documented_outcome():
    py = rec(outcome="failed", reward=1.5)
    rs = rust(outcome="unscored", reward=None, family="verifier_error", code="R302")
    found = compare.compare([row(py)], [row(rs)], mode="cross-engine", spec_rows=SPEC)
    spec = [f for f in found if f.cls == "spec_mismatch"]
    assert len(spec) == 1 and spec[0].detail[0].startswith("python:")


def test_spec_rows_check_the_family_of_their_code():
    py = rec(outcome="unscored", reward=None, family="verifier_timeout")
    rs = rust(outcome="unscored", reward=None, family="verifier_error", code="R302")
    found = compare.compare([row(py)], [row(rs)], mode="cross-engine", spec_rows=SPEC)
    assert [f.detail[0].split(":")[0] for f in found if f.cls == "spec_mismatch"] == [
        "python"
    ]


def test_a_changed_refusal_format_fails_loudly_through_the_comparer(tmp_path):
    """normalise raises on a refusal line that no longer matches
    UnsupportedTaskFeature.format; run.py stores the error, and the comparer
    turns it into a failure no ledger row can match."""
    with pytest.raises(normalise.RefusalFormatError):
        normalise.parse_python_refusal(
            "UnsupportedTaskFeatureError: Task uses parsed runtime features that "
            "BenchFlow cannot execute on the selected sandbox:\n"
            "- [R401] verifier.collect - not executed"
        )
    bad = row(None)
    bad["error"] = "RefusalFormatError: refusal line '- [R401] ...' does not match"
    verdict = compare.judge(
        compare.compare([row(rec())], [bad], mode="cross-engine"), mode="cross-engine"
    )
    assert not verdict.ok


def test_a_changed_outcome_is_not_listed_again_as_its_consequences():
    """Mutant (b)'s shape: unscored -> scored 1.5. The counts, the raw
    category field and the engine's reward.json follow from the outcome."""
    before = rec(outcome="unscored", reward=None)
    before["summary"]["counts"].update(passed=0, verifier_errored=1)
    after = rec(outcome="failed", reward=1.5)
    after["summary"]["counts"].update(passed=0, failed=1)
    after["verifier"] = {**after["verifier"], "reward_json": {"reward": 1.5}}
    raw_a, raw_b = copy.deepcopy(RAW), copy.deepcopy(RAW)
    for raw, value in ((raw_a, "verifier_failure"), (raw_b, None)):
        raw["result.json"]["keys"].append("verifier_error_category")
        raw["result.json"]["values"]["verifier_error_category"] = value
    found = classes([row(before, raw=raw_a)], [row(after, raw=raw_b)])
    assert found == {"t/oracle": {"outcome_flip", "reward_change"}}


def test_a_shift_inside_the_aa_null_of_its_phase_is_noise():
    """Setup moved up to 9.5 s between two installs of one build in the A/A
    run (timing-null.toml); a 3 s setup move is noise there, a 2 s verify
    move is not."""
    before = [row(rec(), rnd=r) for r in (1, 2, 3)]
    after = []
    for r in (1, 2, 3):
        timed = rec()
        timed["timing"] = {**timed["timing"], "setup": 5.0, "verify": 5.0}
        after.append(row(timed, rnd=r))
    null = compare.load_timing_null()
    assert null["setup"] > 3.0 > null["verify"]
    found = compare.compare(before, after, mode="before-after", timing_null=null)
    assert [f.detail[0].split()[0] for f in found] == ["verify"]


def test_when_both_engines_refuse_only_the_refusal_is_compared():
    py = rec(
        outcome="refused_at_load",
        reward=None,
        refusal=REFUSAL,
        family="refusal",
        counts=None,
        prompt=None,
        task_digest=None,
    )
    rs = rust(
        outcome="errored", reward=None, refusal=REFUSAL, family="refusal", code="R401"
    )
    assert classes([row(py)], [row(rs)], mode="cross-engine") == {}
    other = {**REFUSAL, "paths": ["sandbox.clock"]}
    rs2 = rust(
        outcome="errored", reward=None, refusal=other, family="refusal", code="R401"
    )
    assert classes([row(py)], [row(rs2)], mode="cross-engine") == {
        "t/oracle": {"field_change"}
    }
