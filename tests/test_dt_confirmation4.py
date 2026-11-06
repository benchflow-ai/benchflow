"""Guard CONF2-DT1-1..3 against the summary omissions at 6577f861."""

import copy
import json
import tomllib

import pytest

from tests.test_dt_codex_round import DT, _ledger, _summary
from tests.test_dt_compare import rec, row
from tests.test_dt_normalise import materialise
from tools.dt import compare, normalise


@pytest.mark.parametrize("mode", ["cross-engine", "before-after"])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_load_refusal_requires_exact_completed_scoring(mode, reverse):
    """Guards CONF2-DT1-1 at 6577f861, including both transition directions."""
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    completed = rec(
        engine="rust", summary=_summary(score_ratio=1.0, score_excl_errors_ratio=1.0)
    )
    before = [row(refused, name="net-blocklist", rnd=n) for n in (1, 2, 3)]
    after = [row(completed, name="net-blocklist", rnd=n) for n in (1, 2, 3)]
    if reverse:
        before, after = after, before
    findings = compare.compare(before, after, mode=mode)
    scoring = [f for f in findings if f.cls == "summary_change"]
    assert scoring
    # Accept the legitimate transition with its exact scoring exports; the
    # same ledger must reject a malformed completed arm in all three rounds.
    expected = [
        {
            "id": f"transition-{i}",
            "target": "DT1",
            "item": "R3 B10",
            "cells": [f.cell],
            "class": f.cls,
            "fields": list(f.detail),
            "exact_fields": True,
        }
        for i, f in enumerate(findings)
    ]
    assert compare.judge(
        findings, mode=mode, expected=expected, full=False, rounds=3
    ).ok
    completed["summary"]["scoring"]["score_ratio"] = 0.25
    findings = compare.compare(before, after, mode=mode)
    assert not compare.judge(
        findings, mode=mode, expected=expected, full=False, rounds=3
    ).ok


def test_the_load_refusal_ledger_allows_only_the_control_ratios():
    """Guards CONF2-DT1-1's committed cross-engine ledger at 6577f861."""
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    for agent, reward in (("oracle", 1.0), ("nop", 0.0)):
        completed = rec(
            engine="rust",
            agent=agent,
            reward=reward,
            outcome="passed" if reward else "failed",
            summary=_summary(score_ratio=reward, score_excl_errors_ratio=reward),
        )
        before = [row(refused, name="net-blocklist", agent=agent)]
        after = [row(completed, name="net-blocklist", agent=agent)]
        findings = compare.compare(before, after, mode="cross-engine")
        assert compare.judge(findings, mode="cross-engine", **_ledger(), full=False).ok
        completed["summary"]["scoring"]["score_ratio"] = 0.25
        findings = compare.compare(before, after, mode="cross-engine")
        assert not compare.judge(
            findings, mode="cross-engine", **_ledger(), full=False
        ).ok


@pytest.mark.parametrize(
    "numeric,malformed",
    [(1.0, "1.0"), (1.0, True), ([0.2, 1.0], "[0.2, 1.0]"), ([0.2, 1.0], [0.2, True])],
)
def test_exact_scoring_transitions_preserve_json_types(numeric, malformed):
    """Guards CONF2-DT1-2 at 6577f861, including nested boolean siblings."""
    before = rec(summary=_summary(mean_reward=0.0))
    after = rec(summary=_summary(mean_reward=numeric))
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    expected = [
        {
            "id": "typed-summary",
            "target": "DT1",
            "item": "R3 B10",
            "cells": ["t/oracle"],
            "class": "summary_change",
            "fields": findings[0].detail.copy(),
            "exact_fields": True,
        }
    ]
    assert compare.judge(
        findings, mode="before-after", expected=expected, full=False
    ).ok
    after["summary"]["scoring"]["mean_reward"] = malformed
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert not compare.judge(
        findings, mode="before-after", expected=expected, full=False
    ).ok
    # Equality must also distinguish bool from number without a value change.
    before["summary"]["scoring"]["mean_reward"] = numeric
    assert compare.summary_classes(before, after, cross=False)


def test_the_retained_mutant_e_rejects_string_mean_rewards():
    """Guards CONF2-DT1-2 at 6577f861 through the retained exact proof."""
    expected_path = DT / "mutants" / "expected-e.toml"
    expected = tomllib.loads(expected_path.read_text())["change"]
    # These are the control-derived scoring fields in the retained proof.
    fields = next(e["fields"] for e in expected if e["class"] == "summary_change")
    before, after = [], []
    for rnd in (1, 2, 3):
        a, b = {}, {}
        for detail in fields:
            key, transition = detail.split(" ", 1)
            va, vb = transition.split(" -> ")
            # Historical percentage details were unquoted before this fix.
            a[key] = json.loads(va) if not va.endswith("%") else va
            b[key] = json.loads(vb) if not vb.endswith("%") else vb
        before.append(
            row(
                rec(
                    agent="script:forge-reward.sh",
                    outcome="failed",
                    reward=0.0,
                    summary=_summary(**a),
                ),
                name="forge-reward",
                agent="script:forge-reward.sh",
                rnd=rnd,
            )
        )
        after.append(
            row(
                rec(agent="script:forge-reward.sh", summary=_summary(**b)),
                name="forge-reward",
                agent="script:forge-reward.sh",
                rnd=rnd,
            )
        )
    findings = compare.compare(before, after, mode="before-after")
    assert compare.judge(
        findings,
        mode="before-after",
        expected=expected,
        expected_items=("mutant",),
        full=False,
        rounds=3,
    ).ok
    for r in after:
        r["record"]["summary"]["scoring"]["mean_reward"] = "1.0"
    findings = compare.compare(before, after, mode="before-after")
    assert not compare.judge(
        findings,
        mode="before-after",
        expected=expected,
        expected_items=("mutant",),
        full=False,
        rounds=3,
    ).ok


def test_rust_score_interval_changes_are_independent(tmp_path):
    """Guards CONF2-DT1-3 using the Rust fixture present at 6577f861."""
    job = materialise("rust-0.8.2__hello-split", tmp_path / "fixture")
    before = normalise.normalise(job)[0]
    path = job / "summary.json"
    data = json.loads(path.read_text())
    data["score_interval"] = [1, 1]
    path.write_text(json.dumps(data))
    after = normalise.normalise(job)[0]
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert any(
        f.cls == "summary_change" and any("score_interval" in d for d in f.detail)
        for f in findings
    )
    assert not compare.judge(findings, mode="before-after", full=False).ok
    # It must survive an independently accepted outcome transition too.
    after = copy.deepcopy(after)
    after.update(outcome="refused_at_load", reward=None)
    findings = compare.compare([row(before)], [row(after)], mode="before-after")
    assert any(
        f.cls == "summary_change" and any("score_interval" in d for d in f.detail)
        for f in findings
    )
