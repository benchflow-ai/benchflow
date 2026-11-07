"""Guard CONF3-DT1-1 against missing load-transition exports at 631fe6a8."""

import copy
import json

import pytest

from tests.test_dt_codex_round import _ledger, _summary
from tests.test_dt_compare import rec, row
from tests.test_dt_normalise import materialise
from tools.dt import compare, normalise

RATIOS = ("score_ratio", "score_excl_errors_ratio")


def _expect(findings):
    return [
        {
            "id": f"transition-{i}",
            "target": "DT1",
            "item": "DT1",
            "cells": [f.cell],
            "class": f.cls,
            "fields": list(f.detail),
            "exact_fields": True,
        }
        for i, f in enumerate(findings)
    ]


@pytest.mark.parametrize("mode", ["cross-engine", "before-after"])
@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("agent,reward", [("oracle", 1.0), ("nop", 0.0)])
@pytest.mark.parametrize("missing", [(RATIOS[0],), (RATIOS[1],), RATIOS])
def test_load_transitions_require_both_ratios(
    mode, engine, reverse, agent, reward, missing
):
    """Guards CONF3-DT1-1 at 631fe6a8 in both modes, engines and directions."""
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    completed = rec(
        engine=engine,
        agent=agent,
        reward=reward,
        outcome="passed" if reward else "failed",
        summary=_summary(score_ratio=reward, score_excl_errors_ratio=reward),
    )
    before = [row(refused, agent=agent, rnd=n) for n in (1, 2, 3)]
    after = [row(completed, agent=agent, rnd=n) for n in (1, 2, 3)]
    if reverse:
        before, after = after, before
    expected = _expect(compare.compare(before, after, mode=mode))
    for full in (False, True):
        assert compare.judge(
            compare.compare(before, after, mode=mode),
            mode=mode,
            expected=expected,
            full=full,
            rounds=3,
        ).ok
    for key in missing:
        completed["summary"]["scoring"].pop(key)
        completed["summary"]["keys"] = [
            k for k in completed["summary"]["keys"] if k != key
        ]
    findings = compare.compare(before, after, mode=mode, allow_keys=["summary.json:*"])
    for key in missing:
        assert any(
            f.cls == "summary_change"
            and set(f.rounds) == {1, 2, 3}
            and f"{key} <missing> -> <missing>" in f.detail
            for f in findings
        )
    for full in (False, True):
        assert not compare.judge(
            findings, mode=mode, expected=expected, full=full, rounds=3
        ).ok


@pytest.mark.parametrize("key", RATIOS)
@pytest.mark.parametrize("agent,reward", [("oracle", 1.0), ("nop", 0.0)])
def test_committed_load_refusal_ledger_rejects_partial_exports(key, agent, reward):
    """Reproduces CONF3-DT1-1 at 631fe6a8 with the committed ledger."""
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    completed = rec(
        engine="rust",
        reward=reward,
        outcome="passed" if reward else "failed",
        summary=_summary(score_ratio=reward, score_excl_errors_ratio=reward),
    )
    before = [row(refused, name="net-blocklist", agent=agent)]
    after = [row(completed, name="net-blocklist", agent=agent)]
    assert compare.judge(
        compare.compare(before, after, mode="cross-engine"),
        mode="cross-engine",
        **_ledger(),
        full=False,
    ).ok
    completed["summary"]["scoring"].pop(key)
    assert not compare.judge(
        compare.compare(before, after, mode="cross-engine"),
        mode="cross-engine",
        **_ledger(),
        full=False,
    ).ok


@pytest.mark.parametrize("fixture", ["py-0.8__hello-split", "rust-0.8.2__hello-split"])
@pytest.mark.parametrize("key", RATIOS)
@pytest.mark.parametrize("mutation", ["omit", "rename", "object", "nested", "list"])
def test_normalized_load_transitions_require_exact_ratio_fields(
    tmp_path, fixture, key, mutation
):
    """Guards 631fe6a8's raw-summary path, including flattened-key siblings."""
    job = materialise(fixture, tmp_path / "fixture")
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    before = [row(refused)]
    good = normalise.normalise(job)[0]
    expected = _expect(compare.compare(before, [row(good)], mode="cross-engine"))
    assert compare.judge(
        compare.compare(before, [row(good)], mode="cross-engine"),
        mode="cross-engine",
        expected=expected,
        full=True,
    ).ok
    path = job / "summary.json"
    data = json.loads(path.read_text())
    value = data.pop(key)
    if mutation == "rename":
        data[key.upper()] = value
    elif mutation != "omit":
        data[key] = {"object": {}, "nested": {"value": value}, "list": []}[mutation]
    path.write_text(json.dumps(data))
    bad = normalise.normalise(job)[0]
    assert not compare.judge(
        compare.compare(before, [row(bad)], mode="cross-engine"),
        mode="cross-engine",
        expected=expected,
        full=True,
    ).ok


@pytest.mark.parametrize("scoring", [None, {}])
def test_load_transition_cannot_lose_the_entire_scoring_map(scoring):
    """Guards CONF3-DT1-1 at 631fe6a8 when both exports vanish together."""
    refused = rec(outcome="refused_at_load", reward=None, summary=None)
    completed = rec(summary=_summary(score_excl_errors_ratio=1.0))
    expected = _expect(
        compare.compare([row(refused)], [row(completed)], mode="cross-engine")
    )
    completed = copy.deepcopy(completed)
    completed["summary"]["scoring"] = scoring
    findings = compare.compare([row(refused)], [row(completed)], mode="cross-engine")
    assert not compare.judge(
        findings,
        mode="cross-engine",
        expected=expected,
        full=False,
    ).ok
