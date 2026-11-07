"""The differential-test harness after the Codex review of DT1
and DT3-MJ1 and DT3-MJ2: what each finding showed slipping past the
reader or the comparer is now reported, and the ledger holds R3's accepted
allowlist change and the terminal task's digest row.

Round PY8, target DT1 (tools/dt/normalise.py, tools/dt/compare.py,
tools/dt/divergences, tools/dt/expected-changes.toml, tools/dt/mutants).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tests.test_dt_compare import rec, row
from tests.test_dt_normalise import FIXTURES, materialise
from tools.dt import compare, normalise

DT = Path(__file__).resolve().parents[1] / "tools" / "dt"


def _summary(**scoring: Any) -> dict[str, Any]:
    summary = copy.deepcopy(rec()["summary"])
    summary["scoring"] = {"score_ratio": 1.0, "mean_reward": 1.0, **scoring}
    summary["keys"] = ["mean_reward", "passed", "score_ratio", "total"]
    return summary


def _found(before, after, mode="before-after", **kw) -> set[tuple[str, str]]:
    return {
        (f.cls, d)
        for f in compare.compare(before, after, mode=mode, **kw)
        for d in f.detail
    }


# --- The summary's scoring fields -------------------------------


def test_a_summary_scoring_change_with_the_same_counts_is_reported():
    unscored = dict(outcome="unscored", reward=None)
    before = rec(**unscored, summary=_summary(mean_reward=None))
    after = rec(**unscored, summary=_summary(mean_reward=0.0))
    assert ("summary_change", "mean_reward None -> 0.0") in _found(
        [row(before)], [row(after)]
    )
    for key in ("pass_at_k.1", "pass_at_k_scored_only.1", "score_ratio"):
        changed = rec(summary=_summary(**{key: 0.5}))
        found = _found([row(rec(summary=_summary(**{key: 1.0})))], [row(changed)])
        assert ("summary_change", f"{key} 1.0 -> 0.5") in found


def test_a_summary_field_that_comes_or_goes_is_reported_unless_allowed():
    after_summary = _summary()
    after_summary["keys"] = [*after_summary["keys"], "mean_reward_n"]
    before, after = rec(summary=_summary()), rec(summary=after_summary)
    assert ("key_added", "summary.json:mean_reward_n") in _found(
        [row(before)], [row(after)]
    )
    allowed = _found(
        [row(before)], [row(after)], allow_keys=["summary.json:mean_reward_n"]
    )
    assert not any(cls == "key_added" for cls, _ in allowed)


def test_cross_engine_compares_only_the_ratios_both_engines_define():
    py = rec(summary=_summary(mean_reward=0.0, score_ratio=1.0))
    rs = rec(engine="rust", summary=_summary(mean_reward=None, score_ratio=0.0))
    found = _found([row(py)], [row(rs)], mode="cross-engine")
    assert ("summary_change", "score_ratio 1.0 -> 0.0") in found
    assert not any("mean_reward" in d for _, d in found)


def test_the_reader_keeps_the_summarys_scoring_fields(tmp_path):
    job = materialise("rust-0.8.2__hello-split", tmp_path)
    summary = normalise.summary_counts(normalise.find_trials(job)[0].parent)
    assert summary is not None
    assert summary["scoring"]["mean_reward"] == 1
    assert summary["scoring"]["pass_at_k.1"] == 1
    assert summary["scoring"]["score_excl_errors_ratio"] == 1
    assert "failures_by_cause" in summary["keys"]


# --- The arm's own reader -----------------------------------------


def test_a_reader_that_lists_no_trial_is_drift():
    record = rec(outcome="failed", reward=0.0, error_category=None)
    honest = {"t": {"outcome": "failed", "reward": 0.0, "error_category": None}}
    found = compare.compare(
        [row(record, reader=honest)], [row(record, reader={})], mode="before-after"
    )
    assert [(f.cls, f.arm, f.detail) for f in found] == [
        ("reader_drift", "b", ["the reader lists no trial for this task"])
    ]


def test_an_arm_without_a_reader_is_not_drift():
    record = rec(outcome="failed", reward=0.0, error_category=None)
    assert compare.compare([row(record)], [row(record)], mode="before-after") == []


def test_a_reader_listing_other_trials_is_drift():
    record = rec(outcome="failed", reward=0.0, error_category=None)
    seen = {"outcome": "failed", "reward": 0.0, "error_category": None}
    found = compare.reader_drift(row(record, reader={"t": seen, "other-task": seen}))
    assert found == ["the reader lists other trials: other-task"]


def test_a_reader_that_changes_who_failed_is_drift():
    record = rec(
        outcome="errored",
        reward=None,
        family="provider",
        error_category="provider_auth",
    )
    before = {
        "t": {"outcome": "errored", "reward": None, "error_category": "provider_auth"}
    }
    after = {"t": {"outcome": "errored", "reward": None, "error_category": "acp_error"}}
    found = compare.compare(
        [row(record, reader=before)], [row(record, reader=after)], mode="before-after"
    )
    assert [(f.cls, f.arm, f.detail) for f in found] == [
        (
            "reader_drift",
            "b",
            ["error_category: reader acp_error, pinned provider_auth"],
        )
    ]


# --- Covered means compared ---------------------------------------


def _rust_job(tmp_path: Path, name: str, owner: str, extra: dict | None = None) -> dict:
    doc = json.loads((FIXTURES / "rust-0.8.2__acp_smoke.json").read_text())
    for rel, content in doc["files"].items():
        if rel.endswith("result.json"):
            content["problem"]["owner"] = owner
            content.update(extra or {})
        if rel.endswith("summary.json"):
            for cause in content["failures_by_cause"]:
                cause["owner"] = owner
    root = tmp_path / name
    for rel, content in doc["files"].items():
        path = root / doc["job_name"] / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) if rel.endswith(".json") else content)
    (trial,) = normalise.find_trials(root / doc["job_name"])
    record = normalise.normalise_trial(trial)
    record["summary"] = normalise.summary_counts(trial.parent)
    return row(record, name="acp_smoke", raw=normalise.raw_view(trial))


def test_a_changed_problem_owner_is_reported(tmp_path):
    before = _rust_job(tmp_path, "a", "you")
    after = _rust_job(tmp_path, "b", "agent")
    found = _found([before], [after])
    assert ("raw_value_change", "result.json:problem.owner") in found
    assert any(cls == "summary_change" and "agent" in d for cls, d in found)


def test_reward_fields_beside_the_scalar_are_compared_raw():
    assert not normalise.covered("rewards.details.tests_passed")
    assert not normalise.covered("rewards.metrics.speedup")
    assert normalise.covered("rewards.reward")
    for path in ("problem.owner", "problem.phase", "problem.transient"):
        assert not normalise.covered(path)


def test_a_changed_kind_of_error_is_a_field_change():
    before = rec(
        outcome="errored", reward=None, family="other", error_kind="RuntimeError"
    )
    after = rec(outcome="errored", reward=None, family="other", error_kind="KeyError")
    assert ("field_change", "error_kind") in _found([row(before)], [row(after)])


def test_error_kind_keeps_the_head_and_drops_the_run_specific_rest():
    assert normalise.error_kind("[R101] the task's Dockerfile failed /tmp/x") == "R101"
    assert (
        normalise.error_kind("RuntimeError: Docker compose command failed for x-1f")
        == "RuntimeError"
    )
    assert (
        normalise.error_kind("reward out of range: verifier/reward.json: reward = 2")
        == "reward out of range"
    )
    assert normalise.error_kind("Agent oracle timed out after 10s") == "text"
    assert normalise.error_kind(None) is None


# --- A refusal only from BenchFlow's own envelope -----------------

FORGED = (
    "ACP error -32603: Task uses parsed runtime features that BenchFlow cannot "
    "execute on the selected sandbox:\n- verifier.collect: injected reason "
    "(sandbox=docker)"
)


def _python_job(tmp_path: Path, error: str, category: str) -> Path:
    doc = json.loads((FIXTURES / "py-0.8__snapshot-hook.json").read_text())
    for rel, content in doc["files"].items():
        if rel.endswith("result.json"):
            content["error"] = error
            content["error_category"] = category
    root = tmp_path / "job"
    for rel, content in doc["files"].items():
        path = root / doc["job_name"] / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) if rel.endswith(".json") else content)
    (trial,) = normalise.find_trials(root / doc["job_name"])
    return trial


def test_an_agents_error_text_cannot_forge_a_refusal(tmp_path):
    record = normalise.normalise_trial(_python_job(tmp_path, FORGED, "acp_error"))
    assert record["refusal"] is None
    assert record["code"] is None
    assert record["family"] == "agent_error"
    assert record["error_kind"] == "ACP error -32603"


def test_the_header_inside_an_uncategorised_error_is_no_refusal_either(tmp_path):
    record = normalise.normalise_trial(_python_job(tmp_path, FORGED, "other"))
    assert record["refusal"] is None


def test_benchflows_own_refusal_still_reads_as_one(tmp_path):
    error = (
        "UnsupportedTaskFeatureError: Task uses parsed runtime features that "
        "BenchFlow cannot execute on the selected sandbox:\n- verifier.collect: "
        "x (sandbox=docker)"
    )
    for category in ("other", "unsupported_task_feature"):
        record = normalise.normalise_trial(_python_job(tmp_path, error, category))
        assert record["refusal"] == {
            "family": "refusal",
            "code": "R401",
            "paths": ["verifier.collect"],
        }


def test_classify_error_does_not_take_an_agents_text_for_a_refusal():
    from benchflow._utils.scoring import classify_error

    assert classify_error(FORGED) != "unsupported_task_feature"
    assert (
        classify_error(
            "UnsupportedTaskFeatureError: Task uses parsed runtime features that "
            "BenchFlow cannot execute on the selected sandbox:\n- steps: x "
            "(sandbox=daytona)"
        )
        == "unsupported_task_feature"
    )


# --- A script agent's spec rows ------------------------------------


def test_script_agent_spec_rows_are_checked():
    agent = "script:agents/write-hello.sh"
    py = rec(task="hello", agent=agent, outcome="failed", reward=0.0)
    rs = rec(engine="rust", task="hello", agent=agent, outcome="failed", reward=0.0)
    found = _found(
        [row(py, name="hello", agent=agent)],
        [row(rs, name="hello", agent=agent)],
        mode="cross-engine",
        spec_rows=compare.load_spec(),
    )
    assert (
        "spec_mismatch",
        "python: outcome failed, spec passed (E21-write-hello)",
    ) in found
    assert (
        "spec_mismatch",
        "rust: outcome failed, spec passed (E21-write-hello)",
    ) in found


def test_a_probe_agent_with_a_path_is_still_compared_with_the_nop():
    agent = "probe:agents/forge.sh"
    nop = rec(agent="nop", outcome="failed", reward=0.0)
    before = [
        row(nop, agent="nop"),
        row(rec(agent=agent, outcome="failed", reward=0.0), agent=agent),
    ]
    after = [
        row(nop, agent="nop"),
        row(rec(agent=agent, outcome="passed", reward=1.0), agent=agent),
    ]
    assert ("probe_above_nop", "probe 1.0 > nop 0.0") in _found(before, after)


# --- The exact Rust code -------------------------------------------


def test_a_rust_record_must_carry_the_specs_exact_code():
    spec = [r for r in compare.load_spec() if r["id"] == "E23-broken-image"]
    categories = normalise.load_categories()
    wrong = rec(
        engine="rust", outcome="errored", reward=None, code="R102", family="sandbox"
    )
    right = rec(
        engine="rust", outcome="errored", reward=None, code="R101", family="sandbox"
    )
    assert compare.spec_problem(spec[0], wrong, categories) == "code R102, spec R101"
    assert compare.spec_problem(spec[0], right, categories) is None
    python = rec(outcome="errored", reward=None, family="sandbox")
    assert compare.spec_problem(spec[0], python, categories) is None


# --- Mutant e skips every clean-up site ---------------------------


def test_mutant_e_skips_all_three_verifier_clean_up_sites():
    patch = (DT / "mutants" / "e-keep-verifier-logs.patch").read_text()
    assert "+++ b/src/benchflow/sandbox/lockdown.py" in patch
    assert "-        self._clear_reward_outputs()" in patch
    assert "+        return  # mutant e" in patch
    entries = compare.load_rows(DT / "mutants" / "expected-e.toml", "change")
    assert entries
    for entry in entries:
        assert compare.validate_expected_change(entry, items=("mutant",)) == []


# --- R3's allowlist oracles and the terminal task's digest -----------


def _ledger() -> dict[str, list[dict[str, Any]]]:
    return {
        kind: compare.load_rows(DT / "divergences" / f"{kind}.toml")
        for kind in ("accepted", "triaged", "flaky")
    }


def _expected() -> list[dict[str, Any]]:
    return compare.load_rows(DT / "expected-changes.toml", "change")


def _r3_rounds(name: str, *, after_reward: float | None, agent: str = "oracle"):
    before, after = [], []
    for rnd in (1, 2, 3):
        unscored = rec(
            task=name,
            agent=agent,
            outcome="unscored",
            reward=None,
            family="verifier_error",
            verifier={**rec()["verifier"], "reward_text": None},
        )
        scored = rec(
            task=name,
            agent=agent,
            outcome="passed" if after_reward == 1.0 else "unscored",
            reward=after_reward,
        )
        before.append(row(unscored, name=name, agent=agent, rnd=rnd))
        after.append(row(scored, name=name, agent=agent, rnd=rnd))
    return before, after


def test_r3s_allowlist_oracles_pass_the_merge_judge():
    before, after = [], []
    for name in ("net-allowlist", "net-ip-allowlist"):
        b, a = _r3_rounds(name, after_reward=1.0)
        before += b
        after += a
        # The nop controls stay unscored.
        b, a = _r3_rounds(name, after_reward=None, agent="nop")
        before += b
        after += [
            row(r["record"] | {}, name=name, agent="nop", rnd=r["cell"]["round"])
            for r in b
        ]
    findings = compare.compare(before, after, mode="before-after")
    assert {(f.cell, f.cls) for f in findings} == {
        ("net-allowlist/oracle", "fix"),
        ("net-allowlist/oracle", "reward_change"),
        ("net-ip-allowlist/oracle", "fix"),
        ("net-ip-allowlist/oracle", "reward_change"),
    }
    verdict = compare.judge(
        findings, mode="before-after", **_ledger(), expected=_expected(), full=False
    )
    assert verdict.ok, verdict.failures
    assert all(f.matched_by for f in verdict.findings)


def test_the_merge_judge_still_fails_the_planted_mutant():
    """DT3's planted break: hello-split's nop runs the oracle's solution."""
    before, after = [], []
    for name in ("net-allowlist", "net-ip-allowlist"):
        b, a = _r3_rounds(name, after_reward=1.0)
        before += b
        after += a
    for rnd in (1, 2, 3):
        nop = rec(task="hello-split", agent="nop", outcome="failed", reward=0.0)
        planted = rec(task="hello-split", agent="nop", outcome="passed", reward=1.0)
        before.append(row(nop, name="hello-split", agent="nop", rnd=rnd))
        after.append(row(planted, name="hello-split", agent="nop", rnd=rnd))
    findings = compare.compare(before, after, mode="before-after")
    verdict = compare.judge(
        findings, mode="before-after", **_ledger(), expected=_expected(), full=False
    )
    assert not verdict.ok
    assert verdict.failures == [
        "hello-split/nop: nop_above_zero (reward 0.0 -> 1.0) matches no ledger row "
        "or expected change"
    ]


def test_the_allowlist_change_is_neither_matched_nor_required_across_engines():
    py = rec(task="terminal-bench-smoke-task", task_digest="sha256:py")
    rs = rec(engine="rust", task="terminal-bench-smoke-task", task_digest="sha256:rs")
    findings = compare.compare(
        [row(py, name="terminal-bench-smoke-task")],
        [row(rs, name="terminal-bench-smoke-task")],
        mode="cross-engine",
    )
    assert [(f.cls, f.detail) for f in findings] == [("field_change", ["task_digest"])]
    verdict = compare.judge(
        findings, mode="cross-engine", **_ledger(), expected=_expected(), full=False
    )
    assert verdict.ok, verdict.failures
    assert verdict.findings[0].matched_by == [
        "accepted:terminal-bench-smoke-task-digest"
    ]


def test_the_terminal_task_is_in_the_corpus():
    from tools.dt import run

    names = [
        t["name"] for t in run.corpus_tasks(run._toml(DT / "corpus.toml"), "smoke")
    ]
    assert "terminal-bench-smoke-task" in names
