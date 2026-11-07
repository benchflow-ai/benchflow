"""The divergence ledger's rules and the committed ledger and spec files.

Round PY8, target DT1 (tools/dt/compare.py ``judge``). A difference is
explained by an accepted, triaged or flaky row, or an expected-changes
entry, or the run fails; and a ledger entry that no longer earns its place
fails the run too.
"""

from __future__ import annotations

from pathlib import Path

from tools.dt import compare

DT = Path(__file__).parents[1] / "tools" / "dt"


def finding(
    cell="hello-split/oracle", cls="field_change", detail=("prompt.raw_sha256",)
):
    return compare.Finding(cell=cell, cls=cls, rounds=[1], detail=list(detail))


def accepted_row(**over):
    row = {
        "id": "prompt-newline",
        "mode": "cross-engine",
        "cells": ["hello-split/*"],
        "class": "field_change",
        "fields": ["prompt.raw_sha256"],
        "spec_side": "rust",
        "owner": "PT",
        "cite": "src/benchflow/rollout/_setup.py:82,86,92",
    }
    row.update(over)
    return row


def test_a_matched_row_explains_the_finding():
    verdict = compare.judge([finding()], mode="cross-engine", accepted=[accepted_row()])
    assert verdict.ok, verdict.failures
    assert verdict.findings[0].matched_by == ["accepted:prompt-newline"]


def test_an_unexplained_finding_fails():
    verdict = compare.judge([finding()], mode="cross-engine")
    assert not verdict.ok


def test_a_row_naming_other_fields_does_not_explain_a_finding():
    verdict = compare.judge(
        [finding(detail=("task_digest",))],
        mode="cross-engine",
        accepted=[accepted_row()],
    )
    assert not verdict.ok


def test_a_stale_row_fails_in_a_full_run():
    verdict = compare.judge([], mode="cross-engine", accepted=[accepted_row()])
    assert any("stale" in f for f in verdict.failures)


def test_a_stale_row_of_another_tier_or_a_partial_run_does_not_fail():
    other_tier = accepted_row(tiers=["model"])
    assert compare.judge([], mode="cross-engine", accepted=[other_tier]).ok
    assert compare.judge(
        [], mode="cross-engine", accepted=[accepted_row()], full=False
    ).ok


STATUS = (
    "06:44 UTC integrator: merged T0 py8/timing-spans a7d76b9a into integ/py-0.8 as fc2b9e9a\n"
    "09:00 UTC PT: still working\n"
)


def test_a_triaged_row_that_still_matches_after_its_owner_landed_fails():
    row = accepted_row(id="newline", owner="T0")
    verdict = compare.judge(
        [finding()], mode="cross-engine", triaged=[row], status_text=STATUS
    )
    assert any("after its owner T0 landed" in f for f in verdict.failures)
    # Owner not landed yet: the row holds.
    row = accepted_row(id="newline", owner="PT")
    assert compare.judge(
        [finding()], mode="cross-engine", triaged=[row], status_text=STATUS
    ).ok


def test_landed_targets_reads_the_integrators_lines():
    text = (
        "02:27 UTC integrator: M0 merged: py8/merge-main d9e8c4b3 into integ/py-0.8\n"
        "06:44 UTC integrator: merged T0 py8/timing-spans a7d76b9a\n"
        "07:00 UTC U10: T0 landed (fc2b9e9a); merged integ/py-0.8 into py8/fast-teardown\n"
        "10:00 UTC DT3: LANDED V4\n"
    )
    assert compare.landed_targets(text) == {"M0", "T0", "V4"}


def flaky_row(**over):
    row = {
        "id": "r2r",
        "mode": "any",
        "cells": ["r2r-mpc-control/oracle"],
        "class": ["regression", "fix", "reward_change"],
        "flip": "passed <-> failed",
        "reason": "the task's controller tolerance is met only sometimes",
        "repeats": 3,
        "cite": "a recorded run",
    }
    row.update(over)
    return row


def test_a_flaky_row_needs_three_repeats_of_evidence():
    f = finding(
        cell="r2r-mpc-control/oracle", cls="regression", detail=("passed -> failed",)
    )
    assert compare.judge([f], mode="before-after", flaky=[flaky_row()]).ok
    verdict = compare.judge([f], mode="before-after", flaky=[flaky_row(repeats=2)])
    assert any("needs at least 3" in x for x in verdict.failures)
    assert not compare.judge([f], mode="before-after", flaky=[flaky_row(flip="")]).ok


def change(**over):
    entry = {
        "id": "v4-out-of-range",
        "target": "V4",
        "item": "V4",
        "cells": ["reward-out-of-range/oracle"],
        "class": "outcome_flip",
    }
    entry.update(over)
    return entry


def test_an_expected_change_without_cells_target_or_item_is_rejected():
    f = finding(
        cell="reward-out-of-range/oracle",
        cls="outcome_flip",
        detail=("unscored -> failed",),
    )
    assert compare.judge([f], mode="before-after", expected=[change()]).ok
    for bad in (
        change(cells=[]),
        change(cells=["reward-*/oracle"]),
        change(cells=["reward-out-of-range"]),
        change(target=""),
        change(item=""),
        change(item="the whole of V4 and more"),
    ):
        verdict = compare.judge([f], mode="before-after", expected=[bad])
        assert not verdict.ok, bad


def test_an_expected_change_that_matches_nothing_fails():
    verdict = compare.judge([], mode="before-after", expected=[change()])
    assert any("matches nothing" in f for f in verdict.failures)


def test_accepted_and_triaged_rows_need_spec_side_owner_and_cite():
    for missing in ("spec_side", "owner", "cite"):
        row = accepted_row()
        row.pop(missing)
        verdict = compare.judge([finding()], mode="cross-engine", accepted=[row])
        assert not verdict.ok, missing


def test_a_drop_row_names_a_field_the_reader_drops():
    row = accepted_row(id="skill-mode", kind="drop", field="skill_mode", cells=["*"])
    assert compare.validate_ledger_row(row, "accepted") == []
    stale = accepted_row(id="gone", kind="drop", field="no_such_field", cells=["*"])
    assert compare.validate_ledger_row(stale, "accepted")


# --- the committed files ------------------------------------------------------


def test_the_committed_ledger_rows_are_all_usable():
    problems = []
    for kind in ("accepted", "triaged", "flaky"):
        for row in compare.load_rows(DT / "divergences" / f"{kind}.toml"):
            problems += compare.validate_ledger_row(row, kind)
    for entry in compare.load_rows(DT / "expected-changes.toml", "change"):
        problems += compare.validate_expected_change(entry)
    assert problems == []


def test_accepted_toml_holds_the_rows_goals_lists():
    """GOALS DT1: acp_smoke refused by Rust, snapshot-hook refused by Python,
    task_digest on split layouts, skill_mode hard-coded in Rust,
    hello-world-task read through its task.md by Rust, skill deployment on
    tasks that ship skills, net-blocklist and net-ip-blocklist refused at
    load by Python."""
    rows = {
        row["id"]: row
        for row in compare.load_rows(DT / "divergences" / "accepted.toml")
    }
    for rid in (
        "acp_smoke-rust-refuses",
        "snapshot-hook-python-refuses",
        "task-digest-split-layout",
        "skill-mode-rust-hard-coded",
        "hello-world-task-taskmd",
        "task-skills-deployment",
        "net-blocklist-refused-at-load",
        "net-ip-blocklist-refused-at-load",
    ):
        assert rid in rows, rid
        assert rows[rid]["spec_side"] in compare.SPEC_SIDES
        assert rows[rid]["owner"] and rows[rid]["cite"]
    for rid in ("net-blocklist-refused-at-load", "net-ip-blocklist-refused-at-load"):
        assert "python_refuses_rust_runs" in rows[rid]["class"]
        assert "taskmd/plan.py:931-936" in rows[rid]["cite"]


def test_triaged_toml_holds_the_trailing_newline():
    rows = {
        row["id"]: row for row in compare.load_rows(DT / "divergences" / "triaged.toml")
    }
    row = rows["instruction-trailing-newline"]
    assert "rollout/_setup.py:82,86,92" in row["cite"]
    assert ":370-376" in row["cite"]


def test_the_spec_table_covers_the_examples_and_the_goldens():
    rows = compare.load_spec(DT / "spec" / "expected.toml")
    by_task = {}
    for row in rows:
        assert row["source"], row["id"]
        assert row["outcome"] in ("passed", "failed", "errored", "unscored", "refused")
        by_task.setdefault(row["task"], []).append(row)
    examples = {
        "hello": None,
        "hello-split": None,
        "hello-08": None,
        "reward-fraction": None,
        "reward-json": None,
        "reward-json-bad-metric": "R302",
        "reward-json-metrics": "R302",
        "reward-out-of-range": "R302",
        "reward-negative": "R302",
        "reward-nan": "R302",
        "reward-empty": "R302",
        "reward-word": "R302",
        "agent-timeout": None,
        "verifier-timeout": "R303",
        "build-timeout": "R102",
        "broken-image": "R101",
        "ready-never": "R105",
        "mount-outputs": None,
        "separate-verifier": None,
        "net-allowlist": None,
        "net-blocklist": None,
        "net-ip-allowlist": None,
        "net-ip-blocklist": None,
        "net-ip-none": None,
        "sidecar": None,
    }
    for task, code in examples.items():
        assert task in by_task, task
        assert any((r.get("code") or None) == code for r in by_task[task]), task
    goldens = sorted(
        p.stem
        for p in (
            Path(__file__).parent / "integration" / "deterministic" / "golden"
        ).glob("*.json")
    )
    assert len(goldens) == 9
    for name in goldens:
        assert name in by_task, name
        assert by_task[name][0]["source"].startswith(
            "tests/integration/deterministic/golden/"
        )


def test_every_cell_an_expected_change_names_must_show_it():
    entry = change(cells=["reward-out-of-range/oracle", "reward-negative/oracle"])
    f = finding(cell="reward-out-of-range/oracle", cls="outcome_flip", detail=("x",))
    verdict = compare.judge([f], mode="before-after", expected=[entry])
    assert any("reward-negative/oracle does not show" in x for x in verdict.failures)


def test_a_mutant_proof_needs_the_class_in_every_round():
    f = compare.Finding("reward-out-of-range/oracle", "outcome_flip", rounds=[1, 3])
    entry = change(item="mutant:b")
    verdict = compare.judge(
        [f], mode="before-after", expected=[entry], expected_items=("mutant",), rounds=3
    )
    assert any("rounds [1, 3] of 3" in x for x in verdict.failures)
    f.rounds = [1, 2, 3]
    f.matched_by = []
    assert compare.judge(
        [f], mode="before-after", expected=[entry], expected_items=("mutant",), rounds=3
    ).ok


def test_each_detail_of_a_finding_needs_a_row_naming_it():
    f = finding(detail=("prompt.raw_sha256", "task_digest"))
    digest = accepted_row(id="digest", fields=["task_digest"], spec_side="n/a")
    verdict = compare.judge([f], mode="cross-engine", accepted=[accepted_row()])
    assert any(
        "task_digest" in x and "matches no ledger row" in x for x in verdict.failures
    )
    verdict = compare.judge([f], mode="cross-engine", accepted=[accepted_row(), digest])
    assert verdict.ok, verdict.failures
    assert sorted(verdict.findings[0].matched_by) == [
        "accepted:digest",
        "accepted:prompt-newline",
    ]


def test_a_drop_row_is_not_stale_for_matching_no_finding():
    row = accepted_row(id="skill-mode", kind="drop", field="skill_mode", cells=["*"])
    assert compare.judge([], mode="cross-engine", accepted=[row]).ok
