"""The rc gate's own logic (tools/release_gate/run_gate.py, PR #146).

The 0.10 line has no CI (Actions is off on the private repository), so PR
#120 merged with no checks, and v0.10.0rc3 was published while its tests
failed. The gate runs the suites on per-run VMs, one per leg, and writes a
receipt per leg into the internal records folder; these tests cover what
decides its verdict: every failure fails unless the lead waived it by name, a
skip or a left-out stage never reads as a pass, blocked matrix cells are
listed and never counted, no secret reaches a log, the receipt or a test's
environment variables, an upload counts only when every copy's MD5 matches,
records are read only as committed, and the combined receipt passes only
when every leg does.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.release_gate import run_gate as gate

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_a" name="test_ok" file="tests/test_a.py" line="1"/>
  <testcase classname="tests.test_a.TestThing" name="test_param[x-1]" file="tests/test_a.py" line="5">
    <failure message="assert 1 == 2">boom</failure>
  </testcase>
  <testcase classname="tests.test_b" name="test_setup" file="tests/test_b.py" line="2">
    <error message="fixture failed">e</error>
  </testcase>
  <testcase classname="tests.test_b" name="test_skip" file="tests/test_b.py" line="3">
    <skipped type="pytest.skip" message="no docker">s</skipped>
  </testcase>
  <testcase classname="tests.test_b" name="test_xfail" file="tests/test_b.py" line="4">
    <skipped type="pytest.xfail" message="known">x</skipped>
  </testcase>
  <testcase classname="tests.test_b" name="test_teardown" file="tests/test_b.py" line="6"/>
  <testcase classname="tests.test_b" name="test_teardown" file="tests/test_b.py" line="6">
    <error message="teardown">t</error>
  </testcase>
  <testcase classname="" name="tests.test_broken">
    <error message="collection failure">ImportError</error>
  </testcase>
</testsuite></testsuites>
"""

MATRIX = {
    "cells": [
        {"id": "claude-acp-sub", "agent": "claude-agent-acp", "auth": "claude_sub"},
        {
            "id": "gemini-key",
            "agent": "gemini",
            "harness": "acp",
            "auth": "gemini_key",
            "blocked": "no Gemini key: Gemini cost is unverified on this rc",
        },
        {
            "id": "opencode-openrouter",
            "agent": "opencode",
            "auth": "openrouter_key",
            "blocked": "no BenchFlow OpenRouter key",
        },
    ],
    "effort_cells": [
        {
            "id": "effort-gemini-high",
            "agent": "gemini",
            "auth": "gemini_key",
            "blocked": "no Gemini key",
        }
    ],
    "refusal_cells": [],
    "offline_evidence": [
        {
            "check": "check_cost.provider_billed(): a provider cost equals the recorded cost",
            "data": "54 OpenCode-on-OpenRouter trials on BenchFlow 0.10.0rc2",
            "result": "54 agree, 0 disagree",
            "why_offline": "no live cell returns a provider cost",
            "ref": "https://example.invalid/issue#comment",
        }
    ],
}


def test_junit_outcomes_name_tests_as_pytest_prints_them(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT)
    outcomes = gate.junit_outcomes(path)
    assert outcomes == {
        "tests/test_a.py::test_ok": "passed",
        "tests/test_a.py::TestThing::test_param[x-1]": "failed",
        "tests/test_b.py::test_setup": "failed",
        "tests/test_b.py::test_skip": "skipped",
        "tests/test_b.py::test_xfail": "skipped",
        # Passed its call, failed its teardown: failed.
        "tests/test_b.py::test_teardown": "failed",
        "tests.test_broken": "failed",
    }
    assert gate.counts(outcomes) == {
        "passed": 1,
        "failed": 4,
        "skipped": 2,
        "total": 7,
    }


def test_a_missing_report_is_no_result(tmp_path):
    assert gate.junit_outcomes(tmp_path / "absent.xml") is None
    (tmp_path / "cut.xml").write_text("<testsuites><testsuite>")
    assert gate.junit_outcomes(tmp_path / "cut.xml") is None


def test_only_failures_the_base_lacks_are_new():
    result = gate.compare_with_base(["a", "b", "c"], ["b", "c", "d"])
    assert result == {
        "new_failures": ["a"],
        "fixed_since_base": ["d"],
        "shared_failures": ["b", "c"],
    }


def test_a_recorded_failure_list_reads_like_the_known_failures_file(tmp_path):
    path = tmp_path / "base.txt"
    path.write_text(
        "# the base's failures\n\n"
        "tests/test_a.py::test_x  # needs a browser\n"
        "  tests/test_b.py::test_y\n"
        "tests/test_a.py::test_x\n"
    )
    assert gate.read_failure_list(path) == [
        "tests/test_a.py::test_x",
        "tests/test_b.py::test_y",
    ]


def test_blocked_cells_are_listed_from_every_cell_group():
    blocked = gate.blocked_cells(MATRIX)
    assert [(c["id"], c["group"], c["auth"]) for c in blocked] == [
        ("gemini-key", "cells", "gemini_key"),
        ("opencode-openrouter", "cells", "openrouter_key"),
        ("effort-gemini-high", "effort_cells", "gemini_key"),
    ]
    assert blocked[0]["reason"] == MATRIX["cells"][1]["blocked"]


def test_the_release_note_says_which_costs_are_unverified_live():
    record = {
        "ran": True,
        "runs": 40,
        "passed": 40,
        "blocked": gate.blocked_cells(MATRIX),
        "offline_evidence": MATRIX["offline_evidence"],
    }
    note = gate.release_note("0.10.0rc4", "0123456789abcdef", record)
    assert note is not None
    assert "Gemini and OpenRouter cost are unverified live on this rc" in note
    assert "40 live runs; 40 passed" in note
    assert "54 agree, 0 disagree" in note
    assert gate.release_note("0.10.0rc4", "0123456789abcdef", None) is None


ISSUE = "TEAM" + "-999"  # issue-key shaped, spelled in pieces for the public tree guard


def _receipt(**changes) -> dict:
    stage = {"ok": True, "problems": [], "counts": {"passed": 3}}
    receipt = {
        "version": "0.10.0rc4",
        "leg": "py3.12",
        "leg_stages": list(gate.load_legs()["py3.12"]["stages"]),
        "dry_run": False,
        "source": {"commit": "a" * 40, "dirty": False},
        "rc": {
            "branch": "rc-line",
            "head": "a" * 40,
            "remote_head": None,
            "remote_checked": False,
            "last_rc": {"version": "0.10.0rc3", "source_commit": "e" * 40},
            "merged_prs": [
                {"pr": 7, "commit": "f" * 40, "subject": "Merge pull request #7"}
            ],
            "problems": [],
        },
        "base": {"commit": "b" * 40, "failures_file": None, "failures": ["t::x"]},
        "records": {
            "path": "/lab/ops/releases/benchflow",
            "repo": "lab",
            "commit": "c" * 40,
        },
        "classification": None,
        "waivers": None,
        "python": "3.12",
        "gate_script": {"path": "tools/release_gate/run_gate.py", "sha256": "c" * 64},
        "host": {
            "hostname": "vm",
            "os": "Ubuntu",
            "cpus": 8,
            "docker": "29",
            "uv": "0.13.4",
        },
        "started_at": "2026-11-13T08:00:00Z",
        "finished_at": "2026-11-13T12:00:00Z",
        "stages": {name: {**stage, "name": name} for name in ("setup", *gate.STAGES)},
        "matrix": {
            "matrix_json": "tools/cost_matrix/matrix.json",
            "matrix_json_md5": "d" * 32,
            "matrix_commit": None,
            "ran": True,
            "rc": 0,
            "runs": 2,
            "passed": 2,
            "failed_runs": [],
            "blocked": gate.blocked_cells(MATRIX),
            "offline_evidence": MATRIX["offline_evidence"],
        },
        "release_note_draft": "note",
        "logs": {
            "destination": "gs://bucket/gate/0.10.0rc4/aaaa",
            "verified": True,
            "files": [],
            "scrubbed": [],
        },
    }
    receipt["stages"]["unit"].update(
        failures=[], judged={"waived": [], "not_waived": []}
    )
    for key, value in changes.items():
        receipt[key] = value
    return receipt


def test_every_stage_run_whole_and_passed_is_a_pass():
    assert gate.verdict(_receipt())[0] == "PASS"


@pytest.mark.parametrize(
    "rc",
    [
        None,
        {"branch": None, "problems": ["no --rc-branch: a run on a PR branch"]},
        {
            "branch": "rc-line",
            "problems": ["the tested commit is not origin/rc-line's head"],
        },
    ],
    ids=["no-rc-record", "pr-branch", "not-the-head"],
)
def test_a_run_off_the_rc_head_is_never_a_pass(rc):
    """Two PRs can each pass and fail together once merged: only a run on
    the rc branch's head is a release gate."""
    receipt = _receipt(rc=rc)
    result, _fail, missing = gate.verdict(receipt)
    assert result == "INCOMPLETE"
    assert any(m.startswith("not a release gate") for m in missing)
    text = gate.render_markdown(
        {**receipt, "verdict": result, "failures": [], "incomplete": missing}
    )
    assert "NOT A RELEASE GATE" in text


def test_a_leg_is_judged_on_its_own_stages():
    receipt = _receipt(leg="py3.13", leg_stages=["unit"])
    receipt["stages"] = {
        k: v for k, v in receipt["stages"].items() if k in ("setup", "unit")
    }
    assert gate.verdict(receipt)[0] == "PASS"


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["stages"]["matrix"].update(skipped="--skip matrix"),
        lambda r: r["stages"]["unit"].update(subset=True),
        # One sandbox of the matrix is a subset too.
        lambda r: r["stages"]["matrix"].update(subset=True),
        lambda r: r["logs"].update(destination=None, verified=False),
        lambda r: r["logs"].update(destination="file:///home/x/logs"),
        lambda r: r.update(dry_run=True),
    ],
    ids=[
        "stage-left-out",
        "subset",
        "matrix-one-sandbox",
        "no-upload",
        "local-logs",
        "dry-run",
    ],
)
def test_anything_left_out_is_incomplete_never_a_pass(change):
    receipt = _receipt()
    change(receipt)
    assert gate.verdict(receipt)[0] == "INCOMPLETE"


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["stages"]["unit"].update(
            ok=False, problems=["1 tests failed without a waiver"]
        ),
        lambda r: r["stages"]["hillclimb"].update(
            ok=False, problems=["1 tests skipped"]
        ),
        lambda r: r["stages"]["pip-audit"].update(
            ok=False, problems=["pip-audit exited 1"]
        ),
        lambda r: r["stages"]["setup"].update(ok=False),
        lambda r: r["logs"].update(verified=False),
        lambda r: r.update(gate_error="RuntimeError('upload broke')"),
        # A failure outweighs anything left out.
        lambda r: (
            r.update(dry_run=True),
            r["stages"]["docker"].update(ok=False, problems=["2 tests failed"]),
        ),
    ],
    ids=[
        "unit",
        "hillclimb-skipped",
        "pip-audit",
        "setup",
        "md5-mismatch",
        "gate-error",
        "dry-run",
    ],
)
def test_a_failed_check_fails_the_gate(change):
    receipt = _receipt()
    change(receipt)
    assert gate.verdict(receipt)[0] == "FAIL"


def _gate(tmp_path: Path) -> gate.Gate:
    g = gate.Gate.__new__(gate.Gate)
    g.logs = tmp_path
    g.tmp = tmp_path / "tmp"
    g.work = tmp_path
    g.waived = {}
    return g


def test_a_skip_is_not_a_pass_and_every_skip_keeps_its_reason(tmp_path):
    junit = tmp_path / "junit"
    junit.mkdir()
    (junit / "h.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.test_h" name="test_h"'
        ' file="tests/test_h.py"><skipped message="no docker"/></testcase>'
        "</testsuite></testsuites>"
    )
    g = _gate(tmp_path)
    for skips_fail in (True, False):
        stage = {"problems": [], "ok": False}
        g.judge_pytest(stage, 0, "h.xml", skips_fail=skips_fail)
        assert stage["ok"] is False
        assert "no test passed (a skip is not a pass)" in stage["problems"]
        assert stage["skip_reasons"] == {"tests/test_h.py::test_h": "no docker"}


def test_exit_1_with_no_failure_in_the_report_fails(tmp_path):
    junit = tmp_path / "junit"
    junit.mkdir()
    passing = (
        '<testsuites><testsuite><testcase classname="tests.test_a" name="test_a"'
        ' file="tests/test_a.py"/></testsuite></testsuites>'
    )
    (junit / "d.xml").write_text(passing)
    g = _gate(tmp_path)
    g.uv, g.python = "uv", "3.12"
    stage = {"problems": [], "ok": False}
    g.judge_pytest(stage, 1, "d.xml", skips_fail=False)
    assert "exit 1 but the report has no failure" in stage["problems"]

    (junit / "unit-candidate.xml").write_text(passing)
    g.args = gate.parse_args(
        ["--records", str(tmp_path / "records"), "--unit-timeout", "5"]
    )
    g.run = lambda *a, **k: 1
    stage = {"problems": [], "ok": False}
    assert g.unit_run(tmp_path, stage, "candidate", ["tests/"]) is None
    assert any("report has no failure" in p for p in stage["problems"])


def test_every_failure_fails_unless_the_lead_waived_it_by_name():
    """Review of PR #146: a test listed as env in the failure table passed the
    gate whatever made it fail. The table decides nothing now."""
    judged = gate.judge_failures(["t::env_classified", "t::waived"], {"t::waived": {}})
    assert judged == {"waived": ["t::waived"], "not_waived": ["t::env_classified"]}


def _waiver(test: str, version: str = "0.10.0rc4", **changes) -> dict:
    return {
        "test": test,
        "version": version,
        "reason": "fixed in the next rc",
        "issue": ISSUE,
        "waived_by": "lead",
        "date": "2026-11-13",
        **changes,
    }


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_a_waiver_counts_for_its_version_and_its_leg_only(tmp_path):
    path = _write(
        tmp_path / "w.json",
        {
            "schema_version": 1,
            "waivers": [
                _waiver("t::all_legs"),
                _waiver("t::ce_only", leg="docker-ce"),
                _waiver("t::old", version="0.10.0rc3"),
            ],
        },
    )
    assert sorted(gate.load_waivers(path, "0.10.0rc4", "docker-ce")) == [
        "t::all_legs",
        "t::ce_only",
    ]
    assert sorted(gate.load_waivers(path, "0.10.0rc4", "py3.12")) == ["t::all_legs"]


@pytest.mark.parametrize(
    ("candidate", "applies"),
    [
        ("0.10.0rc4", True),
        ("0.10.0rc4.dev0", True),
        ("0.10.0rc4.dev12", True),
        ("0.10.0rc5", False),
        ("0.10.0rc5.dev0", False),
        ("0.10.0rc3", False),
        ("0.10.0rc41", False),
        ("0.10.0", False),
    ],
)
def test_a_waiver_lapses_at_the_next_rc(tmp_path, candidate, applies):
    """The lead's waivers are for one rc: its development builds may use
    them, the next rc (or any other version) may not."""
    path = _write(
        tmp_path / "w.json",
        {"schema_version": 1, "waivers": [_waiver("t::x", version="0.10.0rc4")]},
    )
    assert ("t::x" in gate.load_waivers(path, candidate, "py3.12")) is applies


def test_a_waiver_for_a_leg_the_gate_lacks_is_refused(tmp_path):
    path = _write(
        tmp_path / "w.json",
        {"schema_version": 1, "waivers": [_waiver("t::x", leg="py3.14")]},
    )
    with pytest.raises(gate.BadRecord, match="names no leg"):
        gate.load_waivers(path, "0.10.0rc4", "py3.12")


@pytest.mark.parametrize(
    "change",
    [{"reason": ""}, {"issue": "see the thread"}, {"waived_by": None}],
    ids=["no-reason", "no-issue-key", "no-name"],
)
def test_a_waiver_needs_a_reason_an_issue_and_who_waived_it(tmp_path, change):
    path = _write(
        tmp_path / "w.json",
        {"schema_version": 1, "waivers": [_waiver("t::x", **change)]},
    )
    with pytest.raises(gate.BadRecord):
        gate.load_waivers(path, "0.10.0rc4", "py3.12")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo, check=True, capture_output=True,
    )  # fmt: skip


LAB = "https://github.com/benchflow-ai/lab.git"


def _lab(tmp_path: Path, waivers: list[dict] | None = None) -> Path:
    """A checkout of lab (origin GitHub's lab): its records folder, with
    waivers.json committed when ``waivers`` is given."""
    top = tmp_path / "lab"
    records = top / "ops" / "releases" / "benchflow"
    records.mkdir(parents=True)
    (top / "README.md").write_text("lab\n")
    if waivers is not None:
        _write(records / "waivers.json", {"schema_version": 1, "waivers": waivers})
    _git(top, "init", "-q")
    _git(top, "remote", "add", "origin", LAB)
    _git(top, "add", "-A")
    _git(top, "commit", "-q", "-m", "lab")
    return records


def _head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def test_records_are_read_only_as_committed(tmp_path):
    records = _lab(tmp_path)
    _write(records / "waivers.json", {"schema_version": 1, "waivers": []})
    with pytest.raises(gate.Refused, match="not committed"):
        gate.check_records(records, "0.10.0rc4")
    _git(tmp_path / "lab", "add", "-A")
    _git(tmp_path / "lab", "commit", "-q", "-m", "waivers")
    found = gate.check_records(records, "0.10.0rc4")
    assert found["waivers"]["commit"] == found["records"]["commit"]
    assert found["records"] == {
        "path": str(records),
        "repo": LAB,
        "commit": _head(records),
    }
    assert found["classification"] is None
    _write(
        records / "waivers.json", {"schema_version": 1, "waivers": [_waiver("t::x")]}
    )
    with pytest.raises(gate.Refused, match="not committed"):
        gate.check_records(records, "0.10.0rc4")


@pytest.mark.parametrize(
    ("origin", "folder", "refusal"),
    [
        (None, "ops/releases/benchflow", "not in a checkout of lab"),
        (
            "https://github.com/someone/lab.git",
            "ops/releases/benchflow",
            "not in a checkout of lab",
        ),
        (
            (
                "ht"
                + "tps://x-"
                + "access-t"
                + "oken:mad"
                + "e-up-tok"
                + "en@githu"
                + "b.com/benchflow-ai/lab.git"
            ),
            "ops/releases/benchflow",
            "not in a checkout of lab",
        ),
        (LAB, "records", "is not lab's ops/releases/benchflow"),
    ],
    ids=["git-init", "another-repo", "a-token-in-the-url", "another-folder"],
)
def test_records_anywhere_but_lab_are_refused(tmp_path, origin, folder, refusal):
    """Review of PR #146 (M1): waivers committed in any git repository
    counted. ``git init`` and a commit made a waiver the lead never wrote."""
    top = tmp_path / "repo"
    records = top / folder
    records.mkdir(parents=True)
    _write(
        records / "waivers.json",
        {"schema_version": 1, "waivers": [_waiver("tests/test_x.py::test_made_up")]},
    )
    _git(top, "init", "-q")
    if origin:
        _git(top, "remote", "add", "origin", origin)
    _git(top, "add", "-A")
    _git(top, "commit", "-q", "-m", "a waiver")
    with pytest.raises(gate.Refused, match=refusal) as refused:
        gate.check_records(records, "0.10.0rc4")
    assert "made-up-token" not in str(refused.value)
    with pytest.raises(gate.Refused, match=refusal):
        gate.combine(records, "a" * 40, "0.10.0rc4")


def test_records_never_go_in_this_repository(capsys):
    with pytest.raises(SystemExit):
        gate.parse_args(["--records", str(gate.ROOT / "release")])
    assert "outside this repository" in capsys.readouterr().err


def test_the_legs_match_test_yml_and_the_rulings():
    legs = gate.load_legs()
    assert legs["py3.12"]["python"] == "3.12"
    assert {"static", "unit", "pip-audit", "docker", "hillclimb", "matrix"} == set(
        legs["py3.12"]["stages"]
    )
    assert legs["py3.13"] == {**legs["py3.13"], "python": "3.13", "stages": ["unit"]}
    assert legs["docker-ce"]["docker"] == "ce"


def _read_records(records: Path) -> dict:
    """What a leg run on this lab checkout records of it."""
    found = gate.check_records(records, "0.10.0rc4")
    return {"records": found["records"], "waivers": found["waivers"]}


def _leg_receipt(records: Path, leg: str, verdict: str, passed: int) -> None:
    folder = records / "receipts"
    folder.mkdir(parents=True, exist_ok=True)
    _write(
        folder / f"0.10.0rc4-{'a' * 12}-{leg}.json",
        {
            **_read_records(records),
            "verdict": verdict,
            "finished_at": "2026-11-13T12:00:00Z",
            "python_version": "3.12.15",
            "host": {"docker": "29.1.3", "compose": "2.40.3"},
            "stages": {"unit": {"counts": {"passed": passed, "failed": 0}}},
            "rc": {"branch": "rc-line", "head": "a" * 40, "problems": []},
        },
    )


def _github_says(
    monkeypatch,
    head: str,
    lab: dict[str, str] | None = None,
    lab_answers: bool = True,
) -> None:
    """git ls-remote, as GitHub answers it where --combine runs: the rc
    branch at ``head``; lab's branches ``lab``, by default its main at the
    lab checkout's HEAD."""

    def heads(checkout: Path, *branches: str) -> dict[str, str] | None:
        if Path(checkout) == gate.ROOT:
            return {"rc-line": head}
        if not lab_answers:
            return None
        return lab if lab is not None else {"main": _head(Path(checkout))}

    monkeypatch.setattr(gate, "github_heads", heads)


def _combined(records: Path, commit: str = "a" * 40) -> dict:
    return json.loads(
        (records / "receipts" / f"0.10.0rc4-{commit[:12]}.json").read_text()
    )


def test_the_release_passes_only_when_every_leg_does(tmp_path, capsys, monkeypatch):
    records = _lab(tmp_path)
    _github_says(monkeypatch, "a" * 40)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 82442)
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_PASS
    combined = _combined(records)
    summary = combined["public_summary"]
    assert summary.startswith(
        "BenchFlow 0.10.0rc4 (aaaaaaaaaaaa): release gate 2026-11-13"
    )
    assert "82,442 passed / 0 failed on py3.13" in summary
    assert combined["records"]["repo"] == LAB
    assert combined["records"]["main_on_github"] == _head(records)
    _leg_receipt(records, "docker-ce", "FAIL", 1)
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_FAIL
    (records / "receipts" / f"0.10.0rc4-{'a' * 12}-py3.13.json").unlink()
    _leg_receipt(records, "docker-ce", "PASS", 1)
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE


@pytest.mark.parametrize(
    ("verdict", "rc_problems", "code"),
    [
        ("FAIL", [], gate.EXIT_FAIL),
        ("PASS", ["no --rc-branch: a run on a PR branch"], gate.EXIT_INCOMPLETE),
        ("INCOMPLETE", [], gate.EXIT_INCOMPLETE),
    ],
    ids=["fail", "not-a-release-gate", "incomplete"],
)
def test_only_a_release_gate_that_passed_gets_a_public_line(
    tmp_path, monkeypatch, verdict, rc_problems, code
):
    """Review of PR #146 (L1): --combine wrote "release gate <date>" for the
    public release notes on a FAIL, and on runs that were no release gate."""
    records = _lab(tmp_path)
    _github_says(monkeypatch, "a" * 40)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    path = records / "receipts" / f"0.10.0rc4-{'a' * 12}-py3.12.json"
    data = json.loads(path.read_text())
    data["verdict"] = verdict
    data["rc"]["problems"] = rc_problems
    path.write_text(json.dumps(data))
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == code
    assert _combined(records)["public_summary"] is None
    text = (records / "receipts" / f"0.10.0rc4-{'a' * 12}.md").read_text()
    assert "release gate 2026" not in text
    assert "No public summary line" in text


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda r: r["waivers"].update(md5="0" * 32), "py3.13: its waivers (md5 0000"),
        (
            lambda r: r["records"].update(repo=None),
            "py3.13: its records were not read from a checkout of lab",
        ),
        (
            lambda r: r["records"].update(commit="5" * 40),
            "py3.13: its records commit 555555555555 is not in this lab checkout's history",
        ),
    ],
    ids=["other-waivers", "records-not-lab", "commit-not-in-lab"],
)
def test_a_leg_s_waivers_count_only_as_committed_in_lab(
    tmp_path, monkeypatch, change, problem
):
    """Review of PR #146 (M1): --combine passed legs whose waivers no one
    compared with lab's: three PASS receipts with waivers md5 000... and
    records ``repo: null`` combined to PASS."""
    records = _lab(tmp_path, waivers=[_waiver("t::x")])
    _github_says(monkeypatch, "a" * 40)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_PASS
    path = records / "receipts" / f"0.10.0rc4-{'a' * 12}-py3.13.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE
    combined = _combined(records)
    assert any(p.startswith(problem) for p in combined["problems"]), combined
    assert combined["public_summary"] is None


def test_a_waiver_counts_only_once_it_is_on_lab_s_main(tmp_path, monkeypatch):
    """The lead merges lab: a waiver committed on a branch of lab alone,
    which the legs read, does not count until lab's main has it."""
    records = _lab(tmp_path, waivers=[])
    main = _head(records)
    _write(
        records / "waivers.json", {"schema_version": 1, "waivers": [_waiver("t::x")]}
    )
    _git(tmp_path / "lab", "commit", "-q", "-am", "a waiver on a branch")
    branch = _head(records)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    _github_says(monkeypatch, "a" * 40, lab={"main": main, "waiver": branch})
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE
    problems = _combined(records)["problems"]
    assert len(problems) == len(gate.load_legs())
    assert all(
        "waiver(s) it read are not on lab's main as they were: t::x" in p
        for p in problems
    )
    _github_says(monkeypatch, "a" * 40, lab={"main": branch})
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_PASS


def test_a_leg_whose_records_commit_is_not_on_github_does_not_pass(
    tmp_path, monkeypatch
):
    records = _lab(tmp_path)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    _github_says(monkeypatch, "a" * 40, lab_answers=False)
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE
    assert _combined(records)["problems"] == [
        "lab on GitHub could not be read: its main and the legs' commits are unchecked"
    ]
    (tmp_path / "lab" / "local.md").write_text("never pushed\n")
    _git(tmp_path / "lab", "add", "-A")
    _git(tmp_path / "lab", "commit", "-q", "-m", "local only")
    local = _head(records)
    for leg in gate.load_legs():
        (records / "receipts" / f"0.10.0rc4-{'a' * 12}-{leg}.json").unlink()
        _leg_receipt(records, leg, "PASS", 3)
    _github_says(monkeypatch, "a" * 40, lab={"main": "f" * 40})
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE
    problems = _combined(records)["problems"]
    assert (
        problems[0]
        == "lab's main on GitHub (ffffffffffff) is not in this checkout: git fetch origin"
    )
    assert (
        f"py3.12: its records commit {local[:12]} is on no branch of lab on GitHub"
        in problems
    )


def test_the_release_receipt_needs_github_s_rc_head_to_be_the_tested_one(
    tmp_path, monkeypatch
):
    records = _lab(tmp_path)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    _github_says(monkeypatch, "b" * 40)  # merged into since the legs ran
    assert gate.combine(records, "a" * 40, "0.10.0rc4") == gate.EXIT_INCOMPLETE
    combined = _combined(records)
    assert combined["problems"] == [
        "rc-line on GitHub is at bbbbbbbbbbbb, not aaaaaaaaaaaa"
    ]


def test_the_public_summary_holds_no_path_or_link():
    assert "/" not in gate.public_summary(
        "1.0", "a" * 40, ["3 passed / 0 failed on py3.12"], "2026-11-13"
    ).replace(" / ", "")
    with pytest.raises(ValueError):
        gate.public_summary(
            "1.0", "a" * 40, ["see https://example.invalid"], "2026-11-13"
        )
    with pytest.raises(ValueError):
        gate.public_summary("1.0", "a" * 40, ["logs in /home/x"], "2026-11-13")


def test_the_receipt_lists_blocked_cells_and_offline_evidence_verbatim():
    text = gate.render_markdown(
        {**_receipt(), "verdict": "PASS", "failures": [], "incomplete": []}
    )
    for cell in gate.blocked_cells(MATRIX):
        assert f"| {cell['id']} | {cell['agent']} |" in text
        assert cell["reason"] in text
    evidence = MATRIX["offline_evidence"][0]
    for key, value in evidence.items():
        assert f"{key}: {value}" in text
    assert "Not waived (0)" in text


SECRET = "sk-" + "0123456789abcdef"


def test_secret_values_never_stay_in_any_uploaded_file(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "mimo.env").write_text(
        f'export XIAOMI_API_KEY="{SECRET}"\n# NOTE=commented-out-value\nSHORT=abc\n'
    )
    logs = tmp_path / "logs"
    (logs / "matrix").mkdir(parents=True)
    (logs / "matrix" / "run.log").write_text(f"auth {SECRET} ok abc\n")
    (logs / "matrix" / "trace.bin").write_bytes(b"\x00\x01" + SECRET.encode() + b"\xff")
    (logs / "clean.log").write_text("nothing here\n")
    values = gate.secret_values(secrets)
    assert values == [SECRET]
    assert gate.scrub(logs, values) == ["matrix/run.log", "matrix/trace.bin"]
    assert (logs / "matrix" / "run.log").read_text() == "auth [secret] ok abc\n"
    assert SECRET.encode() not in (logs / "matrix" / "trace.bin").read_bytes()


def test_the_receipt_is_scrubbed_and_written_even_when_it_cannot_render(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "a.env").write_text(f"KEY={SECRET}\n")
    g = _gate(tmp_path)
    g.args = gate.parse_args(
        ["--records", str(tmp_path / "records"), "--secrets", str(secrets)]
    )
    receipt = {**_receipt(), "verdict": "PASS", "failures": [], "incomplete": []}
    receipt["matrix"]["failed_runs"] = [
        {"cell": "c", "sandbox": "docker", "task": "t", "problems": [SECRET]}
    ]
    path = g.write_receipt(receipt, "x")
    assert SECRET not in path.read_text() and "[secret]" in path.read_text()
    assert SECRET not in path.with_suffix(".md").read_text()
    del receipt["host"]  # the renderer reads it
    path = g.write_receipt(receipt, "y")
    assert "could not be rendered" in path.with_suffix(".md").read_text()
    assert json.loads(path.read_text())["verdict"] == "FAIL"


def test_no_host_credential_variable_reaches_a_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-pass")
    monkeypatch.setenv("HARBOR_TELEMETRY", "on")
    g = gate.Gate.__new__(gate.Gate)
    g.args = gate.parse_args(
        ["--work", "/tmp/w", "--skip", "matrix", "--records", str(tmp_path)]
    )
    g.python = "3.12"
    g.uv = "/home/x/.local/bin/uv"
    env, set_here = g.env({"BENCHFLOW_DETERMINISTIC_SANDBOX": "off"})
    assert "ANTHROPIC_API_KEY" not in env
    assert env["HARBOR_TELEMETRY"] == set_here["HARBOR_TELEMETRY"] == "off"
    assert env["PATH"].startswith("/home/x/.local/bin")
    assert set_here["UV_PYTHON"] == "3.12"


CLASSIFICATION = {
    "schema_version": 1,
    "version": "0.10.0rc4",
    "made_from": {"commit": "a" * 40},
    "causes": {
        "umask": {"class": "env", "cause": "the VM's umask 002"},
        "old-text": {"class": "stale test", "cause": "asserts 0.9's message"},
        "proxy-leak": {
            "class": "real bug",
            "cause": "the proxy keeps a socket open",
            "issue": ISSUE,
        },
    },
    "tests": {
        "tests/test_a.py::test_umask": "umask",
        "tests/test_a.py::test_text": "old-text",
        "tests/test_b.py::test_leak": "proxy-leak",
        "tests/test_b.py::test_leak_2": "proxy-leak",
    },
}


def test_an_upload_counts_only_when_every_copy_matches(tmp_path):
    logs = tmp_path / "logs"
    (logs / "junit").mkdir(parents=True)
    (logs / "unit.log").write_text("unit\n")
    (logs / "junit" / "unit.xml").write_text("<testsuites/>\n")
    files = gate.log_files(logs)
    assert [f["path"] for f in files] == ["junit/unit.xml", "unit.log"]
    dest = "gs://bucket/gate/0.10.0rc4/aaaaaaaaaaaa-20261113T080000Z"

    def row(path: str, data: bytes) -> dict:
        md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
        return {
            "bucket": "bucket",
            "name": f"gate/0.10.0rc4/aaaaaaaaaaaa-20261113T080000Z/{path}",
            "md5_hash": md5,
        }

    listing = [
        row("logs/unit.log", b"unit\n"),
        row("logs/junit/unit.xml", b"<testsuites/>\n"),
        {"bucket": "other", "name": "gate/x", "md5_hash": "AAAA"},
    ]
    remote = gate.gcs_md5s(json.loads(json.dumps(listing)), dest)
    assert gate.verify(files, remote, dest, "logs") is True
    assert files[1]["uri"] == f"{dest}/logs/unit.log"
    listing[0] = row("logs/unit.log", b"changed\n")
    files = gate.log_files(logs)
    assert gate.verify(files, gate.gcs_md5s(listing, dest), dest, "logs") is False
    assert [f["verified"] for f in files] == [True, False]


def test_the_gcs_prefix_must_be_a_bucket_path(capsys):
    with pytest.raises(SystemExit):
        gate.parse_args(["--gcs-prefix", "/local/folder"])
    assert "--gcs-prefix must start with gs://" in capsys.readouterr().err


def test_the_table_groups_tests_by_cause_with_their_class(tmp_path):
    text = gate.render_failures(CLASSIFICATION)
    assert "| env | 1 | 1 |" in text
    assert "| real bug | 1 | 2 |" in text
    assert "| all | 3 | 4 |" in text
    assert "## proxy-leak: real bug, 2 tests" in text
    assert f"Issue: {ISSUE}" in text
    assert (
        "| `tests/test_a.py::test_text` | stale test | asserts 0.9's message |" in text
    )


def test_the_gate_script_never_writes_the_waivers():
    source = Path(gate.__file__).read_text()
    assert "load_waivers" in source
    for writer in ("write_text", "open(", "json.dump("):
        for line in source.splitlines():
            if writer in line:
                assert "waiver" not in line.lower(), line


def test_a_basetemp_nested_past_path_max_is_removed(tmp_path):
    """The first full gate run broke removing a basetemp: a test had nested
    folders past PATH_MAX, and Path.chmod raised ENAMETOOLONG."""
    root = tmp_path / "basetemp"
    root.mkdir()
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(30):  # 30 x 201 characters: past Linux's 4096
            os.mkdir("d" * 200, dir_fd=fd)
            inner = os.open("d" * 200, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = inner
        os.mkdir("locked", dir_fd=fd)
        locked = os.open("locked", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        try:  # a file in a read-only folder: rm -rf alone leaves it
            os.close(os.open("kept", os.O_CREAT | os.O_WRONLY, dir_fd=locked))
            os.chmod(locked, 0o500)
        finally:
            os.close(locked)
    finally:
        os.close(fd)
    gate.drop_tree(root)
    assert not root.exists()


@pytest.mark.parametrize(
    ("ignored", "restored"),
    [
        (("SIGINT",), ["SIGINT"]),
        (("SIGINT", "SIGQUIT"), ["SIGINT", "SIGQUIT"]),
        ((), []),
    ],
    ids=["sigint", "sigint-and-sigquit", "neither"],
)
def test_a_background_start_gives_the_tests_ctrl_c_back(ignored, restored):
    """``nohup run_gate.py ... &`` over ssh starts the gate with SIGINT
    ignored, and every test inherited it: a Ctrl-C test's run never
    stopped (tests/test_stop_summary_docker.py).

    The probe sets both signals itself, ignored or default, so the result
    never depends on what this pytest inherited (review of PR #146, N1: a
    pytest started with ``&`` inherits SIGQUIT ignored too)."""
    import signal
    import sys

    def start() -> None:
        for sig in (signal.SIGINT, signal.SIGQUIT):
            ignore = sig.name in ignored
            signal.signal(sig, signal.SIG_IGN if ignore else signal.SIG_DFL)

    probe = (
        "import signal, subprocess, sys\n"
        "from tools.release_gate import run_gate as gate\n"
        "print(gate.restore_stop_signals())\n"
        "child = 'import signal; print(signal.getsignal(signal.SIGINT) is"
        " signal.default_int_handler)'\n"
        "sys.stdout.flush()\n"
        "subprocess.run([sys.executable, '-c', child], check=True)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=gate.ROOT,
        capture_output=True,
        text=True,
        check=True,
        preexec_fn=start,
    )
    assert done.stdout.split("\n")[:2] == [str(restored), "True"]


def test_a_gate_started_through_sg_is_refused():
    """``sg docker -c ...`` gave the gate docker as its real group: the agent
    entry could not switch to the account's own (setresgid: Operation not
    permitted) and 24 tests failed."""
    problem = gate.group_problem(gid=113, account_gid=1002)
    assert problem is not None and "sg or newgrp" in problem
    assert gate.group_problem(gid=os.getgid(), account_gid=os.getgid()) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["causes"]["umask"].update({"class": "flaky"}),
        lambda d: d["causes"]["umask"].update({"cause": " "}),
        lambda d: d["causes"]["proxy-leak"].pop("issue"),
        lambda d: d["tests"].update({"t::x": "nowhere"}),
    ],
    ids=["unknown-class", "no-cause", "real-bug-without-issue", "unknown-cause"],
)
def test_a_classification_the_gate_cannot_trust_is_refused(tmp_path, change):
    data = json.loads(json.dumps(CLASSIFICATION))
    change(data)
    with pytest.raises(gate.BadRecord):
        gate.load_classification(_write(tmp_path / "f.json", data))


@pytest.mark.parametrize(
    ("origin", "ok"),
    [
        ("https://github.com/benchflow-ai/benchflow-private.git", True),
        ("https://github.com/benchflow-ai/benchflow", True),
        ("git@github.com:benchflow-ai/benchflow-private.git", True),
        ("/home/me/bfp-dev", False),
        ("https://github.com/someone/benchflow.git", False),
    ],
)
def test_the_gate_needs_the_github_origin_tests_infer_provenance_from(origin, ok):
    """A clone of a bundle names a local path as origin, and
    test_check_results_accepts_symlinked_current_repo_inferred_source then
    finds no repository (the first full gate run, 2026-11-13)."""
    assert bool(gate.GITHUB_ORIGIN.fullmatch(origin)) is ok


def test_a_must_run_test_that_skips_fails_the_leg():
    """The direct Docker path's real-daemon tests skipped whole on every VM
    without ubuntu:24.04 already pulled, so no gate saw them run."""
    file = "tests/test_direct_docker_container.py"
    ran = {
        f"{file}::test_a": "passed",
        f"{file}::test_b": "skipped",
        "tests/x.py::t": "skipped",
    }
    assert gate.must_run_gaps(ran, [file]) == [f"{file}::test_b"]
    assert gate.must_run_gaps({"tests/x.py::t": "passed"}, [file]) == [
        f"{file} (not in the report)"
    ]
    legs = gate.load_legs()
    for leg in ("py3.12", "py3.13"):
        assert file in legs[leg]["must_run"]
        assert "ubuntu:24.04" in legs[leg]["images"]
    for leg in legs.values():
        assert "tests/test_locked_system_path.py" in leg["must_run"]


def _rc_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository like the rc line: the last rc's acceptance record, then a
    squash merge and a merge commit of two PRs, pushed to a branch."""
    repo = tmp_path / "rc"
    repo.mkdir()

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    run("init", "-q", "-b", "main")
    (repo / "a.txt").write_text("rc3\n")
    run("add", "-A")
    run("commit", "-q", "-m", "the released rc3")
    released = run("rev-parse", "HEAD")
    records = repo / ".github" / "release-acceptance"
    records.mkdir(parents=True)
    (records / "0.10.0rc3.json").write_text(
        json.dumps(
            {"version": "0.10.0rc3", "status": "accepted", "source_commit": released}
        )
    )
    run("add", "-A")
    run("commit", "-q", "-m", "0.10.0rc3 acceptance record")
    (repo / "b.txt").write_text("b\n")
    run("add", "-A")
    run("commit", "-q", "-m", "A fix (#140)")
    run("checkout", "-q", "-b", "feature")
    (repo / "c.txt").write_text("c\n")
    run("add", "-A")
    run("commit", "-q", "-m", "the feature")
    run("checkout", "-q", "main")
    run(
        "merge",
        "-q",
        "--no-ff",
        "feature",
        "-m",
        "Merge pull request #143 from org/feature",
    )
    head = run("rev-parse", "HEAD")
    run("update-ref", "refs/remotes/origin/rc-line", head)
    return repo, head, released


def test_the_rc_head_names_its_branch_and_the_prs_since_the_last_rc(tmp_path):
    repo, head, released = _rc_repo(tmp_path)
    rc = gate.rc_identity(repo, "rc-line", head)
    assert rc["problems"] == []
    assert rc["head"] == head and rc["remote_checked"] is False
    assert rc["last_rc"]["version"] == "0.10.0rc3"
    assert rc["last_rc"]["source_commit"] == released
    assert [p["pr"] for p in rc["merged_prs"]] == [140, 143]


def test_a_pr_branch_or_a_scratch_merge_is_not_a_release_gate(tmp_path):
    repo, head, _released = _rc_repo(tmp_path)
    assert gate.rc_identity(repo, None, head)["problems"] == [
        "no --rc-branch: a run on a PR branch or a scratch merge"
    ]
    older = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD~1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    problems = gate.rc_identity(repo, "rc-line", older)["problems"]
    assert problems == [f"the tested commit is not origin/rc-line's head ({head[:12]})"]
    assert gate.rc_identity(repo, "other", head)["problems"] == [
        "origin/other is not in this checkout: fetch it"
    ]


def test_the_gate_checks_every_generated_page_with_its_own_regenerate_command():
    from tools import cli_help_golden, docs_errors, docs_reference

    commands = {regenerate for _what, _args, regenerate in gate.GENERATED_CHECKS}
    assert commands == {
        cli_help_golden.REGENERATE,
        docs_errors.REGENERATE,
        docs_reference.REGENERATE,
    }
    for _what, args, regenerate in gate.GENERATED_CHECKS:
        assert args[-1] == "--check" and args[1] in regenerate


def test_a_stale_generated_page_fails_static_with_the_command(tmp_path, monkeypatch):
    g = gate.Gate.__new__(gate.Gate)
    g.checkout = tmp_path
    stale = gate.GENERATED_CHECKS[1][1]

    def run(stage, log, cmd, cwd, timeout):
        return 1 if tuple(cmd[-len(stale) :]) == stale else 0

    monkeypatch.setattr(g, "run", run, raising=False)
    monkeypatch.setattr(g, "uv_cmd", lambda *a: list(a), raising=False)
    stage = {"name": "static", "problems": [], "ok": False}
    g.static(stage)
    assert stage["ok"] is False
    assert stage["problems"] == [
        f"{gate.GENERATED_CHECKS[1][0]} is stale on this head: regenerate with "
        f"`{gate.GENERATED_CHECKS[1][2]}` (on a build VM, in the test environment) and commit"
    ]
    assert [c["exit"] for c in stage["generated"]] == [0, 1, 0]


def test_the_release_receipt_needs_every_leg_on_one_rc_head(tmp_path, monkeypatch):
    records = _lab(tmp_path)
    (records / "receipts").mkdir(parents=True)
    commit = "a" * 40
    for leg in gate.load_legs():
        receipt = _receipt(leg=leg, verdict="PASS", **_read_records(records))
        receipt["rc"] = (
            {**receipt["rc"], "problems": ["no --rc-branch: a run on a PR branch"]}
            if leg == "py3.13"
            else receipt["rc"]
        )
        (records / "receipts" / f"0.10.0rc4-{commit[:12]}-{leg}.json").write_text(
            json.dumps(receipt)
        )
    _github_says(monkeypatch, commit)
    assert gate.combine(records, commit, "0.10.0rc4") != gate.EXIT_PASS
    combined = json.loads(
        (records / "receipts" / f"0.10.0rc4-{commit[:12]}.json").read_text()
    )
    assert "py3.13: not a release gate" in combined["problems"]


def test_the_last_rc_is_the_highest_release_not_the_nearest(tmp_path):
    """The 0.10 line merged public main, so 0.9.0's source commit is nearer
    its head than 0.10.0rc3's: the last rc is still 0.10.0rc3."""
    repo, head, released = _rc_repo(tmp_path)
    nearer = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD~1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    records = repo / ".github" / "release-acceptance"
    (records / "0.9.0.json").write_text(
        json.dumps({"version": "0.9.0", "status": "accepted", "source_commit": nearer})
    )
    last = gate.last_rc(repo, head)
    assert last is not None and last["version"] == "0.10.0rc3"
    assert last["source_commit"] == released
    assert gate.release_key("0.9.0") < gate.release_key("0.10.0rc2")
    assert gate.release_key("0.10.0rc2") < gate.release_key("0.10.0rc3")
    assert gate.release_key("0.10.0rc3") < gate.release_key("0.10.0")
    assert gate.release_key("0.10.0rc4.dev0") is None


def test_pip_audit_records_when_it_read_the_advisories(tmp_path, monkeypatch):
    """A pass of pip-audit is dated by its own run on the tested head: the
    receipt says when that run asked each advisory service, and with which
    pip-audit."""
    g = gate.Gate.__new__(gate.Gate)
    g.checkout = tmp_path
    g.work = tmp_path / "work"
    g.logs = g.work / "logs"
    g.logs.mkdir(parents=True)
    ran: list[tuple[str, str]] = []
    clock = iter(["2026-11-14T09:00:00Z", "2026-11-14T09:00:05Z"])
    monkeypatch.setattr(gate, "utc_now", lambda: next(clock, "2026-11-14T09:01:00Z"))

    def run(stage, log, cmd, cwd, timeout):
        ran.append((log, " ".join(cmd)))
        if log == "pip-audit-version.log":
            (g.logs / log).write_text(
                "# $ pip-audit --version\npip-audit 2.9.0\n# exit 0\n"
            )
        return 0

    monkeypatch.setattr(g, "run", run, raising=False)
    monkeypatch.setattr(g, "uv_cmd", lambda *a: ["uv", *a], raising=False)
    stage = {"name": "pip-audit", "problems": [], "ok": False}
    g.pip_audit(stage)
    assert stage["ok"] is True
    assert [log for log, _ in ran] == [
        "pip-audit-version.log",
        "pip-audit-1.log",
        "pip-audit-2.log",
        "pip-audit-3.log",
    ]
    assert stage["advisories"]["audited"] == [
        {
            "file": "uv.lock (all extras)",
            "service": "pypi",
            "queried_at": "2026-11-14T09:00:00Z",
            "rc": 0,
            "log": "pip-audit-2.log",
        },
        {
            "file": "uv.lock (all extras)",
            "service": "osv",
            "queried_at": "2026-11-14T09:00:05Z",
            "rc": 0,
            "log": "pip-audit-3.log",
        },
    ]
    assert stage["advisories"]["services"] == ["pypi", "osv"]
    assert stage["advisories"]["pip_audit"] == "2.9.0"
    # Taken as the audit itself starts, not when the receipt is written.
    assert stage["advisories"]["queried_at"] == "2026-11-14T09:00:00Z"
    receipt = _receipt()
    receipt["stages"]["pip-audit"] = {**receipt["stages"]["pip-audit"], **stage}
    text = gate.render_markdown(
        {**receipt, "verdict": "PASS", "failures": [], "incomplete": []}
    )
    assert (
        "| pip-audit advisories | queried 2026-11-14T09:00:00Z (pip-audit 2.9.0" in text
    )
    assert (
        "uv.lock (all extras): pypi clean at 2026-11-14T09:00:00Z, "
        "osv clean at 2026-11-14T09:00:05Z" in text
    )


def test_pip_audit_audits_the_lock_with_every_extra_and_each_agent_pin_file(
    tmp_path, monkeypatch
):
    """Lead ruling on PR #146 (L2): the gate audited only the lock with
    test.yml's three extras, so an agent pin file (soupsieve 2.9 in
    harvey-lab.txt, four advisories) and the other extras went unaudited.
    Every file is audited with each advisory service, also after one fails,
    and listed with its date."""
    pins = tmp_path / gate.AGENT_REQUIREMENTS
    pins.mkdir(parents=True)
    for name in ("harvey-lab.txt", "deepagents.txt"):
        (pins / name).write_text("soupsieve==2.9\n")
    (pins / "README.md").write_text("not a pin file\n")
    g = gate.Gate.__new__(gate.Gate)
    g.checkout = tmp_path
    g.work = tmp_path / "work"
    g.logs = g.work / "logs"
    g.logs.mkdir(parents=True)
    clock = iter(f"2026-11-14T09:00:{i:02d}Z" for i in range(12))
    monkeypatch.setattr(gate, "utc_now", lambda: next(clock))
    ran: dict[str, list[str]] = {}

    def run(stage, log, cmd, cwd, timeout):
        ran[log] = list(cmd)
        return 1 if "harvey-lab.txt" in " ".join(cmd) else 0

    monkeypatch.setattr(g, "run", run, raising=False)
    monkeypatch.setattr(g, "uv_cmd", lambda *a: ["uv", *a], raising=False)
    stage = {"name": "pip-audit", "problems": [], "ok": True}
    g.pip_audit(stage)
    assert "--all-extras" in ran["pip-audit-1.log"]
    audited = stage["advisories"]["audited"]
    files = [
        "uv.lock (all extras)",
        f"{gate.AGENT_REQUIREMENTS}/deepagents.txt",
        f"{gate.AGENT_REQUIREMENTS}/harvey-lab.txt",
    ]
    assert [(a["file"], a["service"]) for a in audited] == [
        (f, s) for f in files for s in ("pypi", "osv")
    ]
    assert all(a["queried_at"].startswith("2026-11-14T09:00:") for a in audited)
    assert ran[audited[4]["log"]][-6:] == [
        "-s", "pypi", "-r", str(pins / "harvey-lab.txt"), "--no-deps", "--disable-pip",
    ]  # fmt: skip
    assert ran[audited[5]["log"]][-6:] == [
        "-s", "osv", "-r", str(pins / "harvey-lab.txt"), "--no-deps", "--disable-pip",
    ]  # fmt: skip
    assert [a["rc"] for a in audited] == [0, 0, 0, 0, 1, 1]
    assert stage["ok"] is False
    assert stage["problems"] == [
        f"pip-audit (pypi) of {gate.AGENT_REQUIREMENTS}/harvey-lab.txt exited 1",
        f"pip-audit (osv) of {gate.AGENT_REQUIREMENTS}/harvey-lab.txt exited 1",
    ]
    text = gate._advisories_text(stage)
    assert f"{gate.AGENT_REQUIREMENTS}/harvey-lab.txt: pypi FAILED at" in text
    assert "uv.lock (all extras): pypi clean at" in text


def test_pip_audit_fails_on_an_advisory_only_osv_has(tmp_path, monkeypatch):
    """pip-audit's default (PyPI) service missed CVE-2026-118667
    on langgraph 1.2.4, which OSV has, and the gate asked PyPI only, so the
    stage passed. Every file is now audited with both services, and an
    advisory from OSV alone fails the stage and is named in the receipt."""
    pins = tmp_path / gate.AGENT_REQUIREMENTS
    pins.mkdir(parents=True)
    (pins / "deepagents.txt").write_text("langgraph==1.2.4\n")
    g = gate.Gate.__new__(gate.Gate)
    g.checkout = tmp_path
    g.work = tmp_path / "work"
    g.logs = g.work / "logs"
    g.logs.mkdir(parents=True)
    monkeypatch.setattr(gate, "utc_now", lambda: "2026-11-14T09:00:00Z")

    def run(stage, log, cmd, cwd, timeout):
        # What the two services answered for langgraph 1.2.4 on 2026-11-14.
        osv = "-s osv" in " ".join(cmd)
        return 1 if osv and "deepagents.txt" in " ".join(cmd) else 0

    monkeypatch.setattr(g, "run", run, raising=False)
    monkeypatch.setattr(g, "uv_cmd", lambda *a: ["uv", *a], raising=False)
    stage = {"name": "pip-audit", "problems": [], "ok": True}
    g.pip_audit(stage)
    assert stage["ok"] is False
    assert stage["problems"] == [
        f"pip-audit (osv) of {gate.AGENT_REQUIREMENTS}/deepagents.txt exited 1"
    ]
    text = gate._advisories_text(stage)
    assert (
        f"{gate.AGENT_REQUIREMENTS}/deepagents.txt: pypi clean at "
        "2026-11-14T09:00:00Z, osv FAILED at 2026-11-14T09:00:00Z" in text
    )


def test_the_release_receipt_carries_each_leg_s_pip_audit_date(tmp_path, monkeypatch):
    records = _lab(tmp_path)
    _github_says(monkeypatch, "a" * 40)
    for leg in gate.load_legs():
        _leg_receipt(records, leg, "PASS", 3)
    path = records / "receipts" / f"0.10.0rc4-{'a' * 12}-py3.12.json"
    data = json.loads(path.read_text())
    data["stages"]["pip-audit"] = {"advisories": {"queried_at": "2026-11-14T09:00:00Z"}}
    path.write_text(json.dumps(data))
    gate.combine(records, "a" * 40, "0.10.0rc4")
    combined = _combined(records)
    assert combined["pip_audit_advisories"] == {
        "py3.12": {"queried_at": "2026-11-14T09:00:00Z"}
    }
