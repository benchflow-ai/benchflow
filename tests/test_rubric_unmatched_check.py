"""A task.md rubric check that matches no test is named, with a did-you-mean.

A test-judged criterion's ``check`` names tests in the verifier's CTRF report.
A check that matches none (a typo, a renamed test) fails its criterion in
every trial, the oracle's included, and the run's summary showed only the
reward, so it read as the agent's failure. The summary now says which check
matched nothing, what the report holds and the closest name (Rust 0.8.2,
a37d61ce and 8e8d95cc); the criterion's record says the same.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.cli._failure_evidence import artifact_failure_evidence
from benchflow.taskmd import grading

_TESTS = [
    {"name": "tests/test_calc.py::test_add", "status": "passed"},
    {"name": "tests/test_calc.py::test_sub", "status": "passed"},
]


def _trial(job_dir: Path, verdicts: list[dict], tests: list[dict]) -> str:
    rollout = "calc__abc123"
    verifier = job_dir / rollout / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "ctrf.json").write_text(
        json.dumps({"results": {"tool": {"name": "pytest"}, "tests": tests}})
    )
    # review.json as task.md's verifier writes it (record "verifier").
    (verifier / "review.json").write_text(
        json.dumps({"record": "verifier", "summary": {}, "verdicts": verdicts})
    )
    return rollout


def _test_verdict(
    ident: str, check: str, verdict: str, tests: list[dict] = _TESTS
) -> dict:
    """A test-judged verdict with the rationale task.md's verifier writes."""
    matched = grading.matching_tests(check, tests)
    if not matched:
        rationale = grading.unmatched_check_note(check, tests)
    elif verdict == "pass":
        rationale = f"all {len(matched)} matching tests passed"
    else:
        rationale = "not passed: " + ", ".join(
            f"{t['name']} ({t['status']})" for t in matched if t["status"] != "passed"
        )
    return {
        "id": ident,
        "check": check,
        "verdict": verdict,
        "judge": {"role": "test", "tool": "pytest"},
        "rationale": rationale[:300],
    }


def test_the_summary_names_a_check_that_matched_no_test(tmp_path):
    rollout = _trial(
        tmp_path,
        [
            _test_verdict("adds", "test_ad", "fail"),
            _test_verdict("subtracts", "test_sub", "pass"),
        ],
        _TESTS,
    )

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.body.startswith("check test_ad matched no test in ctrf.json (it has ")
    assert "tests/test_calc.py::test_add" in line.body
    assert line.body.endswith("; did you mean test_add?")
    assert line.suffix == ""


def test_more_unmatched_checks_are_counted(tmp_path):
    rollout = _trial(
        tmp_path,
        [
            _test_verdict("a", "test_ad", "fail"),
            _test_verdict("b", "test_mul", "fail"),
            _test_verdict("c", "test_div", "fail"),
        ],
        _TESTS,
    )

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.suffix == " (+2 more checks matched no test)"


def test_matched_checks_leave_the_ctrf_line(tmp_path):
    tests = [
        {"name": "tests/test_calc.py::test_add", "status": "failed", "message": "boom"},
        {"name": "tests/test_calc.py::test_sub", "status": "passed"},
    ]
    rollout = _trial(
        tmp_path, [_test_verdict("adds", "test_add", "fail", tests)], tests
    )

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.body == "test_add failed: boom" or line.body.startswith("test_add")


def test_the_criterion_record_says_it_too():
    from benchflow.taskmd import grading

    note = grading.unmatched_check_note("test_ad", _TESTS)
    assert note == (
        "check test_ad matched no test in ctrf.json (it has "
        "tests/test_calc.py::test_add, tests/test_calc.py::test_sub); "
        "did you mean test_add?"
    )
    assert grading.unmatched_check_note("zzz", _TESTS).endswith("test_sub)")
    assert grading.unmatched_check_note("x", []) == (
        "check x matched no test in ctrf.json (it has none)"
    )
    # The hint survives the record's 300-character cap.
    long = [{"name": "t/" + "x" * 120 + f"::test_{i}"} for i in range(9)]
    assert len(grading.unmatched_check_note("test_9", long)) < 300


def test_a_ctrf_over_the_read_cap_blames_no_check(tmp_path):
    """Guards PT-R1-A-3: a CTRF report over 64 KiB cannot be read here, which
    made every test-judged check (passing ones too) count as matching no test
    and the line read "check test_main matched no test in ctrf.json (it has
    none)". The verifier read the whole report and matched them, so the line
    now names the failed test from the next source."""
    tests = [
        {"name": f"tests/test_x.py::test_case_{i:04d}", "status": "passed"}
        for i in range(1200)
    ]
    tests.append({"name": "tests/test_x.py::test_main", "status": "failed"})
    rollout = _trial(
        tmp_path,
        [
            _test_verdict("main", "test_main", "fail", tests),
            _test_verdict("cases", "test_case_0001", "pass", tests),
        ],
        tests,
    )
    verifier = tmp_path / rollout / "verifier"
    assert (verifier / "ctrf.json").stat().st_size > 64 * 1024
    (verifier / "test-stdout.txt").write_text(
        "FAILED tests/test_x.py::test_main - assert 1 == 2\n"
    )

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert not line.body.startswith("check ")
    assert "test_main" in line.body


def test_an_older_job_with_the_bare_rationale_is_still_named(tmp_path):
    """Jobs written before the note hold the bare sentence; the line rebuilds
    the note from ctrf.json."""
    verdict = _test_verdict("adds", "test_ad", "fail")
    verdict["rationale"] = "no test in ctrf.json matches this check"
    rollout = _trial(tmp_path, [verdict], _TESTS)

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.body.startswith("check test_ad matched no test in ctrf.json (it has ")
    assert line.body.endswith("; did you mean test_add?")


def test_no_ctrf_report_blames_no_check(tmp_path):
    """Guards PT-R2-A-1: when test.sh writes no ctrf.json (pytest's ctrf
    plugin missing, an install failing, test.sh erroring early) every
    test-judged check gets the note "matched no test in ctrf.json (it has
    none)", and the line blamed a check that may be right. The verifier
    recorded why there were no tests; the line now says that."""
    rollout = "calc__abc123"
    verifier = tmp_path / rollout / "verifier"
    verifier.mkdir(parents=True)
    report = grading.read_ctrf(None)
    verdicts = [
        {
            "id": ident,
            "check": check,
            "verdict": "fail",
            "judge": {"role": "test", "tool": report.tool},
            "rationale": grading.unmatched_check_note(check, report.tests),
        }
        for ident, check in (("adds", "test_add"), ("subtracts", "test_sub"))
    ]
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "record": "verifier",
                "summary": {"fail": 2},
                "verdicts": verdicts,
                "x-benchflow": {
                    "tests": {"count": len(report.tests), "problem": report.problem}
                },
            }
        )
    )
    (verifier / "test-stdout.txt").write_text(
        "ERROR: usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n"
        "pytest: error: unrecognized arguments: --ctrf /logs/verifier/ctrf.json\n"
    )
    assert not (verifier / "ctrf.json").exists()

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert not line.body.startswith("check ")
    assert line.body == "2 test-judged checks failed: the verifier wrote no ctrf.json"
    assert line.suffix == ""


def test_an_invalid_ctrf_report_blames_no_check(tmp_path):
    """The same for a ctrf.json that is not a CTRF report (a test with no
    status): the line names what was wrong with the report."""
    bad = [{"name": "tests/test_calc.py::test_add"}]
    report = grading.read_ctrf(json.dumps({"results": {"tests": bad}}).encode())
    assert report.problem is not None and report.tests == []
    verdict = {
        "id": "adds",
        "check": "test_add",
        "verdict": "fail",
        "judge": {"role": "test", "tool": None},
        "rationale": grading.unmatched_check_note("test_add", report.tests),
    }
    rollout = _trial(tmp_path, [verdict], bad)
    review_path = tmp_path / rollout / "verifier" / "review.json"
    review = json.loads(review_path.read_text())
    review["x-benchflow"] = {"tests": {"count": 0, "problem": report.problem}}
    review_path.write_text(json.dumps(review))

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.body == f"1 test-judged check failed: {report.problem}"


def test_a_valid_report_with_no_tests_names_the_verifier_failure(tmp_path):
    """Guards PT-R3-B-1 at c2db61b8: a collection error is not a bad check."""
    verdict = _test_verdict("adds", "test_add", "fail", [])
    rollout = _trial(tmp_path, [verdict], [])
    review_path = tmp_path / rollout / "verifier" / "review.json"
    review = json.loads(review_path.read_text())
    review["x-benchflow"] = {"tests": {"count": 0, "problem": None}}
    review_path.write_text(json.dumps(review))
    (review_path.parent / "reward.json").write_text(
        json.dumps({"reward": 0, "strict": 0, "partial": 0})
    )
    (review_path.parent / "test-stdout.txt").write_text("=== 1 error in 0.08s ===\n")

    line = artifact_failure_evidence(tmp_path, rollout)

    assert line is not None
    assert line.body == "1 error in 0.08s"
