"""Guard PT-R3-B-1 at c2db61b8: no tests is not an unmatched check."""

import json

import pytest

from benchflow.cli._failure_evidence import artifact_failure_evidence


@pytest.mark.parametrize("problem", [None, "the verifier wrote no ctrf.json"])
def test_no_tests_prefers_the_actual_verifier_failure_over_rubric_and_reward(
    tmp_path, problem
):
    """Guards c2db61b8 with reward.json present, as task.md writes it."""
    verifier = tmp_path / "trial/verifier"
    verifier.mkdir(parents=True)
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "check": "test_add",
                        "verdict": "fail",
                        "judge": {"role": "test"},
                        "rationale": "check test_add matched no test in ctrf.json (it has none)",
                    }
                ],
                "x-benchflow": {"tests": {"count": 0, "problem": problem}},
            }
        )
    )
    (verifier / "reward.json").write_text(
        json.dumps({"reward": 0, "strict": 0, "partial": 0})
    )
    (verifier / "ctrf.json").write_text(json.dumps({"results": {"tests": []}}))
    (verifier / "test-stdout.txt").write_text(
        "ImportError: broken collection\n=== 1 error in 0.08s ===\n"
    )
    line = artifact_failure_evidence(tmp_path, "trial")
    assert line is not None
    assert line.body == "1 error in 0.08s"


@pytest.mark.parametrize(
    "rationale",
    [
        "no test in ctrf.json matches this check",
        "check test_add matched no test in ctrf.json (it has none)",
    ],
)
@pytest.mark.parametrize("report", ["empty", "missing", "invalid"])
def test_an_older_job_with_no_tests_names_the_collection_error(
    tmp_path, rationale, report
):
    """Guards PT-CONF-5 at 7a220231: an older job has no x-benchflow record."""
    verifier = tmp_path / "trial/verifier"
    verifier.mkdir(parents=True)
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "check": "test_add",
                        "verdict": "fail",
                        "judge": {"role": "test"},
                        "rationale": rationale,
                    }
                ]
            }
        )
    )
    if report == "empty":
        (verifier / "ctrf.json").write_text(json.dumps({"results": {"tests": []}}))
    elif report == "invalid":
        (verifier / "ctrf.json").write_text("{not json")
    (verifier / "test-stdout.txt").write_text(
        "ImportError: broken collection\n=== 1 error in 0.08s ===\n"
    )
    line = artifact_failure_evidence(tmp_path, "trial")
    assert line is not None
    assert line.body == "1 error in 0.08s"


def test_an_older_job_whose_report_has_tests_still_names_the_check(tmp_path):
    """The boundary: a real report with other tests keeps the check's note."""
    verifier = tmp_path / "trial/verifier"
    verifier.mkdir(parents=True)
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "check": "test_ad",
                        "verdict": "fail",
                        "judge": {"role": "test"},
                        "rationale": "no test in ctrf.json matches this check",
                    }
                ]
            }
        )
    )
    (verifier / "ctrf.json").write_text(
        json.dumps({"results": {"tests": [{"name": "test_add", "status": "passed"}]}})
    )
    (verifier / "test-stdout.txt").write_text("=== 1 passed in 0.08s ===\n")
    line = artifact_failure_evidence(tmp_path, "trial")
    assert line is not None
    assert line.body.startswith("check test_ad matched no test in ctrf.json")


@pytest.mark.parametrize(
    "tests",
    [
        [{"name": "test_add"}],  # no status
        [{"name": "test_add", "status": 1}],  # status not a string
        [{"status": "failed"}],  # no name
        [{"name": "test_add", "status": "failed"}, "test_sub"],  # not an object
    ],
)
def test_an_older_job_whose_report_the_grader_rejects_names_the_collection_error(
    tmp_path, tests
):
    """Guards PT-CONF-7 at 666fb157: the fallback reads the report as the grader does."""
    verifier = tmp_path / "trial/verifier"
    verifier.mkdir(parents=True)
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "check": "test_add",
                        "verdict": "fail",
                        "judge": {"role": "test"},
                        "rationale": "no test in ctrf.json matches this check",
                    }
                ]
            }
        )
    )
    (verifier / "ctrf.json").write_text(json.dumps({"results": {"tests": tests}}))
    (verifier / "test-stdout.txt").write_text(
        "ImportError: broken collection\n=== 1 error in 0.08s ===\n"
    )
    line = artifact_failure_evidence(tmp_path, "trial")
    assert line is not None
    assert line.body == "1 error in 0.08s"


def test_an_older_job_whose_report_is_over_the_cap_is_not_called_empty(tmp_path):
    """PT-CONF-7's boundary: a report too big to read still names the check."""
    from benchflow.cli._failure_evidence import _ARTIFACT_READ_BYTES

    verifier = tmp_path / "trial/verifier"
    verifier.mkdir(parents=True)
    (verifier / "review.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "check": "test_add",
                        "verdict": "fail",
                        "judge": {"role": "test"},
                        "rationale": "no test in ctrf.json matches this check",
                    }
                ]
            }
        )
    )
    many = [{"name": f"t{i}", "status": "passed"} for i in range(4000)]
    report = json.dumps({"results": {"tests": many}})
    assert len(report) > _ARTIFACT_READ_BYTES
    (verifier / "ctrf.json").write_text(report)
    (verifier / "test-stdout.txt").write_text("=== 1 error in 0.08s ===\n")
    line = artifact_failure_evidence(tmp_path, "trial")
    assert line is not None
    assert line.body == "check test_add matched no test in ctrf.json"
