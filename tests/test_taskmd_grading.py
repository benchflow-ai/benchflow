"""Rubric grading: CTRF reports, how a check matches tests, and the score (docs/package.md, docs/rubrics.md)."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from fractions import Fraction

import pytest

from benchflow.taskmd import grading
from tests._taskmd_helpers import require_taskmd_repo

REPORT = [
    {
        "name": "tests/test_retry.py::test_timeout_retries",
        "status": "passed",
        "duration": 41,
    },
    {"name": "tests/test_retry.py::test_backoff[0.5]", "status": "passed"},
    {
        "name": "tests/test_retry.py::test_backoff[1.0]",
        "status": "failed",
        "message": "assert 1.31 <= 1.2",
    },
]


@pytest.mark.parametrize(
    ("check", "matched", "verdict"),
    [
        # docs/package.md, "How a check matches tests": the worked table.
        ("test_timeout_retries", 1, True),
        ("test_retry.py::test_timeout_retries", 1, True),
        ("test_backoff", 2, False),
        ("test_backoff[0.5]", 1, True),
        ("tests/test_retry.py", 3, False),
        ("test_retry.py", 3, False),
        ("retry.py::test_timeout_retries", 0, False),
        ("*", 3, False),
        ("./tests/test_retry.py", 3, False),
    ],
)
def test_the_spec_table_of_check_matches(
    check: str, matched: int, verdict: bool
) -> None:
    assert len(grading.matching_tests(check, REPORT)) == matched
    assert grading.test_verdict(check, REPORT) is verdict


def test_star_passes_only_a_nonempty_report_that_all_passed() -> None:
    assert grading.test_verdict("*", []) is False
    assert grading.test_verdict("*", REPORT[:2]) is True


def test_a_file_path_matches_by_file_only_when_no_name_does() -> None:
    tests = [
        {"name": "a", "status": "passed", "filePath": "/src/tests/test_x.py"},
        {"name": "b", "status": "passed", "filePath": "./tests/test_x.py"},
    ]
    assert len(grading.matching_tests("tests/test_x.py", tests)) == 2
    assert len(grading.matching_tests("a", tests)) == 1


def test_read_ctrf_refuses_what_is_not_a_report() -> None:
    assert grading.read_ctrf(None).problem == "the verifier wrote no ctrf.json"
    assert grading.read_ctrf(b"{").tests == []
    assert grading.read_ctrf(b'{"results": {}}').problem
    assert grading.read_ctrf(b'{"results": {"tests": [{"name": "x"}]}}').problem
    good = grading.read_ctrf(
        json.dumps(
            {
                "results": {
                    "tool": {"name": "pytest", "version": "8.4.1"},
                    "tests": REPORT,
                }
            }
        ).encode()
    )
    assert good.problem is None and good.tool == "pytest 8.4.1" and len(good.tests) == 3


def test_tests_view_is_jcs_with_ids_outcomes_and_durations() -> None:
    report = grading.read_ctrf(
        json.dumps(
            {
                "results": {
                    "tests": [
                        *REPORT,
                        {
                            "name": "tests/test_retry.py::test_backoff[0.5]",
                            "status": "error",
                            "duration": 2.5,
                        },
                    ]
                }
            }
        ).encode()
    )
    view = json.loads(grading.tests_view(report))
    assert grading.tests_view(report).endswith("}\n")
    assert [t["id"] for t in view["tests"]][
        -1
    ] == "tests/test_retry.py::test_backoff[0.5] [2]"
    assert view["tests"][0] == {
        "duration_ms": 41,
        "id": "tests/test_retry.py::test_timeout_retries",
        "outcome": "passed",
    }
    assert (
        view["tests"][-1]["outcome"] == "other"
        and view["tests"][-1]["duration_ms"] == 3
    )
    assert view["summary"] == {"passed": 2, "failed": 1, "other": 1}
    assert grading.tests_view(grading.read_ctrf(None)) == '{"summary":{},"tests":[]}\n'


def _criteria() -> list[dict]:
    return [
        {"id": "gate", "gate": True, "judge": "test", "check": "x"},
        {"id": "earn", "points": 3, "judge": "test", "check": "y"},
        {"id": "charge", "points": -2, "judge": "test", "check": "z"},
        {
            "id": "lvl",
            "points": 4,
            "judge": "agent",
            "levels": {"0": "a", "2": "b", "4": "c"},
        },
        {"id": "cont", "points": 2, "judge": "llm", "score": {"min": 0, "max": 10}},
        {
            "id": "bad",
            "points": 2,
            "outcome": "bad",
            "judge": "llm",
            "score": {"min": 0, "max": 10},
        },
    ]


def test_score_by_the_rules_of_docs_rubrics() -> None:
    rubric = {"scoring": {"method": "points"}}
    full = {
        "gate": True,
        "earn": True,
        "charge": True,
        "lvl": "4",
        "cont": 10,
        "bad": 0,
    }
    result = grading.score(rubric, _criteria(), full)
    assert result.partial == 1 and result.strict and result.maximum == 11
    charged = grading.score(rubric, _criteria(), full | {"charge": False, "cont": 5})
    assert charged.earned == Fraction(
        3 + 0 + 4 + 1 + 2 - 2
    ) and charged.partial == Fraction(8, 11)
    assert not charged.strict
    gated = grading.score(rubric, _criteria(), full | {"gate": False})
    assert gated.partial == 0 and not gated.strict and gated.failed == ["gate"]
    only_gates = grading.score({}, [_criteria()[0]], {"gate": True})
    assert only_gates.partial == 1 and only_gates.strict
    total = grading.score(
        {"scoring": {"method": "sum"}}, _criteria(), full | {"charge": False}
    )
    assert total.partial == 9 and not total.strict
    threshold = grading.score(
        {"scoring": {"pass_threshold": 0.7}},
        _criteria(),
        full | {"charge": False, "cont": 5},
    )
    assert threshold.strict
    assert grading.headline({"scoring": {"headline": "strict"}}, threshold) == 1.0
    with pytest.raises(ValueError, match="not one of its levels"):
        grading.score(rubric, _criteria(), full | {"lvl": "3"})


def test_a_skip_earns_and_charges_nothing() -> None:
    earn, charge, gate = _criteria()[1], _criteria()[2], _criteria()[0]
    assert grading.skip_value(earn) is False
    assert grading.skip_value(charge) is True
    assert grading.skip_value(gate) is False
    assert grading.verdict_points(earn, None) is None
    assert grading.verdict_points(charge, False) == -2.0
    assert grading.verdict_points(_criteria()[3], "2") == 2.0


def _battery(seed: int = 7, count: int = 400) -> list[tuple[dict, list[dict], dict]]:
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        criteria = []
        verdicts = {}
        for n in range(rng.randint(1, 6)):
            kind = rng.choice(["gate", "points", "levels", "score"])
            cid = f"c{n}"
            if kind == "gate":
                criteria.append({"id": cid, "gate": True})
                verdicts[cid] = rng.random() < 0.8
            elif kind == "points":
                criteria.append({"id": cid, "points": rng.choice([-3, -1, 1, 2, 5])})
                verdicts[cid] = rng.random() < 0.6
            elif kind == "levels":
                criteria.append(
                    {"id": cid, "points": 4, "levels": {"0": "a", "2": "b", "4": "c"}}
                )
                verdicts[cid] = rng.choice(["0", "2", "4"])
            else:
                outcome = rng.choice(["good", "bad"])
                criteria.append(
                    {
                        "id": cid,
                        "points": rng.choice([-2, 3]),
                        "outcome": outcome,
                        "score": {"min": 0, "max": 10},
                    }
                )
                verdicts[cid] = rng.choice([0, 2.5, 7, 10])
        scoring = {
            "method": rng.choice(["points", "sum"]),
            "headline": rng.choice(["partial", "strict"]),
        }
        if rng.random() < 0.5:
            scoring["pass_threshold"] = (
                rng.choice([0.5, 0.7, 1.0])
                if scoring["method"] == "points"
                else rng.choice([1, 3])
            )
        cases.append(({"scoring": scoring}, criteria, verdicts))
    return cases


def test_score_equals_the_reference_rubrics_tool() -> None:
    """The port of tools/rubrics.py's score() gives its results on a random battery."""
    repo = require_taskmd_repo()
    cases = _battery()
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(repo / 'tools')!r})\n"
        "import rubrics\n"
        "out = []\n"
        "for rubric, criteria, verdicts in json.load(sys.stdin):\n"
        "    r = rubrics.score(dict(rubric, criteria=criteria), verdicts)\n"
        "    out.append({k: str(r[k]) for k in ('raw', 'reward', 'partial', 'strict', 'earned', 'maximum')} | {'failed': r['failed']})\n"
        "print(json.dumps(out))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    expected = json.loads(run.stdout)
    for (rubric, criteria, verdicts), want in zip(cases, expected, strict=True):
        got = grading.score(rubric, criteria, verdicts)
        assert {
            "raw": str(got.raw),
            "reward": str(got.reward),
            "partial": str(got.partial),
            "strict": str(got.strict),
            "earned": str(got.earned),
            "maximum": str(got.maximum),
            "failed": got.failed,
        } == want


def test_matching_equals_the_reference_rubrics_tool() -> None:
    repo = require_taskmd_repo()
    tests = [
        *REPORT,
        {"name": "a/b.py::t", "status": "passed", "filePath": "./a/b.py"},
        {"name": "plain", "status": "skipped"},
    ]
    checks = [
        "test_timeout_retries",
        "test_backoff",
        "test_backoff[1.0]",
        "tests/test_retry.py",
        "/tests/test_retry.py",
        "b.py",
        "a/b.py",
        "t",
        "plain",
        "*",
        "nothing",
    ]
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(repo / 'tools')!r})\n"
        "import rubrics\n"
        "tests, checks = json.load(sys.stdin)\n"
        "print(json.dumps([[t['name'] for t in rubrics.matching_tests(c, tests)] for c in checks]))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps([tests, checks]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout) == [
        [t["name"] for t in grading.matching_tests(c, tests)] for c in checks
    ]
