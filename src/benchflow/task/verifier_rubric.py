"""Grade a task.md rubric (``verifier/rubric.json``) from a CTRF test report.

task.md draft 1 separates reporting from scoring. The verifier's test script
reports what happened as a Common Test Report Format file at
``/logs/verifier/ctrf.json``; it does not score the rubric. The runtime decides
each ``judge: "test"`` criterion from the tests its ``check`` names, then
scores the rubric (task.md ``docs/rubrics.md``, "Scoring"):

- If any gate fails, the reward is 0.
- Otherwise ``partial`` is the positive points earned minus the penalties that
  apply, over the maximum positive points, clipped to [0, 1]. A rubric with
  only gates is all-or-nothing.
- ``strict`` is 1 when all gates pass and ``partial`` reaches
  ``scoring.pass_threshold`` (default 1.0).
- ``scoring.headline`` says which of the two is the reward, and defaults to
  ``partial``.

A check is a test name or a test file. It matches by name first: every test
whose name, with any pytest parameter id dropped, equals the check or ends with
``::<check>`` or ``/<check>``. pytest's CTRF plugin names tests by node id
(``test_outputs.py::test_mass``), so ``test_mass`` names them. If no name
matches, the check is read as a file: every test whose file (CTRF ``filePath``,
the pytest plugin's ``file_path``, or the node id before ``::``) equals the
check or ends with ``/<check>``. Every matching test must pass; a skipped or
pending test does not pass.

A test-judged criterion passes or fails, so it has no ``levels``. A penalty
criterion names the bad outcome; its test asserts the good behavior, so the
penalty applies when the test fails.

This module is pure: no filesystem or sandbox access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

REVIEW_SCHEMA = "https://task.md/schema/review-1.json"
_HEADLINES = ("strict", "partial")
_THRESHOLD_TOLERANCE = 1e-9


class CtrfReportError(ValueError):
    """The verifier's CTRF report is not a usable test report."""


@dataclass(frozen=True)
class RubricGrade:
    """The verdicts and scores for one graded rubric."""

    verdicts: tuple[dict[str, Any], ...]
    strict: float
    partial: float
    reward: float
    unmatched: tuple[tuple[str, str], ...]  # (criterion id, check) naming no test


def rubric_gaps(rubric: dict[str, Any]) -> list[str]:
    """Why this runtime cannot grade ``rubric``; empty when it can."""

    gaps: list[str] = []
    if rubric.get("extends"):
        gaps.append(
            "extends pulls in a shared rubric this runtime does not fetch or merge"
        )
    gaps += _scoring_gaps(rubric.get("scoring"))
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return [*gaps, "criteria must be a non-empty list"]
    if not all(isinstance(c, dict) for c in criteria):
        return [*gaps, "every criterion must be an object"]

    judges = sorted({str(c.get("judge", "llm")) for c in criteria} - {"test"})
    if judges:
        gaps.append(
            f"criteria judged by {', '.join(judges)} are not graded by this runtime"
        )
    ids = [c.get("id") for c in criteria]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        gaps.append("every criterion needs a unique id")
    for c in criteria:
        where = f"criterion {c.get('id')!r}"
        if c.get("remove"):
            gaps.append(f"{where} removes a criterion, which needs extends")
        is_gate = c.get("gate") is True
        points = c.get("points")
        if is_gate and "points" in c:
            gaps.append(f"{where} is both a gate and a points criterion")
        elif not is_gate and not _is_number(points):
            gaps.append(f"{where} needs gate: true or finite points")
        if c.get("judge", "llm") == "test":
            check = c.get("check")
            if not isinstance(check, str) or not check.strip():
                gaps.append(f"{where} needs a check naming a test or a test file")
            if "levels" in c:
                gaps.append(f"{where} has levels, but a test passes or fails")

    point_values = [c["points"] for c in criteria if _is_number(c.get("points"))]
    max_positive = sum(p for p in point_values if p > 0)
    if max_positive == 0 and any(p < 0 for p in point_values):
        gaps.append(
            "penalties with no positive points leave the partial score undefined"
        )
    return gaps


def _scoring_gaps(scoring: Any) -> list[str]:
    if scoring is None:
        return []
    if not isinstance(scoring, dict):
        return ["scoring must be an object"]
    gaps: list[str] = []
    if scoring.get("method", "points") != "points":
        gaps.append('scoring.method other than "points" is not supported')
    if scoring.get("gates", "all") != "all":
        gaps.append('scoring.gates other than "all" is not supported')
    if scoring.get("headline", "partial") not in _HEADLINES:
        gaps.append('scoring.headline must be "strict" or "partial"')
    threshold = scoring.get("pass_threshold", 1.0)
    if not _is_number(threshold) or not 0 <= threshold <= 1:
        gaps.append("scoring.pass_threshold must be a number from 0 to 1")
    return gaps


def ctrf_tests(report: Any) -> list[dict[str, Any]]:
    """The test entries of a CTRF report, validated.

    Raises :class:`CtrfReportError` for anything but ``{"results": {"tests":
    [{"name": str, "status": str, ...}, ...]}}``.
    """

    results = report.get("results") if isinstance(report, dict) else None
    tests = results.get("tests") if isinstance(results, dict) else None
    if not isinstance(tests, list):
        raise CtrfReportError('expected {"results": {"tests": [...]}}')
    for index, test in enumerate(tests):
        if not _is_test_entry(test):
            raise CtrfReportError(f"results.tests[{index}] needs a name and a status")
    return tests


def _is_test_entry(test: Any) -> bool:
    return (
        isinstance(test, dict)
        and isinstance(test.get("name"), str)
        and isinstance(test.get("status"), str)
    )


def grade_rubric(rubric: dict[str, Any], tests: list[dict[str, Any]]) -> RubricGrade:
    """Decide every criterion from ``tests`` and score the rubric.

    ``rubric`` must be gradable (:func:`rubric_gaps` returns nothing); anything
    else raises ``ValueError`` rather than producing a score.
    """

    gaps = rubric_gaps(rubric)
    if gaps:
        raise ValueError("rubric cannot be graded: " + "; ".join(gaps))

    verdicts: list[dict[str, Any]] = []
    unmatched: list[tuple[str, str]] = []
    gates_pass = True
    earned = penalties = max_positive = 0.0
    for criterion in rubric["criteria"]:
        check = criterion["check"]
        matched = _matching_tests(tests, check)
        not_passed = [t for t in matched if t["status"] != "passed"]
        passed = bool(matched) and not not_passed
        if not matched:
            unmatched.append((criterion["id"], check))

        points = criterion.get("points")
        if criterion.get("gate") is True:
            score = 1.0 if passed else 0.0
            gates_pass = gates_pass and passed
        elif points > 0:
            max_positive += points
            score = float(points) if passed else 0.0
            earned += score
        else:  # a penalty, or zero points: applies when the check fails
            score = 0.0 if passed else float(points)
            penalties -= score

        if not matched:
            rationale = "no test in ctrf.json matches this check"
        elif passed:
            rationale = f"all {len(matched)} matching tests passed"
        else:
            rationale = "not passed: " + ", ".join(
                f"{t['name']} ({t['status']})" for t in not_passed
            )
        verdicts.append(
            {
                "id": criterion["id"],
                "verdict": "pass" if passed else "fail",
                "score": score,
                "judge": {"role": "test"},
                "citations": [
                    {"source": "ctrf", "ref": name, "verified": True}
                    for name in dict.fromkeys(t["name"] for t in matched)
                ],
                "rationale": rationale,
            }
        )

    scoring = rubric.get("scoring") or {}
    if not gates_pass:
        partial = strict = 0.0
    else:
        partial = (
            min(1.0, max(0.0, (earned - penalties) / max_positive))
            if max_positive > 0
            else 1.0
        )
        threshold = float(scoring.get("pass_threshold", 1.0))
        strict = 1.0 if partial >= threshold - _THRESHOLD_TOLERANCE else 0.0
    headline = scoring.get("headline", "partial")
    return RubricGrade(
        verdicts=tuple(verdicts),
        strict=strict,
        partial=partial,
        reward=strict if headline == "strict" else partial,
        unmatched=tuple(unmatched),
    )


def review_document(rubric: dict[str, Any], grade: RubricGrade) -> dict[str, Any]:
    """``review.json``: one verdict per criterion."""

    review: dict[str, Any] = {"$schema": REVIEW_SCHEMA}
    if isinstance(rubric.get("version"), str):
        review["rubric_version"] = rubric["version"]
    review["verdicts"] = list(grade.verdicts)
    return review


def _matching_tests(tests: list[dict[str, Any]], check: str) -> list[dict[str, Any]]:
    by_name = [t for t in tests if _name_matches(t["name"], check)]
    if by_name:
        return by_name
    wanted = PurePosixPath(check).as_posix()
    return [
        t
        for t in tests
        if (path := _test_file(t)) is not None
        and (path == wanted or path.endswith("/" + wanted))
    ]


def _name_matches(name: str, check: str) -> bool:
    for candidate in (name, name.split("[", 1)[0]):
        if candidate == check or candidate.endswith(("::" + check, "/" + check)):
            return True
    return False


def _test_file(test: dict[str, Any]) -> str | None:
    for key in ("filePath", "file_path"):
        value = test.get(key)
        if isinstance(value, str) and value.strip():
            return PurePosixPath(value.strip()).as_posix()
    name = test["name"]
    return PurePosixPath(name.split("::", 1)[0]).as_posix() if "::" in name else None


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
