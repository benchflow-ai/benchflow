"""Rubric grading for task.md draft 2: test reports, test-judged criteria, and the reward.

The verifier's ``test.sh`` reports what happened as a Common Test Report
Format (CTRF) file, ``/logs/verifier/ctrf.json``; it does not score the rubric
(docs/package.md, "Verifier scripts"). The runtime decides each
``judge: "test"`` criterion from the tests its ``check`` names, combines them
with the model judges' verdicts, and scores the rubric (docs/rubrics.md,
"Scoring").

:func:`matching_tests` and :func:`score` are ported from the reference
tools' ``tools/rubrics.py`` (``matching_tests``, ``score``) at the commit the
vendored parser pins; ``tests/test_taskmd_grading.py`` compares them with the
upstream file when a task-md checkout is at hand (``TASKMD_REPO``). The
rubrics module is not vendored because it adds its own folder to
``sys.path`` when imported.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from benchflow.taskmd._util import table
from benchflow.taskmd._vendor import judgeprompt as jp

CTRF_STATUSES = ("passed", "failed", "skipped", "pending", "other")
REVIEW_SCHEMA = "https://task.md/schema/review-1.json"


@dataclass(frozen=True)
class TestReport:
    """The tests of a verifier's CTRF report, or why there are none."""

    tests: list[dict[str, Any]]
    tool: str | None
    problem: str | None = None  # set when the report is missing or not a CTRF report


def read_ctrf(data: bytes | None) -> TestReport:
    """Parse ``ctrf.json``. A missing or malformed report leaves no tests.

    docs/package.md: "results.tests is required, and each test needs a name
    and a status". A report without them is not a CTRF report, so every
    test-judged criterion matches nothing and fails.
    """
    if data is None:
        return TestReport([], None, "the verifier wrote no ctrf.json")
    try:
        report = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return TestReport([], None, f"ctrf.json is not JSON: {exc}")
    results = report.get("results") if isinstance(report, dict) else None
    tests = results.get("tests") if isinstance(results, dict) else None
    if not isinstance(tests, list):
        return TestReport([], None, "ctrf.json has no results.tests list")
    checked: list[dict[str, Any]] = []
    for n, raw in enumerate(tests):
        test = table(raw)
        if not (
            isinstance(raw, dict)
            and isinstance(test.get("name"), str)
            and isinstance(test.get("status"), str)
        ):
            return TestReport(
                [], None, f"ctrf.json results.tests[{n}] needs a name and a status"
            )
        checked.append(test)
    tool = results.get("tool") if isinstance(results, dict) else None
    name = tool.get("name") if isinstance(tool, dict) else None
    version = tool.get("version") if isinstance(tool, dict) else None
    label = (
        " ".join(str(x) for x in (name, version) if isinstance(x, str) and x.strip())
        or None
    )
    return TestReport(checked, label)


def tests_view(report: TestReport) -> str:
    """``/judge/tests.json``: the JCS of ids, outcomes, and durations, then one LF (docs/runtime/judging.md)."""
    seen: Counter[str] = Counter()
    entries = []
    for test in report.tests:
        name = str(test["name"])
        seen[name] += 1
        ident = name if seen[name] == 1 else f"{name} [{seen[name]}]"
        status = test["status"] if test["status"] in CTRF_STATUSES else "other"
        duration = test.get("duration")
        ms = (
            math.floor(duration + 0.5)
            if isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            else 0
        )
        entries.append({"id": ident, "outcome": status, "duration_ms": ms})
    summary = dict(Counter(e["outcome"] for e in entries))
    return jp.jcs({"summary": summary, "tests": entries}) + "\n"


def matching_tests(check: str, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tests of a CTRF report that a test-judged criterion's check names (docs/package.md, Verifier scripts).

    ``"*"`` names every test. Otherwise a test matches by name: its name, or its
    name without a parameter suffix such as ``[0.5]``, equals the check or ends
    with ``::`` or ``/`` and the check. Only when no test matches by name, every
    test in the file the check names matches: its ``filePath``, or else its
    name before the first ``::``, equals the check or ends with ``/`` and the
    check, ignoring a leading ``./`` or ``/`` on either.
    """
    if check.strip() == "*":
        return list(tests)

    def named(name: str) -> bool:
        return any(
            n == check or n.endswith(("::" + check, "/" + check))
            for n in (name, name.split("[", 1)[0])
        )

    by_name = [t for t in tests if named(str(t.get("name", "")))]
    if by_name:
        return by_name

    def bare(path: str) -> str:
        return re.sub(r"^(?:\./|/)+", "", path)

    wanted = bare(check.strip())
    files = [
        (
            t,
            t.get("filePath")
            or (
                str(t.get("name", "")).split("::", 1)[0]
                if "::" in str(t.get("name", ""))
                else None
            ),
        )
        for t in tests
    ]
    return [
        t
        for t, f in files
        if f and (bare(f) == wanted or bare(f).endswith("/" + wanted))
    ]


def test_verdict(check: str, tests: list[dict[str, Any]]) -> bool:
    """A test-judged criterion passes when a test matches its check and every test that matches passed."""
    matched = matching_tests(check, tests)
    return bool(matched) and all(t.get("status") == "passed" for t in matched)


@dataclass
class Score:
    raw: Fraction
    reward: Fraction
    partial: Fraction
    strict: bool
    failed: list[str]
    earned: Fraction
    maximum: Fraction
    penalty: Fraction = field(default_factory=Fraction)


def score(
    rubric: dict[str, Any],
    criteria: list[dict[str, Any]],
    verdicts: dict[str, Any],
    behaviors: tuple[Any, ...] = (),
) -> Score:
    """Score one trial as docs/rubrics.md defines it (``tools/rubrics.py`` ``score``).

    ``criteria`` is the merged rubric's criteria. ``verdicts`` maps each
    criterion id to True or False for pass or fail (pass means the task went as
    it should, whatever the criterion's outcome), a level key for a criterion
    with levels, or the judge's number for a continuous criterion. ``behaviors``
    holds the consequences of detected behaviors, such as ``"fail"`` or
    ``{"penalty": -0.2}``.
    """
    scoring = rubric.get("scoring") or {}
    method = scoring.get("method", "points")
    earned, maximum, failed = Fraction(0), Fraction(0), []
    for c in criteria:
        v = verdicts[c["id"]]
        if c.get("gate"):
            if v is not True:
                failed.append(c["id"])
            continue
        pts = Fraction(str(c.get("points", 0)))
        if pts > 0:
            maximum += pts
        if isinstance(v, bool):
            # Positive points are earned on pass; negative points are charged on fail.
            earned += (
                (pts if v else Fraction(0)) if pts > 0 else (Fraction(0) if v else pts)
            )
        elif c.get("score"):
            # A continuous criterion: v says how far its text holds. For a bad outcome, holding is going wrong.
            lo, hi = Fraction(str(c["score"]["min"])), Fraction(str(c["score"]["max"]))
            f = min(max((Fraction(str(v)) - lo) / (hi - lo), Fraction(0)), Fraction(1))
            well = 1 - f if c.get("outcome") == "bad" else f
            earned += pts * well if pts > 0 else pts * (1 - well)
        else:
            if isinstance(v, str) and c.get("levels") and v not in c["levels"]:
                raise ValueError(
                    f"{c['id']}: {v!r} is not one of its levels"
                )  # judges pick a level; they do not invent scores
            earned += Fraction(str(v))
    if method == "sum":
        base = earned  # neither divided nor clipped
    elif maximum:
        base = earned / maximum
    else:
        base = (
            1 + earned
        )  # no positive points: 1 when nothing is charged, less the charges
    penalty = sum(
        (
            Fraction(str(b["penalty"]))
            for b in behaviors
            if isinstance(b, dict) and "penalty" in b
        ),
        Fraction(0),
    )
    fail = any(b == "fail" for b in behaviors)
    partial = base if method == "sum" else min(max(base, Fraction(0)), Fraction(1))
    if failed or fail:
        partial = Fraction(0)
    elif penalty:
        partial = max(
            partial + penalty, min(partial, Fraction(0))
        )  # floored at 0, and never lifting a negative sum
    threshold = Fraction(
        str(scoring.get("pass_threshold", maximum if method == "sum" else 1))
    )
    return Score(
        raw=base + penalty,
        reward=partial,
        partial=partial,
        strict=not failed and not fail and partial >= threshold,
        failed=failed,
        earned=earned,
        maximum=maximum,
        penalty=penalty,
    )


def headline(rubric: dict[str, Any], result: Score) -> float:
    """The reward written to reward.txt: the partial score, or strict when the rubric's headline says so."""
    head = (rubric.get("scoring") or {}).get("headline", "partial")
    return (
        float(1 if result.strict else 0) if head == "strict" else float(result.partial)
    )


def verdict_points(criterion: dict[str, Any], value: Any) -> float | None:
    """What one verdict adds to the total (a review's ``score``): None for a gate or a skip."""
    if criterion.get("gate") or value is None:
        return None
    pts = Fraction(str(criterion.get("points", 0)))
    if isinstance(value, bool):
        return float((pts if value else 0) if pts > 0 else (0 if value else pts))
    if criterion.get("score"):
        lo, hi = (
            Fraction(str(criterion["score"]["min"])),
            Fraction(str(criterion["score"]["max"])),
        )
        f = min(max((Fraction(str(value)) - lo) / (hi - lo), Fraction(0)), Fraction(1))
        well = 1 - f if criterion.get("outcome") == "bad" else f
        return float(pts * well if pts > 0 else pts * (1 - well))
    return float(Fraction(str(value)))


def skip_value(criterion: dict[str, Any]) -> bool:
    """The score() input for a criterion recorded as ``skip``: it earns and charges nothing.

    A skip scores nothing and is never a pass: a gate fails, positive points
    are not earned, and negative points are not charged.
    """
    if criterion.get("gate"):
        return False
    return Fraction(str(criterion.get("points", 0))) < 0
