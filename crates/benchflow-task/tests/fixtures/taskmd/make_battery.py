"""Record task-md's reference scorer on a random battery, for tests/scoring_battery.rs.

    python3 make_battery.py <task-md checkout> <out.json>

Builds rubrics (gates, points, levels, continuous criteria; methods, headlines, thresholds), verdicts, and behavior
consequences with a fixed seed, and scores each with tools/rubrics.py's score(); builds CTRF test lists and checks, and
records matching_tests() and test_verdict(). Every rubric passes the reference checker, so the Rust side can load it as
a Rubric. Fractions are written as str(Fraction) ("3/4"); a case the scorer refuses records the exception's type.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(repo / "tools"))
import rubrics  # noqa: E402

SCHEMA = "https://task.md/schema/rubric-1.json"
rng = random.Random(20261010)


def criterion(n: int) -> tuple[dict, object]:
    """A criterion that passes the checker, and a verdict for it."""
    cid = f"c{n}"
    kind = rng.choice(["gate", "points", "points", "levels", "score"])
    base = {"id": cid, "text": "T.", "implicit": "x"}
    if kind == "gate":
        return base | {"gate": True, "judge": "test", "check": "t"}, rng.random() < 0.8
    if kind == "points":
        pts = rng.choice([-3, -1.5, -1, 0.5, 1, 2, 2.5, 5, 0.1, 0.3, 7, 0])
        return base | {"points": pts, "judge": "test", "check": "t"}, rng.random() < 0.6
    if kind == "levels":
        pts, levels = rng.choice([(4, ["0", "2", "4"]), (3, ["0", "1.5", "3"]), (1, ["0", "0.25", "0.5", "1"]), (2.5, ["0", "2.5"])])
        crit = base | {"points": pts, "judge": "agent", "evidence": ["r.md"], "levels": {k: f"level {k}" for k in levels}}
        roll = rng.random()
        if roll < 0.75:
            return crit, rng.choice(levels)
        if roll < 0.9:
            return crit, rng.choice([0, 1, 2.5, 0.1])  # a number of points
        return crit, rng.choice(["3", "0.75", "-1"])  # not one of its levels
    lo, hi = rng.choice([(0, 10), (-1, 1), (0, 100), (0.5, 2.5)])
    crit = base | {"points": rng.choice([-2, 3, 1.5, -0.5]), "judge": rng.choice(["llm", "human"]), "evidence": ["r.md"],
                   "outcome": rng.choice(["good", "bad"]), "score": {"min": lo, "max": hi}}
    return crit, rng.choice([lo, hi, (lo + hi) / 2, lo - 1, hi + 5, 0.3, 7, 2.5, 99])


def frac(x) -> str:
    return str(x)


cases = []
for _ in range(1500):
    crits, verdicts = [], {}
    for n in range(rng.randint(1, 7)):
        c, v = criterion(n)
        crits.append(c)
        verdicts[c["id"]] = v
    method = rng.choice(["points", "sum"])
    scoring = {"method": method, "headline": rng.choice(["partial", "strict"])}
    if rng.random() < 0.5:
        scoring["pass_threshold"] = rng.choice([0.5, 0.7, 1.0, 0, 0.25]) if method == "points" else rng.choice([1, 3, 2.5, 0])
    rubric = {"$schema": SCHEMA, "scoring": scoring, "criteria": crits}
    behaviors = rng.sample(["fail", "report", {"penalty": -0.2}, {"penalty": -0.5}, {"penalty": -1}, {"penalty": -0.05}],
                           rng.choice([0, 0, 0, 1, 1, 2]))
    try:
        r = rubrics.score(rubric, verdicts, behaviors)
        expected = {k: frac(r[k]) for k in ("raw", "reward", "partial", "earned", "maximum", "penalty")} | {
            "strict": r["strict"], "failed": r["failed"]}
    except Exception as exc:  # noqa: BLE001 - the battery records what the scorer refuses
        expected = {"error": type(exc).__name__}
    cases.append({"rubric": rubric, "verdicts": verdicts, "behaviors": behaviors, "expected": expected})

FILES = ["tests/test_retry.py", "./tests/test_retry.py", "/abs/tests/test_x.py", "a/b.py", "test_x.py"]
NAMES = ["test_timeout_retries", "test_backoff", "test_backoff[0.5]", "test_backoff[1.0]", "t", "plain", "test_x", "b"]
STATUS = ["passed", "passed", "passed", "failed", "skipped", "pending", "other"]
matching = []
for _ in range(500):
    tests = []
    for _ in range(rng.randint(0, 6)):
        name = rng.choice(NAMES)
        form = rng.random()
        if form < 0.5:
            name = f"{rng.choice(FILES).lstrip('./')}::{name}"
        elif form < 0.6:
            name = f"{rng.choice(FILES)}::{name}"
        test = {"name": name, "status": rng.choice(STATUS)}
        if rng.random() < 0.3:
            test["filePath"] = rng.choice(FILES + [""])
        tests.append(test)
    checks = [rng.choice(NAMES + FILES + ["*", " * ", "retry.py::test_timeout_retries", "test_retry.py::test_timeout_retries",
                                           "b.py", "nothing", "tests/test_retry.py::test_backoff[0.5]"]) for _ in range(4)]
    matching.append({"tests": tests, "checks": [
        {"check": c, "matched": [t["name"] for t in rubrics.matching_tests(c, tests)], "verdict": rubrics.test_verdict(c, tests)}
        for c in checks]})

out.write_text(json.dumps({"score": cases, "matching": matching}, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(cases)} score cases and {len(matching)} matching cases to {out}")
