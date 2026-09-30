"""Verifier for the RL cookbook task family.

Every task's ``verifier/`` holds a copy of this file and the instance's
``expected.json``. The reward is the fraction of checks passed, so a group of
rollouts rarely scores all 0 or all 1; ``passed`` records whether every check
passed. It writes:

- ``/logs/verifier/reward.txt``: the reward, from 0 to 1;
- ``/logs/verifier/reward.json``: ``{"reward": ..., "passed": 0 or 1}``;
- ``/logs/verifier/checks.json``: each check's result, for audit.

It exits 0 whatever the score; a nonzero exit means the verifier itself broke.

- Question tasks: each answer line of ``/workdir/answer.txt`` is one check.
- Bug-fix tasks: the fixed module's functions run on hidden inputs, in a
  process with the unprivileged ``nobody`` identity, and the results are
  compared here, in the root verifier process; the checked code never sees the
  expected outputs. The reward is the share of the cases the bug broke that
  now pass, times the share of the other cases that still pass, so doing
  nothing scores 0.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ANSWER = Path("/workdir/answer.txt")
LOGS = Path("/logs/verifier")
NOBODY = "65534"
MARKER = "@@benchflow-rl-results@@"
RUNNER = f"""
import importlib.util, json, sys
path, name = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = []
for check in json.load(sys.stdin):
    fn = getattr(module, check["function"], None)
    outputs = []
    for args in check["cases"]:
        try:
            outputs.append(json.loads(json.dumps(fn(*args))))
        except Exception as exc:
            outputs.append({{"__raised__": type(exc).__name__}})
    results.append(outputs)
sys.stdout.write("\\n{MARKER}" + json.dumps(results) + "\\n")
"""
# "1. 42" or "Q2) north": a question number, then a separator and a space. A
# bare "267.32" or "10.0.3.17" is an answer, never a number to strip.
NUMBERING = re.compile(r"^(?:q(?:uestion)?\s*)?\d+\s*[.):-]\s+", re.IGNORECASE)


def _text(value: str) -> str:
    return " ".join(value.strip().strip("\"'`").rstrip(".").split()).casefold()


def _number(value: str) -> float | None:
    cleaned = value.strip().strip("\"'`").rstrip(".").replace(",", "").replace("$", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _correct(raw: str, expected: dict[str, Any]) -> bool:
    want = expected["answer"]
    if expected["type"] == "text":
        got, target = _text(raw), _text(want)
        if got.isdigit() and target.isdigit():
            return int(got) == int(target)
        return got == target
    number = _number(raw)
    if number is None:
        return False
    return abs(number - float(want)) <= float(expected.get("tolerance") or 0.0) + 1e-9


def check_answers(expected: dict[str, Any]) -> tuple[float, bool, list[dict[str, Any]]]:
    wanted = expected["answers"]
    try:
        text = ANSWER.read_text(errors="replace")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return (
            0.0,
            False,
            [{"check": "answer file", "passed": False, "note": "missing"}],
        )
    lines = [NUMBERING.sub("", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > len(wanted) or len(text) > 2000:
        note = f"{len(lines)} answer lines for {len(wanted)} questions"
        return 0.0, False, [{"check": "answer file", "passed": False, "note": note}]
    results = []
    for index, want in enumerate(wanted):
        got = lines[index] if index < len(lines) else None
        passed = got is not None and _correct(got, want)
        results.append({"check": f"answer {index + 1}", "passed": passed, "got": got})
    score = sum(r["passed"] for r in results) / len(wanted)
    return score, score == 1.0, results


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, float) or isinstance(got, float):
        if isinstance(got, bool) or isinstance(want, bool):
            return got == want
        if not isinstance(got, int | float) or not isinstance(want, int | float):
            return False
        return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(want, list):
        return (
            isinstance(got, list)
            and len(got) == len(want)
            and all(_same(g, w) for g, w in zip(got, want, strict=True))
        )
    if isinstance(want, dict):
        return (
            isinstance(got, dict)
            and got.keys() == want.keys()
            and all(_same(got[k], want[k]) for k in want)
        )
    return type(got) is type(want) and got == want


def _run_module(expected: dict[str, Any]) -> tuple[list | None, str]:
    path = Path("/workdir") / f"{expected['module']}.py"
    if not path.is_file():
        return None, f"{path} is missing"
    checks = [
        {"function": c["function"], "cases": c["cases"]} for c in expected["checks"]
    ]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"}
    command = [
        "setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups",
        sys.executable, "-I", "-B", "-c", RUNNER, str(path), expected["module"],
    ]  # fmt: skip
    try:
        run = subprocess.run(
            command, input=json.dumps(checks), capture_output=True, text=True,
            timeout=10, cwd="/tmp", env=env,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return None, "the module ran longer than 10 seconds"
    finally:
        # Nothing the checked code started may outlive the check.
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["pkill", "-9", "-u", NOBODY], capture_output=True, timeout=10
            )
    lines = [line for line in run.stdout.splitlines() if line.startswith(MARKER)]
    if run.returncode != 0 or not lines:
        tail = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return None, f"the module did not run: {tail[0][:200]}"
    try:
        results = json.loads(lines[-1][len(MARKER) :])
    except json.JSONDecodeError:
        return None, "unreadable results"
    if not isinstance(results, list) or len(results) != len(expected["checks"]):
        return None, "missing results"
    return results, "ran"


def check_bugfix(expected: dict[str, Any]) -> tuple[float, bool, list[dict[str, Any]]]:
    results, note = _run_module(expected)
    if results is None:
        return 0.0, False, [{"check": "module", "passed": False, "note": note}]
    outcome = []
    for index, check in enumerate(expected["checks"]):
        got = results[index] if isinstance(results[index], list) else []
        for case, want in enumerate(check["outputs"]):
            passed = case < len(got) and _same(got[case], want)
            outcome.append(
                {
                    "check": f"{check['function']}#{case}",
                    "index": [index, case],
                    "passed": passed,
                }
            )
    broken = {tuple(pair) for pair in expected["broken"]}
    repaired = [o for o in outcome if tuple(o["index"]) in broken]
    kept = [o for o in outcome if tuple(o["index"]) not in broken]
    fixed_share = (
        sum(o["passed"] for o in repaired) / len(repaired) if repaired else 1.0
    )
    kept_share = sum(o["passed"] for o in kept) / len(kept) if kept else 1.0
    return fixed_share * kept_share, all(o["passed"] for o in outcome), outcome


def main() -> None:
    # The verifier runs as root; keep the expected answers from the unprivileged process.
    os.chmod(HERE, 0o700)
    expected = json.loads((HERE / "expected.json").read_text())
    check = check_bugfix if expected["type"] == "bugfix" else check_answers
    reward, passed, details = check(expected)
    reward = round(reward, 4)
    summary = ", ".join(
        f"{d['check']}={'ok' if d['passed'] else 'no'}" for d in details[:12]
    )
    print(f"reward {reward} passed {passed}: {summary}")
    LOGS.mkdir(parents=True, exist_ok=True)
    # Written fresh, whatever was there: the reward is only what this run computed.
    (LOGS / "reward.json").write_text(
        json.dumps({"reward": reward, "passed": float(passed)}) + "\n"
    )
    (LOGS / "checks.json").write_text(json.dumps(details, indent=1) + "\n")
    (LOGS / "reward.txt").write_text(f"{reward}\n")


if __name__ == "__main__":
    main()
