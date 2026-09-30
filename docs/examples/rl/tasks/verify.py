"""Verifier for the RL cookbook task family.

Every task's ``verifier/`` holds a copy of this file and the instance's
``expected.json``. It writes ``1`` or ``0`` to ``/logs/verifier/reward.txt``
and exits 0; a nonzero exit means the verifier itself broke.

- Question tasks compare ``/workdir/answer.txt`` with the expected answer.
- Bug-fix tasks run the fixed module's functions on hidden inputs, in a
  process with the unprivileged ``nobody`` identity, and compare the results
  with outputs of the reference functions. The comparison runs here, in the
  root verifier process; the checked code never sees the expected outputs.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ANSWER = Path("/workdir/answer.txt")
REWARD = Path("/logs/verifier/reward.txt")
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


def _text(value: str) -> str:
    return " ".join(value.strip().strip("\"'`").rstrip(".").split()).casefold()


def _number(value: str) -> float | None:
    cleaned = value.strip().strip("\"'`").rstrip(".").replace(",", "").replace("$", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def check_answer(expected: dict[str, Any]) -> tuple[bool, str]:
    try:
        raw = ANSWER.read_text(errors="replace").strip()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False, "no answer in /workdir/answer.txt"
    if not raw or len(raw) > 200:
        return False, f"answer is empty or too long ({len(raw)} characters)"
    want = expected["answer"]
    if expected["type"] == "text":
        got, target = _text(raw), _text(want)
        if got.isdigit() and target.isdigit():
            return int(got) == int(target), f"got {raw!r}"
        return got == target, f"got {raw!r}"
    number = _number(raw)
    if number is None:
        return False, f"not a number: {raw!r}"
    tolerance = float(expected.get("tolerance") or 0.0)
    return abs(number - float(want)) <= tolerance + 1e-9, f"got {raw!r}"


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


def check_bugfix(expected: dict[str, Any]) -> tuple[bool, str]:
    module = expected["module"]
    path = Path("/workdir") / f"{module}.py"
    if not path.is_file():
        return False, f"{path} is missing"
    checks = [
        {"function": c["function"], "cases": c["cases"]} for c in expected["checks"]
    ]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        "setpriv",
        f"--reuid={NOBODY}",
        f"--regid={NOBODY}",
        "--clear-groups",
        sys.executable,
        "-I",
        "-B",
        "-c",
        RUNNER,
        str(path),
        module,
    ]
    try:
        run = subprocess.run(
            command,
            input=json.dumps(checks),
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/tmp",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "the fixed module ran longer than 10 seconds"
    finally:
        # Nothing the checked code started may outlive the check.
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["pkill", "-9", "-u", NOBODY], capture_output=True, timeout=10
            )
    lines = [line for line in run.stdout.splitlines() if line.startswith(MARKER)]
    if run.returncode != 0 or not lines:
        tail = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return False, f"the module did not run: {tail[0][:200]}"
    try:
        results = json.loads(lines[-1][len(MARKER) :])
    except json.JSONDecodeError:
        return False, "unreadable results"
    if not isinstance(results, list) or len(results) != len(expected["checks"]):
        return False, "missing results"
    failed = [
        check["function"]
        for check, got in zip(expected["checks"], results, strict=True)
        if not _same(got, check["outputs"])
    ]
    return not failed, (
        "hidden checks passed" if not failed else f"hidden checks failed: {failed}"
    )


def main() -> None:
    # The verifier runs as root; keep the expected answers from the unprivileged process.
    os.chmod(HERE, 0o700)
    expected = json.loads((HERE / "expected.json").read_text())
    passed, note = (
        check_bugfix(expected)
        if expected["type"] == "bugfix"
        else check_answer(expected)
    )
    print(f"{'PASS' if passed else 'FAIL'}: {note}")
    REWARD.parent.mkdir(parents=True, exist_ok=True)
    # The reward is this file alone: never trust a reward file found in place.
    (REWARD.parent / "reward.json").unlink(missing_ok=True)
    REWARD.write_text("1\n" if passed else "0\n")


if __name__ == "__main__":
    main()
