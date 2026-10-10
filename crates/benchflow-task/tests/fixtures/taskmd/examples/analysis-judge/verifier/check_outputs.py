"""The test-judged criteria of analysis-judge, written as a CTRF report.

Reads the handoff in <submission-dir> and the generating values in <answers.json>. Stdlib only, so it runs in a bare
Python image with no network.

    test_outputs_present     fit.json parses and holds finite numbers alpha, alpha_err, xmin, and n_tail; report.md is
                             not empty; fit.png is a PNG
    test_alpha_accurate      alpha is within alpha_tolerance of the generating value
    test_uncertainty_covers  0 < alpha_err <= max_alpha_err, and alpha is within 2 alpha_err of the generating value

Usage: python3 check_outputs.py <submission-dir> <answers.json> <ctrf-out>
"""

import json
import math
import sys
import time
from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n"
KEYS = ("alpha", "alpha_err", "xmin", "n_tail")


def number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def main(sub: str, answers: str, out: str) -> None:
    start = time.time()
    sub_dir, ans = Path(sub), json.loads(Path(answers).read_text())
    tests: list[dict] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        tests.append({"name": name, "status": "passed" if ok else "failed", "duration": 0} | ({"message": message} if message and not ok else {}))

    fit, problems = None, []
    try:
        fit = json.loads((sub_dir / "fit.json").read_text())
        if not isinstance(fit, dict):
            problems.append("fit.json is not a JSON object")
            fit = None
    except (OSError, ValueError) as e:
        problems.append(f"fit.json: {type(e).__name__}")
    fit_ok = fit is not None and all(number(fit.get(k)) for k in KEYS)
    if fit is not None and not fit_ok:
        problems.append("fit.json lacks finite numbers for " + ", ".join(k for k in KEYS if not number(fit.get(k))))
    report = sub_dir / "report.md"
    if not (report.is_file() and report.read_text(errors="replace").strip()):
        problems.append("report.md is missing or empty")
    png = sub_dir / "fit.png"
    if not (png.is_file() and png.read_bytes()[:8] == PNG):
        problems.append("fit.png is missing or not a PNG")
    record("test_outputs_present", not problems, "; ".join(problems))

    a = fit["alpha"] if fit_ok else None
    e = fit["alpha_err"] if fit_ok else None
    record("test_alpha_accurate", fit_ok and abs(a - ans["alpha"]) <= ans["alpha_tolerance"],
           "no usable alpha" if not fit_ok else f"alpha {a} is off by more than the tolerance")
    record("test_uncertainty_covers", fit_ok and 0 < e <= ans["max_alpha_err"] and abs(a - ans["alpha"]) <= 2 * e,
           "no usable alpha_err" if not fit_ok else f"alpha {a} with alpha_err {e} does not cover the generating value, or alpha_err is out of range")

    passed = sum(t["status"] == "passed" for t in tests)
    report_json = {"results": {
        "tool": {"name": "check_outputs.py", "version": "1.0.0"},
        "summary": {"tests": len(tests), "passed": passed, "failed": len(tests) - passed, "pending": 0, "skipped": 0, "other": 0,
                    "start": int(start * 1000), "stop": int(time.time() * 1000)},
        "tests": tests}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report_json, indent=2) + "\n")
    print(f"{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main(*sys.argv[1:4])
