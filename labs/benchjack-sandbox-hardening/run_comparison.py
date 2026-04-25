#!/usr/bin/env python3
"""Top-level orchestrator for the BenchJack sandbox-hardening labs demo.

Creates two isolated venvs, installs benchflow==0.2.0 in one and editable-HEAD
in the other, runs _attack_runner.py once per venv × pattern, and prints a
pattern-first comparison table.

Usage:
    python run_comparison.py          # create venvs if missing and run
    python run_comparison.py --clean  # delete .venvs/ and .jobs/ first

Requires:
    * Docker daemon running and accessible to the current user
    * Python 3.10+
    * `uv` on PATH (preferred), falls back to `python -m venv`
    * Network access to PyPI on first run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VENVS_DIR = HERE / ".venvs"
JOBS_DIR = HERE / ".jobs"

# Shipped patterns: (id, display_name, task_dir)
PATTERNS: list[tuple[str, str, Path]] = [
    ("P1", "conftest-hook", HERE / "pattern1_conftest_hook"),
    ("P2", "answer-lookup", HERE / "pattern2_answer_lookup"),
    ("P7", "pth-injection", HERE / "pattern7_pth_injection"),
]


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _create_venv(venv_dir: Path, spec: list[str]) -> None:
    if (venv_dir / "bin" / "python").exists():
        return
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    python = venv_dir / "bin" / "python"

    if _have_uv():
        subprocess.check_call(["uv", "venv", str(venv_dir)])
        subprocess.check_call(
            ["uv", "pip", "install", "--python", str(python), *spec]
        )
        return

    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    subprocess.check_call(
        [str(venv_dir / "bin" / "pip"), "install", "--upgrade", "pip"]
    )
    subprocess.check_call([str(venv_dir / "bin" / "pip"), "install", *spec])


def _run_in_venv(venv_dir: Path, label: str, task_path: Path) -> dict:
    python = venv_dir / "bin" / "python"
    env = os.environ.copy()
    env["BENCHJACK_JOBS_DIR"] = str(JOBS_DIR / label / task_path.name)
    env["BENCHJACK_TRIAL_NAME"] = f"attack-{label}-{task_path.name}"

    proc = subprocess.run(
        [str(python), str(HERE / "_attack_runner.py"), str(task_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    payload: dict | None = None
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        payload = {
            "version": None,
            "reward": None,
            "error": f"no JSON on stdout (rc={proc.returncode})",
        }

    payload["label"] = label
    payload["returncode"] = proc.returncode
    if proc.stderr:
        payload["stderr_tail"] = proc.stderr[-2000:]
    return payload


def _fmt_cell(row: dict) -> str:
    """Format one result cell (version × pattern) for the table."""
    reward = row.get("reward")
    if reward is None:
        return "   N/A  ERROR    "
    elif reward >= 0.999:
        return f"  {reward:.2f}  EXPLOITED"
    else:
        return f"  {reward:.2f}  BLOCKED  "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete .venvs/ and .jobs/ before running",
    )
    args = ap.parse_args()

    if args.clean:
        for d in (VENVS_DIR, JOBS_DIR):
            if d.exists():
                print(f"[clean] removing {d}")
                shutil.rmtree(d)

    # Pre-flight: verify all task directories exist
    missing = [str(td) for _, _, td in PATTERNS if not td.is_dir()]
    if missing:
        print("ERROR: missing task directories:")
        for m in missing:
            print(f"  {m}")
        return 1

    n = len(PATTERNS)
    total_steps = 2 + 2 * n  # 2 venvs + 2 versions x n patterns

    print(f"[1/{total_steps}] venv: benchflow==0.2.0 (PyPI)")
    _create_venv(VENVS_DIR / "bf-0.2.0", ["benchflow==0.2.0"])
    print(f"[2/{total_steps}] venv: benchflow@HEAD (editable)")
    _create_venv(VENVS_DIR / "bf-head", ["-e", str(REPO_ROOT)])

    results: dict[str, dict[str, dict]] = {}  # pattern_id -> {label -> payload}
    step = 3
    for pat_id, pat_name, task_path in PATTERNS:
        results[pat_id] = {}
        print(f"[{step}/{total_steps}] run: benchflow 0.2.0 against {pat_id} {pat_name}")
        results[pat_id]["0.2.0"] = _run_in_venv(VENVS_DIR / "bf-0.2.0", "0.2.0", task_path)
        step += 1
        print(f"[{step}/{total_steps}] run: benchflow HEAD  against {pat_id} {pat_name}")
        results[pat_id]["head"] = _run_in_venv(VENVS_DIR / "bf-head", "head", task_path)
        step += 1

    # Print pattern-first table
    print()
    print("=" * 72)
    print("BenchJack sandbox-hardening comparison (0.2.0 vs HEAD)")
    print("=" * 72)
    print(f"{'pattern':<24}  {'benchflow 0.2.0':<20}  {'benchflow HEAD'}")
    print("-" * 72)
    for pat_id, pat_name, _ in PATTERNS:
        cell_020 = _fmt_cell(results[pat_id]["0.2.0"])
        cell_head = _fmt_cell(results[pat_id]["head"])
        label = f"{pat_id} {pat_name}"
        print(f"{label:<24}  {cell_020}  {cell_head}")
    print()

    # Pass/fail assertion
    all_ok = True
    failures: list[dict] = []
    for pat_id, pat_name, _ in PATTERNS:
        r020 = results[pat_id]["0.2.0"].get("reward")
        rhead = results[pat_id]["head"].get("reward")
        ok = (
            r020 is not None
            and rhead is not None
            and r020 >= 0.999
            and rhead < 0.001
        )
        if not ok:
            all_ok = False
            failures.append({
                "pattern": f"{pat_id} {pat_name}",
                "0.2.0": results[pat_id]["0.2.0"],
                "head": results[pat_id]["head"],
            })

    if all_ok:
        print("✓ All patterns: exploit succeeded under 0.2.0, blocked under HEAD.")
        return 0

    print("✗ Unexpected outcome(s). Full payloads for failed patterns below.")
    print(json.dumps(failures, indent=2, default=str))
    return 1


if __name__ == "__main__":
    sys.exit(main())
