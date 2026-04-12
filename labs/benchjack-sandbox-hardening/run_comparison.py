#!/usr/bin/env python3
"""Top-level orchestrator for the BenchJack sandbox-hardening labs demo.

Creates two isolated venvs, installs benchflow==0.2.0 in one and editable-HEAD
in the other, runs _attack_runner.py once per venv, and prints a two-row
comparison table.

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


def _run_in_venv(venv_dir: Path, label: str) -> dict:
    python = venv_dir / "bin" / "python"
    env = os.environ.copy()
    env["BENCHJACK_JOBS_DIR"] = str(JOBS_DIR / label)
    env["BENCHJACK_TRIAL_NAME"] = f"attack-{label}"

    proc = subprocess.run(
        [str(python), str(HERE / "_attack_runner.py")],
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


def _fmt_row(row: dict) -> str:
    version = row.get("version") or "?"
    reward = row.get("reward")

    if reward is None:
        state = f"ERROR ({row.get('error') or 'unknown'})"
        reward_str = "N/A"
    elif reward >= 0.999:
        state = "EXPLOITED — conftest.py hook fired"
        reward_str = f"{reward:.2f}"
    else:
        state = "BLOCKED — hardening layer fired"
        reward_str = f"{reward:.2f}"

    return f"benchflow {version:10s}  reward={reward_str:>6s}  {state}"


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

    print("[1/4] venv: benchflow==0.2.0 (PyPI)")
    _create_venv(VENVS_DIR / "bf-0.2.0", ["benchflow==0.2.0"])
    print("[2/4] venv: benchflow@HEAD (editable)")
    _create_venv(VENVS_DIR / "bf-head", ["-e", str(REPO_ROOT)])

    print("[3/4] run: benchflow 0.2.0 against pattern1_conftest_hook")
    r020 = _run_in_venv(VENVS_DIR / "bf-0.2.0", "0.2.0")
    print("[4/4] run: benchflow HEAD against pattern1_conftest_hook")
    rhead = _run_in_venv(VENVS_DIR / "bf-head", "head")

    print()
    print("=" * 72)
    print("BenchJack sandbox-hardening comparison (0.2.0 vs HEAD)")
    print("=" * 72)
    print(_fmt_row(r020))
    print(_fmt_row(rhead))
    print()

    r020_reward = r020.get("reward")
    rhead_reward = rhead.get("reward")
    ok = (
        r020_reward is not None
        and rhead_reward is not None
        and r020_reward >= 0.999
        and rhead_reward < 0.001
    )

    if ok:
        print("✓ Expected: exploit succeeded under 0.2.0, blocked under HEAD.")
        return 0

    print("✗ Unexpected outcome. Full payloads below.")
    print(json.dumps({"0.2.0": r020, "head": rhead}, indent=2, default=str))
    return 1


if __name__ == "__main__":
    sys.exit(main())
