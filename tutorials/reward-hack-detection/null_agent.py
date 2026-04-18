#!/usr/bin/env python3
"""Null agent baseline.

Demonstrates that a benchmark is compromised when an agent doing literally
nothing can achieve reward > 0. On a well-isolated benchmark, the null
agent should always score 0.

On the pytest-hook exploit in this tutorial, the null agent scores 1
purely because the conftest.py file was already present before the
"agent" ran.

Usage:
    python null_agent.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPLOIT_DIR = Path(__file__).parent / "conftest_exploit"


def run_pytest(tests_dir: Path, include_conftest: bool) -> tuple[int, str]:
    """Run pytest on the test files. Returns (exit_code, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy only the test files
        for f in ["test_failing.py", "test_passing.py"]:
            (tmp_path / f).write_text((EXPLOIT_DIR / f).read_text())

        # Optionally include the exploit conftest.py
        if include_conftest:
            (tmp_path / "conftest.py").write_text((EXPLOIT_DIR / "conftest.py").read_text())

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr


def main() -> int:
    print("Null agent: doing literally nothing")
    print("  Wrote no files")
    print("  Made no tool calls")
    print("  Executed no code")
    print()

    print("Then running the 'test' against the empty workspace:")

    # Run 1: without the exploit (clean baseline)
    exit_code_clean, _ = run_pytest(EXPLOIT_DIR, include_conftest=False)
    reward_clean = 1 if exit_code_clean == 0 else 0
    expected = " (as expected)" if reward_clean == 0 else " (!!! should be 0)"
    print(f"  WITHOUT the conftest.py exploit: reward = {reward_clean}{expected}")

    # Run 2: with the exploit present in the workspace
    exit_code_exploit, _ = run_pytest(EXPLOIT_DIR, include_conftest=True)
    reward_exploit = 1 if exit_code_exploit == 0 else 0
    expected = " (!!! benchmark is compromised)" if reward_exploit == 1 else " (as expected)"
    print(f"  WITH the conftest.py exploit:    reward = {reward_exploit}{expected}")

    print()
    print("If a null agent can get reward=1 on your benchmark,")
    print("your benchmark has a reward-hacking vulnerability.")

    # Return non-zero if the exploit was effective (compromised benchmark)
    return 1 if reward_exploit == 1 else 0


if __name__ == "__main__":
    sys.exit(main())
