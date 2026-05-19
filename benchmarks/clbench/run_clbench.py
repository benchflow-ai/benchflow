#!/usr/bin/env python3
"""Runner: clone CLBench, generate BenchFlow tasks, and run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLBENCH_REPO = "https://github.com/pgasawa/continual-learning-bench"


def main() -> None:
    clbench_dir = Path("/tmp/continual-learning-bench")
    if not clbench_dir.exists():
        print(f"Cloning CLBench to {clbench_dir}...")
        subprocess.run(
            ["git", "clone", CLBENCH_REPO, str(clbench_dir)],
            check=True,
        )

    output_dir = Path("/tmp/clbench-tasks")
    print(f"Generating BenchFlow tasks in {output_dir}...")
    subprocess.run(
        [
            sys.executable,
            "benchmarks/clbench/benchflow.py",
            "--clbench-dir",
            str(clbench_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    print(f"\nTasks generated in {output_dir}")
    print("Run parity tests with:")
    print(f"  python benchmarks/clbench/parity_test.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()
