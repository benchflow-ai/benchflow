#!/usr/bin/env python3
"""Runner: clone CLBench, generate BenchFlow tasks, and run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CLBENCH_REPO = "https://github.com/pgasawa/continual-learning-bench"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CLBench BenchFlow tasks.")
    parser.add_argument(
        "--clbench-dir",
        type=Path,
        default=Path("/tmp/continual-learning-bench"),
        help="Path to the CLBench checkout, cloned if it does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/clbench-tasks"),
        help="Where to write generated BenchFlow task directories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of generated CLBench tasks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing task directories.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated CLBench task ids to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    clbench_dir: Path = args.clbench_dir
    if not clbench_dir.exists():
        print(f"Cloning CLBench to {clbench_dir}...")
        subprocess.run(
            ["git", "clone", CLBENCH_REPO, str(clbench_dir)],
            check=True,
        )

    output_dir: Path = args.output_dir
    print(f"Generating BenchFlow tasks in {output_dir}...")
    cmd = [
        sys.executable,
        "benchmarks/clbench/benchflow.py",
        "--clbench-dir",
        str(clbench_dir),
        "--output-dir",
        str(output_dir),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.task_ids:
        cmd.extend(["--task-ids", args.task_ids])
    subprocess.run(cmd, check=True)

    print(f"\nTasks generated in {output_dir}")
    print("Run parity tests with:")
    print(f"  python benchmarks/clbench/parity_test.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()
