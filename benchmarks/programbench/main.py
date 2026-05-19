"""CLI entry point for ProgramBench → BenchFlow task generation.

Usage::

    python -m benchmarks.programbench.main --output-dir benchmarks/programbench/tasks
    python -m benchmarks.programbench.main --output-dir out --limit 5
    python -m benchmarks.programbench.main --output-dir out --task-ids jqlang__jq.b33a763
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmarks.programbench.benchflow import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BenchFlow tasks from ProgramBench instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated task directories into.",
    )
    parser.add_argument(
        "--programbench-dir",
        type=Path,
        default=None,
        help="Path to ProgramBench repo or data/tasks directory.  "
        "If omitted, tries the installed programbench package.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Generate at most N tasks."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing task directories."
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Only generate these instance IDs.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tasks_dir = _resolve_tasks_dir(args.programbench_dir)
    generated = generate_all(
        tasks_dir,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=args.task_ids,
    )
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


def _resolve_tasks_dir(explicit: Path | None) -> Path:
    """Find the ProgramBench tasks directory."""
    if explicit is not None:
        # Accept either the repo root or the data/tasks dir directly
        candidate = explicit / "src" / "programbench" / "data" / "tasks"
        if candidate.is_dir():
            return candidate
        if (explicit / "task.yaml").exists() or any(explicit.glob("*/task.yaml")):
            return explicit
        raise FileNotFoundError(f"No ProgramBench tasks found at {explicit}")

    # Try the installed package
    try:
        from programbench.constants import TASKS_DIR

        if TASKS_DIR.is_dir():
            return TASKS_DIR
    except ImportError:
        pass

    # Try common local paths
    for path in [
        Path.cwd() / "programbench",
        Path.home() / "programbench",
    ]:
        candidate = path / "src" / "programbench" / "data" / "tasks"
        if candidate.is_dir():
            return candidate

    print(
        "ERROR: Cannot find ProgramBench tasks. Either:\n"
        "  1. Install programbench: pip install programbench\n"
        "  2. Pass --programbench-dir /path/to/programbench",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
