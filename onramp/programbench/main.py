"""CLI entry point for the ProgramBench onramp adapter.

Usage:

    python -m onramp.programbench.main \\
        --output-dir tasks/programbench \\
        --upstream-tasks-dir <path-to-programbench-clone>/src/programbench/data/tasks \\
        --limit 5

If ``--upstream-tasks-dir`` is omitted, the adapter clones
``facebookresearch/ProgramBench`` into ``.ref/programbench/`` first.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from onramp.programbench.adapter import convert

UPSTREAM_REPO = "https://github.com/facebookresearch/ProgramBench.git"
UPSTREAM_REF = "main"


def _repo_root() -> Path:
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def ensure_upstream_clone() -> Path:
    """Clone ProgramBench into .ref/programbench/ if not present and return tasks/ dir."""
    root = _repo_root()
    target = root / ".ref" / "programbench"
    tasks = target / "src" / "programbench" / "data" / "tasks"
    if tasks.exists() and any(tasks.iterdir()):
        return tasks
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", UPSTREAM_REF, UPSTREAM_REPO, str(target)],
        check=True,
    )
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ProgramBench into BenchFlow tasks.")
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Where to write the generated task directories.",
    )
    parser.add_argument(
        "--upstream-tasks-dir", type=Path, default=None,
        help="Path to ProgramBench's src/programbench/data/tasks/. "
             "If omitted, the upstream repo is cloned into .ref/programbench/.",
    )
    parser.add_argument(
        "--task-ids", nargs="*", default=None,
        help="Convert only these instance IDs (space-separated).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N tasks.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing tasks.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tasks_dir = args.upstream_tasks_dir or ensure_upstream_clone()
    if not tasks_dir.exists():
        print(f"upstream tasks dir not found: {tasks_dir}", file=sys.stderr)
        return 2

    generated = convert(
        upstream_tasks_dir=tasks_dir,
        output_dir=args.output_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Generated {len(generated)} BenchFlow tasks in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
