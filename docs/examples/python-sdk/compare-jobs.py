#!/usr/bin/env python3
"""Compare two finished jobs task by task, with fair denominators.

Each side is a job folder (or several folders, e.g. one arm of a paired run
kept in per-task folders). Trials are paired by task name; control runs
(oracle, empty) are left out unless --include-controls. Prints a markdown
report and optionally writes the per-task rows to CSV. Reads files only: no
sandbox, no credentials.

Usage:
  uv run python docs/examples/python-sdk/compare-jobs.py jobs/run-a jobs/run-b
  uv run python docs/examples/python-sdk/compare-jobs.py \\
      --a 'runs/*/a' --b 'runs/*/b' --labels a b
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import benchflow as bf


def _paths(spec: str) -> list[Path]:
    matches = sorted(glob.glob(spec))
    return [Path(m) for m in matches] or [Path(spec)]


def main(args: argparse.Namespace) -> int:
    a_spec = args.a or args.job_a
    b_spec = args.b or args.job_b
    if not a_spec or not b_spec:
        raise SystemExit("give two jobs: JOB_A JOB_B, or --a GLOB --b GLOB")
    comparison = bf.compare(
        bf.load_job(_paths(a_spec)),
        bf.load_job(_paths(b_spec)),
        include_controls=args.include_controls,
        labels=tuple(args.labels) if args.labels else None,
    )
    print(comparison.to_markdown())
    if args.csv:
        rows = comparison.to_records()
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", args.csv)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_a", nargs="?")
    parser.add_argument("job_b", nargs="?")
    parser.add_argument("--a", help="glob for side A's folders")
    parser.add_argument("--b", help="glob for side B's folders")
    parser.add_argument("--labels", nargs=2, metavar=("A", "B"))
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--csv", help="write the per-task rows here")
    raise SystemExit(main(parser.parse_args()))
