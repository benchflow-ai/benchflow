#!/usr/bin/env python3
"""Materialize a computer-use smoke task through BenchFlow's inbound adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchflow.adapters.computer_use import ComputerUseAdapter
from benchflow.adapters.inbound import materialize_inbound_task_md


def materialize_task(task_dir: Path, out_dir: Path, *, overwrite: bool = False) -> Path:
    inbound = ComputerUseAdapter.from_task_dir(task_dir)
    return materialize_inbound_task_md(
        inbound,
        out_dir / inbound.name,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    native_task = materialize_task(
        args.task_dir,
        args.out_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({"native_task_dir": str(native_task)}, indent=2))


if __name__ == "__main__":
    main()
