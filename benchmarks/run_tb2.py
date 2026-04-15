"""Run Terminal-Bench 2.0 — downloads tasks if needed, runs via Job.

Usage:
    python benchmarks/run_tb2.py  # defaults to tb2_single-codex-gpt54.yaml
    python benchmarks/run_tb2.py benchmarks/tb2_multiturn-codex-gpt54.yaml
"""

import asyncio
import logging
import sys
from pathlib import Path

from benchflow.job import Job
from benchflow.task_download import ensure_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main():
    config = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "tb2_single-codex-gpt54.yaml"
    )
    ensure_tasks("terminal-bench-2")
    job = Job.from_yaml(config)
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
