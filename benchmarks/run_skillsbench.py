"""Run SkillsBench — downloads tasks if needed, runs via Job."""

import asyncio
import logging
import sys
from pathlib import Path

from benchflow.job import Job
from benchflow.task_download import ensure_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main():
    config = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "skillsbench-codex-gpt54.yaml"
    )
    ensure_tasks("skillsbench")
    job = Job.from_yaml(config)
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
