"""Run ProgramBench — generates tasks if needed, runs via Evaluation."""

import asyncio
import logging
import sys
from pathlib import Path

from benchflow._utils.benchmark_repos import ensure_tasks
from benchflow.evaluation import Evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main():
    config = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).parent / "programbench-gemini-flash-lite.yaml")
    )
    ensure_tasks("programbench")
    job = Evaluation.from_yaml(config)
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
