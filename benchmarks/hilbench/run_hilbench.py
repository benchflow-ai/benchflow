"""Run HILBench — downloads dataset from HuggingFace, generates tasks, runs via Job.

Usage:
    python benchmarks/hilbench/run_hilbench.py
    python benchmarks/hilbench/run_hilbench.py path/to/config.yaml
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from benchflow.job import Job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONVERTER = _SCRIPT_DIR / "benchflow.py"


def _repo_root() -> Path:
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def ensure_converted_tasks() -> Path:
    """Download HILBench dataset from HuggingFace and convert to BenchFlow format."""
    root = _repo_root()
    converted_dir = root / ".cache" / "hilbench-benchflow"

    if converted_dir.exists() and any(converted_dir.iterdir()):
        logger.info("Converted tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting HILBench SWE tasks to BenchFlow format...")
    result = subprocess.run(
        [
            sys.executable,
            str(_CONVERTER),
            "--output-dir",
            str(converted_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Conversion failed: %s", result.stderr)
        raise RuntimeError(f"HILBench conversion failed: {result.stderr}")

    logger.info("Converted tasks to %s", converted_dir)
    return converted_dir


async def main():
    config = sys.argv[1] if len(sys.argv) > 1 else None
    tasks_dir = ensure_converted_tasks()
    logger.info("Using tasks from %s", tasks_dir)

    if config:
        job = Job.from_yaml(config)
    else:
        logger.info("No config specified; tasks generated at %s", tasks_dir)
        logger.info("Use: bench eval create -f <config.yaml> to run evaluations")
        return

    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
