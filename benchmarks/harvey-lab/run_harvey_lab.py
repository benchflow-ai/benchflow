"""Run Harvey LAB benchmark — downloads raw tasks, converts to BenchFlow format, runs via Evaluation.

Usage:
    python benchmarks/harvey-lab/run_harvey_lab.py                # default config
    python benchmarks/harvey-lab/run_harvey_lab.py path/to/config.yaml

Prefer using `bench eval create --config` with a YAML config that has tasks_dir
pointing to already-converted tasks if you've pre-converted them.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from benchflow._utils.benchmark_repos import resolve_source
from benchflow.evaluation import Evaluation

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
    """Download raw Harvey LAB tasks and convert to BenchFlow format."""
    raw_dir = resolve_source("harveyai/harvey-labs", path="tasks", ref="main")
    root = _repo_root()
    converted_dir = root / ".cache" / "harvey-lab-benchflow"

    if converted_dir.exists() and any(converted_dir.iterdir()):
        logger.info("Converted tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting Harvey LAB tasks to BenchFlow format...")
    result = subprocess.run(
        [
            sys.executable,
            str(_CONVERTER),
            "--output-dir",
            str(converted_dir),
            "--harvey-root",
            str(raw_dir.parent),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Conversion failed: %s", result.stderr)
        raise RuntimeError(f"Harvey LAB conversion failed: {result.stderr}")

    logger.info("Converted tasks to %s", converted_dir)
    return converted_dir


async def main():
    config = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(_SCRIPT_DIR / "harvey-lab-gemini-flash-lite.yaml")
    )
    tasks_dir = ensure_converted_tasks()
    logger.info("Using tasks from %s", tasks_dir)

    # Load job config from YAML, then override tasks_dir with the converted path.
    # The YAML doesn't specify source/tasks_dir since Harvey LAB requires conversion.
    job = Evaluation.from_yaml(config)
    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]

    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
