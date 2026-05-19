"""Run ProgramBench — generates tasks if needed, runs via Evaluation."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProgramBench via BenchFlow.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_SCRIPT_DIR / "programbench-gemini-flash-lite.yaml"),
        help="BenchFlow evaluation YAML config.",
    )
    return parser.parse_args()


def ensure_converted_tasks() -> Path:
    """Download official ProgramBench metadata and convert it to BenchFlow tasks."""
    from benchflow._utils.benchmark_repos import ensure_tasks
    from benchmarks.programbench.benchflow import generate_all

    raw_tasks_dir = ensure_tasks("programbench")
    converted_dir = _REPO_ROOT / ".cache" / "programbench-benchflow"
    if converted_dir.exists() and any(converted_dir.glob("*/task.toml")):
        logger.info("Converted ProgramBench tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting ProgramBench tasks from %s", raw_tasks_dir)
    generated = generate_all(raw_tasks_dir, converted_dir)
    logger.info("Generated %d ProgramBench tasks in %s", len(generated), converted_dir)
    return converted_dir


async def main():
    from benchflow.evaluation import Evaluation

    args = _parse_args()
    tasks_dir = ensure_converted_tasks()
    job = Evaluation.from_yaml(args.config)
    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
