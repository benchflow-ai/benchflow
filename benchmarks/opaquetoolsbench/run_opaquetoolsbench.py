"""Run OpaqueToolsBench BFCL — converts source tasks and runs via Evaluation.

Usage:
    python benchmarks/opaquetoolsbench/run_opaquetoolsbench.py
    python benchmarks/opaquetoolsbench/run_opaquetoolsbench.py path/to/config.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import cast

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
    parser = argparse.ArgumentParser(description="Run OpaqueToolsBench via BenchFlow.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_SCRIPT_DIR / "opaquetoolsbench-gemini-flash-lite.yaml"),
        help="BenchFlow evaluation YAML config.",
    )
    parser.add_argument(
        "--task-format",
        choices=("legacy", "task-md"),
        default="legacy",
        help="Converted task layout to run.",
    )
    return parser.parse_args()


def ensure_converted_tasks(task_format: str = "legacy") -> Path:
    """Clone OpaqueToolsBench and convert BFCL configs to BenchFlow tasks."""
    from benchflow._utils.benchmark_repos import resolve_source
    from benchmarks.opaquetoolsbench.benchflow import TaskFormat, generate_all

    raw_repo = resolve_source("shallinan1/OpaqueToolsBench", ref="main")
    suffix = "task-md" if task_format == "task-md" else "legacy"
    marker = "task.md" if task_format == "task-md" else "task.toml"
    converted_dir = _REPO_ROOT / ".cache" / f"opaquetoolsbench-benchflow-{suffix}"
    if converted_dir.exists() and any(converted_dir.glob(f"*/{marker}")):
        logger.info("Converted tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting OpaqueToolsBench BFCL tasks from %s", raw_repo)
    generated = generate_all(
        raw_repo,
        converted_dir,
        task_format=cast(TaskFormat, task_format),
    )
    logger.info(
        "Generated %d OpaqueToolsBench tasks in %s", len(generated), converted_dir
    )
    return converted_dir


def _jobs_dir_for_task_format(jobs_dir: Path, task_format: str) -> Path:
    """Keep native task.md runs from resuming legacy conversion results."""
    if task_format == "legacy":
        return jobs_dir
    suffix = f"-{task_format}"
    if jobs_dir.name.endswith(suffix):
        return jobs_dir
    return jobs_dir.with_name(f"{jobs_dir.name}{suffix}")


async def main():
    from benchflow.evaluation import Evaluation

    args = _parse_args()
    tasks_dir = ensure_converted_tasks(task_format=args.task_format)
    job = Evaluation.from_yaml(args.config)
    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]
    job._jobs_dir = _jobs_dir_for_task_format(  # type: ignore[attr-defined]
        job._jobs_dir,
        args.task_format,
    )
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
