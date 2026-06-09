"""Run Harvey LAB benchmark — downloads raw tasks, converts to BenchFlow format, runs via Evaluation.

Usage:
    python benchmarks/harvey-lab/run_harvey_lab.py                # default config
    python benchmarks/harvey-lab/run_harvey_lab.py --task-format task-md
    python benchmarks/harvey-lab/run_harvey_lab.py path/to/config.yaml

Prefer using `bench eval create --config` with a YAML config that has tasks_dir
pointing to already-converted tasks if you've pre-converted them.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CONVERTER = _SCRIPT_DIR / "benchflow.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harvey LAB via BenchFlow.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_SCRIPT_DIR / "harvey-lab-gemini-flash-lite.yaml"),
        help="BenchFlow evaluation YAML config.",
    )
    parser.add_argument(
        "--task-format",
        choices=("legacy", "task-md"),
        default="legacy",
        help="Converted task layout to run.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def ensure_converted_tasks(task_format: str = "legacy") -> Path:
    """Download raw Harvey LAB tasks and convert to BenchFlow format."""
    from benchflow._utils.benchmark_repos import resolve_source

    raw_dir = resolve_source("harveyai/harvey-labs", path="tasks", ref="main")
    root = _repo_root()
    suffix = "task-md" if task_format == "task-md" else "legacy"
    marker = "task.md" if task_format == "task-md" else "task.toml"
    converted_dir = root / ".cache" / f"harvey-lab-benchflow-{suffix}"

    if converted_dir.exists() and any(converted_dir.glob(f"*/{marker}")):
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
            "--task-format",
            task_format,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Conversion failed: %s", result.stderr)
        raise RuntimeError(f"Harvey LAB conversion failed: {result.stderr}")

    logger.info("Converted tasks to %s", converted_dir)
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
    logger.info("Using tasks from %s", tasks_dir)

    # Load job config from YAML, then override tasks_dir with the converted path.
    # The YAML doesn't specify source/tasks_dir since Harvey LAB requires conversion.
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
