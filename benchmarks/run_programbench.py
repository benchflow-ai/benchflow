"""Run ProgramBench — generates BenchFlow tasks if needed, runs via Job.

ProgramBench tasks aren't a separate repo of pre-built BenchFlow tasks
(unlike SkillsBench / TB2). They're produced from upstream ProgramBench
metadata by `benchmarks.programbench.benchflow.convert()` — see
`benchmarks/programbench/README.md`.

Usage:
    python benchmarks/run_programbench.py
    python benchmarks/run_programbench.py path/to/custom.yaml
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from benchflow.job import Job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def ensure_programbench_tasks() -> Path:
    """Generate `.ref/programbench-bf/<instance_id>/` for every ProgramBench instance.

    Re-running is cheap (skips existing dirs unless `--overwrite` is passed
    to the `main` CLI directly). The first call clones
    `facebookresearch/ProgramBench` into `.ref/programbench/` for metadata.
    """
    root = _repo_root()
    target = root / ".ref" / "programbench-bf"
    if target.exists() and any(target.iterdir()):
        return target
    logger.info("Generating ProgramBench BenchFlow tasks at %s ...", target)
    subprocess.run(
        [sys.executable, "-m", "benchmarks.programbench.main",
         "--output-dir", str(target)],
        cwd=str(root),
        check=True,
    )
    return target


async def main():
    config = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "run_programbench.yaml"
    )
    ensure_programbench_tasks()
    job = Job.from_yaml(config)
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
