"""Job management — run multiple tasks with concurrency, retries, and persistence.

A Job is a collection of trials (task × agent × attempt). Jobs support:
- Concurrent execution with configurable parallelism
- Automatic retries on transient failures (install timeouts, pipe errors)
- Resume from where a previous run left off
- Result persistence (result.json per trial, summary.json per job)
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from benchflow.sdk import SDK, RunResult

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 2
    retry_on_install: bool = True
    retry_on_pipe: bool = True
    retry_on_acp: bool = True

    def should_retry(self, error: str | None) -> bool:
        """Check if an error is retryable."""
        if not error:
            return False
        if self.retry_on_install and "install failed" in error:
            return True
        if self.retry_on_pipe and "closed stdout" in error:
            return True
        if self.retry_on_acp and "ACP error" in error:
            return True
        return False


@dataclass
class JobConfig:
    """Configuration for a benchmark job."""

    agent: str = "claude-agent-acp"
    model: str = "claude-haiku-4-5-20251001"
    environment: str = "docker"
    concurrency: int = 4
    prompts: list[str | None] | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class JobResult:
    """Aggregated results for a job."""

    job_name: str
    config: JobConfig
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    elapsed_sec: float = 0.0
    results: dict[str, RunResult] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        completed = self.passed + self.failed
        return self.passed / completed if completed > 0 else 0.0


class Job:
    """Run a benchmark job across multiple tasks.

    Usage:
        job = Job(
            tasks_dir=".ref/terminal-bench-2",
            jobs_dir="parity/tb2-haiku",
            config=JobConfig(model="claude-haiku-4-5-20251001"),
        )
        result = await job.run()
        print(result.score)
    """

    def __init__(
        self,
        tasks_dir: str | Path,
        jobs_dir: str | Path,
        config: JobConfig | None = None,
        job_name: str | None = None,
        on_result: Callable[[str, RunResult], None] | None = None,
    ):
        self._tasks_dir = Path(tasks_dir)
        self._jobs_dir = Path(jobs_dir)
        self._config = config or JobConfig()
        self._job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self._on_result = on_result
        self._sdk = SDK()

    def _get_task_dirs(self) -> list[Path]:
        """Get all valid task directories."""
        return sorted(
            d for d in self._tasks_dir.iterdir()
            if d.is_dir() and (d / "task.toml").exists()
        )

    def _get_completed_tasks(self) -> dict[str, dict]:
        """Load tasks that already have results with rewards."""
        completed = {}
        for rfile in self._jobs_dir.rglob("result.json"):
            try:
                r = json.loads(rfile.read_text())
                task = r["task_name"]
                if r.get("rewards") is not None:
                    completed[task] = r
            except Exception:
                pass
        return completed

    def _prune_docker(self):
        """Clean up Docker resources."""
        if self._config.environment != "docker":
            return
        try:
            subprocess.run(["docker", "container", "prune", "-f"], capture_output=True, timeout=30)
            subprocess.run(["docker", "network", "prune", "-f"], capture_output=True, timeout=30)
        except Exception:
            pass

    async def _run_task(self, task_dir: Path) -> RunResult:
        """Run a single task with retries."""
        cfg = self._config
        last_result = None

        for attempt in range(1, cfg.retry.max_retries + 2):  # +2 because range is exclusive and attempt 1 is first try
            result = await self._sdk.run(
                task_path=task_dir,
                agent=cfg.agent,
                model=cfg.model,
                prompts=cfg.prompts,
                agent_env=cfg.agent_env,
                jobs_dir=str(self._jobs_dir),
                environment=cfg.environment,
            )
            last_result = result

            # If succeeded or non-retryable error, return
            if result.rewards is not None or not cfg.retry.should_retry(result.error):
                break

            if attempt <= cfg.retry.max_retries:
                logger.info(f"Retrying {task_dir.name} (attempt {attempt + 1}): {result.error[:60]}")

        self._prune_docker()
        return last_result

    async def run(self) -> JobResult:
        """Execute the job."""
        task_dirs = self._get_task_dirs()
        completed = self._get_completed_tasks()
        remaining = [d for d in task_dirs if d.name not in completed]

        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._prune_docker()

        cfg = self._config
        logger.info(
            f"Job: {len(task_dirs)} tasks, {len(completed)} done, "
            f"{len(remaining)} to run (concurrency={cfg.concurrency})"
        )

        start = time.time()
        sem = asyncio.Semaphore(cfg.concurrency)

        async def bounded(td: Path) -> tuple[str, RunResult]:
            async with sem:
                result = await self._run_task(td)
                # Log result
                reward = result.rewards.get("reward") if result.rewards else None
                status = "PASS" if reward == 1 else ("FAIL" if reward is not None else "ERR")
                err = f" ({result.error[:50]})" if result.error else ""
                logger.info(f"[{status}] {td.name} (tools={result.n_tool_calls}){err}")
                if self._on_result:
                    self._on_result(td.name, result)
                return td.name, result

        pairs = await asyncio.gather(*[bounded(td) for td in remaining])
        elapsed = time.time() - start

        # Merge with previously completed
        all_results = {}
        for task, data in completed.items():
            # Convert dict to minimal RunResult-like for counting
            all_results[task] = data
        for name, result in pairs:
            all_results[name] = {
                "task_name": result.task_name,
                "rewards": result.rewards,
                "error": result.error,
                "n_tool_calls": result.n_tool_calls,
            }

        # Count
        job_result = JobResult(
            job_name=self._job_name,
            config=cfg,
            total=len(task_dirs),
            passed=sum(
                1 for r in all_results.values()
                if isinstance(r, dict) and r.get("rewards") and r["rewards"].get("reward") == 1.0
                or isinstance(r, RunResult) and r.rewards and r.rewards.get("reward") == 1.0
            ),
            failed=sum(
                1 for r in all_results.values()
                if isinstance(r, dict) and r.get("rewards") and r["rewards"].get("reward") == 0.0
                or isinstance(r, RunResult) and r.rewards and r.rewards.get("reward") == 0.0
            ),
            errored=sum(
                1 for r in all_results.values()
                if isinstance(r, dict) and r.get("error") and r.get("rewards") is None
                or isinstance(r, RunResult) and r.error and r.rewards is None
            ),
            elapsed_sec=elapsed,
        )

        # Save summary
        summary = {
            "job_name": self._job_name,
            "agent": cfg.agent,
            "model": cfg.model,
            "environment": cfg.environment,
            "total": job_result.total,
            "passed": job_result.passed,
            "failed": job_result.failed,
            "errored": job_result.errored,
            "score": f"{job_result.score:.1%}",
            "score_excl_errors": f"{job_result.score_excl_errors:.1%}",
            "elapsed_sec": elapsed,
        }
        (self._jobs_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        logger.info(
            f"Job complete: {job_result.passed}/{job_result.total} "
            f"({job_result.score:.1%}), errors={job_result.errored}, "
            f"time={elapsed/60:.1f}min"
        )

        return job_result
