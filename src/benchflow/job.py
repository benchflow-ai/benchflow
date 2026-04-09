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
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from benchflow._scoring import (
    ACP_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    classify_error,
    extract_reward,
    pass_rate,
    pass_rate_excl_errors,
)

import yaml

from benchflow._models import RunResult
from benchflow.sdk import SDK

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
        category = classify_error(error)
        if not category:
            return False
        if self.retry_on_install and category == INSTALL_FAILED:
            return True
        if self.retry_on_pipe and category == PIPE_CLOSED:
            return True
        if self.retry_on_acp and category == ACP_ERROR:
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
    skills_dir: str | None = None
    sandbox_user: str | None = None
    context_root: str | None = None

    def __post_init__(self):
        from benchflow.agents.registry import AGENTS
        if self.agent not in AGENTS:
            available = ", ".join(sorted(AGENTS.keys()))
            logger.warning(
                f"Unknown agent {self.agent!r} — not in registry. "
                f"Available: {available}. Will attempt to use as raw command."
            )


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

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)


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

    Or from YAML:
        job = Job.from_yaml("experiments/tb2.yaml")
        result = await job.run()
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

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> "Job":
        """Create a Job from a YAML config file.

        Supports both benchflow-native and Harbor-compatible YAML formats.

        benchflow format:
            tasks_dir: path/to/tasks
            jobs_dir: jobs/my-run
            agent: claude-agent-acp
            model: claude-haiku-4-5-20251001
            environment: daytona
            concurrency: 64
            max_retries: 1
            prompts:
              - null
              - "Review your solution and fix any issues."

        Harbor-compatible format:
            jobs_dir: jobs
            n_attempts: 1
            orchestrator:
              n_concurrent_trials: 4
            environment:
              type: docker
              env:
                - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
            agents:
              - name: claude-agent-acp
                model_name: anthropic/claude-haiku-4-5-20251001
            datasets:
              - path: path/to/tasks
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Detect format: Harbor uses "agents" + "datasets", benchflow uses "agent"
        if "agents" in raw or "datasets" in raw:
            return cls._from_harbor_yaml(raw, path.parent, **kwargs)
        return cls._from_native_yaml(raw, path.parent, **kwargs)

    @classmethod
    def _from_native_yaml(cls, raw: dict, base_dir: Path, **kwargs) -> "Job":
        """Parse benchflow-native YAML."""
        tasks_dir = base_dir / raw["tasks_dir"]
        jobs_dir = base_dir / raw.get("jobs_dir", "jobs")

        # Parse prompts — YAML null becomes Python None
        prompts = raw.get("prompts")

        config = JobConfig(
            agent=raw.get("agent", "claude-agent-acp"),
            model=raw.get("model", "claude-haiku-4-5-20251001"),
            environment=raw.get("environment", "docker"),
            concurrency=raw.get("concurrency", 4),
            prompts=prompts,
            retry=RetryConfig(max_retries=raw.get("max_retries", 2)),
            skills_dir=str(base_dir / raw["skills_dir"]) if raw.get("skills_dir") else None,
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

    @classmethod
    def _from_harbor_yaml(cls, raw: dict, base_dir: Path, **kwargs) -> "Job":
        """Parse Harbor-compatible YAML."""
        # Agent
        agents = raw.get("agents", [{}])
        agent_cfg = agents[0] if agents else {}
        agent_name = agent_cfg.get("name", "claude-agent-acp")

        # Model — Harbor uses "anthropic/model-name" format
        model = agent_cfg.get("model_name", "")
        if "/" in model:
            model = model.split("/", 1)[1]
        model = model or "claude-haiku-4-5-20251001"

        # Environment
        env_cfg = raw.get("environment", {})
        environment = env_cfg.get("type", "docker")

        # Agent env vars from environment.env
        agent_env: dict[str, str] = {}
        for entry in env_cfg.get("env", []):
            if "=" in entry:
                k, v = entry.split("=", 1)
                # Expand ${VAR} references
                v = os.path.expandvars(v)
                agent_env[k] = v

        # Datasets
        datasets = raw.get("datasets", [{}])
        tasks_dir = base_dir / datasets[0].get("path", "tasks")

        # Orchestrator
        orch = raw.get("orchestrator", {})
        concurrency = orch.get("n_concurrent_trials", 4)

        jobs_dir = base_dir / raw.get("jobs_dir", "jobs")
        max_retries = raw.get("n_attempts", 1) - 1  # Harbor n_attempts includes first try

        # Skills dir (shared with benchflow-native format)
        skills_dir_raw = raw.get("skills_dir")
        skills_dir = str(base_dir / skills_dir_raw) if skills_dir_raw else None

        config = JobConfig(
            agent=agent_name,
            model=model,
            environment=environment,
            concurrency=concurrency,
            agent_env=agent_env,
            retry=RetryConfig(max_retries=max(0, max_retries)),
            skills_dir=skills_dir,
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

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
            except Exception as e:
                logger.debug(f"Skipping corrupt result file {rfile}: {e}")
        return completed

    def _prune_docker(self):
        """Clean up Docker resources."""
        if self._config.environment != "docker":
            return
        try:
            subprocess.run(["docker", "container", "prune", "-f"], capture_output=True, timeout=30)
            subprocess.run(["docker", "network", "prune", "-f"], capture_output=True, timeout=30)
        except Exception as e:
            logger.warning(f"Docker prune failed: {e}")

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
                job_name=self._job_name,
                jobs_dir=str(self._jobs_dir),
                environment=cfg.environment,
                skills_dir=cfg.skills_dir,
                sandbox_user=cfg.sandbox_user,
                context_root=cfg.context_root,
            )
            last_result = result

            # If succeeded or non-retryable error, return
            if result.rewards is not None or not cfg.retry.should_retry(result.error):
                break

            if attempt <= cfg.retry.max_retries:
                logger.info(f"Retrying {task_dir.name} (attempt {attempt + 1}): {result.error[:60]}")

        return last_result

    async def run(self) -> JobResult:
        """Execute the job."""
        task_dirs = self._get_task_dirs()
        completed = self._get_completed_tasks()
        remaining = [d for d in task_dirs if d.name not in completed]

        # Warn if resuming with different config than completed tasks
        if completed:
            # Check config.json (written by SDK.run) for the registry agent name
            sample_dir = next(
                (d for d in self._jobs_dir.iterdir() if d.is_dir()),
                None,
            )
            prev_agent = ""
            if sample_dir:
                for cfg_file in sample_dir.rglob("config.json"):
                    try:
                        cfg = json.loads(cfg_file.read_text())
                        prev_agent = cfg.get("agent", "")
                        break
                    except Exception:
                        pass
            if prev_agent and prev_agent != self._config.agent:
                logger.warning(
                    f"Resuming with agent={self._config.agent!r} but "
                    f"completed tasks used agent={prev_agent!r}. "
                    f"Use a different jobs_dir to avoid mixing results."
                )

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

        results_or_errors = await asyncio.gather(
            *[bounded(td) for td in remaining],
            return_exceptions=True,
        )
        elapsed = time.time() - start

        # Separate successful results from unexpected exceptions
        pairs: list[tuple[str, RunResult]] = []
        for i, r in enumerate(results_or_errors):
            if isinstance(r, BaseException):
                task_name = remaining[i].name
                logger.error(f"[ERR] {task_name}: unexpected exception: {r}")
                pairs.append((task_name, RunResult(
                    task_name=task_name, error=f"Unexpected: {r}",
                )))
            else:
                pairs.append(r)

        # Merge with previously completed — normalize everything to dicts
        all_results: dict[str, dict] = {}
        for task, data in completed.items():
            all_results[task] = data
        for name, result in pairs:
            all_results[name] = {
                "task_name": result.task_name,
                "rewards": result.rewards,
                "error": result.error,
                "n_tool_calls": result.n_tool_calls,
            }

        # Count — all values are dicts now, no type branching needed
        job_result = JobResult(
            job_name=self._job_name,
            config=cfg,
            total=len(task_dirs),
            passed=sum(1 for r in all_results.values() if extract_reward(r) == 1.0),
            failed=sum(1 for r in all_results.values()
                       if extract_reward(r) is not None and extract_reward(r) != 1.0),
            errored=sum(1 for r in all_results.values()
                        if r.get("error") and r.get("rewards") is None),
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
