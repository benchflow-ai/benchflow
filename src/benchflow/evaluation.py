"""Evaluation management — run many tasks against an agent with concurrency, retries, resume.

An ``Evaluation`` wraps ``bf.run()`` with everything needed to drive a benchmark
to completion: task discovery, parallelism, retry policy, resume from
disk, summary aggregation.

Backward-compat aliases: ``Job = Evaluation``, ``JobConfig = EvaluationConfig``,
``JobResult = EvaluationResult``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from benchflow._utils.scoring import (
    ACP_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    classify_error,
    extract_reward,
    pass_rate,
    pass_rate_excl_errors,
)
from benchflow.models import RolloutResult

# Backward-compat alias
RunResult = RolloutResult

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()  # default value for _sdk; tests replace with AsyncMock


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Matches Harbor's RetryConfig pattern: exponential backoff with
    configurable exception filtering. Legacy boolean fields are
    preserved for backwards compat but the category-based check
    covers all cases.
    """

    max_retries: int = 2
    retry_on_install: bool = True
    retry_on_pipe: bool = True
    retry_on_acp: bool = True
    wait_multiplier: float = 2.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    exclude_categories: set[str] = field(default_factory=lambda: {"timeout"})

    def should_retry(self, error: str | None) -> bool:
        """Check if an error is retryable."""
        category = classify_error(error)
        if not category:
            return False
        if category in self.exclude_categories:
            return False
        if self.retry_on_install and category == INSTALL_FAILED:
            return True
        if self.retry_on_pipe and category == PIPE_CLOSED:
            return True
        return bool(self.retry_on_acp and category == ACP_ERROR)

    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff delay for retry attempt."""
        delay = self.min_wait_sec * (self.wait_multiplier**attempt)
        return min(delay, self.max_wait_sec)


# Defaults: works out-of-the-box with `claude login` (subscription auth, no API key needed)
DEFAULT_AGENT = "claude-agent-acp"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Job scheduling modes (architecture.md § "Lifecycles" — the Job lifecycle).
# - parallel-independent: the default — rollouts run concurrently, isolated.
# - sequential-shared: continual learning — rollouts run in order over one
#   persistent, versioned LearnerStore (capability 5).
JOB_MODES = ("parallel-independent", "sequential-shared")
DEFAULT_JOB_MODE = "parallel-independent"


def effective_model(agent: str, model: str | None) -> str | None:
    """Resolve the model an agent should run with.

    Oracle runs solve.sh and never calls an LLM, so it never receives a model
    (the chokepoint in resolve_agent_env defends, but callers should also stop
    materializing DEFAULT_MODEL into oracle configs to keep the data honest —
    e.g. result-summary JSON shows model=null instead of a bogus default).
    """
    if agent == "oracle":
        return None
    return model or DEFAULT_MODEL


@dataclass
class EvaluationConfig:
    """Configuration for a benchmark job."""

    agent: str = DEFAULT_AGENT
    model: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    prompts: list[str | None] | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    skills_dir: str | None = None
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    context_root: str | None = None
    exclude_tasks: set[str] = field(default_factory=set)
    include_tasks: set[str] = field(default_factory=set)
    skill_mode: str = "default"
    skill_creator_dir: str | None = None
    self_gen_no_internet: bool = False
    job_mode: str = DEFAULT_JOB_MODE

    def __post_init__(self):
        from benchflow._utils.config import normalize_agent_name, normalize_sandbox_user
        from benchflow.agents.registry import AGENTS

        self.agent = normalize_agent_name(self.agent)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        if self.job_mode not in JOB_MODES:
            raise ValueError(
                f"unknown job_mode {self.job_mode!r} — "
                f"expected one of {', '.join(JOB_MODES)}"
            )
        if self.agent not in AGENTS:
            available = ", ".join(sorted(AGENTS.keys()))
            logger.warning(
                f"Unknown agent {self.agent!r} — not in registry. "
                f"Available: {available}. Will attempt to use as raw command."
            )


@dataclass
class EvaluationResult:
    """Aggregated results for a job."""

    job_name: str
    config: EvaluationConfig
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    verifier_errored: int = 0
    elapsed_sec: float = 0.0

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)


def _agent_result_from_rollout(result: RunResult) -> dict[str, Any]:
    """Return the serialized agent_result block for an in-memory rollout result."""
    return {
        "n_tool_calls": result.n_tool_calls,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "price_source": result.price_source,
    }


def _usage_summary(results: dict[str, dict]) -> dict[str, Any]:
    """Aggregate provider telemetry fields for summary.json."""
    completed = [
        r
        for r in results.values()
        if r.get("rewards") is not None
        and not r.get("error")
        and not r.get("verifier_error")
    ]
    covered = [
        r
        for r in completed
        if (r.get("agent_result") or {}).get("usage_source") == "provider_response"
    ]

    def total(field: str) -> int:
        return sum((r.get("agent_result") or {}).get(field) or 0 for r in covered)

    total_cost = round(
        sum((r.get("agent_result") or {}).get("cost_usd") or 0.0 for r in covered),
        10,
    )
    return {
        "total_input_tokens": total("n_input_tokens"),
        "total_output_tokens": total("n_output_tokens"),
        "total_cache_read_tokens": total("n_cache_read_tokens"),
        "total_cache_creation_tokens": total("n_cache_creation_tokens"),
        "total_tokens": total("total_tokens"),
        "total_cost_usd": total_cost,
        "avg_cost_per_trial_usd": (
            round(total_cost / len(covered), 10) if covered else None
        ),
        "telemetry_coverage": (len(covered) / len(completed) if completed else 0.0),
    }


class Evaluation:
    """Run a benchmark job across multiple tasks.

    Usage:
        from benchflow._utils.benchmark_repos import resolve_source

        evaluation = Evaluation(
            tasks_dir=resolve_source("harbor-framework/terminal-bench-2"),
            jobs_dir="parity/tb2-haiku",
            config=EvaluationConfig(model="claude-haiku-4-5-20251001"),
        )
        result = await evaluation.run()
        print(result.score)

    Or from YAML:
        evaluation = Evaluation.from_yaml("experiments/tb2.yaml")
        result = await evaluation.run()
    """

    def __init__(
        self,
        tasks_dir: str | Path,
        jobs_dir: str | Path,
        config: EvaluationConfig | None = None,
        job_name: str | None = None,
        on_result: Callable[[str, RunResult], None] | None = None,
    ):
        self._tasks_dir = Path(tasks_dir)
        self._jobs_dir = Path(jobs_dir)
        self._config = config or EvaluationConfig()
        self._job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self._on_result = on_result
        # Kept for test mocking compat; _run_task prefers Rollout
        from benchflow.sdk import SDK

        self._sdk = SDK()
        # The persistent learner store for sequential-shared (continual
        # learning) jobs. Created on demand by run(); parallel-independent
        # jobs leave it None.
        from benchflow.learner_store import LearnerStore

        self.learner_store: LearnerStore | None = (
            LearnerStore()
            if self._config.job_mode == "sequential-shared"
            else None
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> Evaluation:
        """Create a Job from a YAML config file.

        Supports both benchflow-native and legacy YAML formats.

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

        Legacy format (agents + datasets style):
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

        # Detect format: legacy uses "agents" + "datasets", benchflow uses "agent"
        if "agents" in raw or "datasets" in raw:
            return cls._from_legacy_yaml(raw, **kwargs)
        return cls._from_native_yaml(raw, **kwargs)

    @classmethod
    def _from_native_yaml(cls, raw: dict, **kwargs) -> Evaluation:
        """Parse benchflow-native YAML."""
        from benchflow._utils.benchmark_repos import (
            TASK_ALIASES,
            ensure_tasks,
            resolve_source,
        )

        # New two-field format: source.repo + source.path
        if "source" in raw:
            src = raw["source"]
            tasks_dir = resolve_source(
                repo=src["repo"],
                path=src.get("path"),
                ref=src.get("ref"),
            )
        elif "tasks_dir" in raw:
            # Legacy single-string format (backward compat).
            ref = raw["tasks_dir"]
            tasks_dir = Path(ref)
            if not tasks_dir.exists() and ref in TASK_ALIASES:
                tasks_dir = ensure_tasks(ref)
        else:
            raise ValueError("YAML config must have 'source' or 'tasks_dir'")

        jobs_dir = Path(raw.get("jobs_dir", "jobs"))

        # Parse prompts — YAML null becomes Python None
        prompts = raw.get("prompts")

        agent_env_raw = raw.get("agent_env", {})
        exclude = set(raw.get("exclude", []))
        include = set(raw.get("include", []))
        sandbox_user = raw.get("sandbox_user", "agent")
        sandbox_locked_paths = raw.get("sandbox_locked_paths")
        sandbox_setup_timeout = raw.get("sandbox_setup_timeout", 120)

        agent_name = raw.get("agent", DEFAULT_AGENT)
        config = EvaluationConfig(
            agent=agent_name,
            model=effective_model(agent_name, raw.get("model")),
            environment=raw.get("environment", "docker"),
            concurrency=raw.get("concurrency", 4),
            prompts=prompts,
            agent_env=agent_env_raw,
            retry=RetryConfig(max_retries=raw.get("max_retries", 2)),
            skills_dir=str(Path(raw["skills_dir"])) if raw.get("skills_dir") else None,
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            exclude_tasks=exclude,
            include_tasks=include,
            skill_mode=raw.get("skill_mode", "default"),
            skill_creator_dir=(
                str(Path(raw["skill_creator_dir"]))
                if raw.get("skill_creator_dir")
                else None
            ),
            self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
            job_mode=raw.get("job_mode", DEFAULT_JOB_MODE),
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

    @classmethod
    def _from_legacy_yaml(cls, raw: dict, **kwargs) -> Evaluation:
        """Parse legacy-format YAML (agents + datasets style)."""
        # Agent
        agents = raw.get("agents", [{}])
        agent_cfg = agents[0] if agents else {}
        agent_name = agent_cfg.get("name", DEFAULT_AGENT)

        # Model — keep provider prefix intact for downstream resolution
        model = effective_model(agent_name, agent_cfg.get("model_name") or None)

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
        tasks_dir = Path(datasets[0].get("path", "tasks"))

        # Orchestrator
        orch = raw.get("orchestrator", {})
        concurrency = orch.get("n_concurrent_trials", 4)

        jobs_dir = Path(raw.get("jobs_dir", "jobs"))
        max_retries = (
            raw.get("n_attempts", 1) - 1
        )  # legacy n_attempts includes first try

        # Skills dir (shared with benchflow-native format)
        skills_dir_raw = raw.get("skills_dir")
        skills_dir = str(Path(skills_dir_raw)) if skills_dir_raw else None
        sandbox_user = raw.get("sandbox_user", "agent")
        sandbox_locked_paths = raw.get("sandbox_locked_paths")
        sandbox_setup_timeout = raw.get("sandbox_setup_timeout", 120)

        config = EvaluationConfig(
            agent=agent_name,
            model=model,
            environment=environment,
            concurrency=concurrency,
            agent_env=agent_env,
            retry=RetryConfig(max_retries=max(0, max_retries)),
            skills_dir=skills_dir,
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            skill_mode=raw.get("skill_mode", "default"),
            skill_creator_dir=(
                str(Path(raw["skill_creator_dir"]))
                if raw.get("skill_creator_dir")
                else None
            ),
            self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

    def _get_task_dirs(self) -> list[Path]:
        """Get all valid task directories."""
        return sorted(
            d
            for d in self._tasks_dir.iterdir()
            if d.is_dir()
            and (d / "task.toml").exists()
            and d.name not in self._config.exclude_tasks
            and (not self._config.include_tasks or d.name in self._config.include_tasks)
        )

    def _get_completed_tasks(self) -> dict[str, dict]:
        """Load tasks that already have results with rewards or verifier errors."""
        completed = {}
        for rfile in self._jobs_dir.rglob("result.json"):
            try:
                r = json.loads(rfile.read_text())
                task = r["task_name"]
                if r.get("rewards") is not None or r.get("verifier_error"):
                    if r.get("verifier_error"):
                        logger.info(
                            f"Skipping verifier-errored task on resume: {task} ({r['verifier_error'][:80]})"
                        )
                    completed[task] = r
            except Exception as e:
                logger.debug(f"Skipping corrupt result file {rfile}: {e}")
        return completed

    def _prune_docker(self):
        """Clean up Docker resources."""
        if self._config.environment != "docker":
            return
        try:
            subprocess.run(
                ["docker", "container", "prune", "-f"], capture_output=True, timeout=30
            )
            subprocess.run(
                ["docker", "network", "prune", "-f"], capture_output=True, timeout=30
            )
        except Exception as e:
            logger.warning(f"Docker prune failed: {e}")

    def _resolve_skills_dir(self, task_dir: Path, skills_dir: str | None) -> str | None:
        """Resolve skills_dir — 'auto' means per-task environment/skills/."""
        if skills_dir == "auto":
            candidate = task_dir / "environment" / "skills"
            return str(candidate) if candidate.is_dir() else None
        return skills_dir

    async def _run_single_task(
        self, task_dir: Path, cfg: EvaluationConfig
    ) -> RolloutResult:
        """Execute one rollout via Rollout."""
        from benchflow.rollout import Rollout, RolloutConfig

        rollout_config = RolloutConfig.from_legacy(
            task_path=task_dir,
            agent=cfg.agent,
            model=cfg.model,
            prompts=cfg.prompts,
            agent_env=cfg.agent_env,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            environment=cfg.environment,
            skills_dir=self._resolve_skills_dir(task_dir, cfg.skills_dir),
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            context_root=cfg.context_root,
            skill_mode=cfg.skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
        )
        if cfg.skill_mode == "self-gen":
            from benchflow.self_gen import run_self_gen

            return await run_self_gen(rollout_config)
        rollout = await Rollout.create(rollout_config)
        return await rollout.run()

    async def _run_single_task_legacy(
        self, task_dir: Path, cfg: EvaluationConfig
    ) -> RunResult:
        """SDK.run() path — used when _sdk is mocked in tests."""
        return await self._sdk.run(
            task_path=task_dir,
            agent=cfg.agent,
            model=cfg.model,
            prompts=cfg.prompts,
            agent_env=cfg.agent_env,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            environment=cfg.environment,
            skills_dir=self._resolve_skills_dir(task_dir, cfg.skills_dir),
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            context_root=cfg.context_root,
            skill_mode=cfg.skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
        )

    async def _run_task(self, task_dir: Path) -> RunResult:
        """Run a single task with retries."""
        cfg = self._config
        last_result: RunResult | None = None

        for attempt in range(1, cfg.retry.max_retries + 2):
            if attempt > 1:
                delay = cfg.retry.backoff_delay(attempt - 1)
                logger.info(f"Retry backoff: {delay:.1f}s before attempt {attempt}")
                await asyncio.sleep(delay)
                self._prune_docker()
            # Use legacy SDK path if _sdk has been replaced (test compat)
            from benchflow.sdk import SDK

            if not isinstance(self._sdk, SDK):
                result = await self._run_single_task_legacy(task_dir, cfg)
            else:
                result = await self._run_single_task(task_dir, cfg)
            last_result = result

            # If succeeded, verifier-errored (terminal), or non-retryable, stop
            if (
                result.rewards is not None
                or result.verifier_error
                or not cfg.retry.should_retry(result.error)
            ):
                break

            if attempt <= cfg.retry.max_retries:
                err_preview = (result.error or "")[:60]
                logger.info(
                    f"Retrying {task_dir.name} (attempt {attempt + 1}): {err_preview}"
                )

        # The loop always runs at least once (range(1, max_retries + 2)
        # has min 1 iter), so last_result is guaranteed set.
        assert last_result is not None
        return last_result

    def _log_and_report(self, td: Path, result: RunResult) -> None:
        """Log one rollout's outcome and fire the on_result callback."""
        reward = result.rewards.get("reward") if result.rewards else None
        status = "PASS" if reward == 1 else ("FAIL" if reward is not None else "ERR")
        err_msg = result.error or result.verifier_error
        err = f" ({err_msg[:50]})" if err_msg else ""
        logger.info(f"[{status}] {td.name} (tools={result.n_tool_calls}){err}")
        if self._on_result:
            self._on_result(td.name, result)

    async def _run_parallel_independent(
        self, remaining: list[Path]
    ) -> list[tuple[str, RunResult]]:
        """The default schedule — rollouts run concurrently and isolated."""
        cfg = self._config
        sem = asyncio.Semaphore(cfg.concurrency)

        async def bounded(td: Path) -> tuple[str, RunResult]:
            async with sem:
                # Jitter start to avoid SSH connection storms at high concurrency
                import random

                if cfg.concurrency > 16:
                    await asyncio.sleep(
                        random.uniform(0, min(cfg.concurrency / 10, 10))
                    )
                result = await self._run_task(td)
                self._prune_docker()
                self._log_and_report(td, result)
                return td.name, result

        results_or_errors = await asyncio.gather(
            *[bounded(td) for td in remaining],
            return_exceptions=True,
        )

        # Separate successful results from unexpected exceptions
        pairs: list[tuple[str, RunResult]] = []
        for i, r in enumerate(results_or_errors):
            if isinstance(r, BaseException):
                task_name = remaining[i].name
                logger.error(f"[ERR] {task_name}: unexpected exception: {r}")
                pairs.append(
                    (
                        task_name,
                        RunResult(task_name=task_name, error=f"Unexpected: {r}"),
                    )
                )
            else:
                pairs.append(r)
        return pairs

    async def _run_sequential_shared(
        self, remaining: list[Path]
    ) -> list[tuple[str, RunResult]]:
        """The continual-learning schedule — capability 5.

        Rollouts run strictly in order over one persistent, generation-versioned
        ``LearnerStore`` (memory + skills). After each rollout, its reward is
        offered to the store as a learning-curve metric: an improvement stamps a
        new generation, a regression is reverted. The learner store is the one
        snapshot layer that does NOT roll back with a ``Branch`` — its rollback
        is this separate, generation-scoped, curve-driven operation.

        Concurrency is deliberately ignored here: a shared mutable store cannot
        be written by overlapping rollouts.
        """
        from benchflow.learner_store import LearnerStore

        store = self.learner_store
        if store is None:  # defensive — run() sets it for this mode
            store = self.learner_store = LearnerStore()

        pairs: list[tuple[str, RunResult]] = []
        for td in remaining:
            try:
                result = await self._run_task(td)
            except BaseException as e:  # mirror the parallel path's catch
                logger.error(f"[ERR] {td.name}: unexpected exception: {e}")
                pairs.append(
                    (td.name, RunResult(task_name=td.name, error=f"Unexpected: {e}"))
                )
                continue

            self._prune_docker()
            self._log_and_report(td, result)
            pairs.append((td.name, result))

            # Offer the rollout's reward to the learner store. A scored rollout
            # advances or reverts a generation; an errored one (no reward)
            # leaves the store untouched.
            reward = result.rewards.get("reward") if result.rewards else None
            if reward is not None:
                kept = store.commit_or_revert(store.current(), metric=float(reward))
                if not kept:
                    logger.info(
                        f"Learner store: {td.name} regressed "
                        f"(reward={reward}) — reverted, staying at "
                        f"generation {store.generation}"
                    )
        return pairs

    async def run(self) -> EvaluationResult:
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
                    except (json.JSONDecodeError, OSError):
                        logger.debug("Could not read %s", cfg_file)
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

        if cfg.job_mode == "sequential-shared":
            pairs = await self._run_sequential_shared(remaining)
        else:
            pairs = await self._run_parallel_independent(remaining)
        elapsed = time.time() - start

        # Merge with previously completed — normalize everything to dicts
        all_results: dict[str, dict] = {}
        for task, data in completed.items():
            all_results[task] = data
        for name, result in pairs:
            all_results[name] = {
                "task_name": result.task_name,
                "rewards": result.rewards,
                "error": result.error,
                "verifier_error": result.verifier_error,
                "n_tool_calls": result.n_tool_calls,
                "agent_result": _agent_result_from_rollout(result),
            }

        # Count — all values are dicts now, no type branching needed
        job_result = EvaluationResult(
            job_name=self._job_name,
            config=cfg,
            total=len(task_dirs),
            passed=sum(1 for r in all_results.values() if extract_reward(r) == 1.0),
            failed=sum(
                1
                for r in all_results.values()
                if (rw := extract_reward(r)) is not None and rw != 1.0
            ),
            errored=sum(
                1
                for r in all_results.values()
                if r.get("error") and r.get("rewards") is None
            ),
            verifier_errored=sum(
                1 for r in all_results.values() if r.get("verifier_error")
            ),
            elapsed_sec=elapsed,
        )

        assert (
            job_result.passed
            + job_result.failed
            + job_result.errored
            + job_result.verifier_errored
            == job_result.total
        ), (
            f"Counting bug: {job_result.passed}+{job_result.failed}+{job_result.errored}+"
            f"{job_result.verifier_errored} != {job_result.total}"
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
            "verifier_errored": job_result.verifier_errored,
            "score": f"{job_result.score:.1%}",
            "score_excl_errors": f"{job_result.score_excl_errors:.1%}",
            "elapsed_sec": elapsed,
            **_usage_summary(all_results),
        }
        (self._jobs_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        if job_result.verifier_errored > 0:
            pct = job_result.verifier_errored / job_result.total * 100
            logger.warning(
                f"{job_result.verifier_errored} tasks ({pct:.0f}%) had verifier errors — "
                f"check verifier scripts for bugs"
            )
            if pct > 20:
                logger.error(
                    "Over 20% of tasks had verifier errors — results may be unreliable. "
                    "This likely indicates a systemic verifier bug, not agent failure."
                )

        logger.info(
            f"Job complete: {job_result.passed}/{job_result.total} "
            f"({job_result.score:.1%}), errors={job_result.errored}, "
            f"time={elapsed / 60:.1f}min"
        )

        return job_result
