"""Job management — run many tasks against an agent with concurrency, retries, resume.

A ``Job`` wraps ``SDK.run()`` with everything needed to drive a benchmark
to completion: task discovery, parallelism, retry policy, resume from
disk, summary aggregation. The SDK owns *one* task; the Job owns the
fleet.

Pipeline (Job.run, top to bottom)
---------------------------------

::

    _get_task_dirs            scan tasks_dir for valid task.toml dirs
                              (minus JobConfig.exclude_tasks)
    _get_completed_tasks      load result.json files from jobs_dir;
                              completed = has rewards or verifier_error
                              (verifier_error is terminal — no retry)
    resume sanity check       warn if config.json shows a different
                              agent than self._config.agent
    _prune_docker             best-effort container/network cleanup
                              (skipped on non-docker environments)
    bounded(_run_task) ×N     asyncio.Semaphore(concurrency); each task
                              runs through SDK.run, then per-task retry
                              loop in _run_task
    aggregate                 merge new pairs with completed dicts;
                              count passed / failed / errored /
                              verifier_errored — assertion enforces
                              the four buckets sum to total
    summary.json              written into jobs_dir; verifier-error
                              percentage is logged + warned over 20%

Retry policy
------------
Per-task, not per-job. ``_run_task`` calls ``SDK.run`` up to
``RetryConfig.max_retries + 1`` times, breaking out as soon as the
result has rewards, has a verifier_error (terminal), or carries an
error class that ``RetryConfig.should_retry`` rejects. Retryable
classes are ``INSTALL_FAILED``, ``PIPE_CLOSED``, ``ACP_ERROR`` from
``_scoring`` — each independently togglable on ``RetryConfig``.

Config + persistence
--------------------
- ``JobConfig``   dataclass holding agent, model, environment,
                  concurrency, prompts, agent_env, retry, skills_dir,
                  sandbox_user/locked_paths, context_root, exclude_tasks.
                  ``__post_init__`` warns (does *not* fail) on unknown
                  agent so raw-command paths still work.
- ``Job.from_yaml``  dispatcher: detects benchflow-native vs Harbor
                  format by presence of ``agents``/``datasets`` keys,
                  delegates to ``_from_native_yaml`` or
                  ``_from_harbor_yaml``. Both produce a ``JobConfig``.
- ``JobResult``   counts + ``score`` (over total) + ``score_excl_errors``
                  (over passed+failed only). Persisted as ``summary.json``
                  in ``jobs_dir``; per-trial ``result.json`` is written by
                  ``SDK.run``, not here.

Critical invariants
-------------------
- Counting assertion: ``passed + failed + errored + verifier_errored ==
  total``. Any drift means classification logic regressed.
- Resume keys on result.json existence + ``rewards is not None`` or
  ``verifier_error``. The job-level config is *not* part of the resume
  key — see Known issues "Job resume config scoping" in CLAUDE.md.
- Unexpected exceptions inside ``bounded`` are caught by
  ``return_exceptions=True`` and turned into synthetic ``RunResult(error=...)``
  rows so the gather never raises and counts stay consistent.
"""

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

import yaml

from benchflow._scoring import (
    ACP_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    classify_error,
    extract_reward,
    pass_rate,
    pass_rate_excl_errors,
)
from benchflow.models import RunResult
from benchflow.sdk import SDK
from benchflow.trial import Scene

logger = logging.getLogger(__name__)


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
class JobConfig:
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
    # None = unspecified (legacy single-turn execution); non-empty list = multi-scene flow.
    # Empty list `[]` from YAML is coerced to None with a warning in _from_*_yaml; the
    # dataclass itself does not enforce that — direct construction with [] is accepted.
    scenes: list[Scene] | None = None

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
        self._sdk = SDK()  # kept for test mocking compat; _run_task prefers Trial

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
            return cls._from_harbor_yaml(raw, **kwargs)
        return cls._from_native_yaml(raw, **kwargs)

    @classmethod
    def _from_native_yaml(cls, raw: dict, **kwargs) -> "Job":
        """Parse benchflow-native YAML."""
        from benchflow.trial_yaml import parse_scene

        tasks_dir = Path(raw["tasks_dir"])
        jobs_dir = Path(raw.get("jobs_dir", "jobs"))

        # Parse prompts — YAML null becomes Python None
        prompts = raw.get("prompts")

        agent_env_raw = raw.get("agent_env", {})
        exclude = set(raw.get("exclude", []))
        sandbox_user = raw.get("sandbox_user", "agent")
        sandbox_locked_paths = raw.get("sandbox_locked_paths")
        sandbox_setup_timeout = raw.get("sandbox_setup_timeout", 120)

        agent_name = raw.get("agent", DEFAULT_AGENT)
        model = effective_model(agent_name, raw.get("model"))

        # scenes_raw: None → legacy single-turn, [] → warn-then-legacy, list → multi-scene.
        scenes_raw = raw.get("scenes")
        scenes: list[Scene] | None
        if scenes_raw is None:
            scenes = None
        elif not scenes_raw:
            logger.warning(
                "scenes: [] is empty — falling through to legacy single-turn execution. "
                "Remove the key entirely if that is the intent."
            )
            scenes = None
        else:
            scenes = [parse_scene(s) for s in scenes_raw]

        # Derive summary fields from scenes when present so summary.json reflects what runs.
        # Re-resolve through effective_model so the oracle short-circuit applies to the
        # scene's agent (not the top-level one) — otherwise a non-oracle top-level + oracle
        # role would leak DEFAULT_MODEL into JobConfig and disagree with result.json.
        if scenes and scenes[0].roles:
            primary_role = scenes[0].roles[0]
            agent_name = primary_role.agent
            model = effective_model(agent_name, primary_role.model)

        config = JobConfig(
            agent=agent_name,
            model=model,
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
            scenes=scenes,
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

    @classmethod
    def _from_harbor_yaml(cls, raw: dict, **kwargs) -> "Job":
        """Parse Harbor-compatible YAML."""
        from benchflow.trial_yaml import parse_scene

        # Agent — accept both Harbor-canonical agents[].name/model_name and the
        # singular top-level fallback (issue #4: `agent: pi-acp` + top-level
        # `model_name: vllm/...`). agents[].* wins when truthy; falsy values
        # (missing key, empty string) fall through to top-level keys.
        agents = raw.get("agents") or []
        agent_cfg = agents[0] if agents else {}
        agent_name = agent_cfg.get("name") or raw.get("agent", DEFAULT_AGENT)
        model_raw = (
            agent_cfg.get("model_name")
            or raw.get("model_name")
            or raw.get("model")
        )
        # Model — keep provider prefix intact for downstream resolution
        model = effective_model(agent_name, model_raw)

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
        )  # Harbor n_attempts includes first try

        # Skills dir (shared with benchflow-native format)
        skills_dir_raw = raw.get("skills_dir")
        skills_dir = str(Path(skills_dir_raw)) if skills_dir_raw else None
        sandbox_user = raw.get("sandbox_user", "agent")
        sandbox_locked_paths = raw.get("sandbox_locked_paths")
        sandbox_setup_timeout = raw.get("sandbox_setup_timeout", 120)

        # scenes_raw: None → legacy single-turn, [] → warn-then-legacy, list → multi-scene.
        scenes_raw = raw.get("scenes")
        scenes: list[Scene] | None
        if scenes_raw is None:
            scenes = None
        elif not scenes_raw:
            logger.warning(
                "scenes: [] is empty — falling through to legacy single-turn execution. "
                "Remove the key entirely if that is the intent."
            )
            scenes = None
        else:
            scenes = [parse_scene(s) for s in scenes_raw]

        # Derive summary fields from scenes when present so summary.json reflects what runs.
        # See _from_native_yaml: must re-resolve through effective_model so oracle stays None.
        if scenes and scenes[0].roles:
            primary_role = scenes[0].roles[0]
            agent_name = primary_role.agent
            model = effective_model(agent_name, primary_role.model)

        config = JobConfig(
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
            scenes=scenes,
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

    async def _run_single_task(self, task_dir: Path, cfg: JobConfig) -> RunResult:
        """Execute one trial via Trial.

        Uses cfg.scenes when present (multi-scene flow); otherwise synthesizes
        a single Scene from the legacy agent/model/prompts so single-turn jobs
        keep working unchanged.
        """
        from benchflow.trial import Trial, TrialConfig

        resolved_skills = self._resolve_skills_dir(task_dir, cfg.skills_dir)
        scenes = cfg.scenes or [
            Scene.single(
                agent=cfg.agent,
                model=cfg.model,
                prompts=cfg.prompts,
                skills_dir=resolved_skills,
            )
        ]
        trial_config = TrialConfig(
            task_path=task_dir,
            scenes=scenes,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            environment=cfg.environment,
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            context_root=cfg.context_root,
            agent_env=cfg.agent_env,
            skills_dir=resolved_skills,
            # Legacy mirror fields kept for any consumers reading them off TrialConfig:
            agent=cfg.agent,
            model=cfg.model,
            prompts=cfg.prompts,
        )
        trial = await Trial.create(trial_config)
        return await trial.run()

    async def _run_single_task_legacy(
        self, task_dir: Path, cfg: JobConfig
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
                # Log result
                reward = result.rewards.get("reward") if result.rewards else None
                status = (
                    "PASS" if reward == 1 else ("FAIL" if reward is not None else "ERR")
                )
                err_msg = result.error or result.verifier_error
                err = f" ({err_msg[:50]})" if err_msg else ""
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
                pairs.append(
                    (
                        task_name,
                        RunResult(
                            task_name=task_name,
                            error=f"Unexpected: {r}",
                        ),
                    )
                )
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
                "verifier_error": result.verifier_error,
                "n_tool_calls": result.n_tool_calls,
            }

        # Count — all values are dicts now, no type branching needed
        job_result = JobResult(
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
