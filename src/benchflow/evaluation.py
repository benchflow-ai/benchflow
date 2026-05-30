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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from benchflow._utils.evaluation_results import (
    phase_timing_summary,
    rollout_result_payload,
    skill_invocation_summary,
    tool_call_summary,
    usage_summary,
)
from benchflow._utils.learner_memory import (
    attach_memory_score,
    evolved_skills_for_result,
    expected_skills_for_task,
    memory_delta_from_skills,
    patch_learner_generation_artifact,
)
from benchflow._utils.reward_events import memory_summary
from benchflow._utils.scoring import (
    ACP_ERROR,
    IDLE_TIMEOUT,
    INFRA_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    VERIFIER_DEP_INSTALL,
    VERIFIER_INFRA,
    VERIFIER_TIMEOUT,
    classify_error,
    classify_verifier_error,
    count_audit_outcomes,
    count_score_outcomes,
    pass_rate,
    pass_rate_excl_errors,
)
from benchflow._utils.source_provenance import summary_source_fields
from benchflow.diagnostics import DIAGNOSTIC_REGISTRY, summary_warning
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.learner_store import LearnerState, LearnerStore
from benchflow.models import RolloutResult
from benchflow.trajectories.tree import RolloutNode
from benchflow.usage_tracking import UsageTrackingConfig

# Backward-compat alias
RunResult = RolloutResult

logger = logging.getLogger(__name__)

# Label applied to every container/network BenchFlow's compose files create.
# Used to scope Docker prune calls so we only delete our own resources and never
# touch unrelated containers/networks on shared developer or CI hosts.
BENCHFLOW_OWNED_LABEL = "benchflow.owned=true"

# Serialize docker prune across concurrent _run_task retries. When --concurrency
# is high (e.g. 60) and tasks retry in lockstep, parallel `docker container
# prune` calls each block on the daemon and time out at 30s, cascading into
# false install_failure errors. Non-blocking acquire: if a prune is already in
# flight, skip — there's nothing new to clean since the in-flight one started.
_PRUNE_LOCK = threading.Lock()

_SENTINEL: Any = object()  # default value for _sdk; tests replace with AsyncMock


class EmptyTaskSelectionError(ValueError):
    """Raised when task discovery + include/exclude filters resolve to zero tasks.

    Failing fast is preferred over silently writing a 0/0 summary.json that
    downstream dashboards may ingest as evidence (#407).
    """


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
    retry_on_idle_timeout: bool = True
    retry_on_infra: bool = True
    retry_on_verifier_infra: bool = True
    wait_multiplier: float = 2.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    exclude_categories: set[str] = field(default_factory=lambda: {"timeout"})

    def should_retry(
        self,
        error: str | None,
        *,
        category: str | None = None,
    ) -> bool:
        """Check if an error is retryable."""
        category = category or classify_error(error)
        if not category:
            return False
        if category in self.exclude_categories:
            return False
        if self.retry_on_install and category == INSTALL_FAILED:
            return True
        if self.retry_on_pipe and category == PIPE_CLOSED:
            return True
        if self.retry_on_idle_timeout and category == IDLE_TIMEOUT:
            return True
        if self.retry_on_infra and category == INFRA_ERROR:
            return True
        return bool(self.retry_on_acp and category == ACP_ERROR)

    def should_retry_verifier_error(self, verifier_error: str | None) -> bool:
        """Check if a verifier error is infrastructure-retryable."""
        if not self.retry_on_verifier_infra:
            return False
        return classify_verifier_error(verifier_error) in {
            VERIFIER_INFRA,
            VERIFIER_TIMEOUT,
        }

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

    Resolution order:
      1. An explicit ``--model`` always wins.
      2. The agent's own ``default_model`` (e.g. ``gemini-2.5-flash`` for the
         gemini agent) — keeps each agent on its native provider.
      3. ``DEFAULT_MODEL`` only when the caller is on the default agent.
         Substituting it under any other agent silently cross-wires providers
         and was the root cause of #343 (gemini eval demanding ANTHROPIC_API_KEY).

    Oracle runs solve.sh and never calls an LLM, so it never receives a model
    (the chokepoint in resolve_agent_env defends, but callers should also stop
    materializing DEFAULT_MODEL into oracle configs to keep the data honest —
    e.g. result-summary JSON shows model=null instead of a bogus default).
    """
    if agent == "oracle":
        return None
    if model:
        return model
    # Look up the agent's own default. Unknown agents (raw-command fallback)
    # bypass the registry lookup and use the global default.
    from benchflow.agents.registry import AGENTS

    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.default_model:
        return agent_cfg.default_model
    if agent == DEFAULT_AGENT or agent_cfg is None:
        return DEFAULT_MODEL
    raise ValueError(
        f"agent {agent!r} has no default model; pass --model "
        f"(refusing to fall back to {DEFAULT_MODEL!r} from a different provider)"
    )


@dataclass
class EvaluationConfig:
    """Configuration for a benchmark job."""

    agent: str = DEFAULT_AGENT
    model: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    build_concurrency: int | None = None
    prompts: list[str | None] | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    skills_dir: str | None = None
    include_task_skills: bool = False
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    agent_idle_timeout: int | None = 600
    context_root: str | None = None
    exclude_tasks: set[str] = field(default_factory=set)
    include_tasks: set[str] = field(default_factory=set)
    skill_mode: str = "default"
    skill_creator_dir: str | None = None
    self_gen_no_internet: bool = False
    job_mode: str = DEFAULT_JOB_MODE
    source_provenance: dict[str, Any] | None = None
    usage_tracking: UsageTrackingConfig = field(default_factory=UsageTrackingConfig)
    # Environment-plane manifest applied to every rollout in the batch.
    # When set, each task's RolloutConfig.environment_manifest is populated
    # so the Environment plane (manifest-declared stateful environment,
    # readiness gating, teardown) is exercised — closing the gap between
    # single-rollout SDK.run() and the batch Evaluation/Job API (#398).
    environment_manifest: EnvironmentManifest | None = None

    def __post_init__(self):
        from benchflow._utils.config import (
            normalize_agent_idle_timeout,
            normalize_agent_name,
            normalize_sandbox_user,
        )
        from benchflow.agents.registry import AGENTS

        self.agent = normalize_agent_name(self.agent)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        self.usage_tracking = UsageTrackingConfig.coerce(self.usage_tracking)
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
    memory_score: float | None = None
    memory_scores: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)


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

    @staticmethod
    def _resolve_job_name(jobs_dir: Path) -> str:
        """Pick a job_name when none was explicitly provided.

        If ``jobs_dir`` already contains exactly one timestamped job
        directory, reuse it so that a second ``Evaluation.run()`` call
        resumes into the same directory instead of creating an orphan.
        When zero job dirs exist (or ``jobs_dir`` itself does not exist),
        fall back to a fresh timestamp.  When multiple exist, resume into
        the most recent (alphabetically last).

        Guards ENG-160: auto-generated job_name must be stable across
        resume calls.
        """
        if jobs_dir.is_dir():
            job_dirs = sorted(
                d
                for d in jobs_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            if len(job_dirs) == 1:
                logger.info(f"Resuming into existing job directory: {job_dirs[0].name}")
                return job_dirs[0].name
            if len(job_dirs) > 1:
                latest = job_dirs[-1]
                logger.info(
                    f"Multiple job directories found ({len(job_dirs)}); "
                    f"resuming into most recent: {latest.name}"
                )
                return latest.name
        return datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

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
        self._job_name = job_name or self._resolve_job_name(self._jobs_dir)
        self._explicit_job_name = job_name is not None
        self._on_result = on_result
        # Kept for test mocking compat; _run_task prefers Rollout
        from benchflow.sdk import SDK

        self._sdk = SDK()
        # The persistent learner store for sequential-shared (continual
        # learning) jobs — the one owner. parallel-independent jobs leave it
        # None.
        #
        # On resume, the store is restored from the per-job JSON snapshot so
        # rollout N+1 still inherits the (memory + skills) state earlier
        # rollouts evolved. Without this restore an interrupted continual-
        # learning job would silently mix old result rows with a fresh empty
        # store (issue #394).
        self.learner_store: LearnerStore | None = (
            self._load_or_init_learner_store()
            if self._config.job_mode == "sequential-shared"
            else None
        )
        # Per-rollout continual-learning skill dirs, set by
        # _run_sequential_shared before each _run_task call and consumed by
        # _run_single_task. None outside sequential-shared mode.
        self._learner_skills_dir: Path | None = None
        self._learner_export_dir: Path | None = None
        # One RolloutNode per sequential-shared rollout, each carrying that
        # rollout's memory_delta — the Memory-space scorer's input.
        self.learner_nodes: list[RolloutNode] = []

    def _learner_store_path(self) -> Path:
        """Where the persisted LearnerStore snapshot lives for this job."""
        return self._jobs_dir / self._job_name / "learner_store.json"

    def _load_or_init_learner_store(self) -> LearnerStore:
        """Restore the per-job LearnerStore snapshot, or start fresh.

        A corrupt snapshot is a hard failure rather than a silent reset: a
        resumed continual-learning job that secretly started from an empty
        store is exactly the bug this guards (issue #394).
        """
        snapshot = self._learner_store_path()
        if not snapshot.is_file():
            return LearnerStore()
        try:
            store = LearnerStore.load(snapshot)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Could not load persisted LearnerStore from {snapshot}: {e}. "
                f"Delete the file or use a fresh jobs_dir to start a new run."
            ) from e
        logger.info(
            f"Resumed LearnerStore from {snapshot} at generation "
            f"{store.generation} ({len(store.history) - 1} prior rollouts)"
        )
        return store

    def _save_learner_store(self) -> None:
        """Persist the current LearnerStore so the next process can resume it."""
        if self.learner_store is None:
            return
        try:
            self.learner_store.save(self._learner_store_path())
        except OSError as e:
            logger.warning(f"Could not persist LearnerStore: {e}")

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
            resolve_source_with_metadata,
        )

        # New two-field format: source.repo + source.path
        source_provenance = None
        if "source" in raw:
            src = raw["source"]
            resolved = resolve_source_with_metadata(
                repo=src["repo"],
                path=src.get("path"),
                ref=src.get("ref"),
            )
            tasks_dir = resolved.path
            source_provenance = resolved.provenance
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
        # Optional environment-plane manifest path. Keeps YAML and CLI in
        # sync so manifest-backed evaluations can be driven from either
        # (#398).
        env_manifest_raw = raw.get("environment_manifest")
        env_manifest: EnvironmentManifest | None = None
        if env_manifest_raw is not None:
            from benchflow.environment.manifest import load_manifest

            env_manifest = load_manifest(env_manifest_raw)
        config = EvaluationConfig(
            agent=agent_name,
            model=effective_model(agent_name, raw.get("model")),
            environment=raw.get("environment", "docker"),
            concurrency=raw.get("concurrency", 4),
            build_concurrency=raw.get("build_concurrency"),
            prompts=prompts,
            agent_env=agent_env_raw,
            retry=RetryConfig(max_retries=raw.get("max_retries", 2)),
            skills_dir=str(Path(raw["skills_dir"])) if raw.get("skills_dir") else None,
            include_task_skills=bool(raw.get("include_task_skills", False)),
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
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
            source_provenance=source_provenance,
            usage_tracking=UsageTrackingConfig.from_mapping(raw),
            environment_manifest=env_manifest,
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

        # Map legacy include/exclude task filters. Accept both singular and
        # plural spellings ("include"/"includes", "exclude"/"excludes") so
        # ported configs do not silently lose their filtering (#500).
        include: set[str] = set()
        for key in ("include", "includes", "include_tasks"):
            values = raw.get(key)
            if values:
                include.update(values)
        exclude: set[str] = set()
        for key in ("exclude", "excludes", "exclude_tasks"):
            values = raw.get(key)
            if values:
                exclude.update(values)

        config = EvaluationConfig(
            agent=agent_name,
            model=model,
            environment=environment,
            concurrency=concurrency,
            agent_env=agent_env,
            retry=RetryConfig(max_retries=max(0, max_retries)),
            skills_dir=skills_dir,
            include_task_skills=bool(raw.get("include_task_skills", False)),
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
            include_tasks=include,
            exclude_tasks=exclude,
            skill_mode=raw.get("skill_mode", "default"),
            skill_creator_dir=(
                str(Path(raw["skill_creator_dir"]))
                if raw.get("skill_creator_dir")
                else None
            ),
            self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
            usage_tracking=UsageTrackingConfig.from_mapping(raw),
        )
        return cls(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs)

    def _get_task_dirs(self) -> list[Path]:
        """Get all valid task directories."""
        if (self._tasks_dir / "task.toml").exists():
            if self._tasks_dir.name in self._config.exclude_tasks:
                return []
            if (
                self._config.include_tasks
                and self._tasks_dir.name not in self._config.include_tasks
            ):
                return []
            return [self._tasks_dir]
        return sorted(
            d
            for d in self._tasks_dir.iterdir()
            if d.is_dir()
            and (d / "task.toml").exists()
            and d.name not in self._config.exclude_tasks
            and (not self._config.include_tasks or d.name in self._config.include_tasks)
        )

    def _get_completed_tasks(self) -> dict[str, dict]:
        """Load tasks that already have results with rewards or verifier errors.

        Scoped to the current job directory (``_jobs_dir / _job_name``) to
        prevent cross-job contamination.  When multiple result.json files
        exist for the same task (retry artifacts), the newest by mtime wins.

        Guards ENG-160: orphan retry artifacts no longer pollute resume.
        """
        job_dir = self._jobs_dir / self._job_name
        if not job_dir.exists():
            return {}
        # Collect every result keyed by (task_name) → keep newest by mtime.
        best: dict[str, tuple[float, dict]] = {}
        for rfile in job_dir.rglob("result.json"):
            try:
                r = json.loads(rfile.read_text())
                task = r["task_name"]
                if r.get("rewards") is not None or r.get("verifier_error"):
                    mtime = rfile.stat().st_mtime
                    prev = best.get(task)
                    if prev is None or (mtime, str(rfile)) >= (prev[0], ""):
                        best[task] = (mtime, r)
            except Exception as e:
                logger.debug(f"Skipping corrupt result file {rfile}: {e}")
        completed: dict[str, dict] = {}
        for task, (_mt, r) in best.items():
            if r.get("verifier_error"):
                logger.info(
                    f"Skipping verifier-errored task on resume: {task} ({r['verifier_error'][:80]})"
                )
            completed[task] = r
        return completed

    def _prune_docker(self):
        """Clean up Docker resources owned by BenchFlow.

        Scoped via ``--filter label=benchflow.owned=true`` so we only remove
        containers/networks our own compose files created. Unrelated Docker
        workloads on the same host are left untouched. The label is applied in
        ``sandbox/_compose_files/docker-compose-base.yaml``.

        Serialized via ``_PRUNE_LOCK``: parallel retries from high-concurrency
        batches would otherwise each kick off a 30s-timeout docker CLI call,
        all blocking on the same daemon. Non-blocking acquire — if another
        prune is in flight we just skip, since it will catch the same garbage.
        """
        if self._config.environment != "docker":
            return
        if not _PRUNE_LOCK.acquire(blocking=False):
            return
        label_filter = f"label={BENCHFLOW_OWNED_LABEL}"
        try:
            subprocess.run(
                [
                    "docker",
                    "container",
                    "prune",
                    "-f",
                    "--filter",
                    label_filter,
                ],
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                [
                    "docker",
                    "network",
                    "prune",
                    "-f",
                    "--filter",
                    label_filter,
                ],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Docker prune failed: {e}")
        finally:
            _PRUNE_LOCK.release()

    def _resolve_skills_dir(self, task_dir: Path, skills_dir: str | None) -> str | None:
        """Resolve skills_dir — 'auto' means per-task environment/skills/."""
        from benchflow.skill_policy import resolve_runtime_skills_dir

        resolved = resolve_runtime_skills_dir(task_dir, skills_dir)
        return str(resolved) if resolved is not None else None

    def _enrich_payload_with_persisted_timing(
        self, payload: dict, result: RolloutResult
    ) -> None:
        """Copy ``timing`` from the rollout's on-disk result.json into payload.

        ``RolloutResult`` does not carry phase timing, but the rollout writer
        (``rollout.py``) persists it under ``rollout_dir/result.json``. Reading
        it back lets ``phase_timing_summary`` aggregate phase totals for fresh
        runs (issue #501). Best-effort: legacy SDK paths that mock the writer
        — or any case where no rollout_name is set — silently leave timing
        absent rather than crash summary generation.
        """
        if "timing" in payload:
            return
        rollout_name = getattr(result, "rollout_name", "") or ""
        if not rollout_name:
            return
        rfile = self._jobs_dir / self._job_name / rollout_name / "result.json"
        if not rfile.exists():
            return
        try:
            persisted = json.loads(rfile.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Could not read persisted timing from %s: %s", rfile, e)
            return
        timing = persisted.get("timing")
        if isinstance(timing, dict):
            payload["timing"] = timing

    async def _run_single_task(
        self, task_dir: Path, cfg: EvaluationConfig
    ) -> RolloutResult:
        """Execute one rollout via Rollout.

        In sequential-shared mode the per-rollout learner skill dirs override
        the static config: the rollout starts from the LearnerStore's evolved
        skill set (``_learner_skills_dir``) and its agent-evolved skills are
        captured back through ``export_generated_skills_to``.
        """
        from benchflow._utils.benchmark_repos import task_source_provenance
        from benchflow.rollout import Rollout, RolloutConfig

        skills_dir = (
            str(self._learner_skills_dir)
            if self._learner_skills_dir is not None
            else self._resolve_skills_dir(task_dir, cfg.skills_dir)
        )
        export_to = (
            str(self._learner_export_dir)
            if self._learner_export_dir is not None
            else None
        )
        rollout_config = RolloutConfig.from_legacy(
            task_path=task_dir,
            agent=cfg.agent,
            model=cfg.model,
            prompts=cfg.prompts,
            agent_env=cfg.agent_env,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            concurrency=cfg.concurrency,
            environment=cfg.environment,
            environment_manifest=cfg.environment_manifest,
            skills_dir=skills_dir,
            include_task_skills=cfg.include_task_skills,
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            agent_idle_timeout=cfg.agent_idle_timeout,
            context_root=cfg.context_root,
            skill_mode=cfg.skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
            export_generated_skills_to=export_to,
            source_provenance=task_source_provenance(cfg.source_provenance, task_dir),
            usage_tracking=cfg.usage_tracking,
        )
        if cfg.skill_mode == "self-gen":
            from benchflow.self_gen import run_self_gen

            return await run_self_gen(rollout_config)
        rollout = await Rollout.create(rollout_config)
        return await rollout.run()

    async def _run_single_task_legacy(
        self, task_dir: Path, cfg: EvaluationConfig
    ) -> RunResult:
        """SDK.run() path — used when _sdk is mocked in tests.

        Note: this legacy path does NOT thread the continual-learning skill
        dirs (``_learner_skills_dir`` / ``_learner_export_dir``), so it
        cannot materialize or capture evolved skills. It is test-only today;
        a real continual-learning run must go through ``_run_single_task``.
        """
        from benchflow._utils.benchmark_repos import task_source_provenance

        return await self._sdk.run(
            task_path=task_dir,
            agent=cfg.agent,
            model=cfg.model,
            prompts=cfg.prompts,
            agent_env=cfg.agent_env,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            concurrency=cfg.concurrency,
            environment=cfg.environment,
            skills_dir=self._resolve_skills_dir(task_dir, cfg.skills_dir),
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            agent_idle_timeout=cfg.agent_idle_timeout,
            context_root=cfg.context_root,
            skill_mode=cfg.skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
            source_provenance=task_source_provenance(cfg.source_provenance, task_dir),
            usage_tracking=cfg.usage_tracking,
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

            retryable_agent_error = cfg.retry.should_retry(
                result.error,
                category=result.error_category,
            )
            retryable_verifier_error = cfg.retry.should_retry_verifier_error(
                result.verifier_error
            )

            # If succeeded, verifier-errored (terminal), or non-retryable, stop.
            # Retryable infra/idle errors win over fallback rewards so a hung
            # agent lane does not become permanent failed-task data at scale.
            if not (retryable_agent_error or retryable_verifier_error):
                break

            if attempt <= cfg.retry.max_retries:
                err_preview = (result.error or result.verifier_error or "")[:60]
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

    def _preflight_usage_tracking(self) -> None:
        from benchflow.providers.runtime import validate_usage_proxy_preconditions

        cfg = self._config
        usage = cfg.usage_tracking.with_env_defaults()
        usage.validate_parallelism(concurrency=cfg.concurrency)
        failure = validate_usage_proxy_preconditions(
            usage,
            environment=cfg.environment,
            model=cfg.model,
        )
        if failure is None:
            return
        if usage.mode == "required":
            raise RuntimeError(failure.required_message)
        logger.log(
            failure.log_level,
            "%s Results will report usage_source='unavailable'.",
            failure.skip_message,
        )

    async def _run_parallel_independent(
        self, remaining: list[Path]
    ) -> list[tuple[str, RunResult]]:
        """The default schedule — rollouts run concurrently and isolated."""
        cfg = self._config
        sem = asyncio.Semaphore(cfg.concurrency)

        async def bounded(td: Path) -> tuple[str, RunResult]:
            async with sem:
                # Jitter start to avoid SSH/docker-daemon storms at high
                # concurrency. The window scales linearly with --concurrency so
                # the average start rate stays around 2 tasks/sec; the previous
                # 10s cap was too tight for c >= 30 (≈10 starts/sec flooded the
                # daemon's compose-up handler).
                import random

                if cfg.concurrency > 16:
                    jitter_max = max(cfg.concurrency / 2, 8.0)
                    await asyncio.sleep(random.uniform(0, jitter_max))
                result = await self._run_task(td)
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
                if isinstance(r, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise r
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
        ``LearnerStore`` (memory + skills). Each rollout:

        1. **reads** the store's current skills and injects them as its
           ``skills_dir``, so it starts from the *evolved* skill set;
        2. **runs**, with ``export_generated_skills_to`` set so the skills the
           agent generated/evolved are captured;
        3. **records** the before/after skills as ``memory_delta`` on a tree
           node, giving the Memory-space scorer its writer; and
        4. **commits** the captured skills to the store as the next
           ``LearnerState`` — so rollout N+1 inherits them.

        The rollout's reward is offered as a learning-curve metric: an
        improvement stamps a new generation, a regression is rejected and the
        store stays at the better generation. The learner store is the one
        snapshot layer that does NOT roll back with a ``Branch`` — this
        curve-driven rollback is a separate, generation-scoped operation.

        Concurrency is deliberately ignored here: a shared mutable store cannot
        be written by overlapping rollouts.
        """
        import tempfile

        from benchflow.learner_skills import materialize_skills

        # __init__ is the sole owner: it constructs the store whenever
        # job_mode is sequential-shared, the only mode that reaches here.
        store = self.learner_store
        assert store is not None, "sequential-shared job must have a learner_store"

        # Per-run scoring scratch — reset so re-running the same Evaluation
        # does not score stale nodes carried over from a prior invocation.
        self.learner_nodes = []

        pairs: list[tuple[str, RunResult]] = []
        with tempfile.TemporaryDirectory(prefix="bf-learner-") as work:
            work_root = Path(work)
            for i, td in enumerate(remaining):
                # 1. READ — materialize the store's current skills so the
                # rollout starts from the evolved set.
                before_state = store.current()
                before_generation = store.generation
                skills_dir = work_root / f"rollout-{i}-skills"
                export_dir = work_root / f"rollout-{i}-evolved"
                materialize_skills(before_state, skills_dir)
                self._learner_skills_dir = skills_dir
                self._learner_export_dir = export_dir

                try:
                    result = await self._run_task(td)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as e:  # mirror the parallel path's catch
                    logger.error(f"[ERR] {td.name}: unexpected exception: {e}")
                    pairs.append(
                        (
                            td.name,
                            RunResult(task_name=td.name, error=f"Unexpected: {e}"),
                        )
                    )
                    continue
                finally:
                    self._learner_skills_dir = None
                    self._learner_export_dir = None

                self._log_and_report(td, result)
                pairs.append((td.name, result))

                await self._commit_learner_generation(
                    store, td, result, before_state, before_generation, export_dir
                )
        return pairs

    async def _commit_learner_generation(
        self,
        store: LearnerStore,
        td: Path,
        result: RunResult,
        before_state: LearnerState,
        before_generation: int,
        export_dir: Path,
    ) -> None:
        """Capture a rollout's evolved skills and commit the next generation.

        Builds the ``memory_delta`` record the Memory-space scorer reads, then
        offers the captured (memory + skills) state to the store: an
        improvement stamps a new generation, a regression is reverted. An
        errored rollout (no reward) leaves the store untouched.

        Persists the store and stamps generation metadata onto the result
        artifact (which inherited from / which it produced) so a resumed job
        can audit the learning curve across processes — see issue #394.
        """
        # Skip everything when the skill export itself failed (#389 follow-up).
        # The export dir is half-written and ``result.evolved_skills`` is None,
        # so committing would poison the LearnerStore with an empty/partial
        # generation even though the verifier may have produced rewards.
        if result.export_error is not None:
            logger.warning(
                f"Learner store: {td.name} skill export failed — "
                f"skipping generation commit, staying at generation "
                f"{store.generation}"
            )
            return
        # 2/3. CAPTURE — the skills the agent generated/evolved. Prefer the
        # result's own field (the real Rollout populates it); fall back to
        # reading the export dir directly.
        evolved_skills = evolved_skills_for_result(result, export_dir)
        expected_skills = expected_skills_for_task(td)
        # The Memory scorer must NOT derive an answer key from the agent's own
        # diff — that would make precision/recall a tautology. Only a
        # task-authored fixture may switch the scorer from activity to
        # correctness grading.
        after_skills, delta = memory_delta_from_skills(
            before_state=before_state,
            evolved_skills=evolved_skills,
            expected_skills=expected_skills,
        )

        # Record the delta on this rollout's tree node so the Memory-space
        # scorer (rewards/memory_scorer.py) has its writer — the two halves
        # of capability 5 connected end-to-end.
        node = self._learner_node(td)
        result_path = (
            self._jobs_dir / self._job_name / result.rollout_name / "result.json"
            if result.rollout_name
            else None
        )
        await attach_memory_score(
            result=result,
            node=node,
            delta=delta,
            result_path=result_path,
        )

        # 4. COMMIT — offer the evolved (memory + skills) state to the store.
        reward = result.rewards.get("reward") if result.rewards else None
        committed_generation: int | None = None
        kept: bool | None = None
        if reward is not None:
            # Commit the normalized `after_skills` (str-valued) — not the raw
            # `evolved_skills` — so the committed store state is byte-identical
            # to the `memory_delta` recorded above.
            next_state = LearnerState(
                memory=before_state.memory,
                skills=after_skills,
            )
            kept = store.commit_or_revert(next_state, metric=float(reward))
            if kept:
                committed_generation = store.generation
            else:
                logger.info(
                    f"Learner store: {td.name} regressed (reward={reward}) — "
                    f"reverted, staying at generation {store.generation}"
                )

        # Persist the store after every rollout so an interrupted job can
        # resume from the last committed generation (#394). We save even when
        # the rollout did not commit (errored or reverted) so the snapshot's
        # pointer matches the live store.
        self._save_learner_store()

        # Stamp generation metadata on the result artifact so a resumed run
        # can audit which rollout inherited which store generation.
        if result_path is not None:
            patch_learner_generation_artifact(
                result_path,
                inherited_from=before_generation,
                produced=committed_generation,
                committed=kept,
            )

    def _learner_node(self, td: Path) -> RolloutNode:
        """Return a fresh tree node for one continual-learning rollout.

        Each sequential-shared rollout is one node carrying that rollout's
        ``memory_delta``; the Job keeps them on ``learner_nodes`` so the
        Memory-space scorer can score every rollout after the run.
        """
        # Index-prefixed so two rollouts of the same task name still get
        # distinct node ids.
        node = RolloutNode(id=f"{len(self.learner_nodes)}-{td.name}")
        self.learner_nodes.append(node)
        return node

    async def run(self) -> EvaluationResult:
        """Execute the job."""
        task_dirs = self._get_task_dirs()
        if not task_dirs:
            # Fail fast on an empty selection (#407). Silently writing a
            # 0/0 summary.json would surface as an apparently successful
            # eval in downstream dashboards and release evidence.
            cfg = self._config
            detail_parts = [f"tasks_dir={self._tasks_dir}"]
            if cfg.include_tasks:
                detail_parts.append(f"include={sorted(cfg.include_tasks)}")
            if cfg.exclude_tasks:
                detail_parts.append(f"exclude={sorted(cfg.exclude_tasks)}")
            raise EmptyTaskSelectionError(
                "No tasks selected after include/exclude filtering "
                f"({', '.join(detail_parts)}). Refusing to publish an "
                "empty 0/0 summary."
            )
        completed = self._get_completed_tasks()
        remaining = [d for d in task_dirs if d.name not in completed]
        if remaining:
            self._preflight_usage_tracking()

        # A resumed sequential-shared job rebuilds the LearnerStore from the
        # per-job snapshot under ``<job>/learner_store.json``. If that file
        # is missing while completed rollouts exist, the run cannot honestly
        # continue the learning curve — the older rollouts' evolved skills
        # are lost. Fail closed (#394) rather than silently mix old result
        # rows with a fresh empty store.
        if completed and self._config.job_mode == "sequential-shared":
            snapshot = self._learner_store_path()
            if not snapshot.is_file():
                raise RuntimeError(
                    f"Cannot resume sequential-shared job: "
                    f"{len(completed)} completed task(s) but no persisted "
                    f"LearnerStore at {snapshot}. The learning curve would "
                    f"restart at generation 0 and earlier rollouts' evolved "
                    f"skills are lost. Use a fresh jobs_dir for a clean run, "
                    f"or restore the snapshot from a backup."
                )
            assert self.learner_store is not None
            logger.info(
                f"Resuming sequential-shared job at generation "
                f"{self.learner_store.generation} "
                f"({len(completed)} completed task(s), "
                f"{len(remaining)} remaining)"
            )

        # Warn if resuming with different config than completed tasks
        if completed:
            # Check config.json (written by SDK.run) for the registry agent name
            job_dir = self._jobs_dir / self._job_name
            sample_dir = (
                next(
                    (d for d in job_dir.iterdir() if d.is_dir()),
                    None,
                )
                if job_dir.exists()
                else None
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

        if cfg.build_concurrency is not None and cfg.environment == "docker":
            from benchflow.sandbox.docker import DockerSandbox

            DockerSandbox.set_build_concurrency(cfg.build_concurrency)

        logger.info(
            f"Job: {len(task_dirs)} tasks, {len(completed)} done, "
            f"{len(remaining)} to run (concurrency={cfg.concurrency})"
        )

        start = time.time()

        if cfg.job_mode == "sequential-shared":
            pairs = await self._run_sequential_shared(remaining)
        else:
            pairs = await self._run_parallel_independent(remaining)
        self._prune_docker()
        elapsed = time.time() - start

        all_results: dict[str, dict] = {}
        for task, data in completed.items():
            all_results[task] = data
        for name, result in pairs:
            payload = rollout_result_payload(
                result,
                source_provenance=cfg.source_provenance,
                tasks_dir=self._tasks_dir,
                task_name=name,
            )
            # ``rollout_result_payload`` is RolloutResult-driven and so cannot
            # see ``timing`` (it lives only in the persisted result.json).
            # Pull it from disk so phase-timing aggregates cover fresh pairs
            # the same way they cover resumed tasks (issue #501).
            self._enrich_payload_with_persisted_timing(payload, result)
            all_results[name] = payload

        # EvaluationResult is the score/invariant view. summary.json is the
        # audit view consumed by result checkers, so verifier evidence remains
        # visible there even when the score view gives agent errors precedence.
        score_counts = count_score_outcomes(all_results.values())
        audit_counts = count_audit_outcomes(all_results.values())
        memory, memory_scores = memory_summary(all_results)
        job_result = EvaluationResult(
            job_name=self._job_name,
            config=cfg,
            total=len(task_dirs),
            passed=score_counts["passed"],
            failed=score_counts["failed"],
            errored=score_counts["errored"],
            verifier_errored=score_counts["verifier_errored"],
            elapsed_sec=elapsed,
            memory_score=memory["avg_score"],
            memory_scores=memory_scores,
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

        # Count error categories across all results for summary diagnostics.
        error_category_counts: dict[str, int] = {}
        verifier_error_category_counts: dict[str, int] = {}
        for r in all_results.values():
            cat = r.get("error_category") or classify_error(r.get("error"))
            if cat:
                error_category_counts[cat] = error_category_counts.get(cat, 0) + 1
            vcat = r.get("verifier_error_category") or classify_verifier_error(
                r.get("verifier_error")
            )
            if vcat:
                verifier_error_category_counts[vcat] = (
                    verifier_error_category_counts.get(vcat, 0) + 1
                )

        # Save summary
        summary = {
            "job_name": self._job_name,
            "agent": cfg.agent,
            "model": cfg.model,
            "environment": cfg.environment,
            "concurrency": cfg.concurrency,
            "agent_idle_timeout_sec": cfg.agent_idle_timeout,
            "usage_tracking": cfg.usage_tracking.with_env_defaults().to_config_artifact(),
            "total": job_result.total,
            "passed": audit_counts["passed"],
            "failed": audit_counts["failed"],
            "errored": audit_counts["errored"],
            "verifier_errored": audit_counts["verifier_errored"],
            "idle_timeout": error_category_counts.get(IDLE_TIMEOUT, 0),
            "error_categories": error_category_counts or None,
            "verifier_error_categories": verifier_error_category_counts or None,
            "score": f"{pass_rate(passed=audit_counts['passed'], total=job_result.total):.1%}",
            "score_excl_errors": f"{pass_rate_excl_errors(passed=audit_counts['passed'], failed=audit_counts['failed']):.1%}",
            "elapsed_sec": elapsed,
            "memory_score": job_result.memory_score,
            "memory_score_coverage": (
                len(memory_scores) / job_result.total if job_result.total else 0.0
            ),
            "memory": memory,
            "memory_scores": memory_scores,
            **skill_invocation_summary(all_results),
            **usage_summary(all_results),
            **tool_call_summary(all_results),
            **phase_timing_summary(all_results),
            **summary_source_fields(cfg.source_provenance, all_results),
        }
        # Surface continual-learning provenance — generation, curve — so a
        # resumed run can be audited end-to-end (#394).
        if cfg.job_mode == "sequential-shared" and self.learner_store is not None:
            summary["learner_store"] = {
                "generation": self.learner_store.generation,
                "learning_curve": self.learner_store.learning_curve(),
                "snapshot_path": str(
                    self._learner_store_path().relative_to(self._jobs_dir)
                ),
            }
        # Write summary into the job directory so each run is self-contained.
        job_dir = self._jobs_dir / self._job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        summary_text = json.dumps(summary, indent=2)
        (job_dir / "summary.json").write_text(summary_text)
        # Backward-compat: also write to jobs_dir root for tooling that
        # expects summary.json at the top level.
        (self._jobs_dir / "summary.json").write_text(summary_text)

        # Aggregate per-rollout trainer artifacts into job_dir/verifiers.jsonl
        # — the architecture's train-mode seam (issue #385).
        try:
            from benchflow.trajectories.export import write_job_verifiers_jsonl

            write_job_verifiers_jsonl(job_dir)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Job-level trainer artifact aggregation failed: %s", e)

        # Per-diagnostic summary warnings — driven by the registry so a
        # new diagnostic class adds its warning automatically (issue #503).
        for diag_cls in DIAGNOSTIC_REGISTRY:
            if diag_cls.category is None:
                continue
            counts = (
                error_category_counts
                if diag_cls.channel == "error"
                else verifier_error_category_counts
            )
            count = counts.get(diag_cls.category, 0)
            if count > 0:
                logger.warning(summary_warning(diag_cls, count, job_result.total))

        # ENG-151: dep-install failures don't have a structured diagnostic
        # yet — keep the standalone warning until they do.
        dep_install_count = verifier_error_category_counts.get(VERIFIER_DEP_INSTALL, 0)
        if dep_install_count > 0:
            pct = dep_install_count / job_result.total * 100
            logger.warning(
                f"{dep_install_count} tasks ({pct:.0f}%) failed during verifier "
                f"dependency install — check verifier_error_category in result.json "
                f"and fix the task's index policy"
            )
        if audit_counts["verifier_errored"] > 0:
            pct = audit_counts["verifier_errored"] / job_result.total * 100
            logger.warning(
                f"{audit_counts['verifier_errored']} tasks ({pct:.0f}%) had verifier errors — "
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
            f"idle_timeouts={error_category_counts.get(IDLE_TIMEOUT, 0)}, "
            f"time={elapsed / 60:.1f}min"
        )

        return job_result
