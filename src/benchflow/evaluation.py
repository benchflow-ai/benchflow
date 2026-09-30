"""Evaluation management — run many tasks against an agent with concurrency, retries, resume.

An ``Evaluation`` wraps ``bf.run()`` with everything needed to drive a benchmark
to completion: task discovery, parallelism, retry policy, resume from
disk, summary aggregation.

Backward-compat aliases: ``Job = Evaluation``, ``JobConfig = EvaluationConfig``,
``JobResult = EvaluationResult``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from benchflow._utils.evaluation_results import (
    loop_summary,
    phase_timing_summary,
    rollout_result_payload,
    skill_invocation_summary,
    solve_rate_summary,
    tool_call_summary,
    trajectory_step_summary,
    usage_summary,
)
from benchflow._utils.learner_memory import (
    attach_memory_score,
    evolved_skills_for_result,
    expected_skills_for_task,
    memory_delta_from_skills,
    patch_learner_generation_artifact,
)
from benchflow._utils.result_paths import load_task_results
from benchflow._utils.reward_events import memory_summary
from benchflow._utils.scoring import (
    ACP_ERROR,
    AGENT_INTEGRATION,
    API_ERROR,
    IDLE_TIMEOUT,
    INFRA_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    PROVIDER_AUTH,
    PROVIDER_RATE_LIMIT,
    PROVIDER_REJECTED,
    SANDBOX_SETUP,
    SUSPECTED_API_ERROR,
    USAGE_LIMIT,
    VERIFIER_DEP_INSTALL,
    VERIFIER_INFRA,
    VERIFIER_TIMEOUT,
    api_error_is_transient,
    classify_error,
    classify_score_outcome,
    classify_verifier_error,
    count_score_outcomes,
    is_unrecoverable_startup_error,
    mean_scored_reward,
    pass_rate,
    pass_rate_excl_errors,
    score_summary_fields,
)
from benchflow._utils.source_provenance import summary_source_fields
from benchflow._utils.text import truncate_end
from benchflow.agents.errors import UsageLimitError
from benchflow.budget import Budget, BudgetGuard
from benchflow.checkpoint_retry import retry_summary, run_checkpoint_retry
from benchflow.diagnostics import DIAGNOSTIC_REGISTRY, summary_warning
from benchflow.environment.manifest import EnvironmentManifest, load_manifest
from benchflow.errors import UserError
from benchflow.learner_store import LearnerState, LearnerStore
from benchflow.loop_strategies import (
    LoopStrategySpec,
    loop_block,
    parse_loop_strategy_spec,
)
from benchflow.models import RolloutResult
from benchflow.review.options import ReviewerConfig
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    normalize_skill_mode,
)
from benchflow.task.discovery import (
    is_task_dir as _is_structural_task_dir,
)
from benchflow.task.discovery import (
    resolve_task_collection_root,
)
from benchflow.trajectories.tree import RolloutNode
from benchflow.usage_tracking import UsageTrackingConfig

# Backward-compat alias
RunResult = RolloutResult

if TYPE_CHECKING:
    from benchflow.checkpoint_retry import RetryPolicy
    from benchflow.checkpoints import CheckpointPolicy
    from benchflow.review.outcome import ScoringResult

logger = logging.getLogger(__name__)

# Label applied to every container/network BenchFlow's compose files create.
# The leftover sweep lists only resources carrying it, so it never touches
# unrelated containers/networks on shared developer or CI hosts. Its value was
# "true" before the sweep replaced the daemon-wide prune; a new value keeps an
# older BenchFlow's prune on the same daemon off this version's containers.
BENCHFLOW_OWNED_LABEL = "benchflow.owned=process"

# Serialize the leftover sweep across concurrent _run_task retries. When
# --concurrency is high (e.g. 60) and tasks retry in lockstep, parallel sweeps
# each block on the daemon, cascading into false install_failure errors.
# Non-blocking acquire: if a sweep is already in flight, skip — there's nothing
# new to clean since the in-flight one started.
_PRUNE_LOCK = threading.Lock()


def _environment_manifest_from_task_document(
    task_dir: Path,
) -> EnvironmentManifest | None:
    task_md = task_dir / "task.md"
    if not task_md.is_file():
        return None

    from benchflow.task.document import TaskDocument

    document = TaskDocument.from_path(task_md)
    environment = document.benchflow.get("environment")
    if environment is None:
        return None
    if not isinstance(environment, dict):
        raise ValueError("task.md benchflow.environment must be a mapping")
    manifest = environment.get("manifest")
    if manifest is None:
        return None
    if not isinstance(manifest, str) or not manifest.strip():
        raise ValueError("task.md benchflow.environment.manifest must be a path")

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = task_dir / manifest_path
    return load_manifest(manifest_path)


def _is_task_dir(path: Path) -> bool:
    from benchflow._utils.task_authoring import task_symlink_issues

    if task_symlink_issues(path):
        return False
    if not (path / "task.md").exists():
        return _is_structural_task_dir(path) and _task_parse_error(path) is None
    from benchflow._utils.task_authoring import check_task

    return check_task(path) == []


def _task_parse_error(path: Path) -> tuple[Path, str] | None:
    """The task file in *path* that exists but fails to parse, and its error.

    ``task.md`` is checked when present, as ``Task`` loads it first; a legacy
    ``task.toml`` only without one.
    """
    from benchflow._utils.task_authoring import (
        task_config_parse_error,
        task_document_parse_error,
    )

    task_md = path / "task.md"
    if task_md.is_file():
        error = task_document_parse_error(task_md)
        return None if error is None else (task_md, error)
    task_toml = path / "task.toml"
    if task_toml.is_file():
        error = task_config_parse_error(task_toml)
        return None if error is None else (task_toml, error)
    return None


class EmptyTaskSelectionError(ValueError, UserError):
    """Raised when task discovery + include/exclude filters resolve to zero tasks.

    Failing fast is preferred over silently writing a 0/0 summary.json that
    downstream dashboards may ingest as evidence (#407).
    """


class ResumeMismatchError(ValueError, UserError):
    """Raised when resuming a jobs_dir whose completed tasks ran a different agent.

    A jobs_dir holds one (agent, model) run. Folding a *different* agent's cached
    rollouts into this run would publish a blended ``Score: X/N`` that belongs to
    neither agent (the symptom: scores appear for tasks this agent never ran).
    Refuse rather than warn-and-proceed — and rather than re-run over the prior
    rows, which would destroy the earlier agent's results. The fix is a fresh
    --jobs-dir, which also preserves the existing data.
    """


class MalformedTaskError(ValueError, UserError):
    """A single-task input whose ``task.md`` (or legacy ``task.toml``) exists
    but fails to parse (#3).

    Subclasses ``ValueError`` so the CLI's existing run-error handlers surface it
    as a clean red message + exit 1. The message names the offending file —
    silently treating a typo'd task.md as "not a task" would make the task vanish.
    """

    fault = "task"


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
    # Provider API errors: only TRANSIENT ones (rate limit, 5xx) are
    # retryable — auth/quota/model-not-found are permanent until a human
    # fixes the credential or model id, so retrying only burns wall-clock.
    retry_on_api_error: bool = True
    wait_multiplier: float = 2.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    exclude_categories: set[str] = field(
        default_factory=lambda: {
            "timeout",
            PROVIDER_AUTH,
            PROVIDER_RATE_LIMIT,
            PROVIDER_REJECTED,
        }
    )

    @classmethod
    def from_mapping(cls, raw: dict | None) -> RetryConfig:
        """Build from a serialized ``retry`` payload (e.g. a worker config).

        Any omitted field falls back to this dataclass's own default — never a
        hard-coded literal — so a partial/older payload that drops, say,
        ``exclude_categories`` still excludes ``provider_auth`` (#564 finding 2).
        """
        raw = raw or {}
        defaults = cls()
        exclude = raw.get("exclude_categories")
        return cls(
            max_retries=int(raw.get("max_retries", defaults.max_retries)),
            retry_on_install=bool(
                raw.get("retry_on_install", defaults.retry_on_install)
            ),
            retry_on_pipe=bool(raw.get("retry_on_pipe", defaults.retry_on_pipe)),
            retry_on_acp=bool(raw.get("retry_on_acp", defaults.retry_on_acp)),
            retry_on_idle_timeout=bool(
                raw.get("retry_on_idle_timeout", defaults.retry_on_idle_timeout)
            ),
            retry_on_infra=bool(raw.get("retry_on_infra", defaults.retry_on_infra)),
            retry_on_api_error=bool(
                raw.get("retry_on_api_error", defaults.retry_on_api_error)
            ),
            retry_on_verifier_infra=bool(
                raw.get("retry_on_verifier_infra", defaults.retry_on_verifier_infra)
            ),
            wait_multiplier=float(raw.get("wait_multiplier", defaults.wait_multiplier)),
            min_wait_sec=float(raw.get("min_wait_sec", defaults.min_wait_sec)),
            max_wait_sec=float(raw.get("max_wait_sec", defaults.max_wait_sec)),
            exclude_categories=(
                set(exclude) if exclude is not None else defaults.exclude_categories
            ),
        )

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
        if category == USAGE_LIMIT:
            # Whatever exclude_categories says: every retry on the same login
            # fails the same way until the window resets.
            return False
        if category in self.exclude_categories:
            return False
        if self.retry_on_install and category == INSTALL_FAILED:
            return True
        if self.retry_on_pipe and category == PIPE_CLOSED:
            return True
        if self.retry_on_idle_timeout and category == IDLE_TIMEOUT:
            return True
        if self.retry_on_infra and category in {INFRA_ERROR, SANDBOX_SETUP}:
            # A missing build-context path fails the same way every time.
            return not is_unrecoverable_startup_error(error)
        if category == API_ERROR:
            # Transient-only: rate limit / provider 5xx self-heal on backoff;
            # permanent (auth, quota, model_not_found, rejected_request) do not.
            return self.retry_on_api_error and api_error_is_transient(error)
        if category == SUSPECTED_API_ERROR:
            # Zero-signal verdicts have an unknown subcategory — never provably
            # transient, so never auto-retried (rerun is an operator action).
            return False
        return bool(self.retry_on_acp and category == ACP_ERROR)

    def reruns_unjudged_solver(
        self,
        scoring: ScoringResult | None,
        error: str | None,
        *,
        category: str | None = None,
    ) -> bool:
        """Whether a rubric trial's solver runs again despite its scoring block.

        A rubric trial commits a scoring block even when its solver failed on
        the sandbox or transport and nothing was judged: a scoring error with
        no verifier reward. That is a retryable infrastructure failure like
        any other (#1059), not a verdict that pins the trial.
        """
        return (
            scoring is not None
            and scoring.status == "error"
            and scoring.verifier_reward is None
            and self.should_retry(error, category=category)
        )

    def should_retry_verifier_error(self, verifier_error: str | None) -> bool:
        """Check if a verifier error is infrastructure-retryable."""
        from benchflow.rollout._verifier_recovery import PRESERVED_SOLVER

        if not self.retry_on_verifier_infra or (
            verifier_error and PRESERVED_SOLVER in verifier_error
        ):
            return False
        return classify_verifier_error(verifier_error) in {
            VERIFIER_INFRA,
            VERIFIER_TIMEOUT,
        }

    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff delay for retry attempt."""
        delay = self.min_wait_sec * (self.wait_multiplier**attempt)
        return min(delay, self.max_wait_sec)


class ApiErrorCircuitBreaker:
    """Trip after N consecutive permanent provider-API failures with the SAME
    fingerprint (classic dead key / wrong model id), so a doomed batch stops
    burning sandbox-hours producing all-unhealthy artifacts.

    Isolated api_errors never interrupt the batch — any completion that is not
    a permanent api_error resets the streak. Threshold comes from
    ``BENCHFLOW_API_ERROR_BREAKER_THRESHOLD`` (default 5; ``0`` disables).
    Already-running tasks finish; only not-yet-started tasks are skipped.
    """

    ENV_VAR = "BENCHFLOW_API_ERROR_BREAKER_THRESHOLD"
    DEFAULT_THRESHOLD = 5

    def __init__(self, threshold: int | None = None) -> None:
        if threshold is None:
            raw = os.environ.get(self.ENV_VAR, "")
            try:
                threshold = int(raw) if raw.strip() else self.DEFAULT_THRESHOLD
            except ValueError:
                threshold = self.DEFAULT_THRESHOLD
        self.threshold = max(threshold, 0)
        self._fingerprint: str | None = None
        self._streak = 0
        self.tripped = False

    @staticmethod
    def _fingerprint_of(result: RunResult) -> str | None:
        """Permanent-api-error fingerprint, or None when not breaker-relevant."""
        category = result.error_category or classify_error(result.error)
        if category == SUSPECTED_API_ERROR:
            return "suspected:zero_signal"
        if category == AGENT_INTEGRATION:
            # An auth or install failure repeats on every trial of the batch
            # (e.g. an exhausted credit balance); the other causes can be
            # transient and never trip the breaker.
            from benchflow.integration_health import PERMANENT_CAUSES

            match = re.match(
                r"agent integration failure \[([a-z_]+)\]", result.error or ""
            )
            cause = match.group(1) if match else None
            return f"integration:{cause}" if cause in PERMANENT_CAUSES else None
        if category == API_ERROR and not api_error_is_transient(result.error):
            match = re.search(r"\[([a-z_]+)/permanent\] HTTP (\d+)", result.error or "")
            return (
                f"{match.group(1)}:{match.group(2)}" if match else "api_error:unknown"
            )
        return None

    def record(self, result: RunResult) -> None:
        """Track one completed task; trip when the same-fingerprint streak hits
        the threshold."""
        if self.threshold == 0 or self.tripped:
            return
        fingerprint = self._fingerprint_of(result)
        if fingerprint is None:
            self._fingerprint = None
            self._streak = 0
            return
        if fingerprint == self._fingerprint:
            self._streak += 1
        else:
            self._fingerprint = fingerprint
            self._streak = 1
        if self._streak >= self.threshold:
            self.tripped = True
            logger.error(
                f"API-error circuit breaker OPEN: {self._streak} consecutive "
                f"permanent provider failures [{fingerprint}] — skipping "
                f"remaining unstarted tasks (set {self.ENV_VAR}=0 to disable)"
            )

    def skip_error(self) -> str:
        return (
            f"skipped: api-error circuit breaker open "
            f"([{self._fingerprint}] x{self._streak} consecutive)"
        )


class UsageLimitStop:
    """Stop starting trials once one ends on its login's usage limit.

    Every trial of an Evaluation runs on the same login, so once one reports
    the limit the rest would fail the same way until the window resets.
    Running trials finish; trials not yet started are left out of the job
    (not counted as results) for a resume on another login or after the
    reset. ``Evaluation.run`` raises the first ``UsageLimitError`` at the end.
    """

    def __init__(self) -> None:
        self.error: UsageLimitError | None = None
        self.not_started: list[str] = []

    def record(self, result: RunResult) -> None:
        if self.error is not None:
            return
        self.error = UsageLimitError.from_result(result)
        if self.error is not None:
            login = f"login {self.error.login}" if self.error.login else "the login"
            logger.error(
                f"Stopping the job: {login} is out of usage; running trials "
                "finish, no new ones start"
            )

    def skip(self, name: str) -> bool:
        """True (and recorded) when ``name`` must not start."""
        if self.error is None:
            return False
        self.not_started.append(name)
        return True

    def summary(self) -> dict[str, Any] | None:
        if self.error is None:
            return None
        return {**self.error.to_dict(), "not_started": list(self.not_started)}


# Defaults: works out-of-the-box with `claude login` (subscription auth, no API key needed)
DEFAULT_AGENT = "claude-agent-acp"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Job scheduling modes (architecture.md § "Lifecycles" — the Job lifecycle).
# - parallel-independent: the default — rollouts run concurrently, isolated.
# - sequential-shared: continual learning — rollouts run in order over one
#   persistent, versioned LearnerStore (capability 5).
JOB_MODES = ("parallel-independent", "sequential-shared")
DEFAULT_JOB_MODE = "parallel-independent"


def _check_resume_mismatch(job_dir: Path, config: EvaluationConfig) -> None:
    """Guard against resuming a jobs_dir whose completed tasks ran differently.

    Reads one completed rollout's config.json (written by SDK.run) and
    compares its agent and ``loop`` block against the resuming config.
    Pre-loop-strategy config.json files have no ``loop`` key — they ran
    single-shot, so they default to ``loop_block(None)`` and still warn
    when the resume requests a strategy.

    An *agent* mismatch raises :class:`ResumeMismatchError` (a blended score is
    meaningless and silently mixing one in is the bug this guards). A
    *loop_strategy* mismatch — same agent, different tuning — only warns.
    """
    sample_dir = (
        next((d for d in job_dir.iterdir() if d.is_dir()), None)
        if job_dir.exists()
        else None
    )
    prev_agent = ""
    prev_loop: dict | None = None
    if sample_dir:
        for cfg_file in sample_dir.rglob("config.json"):
            try:
                cfg = json.loads(cfg_file.read_text())
                prev_agent = cfg.get("agent", "")
                prev_loop = cfg.get("loop") or loop_block(None)
                break
            except (json.JSONDecodeError, OSError):
                logger.debug("Could not read %s", cfg_file)
    if prev_agent and prev_agent != config.agent:
        raise ResumeMismatchError(
            f"refusing to resume: this jobs_dir's completed tasks ran "
            f"agent={prev_agent!r}, but this run uses agent={config.agent!r}. "
            f"Mixing them would publish a blended score that belongs to neither. "
            f"Use a fresh --jobs-dir (the existing results are preserved)."
        )
    current_loop = loop_block(config.loop_strategy)
    if prev_loop is not None and prev_loop != current_loop:
        logger.warning(
            f"Resuming with loop_strategy={current_loop} but "
            f"completed tasks used loop_strategy={prev_loop}. "
            f"Use a different jobs_dir to avoid mixing results."
        )


def _scoring_block(result: dict[str, Any]) -> ScoringResult | None:
    """A persisted result's scoring block; None when absent or malformed."""
    from benchflow.review.outcome import scoring_from_result

    try:
        return scoring_from_result(result)
    except ValueError:
        return None


def _classify_completed_outcomes(
    completed: dict[str, dict],
) -> tuple[int, int, int]:
    """(passed, failed, errored) over already-complete (resumed) result payloads.

    Use the same explicit gate outcome as fresh results, retaining the legacy
    reward-equals-one rule only for results without integrated scoring.
    """
    passed = failed = errored = 0
    for r in completed.values():
        outcome = classify_score_outcome(r)
        if outcome == "passed":
            passed += 1
        elif outcome == "failed":
            failed += 1
        else:
            errored += 1
    return passed, failed, errored


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
    if agent in ("oracle", "nop"):
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
    reasoning_effort: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    build_concurrency: int | None = None
    prompts: list[str | None] | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    skills_dir: str | None = None
    codex_apps_policy: str | None = field(default=None, kw_only=True)
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    skip_agent_install: bool = False
    agent_idle_timeout: int | None = 600
    context_root: str | None = None
    base_image_override: str | None = None
    exclude_tasks: set[str] = field(default_factory=set)
    include_tasks: set[str] = field(default_factory=set)
    skill_mode: str = SKILL_MODE_NO_SKILL
    skill_creator_dir: str | None = None
    self_gen_no_internet: bool = False
    job_mode: str = DEFAULT_JOB_MODE
    source_provenance: dict[str, Any] | None = None
    # Registry dataset identity (`bench eval run -d name@version`). When
    # set, every result.json/config.json is stamped with dataset_name,
    # dataset_version, and the task's registry content digest — see
    # docs/dataset-versioning.md in benchflow-ai/skillsbench.
    dataset_name: str | None = None
    dataset_version: str | None = None
    dataset_task_digests: dict[str, str] = field(default_factory=dict)
    usage_tracking: UsageTrackingConfig = field(default_factory=UsageTrackingConfig)
    # Environment-plane manifest applied to every rollout in the batch.
    # When set, each task's RolloutConfig.environment_manifest is populated
    # so the Environment plane (manifest-declared stateful environment,
    # readiness gating, teardown) is exercised — closing the gap between
    # single-rollout SDK.run() and the batch Evaluation/Job API (#398).
    environment_manifest: EnvironmentManifest | None = None
    # C-axis overlay (parsed dict) deep-merged into each task's resolved config.
    config_override: dict | None = None
    # Harness loop strategy applied to every rollout (e.g.
    # "verify-retry:k=3,feedback=names"). Threaded to RolloutConfig.from_legacy
    # and stamped in summary.json; None = single-shot. A dict (the to_mapping()
    # shape) is also accepted at runtime — __post_init__ materializes it.
    loop_strategy: LoopStrategySpec | str | None = None
    # Opt-in automatic checkpoints (benchflow.checkpoints): "every-prompt" or
    # "prompt:N,M", and how many retained snapshots each trial keeps.
    checkpoints: str | None = None
    checkpoint_keep: int = 3
    # Freeze each trial's final workspace for later `bench eval regrade`.
    freeze_workspace: bool = False
    # Opt-in retry of a failed/timed-out trial from its last checkpoint
    # (benchflow.checkpoint_retry): "on-failure", "on-timeout" or both.
    retry_from_checkpoint: str | None = None
    retry_prompt: str | None = None
    retry_resume_session: bool = False
    # Hard per-job budget (benchflow.budget): stop launching and cancel
    # running trials once USD, sandbox-seconds or tokens reach a cap.
    budget: Budget | None = None

    def retry_policy(self) -> RetryPolicy | None:
        """The parsed retry policy, or None when not requested."""
        from benchflow.checkpoint_retry import parse_retry_policy

        if not self.retry_from_checkpoint:
            return None
        if not self.checkpoints:
            raise ValueError(
                "--retry-from-checkpoint needs --checkpoints: a trial can only be "
                "retried from a checkpoint it kept"
            )
        return parse_retry_policy(
            self.retry_from_checkpoint,
            prompt=self.retry_prompt,
            resume_session=self.retry_resume_session,
        )

    def checkpoint_policy(self) -> CheckpointPolicy | None:
        """The parsed checkpoint policy, or None when not requested."""
        from benchflow.checkpoints import parse_checkpoint_policy

        if not self.checkpoints:
            return None
        return parse_checkpoint_policy(self.checkpoints, keep=self.checkpoint_keep)

    def __post_init__(self):
        self.budget = Budget.coerce(self.budget)
        from benchflow._utils.config import (
            normalize_agent_idle_timeout,
            normalize_agent_name,
            normalize_reasoning_effort,
            normalize_sandbox_user,
        )
        from benchflow.agents.registry import AGENTS

        # normalize_agent_name delegates to registry.resolve_agent_key. Unknown
        # bare IDs fail before sandbox work; explicit commands remain supported.
        self.agent = normalize_agent_name(self.agent)
        self.reasoning_effort = normalize_reasoning_effort(self.reasoning_effort)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        if self.codex_apps_policy not in (None, "disabled", "inherit"):
            raise ValueError("codex_apps_policy must be disabled, inherit, or None")
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        self.checkpoint_policy()  # refuse a bad --checkpoints before any run
        self.retry_policy()  # and a bad --retry-from-checkpoint
        self.usage_tracking = UsageTrackingConfig.coerce(self.usage_tracking)
        self.reviewer = ReviewerConfig.coerce(self.reviewer)
        self.skill_mode = normalize_skill_mode(self.skill_mode)
        if isinstance(self.loop_strategy, str):
            self.loop_strategy = parse_loop_strategy_spec(self.loop_strategy)
        elif isinstance(self.loop_strategy, dict):
            # The to_mapping() dict shape (e.g. a stamped spec round-tripped back
            # through a --config YAML, or an SDK EvaluationConfig(loop_strategy={...}))
            # must materialize too — not silently fall through and mislabel the run
            # single-shot. Mirror the sharding guard's loud-failure stance.
            # cast: isinstance narrows to dict[Unknown, Unknown]; from_mapping
            # validates the keys at runtime.
            self.loop_strategy = LoopStrategySpec.from_mapping(
                cast("dict[str, Any]", self.loop_strategy)
            )
        elif self.loop_strategy is not None and not isinstance(
            self.loop_strategy, LoopStrategySpec
        ):
            raise ValueError(
                "loop_strategy must be a spec string, mapping, or LoopStrategySpec, "
                f"got {type(self.loop_strategy).__name__}"
            )
        if self.skills_dir is not None and self.skill_mode != SKILL_MODE_WITH_SKILL:
            raise ValueError("skills_dir requires skill_mode='with-skill'")
        if self.job_mode not in JOB_MODES:
            raise ValueError(
                f"unknown job_mode {self.job_mode!r} — "
                f"expected one of {', '.join(JOB_MODES)}"
            )
        if self.agent not in ("oracle", "nop") and self.agent not in AGENTS:
            available = ", ".join(sorted(AGENTS.keys()))
            logger.warning(
                f"Unknown agent {self.agent!r} — not in registry. "
                f"Available: {available}. Will attempt to use as raw command."
            )


@dataclass(frozen=True)
class TaskFailure:
    """Cheap failure evidence for one task with a failed scoring outcome.

    Carried on :class:`EvaluationResult` so the CLI's final block can print a
    one-line reason per failed task from data the engine already holds —
    without re-reading result.json files. Errored tasks are excluded (they
    already surface through the error counters and warning replay).
    """

    task_name: str
    rewards: dict[str, Any] | None
    verifier_error: str | None
    # The task's rollout dir name under the job dir (``<task>__<uuid8>``), so
    # the CLI can find the rollout's verifier artifacts without guessing.
    # None on results persisted before the key existed.
    rollout_name: str | None = None


@dataclass
class EvaluationResult:
    """Aggregated results for a job.

    ``results`` maps each task name to its :class:`RolloutResult` (tasks
    reused on resume are read back from their ``result.json``), and
    ``job_dir`` is where the job's artifacts and ``summary.json`` live.
    """

    job_name: str
    config: EvaluationConfig = field(repr=False)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    verifier_errored: int = 0
    elapsed_sec: float = 0.0
    memory_score: float | None = None
    memory_scores: dict[str, float] = field(default_factory=dict)
    task_failures: list[TaskFailure] = field(default_factory=list)
    # Mean of rewards over scored rollouts (None when nothing scored). The
    # pass/fail counts describe hard gates, while quality retains partial credit —
    # a 0.3 rubric score and a flat 0 both print as FAIL without this.
    mean_reward: float | None = None
    # The job's artifact directory (``jobs_dir/job_name``).
    job_dir: Path | None = None
    # One typed result per task name: this run's results as returned by the
    # rollouts, plus resumed tasks read back from their result.json.
    results: dict[str, RolloutResult] = field(default_factory=dict)
    # summary.json's ``budget`` block when the job had a Budget (caps, spent,
    # stopped, reason, cancelled and not-started trials), else None.
    budget: dict[str, Any] | None = None
    # Tasks reused from an earlier run of the same job (resume) and tasks that
    # ran now; ran == 0 with reused > 0 means the results are all earlier ones.
    reused: int = 0
    ran: int = 0

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per task, sorted by task name (``RolloutResult.to_record``)."""
        return [result.to_record() for _, result in sorted(self.results.items())]

    def to_csv(self, path: str | Path) -> Path:
        """Write one CSV row per task and return the path."""
        from benchflow.batch import write_csv

        return write_csv((r for _, r in sorted(self.results.items())), path)

    def to_jsonl(self, path: str | Path) -> Path:
        """Write one JSON line per task and return the path."""
        from benchflow.batch import write_jsonl

        return write_jsonl((r for _, r in sorted(self.results.items())), path)

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)


EVALUATION_RECORD = "evaluation.json"
JOB_LOCK = ".evaluation.lock"


def _pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OverflowError:
        return False
    return True


def _config_to_record(config: EvaluationConfig) -> dict[str, Any]:
    """Serialize a config for ``evaluation.json``; agent_env values are left out."""
    import dataclasses

    record: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.name == "agent_env":
            record["agent_env_keys"] = sorted(value)
        elif f.name == "retry":
            retry = dataclasses.asdict(value)
            retry["exclude_categories"] = sorted(retry["exclude_categories"])
            record["retry"] = retry
        elif f.name == "reviewer":
            record["reviewer"] = value.to_config_artifact()
        elif f.name == "usage_tracking":
            record["usage_tracking"] = value.to_mapping()
        elif f.name == "environment_manifest":
            record[f.name] = None if value is None else value.model_dump(mode="json")
        elif f.name == "loop_strategy":
            record[f.name] = None if value is None else value.to_mapping()
        elif f.name == "budget":
            record[f.name] = None if value is None else value.to_dict()
        elif isinstance(value, set):
            record[f.name] = sorted(value)
        else:
            record[f.name] = value
    return json.loads(json.dumps(record, default=str))


def _config_from_record(
    raw: dict[str, Any], agent_env: dict[str, str] | None
) -> EvaluationConfig:
    """Rebuild an EvaluationConfig from ``_config_to_record`` output."""
    import dataclasses

    known = {f.name for f in dataclasses.fields(EvaluationConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    missing = sorted(set(raw.get("agent_env_keys") or []) - set(agent_env or {}))
    if missing:
        logger.warning(
            "Resuming without agent_env %s: their values are not stored in %s. "
            "Pass agent_env={...} to Evaluation.resume() if the agent needs them.",
            ", ".join(missing),
            EVALUATION_RECORD,
        )
    kwargs["agent_env"] = dict(agent_env or {})
    kwargs["retry"] = RetryConfig.from_mapping(raw.get("retry"))
    reviewer = dict(raw.get("reviewer") or {})
    reviewer.pop("agent_env_keys", None)
    kwargs["reviewer"] = ReviewerConfig.coerce(reviewer)
    if raw.get("environment_manifest") is not None:
        kwargs["environment_manifest"] = EnvironmentManifest.model_validate(
            raw["environment_manifest"]
        )
    for name in ("exclude_tasks", "include_tasks"):
        kwargs[name] = set(raw.get(name) or [])
    return EvaluationConfig(**kwargs)


class Evaluation:
    """Run a benchmark job across multiple tasks.

    Usage:
        from benchflow import resolve_source

        evaluation = Evaluation(
            tasks_dir=resolve_source("harbor-framework/terminal-bench-2"),
            jobs_dir="parity/tb2-haiku",
            config=EvaluationConfig(model="claude-haiku-4-5-20251001"),
        )
        result = await evaluation.run()
        print(result.score)

    Or from YAML:
        evaluation = Evaluation.from_yaml("tb2.yaml")
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
                logger.warning(
                    f"Resuming into existing job directory: {job_dirs[0].name} "
                    "(finished tasks are reused; pass a new job_name, or "
                    "--fresh on the CLI, for a new run)"
                )
                return job_dirs[0].name
            if len(job_dirs) > 1:
                latest = job_dirs[-1]
                logger.warning(
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
        on_task_start: Callable[[str], None] | None = None,
        on_plan: Callable[[int, int, int, tuple[int, int, int]], None] | None = None,
        preflight: bool = True,
        budget: Budget | dict[str, Any] | None = None,
    ):
        self._tasks_dir = resolve_task_collection_root(tasks_dir)
        self._jobs_dir = Path(jobs_dir)
        self._config = config or EvaluationConfig()
        if budget is not None:
            # A hard per-job cap (benchflow.budget); same as config.budget.
            self._config.budget = Budget.coerce(budget)
        self._budget_guard: BudgetGuard | None = None
        self._usage_stop = UsageLimitStop()
        # agent_env names a loaded config declared without values; to_dict
        # keeps listing them so a second save does not forget them.
        self._declared_env_keys: list[str] = []
        if self._config.source_provenance is None:
            from benchflow._utils.hf_datasets import load_source_sidecar

            self._config.source_provenance = load_source_sidecar(self._tasks_dir)
        self._job_name = job_name or self._resolve_job_name(self._jobs_dir)
        self._on_result = on_result
        # Pre-run checks in run(); the CLI passes False (it runs its own).
        self._preflight = preflight
        # The last run's EvaluationResult (set by run(), stream(), run_sync()).
        self.result: EvaluationResult | None = None
        # UI-progress hooks (the CLI live dashboard; None everywhere else). Fired
        # best-effort via _fire_progress so a display bug never aborts a run.
        self._on_task_start = on_task_start
        self._on_plan = on_plan
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

    def to_dict(self, *, include_agent_env: bool = False) -> dict[str, Any]:
        """This job as the native config mapping that ``from_yaml`` and
        ``bench eval run --config`` read.

        agent_env values (the agent's and the reviewer's) are left out unless
        ``include_agent_env=True``; their key names are listed under
        ``agent_env_keys`` so a reader knows what to supply. The job name is
        not written, so each run of the saved config starts a new job.
        """
        cfg = self._config
        record = _config_to_record(cfg)
        reviewer = cfg.reviewer.to_dict()
        if not include_agent_env:
            reviewer["agent_env"] = {}
        out: dict[str, Any] = {
            "tasks_dir": str(self._tasks_dir),
            "jobs_dir": str(self._jobs_dir),
            "agent": cfg.agent,
            "model": cfg.model,
            "reasoning_effort": cfg.reasoning_effort,
            "environment": cfg.environment,
            "concurrency": cfg.concurrency,
            "build_concurrency": cfg.build_concurrency,
            "prompts": cfg.prompts,
            "agent_env": dict(cfg.agent_env) if include_agent_env else {},
            "agent_env_keys": sorted(set(cfg.agent_env) | set(self._declared_env_keys)),
            "retry": record["retry"],
            "reviewer": reviewer,
            "skills_dir": cfg.skills_dir,
            "codex_apps_policy": cfg.codex_apps_policy,
            "sandbox_user": cfg.sandbox_user,
            "sandbox_locked_paths": cfg.sandbox_locked_paths,
            "sandbox_setup_timeout": cfg.sandbox_setup_timeout,
            "skip_install": cfg.skip_agent_install,
            "agent_idle_timeout_sec": cfg.agent_idle_timeout,
            "context_root": cfg.context_root,
            "base_image_override": cfg.base_image_override,
            "include": sorted(cfg.include_tasks),
            "exclude": sorted(cfg.exclude_tasks),
            "skill_mode": cfg.skill_mode,
            "skill_creator_dir": cfg.skill_creator_dir,
            "self_gen_no_internet": cfg.self_gen_no_internet,
            "job_mode": cfg.job_mode,
            **cfg.usage_tracking.to_mapping(),
            "environment_manifest": record["environment_manifest"],
            "config_override": cfg.config_override,
            "loop_strategy": record["loop_strategy"],
            "checkpoints": cfg.checkpoints,
            "checkpoint_keep": cfg.checkpoint_keep,
            "freeze_workspace": cfg.freeze_workspace,
            "retry_from_checkpoint": cfg.retry_from_checkpoint,
            "retry_prompt": cfg.retry_prompt,
            "retry_resume_session": cfg.retry_resume_session,
            "budget": None if cfg.budget is None else cfg.budget.to_dict(),
            "source_provenance": cfg.source_provenance,
            "dataset_name": cfg.dataset_name,
            "dataset_version": cfg.dataset_version,
            "dataset_task_digests": dict(cfg.dataset_task_digests),
        }
        return json.loads(json.dumps(out, default=str))

    def to_yaml(self, path: str | Path, *, include_agent_env: bool = False) -> Path:
        """Write :meth:`to_dict` as YAML (``bench eval run --config`` reads it)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                self.to_dict(include_agent_env=include_agent_env), sort_keys=False
            )
        )
        return path

    @classmethod
    def from_dict(cls, raw: dict[str, Any], **kwargs: Any) -> Evaluation:
        """Build an Evaluation from a native config mapping (``to_dict``'s shape)."""
        return cls._from_native_yaml(dict(raw), **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> Evaluation:
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

        # A non-mapping document (empty file → None, or a top-level list/scalar)
        # is not a valid config. Reject it up front: the membership tests below
        # would otherwise TypeError on None, or silently substring-match a scalar
        # string that happens to contain "agents"/"datasets" and mis-route to
        # the legacy parser.
        if not isinstance(raw, dict):
            raise ValueError(
                f"Eval config {path} must be a YAML mapping with 'source' or "
                f"'tasks_dir' (or legacy 'agents'/'datasets'); got "
                f"{type(raw).__name__}."
            )

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
        from benchflow.adapters.source import adapt_resolved_source_if_needed

        # New two-field format: source.repo + source.path
        source_provenance = None
        if "source" in raw:
            src = raw["source"]
            if not isinstance(src, dict):
                raise ValueError(
                    f"YAML 'source' must be a mapping with a 'repo' key; got "
                    f"{type(src).__name__}."
                )
            repo = src.get("repo")
            if not isinstance(repo, str) or not repo:
                raise ValueError(
                    "YAML 'source.repo' must be a non-empty string (e.g. 'org/repo')."
                )
            resolved = resolve_source_with_metadata(
                repo=repo,
                path=src.get("path"),
                ref=src.get("ref"),
            )
            resolved = adapt_resolved_source_if_needed(resolved)
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

        # Parse prompts — YAML null becomes Python None. A bare string must be
        # wrapped: otherwise Scene.single iterates it character-by-character into
        # one garbage turn per character (mirrors the SDK YAML loader).
        raw_prompts = raw.get("prompts")
        prompts: list[str | None] | None = (
            [raw_prompts] if isinstance(raw_prompts, str) else raw_prompts
        )

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
        if isinstance(env_manifest_raw, dict):
            # Inline manifest, as Evaluation.to_dict() writes it.
            env_manifest = EnvironmentManifest.model_validate(env_manifest_raw)
        elif env_manifest_raw is not None:
            env_manifest = load_manifest(env_manifest_raw)
        missing_env = sorted(set(raw.get("agent_env_keys") or []) - set(agent_env_raw))
        if missing_env:
            logger.warning(
                "The config names agent_env %s without values; pass them in "
                "agent_env (or the environment) if the agent needs them.",
                ", ".join(missing_env),
            )
        config = EvaluationConfig(
            agent=agent_name,
            model=effective_model(agent_name, raw.get("model")),
            reasoning_effort=raw.get("reasoning_effort"),
            environment=raw.get("environment", "docker"),
            concurrency=raw.get("concurrency", 4),
            build_concurrency=raw.get("build_concurrency"),
            prompts=prompts,
            agent_env=agent_env_raw,
            reviewer=ReviewerConfig.coerce(raw.get("reviewer")),
            retry=RetryConfig.from_mapping(raw["retry"])
            if isinstance(raw.get("retry"), dict)
            else RetryConfig(max_retries=raw.get("max_retries", 2)),
            skills_dir=str(Path(raw["skills_dir"])) if raw.get("skills_dir") else None,
            codex_apps_policy=raw.get("codex_apps_policy"),
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            skip_agent_install=bool(raw.get("skip_install", False)),
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
            context_root=raw.get("context_root"),
            base_image_override=raw.get("base_image_override"),
            exclude_tasks=exclude,
            include_tasks=include,
            skill_mode=raw.get("skill_mode", SKILL_MODE_NO_SKILL),
            skill_creator_dir=(
                str(Path(raw["skill_creator_dir"]))
                if raw.get("skill_creator_dir")
                else None
            ),
            self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
            job_mode=raw.get("job_mode", DEFAULT_JOB_MODE),
            source_provenance=source_provenance or raw.get("source_provenance"),
            dataset_name=raw.get("dataset_name"),
            dataset_version=raw.get("dataset_version"),
            dataset_task_digests=raw.get("dataset_task_digests") or {},
            usage_tracking=UsageTrackingConfig.from_mapping(raw),
            environment_manifest=env_manifest,
            config_override=raw.get("config_override"),
            loop_strategy=raw.get("loop_strategy"),
            checkpoints=raw.get("checkpoints"),
            checkpoint_keep=raw.get("checkpoint_keep", 3),
            freeze_workspace=bool(raw.get("freeze_workspace", False)),
            retry_from_checkpoint=raw.get("retry_from_checkpoint"),
            retry_prompt=raw.get("retry_prompt"),
            retry_resume_session=bool(raw.get("retry_resume_session", False)),
            budget=Budget.coerce(raw.get("budget")),
        )
        evaluation = cls(
            tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs
        )
        declared = raw.get("agent_env_keys")
        if isinstance(declared, list):
            evaluation._declared_env_keys = sorted(
                {k for k in declared if isinstance(k, str)}
            )
        return evaluation

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
            reasoning_effort=agent_cfg.get(
                "reasoning_effort", raw.get("reasoning_effort")
            ),
            environment=environment,
            concurrency=concurrency,
            agent_env=agent_env,
            reviewer=ReviewerConfig.coerce(raw.get("reviewer")),
            retry=RetryConfig(max_retries=max(0, max_retries)),
            skills_dir=skills_dir,
            codex_apps_policy=raw.get("codex_apps_policy"),
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            skip_agent_install=bool(agent_cfg.get("skip_install", False)),
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
            context_root=raw.get("context_root"),
            base_image_override=raw.get("base_image_override"),
            include_tasks=include,
            exclude_tasks=exclude,
            skill_mode=raw.get("skill_mode", SKILL_MODE_NO_SKILL),
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
        """Get all valid task directories.

        A directory whose ``task.md`` (or, without one, legacy ``task.toml``)
        *exists but fails to parse* is a malformed task, not a non-task: in the
        single-task case that is a hard error (the user named exactly one thing
        and it is broken); in the batch case it is loudly warned and skipped (a
        typo must never make a task silently vanish from a 50-task suite, #3)
        while the healthy tasks still run. A task.md
        that PARSES but is structurally incomplete (e.g. a schema-only fixture)
        keeps its existing silent skip.
        """
        # A valid task at the root → that IS the whole job (single-task input).
        if _is_task_dir(self._tasks_dir):
            if self._tasks_dir.name in self._config.exclude_tasks:
                return []
            if (
                self._config.include_tasks
                and self._tasks_dir.name not in self._config.include_tasks
            ):
                return []
            return [self._tasks_dir]

        from benchflow._utils.task_authoring import task_symlink_issues

        # Batch input: collect valid child tasks; warn (don't silently drop) on
        # any child whose task.md fails to PARSE. Selection filters are applied
        # first, so excluded dirs are never warned about. A child task.md that
        # PARSES but is structurally incomplete (schema-only fixture) keeps its
        # silent skip.
        selected: list[Path] = []
        for d in sorted(self._tasks_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name in self._config.exclude_tasks:
                continue
            if self._config.include_tasks and d.name not in self._config.include_tasks:
                continue
            if _is_task_dir(d):
                selected.append(d)
                continue
            malformed = _task_parse_error(d)
            if malformed is not None:
                logger.warning("Skipping malformed task %r: %s", d.name, malformed[1])
                continue
            linked = task_symlink_issues(d)
            if linked and ((d / "task.md").is_file() or (d / "task.toml").is_file()):
                # A task that would run without its linked files must not
                # vanish silently from a batch.
                logger.warning("Skipping task %r: %s", d.name, linked[0])

        # A malformed task file at the tasks-dir ROOT is a hard error ONLY when
        # no valid child tasks were found — i.e. the root was meant as a single
        # task and it is broken. If the dir is a batch container that also
        # happens to carry a stray broken root task file, warn but still run the
        # healthy children rather than aborting the whole batch.
        malformed = _task_parse_error(self._tasks_dir)
        if malformed is not None:
            task_file, parse_error = malformed
            if selected:
                logger.warning(
                    "Ignoring malformed %s at the tasks-dir root %r: %s",
                    task_file.name,
                    self._tasks_dir.name,
                    parse_error,
                )
            else:
                raise MalformedTaskError(f"{task_file}: {parse_error}")
        return selected

    def _empty_selection_message(self) -> str:
        """Why no task was selected, naming structural problems when no filter did it.

        Discovery skips a task file that parses but fails ``bench tasks
        check`` (a schema-only fixture, a task with no environment/), so
        without filters the only visible symptom would be an empty selection.
        """
        cfg = self._config
        root = self._tasks_dir
        detail_parts = [f"tasks_dir={root}"]
        if cfg.include_tasks:
            detail_parts.append(f"include={sorted(cfg.include_tasks)}")
        if cfg.exclude_tasks:
            detail_parts.append(f"exclude={sorted(cfg.exclude_tasks)}")
        tail = " Refusing to publish an empty 0/0 summary."
        if cfg.include_tasks or cfg.exclude_tasks:
            return (
                "No tasks selected after include/exclude filtering "
                f"({', '.join(detail_parts)})." + tail
            )
        from benchflow._utils.task_authoring import check_task

        def _has_task_file(path: Path) -> bool:
            return (path / "task.md").is_file() or (path / "task.toml").is_file()

        if _has_task_file(root):
            issues = check_task(root)
            if issues:
                return (
                    f"No tasks selected: {root} is not a runnable task: "
                    f"{'; '.join(issues)}. Run `bench tasks check {root}` "
                    "for details." + tail
                )
        unrunnable = (
            sorted(d for d in root.iterdir() if d.is_dir() and _has_task_file(d))
            if root.is_dir()
            else []
        )
        if unrunnable:
            first = unrunnable[0]
            issues = check_task(first)
            example = f" (first: {first.name}: {issues[0]})" if issues else ""
            noun = (
                "1 subdirectory has a task file but is not a runnable task"
                if len(unrunnable) == 1
                else f"{len(unrunnable)} subdirectories have task files but "
                "are not runnable tasks"
            )
            return (
                f"No tasks selected in {root}: {noun}{example}. Run "
                f"`bench tasks check <task>` for details." + tail
            )
        return f"No tasks selected: no task found in {root}." + tail

    def _get_completed_tasks(self) -> dict[str, dict]:
        """Load tasks that already have results with rewards or verifier errors.

        Scoreless results whose verifier error is infra-retryable (per
        ``RetryConfig.should_retry_verifier_error``) are not reused: the
        verifier never scored the frozen workspace, so the task re-runs.

        Scoped to the current job directory (``_jobs_dir / _job_name``) to
        prevent cross-job contamination. When multiple result.json files exist
        for the same task (retry artifacts), a scored result always wins over
        a scoreless verifier error; otherwise the newest artifact wins.

        Guards ENG-160: orphan retry artifacts no longer pollute resume.
        """
        job_dir = self._jobs_dir / self._job_name
        if not job_dir.exists():
            return {}
        latest = load_task_results(job_dir)
        # A process may have died after the solver completed but before
        # capture/cleanup produced a terminal result. Preserve the stage
        # checkpoint without pretending it is a completed scoring verdict.
        for checkpoint in job_dir.glob("*/solver-complete.json"):
            if (checkpoint.parent / "result.json").exists():
                continue
            pending = json.loads(checkpoint.read_text())
            name = pending.get("task_name")
            if name and name not in latest and pending.get("purpose", "task") == "task":
                latest[name] = pending
        # Likewise a solver.json without result.json: a reviewed trial whose
        # scoring never committed and that resume_pending_reviews above could
        # not finish. Its solver is kept, never replayed, and it stays
        # unscored until `bench eval score` commits its review. solver.json's
        # own reward is the unreviewed verifier reward, so it is withheld.
        from benchflow.rollout._verifier_recovery import PRESERVED_SOLVER

        # Re-running an errored task is only safe when rollouts are
        # independent. A sequential-shared job advances one persisted learner
        # state in task order, so replaying an earlier task after later tasks
        # committed their skills would corrupt the learning curve; there the
        # errored result stays reused, matching the pre-existing behavior.
        rerun_ok = self._config.job_mode != "sequential-shared"
        for snapshot in job_dir.glob("*/solver.json"):
            if (snapshot.parent / "result.json").exists():
                continue
            pending = json.loads(snapshot.read_text())
            name = pending.get("task_name")
            if name and name not in latest and pending.get("purpose", "task") == "task":
                if (
                    rerun_ok
                    and pending.get("rewards") is None
                    and self._config.retry.should_retry(
                        pending.get("error"), category=pending.get("error_category")
                    )
                ):
                    # Its solver failed on infrastructure: nothing to review.
                    continue
                latest[name] = {
                    **pending,
                    "rewards": None,
                    "verifier_error": (
                        f"{PRESERVED_SOLVER} scoring did not commit; "
                        "finish it with bench eval score"
                    ),
                    "verifier_error_category": VERIFIER_INFRA,
                }
        completed: dict[str, dict] = {}
        for task, r in latest.items():
            # A durable solver snapshot makes rubric retries independent of the
            # solver. Even an unsuccessful retry must never replay that solver,
            # unless the solver failed on infrastructure and nothing was judged.
            if r.get("scoring") is not None:
                if rerun_ok and self._config.retry.reruns_unjudged_solver(
                    _scoring_block(r), r.get("error"), category=r.get("error_category")
                ):
                    logger.info(
                        f"Re-running task whose solver failed on infrastructure "
                        f"before anything was judged: {task} "
                        f"({truncate_end(r.get('error') or '', 80)})"
                    )
                    continue
                completed[task] = r
                continue
            if r.get("verifier_error"):
                # A scoreless result whose verifier error is infra-retryable
                # (same taxonomy as the within-run retry) records no signal
                # about the task; reusing it pins a lost score forever.
                retryable = self._config.retry.should_retry_verifier_error(
                    r["verifier_error"]
                )
                if rerun_ok and r.get("rewards") is None and retryable:
                    logger.info(
                        f"Re-running verifier-errored task on resume: {task} "
                        f"({truncate_end(r['verifier_error'], 80)})"
                    )
                    continue
                logger.info(
                    f"Reusing completed verifier-errored task on resume: {task} "
                    f"({truncate_end(r['verifier_error'], 80)})"
                )
            completed[task] = r
        return completed

    def _prune_docker(self):
        """Remove leftover Docker containers and networks of finished rollouts.

        Only resources labelled ``benchflow.owned=process`` (applied in
        ``sandbox/_compose_files/docker-compose-base.yaml``) are listed, so
        unrelated Docker workloads on the same host are left untouched. Of
        those, only the ones whose BenchFlow process is gone, or whose sandbox
        in this process is torn down, are removed
        (:func:`benchflow.sandbox._docker_sweep.sweep_leftovers`): a daemon-wide
        ``docker container prune`` / ``docker network prune`` deleted other
        live rollouts' just-created containers and networks, whether they
        belonged to this job, another job in this process, or another process.

        Serialized via ``_PRUNE_LOCK``: parallel retries from high-concurrency
        batches would otherwise each kick off docker CLI calls, all blocking
        on the same daemon. Non-blocking acquire — if another sweep is in
        flight we just skip, since it will catch the same garbage. Blocking;
        async callers run it in a thread (:meth:`_sweep_docker`).
        """
        if self._config.environment != "docker":
            return
        if not _PRUNE_LOCK.acquire(blocking=False):
            return
        try:
            from benchflow.sandbox._docker_sweep import sweep_leftovers

            sweep_leftovers()
        except Exception as e:
            logger.warning(f"Docker leftover sweep failed: {e}")
        finally:
            _PRUNE_LOCK.release()

    async def _sweep_docker(self) -> None:
        """:meth:`_prune_docker` off the event loop, so live rollouts keep
        streaming while the docker CLI calls wait on the daemon."""
        if self._config.environment != "docker":
            return
        await asyncio.to_thread(self._prune_docker)

    def _enrich_payload_with_persisted_timing(
        self, payload: dict, result: RolloutResult
    ) -> None:
        """Copy ``timing`` (and the integration-failure record) from the
        rollout's on-disk result.json into payload.

        ``RolloutResult`` does not carry phase timing, but the rollout writer
        (``rollout.py``) persists it under ``rollout_dir/result.json``. Reading
        it back lets ``phase_timing_summary`` aggregate phase totals for fresh
        runs (issue #501). Best-effort: a result with no rollout_name or no
        persisted result.json leaves timing absent rather than crash summary
        generation.
        """
        if "timing" in payload and "integration_failure_info" in payload:
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
        if isinstance(timing, dict) and "timing" not in payload:
            payload["timing"] = timing
        # The integration-failure record (its cause) exists only in the
        # persisted result; summary.json counts integration failures by it.
        integration = persisted.get("integration_failure_info")
        if isinstance(integration, dict):
            payload.setdefault("integration_failure_info", integration)

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

        dataset = None
        if cfg.dataset_name:
            dataset = {"name": cfg.dataset_name, "version": cfg.dataset_version}
        task_digest_value = (
            cfg.dataset_task_digests.get(task_dir.name) if cfg.dataset_name else None
        )
        if task_digest_value is None:
            # Dev runs (--tasks-dir / --source-repo) stamp a live-computed
            # digest so every trajectory stays attributable to the exact
            # task content it ran, not just a directory name.
            from benchflow._utils.task_authoring import task_digest

            try:
                task_digest_value = task_digest(task_dir)
            except (OSError, ValueError, UnicodeError) as e:
                logger.debug("Could not compute task digest for %s: %s", task_dir, e)
        skills_dir = (
            str(self._learner_skills_dir)
            if self._learner_skills_dir is not None
            else cfg.skills_dir
        )
        skill_mode = (
            SKILL_MODE_WITH_SKILL
            if self._learner_skills_dir is not None
            else cfg.skill_mode
        )
        export_to = (
            str(self._learner_export_dir)
            if self._learner_export_dir is not None
            else None
        )
        environment_manifest = cfg.environment_manifest
        if environment_manifest is None:
            environment_manifest = _environment_manifest_from_task_document(task_dir)
        rollout_config = RolloutConfig.from_legacy(
            task_path=task_dir,
            agent=cfg.agent,
            model=cfg.model,
            reasoning_effort=cfg.reasoning_effort,
            prompts=cfg.prompts,
            agent_env=cfg.agent_env,
            reviewer=cfg.reviewer,
            job_name=self._job_name,
            jobs_dir=str(self._jobs_dir),
            concurrency=cfg.concurrency,
            environment=cfg.environment,
            environment_manifest=environment_manifest,
            config_override=cfg.config_override,
            skills_dir=skills_dir,
            codex_apps_policy=cfg.codex_apps_policy,
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            skip_agent_install=cfg.skip_agent_install,
            agent_idle_timeout=cfg.agent_idle_timeout,
            context_root=cfg.context_root,
            base_image_override=cfg.base_image_override,
            skill_mode=skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
            export_generated_skills_to=export_to,
            source_provenance=task_source_provenance(cfg.source_provenance, task_dir),
            dataset=dataset,
            task_digest=task_digest_value,
            usage_tracking=cfg.usage_tracking,
            loop_strategy=cfg.loop_strategy,
        )
        rollout_config.checkpoints = cfg.checkpoint_policy()
        rollout_config.freeze_workspace = cfg.freeze_workspace
        if skill_mode == SKILL_MODE_SELF_GEN:
            from benchflow.self_gen import run_self_gen

            return await run_self_gen(rollout_config)
        rollout = await Rollout.create(rollout_config)
        # Expose the live rollout to the eval dashboard's activity cell —
        # a same-process poll of the session's heartbeat counters, see
        # benchflow._utils.live_activity.
        from benchflow._utils import live_activity

        live_activity.register(task_dir.name, rollout)
        try:
            # Rollout.run() enforces its own host-side hard deadline against
            # awaits wedged below the phase-level timeouts — see
            # benchflow.rollout._deadline. A trip surfaces here as a normal
            # infra-retryable error result.
            result = await rollout.run()
            policy = cfg.retry_policy()
            if policy is not None:
                try:
                    await run_checkpoint_retry(rollout, result, policy)
                except Exception:
                    # A retry never changes or loses the trial's own result.
                    logger.warning(
                        "Checkpoint retry of %s failed", task_dir.name, exc_info=True
                    )
            return result
        finally:
            live_activity.unregister(task_dir.name)

    async def _run_task(self, task_dir: Path) -> RunResult:
        """Run a single task with retries."""
        cfg = self._config
        last_result: RunResult | None = None

        for attempt in range(1, cfg.retry.max_retries + 2):
            if attempt > 1:
                delay = cfg.retry.backoff_delay(attempt - 1)
                logger.info(f"Retry backoff: {delay:.1f}s before attempt {attempt}")
                await asyncio.sleep(delay)
                await self._sweep_docker()
            result = await self._run_single_task(task_dir, cfg)
            last_result = result
            if result.scoring is not None and not cfg.retry.reruns_unjudged_solver(
                result.scoring, result.error, category=result.error_category
            ):
                # Once solver evidence is committed, only the scoring stage may
                # be retried. Replaying the solver would change the trial. A
                # solver that failed on infrastructure left nothing judged.
                return result

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
                err_preview = truncate_end(
                    result.error or result.verifier_error or "", 60
                )
                logger.info(
                    f"Retrying {task_dir.name} (attempt {attempt + 1}): {err_preview}"
                )

        # The loop always runs at least once (range(1, max_retries + 2)
        # has min 1 iter), so last_result is guaranteed set.
        assert last_result is not None
        return last_result

    @staticmethod
    def _fire_progress(callback, *args) -> None:
        """Invoke a UI-progress hook, swallowing any error.

        A live-display callback must never abort or perturb the run, so failures
        are logged at debug and ignored.
        """
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            # Display is best-effort: a render bug must never abort the run.
            logger.debug("progress callback failed: %s", exc)

    def _log_and_report(self, td: Path, result: RunResult) -> None:
        """Log one rollout's outcome and fire the on_result callback."""
        reward = result.rewards.get("reward") if result.rewards else None
        status = {"passed": "PASS", "failed": "FAIL"}.get(result.score_outcome, "ERR")
        err_msg = result.error or result.verifier_error
        err = f" ({truncate_end(err_msg, 50)})" if err_msg else ""
        # Show quality separately from the hard-gate success classification.
        reward_part = (
            f"reward={reward:.2f}, "
            if isinstance(reward, (int, float)) and not isinstance(reward, bool)
            else ""
        )
        logger.info(
            f"[{status}] {td.name} ({reward_part}tools={result.n_tool_calls}){err}"
        )
        self._fire_progress(self._on_result, td.name, result)

    async def _run_parallel_independent(
        self, remaining: list[Path]
    ) -> list[tuple[str, RunResult]]:
        """The default schedule — rollouts run concurrently and isolated."""
        cfg = self._config
        # Console heartbeat auto-gate: interleaved per-task progress lines are
        # noise at high concurrency, so the sessions' heartbeat defaults off
        # when several tasks run at once. It counts tasks actually running, not
        # --concurrency: a job with fewer running tasks than --concurrency
        # would stay silent long enough for CI to treat it as a hang. An explicit
        # BENCHFLOW_PROGRESS=on/off from the operator always wins (checked
        # first in the session layer).
        running = min(cfg.concurrency, len(remaining))
        os.environ["BENCHFLOW_PROGRESS_AUTO"] = "1" if running <= 1 else "0"
        # Floor at 1: Semaphore(0) deadlocks on first acquire. eval-create already
        # rejects <1 at plan time, but this guards every other caller (skills eval,
        # SDK) against a silent forever-hang on a bad concurrency.
        sem = asyncio.Semaphore(max(1, cfg.concurrency))

        breaker = ApiErrorCircuitBreaker()
        guard = self._budget_guard
        usage_stop = self._usage_stop

        async def bounded(td: Path) -> tuple[str, RunResult | None]:
            async with sem:
                if guard is not None and guard.stopped:
                    guard.start(td.name)  # records it as not started
                    return td.name, None
                if usage_stop.skip(td.name):
                    return td.name, None
                if breaker.tripped:
                    result = RunResult(task_name=td.name, error=breaker.skip_error())
                    self._log_and_report(td, result)
                    return td.name, result
                # Jitter start to avoid SSH/docker-daemon storms at high
                # concurrency. The window scales linearly with --concurrency so
                # the average start rate stays around 2 tasks/sec; the previous
                # 10s cap was too tight for c >= 30 (≈10 starts/sec flooded the
                # daemon's compose-up handler).
                import random

                if cfg.concurrency > 16:
                    jitter_max = max(cfg.concurrency / 2, 8.0)
                    await asyncio.sleep(random.uniform(0, jitter_max))
                if guard is not None and not guard.start(
                    td.name, asyncio.current_task()
                ):
                    return td.name, None
                self._fire_progress(self._on_task_start, td.name)
                result = await self._run_budgeted(td, guard)
                if result is None:
                    return td.name, None
                usage_stop.record(result)
                breaker.record(result)
                self._log_and_report(td, result)
                return td.name, result

        watcher = asyncio.ensure_future(guard.watch()) if guard is not None else None
        try:
            results_or_errors = await asyncio.gather(
                *[bounded(td) for td in remaining],
                return_exceptions=True,
            )
        finally:
            if watcher is not None:
                watcher.cancel()

        # Separate successful results from unexpected exceptions
        pairs: list[tuple[str, RunResult]] = []
        for i, r in enumerate(results_or_errors):
            if isinstance(r, BaseException):
                if (
                    isinstance(r, asyncio.CancelledError)
                    and guard is not None
                    and guard.was_cancelled(remaining[i].name)
                ):
                    continue  # the budget cancelled it between awaits
                if isinstance(r, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise r
                task_name = remaining[i].name
                logger.error(f"[ERR] {task_name}: unexpected exception: {r}")
                err_result = RunResult(task_name=task_name, error=f"Unexpected: {r}")
                # _run_task raised after on_task_start fired, so the live
                # dashboard still has this task "running" — fire on_result to
                # remove it and count it errored.
                self._fire_progress(self._on_result, task_name, err_result)
                pairs.append((task_name, err_result))
            elif r[1] is not None:
                pairs.append((r[0], r[1]))
        return pairs

    async def _run_budgeted(
        self, td: Path, guard: BudgetGuard | None
    ) -> RunResult | None:
        """Run one task; None when the budget cancelled it while it ran.

        The cancellation reaches the rollout, whose lifecycle cleans up its
        sandbox and writes no result.json, so a resume runs the task again.
        """
        if guard is None:
            return await self._run_task(td)
        try:
            result = await self._run_task(td)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if guard.was_cancelled(td.name) and current is not None:
                current.uncancel()
                # Take it off the live dashboard; it is not a job result.
                self._fire_progress(
                    self._on_result,
                    td.name,
                    RunResult(task_name=td.name, error=f"cancelled: {guard.reason}"),
                )
                logger.info(f"[CANCELLED] {td.name}: {guard.reason}")
                return None
            raise
        except Exception:
            guard.finish(td.name, None)  # stop counting its sandbox time
            raise
        guard.finish(td.name, result)
        return result

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

                guard = self._budget_guard
                if (guard is not None and guard.stopped) or self._usage_stop.error:
                    if self._usage_stop.error is not None:
                        self._usage_stop.skip(td.name)
                    else:
                        assert guard is not None
                        guard.start(td.name)  # records it as not started
                    self._learner_skills_dir = None
                    self._learner_export_dir = None
                    continue
                self._fire_progress(self._on_task_start, td.name)
                try:
                    if guard is None:
                        result = await self._run_task(td)
                    else:
                        # Its own task so the budget can cancel the trial
                        # without cancelling the job.
                        trial = asyncio.ensure_future(self._run_budgeted(td, guard))
                        guard.start(td.name, trial)
                        watcher = asyncio.ensure_future(guard.watch())
                        try:
                            maybe = await trial
                        finally:
                            watcher.cancel()
                        if maybe is None:
                            continue
                        result = maybe
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as e:  # mirror the parallel path's catch
                    logger.error(f"[ERR] {td.name}: unexpected exception: {e}")
                    err_result = RunResult(task_name=td.name, error=f"Unexpected: {e}")
                    self._fire_progress(self._on_result, td.name, err_result)
                    pairs.append((td.name, err_result))
                    continue
                finally:
                    self._learner_skills_dir = None
                    self._learner_export_dir = None

                self._usage_stop.record(result)
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

    def _maybe_start_daytona_reap(self) -> None:
        """Fire-and-forget auto-reap of orphaned Daytona sandboxes (issue: leakage at scale).

        Gated by ``BENCHFLOW_DAYTONA_AUTO_REAP`` (default on; any of
        ``0``/``false``/``no``/``off`` case-insensitively disables it).
        Conservative TTLs (24h general / 2h failed states) plus an idle-activity
        guard mean concurrent live runs are never reaped. Runs in a daemon
        thread so eval startup never blocks or fails on reaping.
        """
        if self._config.environment != "daytona":
            return
        if os.environ.get("BENCHFLOW_DAYTONA_AUTO_REAP", "1").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return

        def _reap() -> None:
            try:
                from benchflow.sandbox.daytona import reap_stale_sandboxes

                counts = reap_stale_sandboxes()
                if counts["deleted"] or counts["failed"]:
                    logger.info(
                        "Daytona auto-reap: %s stale sandboxes deleted (%s failed)",
                        counts["deleted"],
                        counts["failed"],
                    )
            except Exception as e:
                logger.debug("Daytona auto-reap skipped: %s", e)
            try:
                from benchflow.sandbox.daytona import reap_stale_snapshots

                snaps = reap_stale_snapshots()
                if snaps["deleted"] or snaps["failed"]:
                    logger.info(
                        "Daytona auto-reap: %s stale branch snapshots deleted "
                        "(%s failed)",
                        snaps["deleted"],
                        snaps["failed"],
                    )
            except Exception as e:
                logger.debug("Daytona snapshot auto-reap skipped: %s", e)

        threading.Thread(target=_reap, name="daytona-auto-reap", daemon=True).start()

    @classmethod
    def resume(
        cls,
        job_dir: str | Path,
        *,
        tasks_dir: str | Path | None = None,
        agent_env: dict[str, str] | None = None,
        **config_overrides: Any,
    ) -> Evaluation:
        """Rebuild the Evaluation that created ``job_dir``, ready to finish it.

        Reads the ``evaluation.json`` a job writes when it starts (tasks
        directory and config). Running the returned Evaluation reuses the
        finished tasks and runs only the rest. agent_env values are never
        stored, so pass ``agent_env`` again if the agent needs it;
        ``config_overrides`` replace individual config fields (e.g.
        ``concurrency=8``). Jobs started before this record existed need
        ``Evaluation(tasks_dir, jobs_dir=job_dir.parent, config=...,
        job_name=job_dir.name)`` instead.
        """
        import dataclasses

        job_dir = Path(job_dir)
        record_path = job_dir / EVALUATION_RECORD
        if not record_path.is_file():
            raise FileNotFoundError(
                f"No {EVALUATION_RECORD} in {job_dir} (the job predates it or this "
                "is not a job directory). Resume it with Evaluation(tasks_dir=..., "
                f"jobs_dir={str(job_dir.parent)!r}, config=..., "
                f"job_name={job_dir.name!r})."
            )
        record = json.loads(record_path.read_text())
        config = _config_from_record(record["config"], agent_env)
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)
        return cls(
            tasks_dir=tasks_dir or record["tasks_dir"],
            jobs_dir=job_dir.parent,
            config=config,
            job_name=job_dir.name,
        )

    @property
    def job_dir(self) -> Path:
        """Where this job writes its rollouts, known before it runs.

        Pass it to :func:`benchflow.astream_rollouts` to train on rollouts
        while :meth:`run` is still going.
        """
        return self._jobs_dir / self._job_name

    def _write_evaluation_record(self) -> None:
        """Record tasks_dir and config so Evaluation.resume(job_dir) can finish the job."""
        job_dir = self._jobs_dir / self._job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "tasks_dir": str(self._tasks_dir),
            "job_name": self._job_name,
            "config": _config_to_record(self._config),
        }
        (job_dir / EVALUATION_RECORD).write_text(json.dumps(record, indent=2) + "\n")

    async def stream(self) -> AsyncIterator[tuple[str, RolloutResult]]:
        """Run the job, yielding ``(task_name, RolloutResult)`` as each task finishes.

        Tasks reused from an earlier run of the same job are not yielded; they
        are in ``evaluation.result.results`` once the stream ends. Leaving the
        loop early cancels the job (wrap the stream in
        ``contextlib.aclosing`` so that happens immediately).
        """
        queue: asyncio.Queue[tuple[str, RolloutResult]] = asyncio.Queue()
        previous = self._on_result

        def _push(name: str, result: RolloutResult) -> None:
            if previous is not None:
                previous(name, result)
            queue.put_nowait((name, result))

        self._on_result = _push
        runner = asyncio.create_task(self.run())
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {getter, runner}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    yield getter.result()
                    continue
                getter.cancel()
                while not queue.empty():
                    yield queue.get_nowait()
                runner.result()  # re-raise a failed run
                return
        finally:
            self._on_result = previous
            if not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

    def run_sync(self) -> EvaluationResult:
        """Blocking form of :meth:`run` (also works inside a running event loop)."""
        from benchflow.batch import run_blocking

        return run_blocking(self.run)

    async def run(self) -> EvaluationResult:
        """Execute the job.

        Makes bf.run's pre-run checks first (see ``preflight``), then holds
        ``<job_dir>/.evaluation.lock`` while it runs, so a second run of
        the same job (another process, or ``Evaluation.resume`` of a job that
        is still going) is refused instead of running the same tasks twice.
        """
        if self._preflight:
            self._check_before_run()
        lock = self._acquire_job_lock()
        try:
            return await self._run_unlocked()
        finally:
            with contextlib.suppress(OSError):
                lock.unlink()
            # A run refused before it wrote anything (e.g. an empty task
            # selection) leaves no empty job directory behind.
            with contextlib.suppress(OSError):
                lock.parent.rmdir()

    def _check_before_run(self) -> None:
        """The checks bf.run and ``bench eval run`` make before a job exists.

        A misspelt agent or unknown sandbox raises ``ValueError``; Docker not
        ready for ``environment="docker"`` raises ``RuntimeError`` with
        ``bench doctor``'s fix; a Claude agent that would fall back to an
        expired login file gets a ``UserWarning``. ``Evaluation(...,
        preflight=False)`` skips all of them; ``BENCHFLOW_SKIP_PREFLIGHT=1``
        skips the host checks (Docker, login file), as for the CLI.
        """
        from benchflow.runtime import check_agent_names, check_host, check_sandbox_name

        cfg = self._config
        check_sandbox_name(cfg.environment)
        check_agent_names([cfg.agent])
        check_host([cfg])

    def _acquire_job_lock(self) -> Path:
        """Create the job lock, refusing a live holder and taking over a dead one."""
        import socket

        job_dir = self._jobs_dir / self._job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        lock = job_dir / JOB_LOCK
        me = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now().isoformat(),
        }
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    holder = json.loads(lock.read_text())
                except (OSError, ValueError):
                    holder = {}
                pid, host = holder.get("pid"), holder.get("host")
                if host and host != me["host"]:
                    raise RuntimeError(
                        f"Job {job_dir} is already running (or crashed) on host "
                        f"{host} (pid {pid}); delete {lock} if that run is gone."
                    ) from None
                if isinstance(pid, int) and _pid_alive(pid):
                    raise RuntimeError(
                        f"Job {job_dir} is already running in process {pid} "
                        f"(started {holder.get('started_at', '?')}); wait for it "
                        "to finish or stop it before resuming."
                    ) from None
                logger.warning(
                    "Taking over a stale job lock from process %s, which is gone: %s",
                    pid,
                    lock,
                )
                with contextlib.suppress(FileNotFoundError):
                    lock.unlink()
                continue
            with os.fdopen(fd, "w") as handle:
                json.dump(me, handle)
            return lock
        raise RuntimeError(f"Could not take the job lock {lock}")

    async def _run_unlocked(self) -> EvaluationResult:
        self._maybe_start_daytona_reap()
        task_dirs = self._get_task_dirs()
        if not task_dirs:
            # Fail fast on an empty selection (#407). Silently writing a
            # 0/0 summary.json would surface as an apparently successful
            # eval in downstream dashboards and release evidence.
            raise EmptyTaskSelectionError(self._empty_selection_message())
        from benchflow.review.resume import resume_pending_reviews

        await resume_pending_reviews(
            self._jobs_dir / self._job_name,
            tasks_root=self._tasks_dir,
            reviewer=self._config.reviewer,
            task_names={task_dir.name for task_dir in task_dirs},
        )
        completed = self._get_completed_tasks()
        remaining = [d for d in task_dirs if d.name not in completed]

        # Validate every selected review plan before spending on any solver.
        # Workers repeat this against their own effective credential context.
        from benchflow.review.automatic import prepare_review

        for task_dir in remaining:
            prepare_review(task_dir, self._config.reviewer)

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
            _check_resume_mismatch(self._jobs_dir / self._job_name, self._config)

        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._write_evaluation_record()
        await self._sweep_docker()

        cfg = self._config

        if cfg.build_concurrency is not None and cfg.environment in (
            "docker",
            "remote-docker",
        ):
            from benchflow.sandbox.docker import DockerSandbox

            DockerSandbox.set_build_concurrency(cfg.build_concurrency)

        # The denominator is the number of tasks that will appear in the summary:
        # the resumed-complete set plus the to-run set (disjoint by construction).
        # This equals len(task_dirs) for a clean run, but a resume whose jobs_dir
        # holds results from a *wider* prior selection has more completed rows than
        # the current selection — keying off len(task_dirs) there rendered nonsense
        # like "11/1 · 1100%" and a denominator that disagreed with the final
        # "Score: 8/11". completed-plus-remaining is exactly what gets scored.
        planned_total = len(completed) + len(remaining)
        logger.info(
            f"Job: {planned_total} tasks, {len(completed)} done, "
            f"{len(remaining)} to run (concurrency={cfg.concurrency})"
        )
        # Hand the live dashboard an honest denominator: total / already-done /
        # to-run. Resumed-complete tasks fold in below without a finish event, so
        # a finish-event-only counter would mis-read the total on resume. Pass the
        # resumed tasks' (passed, failed, errored) breakdown too, so the live
        # counts + pass-rate cover the whole job, not just this process's tasks.
        self._fire_progress(
            self._on_plan,
            planned_total,
            len(completed),
            len(remaining),
            _classify_completed_outcomes(completed),
        )

        start = time.time()

        self._usage_stop = UsageLimitStop()
        self._budget_guard = None
        if cfg.budget is not None:
            self._budget_guard = BudgetGuard(cfg.budget)
            # A resumed job has already spent what its finished trials used.
            self._budget_guard.seed(completed.values())

        if cfg.job_mode == "sequential-shared":
            pairs = await self._run_sequential_shared(remaining)
        else:
            pairs = await self._run_parallel_independent(remaining)
        await self._sweep_docker()
        elapsed = time.time() - start

        job_dir = self._jobs_dir / self._job_name
        all_results: dict[str, dict] = {}
        typed_results: dict[str, RolloutResult] = {}
        for task, data in completed.items():
            all_results[task] = data
            resumed_dir = job_dir / (data.get("rollout_name") or "")
            typed_results[task] = RolloutResult.from_dict(
                data,
                rollout_dir=resumed_dir if data.get("rollout_name") else None,
            )
        for name, result in pairs:
            typed_results[name] = result
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
        memory, memory_scores = memory_summary(all_results)
        # Per-task failure evidence for the CLI's final block — FAILED (scored,
        # failed gate outcome) tasks only, from data already in memory. Sorted by name
        # so the printed lines are deterministic across resume/concurrency.
        task_failures = [
            TaskFailure(
                task_name=name,
                rewards=r.get("rewards"),
                verifier_error=r.get("verifier_error"),
                # `or None`: RolloutResult defaults rollout_name to "" — don't
                # let that masquerade as a resolvable rollout dir.
                rollout_name=r.get("rollout_name") or None,
            )
            for name, r in sorted(all_results.items())
            if classify_score_outcome(r) == "failed"
        ]
        job_result = EvaluationResult(
            job_name=self._job_name,
            config=cfg,
            reused=len(completed),
            ran=len(pairs),
            # Score counts cover one entry per scored rollout. Skill-eval expands
            # a single task into multiple rollouts (baseline/skill x trials), so
            # the denominator must be the number of results, not task dirs, or the
            # invariant below (and every pass-rate/percentage) is wrong.
            total=len(all_results),
            passed=score_counts["passed"],
            failed=score_counts["failed"],
            errored=score_counts["errored"],
            verifier_errored=score_counts["verifier_errored"],
            elapsed_sec=elapsed,
            memory_score=memory["avg_score"],
            memory_scores=memory_scores,
            task_failures=task_failures,
            mean_reward=mean_scored_reward(all_results.values()),
            job_dir=job_dir,
            results=typed_results,
            budget=(
                self._budget_guard.summary() if self._budget_guard is not None else None
            ),
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

        scoring_fields = score_summary_fields(all_results.values())
        error_category_counts = scoring_fields["error_categories"] or {}
        verifier_error_category_counts = (
            scoring_fields["verifier_error_categories"] or {}
        )

        # Save summary
        summary = {
            "job_name": self._job_name,
            # The reviewer settings are recorded for every job; "ran" says
            # whether any trial was actually scored by one.
            "reviewer": {
                **cfg.reviewer.to_config_artifact(),
                "ran": any(r.get("scoring") is not None for r in all_results.values()),
            },
            "agent": cfg.agent,
            "codex_apps_policy": cfg.codex_apps_policy,
            "model": cfg.model,
            "environment": cfg.environment,
            "concurrency": cfg.concurrency,
            "agent_idle_timeout_sec": cfg.agent_idle_timeout,
            "usage_tracking": cfg.usage_tracking.with_env_defaults().to_config_artifact(),
            "loop": loop_block(cfg.loop_strategy),
            **scoring_fields,
            "elapsed_sec": elapsed,
            "memory_score": job_result.memory_score,
            "memory_score_coverage": (
                len(memory_scores) / job_result.total if job_result.total else 0.0
            ),
            "memory": memory,
            "memory_scores": memory_scores,
            **skill_invocation_summary(all_results),
            **usage_summary(all_results),
            **loop_summary(all_results),
            **solve_rate_summary(all_results),
            **tool_call_summary(all_results),
            # Retries from checkpoints, next to (never merged into) the score.
            **(
                {"checkpoint_retries": retries}
                if (retries := retry_summary(list(typed_results.values()))) is not None
                else {}
            ),
            **trajectory_step_summary(all_results),
            **phase_timing_summary(all_results),
            **({"budget": job_result.budget} if job_result.budget is not None else {}),
            **(
                {"usage_limit": stop}
                if (stop := self._usage_stop.summary()) is not None
                else {}
            ),
            **summary_source_fields(cfg.source_provenance, all_results),
            **(
                {
                    "dataset_name": cfg.dataset_name,
                    "dataset_version": cfg.dataset_version,
                }
                if cfg.dataset_name
                else {}
            ),
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
        try:
            from benchflow.trajectories.export_adp import write_job_adp_jsonl

            write_job_adp_jsonl(job_dir)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Job-level ADP aggregation failed: %s", e)
        try:
            from benchflow.trajectories.results import write_job_results_jsonl

            write_job_results_jsonl(job_dir)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Job-level results.jsonl aggregation failed: %s", e)

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
        if scoring_fields["verifier_errored"] > 0:
            pct = scoring_fields["verifier_errored"] / job_result.total * 100
            logger.warning(
                f"{scoring_fields['verifier_errored']} tasks ({pct:.0f}%) had verifier errors — "
                f"check verifier scripts for bugs"
            )
            if pct > 20:
                logger.error(
                    "Over 20% of tasks had verifier errors — results may be unreliable. "
                    "This likely indicates a systemic verifier bug, not agent failure."
                )

        mean_part = (
            f"mean_reward={job_result.mean_reward:.2f}, "
            if job_result.mean_reward is not None
            else ""
        )
        logger.info(
            f"Job complete: {job_result.passed}/{job_result.total} "
            f"({job_result.score:.1%}), {mean_part}errors={job_result.errored}, "
            f"idle_timeouts={error_category_counts.get(IDLE_TIMEOUT, 0)}, "
            f"time={elapsed / 60:.1f}min"
        )

        self.result = job_result
        if self._usage_stop.error is not None:
            # The job stopped on its login's usage limit: raise the typed
            # error (with this result) so a caller can switch logins and
            # resume; summary.json and every finished trial are written.
            self._usage_stop.error.result = job_result
            raise self._usage_stop.error
        return job_result
