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
import contextvars
import copy
import glob
import json
import logging
import os
import re
import shlex
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from benchflow import _park
from benchflow._diagnostics import render_text
from benchflow._utils import timing_spans
from benchflow._utils.command_hints import command_arg
from benchflow._utils.env_values import env_value_text, normalize_env_mapping
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
from benchflow._utils.job_lock import JobLock, JobLockedError
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
    DISK_FULL,
    IDLE_TIMEOUT,
    INFRA_ERROR,
    INSTALL_FAILED,
    OTHER_ERROR,
    PIPE_CLOSED,
    PROVIDER_AUTH,
    PROVIDER_RATE_LIMIT,
    PROVIDER_REJECTED,
    SANDBOX_SETUP,
    SUSPECTED_API_ERROR,
    USAGE_LIMIT,
    VERIFIER_DEP_INSTALL,
    VERIFIER_INFRA,
    api_error_is_transient,
    classify_error,
    classify_score_outcome,
    classify_verifier_error,
    count_guard_refusals,
    count_score_outcomes,
    is_unrecoverable_startup_error,
    mean_reward_caveat,
    mean_reward_every_trial,
    mean_scored_reward,
    pass_rate,
    pass_rate_excl_errors,
    plugin_guard_error_message,
    retry_category,
    score_summary_fields,
    scored_trial_count,
)
from benchflow._utils.source_provenance import summary_source_fields
from benchflow._utils.start_pacing import PACE_ABOVE_CONCURRENCY, StartPacer
from benchflow._warm_pool import (
    WarmPool,
    task_shares_nothing,
    warm_pool_note,
    warm_pool_size,
)
from benchflow.agents.errors import UsageLimitError
from benchflow.budget import (
    CANCELLED_BUDGET_RECORD,
    JOB_BUDGET_RECORDS,
    Budget,
    BudgetGuard,
    task_budget_path,
    with_cancelled_budget,
)
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
from benchflow.task.verifier_sandbox import effective_verifier_sandbox
from benchflow.trajectories.tree import RolloutNode
from benchflow.usage_tracking import UsageTrackingConfig

# Backward-compat alias
RunResult = RolloutResult

if TYPE_CHECKING:
    from benchflow.checkpoint_retry import RetryPolicy
    from benchflow.checkpoints import CheckpointPolicy
    from benchflow.review.outcome import ScoringResult

logger = logging.getLogger(__name__)

# The prepared rollout the warm pool handed the trial running in this task,
# taken once by its first attempt (_run_single_task).
_WARM_ROLLOUT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "benchflow_warm_rollout", default=None
)

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
    """The manifest a task.md names, refused if it forwards an unnamed variable.

    ``forward_env`` copies variables of the operator's environment into the
    sandbox, so a task's own manifest gets only the ones the operator named
    (``--allow-task-env``), like a ``${VAR}`` in the task's env tables.
    """
    from benchflow.task.env import TaskEnvNotAllowedError

    manifest = _load_task_document_manifest(task_dir)
    refused = _refused_forward_env(manifest)
    if refused:
        raise TaskEnvNotAllowedError({task_dir.name: refused})
    return manifest


def _refused_forward_env(manifest: EnvironmentManifest | None) -> list[Any]:
    from benchflow.task.env import TaskEnvRequest, refused_task_env

    if manifest is None:
        return []
    return refused_task_env(
        TaskEnvRequest(
            "forward_env", key, None, f"its environment manifest forwards {key}"
        )
        for key in manifest.forward_env.keys
    )


def _task_env_refusals(
    task_dir: Path,
    *,
    manifest: EnvironmentManifest | None = None,
    sandbox: str | None = "docker",
) -> list[Any]:
    """What the task in *task_dir* reads from the operator without being named.

    Its env tables' ``${VAR}``, its task.md manifest's ``forward_env`` (when
    the operator passed no *manifest* of their own) and, on a backend whose
    Compose runs here, its docker-compose.yaml. A task that does not load is
    skipped here: its trial reports the error.
    """
    return _task_env_findings(task_dir, manifest=manifest, sandbox=sandbox)[0]


def _task_env_findings(
    task_dir: Path,
    *,
    manifest: EnvironmentManifest | None = None,
    sandbox: str | None = "docker",
) -> tuple[list[Any], list[Any]]:
    """The task's refusals (:func:`_task_env_refusals`), and the unnamed
    ``${VAR:-default}`` reads of its env tables and Compose files whose
    variable the operator has set: those get the default, never that value,
    so the caller warns (:func:`benchflow.task.env.defaulted_task_env`)."""
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config
    from benchflow.task import Task
    from benchflow.task.compose_env import (
        COMPOSE_CHECKED,
        check_compose,
        task_compose_values,
    )
    from benchflow.task.env import (
        defaulted_task_env,
        refused_task_env,
        task_env_requests,
    )

    refused: list[Any] = []
    defaulted: list[Any] = []
    try:
        task = Task(task_dir)
    except Exception:
        return refused, defaulted
    requests = task_env_requests(task.config)
    refused += refused_task_env(requests)
    if manifest is None:
        with contextlib.suppress(Exception):
            manifest = _load_task_document_manifest(task_dir)
            refused += _refused_forward_env(manifest)
    if sandbox in COMPOSE_CHECKED:
        provided = task_compose_values(
            task.config.sandbox.env, manifest, task_id=task_dir.name
        )
        with contextlib.suppress(OSError):
            checked = check_compose(
                task.paths.environment_dir,
                provided,
                sandbox_type=sandbox or "docker",
                sandbox_config=agent_network_sandbox_config(task.config),
            )
            refused += checked.refused
            requests += checked.defaulted
    defaulted += defaulted_task_env(requests)
    return refused, defaulted


def _refuse_placeholder_tasks(task_dirs: list[Path]) -> None:
    """Refuse the job if a task still holds the ``[REPLACE: ...]``
    placeholders ``bench new`` writes (DFR-p1-7).

    The same check as ``bench check``: such a task builds its image and then
    scores 0. Runs before any image is built or sandbox started."""
    from benchflow._utils.task_authoring import check_task

    unfilled: dict[Path, list[str]] = {}
    for task_dir in task_dirs:
        try:
            issues = check_task(task_dir)
        except Exception:  # the run reports a task it cannot read
            continue
        found = [issue for issue in issues if "unreplaced placeholder" in issue]
        if found:
            unfilled[task_dir] = found
    if not unfilled:
        return
    first, issues = next(iter(unfilled.items()))
    names = _some_names(sorted(d.name for d in unfilled))
    error = MalformedTaskError(
        f"{len(unfilled)} task(s) still hold [REPLACE: ...] placeholders, so "
        f"they would build and then score 0: {names} (first: {first.name}: "
        f"{'; '.join(issues)})"
    )
    error.hint = (
        "fill in every placeholder; "
        f"`bench check {command_arg(first, '<task folder>')}` lists them"
    )
    raise error


def _refuse_unnamed_task_env(
    task_dirs: list[Path],
    *,
    manifest: EnvironmentManifest | None,
    sandbox: str,
) -> None:
    """Refuse the job if any task reads a variable the operator did not name.

    Runs before any sandbox (review sandboxes included) starts, so one
    message lists every task and variable instead of each trial failing.
    """
    from benchflow.task.env import TaskEnvNotAllowedError, defaulted_task_env_warning

    refused: dict[str, list[Any]] = {}
    defaulted: dict[str, list[Any]] = {}
    for task_dir in task_dirs:
        found, unfilled = _task_env_findings(
            task_dir, manifest=manifest, sandbox=sandbox
        )
        if found:
            refused[task_dir.name] = found
        if unfilled:
            defaulted[task_dir.name] = unfilled
    if refused:
        raise TaskEnvNotAllowedError(refused)
    # A task that runs with its default where the operator has a value set
    # (a Toolathlon token, say) would score as if the credential were wrong.
    warning = defaulted_task_env_warning(defaulted)
    if warning:
        logger.warning("warning: %s", warning)


def _load_task_document_manifest(task_dir: Path) -> EnvironmentManifest | None:
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


def _pin_evaluation_subnets(task_dirs, config):
    """Reserve each task with the same override and manifest as _run_task."""
    from benchflow.sandbox._docker_subnets import pin_task_subnets

    pinned = []
    for task_dir in task_dirs:
        manifest = config.environment_manifest
        if manifest is None:
            manifest = _environment_manifest_from_task_document(task_dir)
        pinned.extend(
            pin_task_subnets(
                [task_dir / "environment" / "docker-compose.yaml"],
                backend=config.environment,
                environment_manifest=manifest,
                config_override=config.config_override,
            )
        )
    return pinned


def _is_task_dir(path: Path, *, image_supplied: bool = False) -> bool:
    # The rule the Python API's check_rollout_config applies too.
    from benchflow._utils.task_authoring import runnable_task_issues

    return runnable_task_issues(path, image_supplied=image_supplied) == []


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


class JobMatrixError(ValueError, UserError):
    """A YAML job config of several jobs (agents x datasets x n_attempts)
    loaded as one Evaluation: the first job alone would be a fraction of the
    run the file asks for."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class ResumeMismatchError(ValueError, UserError):
    """Raised when a job's earlier trials ran with other settings and could not
    be set aside.

    A job holds one run. Folding trials of another agent, model, timeout or
    any other setting in :data:`_RESUME_SETTINGS` into this run would publish
    a blended ``Score: X/N`` that belongs to neither, and re-running over the
    prior rows would destroy them; so they move to ``<job>/.set-aside/``
    (:func:`_check_resume_mismatch`). When that move fails the run is refused;
    the fix is a new job (``--fresh``, another ``--job-name``).
    """


class Pre08CopyError(RuntimeError, UserError):
    """A 0.7 job's own files could not be kept before resuming it rewrites them."""

    def __init__(self, message: str, *, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


# A key every 0.8 trial's config.json has and BenchFlow 0.7's never had (the
# harness record, PY8); trials since DX-e also record their benchflow_version.
_PY08_TRIAL_KEYS = ("benchflow_version", "harness_mode")

# Keys every BenchFlow 0.7.x trial's config.json has from the trial's start
# (0.7.0 to 0.7.11 write them unconditionally, read from each release's
# trial writer); together they mark a trial BenchFlow 0.7 started, even one
# stopped before its result.json.
_PY07_TRIAL_KEYS = (
    "task_path",
    "agent",
    "model",
    "reasoning_effort",
    "environment",
    "acp_transport",
    "environment_manifest",
    "sandbox_user",
    "sandbox_locked_paths",
    "sandbox_setup_timeout",
    "skip_install",
    "agent_install_timeout",
    "context_root",
    "base_image_override",
    "timeout_sec",
    "concurrency",
    "agent_idle_timeout_sec",
    "started_at",
    "agent_env",
    "scenes",
    "loop",
)

# Keys every BenchFlow 0.7.x job's summary.json has (0.7.0 to 0.7.11 write
# them unconditionally at the job's end).
_PY07_SUMMARY_KEYS = (
    "job_name",
    "agent",
    "model",
    "environment",
    "concurrency",
    "agent_idle_timeout_sec",
    "usage_tracking",
    "loop",
    "total",
    "passed",
    "failed",
    "errored",
    "verifier_errored",
    "pass",
    "fail",
    "error",
    "idle_timeout",
    "error_categories",
    "verifier_error_categories",
    "score",
    "score_ratio",
    "score_excl_errors",
    "score_excl_errors_ratio",
    "mean_reward",
    "elapsed_sec",
    "memory_score",
    "memory_score_coverage",
    "memory",
    "memory_scores",
)

# Keys of the job record (evaluation.json) every 0.8 job writes as it starts.
_RECORD_KEYS = ("schema_version", "tasks_dir", "job_name", "config")


def _made_by_07(trials: list[Path]) -> bool:
    """Whether BenchFlow 0.7 ran any of these trials, read from their config.json.

    Positive evidence only (L3): a trial whose config.json holds none of
    :data:`_PY08_TRIAL_KEYS`. A folder 0.8's ``bf.run(job_name=...)`` or
    ``bf.run_batch`` made has no evaluation.json either, but its trials do
    hold them; a trial without a readable config.json says nothing.
    """
    for trial in trials:
        try:
            config = json.loads((trial / "config.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(config, dict) and not any(k in config for k in _PY08_TRIAL_KEYS):
            return True
    return False


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_matches(kept: Path, job_dir: Path, names: list[str]) -> bool:
    """Whether ``kept`` holds every one of the job's ``names``, byte for byte."""
    if not kept.is_dir():
        return False
    try:
        return all(
            (kept / n).is_file()
            and (kept / n).stat().st_size == (job_dir / n).stat().st_size
            and _file_digest(kept / n) == _file_digest(job_dir / n)
            for n in names
        )
    except OSError:
        return False


def _copy_pre_08_files(job_dir: Path, kept: Path, names: list[str]) -> None:
    """Copy ``names`` into a temporary folder, then rename it to ``kept``.

    A copy an earlier, interrupted run left (``.pre-0.8.tmp-*``, or a
    ``.pre-0.8`` that does not match) is replaced. Raises
    :class:`Pre08CopyError` before the job is changed when it cannot copy.
    """
    import shutil

    for stale in job_dir.glob(f"{PRE_08_DIR}.tmp-*"):
        shutil.rmtree(stale, ignore_errors=True)
    temp = job_dir / f"{PRE_08_DIR}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        temp.mkdir()
        for name in names:
            shutil.copy2(job_dir / name, temp / name)
        if kept.exists():
            old = job_dir / f"{PRE_08_DIR}.tmp-{uuid.uuid4().hex[:8]}"
            kept.rename(old)
            temp.rename(kept)
            shutil.rmtree(old, ignore_errors=True)
        else:
            temp.rename(kept)
    except OSError as exc:
        shutil.rmtree(temp, ignore_errors=True)
        raise Pre08CopyError(
            f"Could not copy the BenchFlow 0.7 files of job {job_dir.name} to "
            f"{kept} before resuming it ({exc.strerror or exc}); nothing in the "
            "job was changed.",
            hint=(
                "free space in that folder's file system and run the same "
                "command again, or pass --fresh to start a new job and leave "
                "this one as it was"
            ),
        ) from None


class NoEvaluationRecordError(FileNotFoundError, UserError):
    """``Evaluation.resume`` was given a folder with no evaluation.json.

    ``hint`` is the next step: for a job an older BenchFlow made, the
    command that resumes it (DF-p4-upgrader-4).
    """

    def __init__(self, message: str, *, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def _pre_08_resume_command(job_dir: Path, trials: list[Path]) -> str:
    """The ``bench run`` line that resumes a job from before evaluation.json.

    Built from what its trials recorded (agent, model, sandbox, concurrency);
    the tasks folder is not recorded, so it stays a placeholder.
    """
    recorded: dict[str, Any] = {}
    for trial in trials:
        try:
            recorded = json.loads((trial / "config.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(recorded, dict):
            break
        recorded = {}
    words = ["bench", "run", "<tasks folder>"]
    for flag, key in (
        ("--agent", "agent"),
        ("--model", "model"),
        ("--sandbox", "environment"),
        ("--concurrency", "concurrency"),
    ):
        value = recorded.get(key)
        if (
            isinstance(value, (str, int))
            and not isinstance(value, bool)
            and value != ""
        ):
            words += [flag, shlex.quote(str(value))]
    words += [
        "--jobs-dir",
        shlex.quote(str(job_dir.parent)),
        "--job",
        shlex.quote(job_dir.name),
    ]
    return " ".join(words)


def _budget_overrides(cfg: Any, task_dir: Path) -> dict[str, Any]:
    """Per-rollout overrides from ``--timeout-multiplier`` and ``--extra-instruction``."""
    out: dict[str, Any] = {}
    if cfg.timeout_multiplier and cfg.timeout_multiplier != 1:
        from benchflow.task import Task
        from benchflow.task.config import effective_agent_timeout_sec

        base = effective_agent_timeout_sec(Task(task_dir).config.agent)
        if base is None:
            # No limit of its own: scale the rollout-deadline budget it runs
            # under, so a multiplier above 1 never shortens it.
            from benchflow.rollout import RolloutConfig
            from benchflow.rollout._deadline import uncapped_agent_budget_sec

            base = uncapped_agent_budget_sec(RolloutConfig(task_path=task_dir))
        out["timeout"] = max(1, round(float(base) * float(cfg.timeout_multiplier)))
    if cfg.extra_instruction:
        out["prompt_suffix"] = cfg.extra_instruction
    return out


def _seeds_from_config(value: Any) -> list[int] | None:
    """``seeds:`` in a job config: a list of ints, or the CLI form (``0-4``)."""
    if value is None or isinstance(value, list):
        return value
    from benchflow.embodied.rollouts import parse_seeds

    return parse_seeds(str(value))


# Every key a run config may hold, as ``Evaluation.from_yaml``'s two loaders
# read them (tests/test_run_config_keys.py keeps these equal to the loaders).
# Any other key is refused with a did-you-mean: a misspelled ``concurrenzy``
# must never run with the default (dogfood p5-3).
_USAGE_KEYS = frozenset(
    {
        "usage_tracking",
        "usage_proxy",
        "usage_proxy_advertised_base_url",
        "usage_proxy_bind_host",
        "usage_proxy_port",
        "usage_proxy_url",
    }
)
RUN_CONFIG_KEYS: frozenset[str] = _USAGE_KEYS | frozenset(
    {
        "agent", "agent_env", "agent_env_keys", "agent_idle_timeout",
        "agent_idle_timeout_sec", "allow_cloud_metadata",
        "base_image_override", "budget",
        "build_concurrency", "checkpoint_keep", "checkpoints", "codex_apps_policy",
        "concurrency", "config_override", "context_root", "dataset_name",
        "dataset_task_digests", "dataset_version", "environment",
        "environment_manifest", "exclude", "extra_instruction", "freeze_workspace",
        "fresh_install", "harness", "include", "integrity", "job_mode", "jobs_dir",
        "loop_strategy", "max_retries", "model", "n_tasks", "prompts",
        "reasoning_effort", "retry", "retry_from_checkpoint", "retry_prompt",
        "retry_resume_session", "reviewer", "sample_seed", "sandbox_locked_paths",
        "sandbox_setup_timeout", "sandbox_user", "seeds", "self_gen_no_internet",
        "separate_verifier", "separate_verifier_include", "skill_creator_dir",
        "skill_mode", "skills_dir", "skip_install", "source", "source_provenance",
        "tasks_dir", "timeout_multiplier", "verifier_sandbox", "warm_pool",
    }
)  # fmt: skip
LEGACY_RUN_CONFIG_KEYS: frozenset[str] = _USAGE_KEYS | frozenset(
    {
        "agents", "datasets", "environment", "orchestrator", "n_attempts",
        "jobs_dir", "agent_idle_timeout", "agent_idle_timeout_sec",
        "allow_cloud_metadata",
        "base_image_override", "codex_apps_policy", "context_root",
        "fresh_install", "harness", "integrity", "reasoning_effort", "reviewer",
        "sandbox_locked_paths", "sandbox_setup_timeout", "sandbox_user",
        "self_gen_no_internet", "separate_verifier", "separate_verifier_include",
        "skill_creator_dir", "skill_mode", "skills_dir", "verifier_sandbox",
        "include", "includes", "include_tasks", "exclude", "excludes",
        "exclude_tasks",
        # Job keys that the legacy loader honours (review F2).
        "job_name", "timeout_multiplier", "n_concurrent_trials",
        "extra_instructions", "tasks",
    }
)  # fmt: skip
# Other YAML job config keys are validated in benchflow.job_config.
# A setting BenchFlow cannot honour is refused by name.
# Any unknown key is refused
# with a did-you-mean (review F2).


# The include and exclude spellings both run-config formats read.
_INCLUDE_KEYS = ("include", "includes", "include_tasks")
_EXCLUDE_KEYS = ("exclude", "excludes", "exclude_tasks")


def default_oracle_note(path: Path | str | None) -> str:
    """The one line a YAML job config that names no agent prints (D923)."""
    where = f"{path} " if path is not None else "this YAML job config "
    return (
        f"note: {where}names no agent, so the oracle runs; "
        "pass -a <agent> (or add agents:) to run a model agent"
    )


_SOURCE_KEYS = frozenset({"repo", "path", "ref"})


def check_run_config_keys(
    raw: dict[str, Any], *, path: Path, text: str, legacy: bool = False
) -> dict[str, Any]:
    """Refuse a run config key the loader would not read, naming its line.

    ``sandbox:`` is taken as ``environment:`` (the flag is ``--sandbox``);
    both at once is refused. Returns the config to load.
    """
    from benchflow._utils.config_keys import ConfigKeyError, check_keys, key_lines

    lines = key_lines(text)
    problems: list[str] = []
    if not legacy and "sandbox" in raw:
        if "environment" in raw:
            problems.append(
                f"{path}:{lines.get(('sandbox',), '?')}: sandbox and environment "
                "name the same setting; keep one"
            )
        else:
            raw = {
                ("environment" if key == "sandbox" else key): value
                for key, value in raw.items()
            }
    from benchflow.job_config import JOB_KEYS, job_config_problems

    allowed = LEGACY_RUN_CONFIG_KEYS | JOB_KEYS if legacy else RUN_CONFIG_KEYS
    problems += check_keys(raw, allowed, file=path, lines=lines)
    source = raw.get("source")
    if not legacy and isinstance(source, dict):
        problems += check_keys(
            source, _SOURCE_KEYS, file=path, lines=lines, prefix=("source",)
        )
    docs = "https://github.com/benchflow-ai/benchflow/blob/main/docs/jobs/configs.md"
    if problems:
        raise ConfigKeyError(
            "\n".join(problems),
            hint=f"fix the key; a run config's keys are listed in {docs}#keys",
        )
    if legacy:
        # Never a fraction of the run the file asks for.
        problems = job_config_problems(raw, path=path, lines=lines)
        if problems:
            raise ConfigKeyError(
                "\n".join(problems),
                hint="change or drop each setting named; what a YAML job config "
                f"runs is in {docs}#yaml-job-configs",
            )
    return raw


def task_name_selected(name: str, include: set[str], exclude: set[str]) -> bool:
    """``--include`` / ``--exclude`` matching: exact names or fnmatch globs
    (``libero-10-*``, ``*-hard``). Exclude wins over include."""
    import fnmatch

    def hit(patterns: set[str]) -> bool:
        return any(
            name == p or (any(c in p for c in "*?[") and fnmatch.fnmatchcase(name, p))
            for p in patterns
        )

    if exclude and hit(exclude):
        return False
    return not include or hit(include)


def unmatched_selection(
    names: Iterable[str], include: set[str], exclude: set[str]
) -> list[str]:
    """One line per ``--include`` / ``--exclude`` pattern that matches none of
    *names*, with the closest name when there is one (DF-p3-eval-runner-1).

    A typo (``--include fix-gitt``) must never vanish silently: alone it
    selects nothing, and beside a valid name it quietly drops a task.
    """
    import difflib

    pool = sorted(set(names))
    lines: list[str] = []
    for flag, patterns in (("--include", include), ("--exclude", exclude)):
        for pattern in sorted(patterns):
            if any(task_name_selected(n, {pattern}, set()) for n in pool):
                continue
            close = difflib.get_close_matches(pattern, pool, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            lines.append(f"{flag} {pattern!r} matches no task{hint}")
    return lines


def empty_selection_message(
    source: str, include: set[str], exclude: set[str], names: Iterable[str]
) -> str:
    """The refusal every task source prints when include/exclude select nothing.

    *source* names where the tasks came from (``tasks_dir=tb2``,
    ``dataset=terminal-bench@2.0``); *names* are the tasks it offered.
    """
    detail = [source]
    if include:
        detail.append(f"include={sorted(include)}")
    if exclude:
        detail.append(f"exclude={sorted(exclude)}")
    unmatched = unmatched_selection(names, include, exclude)
    why = f" {'; '.join(unmatched)}." if unmatched else ""
    return (
        f"No tasks selected after include/exclude filtering ({', '.join(detail)})."
        f"{why} Refusing to publish an empty 0/0 summary."
    )


def sample_task_dirs(
    dirs: list[Path], n_tasks: int | None, seed: int | None, is_task=None
) -> list[Path]:
    """``--n-tasks N [--sample-seed S]``: the first N task dirs in sorted order, or a
    seeded random sample of N (reproducible: same seed, same tasks), kept sorted."""
    if not n_tasks or n_tasks <= 0:
        return dirs
    tasks = [d for d in dirs if is_task is None or is_task(d)]
    if len(tasks) <= n_tasks:
        return tasks
    if seed is None:
        return tasks[:n_tasks]
    import random

    return sorted(random.Random(seed).sample(tasks, n_tasks))


class MalformedTaskError(ValueError, UserError):
    """A single-task input whose ``task.md`` (or legacy ``task.toml``) exists
    but fails to parse (#3).

    Subclasses ``ValueError`` so the CLI's existing run-error handlers surface it
    as a clean red message + exit 1. The message names the offending file —
    silently treating a typo'd task.md as "not a task" would make the task vanish.
    """

    fault = "task"


# describe_exception's "SomeError: " opening.
_EXCEPTION_CLASS_PREFIX = re.compile(r"^[A-Z][A-Za-z0-9]*(?:Error|Exception): ")


def _unexpected_text(exc: BaseException) -> str:
    """The [ERR] line's words for a trial that raised instead of returning."""
    from benchflow._utils.disk_space import disk_full_text, is_disk_full

    if is_disk_full(exc):
        # The host's disk, which the summary files under disk_full; the file
        # it could not write names which disk (review L-A2).
        name = getattr(exc, "filename", None)
        return disk_full_text(Path(name) if name else None, job_file=False)
    return f"unexpected exception: {exc}"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Exponential backoff with
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
        category = retry_category(error, category or classify_error(error))
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

    def reruns_on_resume(
        self, error: str | None, *, category: str | None = None
    ) -> bool:
        """Whether a resumed job runs an unscored trial again.

        Everything ``should_retry`` retries, and a trial that ended on its
        login's usage limit: never retried within the run (the same login
        would only hit it again), but a resume is how a caller runs it again
        on another login or after the reset.
        """
        category = retry_category(error, category or classify_error(error))
        # A full disk is not retried within the run; once space is freed a
        # resume runs the trial again (benchflow._utils.disk_space).
        return category in (USAGE_LIMIT, DISK_FULL) or self.should_retry(
            error, category=category
        )

    def reruns_unjudged_solver(
        self,
        scoring: ScoringResult | None,
        error: str | None,
        *,
        category: str | None = None,
        on_resume: bool = False,
    ) -> bool:
        """Whether a rubric trial's solver runs again despite its scoring block.

        A rubric trial commits a scoring block even when its solver failed on
        the sandbox or transport and nothing was judged: a scoring error with
        no verifier reward. That is a retryable infrastructure failure like
        any other (#1059), not a verdict that pins the trial. ``on_resume``
        also runs again a solver that ended on a usage limit
        (:meth:`reruns_on_resume`).
        """
        again = self.reruns_on_resume if on_resume else self.should_retry
        return (
            scoring is not None
            and scoring.status == "error"
            and scoring.verifier_reward is None
            and again(error, category=category)
        )

    def should_retry_verifier_error(self, verifier_error: str | None) -> bool:
        """Check if a verifier error is infrastructure-retryable.

        A verifier timeout is not: test.sh past the task's verifier timeout
        runs past it again, and the whole trial (with a paid agent) would
        repeat for nothing (DF-p5-cli-craft-17). A verifier that never
        started is ``verifier_wedge:``, which is infrastructure and retried.
        """
        from benchflow.rollout._verifier_recovery import PRESERVED_SOLVER

        if (
            not self.retry_on_verifier_infra
            or (verifier_error and PRESERVED_SOLVER in verifier_error)
            # A guard refusal is decided by what the agent left; another
            # attempt would be a re-roll (V2-R1-A-1).
            or plugin_guard_error_message(verifier_error) is not None
        ):
            return False
        return classify_verifier_error(verifier_error) == VERIFIER_INFRA

    def reruns_verifier_error_on_resume(self, verifier_error: str | None) -> bool:
        """Whether a resumed job runs a scoreless verifier-errored trial again.

        Everything :meth:`should_retry_verifier_error` retries, and a reward
        lost to a full disk (a resume is how a job whose disk was freed gets
        scored). Never a verifier timeout: re-running the trial would give its
        agent a second attempt. A trial whose workspace was
        preserved gets its one verifier-only re-run through verifier recovery
        instead (``needs_verifier_recovery``), in the run or on resume.
        """
        from benchflow.rollout._verifier_recovery import PRESERVED_SOLVER

        if self.should_retry_verifier_error(verifier_error):
            return True
        return (
            self.retry_on_verifier_infra
            and not (verifier_error and PRESERVED_SOLVER in verifier_error)
            and classify_verifier_error(verifier_error) == DISK_FULL
        )

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
        category = retry_category(
            result.error, result.error_category or classify_error(result.error)
        )
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


def default_agent_warning(model: str | None) -> str:
    """What a run that names no agent says once before it starts.

    0.8 keeps running :data:`DEFAULT_AGENT` then, but no longer silently: a
    paid model run nobody asked for by name should be visible, and the job
    records it (``agent_defaulted`` in evaluation.json). ``model`` is the model
    the run resolved (``--model``, else :data:`DEFAULT_MODEL`).
    """
    return (
        f"No agent named: running {DEFAULT_AGENT} with {model or DEFAULT_MODEL}, "
        "BenchFlow 0.8's default. Name one with --agent (and --model); "
        "--agent oracle runs the task's own solution."
    )


# Job scheduling modes (architecture.md § "Lifecycles" — the Job lifecycle).
# - parallel-independent: the default — rollouts run concurrently, isolated.
# - sequential-shared: continual learning — rollouts run in order over one
#   persistent, versioned LearnerStore (capability 5).
JOB_MODES = ("parallel-independent", "sequential-shared")
DEFAULT_JOB_MODE = "parallel-independent"


# The job names Evaluation generates: a timestamp, and a -N suffix when the
# second is already taken (bench eval run --fresh).
_AUTO_JOB_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}(-\d+)?$")


def _is_external_job(folder: Path) -> bool:
    """A nonempty folder that carries no BenchFlow marker
    (:func:`_is_benchflow_job`): a folder BenchFlow did not make, never
    resumed into. The dot-entries BenchFlow writes, :data:`_OWN_DOT_ENTRY`,
    are ignored. Every other entry counts,
    including an empty subfolder."""
    if not folder.is_dir() or _is_benchflow_job(folder):
        return False
    try:
        for child in folder.iterdir():
            if _OWN_DOT_ENTRY.match(child.name):
                continue
            return True
    except OSError:
        return True
    return False


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_job_record(record: dict[str, Any]) -> bool:
    """Whether ``record`` has the keys of a 0.8 job record
    (:data:`_RECORD_KEYS`) with the types every 0.8 record has written them
    in: an int ``schema_version``, str ``tasks_dir`` and ``job_name``, and a
    ``config`` object (FIX-49 r3 L3)."""
    version = record.get("schema_version")
    return (
        all(k in record for k in _RECORD_KEYS)
        and isinstance(version, int)
        and not isinstance(version, bool)
        and isinstance(record.get("tasks_dir"), str)
        and isinstance(record.get("job_name"), str)
        and isinstance(record.get("config"), dict)
    )


# Dot-entries BenchFlow itself leaves in a job folder: the job lock, the
# local runtime's prepared marker, the kept 0.7 files and their temporary
# copy, the set-aside trials, the budget's spend records, and the temporary
# files of an atomic write (``.<name>.<random>``);
# every other dot-entry, including Finder's ``.DS_Store``, is an entry
# BenchFlow did not write (FIX-49 r4 L1).
_OWN_DOT_ENTRY = re.compile(
    r"^(\.evaluation\.lock|\.benchflow-prepared|\.pre-0\.8(\.tmp-[0-9a-f]+)?"
    r"|\.set-aside|\.budget-spend"
    r"|\.[^./][^/]*\.(json|jsonl)\.[^./]+)$"
)


def _is_benchflow_job(folder: Path) -> bool:
    """A folder carrying a marker BenchFlow writes as the job starts or
    later: its job record (an ``evaluation.json`` holding every key of
    :data:`_RECORD_KEYS`, written as a 0.8 job starts), or a trial BenchFlow
    started. A trial is BenchFlow's when the config.json it writes as it
    starts names the BenchFlow that ran it (:data:`_PY08_TRIAL_KEYS`) or has
    every key a 0.7 trial's has (:data:`_PY07_TRIAL_KEYS`), or when its
    result.json (or solver checkpoint) is a BenchFlow rollout result
    (``rollout_name``). ``bf.run(job_name=...)``, ``bf.run_batch`` and a 0.7
    job stopped before its end have no job record.

    A ``summary.json`` is no marker: other programs write one too, and a
    resume replaces it. A folder known only by its 0.7 trials' config.json
    is BenchFlow's only when its ``summary.json``, if it has one, holds every
    key a 0.7 summary has (:data:`_PY07_SUMMARY_KEYS`).

    The job lock (:data:`JOB_LOCK`) is no marker: the run asking takes it
    in the folder before this check, foreign folder or not."""
    record = _json_object(folder / EVALUATION_RECORD)
    if record is not None and _is_job_record(record):
        return True
    made_by_07 = False
    for trial in _trial_folders(folder):
        for name, any_of, all_of in (
            ("config.json", _PY08_TRIAL_KEYS, _PY07_TRIAL_KEYS),
            ("result.json", ("rollout_name",), ()),
            ("solver.json", ("rollout_name",), ()),
            ("solver-complete.json", ("rollout_name",), ()),
        ):
            try:
                data = json.loads((trial / name).read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if any(k in data for k in any_of):
                return True
            if all_of and all(k in data for k in all_of):
                made_by_07 = True
    if not made_by_07:
        return False
    if not (folder / "summary.json").exists():
        return True
    summary = _json_object(folder / "summary.json")
    return summary is not None and all(k in summary for k in _PY07_SUMMARY_KEYS)


def _is_job_folder(folder: Path) -> bool:
    """Whether ``folder`` is one job (not, say, a folder of jobs): an
    auto-generated name, a job record, or a trial folder inside it.

    An external job can use the same timestamps and `jobs/` directory.
    Resuming into one would count or move another runner's trials (review R2-1).
    """
    if _is_external_job(folder):
        return False
    if _AUTO_JOB_NAME.match(folder.name):
        return True
    if (folder / EVALUATION_RECORD).is_file() or (folder / "summary.json").is_file():
        return True
    return any(
        (child / "result.json").is_file() or (child / "config.json").is_file()
        for child in folder.iterdir()
        if child.is_dir()
    )


def _job_folders(jobs_dir: Path) -> list[Path]:
    """The job folders directly under ``jobs_dir`` (see :func:`_is_job_folder`),
    sorted by name, so the last one is the job an unnamed rerun resumes."""
    if not jobs_dir.is_dir():
        return []
    return sorted(
        d
        for d in jobs_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and _is_job_folder(d)
        # A job the local runtime prepared ahead is no job until taken.
        and not (d / _park.PREPARED_MARKER).exists()
    )


# The settings evaluation.json records (its "config") that change what a trial
# measures. A rerun that differs in any of them does not resume the job: with
# no job name it starts a new job, and on a named job the earlier trials are
# set aside (_check_resume_mismatch), so one score never blends two runs (P2;
# the Rust bench's dx-006). The tasks folder is compared too. Not compared, as
# they do not change what a finished trial measured: how many trials or
# reviews run at once or are prepared ahead (concurrency, build_concurrency,
# warm_pool, reviewer.concurrency),
# whether the agent installs from its cached layer or in the trial
# (fresh_install: the layer holds the same pinned install),
# which tasks are picked (include_tasks,
# exclude_tasks, n_tasks, sample_seed, and seeds, as each seed is its own task
# <task>--seed-<n>: a resume may run a wider selection),
# infrastructure retries (retry: a resume reruns those trials anyway), what is
# kept or checked afterwards (checkpoint_keep, freeze_workspace, integrity,
# usage_tracking), provenance (source_provenance, dataset_task_digests), how
# the install is done (fresh_install: the proxy and agent install in the trial
# instead of from a cached layer) and agent_defaulted. agent_env is compared by the entries evaluation.json
# records (secret-looking names and values are never stored, so a rotated key
# does not count as a change; recognized execution settings such as
# MAX_THINKING_TOKENS=32000 are stored and compared). Supported inline JSON
# configs also record a fingerprint after credential leaves are removed. In the
# other settings an env or headers mapping (config_override's sandbox env,
# MCP headers) leaves out the entries that look secret by name or shape, and
# a secret-looking string anywhere else (a prompt with "password: ...") is
# compared as a digest of itself, so changing it is a change
# (_without_secrets); a config_override change is named by path only. The reviewer is compared only for a job with a reviewed trial: it
# is recorded, with its defaults, for every job. verifier_sandbox is compared
# as the effective isolation: integrity=strict implies separate too.
_RESUME_SETTINGS = (
    "agent",
    "model",
    "reasoning_effort",
    "harness",
    "codex_apps_policy",
    "prompts",
    "extra_instruction",
    "timeout_multiplier",
    "agent_idle_timeout",
    "sandbox_setup_timeout",
    "budget",
    "agent_env",
    "environment",
    "sandbox_user",
    "sandbox_locked_paths",
    # The agent could reach the cloud metadata endpoints.
    "allow_cloud_metadata",
    "verifier_sandbox",
    "base_image_override",
    # Where the verifier runs and what crosses into it (0.7.11's flags).
    "separate_verifier",
    "separate_verifier_include",
    "context_root",
    "skip_agent_install",
    "environment_manifest",
    "config_override",
    "skill_mode",
    "skills_dir",
    "skill_creator_dir",
    "self_gen_no_internet",
    "job_mode",
    "loop_strategy",
    "checkpoints",
    "retry_from_checkpoint",
    "retry_prompt",
    "retry_resume_session",
    "reviewer",
    "pre_agent_hooks",
    "dataset_name",
    "dataset_version",
)

# Settings recorded as a mapping of BenchFlow's own fields: a field an older
# record lacks is unknown there, not changed (as for a whole setting), so an
# upgrade that adds one does not set every earlier trial aside.
_RESUME_SCHEMA_SETTINGS = (
    "reviewer",
    "environment_manifest",
    "budget",
    "loop_strategy",
)

# Files of a job folder that are its trials' (beside the trial folders): set
# aside with them when a rerun with other settings starts the job over.
_JOB_RESULT_FILES = (
    "summary.json",
    "learner_store.json",
    "results.jsonl",
    "results.errors.json",
    "verifiers.jsonl",
    "adp.jsonl",
)


def _trial_folders(job_dir: Path) -> list[Path]:
    """The job's own trial (attempt) folders: its children holding a trial's
    config.json, result.json or solver checkpoint (not set-aside ones)."""
    from benchflow._utils.result_paths import SET_ASIDE_DIR

    if not job_dir.is_dir():
        return []
    markers = ("config.json", "result.json", "solver.json", "solver-complete.json")
    return sorted(
        child
        for child in job_dir.iterdir()
        if child.is_dir()
        and child.name != SET_ASIDE_DIR
        and any((child / name).is_file() for name in markers)
    )


def _shown_setting(value: Any) -> str:
    """A recorded setting as a log line shows it (unset caps left out, cut at 80)."""
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if v is not None}
    return render_text(repr(value), 80)


# Mappings whose entries are free-form names a user picks, where a credential
# can sit under any name: a sandbox's or MCP server's env and headers, the
# reviewer's agent_env. Elsewhere a name is a schema field
# (environment_manifest.task_selection.key), never a reason to drop a value.
_FREE_FORM_MAPPINGS = ("env", "headers", "agent_env")
# What a secret-looking string outside those mappings is compared as.
_REDACTED_PREFIX = "<redacted sha256:"


def _redacted(text: str) -> str:
    """A stable stand-in for a secret-looking string: two different strings
    stay different, and no part of the text reaches a log line."""
    import hashlib

    return f"{_REDACTED_PREFIX}{hashlib.sha256(text.encode()).hexdigest()[:16]}>"


def _without_secrets(value: Any, free_form: bool = False) -> Any:
    """``value`` with the strings that look secret made safe to compare and
    show. Inside an env or headers mapping (free-form names) such an entry
    is left out: by its shape (a key's prefix, a bearer header) or its name
    (HF_TOKEN, Authorization), unless it is a recognized execution setting
    (MAX_THINKING_TOKENS=32000), so a rotated key in a
    sandbox env or MCP headers is no change. Anywhere else (a prompt that
    says "password: admin123", a list item, a reviewer or manifest field) a
    string shaped like a secret becomes a digest of itself
    (:func:`_redacted`): a change to it is still a change, and its value
    never reaches a log line."""
    from benchflow._utils.config_redaction import (
        _inline_config_fingerprint,
        _is_secret_env_value,
        _should_compare_env_entry,
    )

    if isinstance(value, dict):
        safe = {}
        for k, v in value.items():
            if free_form and (fingerprint := _inline_config_fingerprint(str(k), v)):
                safe[k] = fingerprint
            elif not (
                free_form
                and isinstance(v, (str, int, float, bool))
                and not _should_compare_env_entry(str(k), v)
            ):
                safe[k] = _without_secrets(v, str(k).lower() in _FREE_FORM_MAPPINGS)
        return safe
    if isinstance(value, list):
        return [_without_secrets(v) for v in value]
    if isinstance(value, str) and _is_secret_env_value("", value):
        return _redacted(value)
    return value


def _changed_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    """The dotted paths below which ``old`` and ``new`` differ (a list or a
    scalar is one leaf), e.g. ["sandbox.env.MY_FLAG"]."""
    if not isinstance(old, dict) and not isinstance(new, dict):
        return [prefix] if prefix and old != new else []
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    paths: list[str] = []
    for sub in sorted(old.keys() | new.keys(), key=str):
        if old.get(sub) != new.get(sub):
            paths.extend(_changed_paths(old.get(sub), new.get(sub), f"{prefix}{sub}."))
    return [p.rstrip(".") for p in paths]


def _setting_change(key: str, old: Any, new: Any) -> str:
    """One setting's difference, e.g. "model='a' (this run: 'b')"."""
    if key == "config_override":
        # Paths only: the overlay is free-form (sandbox env, MCP headers), so
        # any of its values may be a credential under a harmless name.
        paths = _changed_paths(old, new)
        return f"{key} " + (f"{', '.join(paths)} changed" if paths else "changed")
    if key.endswith("agent_env"):
        # Names only: values stay out of logs even when config.json keeps them.
        old_env = old if isinstance(old, dict) else {}
        new_env = new if isinstance(new, dict) else {}
        both = old_env.keys() & new_env.keys()
        groups = (
            ("changed", sorted(k for k in both if old_env[k] != new_env[k])),
            ("added", sorted(new_env.keys() - old_env.keys())),
            ("removed", sorted(old_env.keys() - new_env.keys())),
        )
        return f"{key} " + "; ".join(
            f"{', '.join(names)} {what}" for what, names in groups if names
        )
    if isinstance(old, dict) and isinstance(new, dict):
        return ", ".join(
            _setting_change(f"{key}.{sub}", old.get(sub), new.get(sub))
            for sub in sorted(old.keys() | new.keys())
            if old.get(sub) != new.get(sub)
        )
    shown_old, shown_new = _shown_setting(old), _shown_setting(new)
    redacted = _REDACTED_PREFIX in repr(old) + repr(new)
    shaped = _without_secrets([shown_old, shown_new]) != [shown_old, shown_new]
    if redacted or shaped:
        return f"{key} changed"  # a value shaped like a key stays out of logs
    return f"{key}={shown_old} (this run: {shown_new})"


def _budget_raised_after_stop(job_dir: Path, old: Any, new: Any) -> bool:
    """Whether ``new`` only raises or lifts the caps of a job its budget stopped.

    That rerun is how the trials a budget left unstarted get run (resuming
    with the same caps starts none of them), and a bigger cap does not change
    what a finished trial measured. A lowered or a new cap is a change.
    """
    try:
        summary = json.loads((job_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return False
    block = summary.get("budget") if isinstance(summary, dict) else None
    if not isinstance(block, dict) or not block.get("stopped"):
        return False
    if not isinstance(old, dict):
        return False
    if new is None:
        return True
    if not isinstance(new, dict):
        return False
    for cap in set(old) | set(new):
        was, now = old.get(cap), new.get(cap)
        if now is None:
            continue
        if not isinstance(was, int | float) or not isinstance(now, int | float):
            return False  # a new kind of cap, or a record this cannot read
        if now < was:
            return False
    return True


def _job_was_reviewed(job_dir: Path) -> bool:
    """Whether any trial of the job has a reviewer plan (its config.json's
    "review" block, written when the task's rubric review starts)."""
    for trial in _trial_folders(job_dir):
        try:
            recorded = json.loads((trial / "config.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(recorded, dict) and "review" in recorded:
            return True
    return False


def _task_digest_for(task_dir: Path, config: EvaluationConfig) -> str | None:
    """The ``task_digest`` a trial of ``task_dir`` records in its config.json.

    A dataset run's registry digest; else (``--tasks-dir`` / ``--source-repo``)
    one computed from the folder's files, so every trajectory stays
    attributable to the exact task content it ran, not just a folder name.
    """
    digest = (
        config.dataset_task_digests.get(task_dir.name) if config.dataset_name else None
    )
    if digest is None:
        from benchflow._utils.task_authoring import task_digest

        try:
            digest = task_digest(task_dir)
        except (OSError, ValueError, UnicodeError) as e:
            logger.debug("Could not compute task digest for %s: %s", task_dir, e)
    return digest


#: result.json's mark on a trial whose folder the job wrote after the trial
#: left none (it failed before its rollout folder existed): never reused as
#: finished, never compared as a trial that ran (review DX-e R2-M1, R2-L1).
NO_ROLLOUT_FOLDER = "no_rollout_folder"


def _left_no_rollout_folder(trial_dir: Path) -> bool:
    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(result, dict) and result.get(NO_ROLLOUT_FOLDER) is True


def _rollout_skill_mode(requested: str, *, learner: bool) -> str:
    """Learner skills replace the requested skill mode on scheduled trials."""
    return SKILL_MODE_WITH_SKILL if learner else requested


def _resolved_primary_settings(
    task_dir: Path, config: EvaluationConfig
) -> tuple[str, str | None, str | None]:
    """The primary agent, model and effort a rollout of ``task_dir`` records.

    Follow scheduling and self-gen transformations before resolving document
    scenes; never a stored absolute path. A scene's unset model or effort stays
    unset, as in RolloutConfig's primary_* properties.
    """
    from benchflow._utils.config import normalize_agent_name, normalize_reasoning_effort
    from benchflow.rollout._config import _task_document_scenes
    from benchflow.task.formats import materialize_task_dir

    # _run_sequential_shared supplies learner skills before every trial;
    # _rollout_config consequently uses with-skill, even for requested self-gen.
    skill_mode = _rollout_skill_mode(
        config.skill_mode, learner=config.job_mode == "sequential-shared"
    )
    if skill_mode == SKILL_MODE_SELF_GEN:
        # run_self_gen replaces the initial config with creator/solver scenes.
        # Use its primary role constructor, including its unset effort.
        from benchflow.self_gen import _creator_role

        role = _creator_role(config.agent, config.model)
    else:
        # A retained task outside the current selection still points at its
        # source. Resolve its format just as fresh rollouts do; raw draft-2
        # task.md documents cannot be read by the native scene parser.
        scenes = _task_document_scenes(
            materialize_task_dir(task_dir),
            prompts=config.prompts,
            skill_mode=skill_mode,
            agent=config.agent,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
        role = scenes[0].roles[0] if scenes and scenes[0].roles else None
    agent, model, effort = config.agent, config.model, config.reasoning_effort
    if role is not None:
        agent, model, effort = role.agent, role.model, role.reasoning_effort
    return normalize_agent_name(agent), model, normalize_reasoning_effort(effort)


def _resolved_agent(task_dir: Path, config: EvaluationConfig) -> str:
    """The agent a rollout records, including task-pinned scene overrides."""
    return _resolved_primary_settings(task_dir, config)[0]


def _task_image_ref(task_dir: Path, config: EvaluationConfig) -> str | None:
    """The prebuilt image a trial of ``task_dir`` starts from, or None (a build).

    Read after ``config_override``, as the trial's sandbox reads it.
    """
    from benchflow._utils.config_override import apply_config_override
    from benchflow.task import Task
    from benchflow.task.formats import materialize_task_dir

    task_config = Task(materialize_task_dir(task_dir)).config
    if config.config_override:
        task_config = apply_config_override(task_config, config.config_override)
    image = getattr(getattr(task_config, "sandbox", None), "docker_image", None)
    return image or None


def _warm_image_key(task_dir: Path, config: EvaluationConfig) -> tuple[Any, ...] | None:
    """What a warm-pool entry must still match when its trial takes it.

    Only immutable prebuilt images can be prepared without changing the
    requested configuration. Inspecting a mutable tag here would not bind
    the later launch: the tag can change and change back during setup.
    Builds and custom Compose graphs can also contain unbound image inputs;
    those tasks keep their normal cold start inside the trial's slot.
    """
    from benchflow.task.compose_env import docker_connection_args
    from benchflow.task.formats import materialize_task_dir

    if config.base_image_override or config.environment_manifest is not None:
        return None
    environment = materialize_task_dir(task_dir) / "environment"
    if any(
        (environment / name).exists()
        for name in ("docker-compose.yaml", "docker-compose.yml")
    ):
        return None
    image = _task_image_ref(task_dir, config)
    if (
        image is None
        or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", image) is None
    ):
        return None
    return (
        _task_digest_for(task_dir, config),
        image,
        tuple(docker_connection_args()),
        os.environ.get("DOCKER_DEFAULT_PLATFORM"),
    )


def _trial_settings_differ(
    trial_dir: Path,
    config: EvaluationConfig,
    tasks: dict[str, Path] | None = None,
    digests: dict[Path, str | None] | None = None,
    *,
    compare_legacy_settings: bool = False,
) -> str | None:
    """What one trial's own config.json records differently from ``config``.

    The effective agent, loop and task digest are checked for every job. Without
    evaluation.json, also compare the effective model, effort, backend and user
    where recorded (IU-R1-1). Missing historical fields are unknown, not proved
    equal; null is a recorded value. Other settings keep their existing
    evaluation.json policy: derived timeouts, filtered env and scene prompt
    presence do not reconstruct the old requested configuration.

    ``digests`` caches each task folder's digest across the trials of one check.
    """
    if _left_no_rollout_folder(trial_dir):
        # It ran nothing, so no setting of it can differ (R2-L1).
        return None
    try:
        recorded = json.loads((trial_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(recorded, dict) or recorded.get("purpose", "task") != "task":
        return None
    diffs = []
    agent = recorded.get("agent")
    current_agent = config.agent
    # config.json records primary_agent, which a task's scenes can pin.
    # Use the same scene compiler as a fresh rollout, including the explicit
    # prompt and self-gen overrides. Never resolve a stored absolute path.
    task_name = Path(str(recorded.get("task_path") or "")).name
    if not task_name:
        task_name = trial_dir.name.rsplit("__", 1)[0]
    task_dir = (tasks or {}).get(task_name)
    current_settings: dict[str, Any] = {
        "environment": config.environment,
        "sandbox_user": config.sandbox_user,
    }
    if task_dir is not None:
        current_agent, model, effort = _resolved_primary_settings(task_dir, config)
        current_settings.update(model=model, reasoning_effort=effort)
    if agent and agent != current_agent:
        diffs.append(
            _setting_change(
                "agent", _without_secrets(agent), _without_secrets(current_agent)
            )
        )
    if compare_legacy_settings:
        from benchflow._utils.config import (
            normalize_reasoning_effort,
            normalize_sandbox_user,
        )

        # Without a current task we cannot know its scene overrides. Compare
        # only the backend/user in that case, never the stored task_path.
        for key, new in current_settings.items():
            if key not in recorded:
                continue
            old = recorded[key]
            if key == "reasoning_effort":
                # An unrecognized historical value is not an unset effort.
                # Compare it as recorded, with the usual secret redaction.
                with contextlib.suppress(ValueError):
                    old = normalize_reasoning_effort(old)
            elif key == "sandbox_user":
                if isinstance(old, str):
                    old = normalize_sandbox_user(old)
                new = normalize_sandbox_user(new)
            old, new = _without_secrets(old), _without_secrets(new)
            if old != new:
                diffs.append(_setting_change(key, old, new))
    if agent:
        # A config.json from before loop strategies has no "loop": single-shot.
        loop = recorded.get("loop") or loop_block(None)
        current = loop_block(config.loop_strategy)
        if loop != current:
            diffs.append(f"loop={loop} (this run: {current})")
    recorded_digest = recorded.get("task_digest")
    if task_dir is not None and isinstance(recorded_digest, str) and recorded_digest:
        cache = digests if digests is not None else {}
        if task_dir not in cache:
            cache[task_dir] = _task_digest_for(task_dir, config)
        current_digest = cache[task_dir]
        if current_digest is not None and current_digest != recorded_digest:
            diffs.append(
                f"task {task_name!r} at task_digest {recorded_digest[:19]} (this "
                f"run: {current_digest[:19]}; the task's files changed since)"
            )
    return ", ".join(diffs) or None


def _resume_task_paths(
    tasks_root: Path | None, selected: list[Path] | None = None
) -> dict[str, Path]:
    """Resolve recorded task names under the current root, including unselected tasks."""
    tasks = {}
    if tasks_root is not None and tasks_root.is_dir():
        tasks = {p.name: p for p in tasks_root.iterdir() if p.is_dir()}
        tasks[tasks_root.name] = tasks_root
    tasks.update({p.name: p for p in selected or []})
    return tasks


def _without_unrecorded_env(
    new: Any, old: Any, old_keys: Any, fingerprints: dict[str, str]
) -> Any:
    """``new``'s env without the names the record lists but holds no value
    for (a value that looked secret, or one an older version did not store,
    such as MAX_THINKING_TOKENS=32000): unknown there, not changed.

    A supported inline config carries public execution policy as well as
    credentials. If an older record dropped it without a fingerprint, a
    supplied policy has no known identity to match and must rerun once.
    """
    if not isinstance(new, dict) or not isinstance(old, dict):
        return new
    unknown = (
        set(old_keys if isinstance(old_keys, list) else [])
        - set(old)
        - fingerprints.keys()
    )
    return {k: v for k, v in new.items() if k not in unknown}


def verifier_isolation_summary(
    results: dict[str, dict[str, Any]],
    *,
    task_dirs: dict[str, Path],
    tasks_dir: Path,
    config_override: dict | None = None,
) -> dict[str, Any]:
    """summary.json's ``verifier_isolation`` block.

    ``modes`` counts trials by where the verifier ran (``shared``,
    ``separate``; ``unknown`` for trials from before the record or that
    stopped before setup), ``sources`` by what decided it. ``run_flag_without_artifacts``
    names the tasks moved by ``--verifier-sandbox separate`` that declare no
    artifacts: their verifier gets only the workspace and ``/logs/artifacts``,
    so an answer kept anywhere else scores 0, which looks exactly like an
    agent failure.
    """
    from collections import Counter

    modes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    run_flag: set[str] = set()
    for name, result in results.items():
        mode = result.get("verifier_isolation") or "unknown"
        modes[str(mode)] += 1
        source = result.get("verifier_isolation_source")
        if source:
            sources[str(source)] += 1
        if source == "run-flag":
            run_flag.add(str(result.get("task_name") or name))
    without = []
    for name in sorted(run_flag):
        task_dir = task_dirs.get(name) or (tasks_dir / name)
        if not _declares_artifacts(task_dir, config_override):
            without.append(name)
    return {
        "modes": dict(sorted(modes.items())),
        "sources": dict(sorted(sources.items())),
        "run_flag_without_artifacts": without,
    }


def _declares_artifacts(task_dir: Path, config_override: dict | None) -> bool:
    """Whether the task (after the run's overlay) declares any artifact; a
    task that no longer loads counts as declaring none."""
    from benchflow.integrity.trial import _load_task_config

    config = _load_task_config(task_dir) if task_dir.is_dir() else None
    if config is None:
        return False
    if config_override:
        from benchflow._utils.config_override import apply_config_override

        with contextlib.suppress(Exception):
            config = apply_config_override(config, config_override)
    return bool(getattr(config, "artifacts", None))


def _log_verifier_isolation(isolation: dict[str, Any]) -> None:
    """The per-mode count once any trial ran separate, and the warning for
    run-flag tasks that declare no artifacts."""
    modes = isolation.get("modes") or {}
    if not modes.get("separate"):
        return
    logger.info(
        "Verifier: %s",
        ", ".join(f"{count} {mode}" for mode, count in modes.items()),
    )
    without = isolation.get("run_flag_without_artifacts") or []
    if without:
        logger.warning(
            "--verifier-sandbox separate moved %d task(s) that declare no "
            "artifacts: %s. Their verifier gets only the workspace and "
            "/logs/artifacts, so an answer kept elsewhere (a service, a "
            "database, a file outside the workspace) scores 0 like an agent "
            "failure; check each task's oracle under the switch before "
            "comparing these zeros with task-mode runs.",
            len(without),
            ", ".join(without),
        )


def _run_verifier_sandbox(record: dict[str, Any]) -> str:
    """``separate`` when the recorded run moved every verifier into the
    separate verifier sandbox (the switch, ``--separate-verifier`` or strict
    integrity), else ``task``."""
    if (
        record.get("verifier_sandbox") == "separate"
        or record.get("separate_verifier") is True
        or record.get("integrity") == "strict"
    ):
        return "separate"
    return "task"


def _env_comparison_record(record: dict[str, Any]) -> dict[str, Any]:
    """Public replay env plus separate inline-config identities, if recorded."""
    from benchflow._utils.config_redaction import _inline_env_fingerprints

    env = record.get("agent_env") or {}
    return {
        **env,
        **_inline_env_fingerprints(env),
        **(record.get("agent_env_fingerprints") or {}),
    }


def _job_record_differs(
    job_dir: Path, config: EvaluationConfig, tasks_dir: Path | None
) -> str | None:
    """What of ``config``/``tasks_dir`` the job's ``evaluation.json`` records
    differently, or None when it matches, is missing or records nothing.

    Every setting in :data:`_RESUME_SETTINGS` the record has is compared; one
    an older record lacks is not (its value then is unknown, not different).
    """
    try:
        record = json.loads((job_dir / EVALUATION_RECORD).read_text())
    except (OSError, ValueError):
        return None
    recorded = record.get("config") if isinstance(record, dict) else None
    if not isinstance(recorded, dict):
        return None
    current = _config_to_record(config)
    diffs = []
    for key in _RESUME_SETTINGS:
        if key == "verifier_sandbox":
            # Compared as where the run put the verifiers: strict integrity
            # implies the switch, and a record from before the switch (no
            # key) ran "task", or "separate" under strict.
            old_mode = _run_verifier_sandbox(recorded)
            new_mode = _run_verifier_sandbox(current)
            if old_mode != new_mode:
                diffs.append(_setting_change(key, old_mode, new_mode))
            continue
        if key not in recorded:
            continue
        old, new = recorded[key], current.get(key)
        if key == "agent_env":
            old, new = _env_comparison_record(recorded), _env_comparison_record(current)
        if key != "agent_env":
            # agent_env's record already holds only the values that do not
            # look secret; it is compared and shown by name below.
            old, new = _without_secrets(old), _without_secrets(new)
        if key == "agent_env":
            new = _without_unrecorded_env(
                new,
                old,
                recorded.get("agent_env_keys"),
                current["agent_env_fingerprints"],
            )
        if key == "sandbox_user" and isinstance(old, str):
            # A job recorded before root was one case ("root" or "0") ran
            # its agent as root, as None does now.
            from benchflow._utils.config import normalize_sandbox_user

            old = normalize_sandbox_user(old)
        if key == "reviewer" and isinstance(old, dict) and isinstance(new, dict):
            old = {**old, "agent_env": _env_comparison_record(old)}
            new = {**new, "agent_env": _env_comparison_record(new)}
            new = {
                **new,
                "agent_env": _without_unrecorded_env(
                    new.get("agent_env"),
                    old.get("agent_env"),
                    old.get("agent_env_keys"),
                    new["agent_env_fingerprints"],
                ),
            }
            # Like the agent's: names that look secret have no stored value,
            # so a resume cannot pass them back; their values are compared.
            # How many reviews run at once changes no score.
            dropped = ("agent_env_keys", "agent_env_fingerprints", "concurrency")
            old = {k: v for k, v in old.items() if k not in dropped}
            new = {k: v for k, v in new.items() if k not in dropped}
        if (
            key in _RESUME_SCHEMA_SETTINGS
            and isinstance(old, dict)
            and isinstance(new, dict)
        ):
            new = {k: v for k, v in new.items() if k in old}
        if old == new:
            continue
        if key == "reviewer" and not _job_was_reviewed(job_dir):
            # The reviewer is recorded for every job, with its defaults; it
            # changed nothing in a job none of whose trials was reviewed.
            continue
        if key == "budget" and _budget_raised_after_stop(job_dir, old, new):
            logger.info(
                "Job %s stopped at its budget; resuming it with the caps %s (were %s)",
                job_dir.name,
                _shown_setting(new),
                _shown_setting(old),
            )
            continue
        diffs.append(_setting_change(key, old, new))
    recorded_tasks = record.get("tasks_dir")
    if (
        tasks_dir is not None
        and isinstance(recorded_tasks, str)
        and Path(recorded_tasks).resolve() != Path(tasks_dir).resolve()
    ):
        diffs.append(f"tasks_dir={recorded_tasks!r} (this run: {str(tasks_dir)!r})")
    return ", ".join(diffs) or None


def _first_line(text: str) -> str:
    """The first non-blank line of a task's parse error, for a one-line
    message, without the task file's own path (the message names the task)."""
    line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    return re.sub(r"\S*/task\.(?:md|toml): ", "", line)


def _some_names(names: list[str], shown: int = 3) -> str:
    """``a, b, c and 4 more``: the first few of ``names`` for a log line."""
    more = len(names) - shown
    return ", ".join(names[:shown]) + (f" and {more} more" if more > 0 else "")


def _recorded_run_differs(
    job_dir: Path, config: EvaluationConfig | None, tasks_dir: Path | None
) -> str | None:
    """What of ``config``/``tasks_dir`` the job ran differently (e.g.
    "agent='oracle' (this run: 'nop')"), or None when it matches or records
    nothing to compare.

    The job's ``evaluation.json`` is compared setting by setting
    (:data:`_RESUME_SETTINGS`); a job from before that record is compared by
    what its trials' config.json files record (:func:`_trial_settings_differ`).
    """
    if config is None:
        return None
    if (job_dir / EVALUATION_RECORD).is_file():
        return _job_record_differs(job_dir, config, tasks_dir)
    tasks = _resume_task_paths(tasks_dir)
    digests: dict[Path, str | None] = {}
    for trial in _trial_folders(job_dir):
        differs = _trial_settings_differ(
            trial, config, tasks, digests, compare_legacy_settings=True
        )
        if differs:
            return differs
    return None


def _set_aside(
    job_dir: Path, trials: list[Path], reason: str, *, whole_job: bool
) -> Path:
    """Move ``trials`` (and, for ``whole_job``, the job's result files and a
    copy of its evaluation.json) into a new ``<job>/.set-aside/<UTC time>/``
    folder with a ``set-aside.json`` naming why. Nothing is deleted."""
    import shutil

    from benchflow._utils.result_paths import SET_ASIDE_DIR

    now = datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    dest, n = job_dir / SET_ASIDE_DIR / stamp, 1
    while dest.exists():
        n += 1
        dest = job_dir / SET_ASIDE_DIR / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    moved: list[str] = []
    try:
        if whole_job and (job_dir / EVALUATION_RECORD).is_file():
            shutil.copy2(job_dir / EVALUATION_RECORD, dest / EVALUATION_RECORD)
        for trial in trials:
            # The cumulative budget belongs to the same task as its attempts.
            # Partial reruns retain the other tasks' totals; full reruns move
            # even receipts from cancellation before a trial folder existed.
            budget_path = task_budget_path(job_dir, trial.name.rsplit("__", 1)[0])
            if not whole_job and budget_path.is_file():
                (dest / JOB_BUDGET_RECORDS).mkdir(exist_ok=True)
                budget_path.rename(dest / JOB_BUDGET_RECORDS / budget_path.name)
            trial.rename(dest / trial.name)
            moved.append(trial.name)
        if whole_job:
            if (job_dir / JOB_BUDGET_RECORDS).exists():
                (job_dir / JOB_BUDGET_RECORDS).rename(dest / JOB_BUDGET_RECORDS)
                moved.append(JOB_BUDGET_RECORDS)
            for name in _JOB_RESULT_FILES:
                if (job_dir / name).is_file():
                    (job_dir / name).rename(dest / name)
                    moved.append(name)
    except OSError as e:
        raise ResumeMismatchError(
            f"refusing to resume {job_dir}: its trials ran {reason}, and setting "
            f"them aside in {dest} failed ({e}). {len(moved)} item(s) were moved "
            "there; the rest are where they were. Start a new job instead: bench "
            "eval run --fresh (or another --job-name); in Python, a new job_name."
        ) from e
    (dest / "set-aside.json").write_text(
        json.dumps(
            {"set_aside_at": now.isoformat(), "reason": reason, "moved": moved},
            indent=2,
        )
        + "\n"
    )
    return dest


def _refuse_external_job(job_dir: Path) -> None:
    """Raise :class:`ResumeMismatchError` when ``job_dir`` is a folder
    BenchFlow did not make (:func:`_is_external_job`)."""
    if _is_external_job(job_dir):
        error = ResumeMismatchError(
            f"{job_dir} is a folder BenchFlow did not make; nothing in it moves"
        )
        error.hint = "pick another --job or another --jobs-dir"
        raise error


def _check_resume_mismatch(
    job_dir: Path,
    config: EvaluationConfig,
    before_set_aside: Callable[[], None] | None = None,
    task_dirs: list[Path] | None = None,
    tasks_root: Path | None = None,
) -> Path | None:
    """Set aside the job's earlier trials that ran with other settings.

    Runs before a job starts (or resumes) under its name. When the job's
    ``evaluation.json`` records another value of any setting in
    :data:`_RESUME_SETTINGS`, every trial folder, with the job's
    summary.json, learner store and results files, moves to
    ``<job>/.set-aside/<UTC time>/`` and the job starts over; otherwise each
    trial whose own config.json records another agent or loop strategy, or
    an earlier version of its task (another ``task_digest``, DF-13), is
    moved (all of them in a sequential-shared job, whose learner store holds
    every trial's lessons). Without evaluation.json, the recorded effective
    model, effort, backend and user are checked too. Readers skip that folder
    (:data:`benchflow._utils.result_paths.SET_ASIDE_DIR`) and nothing is
    deleted: a blended score belongs to neither run, and re-running over the
    old rows would lose them. A warning names the setting that changed.
    The tasks folder is not compared here, only when an unnamed run picks
    its job (:meth:`Evaluation._resolve_job_name`): naming the job is how a
    moved tasks folder is resumed (``Evaluation.resume(job_dir,
    tasks_dir=...)``). ``before_set_aside`` runs once trials are to move
    and before any does: the checks that can still refuse the run go there,
    so a refused run leaves the job's trials where its record expects them.
    Returns the folder the trials went to, or None when all of them matched.
    Raises :class:`ResumeMismatchError` when they cannot be moved.
    """
    # A named run never resumes into or moves trials out of an external job
    # (review R2-1).
    _refuse_external_job(job_dir)
    trials = _trial_folders(job_dir)
    if not trials:
        return None
    whole = _job_record_differs(job_dir, config, None)
    if whole:
        reason, chosen = whole, trials
    else:
        tasks = _resume_task_paths(tasks_root, task_dirs)
        digests: dict[Path, str | None] = {}
        legacy = not (job_dir / EVALUATION_RECORD).is_file()
        differing = {
            t: _trial_settings_differ(
                t, config, tasks, digests, compare_legacy_settings=legacy
            )
            for t in trials
        }
        chosen = [t for t, d in differing.items() if d]
        if not chosen:
            return None
        reason = "; ".join(sorted({d for d in differing.values() if d}))
        if config.job_mode == "sequential-shared":
            chosen = trials
    whole_job = chosen == trials
    if before_set_aside is not None:
        before_set_aside()
    dest = _set_aside(job_dir, chosen, reason, whole_job=whole_job)
    logger.warning(
        f"Setting aside {len(chosen)} earlier trial(s) of job {job_dir.name} in "
        f"{dest}: they ran {reason}. They are kept there and not counted; this "
        "run runs those tasks again with its own settings (bench run "
        "--fresh, or another --job, keeps two runs as two jobs)."
    )
    return dest


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
    from benchflow.agents.registry import is_scripted_agent

    if is_scripted_agent(agent):
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
    # "acp" (default) or "native": run the agent through its own CLI in
    # headless JSON mode instead of its ACP adapter (benchflow.native_harness).
    harness: str = "acp"
    environment: str = "docker"
    concurrency: int = 4
    build_concurrency: int | None = None
    # Docker: sandboxes prepared ahead of their trial's slot, beyond the
    # running trials (benchflow._warm_pool.warm_pool_size; None: its default).
    warm_pool: int | None = None
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
    # Every trial installs the model proxy and the agent itself, with no
    # cached layer (``--fresh-install``).
    fresh_install: bool = False
    # The agent may reach the cloud metadata endpoints under an open network
    # (``--allow-cloud-metadata``): no block, recorded per trial.
    allow_cloud_metadata: bool = False
    agent_idle_timeout: int | None = 600
    context_root: str | None = None
    base_image_override: str | None = None
    separate_verifier: bool = False
    separate_verifier_include: tuple[str, ...] = ()
    exclude_tasks: set[str] = field(default_factory=set)
    include_tasks: set[str] = field(default_factory=set)
    # Sample limit (``--n-tasks``): after include/exclude, run a random sample of
    # this many tasks, drawn with ``sample_seed`` (sorted order when seed is None).
    n_tasks: int | None = None
    sample_seed: int | None = None
    # Scale every task's agent time budget (``--timeout-multiplier``).
    timeout_multiplier: float | None = None
    # Text appended to every task prompt (``--extra-instruction``).
    extra_instruction: str | None = None
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
    # Reward integrity for every trial (benchflow.integrity): "off", "audit"
    # or "strict". Writes integrity/claim_verdict.json; never changes a reward.
    integrity: str = "off"
    # Where every trial's verifier runs (benchflow.task.verifier_sandbox):
    # "task" keeps each task's declared mode, "separate" moves every task's
    # verifier into the separate verifier sandbox and refuses, before the job
    # folder exists, a selection with tasks that cannot use it there. There is
    # no "shared"; integrity="strict" implies "separate".
    verifier_sandbox: str = "task"
    # Opt-in retry of a failed/timed-out trial from its last checkpoint
    # (benchflow.checkpoint_retry): "on-failure", "on-timeout" or both.
    retry_from_checkpoint: str | None = None
    retry_prompt: str | None = None
    retry_resume_session: bool = False
    # Hard per-job budget (benchflow.budget): stop launching and cancel
    # running trials once USD, sandbox-seconds or tokens reach a cap.
    budget: Budget | None = None
    # Seeded rollouts (`--seeds 0-4`): every task whose task format writes seeded
    # variants runs once per seed; summary.json then carries a `seeded` report
    # (pass@k, variance, reset reproducibility). See docs/tasks/embodied.md.
    seeds: list[int] | None = None
    # Python hooks run in every rollout's sandbox after it starts and before
    # the agent does (RolloutConfig.pre_agent_hooks): ``async def hook(sandbox)``.
    # Callables cannot be written to a config file; evaluation.json records
    # their names, and Evaluation.resume(job_dir, pre_agent_hooks=[...]) takes
    # them again.
    pre_agent_hooks: list[Callable[..., Any]] | None = None
    # The run named no agent and got DEFAULT_AGENT: `bench eval run`
    # warned once (default_agent_warning) and evaluation.json records it beside
    # the agent and model it ran.
    agent_defaulted: bool = False

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
        # A YAML or JSON config can give a number or a bool (MAX_THINKING_TOKENS:
        # 32000); the agent gets a string, and resume compares strings.
        self.agent_env = normalize_env_mapping(self.agent_env)
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
        from benchflow.agents.registry import is_operator_script, is_scripted_agent
        from benchflow.agents.scripts import normalize_script, validate_script_user

        if is_operator_script(self.agent):
            self.agent = normalize_script(self.agent)
            validate_script_user(self.sandbox_user)
            if self.agent_env or self.harness != "acp":
                raise ValueError(
                    "script: has no agent options or model harness; drop them or choose a model agent"
                )
            self.model = None
        self.reasoning_effort = normalize_reasoning_effort(self.reasoning_effort)
        from benchflow.native_harness.harnesses import (
            check_harness,
            normalize_harness,
        )

        self.harness = normalize_harness(self.harness)
        check_harness(self.harness, [self.agent])
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        if self.codex_apps_policy not in (None, "disabled", "inherit"):
            raise ValueError("codex_apps_policy must be disabled, inherit, or None")
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        self.checkpoint_policy()  # refuse a bad --checkpoints before any run
        self.retry_policy()  # and a bad --retry-from-checkpoint
        from benchflow.integrity.constants import normalize_integrity_mode

        self.integrity = normalize_integrity_mode(self.integrity)
        from benchflow.task.verifier_sandbox import normalize_verifier_sandbox

        self.verifier_sandbox = normalize_verifier_sandbox(self.verifier_sandbox)
        if self.separate_verifier:
            # --separate-verifier is the same setting as --verifier-sandbox
            # separate: one effective isolation for plan validation, the
            # rollout's resolver and the records (b127 PY8-MAIN-27).
            self.verifier_sandbox = "separate"
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
        if not is_scripted_agent(self.agent) and self.agent not in AGENTS:
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
    ``job_dir`` is where the job's artifacts and ``summary.json`` live. For
    the solve rate with its interval, unscored trials and cost,
    ``print(bf.load_job(result.job_dir))``.
    """

    job_name: str
    config: EvaluationConfig = field(repr=False)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    verifier_errored: int = 0
    # Unscored trials the plugin guard refused, failures in score_excl_errors
    # .
    guard_refused: int = 0
    elapsed_sec: float = 0.0
    memory_score: float | None = None
    memory_scores: dict[str, float] = field(default_factory=dict)
    task_failures: list[TaskFailure] = field(default_factory=list)
    # Mean reward over every trial, an unscored one counting as 0 (None when
    # there is none), over mean_reward_n trials; and over scored trials only,
    # BenchFlow 0.7's mean_reward (None when nothing scored), over
    # mean_reward_scored_only_n. The pass/fail counts describe hard gates,
    # while quality retains partial credit — a 0.3 rubric score and a flat 0
    # both print as FAIL without this.
    mean_reward: float | None = None
    mean_reward_scored_only: float | None = None
    mean_reward_n: int = 0
    mean_reward_scored_only_n: int = 0
    # Set when a scored reward is below 0 and some rollout is unscored: the
    # every-trial mean then counts an error above those rewards.
    mean_reward_caveat: str | None = None
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
    # Planned tasks this run never finished: a budget or a login's usage
    # limit stopped the job first (cancelled while running, or never
    # started). Not in ``total`` and not failures; ``planned`` and
    # ``score_of_plan`` count them (DF-p3-eval-runner-5).
    not_run: int = 0
    # The job was stopped (Ctrl-C, SIGTERM): the counts are over the trials
    # that finished first; ``not_run`` holds every other planned task, of
    # which ``interrupted`` had started (DFR-p3-11).
    stopped: bool = False
    interrupted: int = 0

    def __repr__(self) -> str:
        # The counts, not the config or every task's RolloutResult.
        job_dir = str(self.job_dir) if self.job_dir is not None else None
        return (
            f"EvaluationResult(job_name={self.job_name!r}, total={self.total}, "
            f"passed={self.passed}, failed={self.failed}, errored={self.errored}, "
            f"verifier_errored={self.verifier_errored}, score={self.score:.3f}, "
            f"mean_reward={self.mean_reward!r} (over "
            f"{self.mean_reward_n or self.total} trials, unscored as 0), "
            f"mean_reward_scored_only={self.mean_reward_scored_only!r} (over "
            f"{self.mean_reward_scored_only_n}), results=<{len(self.results)} "
            f"task(s)>, job_dir={job_dir!r})"
        )

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
    def planned(self) -> int:
        """The tasks the job set out to run: ``total`` plus ``not_run``."""
        return self.total + self.not_run

    @property
    def score_of_plan(self) -> float:
        """Pass rate over the planned tasks (a task not run counts as not
        passed); the same as ``score`` when every planned task ran."""
        return pass_rate(passed=self.passed, total=self.planned)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(
            passed=self.passed, failed=self.failed, refused=self.guard_refused
        )


EVALUATION_RECORD = "evaluation.json"
# What a job from before evaluation.json records as the BenchFlow that made
# it, and where resuming it keeps its own result files (DF-p4-upgrader-3).
PRE_08_VERSION = "before 0.8"
# A job whose maker no record names (L2, L3).
UNKNOWN_VERSION = "unknown"
PRE_08_DIR = ".pre-0.8"


def _benchflow_version() -> str:
    """The installed BenchFlow version, as `bench --version` prints it."""
    from benchflow import __version__

    return __version__


JOB_LOCK = ".evaluation.lock"


def _recorded_attempts(job_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Every rollout an earlier run of this job left: the result of each
    attempt that wrote one (``result.json``, else the solver snapshot a
    reviewed or interrupted attempt keeps), cancellation spend receipts,
    and how many legacy interrupted folders have no spend record."""
    from benchflow._utils.result_paths import iter_task_result_paths
    from benchflow.jobs import _interrupted

    if not job_dir.is_dir():
        return [], 0
    results: list[dict[str, Any]] = []
    seen: set[Path] = set()
    recorded: set[Path] = set()
    # Evaluation attempts live directly below the job. A cancelled parent
    # already includes nested verifier intervals; child results/artifacts must
    # not become additional attempts merely because it has no result.json.
    cancelled = {p.parent for p in job_dir.glob(f"*/{CANCELLED_BUDGET_RECORD}")}
    for path in iter_task_result_paths(job_dir):
        if any(parent in cancelled for parent in path.parent.parents):
            continue
        seen.add(path.parent)
        data = None
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("purpose", "task") == "task":
            row = with_cancelled_budget(path.parent, data) or data
            row["_budget_task_name"] = (
                data.get("task_name") or path.parent.name.rsplit("__", 1)[0]
            )
            results.append(row)
            recorded.add(path.parent)
    for name in ("solver.json", "solver-complete.json"):
        for path in sorted(job_dir.glob(f"*/{name}")):
            if path.parent in seen:
                continue
            seen.add(path.parent)
            data = None
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(path.read_text())
            if isinstance(data, dict):
                row = with_cancelled_budget(path.parent, data) or data
                row["_budget_task_name"] = (
                    data.get("task_name") or path.parent.name.rsplit("__", 1)[0]
                )
                results.append(row)
                recorded.add(path.parent)
    # Discovery is not accounting: an unreadable result/checkpoint must not
    # hide the attempt's independent, finalized cancellation receipt.
    for folder in sorted(cancelled - recorded):
        receipt = with_cancelled_budget(folder)
        if receipt is not None:
            receipt["_budget_task_name"] = folder.name.rsplit("__", 1)[0]
            results.append(receipt)
            seen.add(folder)
    interrupted = [p for p in _interrupted(job_dir) if p not in seen]
    without_receipt = 0
    for folder in interrupted:
        if any(parent in cancelled for parent in folder.parents):
            continue
        receipt = with_cancelled_budget(folder)
        if receipt is None:
            without_receipt += 1
        else:
            receipt["_budget_task_name"] = folder.name.rsplit("__", 1)[0]
            results.append(receipt)
    return results, without_receipt


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


def _reserve_new_job(jobs_dir: Path) -> str:
    """A new timestamp job name under ``jobs_dir`` whose folder this call made.

    The folder is created without ``exist_ok``, so two runs that start in the
    same second never take the same new job: the second gets ``<stamp>-2``.
    """
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    name, n = stamp, 1
    while True:
        try:
            (jobs_dir / name).mkdir()
        except FileExistsError:
            n += 1
            name = f"{stamp}-{n}"
            continue
        return name


def _acquire_job_lock(job_dir: Path) -> JobLock:
    """Serialize evaluation and independent scoring against the same job."""
    import socket

    job_dir.mkdir(parents=True, exist_ok=True)
    lock = job_dir / JOB_LOCK
    ownership = JobLock.acquire(lock)
    me = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": datetime.now().isoformat(),
    }
    try:
        if lock.exists():
            try:
                holder = json.loads(lock.read_text())
            except (OSError, ValueError):
                holder = {}
            if (
                not isinstance(holder, dict)
                or not isinstance(holder.get("pid"), int)
                or not holder.get("host")
            ):
                raise RuntimeError(
                    f"Job {job_dir} has an unreadable lock; confirm the old run "
                    f"has stopped before removing {lock}"
                )
            pid, host = holder.get("pid"), holder.get("host")
            # JobLockedError (a RuntimeError): another run holds this job, so
            # an unnamed run starts a new job instead (Evaluation._lock_auto_job).
            if host and host != me["host"]:
                raise JobLockedError(
                    f"Job {job_dir} is already running (or crashed) on host "
                    f"{host} (pid {pid}); delete {lock} if that run is gone."
                ) from None
            if isinstance(pid, int) and _pid_alive(pid):
                raise JobLockedError(
                    f"Job {job_dir} is already running in process {pid} "
                    f"(started {holder.get('started_at', '?')}); wait for it "
                    "to finish or stop it before resuming."
                ) from None
            logger.warning(
                "Taking over a stale job lock from process %s, which is gone: %s",
                pid,
                lock,
            )
        with lock.open("w") as handle:
            json.dump(me, handle)
        return ownership
    except BaseException:
        ownership.close()
        raise


def _config_to_record(config: EvaluationConfig) -> dict[str, Any]:
    """Serialize a config for ``evaluation.json``.

    agent_env is recorded with every name (``agent_env_keys``) and only the
    values that do not look secret (``agent_env``), which a resume compares
    and reuses: the ones a trial's config.json records, and recognized numeric
    execution settings (MAX_THINKING_TOKENS=32000). Supported inline configs
    record a separate fingerprint with credential leaves removed; this is
    comparison metadata and is never replayed. The reviewer's env likewise.
    """
    import dataclasses

    from benchflow._utils.config_redaction import (
        _inline_env_fingerprints,
        _should_compare_env_entry,
    )

    record: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.name == "agent_env":
            record["agent_env_keys"] = sorted(value)
            # As strings: a value set after construction may still be a
            # YAML number, and the record must compare equal to its string.
            record["agent_env"] = {
                k: env_value_text(v)
                for k, v in value.items()
                if _should_compare_env_entry(k, v)
            }
            record["agent_env_fingerprints"] = _inline_env_fingerprints(value)
        elif f.name == "retry":
            retry = dataclasses.asdict(value)
            retry["exclude_categories"] = sorted(retry["exclude_categories"])
            record["retry"] = retry
        elif f.name == "reviewer":
            record["reviewer"] = value.to_config_artifact()
            record["reviewer"]["agent_env"] = {
                k: v
                for k, v in value.agent_env.items()
                if _should_compare_env_entry(k, v)
            }
            record["reviewer"]["agent_env_fingerprints"] = _inline_env_fingerprints(
                value.agent_env
            )
        elif f.name == "usage_tracking":
            record["usage_tracking"] = value.to_mapping()
        elif f.name == "environment_manifest":
            record[f.name] = None if value is None else value.model_dump(mode="json")
        elif f.name == "loop_strategy":
            record[f.name] = None if value is None else value.to_mapping()
        elif f.name == "budget":
            record[f.name] = None if value is None else value.to_dict()
        elif f.name == "pre_agent_hooks":
            record[f.name] = (
                None if not value else [_callable_name(hook) for hook in value]
            )
        elif isinstance(value, set):
            record[f.name] = sorted(value)
        else:
            record[f.name] = value
    return json.loads(json.dumps(record, default=str))


def _callable_name(hook: Any) -> str:
    """``module.qualname`` of a hook, for the record."""
    name = getattr(hook, "__qualname__", None) or type(hook).__qualname__
    return f"{getattr(hook, '__module__', '?')}.{name}"


def _config_from_record(
    raw: dict[str, Any], agent_env: dict[str, str] | None
) -> EvaluationConfig:
    """Rebuild an EvaluationConfig from ``_config_to_record`` output."""
    import dataclasses

    known = {f.name for f in dataclasses.fields(EvaluationConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    kwargs.pop("pre_agent_hooks", None)  # names only; see Evaluation.resume
    # The values the record keeps (none that look secret), then the caller's.
    recorded_env = raw.get("agent_env")
    env = {
        **(
            {str(k): str(v) for k, v in recorded_env.items()}
            if isinstance(recorded_env, dict)
            else {}
        ),
        **(agent_env or {}),
    }
    missing = sorted(set(raw.get("agent_env_keys") or []) - set(env))
    if missing:
        logger.warning(
            "Resuming without agent_env %s: their values are not stored in %s. "
            "Pass agent_env={...} to Evaluation.resume() if the agent needs them.",
            ", ".join(missing),
            EVALUATION_RECORD,
        )
    kwargs["agent_env"] = env
    kwargs["retry"] = RetryConfig.from_mapping(raw.get("retry"))
    reviewer = dict(raw.get("reviewer") or {})
    reviewer.pop("agent_env_keys", None)
    reviewer.pop("agent_env_fingerprints", None)
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
            tasks_dir=resolve_source("terminal-bench-2"),
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
    def _resolve_job_name(
        jobs_dir: Path,
        config: EvaluationConfig | None = None,
        tasks_dir: Path | None = None,
    ) -> str:
        """Pick a job_name when none was explicitly provided.

        The latest job folder under ``jobs_dir`` (alphabetically last) is
        reused, so a second ``Evaluation.run()`` (or plain ``bench eval
        run``) resumes into the same directory instead of creating an orphan.
        A job folder has an auto-generated (timestamp) name, a job record
        (``evaluation.json``, ``summary.json``) or trial folders; a folder
        that is none of these, such as a folder of jobs (``jobs/smoke/``), is
        never picked. When that job ran with other settings than ``config``
        and ``tasks_dir`` (any of :data:`_RESUME_SETTINGS` its
        ``evaluation.json`` records, or the tasks folder), a new job starts
        instead (resuming would blend two runs' results), with a warning
        naming the setting. Without such a folder: a fresh timestamp.
        ``run()`` checks again with the settings it finally runs.

        auto-generated job_name must be stable across
        resume calls.
        """
        fresh = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        job_dirs = _job_folders(jobs_dir)
        if not job_dirs:
            return fresh
        latest = job_dirs[-1]
        differs = _recorded_run_differs(latest, config, tasks_dir)
        if differs:
            return Evaluation._new_job_instead(jobs_dir, latest, differs)
        # run() says it resumes this job once it has also compared the tasks
        # the run selects (_settle_auto_job_name).
        return latest.name

    @staticmethod
    def _new_job_instead(jobs_dir: Path, latest: Path, differs: str) -> str:
        """A fresh job name, logged with why ``latest`` is not resumed."""
        fresh = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        name, n = fresh, 1
        while (jobs_dir / name).exists():
            n += 1
            name = f"{fresh}-{n}"
        logger.warning(
            f"Starting a new job {name}: the latest job in {jobs_dir} "
            f"({latest.name}) ran {differs}. Its results stay in their own job; "
            f"bench eval resume {command_arg(latest, '<job dir>')} "
            "finishes it with its own settings."
        )
        return name

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
        # A saved job may be resumed from a different working directory. Bind
        # relative input paths when selected, before writing evaluation.json.
        self._tasks_dir = resolve_task_collection_root(tasks_dir).absolute()
        self._materialized_task_sources: dict[Path, Path] = {}
        self._jobs_dir = Path(jobs_dir)
        # A copy: ``budget=`` and the task source's provenance are set on the
        # job's own config, never on the caller's object, which may be reused
        # for another Evaluation.
        self._config = copy.copy(config) if config is not None else EvaluationConfig()
        if budget is not None:
            # A hard per-job cap (benchflow.budget); same as config.budget.
            self._config.budget = Budget.coerce(budget)
        self._budget_guard: BudgetGuard | None = None
        self._usage_stop = UsageLimitStop()
        self._selection_warned = False
        # Set by resume(): it runs the selection the job recorded.
        self._resuming = False
        # agent_env names a loaded config declared without values; to_dict
        # keeps listing them so a second save does not forget them.
        self._declared_env_keys: list[str] = []
        if self._config.source_provenance is None:
            from benchflow._utils.hf_datasets import load_source_sidecar

            self._config.source_provenance = load_source_sidecar(self._tasks_dir)
        self._job_name = job_name or self._resolve_job_name(
            self._jobs_dir, self._config, self._tasks_dir
        )
        # The name _resolve_job_name picked (None for a named job): run()
        # checks it again against the settings it finally runs, which a caller
        # (the CLI's --config path) may still change after construction.
        self._auto_job_name: str | None = None if job_name else self._job_name
        # Set by the YAML job config loader: the task names a `tasks:` list
        # fixes, and whether the oracle runs because the file names no agent
        # . The CLI reads them once its flags are applied.
        self._config_task_list: list[str] | None = None
        self._default_oracle = False
        self._config_explicit_timeout_phases: set[str] | None = None
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
        # The job's warm pool while _run_parallel_independent runs (Docker
        # jobs with more trials than --concurrency), else None.
        self._warm_pool: WarmPool | None = None

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
        if cfg.pre_agent_hooks:
            raise ValueError(
                "pre_agent_hooks are Python callables and cannot be written to a "
                "config file; save the config without them and pass them in Python"
            )
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
            "harness": cfg.harness,
            "environment": cfg.environment,
            "concurrency": cfg.concurrency,
            "build_concurrency": cfg.build_concurrency,
            "warm_pool": cfg.warm_pool,
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
            "fresh_install": cfg.fresh_install,
            "allow_cloud_metadata": cfg.allow_cloud_metadata,
            "agent_idle_timeout_sec": cfg.agent_idle_timeout,
            "context_root": cfg.context_root,
            "base_image_override": cfg.base_image_override,
            "separate_verifier": cfg.separate_verifier,
            "separate_verifier_include": list(cfg.separate_verifier_include),
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
            "integrity": cfg.integrity,
            "verifier_sandbox": effective_verifier_sandbox(
                cfg.verifier_sandbox, cfg.separate_verifier
            ),
            "retry_from_checkpoint": cfg.retry_from_checkpoint,
            "retry_prompt": cfg.retry_prompt,
            "retry_resume_session": cfg.retry_resume_session,
            "budget": None if cfg.budget is None else cfg.budget.to_dict(),
            "seeds": cfg.seeds,
            "n_tasks": cfg.n_tasks,
            "sample_seed": cfg.sample_seed,
            "timeout_multiplier": cfg.timeout_multiplier,
            "extra_instruction": cfg.extra_instruction,
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
    def from_dict(
        cls, raw: dict[str, Any], *, warn_default_agent: bool = True, **kwargs: Any
    ) -> Evaluation:
        """Build an Evaluation from a native config mapping (``to_dict``'s shape)."""
        return cls._from_native_yaml(
            dict(raw), warn_default_agent=warn_default_agent, **kwargs
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        warn_default_agent: bool = True,
        job_cell: int | None = None,
        **kwargs: Any,
    ) -> Evaluation:
        """Create a Job from a YAML config file.

        Supports both benchflow-native and legacy YAML formats.

        A native config with no ``agent`` key runs :data:`DEFAULT_AGENT` and
        records ``agent_defaulted``; it logs :func:`default_agent_warning`
        unless ``warn_default_agent`` is false (``bench eval run --config``
        passes false and warns itself once its --agent/--model overrides are
        applied).

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

        Legacy format (a YAML job config, agents + datasets style):
            jobs_dir: jobs
            n_attempts: 1
            n_concurrent_trials: 4
            environment:
              type: docker
            agents:
              - name: claude-agent-acp
                model_name: anthropic/claude-haiku-4-5-20251001
                env:
                  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
            datasets:
              - path: path/to/tasks

        A YAML job config runs every agent with every dataset, each
        ``n_attempts`` times. One Evaluation is one of
        those jobs: a file of several is refused here unless ``job_cell``
        picks one (``benchflow.job_config.job_plan(path).cells``
        lists them, and ``bench run --config`` runs them all).
        """
        path = Path(path)
        text = path.read_text()
        from benchflow._utils.config_keys import parse_run_config

        raw = parse_run_config(text, path)

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

        # Detect format: YAML job configs use "agents" + "datasets", or a
        # "tasks" list;
        # benchflow uses "agent".
        from benchflow.job_config import is_job_config

        legacy = is_job_config(raw)
        raw = check_run_config_keys(raw, path=path, text=text, legacy=legacy)
        if legacy:
            return cls._from_legacy_yaml(
                raw,
                warn_default_agent=warn_default_agent,
                path=path,
                job_cell=job_cell,
                **kwargs,
            )
        if job_cell is not None:
            raise ValueError(f"{path} is not a YAML job config; it has no cells")
        return cls._from_native_yaml(
            raw, warn_default_agent=warn_default_agent, **kwargs
        )

    @classmethod
    def _from_native_yaml(
        cls, raw: dict, *, warn_default_agent: bool = True, **kwargs
    ) -> Evaluation:
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
            harness=raw.get("harness", "acp"),
            environment=raw.get("environment", "docker"),
            concurrency=raw.get("concurrency", 4),
            build_concurrency=raw.get("build_concurrency"),
            warm_pool=raw.get("warm_pool"),
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
            fresh_install=bool(raw.get("fresh_install", False)),
            allow_cloud_metadata=bool(raw.get("allow_cloud_metadata", False)),
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
            context_root=raw.get("context_root"),
            base_image_override=raw.get("base_image_override"),
            separate_verifier=bool(raw.get("separate_verifier", False)),
            separate_verifier_include=tuple(raw.get("separate_verifier_include") or ()),
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
            integrity=raw.get("integrity") or "off",
            verifier_sandbox=raw.get("verifier_sandbox", "task"),
            retry_from_checkpoint=raw.get("retry_from_checkpoint"),
            retry_prompt=raw.get("retry_prompt"),
            retry_resume_session=bool(raw.get("retry_resume_session", False)),
            budget=Budget.coerce(raw.get("budget")),
            seeds=_seeds_from_config(raw.get("seeds")),
            n_tasks=raw.get("n_tasks"),
            sample_seed=raw.get("sample_seed"),
            timeout_multiplier=raw.get("timeout_multiplier"),
            extra_instruction=raw.get("extra_instruction"),
            agent_defaulted="agent" not in raw,
        )
        if config.agent_defaulted and warn_default_agent:
            logger.warning(default_agent_warning(config.model))
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
    def _from_legacy_yaml(
        cls,
        raw: dict,
        *,
        warn_default_agent: bool = True,
        path: Path | None = None,
        job_cell: int | None = None,
        **kwargs,
    ) -> Evaluation:
        """Parse a YAML job config (agents + datasets style): one cell of
        it, an agent with a dataset, run once."""
        from benchflow.job_config import (
            DATASET_REMOTE_KEYS,
            agent_timeout_multiplier,
            cell_raw,
            extra_instruction,
            phase_timeout_multipliers,
            plan_from_raw,
        )
        from benchflow.job_config import (
            agent_env as job_agent_env,
        )

        where = f"{path}: " if path is not None else ""
        plan = plan_from_raw(raw)
        if job_cell is None:
            if plan.is_matrix:
                # One Evaluation is one job; never the first of several.
                raise JobMatrixError(
                    f"{where}this YAML job config runs {len(plan.cells)} job(s) "
                    f"({plan.describe()}), and an Evaluation is one of them",
                    hint="run them all with bench run --config "
                    f"{command_arg(path, '<config>') if path else '<config>'}; "
                    "from Python, pass "
                    "job_cell=<n> for one of job_plan(path).cells",
                )
            job_cell = 0
        if not 0 <= job_cell < len(plan.cells):
            raise ValueError(
                f"{where}job_cell {job_cell}: the config has {len(plan.cells)} cell(s)"
            )
        cell = plan.cells[job_cell]
        raw = cell_raw(raw, cell)
        # A YAML job config with no `agents:` or an empty list runs the oracle,
        # and never starts a
        # model. An entry without a name runs the default
        # agent, recorded and warned about as in the native format.
        default_oracle = not raw.get("agents")
        agents = [{"name": "oracle"}] if default_oracle else raw["agents"]
        agent_cfg = agents[0]
        agent_defaulted = "name" not in agent_cfg
        agent_name = agent_cfg.get("name", DEFAULT_AGENT)

        # Model — keep provider prefix intact for downstream resolution
        model = effective_model(agent_name, agent_cfg.get("model_name") or None)

        # Environment
        env_cfg = raw.get("environment", {})
        environment = env_cfg.get("type", "docker")

        # Only the agent receives this env; job_config_problems refuses the
        # sandbox-wide environment.env setting.
        agent_env = job_agent_env(agent_cfg, where, cell.agent_index)

        # Tasks: the cell's dataset folder, or its explicit `tasks:` list:
        # one task folder, or tasks that share
        # a folder, run as that folder with an include list.
        dataset_filter: dict[str, Any] = {}
        datasets = raw.get("datasets") or []
        config_tasks = raw.get("tasks") or []
        task_names: set[str] = set()
        config_task_list: list[str] | None = None
        if datasets and config_tasks:
            raise ValueError(
                f"{where}datasets and tasks are both set; BenchFlow runs one "
                "folder of tasks, so keep one of them"
            )
        if config_tasks:
            for entry in config_tasks:
                if isinstance(entry, dict) and (
                    entry.get("git_url") or entry.get("git_commit_id")
                ):
                    # A pinned version BenchFlow does not fetch; a local folder
                    # of that name may be another one (review L-R3-1).
                    raise ValueError(
                        f"{where}tasks: {entry.get('path')!r} is pinned to "
                        f"{entry.get('git_url') or 'a git repository'} at "
                        f"{entry.get('git_commit_id') or 'its default branch'}; "
                        "BenchFlow does not fetch git tasks from job configs. Clone the "
                        "repository at that commit and run the task folder "
                        "(bench run <folder>)"
                    )
            if any(raw.get(key) for key in _INCLUDE_KEYS + _EXCLUDE_KEYS):
                # The list names the tasks; a filter would widen or narrow it
                # (review L-R3-2).
                raise ValueError(
                    f"{where}tasks: lists the tasks to run, and an include or "
                    "exclude list in the same file would change it; keep one"
                )
            paths = [
                Path(t["path"])
                for t in config_tasks
                if isinstance(t, dict) and t.get("path")
            ]
            if len(paths) != len(config_tasks):
                raise ValueError(
                    f"{where}tasks: every entry needs a path (a registry task "
                    "runs with --dataset)"
                )
            config_task_list = sorted(p.name for p in paths)
            parents = {p.parent for p in paths}
            if len(paths) == 1:
                tasks_dir = paths[0]
            elif len(parents) == 1:
                tasks_dir = parents.pop()
                # By exact name: a folder named with [, * or ? is not a glob.
                task_names = {glob.escape(p.name) for p in paths}
            else:
                raise ValueError(
                    f"{where}tasks: these tasks are in different folders; "
                    "BenchFlow runs one folder of tasks"
                )
        else:
            dataset = datasets[0] if datasets else {}
            at = f"datasets[{cell.dataset_index or 0}]"
            if datasets and not dataset.get("path"):
                raise ValueError(
                    f"{where}{at} has no path (a registry dataset runs "
                    "with --dataset <name>@<version>)"
                )
            remote = [
                key for key in DATASET_REMOTE_KEYS if dataset.get(key) is not None
            ]
            if remote:
                raise ValueError(
                    f"{where}{at}.{remote[0]}: names a remote dataset, and "
                    "BenchFlow runs a local folder (path); clone it and give "
                    "its folder as path, or run it with --dataset <name>@<version>"
                )
            tasks_dir = Path(dataset.get("path", "tasks"))
            # Per-dataset selection: globs, then the first n_tasks.
            dataset_filter = {
                "include": set(dataset.get("task_names") or ()),
                "exclude": set(dataset.get("exclude_task_names") or ()),
                "n_tasks": dataset.get("n_tasks"),
            }
            if (dataset_filter["include"] or dataset_filter["exclude"]) and any(
                raw.get(key) for key in _INCLUDE_KEYS + _EXCLUDE_KEYS
            ):
                raise ValueError(
                    f"{where}{at} selects its tasks (task_names, "
                    "exclude_task_names), and an include or exclude list in the "
                    "same file would change that; keep one"
                )

        # Concurrency: n_concurrent_trials is a top-level job setting,
        # and it wins over the older orchestrator block (review L-R2-4).
        orch = raw.get("orchestrator", {})
        top = raw.get("n_concurrent_trials")
        nested = orch.get("n_concurrent_trials")
        if top is not None and nested is not None and top != nested:
            logger.warning(
                f"{where}n_concurrent_trials: {top} at the top level wins over "
                f"orchestrator.n_concurrent_trials: {nested}"
            )
        concurrency = top if top is not None else (nested if nested is not None else 4)
        # The agent phase's cap stays under the job's cap.
        if agent_cfg.get("n_concurrent") is not None:
            concurrency = min(int(concurrency), int(agent_cfg["n_concurrent"]))

        jobs_dir = Path(raw.get("jobs_dir", "jobs"))
        # The requested job name applies only when no folder of that name
        # exists or the folder is a BenchFlow job: an external job folder
        # (`bench run --config jobs/<job>/config.json`) is never resumed into
        # (review R2-1). A job_name given to from_yaml wins.
        name = raw.get("job_name")
        if name and kwargs.get("job_name") is None:
            folder = jobs_dir / str(name)
            if not folder.exists() or _is_benchflow_job(folder):
                kwargs["job_name"] = str(name)
            else:
                what = (
                    "a folder BenchFlow did not make"
                    if _is_external_job(folder)
                    else "not a BenchFlow job"
                )
                logger.warning(
                    f"{where}job_name {str(name)!r}: {folder} is {what}; "
                    "this run starts a new job instead"
                )
        # n_attempts is a number of trials (the cells' trial-NN jobs), never
        # A YAML job config never retries a trial.

        # Skills dir (shared with benchflow-native format)
        skills_dir_raw = raw.get("skills_dir")
        skills_dir = str(Path(skills_dir_raw)) if skills_dir_raw else None
        sandbox_user = raw.get("sandbox_user", "agent")
        sandbox_locked_paths = raw.get("sandbox_locked_paths")
        sandbox_setup_timeout = raw.get("sandbox_setup_timeout", 120)

        # Map legacy include/exclude task filters. Accept both singular and
        # plural spellings ("include"/"includes", "exclude"/"excludes") so
        # ported configs do not silently lose their filtering (#500).
        include: set[str] = set(task_names) | dataset_filter.get("include", set())
        for key in _INCLUDE_KEYS:
            values = raw.get(key)
            if values:
                include.update(values)
        exclude: set[str] = set(dataset_filter.get("exclude", set()))
        for key in _EXCLUDE_KEYS:
            values = raw.get(key)
            if values:
                exclude.update(values)

        config = EvaluationConfig(
            agent=agent_name,
            model=model,
            reasoning_effort=agent_cfg.get(
                "reasoning_effort", raw.get("reasoning_effort")
            ),
            harness=agent_cfg.get("harness", raw.get("harness", "acp")),
            environment=environment,
            concurrency=concurrency,
            agent_env=agent_env,
            reviewer=ReviewerConfig.coerce(raw.get("reviewer")),
            retry=RetryConfig(max_retries=0),
            skills_dir=skills_dir,
            codex_apps_policy=raw.get("codex_apps_policy"),
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            skip_agent_install=bool(agent_cfg.get("skip_install", False)),
            fresh_install=bool(raw.get("fresh_install", False)),
            allow_cloud_metadata=bool(raw.get("allow_cloud_metadata", False)),
            agent_idle_timeout=raw.get(
                "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
            ),
            context_root=raw.get("context_root"),
            base_image_override=raw.get("base_image_override"),
            separate_verifier=bool(raw.get("separate_verifier", False)),
            separate_verifier_include=tuple(raw.get("separate_verifier_include") or ()),
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
            agent_defaulted=agent_defaulted,
            # Where the verifiers run, as in the native format: a legacy file
            # that asks for separate verifiers (or strict integrity, which
            # implies them) must get them, never a silent shared run
            # (SV2-codex-1).
            integrity=raw.get("integrity") or "off",
            verifier_sandbox=raw.get("verifier_sandbox", "task"),
            # The agent phase's multiplier; job_config_problems refused any other
            # unsupported phase multiplier.
            timeout_multiplier=agent_timeout_multiplier(raw),
            # The verifier and build phases, scaled in the run's overlay.
            config_override=(
                {"timeout_multipliers": scales}
                if (scales := phase_timeout_multipliers(raw))
                else None
            ),
            extra_instruction=extra_instruction(raw),
            n_tasks=dataset_filter.get("n_tasks"),
        )
        if config.agent_defaulted and warn_default_agent:
            logger.warning(default_agent_warning(config.model))
        evaluation = cls(
            tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config, **kwargs
        )
        evaluation._config_task_list = config_task_list
        evaluation._default_oracle = default_oracle
        evaluation._config_explicit_timeout_phases = set(
            phase_timeout_multipliers(raw, explicit_only=True)
        )
        if default_oracle and warn_default_agent:
            logger.warning(default_oracle_note(path))
        return evaluation

    def _is_runnable(self, path: Path) -> bool:
        """The runnable-task rule, with this run's environment manifest."""
        return _is_task_dir(
            path, image_supplied=self._config.environment_manifest is not None
        )

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
        if not self._tasks_dir.is_dir():
            raise FileNotFoundError(
                f"Tasks directory not found: {self._tasks_dir} (resolved against "
                f"{Path.cwd()}); pass a folder of task folders, or one task "
                "folder (each holds task.md or task.toml)"
            )

        from benchflow._utils.task_authoring import (
            run_folder_link_issues,
            task_symlink_issues,
        )
        from benchflow.task.formats import detect_task_format, materialize_task_dir

        seeds = self._config.seeds
        include, exclude = self._config.include_tasks, self._config.exclude_tasks

        def picked(name: str) -> bool:
            return task_name_selected(name, include, exclude)

        def materialize(d: Path) -> list[Path]:
            if not seeds:
                natives = [materialize_task_dir(d)]
            else:
                natives = [materialize_task_dir(d, seed=s) for s in seeds]
            # The converted package can live in a cache outside the source
            # checkout. Derive source provenance from the selected original,
            # never from an untrusted path claimed by package metadata.
            for native in natives:
                self._materialized_task_sources[native] = d
            return natives

        # Decide whether the root is a task before rejecting its descendants.
        # A malformed root can be a batch wrapper; an excluded child's link
        # must not suppress every healthy task in that batch. Check a linked
        # task file itself before reading it or detecting its source format.
        root = self._tasks_dir
        if any((root / name).is_symlink() for name in ("task.md", "task.toml")):
            logger.warning(
                "Skipping task %r: %s", root.name, task_symlink_issues(root)[0]
            )
            return []
        source_format = detect_task_format(root)
        if source_format is not None or (
            _is_structural_task_dir(root) and _task_parse_error(root) is None
        ):
            # A task in another format is copied when it is materialized, so
            # any link in it is refused; a native task only for a link in a
            # folder the run uploads (runnable_task_issues).
            blocking = (
                task_symlink_issues(root)
                if source_format is not None
                else run_folder_link_issues(root)
            )
            if blocking:
                logger.warning("Skipping task %r: %s", root.name, blocking[0])
                return []
            linked = task_symlink_issues(root)
            if linked:
                # A link elsewhere in the task does not stop the run: the file
                # it points at never arrives, so say so and go on.
                logger.warning("Task %r: %s", root.name, linked[0])

        # A task in a registered task format (benchflow.task.formats) at the
        # root is a single-task input too; it runs as its materialized package.
        if source_format is not None:
            natives = materialize(self._tasks_dir)
            # with seeds, filters match the source folder (like child folders below)
            base = self._tasks_dir.name if seeds else natives[0].name
            if not picked(base):
                return []
            return natives

        # A valid task at the root → that IS the whole job (single-task input).
        if self._is_runnable(self._tasks_dir):
            if seeds:
                raise ValueError(
                    f"--seeds: {self._tasks_dir.name} is a native task package; seeded rollouts "
                    "need a task format that writes seeded variants (docs/tasks/embodied.md)"
                )
            if not picked(self._tasks_dir.name):
                return []
            return [self._tasks_dir]

        # Batch input: collect valid child tasks; warn (don't silently drop) on
        # any child whose task.md fails to PARSE. Selection filters are applied
        # first, so excluded dirs are never warned about. A child task.md that
        # PARSES but is structurally incomplete (schema-only fixture) keeps its
        # silent skip.
        selected: list[Path] = []
        # Selected tasks skipped as malformed, with their error: an empty
        # selection names them, not the filters (DFR-p1-2).
        malformed_tasks: list[tuple[str, str]] = []
        self._malformed_tasks = malformed_tasks
        children = [
            d
            for d in sorted(self._tasks_dir.iterdir())
            if d.is_dir() and picked(d.name)
        ]
        children = sample_task_dirs(
            children,
            self._config.n_tasks,
            self._config.sample_seed,
            is_task=lambda d: detect_task_format(d) is not None or self._is_runnable(d),
        )
        for d in children:
            linked = task_symlink_issues(d)
            has_task_file = (d / "task.md").is_file() or (d / "task.toml").is_file()
            if has_task_file and (
                any((d / name).is_symlink() for name in ("task.md", "task.toml"))
                or run_folder_link_issues(d)
                or (linked and detect_task_format(d) is not None)
            ):
                logger.warning("Skipping task %r: %s", d.name, linked[0])
                continue
            if detect_task_format(d) is not None:
                from benchflow.taskmd import TaskMdError

                try:
                    selected.extend(materialize(d))
                except TaskMdError as exc:
                    # A task.md draft 2 package BenchFlow refuses must not
                    # vanish from a batch silently, nor stop the healthy ones.
                    logger.warning("Skipping task %r: %s", d.name, exc)
                    malformed_tasks.append((d.name, str(exc)))
                continue
            if self._is_runnable(d):
                if seeds:
                    raise ValueError(
                        f"--seeds: {d.name} is a native task package; seeded rollouts need "
                        "a task format that writes seeded variants (docs/tasks/embodied.md)"
                    )
                if linked:
                    # The same rule as a task named on its own (above).
                    logger.warning("Task %r: %s", d.name, linked[0])
                selected.append(d)
                continue
            malformed = _task_parse_error(d)
            if malformed is not None:
                logger.warning("Skipping malformed task %r: %s", d.name, malformed[1])
                malformed_tasks.append((d.name, malformed[1]))
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

    def _selectable_names(self) -> list[str]:
        """The task names include/exclude patterns are matched against."""
        root = self._tasks_dir
        if _is_task_dir(root):
            return [root.name]
        if not root.is_dir():
            return []
        return [d.name for d in root.iterdir() if d.is_dir() and d.name[:1] != "."]

    def warn_unmatched_selection(self) -> None:
        """Warn once per pattern that matched no task, though others did."""
        if self._selection_warned:
            return
        self._selection_warned = True
        cfg = self._config
        if not (cfg.include_tasks or cfg.exclude_tasks):
            return
        for line in unmatched_selection(
            self._selectable_names(), cfg.include_tasks, cfg.exclude_tasks
        ):
            logger.warning("%s in %s", line, self._tasks_dir)

    def _empty_selection_message(self) -> str:
        """Why no task was selected, naming structural problems when no filter did it.

        Discovery skips a task file that parses but fails ``bench tasks
        check`` (a schema-only fixture, a task with no environment/), so
        without filters the only visible symptom would be an empty selection.
        """
        cfg = self._config
        root = self._tasks_dir
        tail = " Refusing to publish an empty 0/0 summary."
        malformed = getattr(self, "_malformed_tasks", None)
        if malformed:
            # Every selected task was refused as malformed: a fix to the task
            # files, not to the filters (DFR-p1-2).
            name, error = malformed[0]
            n = len(malformed)
            what = (
                "1 selected task is malformed"
                if n == 1
                else f"all {n} selected tasks are malformed"
            )
            filtered = ""
            if cfg.include_tasks or cfg.exclude_tasks:
                out = sum(
                    not task_name_selected(t, cfg.include_tasks, cfg.exclude_tasks)
                    for t in self._selectable_names()
                )
                filtered = f"; {out} other task(s) did not match --include/--exclude"
            return (
                f"No tasks selected in {root}: {what}, so none can run "
                f"(first: {name}: {render_text(_first_line(error), 200)}){filtered}."
                + tail
            )
        if cfg.include_tasks or cfg.exclude_tasks:
            return empty_selection_message(
                f"tasks_dir={root}",
                cfg.include_tasks,
                cfg.exclude_tasks,
                self._selectable_names(),
            )
        from benchflow._utils.task_authoring import (
            not_runnable_message,
            runnable_task_issues,
        )

        supplied = cfg.environment_manifest is not None

        def _has_task_file(path: Path) -> bool:
            return (path / "task.md").is_file() or (path / "task.toml").is_file()

        if _has_task_file(root):
            issues = runnable_task_issues(root, image_supplied=supplied)
            if issues:
                # One named task: its issues, one per line; the next step
                # (_empty_selection_hint) names it (DF-p1-first-run-18).
                return f"No tasks selected: {not_runnable_message(root, issues)}"
        unrunnable = (
            sorted(d for d in root.iterdir() if d.is_dir() and _has_task_file(d))
            if root.is_dir()
            else []
        )
        if unrunnable:
            first = unrunnable[0]
            issues = runnable_task_issues(first, image_supplied=supplied)
            example = f" (first: {first.name}: {issues[0]})" if issues else ""
            noun = (
                "1 subdirectory has a task file but is not a runnable task"
                if len(unrunnable) == 1
                else f"{len(unrunnable)} subdirectories have task files but "
                "are not runnable tasks"
            )
            return (
                f"No tasks selected in {root}: {noun}{example}. Run "
                f"`bench check <task>` for details." + tail
            )
        return f"No tasks selected: no task found in {root}." + tail

    def _empty_selection_hint(self) -> str | None:
        """The next step for an empty selection of one named task folder, or
        of tasks that are all malformed."""
        cfg = self._config
        root = self._tasks_dir
        malformed = getattr(self, "_malformed_tasks", None)
        if malformed:
            name, error = malformed[0]
            folder = root / name if len(malformed) == 1 else root
            return (
                f"fix {name}'s task file ({render_text(_first_line(error), 160)}); "
                f"`bench check {command_arg(folder, '<folder>')}` lists each issue"
            )
        if cfg.include_tasks or cfg.exclude_tasks:
            return None
        if not ((root / "task.md").is_file() or (root / "task.toml").is_file()):
            return None
        return f"`bench check {command_arg(root, '<task folder>')}` explains each issue"

    def _get_completed_tasks(self) -> dict[str, dict]:
        """Load tasks that already have results with rewards or verifier errors.

        Scoreless results whose verifier error a resume reruns (per
        ``RetryConfig.reruns_verifier_error_on_resume``) are not reused: the
        verifier never scored the frozen workspace, so the task re-runs.

        Scoped to the current job directory (``_jobs_dir / _job_name``) to
        prevent cross-job contamination. When multiple result.json files exist
        for the same task (retry artifacts), a scored result always wins over
        a scoreless verifier error; otherwise the newest artifact wins.

        orphan retry artifacts no longer pollute resume.
        """
        job_dir = self._jobs_dir / self._job_name
        if not job_dir.exists():
            return {}
        latest = load_task_results(
            job_dir,
            include_terminal_errors=self._config.job_mode == "sequential-shared",
        )
        # A trial that left no rollout folder never ran: the task runs again in
        # every job mode, so a fix made since is used (review DX-e R2-M1).
        latest = {
            task: r
            for task, r in latest.items()
            if r.get(NO_ROLLOUT_FOLDER) is not True
        }
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
                    and self._config.retry.reruns_on_resume(
                        pending.get("error"), category=pending.get("error_category")
                    )
                ):
                    # Its solver failed on infrastructure or ended on a usage
                    # limit: nothing to review.
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
                    _scoring_block(r),
                    r.get("error"),
                    category=r.get("error_category"),
                    on_resume=True,
                ):
                    logger.info(
                        f"Re-running task whose solver failed on infrastructure "
                        f"before anything was judged: {task} "
                        f"({render_text(r.get('error') or '', 80)})"
                    )
                    continue
                completed[task] = r
                continue
            if r.get("verifier_error"):
                # A scoreless result whose verifier error is infra-retryable
                # (same taxonomy as the within-run retry) records no signal
                # about the task; reusing it pins a lost score forever.
                retryable = self._config.retry.reruns_verifier_error_on_resume(
                    r["verifier_error"]
                )
                if rerun_ok and r.get("rewards") is None and retryable:
                    logger.info(
                        f"Re-running verifier-errored task on resume: {task} "
                        f"({render_text(r['verifier_error'], 80)})"
                    )
                    continue
                logger.info(
                    f"Reusing completed verifier-errored task on resume: {task} "
                    f"({render_text(r['verifier_error'], 80)})"
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
        """Copy timing, loop and integration-failure records from the
        rollout's on-disk result.json into payload.

        ``RolloutResult`` does not carry phase timing, but the rollout writer
        (``rollout.py``) persists it under ``rollout_dir/result.json``. Reading
        it back lets ``phase_timing_summary`` aggregate phase totals for fresh
        runs (issue #501). Best-effort: a result with no rollout_name or no
        persisted result.json leaves timing absent rather than crash summary
        generation.
        """
        if all(
            key in payload
            for key in (
                "timing",
                "integration_failure_info",
                "loop",
                "verifier_isolation",
            )
        ):
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
        # Where the verifier ran, for summary.json's verifier_isolation block.
        for key in ("verifier_isolation", "verifier_isolation_source"):
            if key in persisted:
                payload.setdefault(key, persisted[key])
        # Fresh and resumed trials must contribute the same loop population.
        loop = persisted.get("loop")
        if isinstance(loop, dict):
            payload.setdefault("loop", loop)

    async def _run_single_task(
        self, task_dir: Path, cfg: EvaluationConfig
    ) -> RolloutResult:
        """Execute one rollout via Rollout.

        In sequential-shared mode the per-rollout learner skill dirs override
        the static config: the rollout starts from the LearnerStore's evolved
        skill set (``_learner_skills_dir``) and its agent-evolved skills are
        captured back through ``export_generated_skills_to``.

        The first attempt of a trial the warm pool prepared runs that
        prepared rollout (``benchflow._warm_pool``); a retry starts cold.
        """
        from benchflow.rollout import Rollout

        rollout = _WARM_ROLLOUT.get()
        _WARM_ROLLOUT.set(None)  # a retried attempt starts cold
        if rollout is None:
            rollout_config = self._rollout_config(task_dir, cfg)
            if rollout_config.skill_mode == SKILL_MODE_SELF_GEN:
                from benchflow.self_gen import run_self_gen

                return await run_self_gen(rollout_config)
            rollout = await Rollout.create(rollout_config)
        pool = getattr(self, "_warm_pool", None)
        if pool is not None:
            rollout.on_sandbox_ready(pool.ready)
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

    def _rollout_config(self, task_dir: Path, cfg: EvaluationConfig) -> Any:
        """The RolloutConfig a trial of ``task_dir`` runs with.

        The warm pool prepares a queued trial from this same config, so a
        prepared trial and a cold one are made alike.
        """
        from benchflow._utils.benchmark_repos import task_source_provenance
        from benchflow.rollout import RolloutConfig

        dataset = None
        if cfg.dataset_name:
            dataset = {"name": cfg.dataset_name, "version": cfg.dataset_version}
        task_digest_value = _task_digest_for(task_dir, cfg)
        skills_dir = (
            str(self._learner_skills_dir)
            if self._learner_skills_dir is not None
            else cfg.skills_dir
        )
        skill_mode = _rollout_skill_mode(
            cfg.skill_mode, learner=self._learner_skills_dir is not None
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
            harness=cfg.harness,
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
            # The learner store holds skills earlier agents generated.
            skills_agent_derived=self._learner_skills_dir is not None,
            codex_apps_policy=cfg.codex_apps_policy,
            sandbox_user=cfg.sandbox_user,
            sandbox_locked_paths=cfg.sandbox_locked_paths,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            skip_agent_install=cfg.skip_agent_install,
            agent_idle_timeout=cfg.agent_idle_timeout,
            context_root=cfg.context_root,
            base_image_override=cfg.base_image_override,
            separate_verifier=cfg.separate_verifier,
            separate_verifier_include=cfg.separate_verifier_include,
            skill_mode=skill_mode,
            skill_creator_dir=cfg.skill_creator_dir,
            self_gen_no_internet=cfg.self_gen_no_internet,
            export_generated_skills_to=export_to,
            source_provenance=task_source_provenance(
                cfg.source_provenance,
                self._materialized_task_sources.get(task_dir, task_dir),
            ),
            dataset=dataset,
            task_digest=task_digest_value,
            usage_tracking=cfg.usage_tracking,
            loop_strategy=cfg.loop_strategy,
            **_budget_overrides(cfg, task_dir),
        )
        rollout_config.checkpoints = cfg.checkpoint_policy()
        if cfg.pre_agent_hooks:
            rollout_config.pre_agent_hooks = [
                *(rollout_config.pre_agent_hooks or []),
                *cfg.pre_agent_hooks,
            ]
        rollout_config.freeze_workspace = cfg.freeze_workspace
        rollout_config.fresh_install = cfg.fresh_install
        rollout_config.allow_cloud_metadata = cfg.allow_cloud_metadata
        rollout_config.integrity = cfg.integrity
        rollout_config.verifier_sandbox = effective_verifier_sandbox(
            cfg.verifier_sandbox, cfg.separate_verifier
        )
        return rollout_config

    async def _run_task(self, task_dir: Path) -> RunResult:
        """Run a single task with retries.

        With a budget, every attempt's USD and tokens count against it, and a
        retry starts only while the budget allows another rollout; otherwise
        the task keeps its last attempt's result.
        """
        cfg = self._config
        guard = self._budget_guard
        last_result: RunResult | None = None

        for attempt in range(1, cfg.retry.max_retries + 2):
            if attempt > 1:
                delay = cfg.retry.backoff_delay(attempt - 1)
                logger.info(f"Retry backoff: {delay:.1f}s before attempt {attempt}")
                await asyncio.sleep(delay)
                await self._sweep_docker()
            result = await self._run_single_task(task_dir, cfg)
            last_result = result
            if guard is not None:
                guard.attempt_done(task_dir.name, result)
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
                if guard is not None and not guard.retry(task_dir.name):
                    logger.info(f"Not retrying {task_dir.name}: {guard.reason}")
                    break
                err_preview = render_text(
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
        error = result.error
        if isinstance(error, str) and error and err_msg == error:
            # A classified error reads as what happened, not as the Python
            # class that carried it (DF-p5-cli-craft-7): "the task's image did
            # not build: ...", not "ImageBuildError: the task's ...". An
            # unclassified one keeps its class, its only clue.
            category = result.error_category or classify_error(error)
            if category not in (None, OTHER_ERROR):
                err_msg = _EXCEPTION_CLASS_PREFIX.sub("", error, count=1)
        err = f" ({render_text(err_msg, 50)})" if err_msg else ""
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

    def _runs_stock_trials(self) -> bool:
        """Whether trials run through this class's own ``_run_single_task``.

        A subclass or caller that replaced it runs something the job cannot
        prepare for (its images, its sandboxes).
        """
        method = getattr(self, "_run_single_task", None)
        return getattr(method, "__func__", None) is _STOCK_RUN_SINGLE_TASK

    def _start_prepull(self, remaining: list[Path]) -> asyncio.Future[Any] | None:
        """Begin pulling the job's prebuilt task images (Docker only), or None.

        In queue order, so the first trials' images come first. A job of one
        trial has nothing to overlap the pull with and pulls in its trial.
        """
        from benchflow.sandbox._prepull import (
            prepull_images,
            prepull_requested,
            task_sets_platform,
        )

        cfg = self._config
        if (
            cfg.environment != "docker"
            or len(remaining) < 2
            or not prepull_requested()
            or not self._runs_stock_trials()
        ):
            return None
        refs: list[str] = []
        for task_dir in remaining:
            if task_sets_platform(task_dir):
                # The trial pulls the pinned platform itself; a pull here
                # would fetch the host's variant beside it.
                continue
            try:
                ref = _task_image_ref(task_dir, cfg)
            except Exception:
                ref = None  # the trial reports its own task problem
            if ref:
                refs.append(ref)
        if not refs:
            return None
        return asyncio.ensure_future(prepull_images(refs))

    def _make_warm_pool(
        self, remaining: list[Path], stopped: Callable[[], bool]
    ) -> WarmPool | None:
        """The job's warm pool (``benchflow._warm_pool``), or None.

        Only on Docker, for a job with more trials than ``--concurrency``
        (otherwise no trial waits for a slot), with room for
        ``warm_pool_size`` prepared sandboxes (``--warm-pool``; 0 turns it
        off, as ``BENCHFLOW_WARM_POOL=off`` does). Only when trials run through this class's own
        ``_run_single_task``: a subclass or caller that replaced it runs
        something the pool cannot prepare.
        """
        cfg = self._config
        # A call the local runtime prepares ahead (benchflow._park) holds its
        # one trial in a pool of one until its caller arrives.
        parked = _park.role() == "parked"
        size = 1 if parked else warm_pool_size(cfg.concurrency, cfg.warm_pool)
        if (
            cfg.environment != "docker"
            or (len(remaining) <= cfg.concurrency and not parked)
            or size == 0
            or self._learner_skills_dir is not None
            or not self._runs_stock_trials()
        ):
            return None
        from benchflow.rollout import Rollout, lifecycle_is_stock

        if not lifecycle_is_stock(Rollout):
            return None

        def key(index: int) -> tuple[Any, ...] | None:
            return _warm_image_key(remaining[index], cfg)

        def make(index: int) -> tuple[Any, tuple[Any, ...]] | None:
            task_dir = remaining[index]
            if not task_shares_nothing(task_dir):
                return None
            image_key = key(index)
            if image_key is None:
                return None
            rollout_config = self._rollout_config(task_dir, cfg)
            if rollout_config.skill_mode == SKILL_MODE_SELF_GEN:
                return None
            rollout = Rollout(rollout_config)
            rollout.on_sandbox_ready(pool.ready)
            return rollout, image_key

        pool = WarmPool(
            size=len(remaining),
            capacity=size,
            make=make,
            key=key,
            stopped=stopped,
        )
        return pool

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
        # Pace starts above --concurrency 16 so SSH and the Docker daemon's
        # compose-up handler never see a storm (at about 10 starts/s they
        # did). The pacer waits for its token before the slot is taken, so
        # no slot idles while a trial waits to start.
        pacer = StartPacer() if cfg.concurrency > PACE_ABOVE_CONCURRENCY else None

        def stopped() -> bool:
            return (
                (guard is not None and guard.stopped)
                or usage_stop.error is not None
                or breaker.tripped
            )

        async def admit(td: Path) -> tuple[bool, RunResult | None]:
            # Holding the slot. These checks run after any pacing wait: a
            # trial that hit the usage limit during it stops this one too.
            if guard is not None and guard.stopped:
                guard.start(td.name)  # records it as not started
                return False, None
            if usage_stop.skip(td.name):
                return False, None
            if breaker.tripped:
                result = RunResult(task_name=td.name, error=breaker.skip_error())
                self._log_and_report(td, result)
                return False, result
            if guard is not None and not await guard.admit(
                td.name, asyncio.current_task()
            ):
                return False, None
            # Admission can suspend behind another trial. Its completion may
            # stop the login or open the breaker while waking budget waiters.
            if usage_stop.error is not None or breaker.tripped:
                if guard is not None:
                    guard.abandon(td.name)
                if usage_stop.skip(td.name):
                    return False, None
                result = RunResult(task_name=td.name, error=breaker.skip_error())
                self._log_and_report(td, result)
                return False, result
            return True, None

        async def start(
            index: int, td: Path, slot_taken: float, pace_wait: float | None
        ) -> tuple[str, RunResult | None]:
            self._fire_progress(self._on_task_start, td.name)
            waits = {"slot_wait": timing_spans.clock() - slot_taken}
            if pace_wait is not None:
                waits["start_pace_wait"] = pace_wait
            timing_spans.set_pending_trial_timing(**waits)
            prepared = pool.take(index) if pool is not None else None
            _WARM_ROLLOUT.set(prepared)
            try:
                result = await self._run_budgeted(td, guard)
            finally:
                if prepared is not None and not prepared.adopted:
                    # Stopped before its lifecycle began: nothing else
                    # removes the prepared sandbox.
                    await prepared.discard_prestart()
                if pool is not None:
                    pool.ready()
            if result is None:
                return td.name, None
            usage_stop.record(result)
            breaker.record(result)
            self._log_and_report(td, result)
            return td.name, result

        def not_started(
            index: int, td: Path, skipped: RunResult | None
        ) -> tuple[str, RunResult | None]:
            if pool is not None:
                pool.drop(index)
            return td.name, skipped

        async def bounded(index: int, td: Path) -> tuple[str, RunResult | None]:
            if pacer is None:
                async with sem:
                    # slot_wait: from here to the trial start.
                    slot_taken = timing_spans.clock()
                    allowed, skipped = await admit(td)
                    return (
                        await start(index, td, slot_taken, None)
                        if allowed
                        else not_started(index, td, skipped)
                    )
            allowed, skipped = False, None
            slot_taken = 0.0

            async def admission() -> bool:
                nonlocal allowed, skipped, slot_taken
                slot_taken = timing_spans.clock()
                allowed, skipped = await admit(td)
                return allowed

            # Keep the FIFO turn across budget waits. Spending a token before
            # admission let many waiters bank starts, then all run together
            # when a completion lowered the projected spend.
            pace_wait = await pacer.take_slot(sem, stopped, admit=admission)
            try:
                return (
                    await start(index, td, slot_taken, pace_wait)
                    if allowed
                    else not_started(index, td, skipped)
                )
            finally:
                sem.release()

        # The next trials' sandboxes, made while the running ones work; the
        # job's end (a stop or Ctrl-C included) removes those no trial took.
        pool = self._warm_pool = self._make_warm_pool(remaining, stopped)
        watcher = asyncio.ensure_future(guard.watch()) if guard is not None else None
        try:
            if _park.role() == "parked":
                # Returns once an identical call takes this one; else the
                # process removes what it made and ends.
                await _park.hold(pool)
            results_or_errors = await asyncio.gather(
                *[bounded(index, td) for index, td in enumerate(remaining)],
                return_exceptions=True,
            )
        finally:
            if watcher is not None:
                watcher.cancel()
            if pool is not None:
                self._warm_pool = None
                await pool.drain()

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
                logger.error(f"[ERR] {task_name}: {_unexpected_text(r)}")
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
        from benchflow.rollout._teardown_queue import counting_stop_seconds

        try:
            with (
                guard.track_trial(td.name),
                counting_stop_seconds(
                    lambda seconds: guard.add_stop_seconds(td.name, seconds)
                ),
            ):
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
            guard.finish(td.name)  # stop counting its sandbox time
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
                if self._usage_stop.error is not None or (
                    guard is not None and not guard.start(td.name)
                ):
                    # A usage limit stopped the job, or a cap was reached (the guard
                    # records the trial as not started).
                    if self._usage_stop.error is not None:
                        self._usage_stop.skip(td.name)
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
                        guard.attach(td.name, trial)
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
                    logger.error(f"[ERR] {td.name}: {_unexpected_text(e)}")
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
        """Fire-and-forget auto-reap of abandoned Daytona sandboxes (issue: leakage at scale).

        Gated by ``BENCHFLOW_DAYTONA_AUTO_REAP`` (default on; any of
        ``0``/``false``/``no``/``off`` case-insensitively disables it).
        Deletes only this owner's sandboxes that Daytona has already stopped,
        archived or errored and whose run lease lapsed; a STARTED sandbox is
        never deleted (DAYTONA-START round 3: liveness is judged by Daytona's
        inactivity stop, never by a clock outside it). Old sandboxes with no
        lease, and STARTED ones with a lapsed lease, are only reported;
        branch snapshots carry no lease, so they are left to ``bench sandbox
        cleanup``. Runs in a daemon thread so eval startup never blocks or
        fails on reaping.
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
                from benchflow.sandbox.daytona import reap_abandoned_sandboxes

                counts = reap_abandoned_sandboxes()
                if counts["deleted"] or counts["failed"]:
                    logger.info(
                        "Daytona auto-reap: %s abandoned sandboxes deleted "
                        "(stopped by Daytona, lease lapsed; %s failed)",
                        counts["deleted"],
                        counts["failed"],
                    )
                if counts.get("started_lapsed"):
                    logger.info(
                        "Daytona auto-reap: %s STARTED sandboxes have a lapsed "
                        "run lease; left alone until Daytona's inactivity stop",
                        counts["started_lapsed"],
                    )
                if counts["unleased"]:
                    logger.info(
                        "Daytona auto-reap: %s sandboxes older than a day carry "
                        "no run lease, so nothing proves them abandoned; left "
                        "alone. `bench sandbox cleanup --dry-run` lists them.",
                        counts["unleased"],
                    )
            except Exception as e:
                logger.debug("Daytona auto-reap skipped: %s", e)

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
        finished tasks and runs only the rest. agent_env values that look
        secret are never stored, so pass ``agent_env`` again if the agent
        needs them (the others are restored), and ``pre_agent_hooks`` too
        (only their names are recorded); ``config_overrides`` replace
        individual config fields (e.g. ``concurrency=8``). An override or a
        missing hook that changes a setting the trials ran with
        (:data:`_RESUME_SETTINGS`) sets the finished trials aside, as for any
        named job, and the run starts them over. Jobs started before this
        record existed need ``Evaluation(tasks_dir, jobs_dir=job_dir.parent,
        config=..., job_name=job_dir.name)`` instead.
        """
        import dataclasses

        job_dir = Path(job_dir)
        record_path = job_dir / EVALUATION_RECORD
        if not record_path.is_file():
            trials = _trial_folders(job_dir)
            if trials:
                made_by = (
                    "was made by BenchFlow 0.7 or older"
                    if _made_by_07(trials)
                    else "was made without a run record (by bf.run or bf.run_batch)"
                )
                raise NoEvaluationRecordError(
                    f"{job_dir} {made_by}: it has no "
                    f"{EVALUATION_RECORD}, so its run cannot be rebuilt from the "
                    "folder. In Python, pass the run again: Evaluation(tasks_dir=..., "
                    f"jobs_dir={str(job_dir.parent)!r}, config=..., "
                    f"job_name={job_dir.name!r}).",
                    hint=(
                        "re-run the command that started it, which resumes this job "
                        "and reuses its finished trials: "
                        f"`{_pre_08_resume_command(job_dir, trials)}` plus any "
                        "other flags it had (--agent-env, --sandbox-user, a "
                        "source, skills); a setting that differs sets the "
                        "finished trials aside instead"
                    ),
                )
            raise NoEvaluationRecordError(
                f"{job_dir} is not a job folder: it has no {EVALUATION_RECORD} "
                "and no trials.",
                hint="pass a job folder, <jobs-dir>/<job name>; "
                "`bench results <jobs-dir>` lists them",
            )
        record = json.loads(record_path.read_text())
        config = _config_from_record(record["config"], agent_env)
        hooks = record["config"].get("pre_agent_hooks")
        if hooks and "pre_agent_hooks" not in config_overrides:
            logger.warning(
                "The job ran pre_agent_hooks %s, which %s cannot store; pass "
                "pre_agent_hooks=[...] to Evaluation.resume() to run them again "
                "(without them the finished trials are set aside).",
                ", ".join(map(str, hooks)),
                EVALUATION_RECORD,
            )
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)
        evaluation = cls(
            tasks_dir=tasks_dir or record["tasks_dir"],
            jobs_dir=job_dir.parent,
            config=config,
            job_name=job_dir.name,
        )
        evaluation._resuming = True  # no unselected-results warning
        return evaluation

    @property
    def job_dir(self) -> Path:
        """Where this job writes its rollouts, known before it runs.

        Pass it to :func:`benchflow.astream_rollouts` to train on rollouts
        while :meth:`run` is still going.
        """
        return self._jobs_dir / self._job_name

    def _write_evaluation_record(self) -> None:
        """Record tasks_dir and config so Evaluation.resume(job_dir) can finish the job.

        ``self._planned_tasks``, once a run has chosen them, names the tasks
        it sets out to finish (finished ones included); readers count an interrupted job against it
        (``Job.planned_tasks``, ``bench eval list``; DF-p3-eval-runner-6).

        Also which BenchFlow wrote the record (``benchflow_version``) and
        which one made the job (``created_by_benchflow_version``, kept from
        the first record; :data:`PRE_08_VERSION` for a job from before
        evaluation.json), so the trials of a job resumed across versions can
        be told apart (each trial's config.json has its own version).
        """
        job_dir = self._jobs_dir / self._job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / EVALUATION_RECORD
        created = getattr(self, "_created_by", None)
        if created is None:
            if path.is_file():
                # An earlier record without the key: an older 0.8 build made
                # the job, not this one (L2).
                try:
                    created = json.loads(path.read_text()).get(
                        "created_by_benchflow_version"
                    )
                except (OSError, ValueError, AttributeError):
                    created = None
                created = created or UNKNOWN_VERSION
            else:
                created = _benchflow_version()
            self._created_by = created
        record = {
            "schema_version": 1,
            "tasks_dir": str(self._tasks_dir),
            "job_name": self._job_name,
            "benchflow_version": _benchflow_version(),
            "created_by_benchflow_version": created,
            "config": _config_to_record(self._config),
            **(
                {"planned_tasks": sorted(planned)}
                if (planned := getattr(self, "_planned_tasks", None)) is not None
                else {}
            ),
        }
        path.write_text(json.dumps(record, indent=2) + "\n")

    def _record_trial_without_folder(
        self,
        job_dir: Path,
        task_name: str,
        payload: dict[str, Any],
        task_dir: Path | None = None,
    ) -> None:
        """Give a counted trial that left no folder one, with its result.json.

        A task that fails before its rollout folder exists (a task.toml typo,
        an unexpected error) is a trial of the job: the summary counts it, an
        unscored one as 0. Readers (``bf.load_job``, ``bench eval metrics``)
        count trial folders, so without one the same job had two every-trial
        means (DX-E-P4-TWO-MEANS: 0.40 over 5 trials in summary.json, 0.50
        over 4 in metrics). The result is marked ``no_rollout_folder``: no
        resume reuses it as finished, in any job mode, so a task fixed since
        runs again (review DX-e R2-M1), and its config.json carries what the
        rollout's own records (task_digest, loop, the resolved agent), so the
        resume checks read it truly (R2-L1).
        """
        name = payload.get("rollout_name")
        if name and (job_dir / name).is_dir():
            return
        if not name or Path(name).name != name or name in (".", ".."):
            # Only a trial that named no folder gets a new name.
            name = f"{task_name}__{uuid.uuid4().hex[:8]}"
        folder = job_dir / name
        folder.mkdir(parents=True, exist_ok=False)
        payload["rollout_name"] = folder.name
        payload[NO_ROLLOUT_FOLDER] = True
        config: dict[str, Any] = {
            "benchflow_version": _benchflow_version(),
            "task_path": task_name,
            "agent": self._config.agent,
            "model": self._config.model,
            "environment": self._config.environment,
            "purpose": "task",
            "loop": loop_block(self._config.loop_strategy),
        }
        if task_dir is not None:
            digest = _task_digest_for(task_dir, self._config)
            if digest is not None:
                config["task_digest"] = digest
            try:
                config["agent"] = _resolved_agent(task_dir, self._config)
            except Exception as exc:  # a task that does not load (the typo)
                logger.debug("Could not resolve the agent of %s: %s", task_name, exc)
        # dx-a's writer: a full disk names the job folder (DiskFullError,
        # DF-p5-cli-craft-5) and never leaves a partial result.json.
        from benchflow._utils.disk_space import write_text_atomic

        write_text_atomic(folder / "config.json", json.dumps(config, indent=2) + "\n")
        write_text_atomic(folder / "result.json", json.dumps(payload, indent=2) + "\n")

    def _keep_pre_08_files(self) -> None:
        """Copy a 0.7 job's result files to ``<job>/.pre-0.8/`` and say so.

        A job whose trials BenchFlow 0.7 ran (:func:`_made_by_07`) has no
        evaluation.json; resuming it rewrites its summary.json and results
        files, and its summary's ``mean_reward`` was over scored trials only
        (DF-p4-upgrader-3). The originals are copied into a temporary folder
        that is renamed to ``.pre-0.8`` only once complete, so a full disk or
        an interrupted copy leaves either no copy or the whole one; a copy
        that does not match the originals byte for byte is made again. When
        the copy cannot be made the run stops before anything in the job is
        rewritten (review DX-e M2). Readers skip dot-folders, so the copy is
        never read as the job's results.
        """
        job_dir = self._jobs_dir / self._job_name
        if (job_dir / EVALUATION_RECORD).is_file() or _is_external_job(job_dir):
            # Another runner's folder is refused before anything moves
            # (_check_resume_mismatch); nothing is copied into it.
            return
        trials = _trial_folders(job_dir)
        if not trials:
            return
        if not _made_by_07(trials):
            # bf.run(job_name=...) and bf.run_batch write no evaluation.json
            # either; their trials say which BenchFlow ran them (L3).
            self._created_by = UNKNOWN_VERSION
            return
        self._created_by = PRE_08_VERSION
        kept = job_dir / PRE_08_DIR
        originals = [n for n in _JOB_RESULT_FILES if (job_dir / n).is_file()]
        if not _copy_matches(kept, job_dir, originals):
            _copy_pre_08_files(job_dir, kept, originals)
        try:
            old = json.loads((kept / "summary.json").read_text())
        except (OSError, ValueError):
            old = {}
        mean = old.get("mean_reward") if isinstance(old, dict) else None
        if isinstance(mean, (int, float)) and not isinstance(mean, bool):
            self._pre_08_mean_reward = float(mean)
        logger.warning(
            f"Job {job_dir.name} was made by BenchFlow 0.7 or older (it has no "
            f"{EVALUATION_RECORD}): this run reuses its finished trials and "
            "rewrites its summary.json and results files. Its own files are "
            f"kept in {kept}. Pass --fresh (or another --job) to leave it "
            "as it was."
        )

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
        prefetch = None
        if getattr(getattr(self, "_config", None), "environment", None) == "docker":
            # The local daemon's subnet read runs beside the job's own setup
            # (PERF-SBX); the first sandbox takes it.
            from benchflow.sandbox.docker import prefetch_local_subnet_allocator

            prefetch = prefetch_local_subnet_allocator()
        try:
            if self._preflight:
                self._check_before_run()
            # Not a preflight check: the CLI builds Evaluation(preflight=False),
            # and this refusal must hold there too, before the job folder exists.
            self._check_verifier_sandbox_plan()
            self._settle_auto_job_name()
            # In the local runtime, a call prepared ahead that cannot be ends
            # here, before its job folder exists (benchflow._park).
            _park.note_job(self)
            lock = self._acquire_job_lock()
        except BaseException:
            if prefetch is not None:
                prefetch.cancel()
            raise
        try:
            return await self._run_unlocked()
        finally:
            if prefetch is not None and not prefetch.done():
                prefetch.cancel()
            lock.release()
            # A collected sandbox whose finalizer found the staging lock
            # taken left its task copy queued; a job's end removes it.
            from benchflow.task.staging import remove_released

            remove_released()
            # The warm host model proxies no trial took (the job's pool).
            from benchflow.providers.runtime import drain_host_proxies

            await drain_host_proxies()

    def _settle_auto_job_name(self) -> None:
        """Start a new job when the job an unnamed run would resume ran with
        other settings than the ones this run has now (see
        :meth:`_resolve_job_name`), or when it holds results of tasks this run
        does not select (:meth:`_unselected_results`): its score would count
        them. A named job is left to :func:`_check_resume_mismatch`, which
        sets its old trials aside, and is warned about unselected results."""
        auto = getattr(self, "_auto_job_name", None)
        if auto is None or self._job_name != auto:
            self._warn_unselected_results()
            return
        job_dir = self._jobs_dir / self._job_name
        if job_dir.is_dir():
            differs = _recorded_run_differs(
                job_dir, self._config, self._tasks_dir
            ) or self._unselected_results(job_dir)
            if not differs:
                job_dirs = _job_folders(self._jobs_dir)
                if len(job_dirs) > 1:
                    logger.warning(
                        f"Multiple job directories found ({len(job_dirs)}); "
                        f"resuming into most recent: {job_dir.name}"
                    )
                else:
                    logger.warning(
                        f"Resuming into existing job directory: {job_dir.name} "
                        "(finished tasks are reused; pass a new job_name, or "
                        "--fresh on the CLI, for a new run)"
                    )
                return
            name = self._new_job_instead(self._jobs_dir, job_dir, differs)
        else:
            # The picked name is a new job because the latest one differed
            # from the settings at construction. Flags applied since (the
            # CLI's --config path sets --agent-env, --prompt, a budget and
            # the rest after from_yaml) may make them match it again: then
            # that job resumes, or an unchanged rerun could never resume.
            job_dirs = _job_folders(self._jobs_dir)
            if (
                not job_dirs
                or _recorded_run_differs(job_dirs[-1], self._config, self._tasks_dir)
                or self._unselected_results(job_dirs[-1])
            ):
                return
            name = job_dirs[-1].name
            logger.warning(
                f"Resuming into existing job directory: {name} (its settings "
                "match the ones this run finally has; finished tasks are reused)"
            )
        self._job_name = name
        self._auto_job_name = name
        if self._config.job_mode == "sequential-shared":
            self.learner_store = self._load_or_init_learner_store()

    def _unselected_task_names(self, job_dir: Path) -> list[str]:
        """Tasks ``job_dir`` holds a result of that this run does not select.

        Empty when the selection cannot be read here (run() then reports why)
        or selects nothing (run() refuses an empty selection)."""
        held = set(load_task_results(job_dir, include_terminal_errors=True))
        if not held:
            return []
        try:
            selected = {d.name for d in self._get_task_dirs()}
        except Exception:
            return []
        return sorted(held - selected) if selected else []

    def _unselected_results(self, job_dir: Path) -> str | None:
        """Why an unnamed run does not resume ``job_dir``: it holds results of
        tasks this run does not select (DFR-p5-4). A wider selection resumes."""
        names = self._unselected_task_names(job_dir)
        if not names:
            return None
        return (
            f"another task selection: it holds results of {len(names)} task(s) "
            f"this run does not select ({_some_names(names)}); the selection "
            "(--include, --exclude, --limit) changed"
        )

    def _warn_unselected_results(self) -> None:
        """A named job keeps every task it holds: say that its score counts
        the ones this run did not select."""
        job_dir = self._jobs_dir / self._job_name
        if self._resuming or not job_dir.is_dir():
            return
        names = self._unselected_task_names(job_dir)
        if names:
            logger.warning(
                f"Job {job_dir.name} also holds results of {len(names)} task(s) "
                f"this run does not select ({_some_names(names)}); its score "
                "counts them. Name another job (--job) to score only this "
                "selection."
            )

    def _check_verifier_sandbox_plan(self) -> None:
        """Refuse, before any job folder or sandbox exists, a run that moves
        every verifier into the separate verifier sandbox (``verifier_sandbox
        = "separate"`` or ``integrity = "strict"``) when a selected task cannot
        use it on this sandbox: one error names every refused task, its reason
        and the next step (:func:`benchflow.integrity.trial.check_separate_plan`).
        """
        from benchflow.integrity.trial import check_separate_plan, run_forces_separate

        cfg = self._config
        setting = effective_verifier_sandbox(
            cfg.verifier_sandbox, cfg.separate_verifier
        )
        if not run_forces_separate(setting, cfg.integrity):
            return
        check_separate_plan(
            self._get_task_dirs(),
            sandbox=cfg.environment,
            run_setting=setting,
            integrity=cfg.integrity,
            config_override=cfg.config_override,
            environment_manifest=cfg.environment_manifest,
        )

    def _check_before_run(self) -> None:
        """The checks bf.run and ``bench eval run`` make before a job exists.

        A misspelt agent, an unknown sandbox or an agent/model pair no
        provider route serves raises ``ValueError``; Docker not
        ready for ``environment="docker"`` raises ``RuntimeError`` with
        ``bench doctor``'s fix; a Claude agent that would fall back to an
        expired login file gets a ``UserWarning``. ``Evaluation(...,
        preflight=False)`` skips all of them; ``BENCHFLOW_SKIP_PREFLIGHT=1``
        skips the host checks (Docker, login file), as for the CLI.
        """
        from benchflow.runtime import (
            check_agent_names,
            check_agent_routes,
            check_host,
            check_sandbox_name,
        )

        cfg = self._config
        check_sandbox_name(cfg.environment)
        check_agent_names([cfg.agent])
        check_agent_routes([cfg])
        check_host([cfg])

    def _acquire_job_lock(self) -> JobLock:
        """Create the job lock, refusing a live holder and taking over a dead one.

        A folder BenchFlow did not make is refused first: taking the lock
        writes in the folder, which is left as it was, its times included.

        A named job (``--job-name``, ``resume``) is refused while another run
        holds it. An unnamed run takes a new job instead
        (:meth:`_lock_auto_job`), except a call the local runtime prepares
        ahead, whose folder ``_park.note_job`` already made and recorded."""
        auto = getattr(self, "_auto_job_name", None)
        if auto is None or auto != self._job_name or _park.role() == "parked":
            job_dir = self._jobs_dir / self._job_name
            _refuse_external_job(job_dir)
            return _acquire_job_lock(job_dir)
        return self._lock_auto_job()

    def _lock_auto_job(self) -> JobLock:
        """Lock the job an unnamed run picked, or a new job of its own.

        An unnamed run never fails because another run holds its job, and two
        runs never share one: per-task runners launched together into
        one jobs dir each got the latest job, or one new timestamp, and every
        run but the first died with :class:`JobLockedError`. A job another
        run holds, or a new name another run created first, moves this run
        to a new job whose folder it creates itself (:func:`_reserve_new_job`).
        """
        while True:
            job_dir = self._jobs_dir / self._job_name
            _refuse_external_job(job_dir)
            if not job_dir.exists():
                try:
                    job_dir.mkdir(parents=True)
                except FileExistsError:
                    self._start_new_job(
                        f"another run created the new job {job_dir.name} first"
                    )
                    continue
            elif (job_dir / _park.PREPARED_MARKER).exists():
                self._start_new_job(f"job {job_dir.name} is prepared for another call")
                continue
            try:
                return _acquire_job_lock(job_dir)
            except JobLockedError:
                self._start_new_job(f"job {job_dir.name} is held by another run")

    def _start_new_job(self, why: str) -> None:
        """Move this unnamed run to a new job folder it made, saying ``why``."""
        name = _reserve_new_job(self._jobs_dir)
        logger.warning(
            f"Starting a new job {name}: {why}. An unnamed run never shares a "
            "job; pass --job-name to resume that job once it has stopped."
        )
        self._job_name = name
        self._auto_job_name = name
        if self._config.job_mode == "sequential-shared":
            self.learner_store = self._load_or_init_learner_store()

    async def _run_unlocked(self) -> EvaluationResult:
        self._maybe_start_daytona_reap()
        task_dirs = self._get_task_dirs()
        if not task_dirs:
            # Fail fast on an empty selection (#407). Silently writing a
            # 0/0 summary.json would surface as an apparently successful
            # eval in downstream dashboards and release evidence.
            error = EmptyTaskSelectionError(self._empty_selection_message())
            error.hint = self._empty_selection_hint()
            raise error
        self.warn_unmatched_selection()
        _refuse_placeholder_tasks(task_dirs)
        _refuse_unnamed_task_env(
            task_dirs,
            manifest=self._config.environment_manifest,
            sandbox=self._config.environment,
        )
        from benchflow.review.automatic import prepare_review

        reviews_checked: set[Path] = set()

        def _check_reviews(dirs: list[Path]) -> None:
            for task_dir in dirs:
                if task_dir not in reviews_checked:
                    prepare_review(task_dir, self._config.reviewer)
                    reviews_checked.add(task_dir)

        # A job an older BenchFlow made: keep its files before anything here
        # moves or rewrites them, and say so (DF-p4-upgrader-3).
        self._keep_pre_08_files()
        # Trials an earlier run of this job ran with other settings are set
        # aside before anything reads them (pending reviews, completed tasks,
        # the budget already spent), so they are neither reused nor counted.
        # Before any moves, every selected task's review plan is checked (a
        # set-aside reruns the tasks): a run refused for a missing reviewer
        # key must not leave the job with its record but without its trials.
        set_aside = _check_resume_mismatch(
            self._jobs_dir / self._job_name,
            self._config,
            before_set_aside=lambda: _check_reviews(task_dirs),
            task_dirs=task_dirs,
            tasks_root=self._tasks_dir,
        )
        if set_aside is not None:
            # The record names the settings of the trials the folder now
            # holds, even if this run stops before it writes it again below.
            self._write_evaluation_record()
            if self._config.job_mode == "sequential-shared":
                # Its learner store moved with the trials whose lessons it holds.
                self.learner_store = LearnerStore()
        from benchflow.review.resume import resume_pending_reviews

        cfg = self._config
        self._budget_guard = None
        if cfg.budget is not None:
            self._budget_guard = BudgetGuard(cfg.budget)
            # Recovery and review spend on the same job as the solver. Restore
            # its durable budget before any path can start a scoring sandbox.
            attempts, interrupted = _recorded_attempts(self._jobs_dir / self._job_name)
            self._budget_guard.seed(attempts, interrupted=interrupted)
            self._budget_guard.restore(self._jobs_dir / self._job_name)

        await resume_pending_reviews(
            self._jobs_dir / self._job_name,
            tasks_root=self._tasks_dir,
            reviewer=self._config.reviewer,
            task_names={task_dir.name for task_dir in task_dirs},
            budget_guard=self._budget_guard,
        )
        completed = self._get_completed_tasks()
        remaining = [d for d in task_dirs if d.name not in completed]

        # Validate every selected review plan before spending on any solver.
        # Workers repeat this against their own effective credential context.
        _check_reviews(remaining)

        # Extra requirements for the model proxy are checked against its
        # locked set once, before any trial starts a sandbox; each
        # trial's setup repeats the check from the same cached answer.
        if remaining:
            from benchflow.providers.litellm_runtime import proxy_install_record

            proxy_install_record(
                agent=self._config.agent,
                model=self._config.model,
                agent_env=dict(self._config.agent_env or {}),
                environment=self._config.environment,
            )

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

        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._planned_tasks = [*completed, *(d.name for d in remaining)]
        self._write_evaluation_record()
        await self._sweep_docker()

        cfg = self._config

        if cfg.build_concurrency is not None and cfg.environment in (
            "docker",
            "remote-docker",
        ):
            from benchflow.sandbox.docker import DockerSandbox

            DockerSandbox.set_build_concurrency(cfg.build_concurrency)

        if cfg.environment in ("docker", "remote-docker"):
            # Before any trial starts: no trial may take a /28 inside a
            # subnet another task's network pins (_docker_subnets.py).
            _pin_evaluation_subnets(remaining, cfg)

        # The denominator is the number of tasks that will appear in the summary:
        # the resumed-complete set plus the to-run set (disjoint by construction).
        # This equals len(task_dirs) for a clean run, but a resume whose jobs_dir
        # holds results from a *wider* prior selection has more completed rows than
        # the current selection — keying off len(task_dirs) there rendered nonsense
        # like "11/1 · 1100%" and a denominator that disagreed with the final
        # "Score: 8/11". completed-plus-remaining is exactly what gets scored.
        planned_total = len(completed) + len(remaining)
        pool_note = (
            f", {warm_pool_note(warm_pool_size(cfg.concurrency, cfg.warm_pool))}"
            if cfg.environment == "docker"
            else ""
        )
        logger.info(
            f"Job: {planned_total} task{'' if planned_total == 1 else 's'}, "
            f"{len(completed)} done, "
            f"{len(remaining)} to run (concurrency={cfg.concurrency}{pool_note})"
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

        from benchflow.rollout._teardown_queue import background_teardown

        # Every task image the job lacks starts pulling now, a few at a
        # time, while the first trials run (sandbox/_prepull.py). Trials stop
        # their sandboxes in the background; every stop has ended before the
        # sweep (rollout/_teardown_queue.py). As PERF-ALL §2 resolved it.
        prepull = self._start_prepull(remaining)
        job_dir = self._jobs_dir / self._job_name
        stopped = False
        # The tasks finished before this run: a stop reloads ``completed``
        # from disk, with this run's trials in it (FIX-49 r3 L2).
        resumed = set(completed)
        try:
            async with background_teardown(limit=cfg.concurrency):
                if cfg.job_mode == "sequential-shared":
                    pairs = await self._run_sequential_shared(remaining)
                else:
                    pairs = await self._run_parallel_independent(remaining)
        except asyncio.CancelledError:
            # Ctrl-C or SIGTERM cancelled the job, and the trials' cleanup
            # has run. The summary below is then over the trials that wrote a
            # result, as the job's readers count them; summary.json says the
            # job stopped, and the stop goes on after it (DFR-p3-11).
            stopped = True
            pairs = []
            completed = {
                task: r
                for task, r in load_task_results(
                    job_dir, include_terminal_errors=True
                ).items()
                if task in self._planned_tasks
            }
        finally:
            if prepull is not None and not prepull.done():
                prepull.cancel()
                await asyncio.gather(prepull, return_exceptions=True)
        if not stopped:
            await self._sweep_docker()
        # A prepared call's wait for its caller is no part of the job.
        elapsed = time.time() - start - _park.admission_wait()

        task_paths = _resume_task_paths(self._tasks_dir, task_dirs)
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
            self._record_trial_without_folder(
                job_dir, name, payload, task_paths.get(name)
            )
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
        # Planned tasks a budget or a usage limit kept from finishing: not in
        # the score's denominator, but the job's plan (DF-p3-eval-runner-5).
        stopped_budget = (
            self._budget_guard.summary() if self._budget_guard is not None else {}
        )
        stopped_usage = self._usage_stop.summary() or {}
        not_run = sorted(
            (
                set(stopped_budget.get("cancelled") or ())
                | set(stopped_budget.get("not_started") or ())
                | set(stopped_usage.get("not_started") or ())
            )
            - set(all_results)
        )
        interrupted: list[str] = []
        if stopped:
            # Every planned task with no result: those whose trial started
            # (a trial folder with no result.json) and those never started.
            from benchflow.jobs import _attempt_task_name, _interrupted

            unfinished = set(self._planned_tasks) - set(all_results)
            started = {_attempt_task_name(p) for p in _interrupted(job_dir)}
            interrupted = sorted(unfinished & started)
            not_run = sorted(set(not_run) | unfinished)
        job_result = EvaluationResult(
            job_name=self._job_name,
            config=cfg,
            reused=len(resumed & set(completed)) if stopped else len(completed),
            ran=len(set(completed) - resumed) if stopped else len(pairs),
            # Score counts cover one entry per scored rollout. Skill-eval expands
            # a single task into multiple rollouts (baseline/skill x trials), so
            # the denominator must be the number of results, not task dirs, or the
            # invariant below (and every pass-rate/percentage) is wrong.
            total=len(all_results),
            passed=score_counts["passed"],
            failed=score_counts["failed"],
            errored=score_counts["errored"],
            verifier_errored=score_counts["verifier_errored"],
            guard_refused=count_guard_refusals(all_results.values()),
            elapsed_sec=elapsed,
            memory_score=memory["avg_score"],
            memory_scores=memory_scores,
            task_failures=task_failures,
            mean_reward=mean_reward_every_trial(all_results.values()),
            mean_reward_scored_only=mean_scored_reward(all_results.values()),
            mean_reward_n=len(all_results),
            mean_reward_scored_only_n=scored_trial_count(all_results.values()),
            mean_reward_caveat=mean_reward_caveat(all_results.values()),
            job_dir=job_dir,
            results=typed_results,
            budget=(
                self._budget_guard.summary() if self._budget_guard is not None else None
            ),
            not_run=len(not_run),
            stopped=stopped,
            interrupted=len(interrupted),
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
            "agent": cfg.agent,
            # Which BenchFlow wrote this summary (DF-p4-upgrader-3), after the
            # run and its agent (test_cli_ci reads those two first).
            "benchflow_version": _benchflow_version(),
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
            # score stays the pass rate over the tasks that ran; these give
            # the plan beside it: planned tasks, tasks that
            # ran, passes over planned tasks, and which did not run.
            "planned": job_result.planned,
            "ran": job_result.total,
            "score_of_plan": job_result.score_of_plan,
            **(
                {"not_run": job_result.not_run, "not_run_tasks": not_run}
                if not_run
                else {}
            ),
            # A stopped job's summary is partial: what finished, then which
            # planned tasks had started and which had not.
            **(
                {
                    "stopped": True,
                    "interrupted": len(interrupted),
                    "interrupted_tasks": interrupted,
                    "not_started": len(not_run) - len(interrupted),
                }
                if stopped
                else {}
            ),
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
        # The reviewer settings are recorded for every job; "ran" says
        # whether any trial was actually scored by one. After the run and its
        # counts: first in the file, a block naming another agent (opencode)
        # read as if that agent had run (DF-p1-first-run-12).
        summary["reviewer"] = {
            **cfg.reviewer.to_config_artifact(),
            "ran": any(r.get("scoring") is not None for r in all_results.values()),
        }
        isolation = verifier_isolation_summary(
            all_results,
            task_dirs={d.name: d for d in task_dirs},
            tasks_dir=self._tasks_dir,
            config_override=cfg.config_override,
        )
        summary["verifier_isolation"] = isolation
        _log_verifier_isolation(isolation)
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
        if cfg.seeds:
            # Seeded rollouts: per base task pass@k, mean/std, reset reproducibility.
            try:
                from benchflow.embodied.rollouts import seed_report

                summary["seeds"] = list(cfg.seeds)
                summary["seeded"] = seed_report(job_dir)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Seeded-rollout report failed: %s", e)
        summary_text = json.dumps(summary, indent=2)
        # Whole or not at all: a full disk leaves no 0-byte summary.json.
        from benchflow._utils.disk_space import write_text_atomic

        write_text_atomic(job_dir / "summary.json", summary_text)
        # Backward-compat: also write to jobs_dir root for tooling that
        # expects summary.json at the top level.
        write_text_atomic(self._jobs_dir / "summary.json", summary_text)
        if stopped:
            # The partial result, for --summary-out; then the stop goes on.
            self.result = job_result
            raise asyncio.CancelledError

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

        # dep-install failures don't have a structured diagnostic
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
        scored_mean = job_result.mean_reward_scored_only
        if (
            mean_part
            and scored_mean is not None
            and scored_mean != job_result.mean_reward
        ):
            # Unscored trials count as 0 here; 0.7's mean_reward left them out.
            mean_part = (
                f"mean_reward={job_result.mean_reward:.2f} (every trial; "
                f"{scored_mean:.2f} over scored ones, 0.7's mean_reward), "
            )
        logger.info(
            f"Job complete: {job_result.passed}/{job_result.total} "
            f"({job_result.score:.1%})"
            + (
                f" run, {job_result.passed}/{job_result.planned} "
                f"({job_result.score_of_plan:.1%}) of the plan"
                if job_result.not_run
                else ""
            )
            + f", {mean_part}errors={job_result.errored}, "
            f"idle_timeouts={error_category_counts.get(IDLE_TIMEOUT, 0)}, "
            f"time={elapsed / 60:.1f}min"
        )
        old_mean = getattr(self, "_pre_08_mean_reward", None)
        if (
            old_mean is not None
            and job_result.mean_reward is not None
            and round(old_mean, 2) != round(job_result.mean_reward, 2)
        ):
            # The same trials, another figure: say why (DF-p4-upgrader-1).
            scored = f"{scored_mean:.2f}" if scored_mean is not None else "none scored"
            logger.warning(
                f"This job's BenchFlow 0.7 summary said mean_reward={old_mean:.2f}, "
                "the mean over scored trials only. BenchFlow 0.8's mean_reward "
                f"counts every trial, an unscored one as 0: {job_result.mean_reward:.2f}; "
                f"its mean_reward_scored_only is 0.7's figure: {scored}."
            )

        self.result = job_result
        if self._usage_stop.error is not None:
            # The job stopped on its login's usage limit: raise the typed
            # error (with this result) so a caller can switch logins and
            # resume; summary.json and every finished trial are written.
            self._usage_stop.error.result = job_result
            raise self._usage_stop.error
        return job_result


# What _make_warm_pool checks trials still run through (see there).
_STOCK_RUN_SINGLE_TASK = Evaluation._run_single_task
