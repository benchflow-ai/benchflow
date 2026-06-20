"""Planning logic for ``bench eval run`` — the pure core of the CLI command.

``bench eval run`` (defined in :mod:`benchflow.cli.main`, pinned there by the
oracle-chokepoint tests) is a thin parse → request → plan → run → report shell.
This module owns the **plan** step: it takes an :class:`EvalCreateRequest` of raw
CLI flags and turns them into an :class:`EvalPlan` — the disambiguated source,
the normalized agent/model/timeout/effort values, the resolved Environment-plane
manifest, and a factory for the :class:`~benchflow.evaluation.EvaluationConfig`
shared by the ``--source-repo`` and ``--tasks-dir`` batch paths.

The planning logic has no console or process-exit side effects. Validation
failures raise :class:`EvalPlanError` carrying the exact message the CLI used to
print; the CLI catches it, renders it in red, and exits non-zero. This keeps the
"choose one source", "worker-concurrency needs a batch source", manifest-load,
usage-tracking, idle-timeout, and reasoning-effort rules in one testable place
while leaving the actual run/report (asyncio, hosted-env dispatch, result
printing) to the CLI shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from benchflow._utils.config import (
    DEFAULT_AGENT_IDLE_TIMEOUT_SEC,
    normalize_agent_idle_timeout,
    normalize_reasoning_effort,
    normalize_sandbox_user,
)
from benchflow.agents.registry import parse_agent_spec
from benchflow.evaluation import DEFAULT_AGENT, EvaluationConfig, effective_model
from benchflow.loop_strategies import (
    SINGLE_SHOT,
    LoopStrategySpec,
    parse_loop_strategy_spec,
)
from benchflow.sandbox.providers import (
    is_known_provider,
    provider_extra,
    providers_phrase,
)
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
)
from benchflow.usage_tracking import UsageTrackingConfig

if TYPE_CHECKING:
    from benchflow.environment.manifest import EnvironmentManifest

__all__ = [
    "EvalCreateRequest",
    "EvalPlan",
    "EvalPlanError",
    "build_eval_plan",
]


class EvalPlanError(ValueError):
    """A ``bench eval run`` validation failure.

    Carries the operator-facing ``message`` exactly as the CLI used to print it
    (without Rich markup). The CLI shell renders it ``[red]...[/red]`` and exits
    with code 1, preserving the original behavior.
    """


@dataclass
class EvalCreateRequest:
    """Raw ``bench eval run`` flags relevant to planning.

    Mirrors the subset of ``eval_run`` parameters consumed by
    :func:`build_eval_plan`. Execution-only flags (the ``source_env_*`` hosted-run
    knobs, ``source_path``/``source_ref``, and the worker run details) are passed
    straight through by the CLI and are not modeled here.
    """

    config_file: Path | None = None
    tasks_dir: Path | None = None
    source_repo: str | None = None
    source_env: str | None = None
    agent: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    environment: str | None = None
    usage_tracking: str | None = None
    environment_manifest: Path | None = None
    state: str | None = None
    config_override: str | None = None
    prompt: list[str] | None = None
    concurrency: int | None = None
    build_concurrency: int | None = None
    worker_concurrency: int | None = None
    worker_retries: int = 1
    worker_start_stagger_sec: float = 1.0
    agent_idle_timeout: str | None = None
    jobs_dir: str | None = None
    sandbox_user: str | None = "agent"
    sandbox_setup_timeout: int = 120
    context_root: Path | None = None
    skills_dir: Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    skill_creator_dir: Path | None = None
    self_gen_no_internet: bool = False
    loop_strategy: str | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    include: list[str] | None = None
    exclude: list[str] | None = None
    dataset: str | None = None
    registry: str | None = None
    ignore_bench_version: bool = False


@dataclass
class EvalPlan:
    """Normalized, validated inputs for the ``bench eval run`` run step.

    Holds every value the CLI shell needs after planning: the normalized
    agent/timeout/effort, the resolved manifest, the parsed include/exclude sets,
    and a :meth:`make_eval_config` factory for the ``EvaluationConfig`` shared by
    the ``--source-repo`` and ``--tasks-dir`` batch paths. Flags like
    ``usage_tracking_overridden`` and ``output_jobs_dir`` are precomputed so the
    shell stays declarative.
    """

    request: EvalCreateRequest
    eval_agent: str
    eval_reasoning_effort: str | None
    eval_environment: str
    eval_concurrency: int
    eval_prompts: list[str | None] | None
    eval_agent_idle_timeout: int | None
    eval_usage_tracking: UsageTrackingConfig
    usage_tracking_overridden: bool
    sandbox_user: str | None
    output_jobs_dir: str
    eval_env_manifest: EnvironmentManifest | None
    eval_config_override: dict | None
    eval_loop_strategy: LoopStrategySpec | None
    parsed_env: dict[str, str]
    include_tasks: set[str]
    exclude_tasks: set[str]

    def make_eval_config(
        self,
        source_provenance: dict[str, Any] | None = None,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        dataset_task_digests: dict[str, str] | None = None,
        include_tasks: set[str] | None = None,
    ) -> EvaluationConfig:
        """Build the ``EvaluationConfig`` shared by the source-repo / tasks-dir paths.

        ``effective_model`` is resolved here (lazily, per call) rather than during
        planning so that, exactly as before, the no-source path can fall through
        to its "provide a source" error without an agent-without-default-model
        first raising ``ValueError``.

        ``dataset_*`` are set only by the ``--dataset`` registry path, so every
        result.json/config.json from a pinned run carries its dataset identity
        and per-task content digest.
        """
        req = self.request
        return EvaluationConfig(
            agent=self.eval_agent,
            model=effective_model(self.eval_agent, req.model),
            reasoning_effort=self.eval_reasoning_effort,
            environment=self.eval_environment,
            concurrency=self.eval_concurrency,
            build_concurrency=req.build_concurrency,
            prompts=self.eval_prompts,
            agent_idle_timeout=self.eval_agent_idle_timeout,
            agent_env=self.parsed_env,
            sandbox_user=self.sandbox_user,
            sandbox_setup_timeout=req.sandbox_setup_timeout,
            context_root=str(req.context_root) if req.context_root else None,
            skills_dir=str(req.skills_dir) if req.skills_dir else None,
            skill_mode=req.skill_mode,
            skill_creator_dir=(
                str(req.skill_creator_dir) if req.skill_creator_dir else None
            ),
            self_gen_no_internet=req.self_gen_no_internet,
            source_provenance=source_provenance,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_task_digests=dataset_task_digests or {},
            include_tasks=(
                include_tasks if include_tasks is not None else self.include_tasks
            ),
            exclude_tasks=self.exclude_tasks,
            usage_tracking=self.eval_usage_tracking,
            environment_manifest=self.eval_env_manifest,
            config_override=self.eval_config_override,
            loop_strategy=self.eval_loop_strategy,
        )


def _normalize_eval_agent(agent_spec: str) -> str:
    """Normalize an eval agent spec, rejecting non-ACP protocols.

    Mirrors the old ``_normalize_eval_agent_or_exit`` rule but raises
    :class:`EvalPlanError` instead of exiting, so the CLI shell owns the exit.
    """
    protocol, canonical_agent = parse_agent_spec(agent_spec)
    if protocol not in ("acp", "acpx"):
        raise EvalPlanError(f"Unsupported eval agent protocol: {protocol}")
    if protocol == "acpx":
        return f"acpx/{canonical_agent}"
    return canonical_agent


def build_eval_plan(request: EvalCreateRequest) -> EvalPlan:
    """Validate and normalize ``bench eval run`` flags into an :class:`EvalPlan`.

    Raises :class:`EvalPlanError` for any validation failure, carrying the exact
    operator-facing message (no Rich markup). Performs no console or process-exit
    side effects.
    """
    parsed_env = request.agent_env
    include_tasks = set(request.include) if request.include else set()
    exclude_tasks = set(request.exclude) if request.exclude else set()

    sources = [
        bool(request.config_file),
        bool(request.tasks_dir),
        bool(request.source_repo),
        bool(request.source_env),
        bool(request.dataset),
    ]
    if sum(sources) > 1:
        raise EvalPlanError(
            "Choose only one source: --config, --tasks-dir, --source-repo, "
            "--source-env, or --dataset"
        )
    if request.registry and not request.dataset:
        raise EvalPlanError("--registry requires --dataset")
    if request.ignore_bench_version and not request.dataset:
        raise EvalPlanError("--ignore-bench-version requires --dataset")
    if request.tasks_dir and not Path(request.tasks_dir).exists():
        raise EvalPlanError(f"--tasks-dir not found: {request.tasks_dir}")
    # Validate --config here so a typo'd / missing / non-file path becomes a clean
    # CLI error instead of a raw FileNotFoundError/IsADirectoryError traceback from
    # the bare open() in Evaluation.from_yaml.
    if request.config_file and not Path(request.config_file).is_file():
        raise EvalPlanError(f"--config not found: {request.config_file}")
    # Validate the --source-repo shape up front (it's otherwise checked deep in
    # resolve_source_with_metadata, after planning, where the ValueError escapes
    # as a traceback). Match that resolver's semantics — split("/", 1) into two
    # non-empty parts — so this guard never rejects a shape the resolver accepts.
    if request.source_repo is not None:
        parts = str(request.source_repo).split("/", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise EvalPlanError(
                f"Invalid --source-repo {request.source_repo!r}; expected 'org/repo' "
                "(e.g. benchflow-ai/skillsbench)"
            )
    if request.worker_concurrency is not None and not (
        request.tasks_dir or request.source_repo
    ):
        raise EvalPlanError(
            "--worker-concurrency is supported for --tasks-dir and --source-repo batch runs"
        )
    if request.worker_retries < 0:
        raise EvalPlanError("--worker-retries must be >= 0")
    if request.worker_start_stagger_sec < 0:
        raise EvalPlanError("--worker-start-stagger-sec must be >= 0")

    eval_agent = (
        _normalize_eval_agent(request.agent)
        if request.agent is not None
        else DEFAULT_AGENT
    )
    # Only None means "unset" → default docker; an empty/whitespace --sandbox is a
    # typo and must reach the Invalid-sandbox check below, not be swallowed.
    eval_environment = (
        request.environment if request.environment is not None else "docker"
    )
    # --sandbox is ignored by hosted source-env runs (the hosted Verifiers
    # environment owns its harness), so only validate / preflight it for the
    # paths that actually use the local sandbox.
    if not request.source_env:
        if not is_known_provider(eval_environment):
            # Unknown sandbox values otherwise surface as a raw traceback per-task
            # once the rollout starts — reject them at planning instead.
            raise EvalPlanError(
                f"Invalid --sandbox {eval_environment!r}: choose {providers_phrase()}"
            )
        if eval_environment == "modal":
            # Fail fast with the actionable extra hint instead of surfacing a raw
            # ModuleNotFoundError deep inside the rollout (the in-sandbox guard in
            # sandbox/setup.py remains as defense-in-depth for programmatic callers).
            try:
                import modal  # noqa: F401
            except ModuleNotFoundError as exc:
                raise EvalPlanError(
                    "Missing optional dependency for 'modal' sandbox. "
                    f"Install it with `uv sync --extra {provider_extra('modal')}`."
                ) from exc
    eval_prompts = cast("list[str | None] | None", request.prompt)
    sandbox_user = normalize_sandbox_user(request.sandbox_user)
    eval_concurrency = request.concurrency if request.concurrency is not None else 4
    if eval_concurrency < 1:
        # A non-positive concurrency builds asyncio.Semaphore(0), which can never
        # be acquired and deadlocks the run — reject it up front instead.
        raise EvalPlanError(f"--concurrency must be >= 1 (got {eval_concurrency})")
    if request.build_concurrency is not None and request.build_concurrency < 1:
        raise EvalPlanError(
            f"--build-concurrency must be >= 1 (got {request.build_concurrency})"
        )
    if request.skill_mode not in {
        SKILL_MODE_NO_SKILL,
        SKILL_MODE_WITH_SKILL,
        SKILL_MODE_SELF_GEN,
    }:
        raise EvalPlanError(
            f"Invalid --skill-mode {request.skill_mode!r}: "
            "choose no-skill, with-skill, or self-gen"
        )
    eval_loop_strategy = None
    if request.loop_strategy is not None:
        try:
            eval_loop_strategy = parse_loop_strategy_spec(request.loop_strategy)
        except ValueError as exc:
            raise EvalPlanError(
                f"Invalid --loop-strategy {request.loop_strategy!r}: {exc}"
            ) from None
        k = eval_loop_strategy.params.get("k")
        if k is not None and not 1 <= k <= 10:
            raise EvalPlanError(
                f"Invalid --loop-strategy {request.loop_strategy!r}: "
                f"k must be between 1 and 10 (got {k})"
            )
        if eval_loop_strategy.name != SINGLE_SHOT:
            if request.prompt and len(request.prompt) > 1:
                raise EvalPlanError(
                    "--loop-strategy drives the prompt loop and conflicts "
                    "with multiple --prompt values"
                )
            if request.skill_mode == SKILL_MODE_SELF_GEN:
                raise EvalPlanError(
                    "--loop-strategy is not supported with --skill-mode self-gen"
                )
    if request.tasks_dir or request.source_repo:
        # Validate the agent/model pairing up front so an agent with no default
        # model (e.g. codex) reports a clean error instead of an uncaught
        # ValueError once the rollout starts. Only the --tasks-dir / --source-repo
        # paths take the model from --model here; --config and --source-env resolve
        # it from the YAML / hosted source later, so pre-validating those would
        # falsely reject a legitimately model-bearing config. The no-source case
        # is left to the CLI's "provide a source" error.
        try:
            effective_model(eval_agent, request.model)
        except ValueError as exc:
            raise EvalPlanError(str(exc)) from None

    usage_tracking_overridden = request.usage_tracking is not None
    try:
        eval_usage_tracking = UsageTrackingConfig(mode=request.usage_tracking)
    except (TypeError, ValueError) as exc:
        raise EvalPlanError(f"Invalid usage tracking config: {exc}") from None
    try:
        eval_agent_idle_timeout = normalize_agent_idle_timeout(
            request.agent_idle_timeout
            if request.agent_idle_timeout is not None
            else DEFAULT_AGENT_IDLE_TIMEOUT_SEC
        )
    except ValueError as exc:
        raise EvalPlanError(
            f"Invalid --agent-idle-timeout {request.agent_idle_timeout!r}: {exc}"
        ) from None
    try:
        eval_reasoning_effort = normalize_reasoning_effort(request.reasoning_effort)
    except ValueError as exc:
        raise EvalPlanError(
            f"Invalid --reasoning-effort {request.reasoning_effort!r}: {exc}"
        ) from None
    output_jobs_dir = request.jobs_dir or "jobs"

    # Resolve the optional Environment-plane manifest once and reuse across
    # every source branch (config / source_repo / tasks_dir / source_env).
    eval_env_manifest = None
    if request.state is not None:
        from benchflow._utils.env_registry import resolve_state

        try:
            eval_env_manifest = resolve_state(request.state)
        except (OSError, ValueError) as exc:
            raise EvalPlanError(f"Invalid --state: {exc}") from None
    elif request.environment_manifest is not None:
        from benchflow.environment.manifest import load_manifest

        try:
            eval_env_manifest = load_manifest(request.environment_manifest)
        except (OSError, ValueError) as exc:
            raise EvalPlanError(
                f"Could not load --environment-manifest {request.environment_manifest}: {exc}"
            ) from None

    # Parse + allowlist-validate the C-axis config overlay once (fail fast),
    # mirroring the manifest resolution above. Threaded as typed data from here.
    eval_config_override = None
    if request.config_override is not None:
        from benchflow._utils.config_override import (
            load_config_override,
            validate_overlay,
        )

        try:
            eval_config_override = load_config_override(request.config_override)
            if eval_config_override:
                validate_overlay(eval_config_override)
        except (OSError, ValueError) as exc:
            raise EvalPlanError(f"Invalid --config-override: {exc}") from None

    return EvalPlan(
        request=request,
        eval_agent=eval_agent,
        eval_reasoning_effort=eval_reasoning_effort,
        eval_environment=eval_environment,
        eval_concurrency=eval_concurrency,
        eval_prompts=eval_prompts,
        eval_agent_idle_timeout=eval_agent_idle_timeout,
        eval_usage_tracking=eval_usage_tracking,
        usage_tracking_overridden=usage_tracking_overridden,
        sandbox_user=sandbox_user,
        output_jobs_dir=output_jobs_dir,
        eval_env_manifest=eval_env_manifest,
        eval_config_override=eval_config_override,
        eval_loop_strategy=eval_loop_strategy,
        parsed_env=parsed_env,
        include_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
    )
