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
from benchflow.budget import Budget
from benchflow.errors import UserError
from benchflow.evaluation import DEFAULT_AGENT, EvaluationConfig, effective_model
from benchflow.loop_strategies import (
    SINGLE_SHOT,
    LoopStrategySpec,
    parse_loop_strategy_spec,
)
from benchflow.review.options import ReviewerConfig
from benchflow.sandbox.providers import (
    extra_install_hint,
    is_known_provider,
    provider_extra,
    providers_phrase,
    sandbox_sdk_missing,
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


class EvalPlanError(ValueError, UserError):
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
    harness: str | None = None
    environment: str | None = None
    usage_tracking: str | None = None
    environment_manifest: Path | None = None
    state: str | None = None
    config_override: str | None = None
    checkpoints: str | None = None
    checkpoint_keep: int = 3
    freeze_workspace: bool = False
    integrity: str | None = None
    retry_from_checkpoint: str | None = None
    retry_prompt: str | None = None
    retry_resume_session: bool = False
    max_cost_usd: float | None = None
    max_sandbox_seconds: float | None = None
    max_tokens: int | None = None
    prompt: list[str] | None = None
    concurrency: int | None = None
    build_concurrency: int | None = None
    worker_concurrency: int | None = None
    worker_retries: int = 1
    worker_start_stagger_sec: float = 1.0
    agent_idle_timeout: str | None = None
    jobs_dir: str | None = None
    # A new timestamped job instead of resuming the latest one in jobs_dir,
    # or an explicit job folder name (new or to resume).
    fresh: bool = False
    job_name: str | None = None
    # CI gates and a machine-readable result.
    fail_under: float | None = None
    fail_on: list[str] | None = None
    summary_out: Path | None = None
    codex_apps_policy: str | None = field(default=None, kw_only=True)
    sandbox_user: str | None = "agent"
    sandbox_setup_timeout: int = 120
    context_root: Path | None = None
    base_image_override: str | None = None
    skills_dir: Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    skill_creator_dir: Path | None = None
    self_gen_no_internet: bool = False
    loop_strategy: str | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    reviewer: ReviewerConfig | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    dataset: str | None = None
    registry: str | None = None
    ignore_bench_version: bool = False
    task_manifest_out: Path | None = None
    run_config_out: Path | None = None
    health_summary_out: Path | None = None
    expected_tasks: int | None = None
    canonicalize: str = "none"
    canonical_selection_out: Path | None = None
    canonical_jobs_dir: Path | None = None
    retry_policy: str = "default"
    retry_attempts: int | None = None
    retry_concurrency: int | None = None
    publish_hf: str | None = None
    hf_prefix: str | None = None
    hf_public_read_check: bool = False
    publish_bucket: str | None = None
    eval_results_model: str | None = None
    eval_results_dataset: str | None = None
    eval_results_task: str | None = None
    matrix: Path | None = None
    trials: int = 1
    seeds: str | None = None
    n_tasks: int | None = None
    sample_seed: int | None = None
    timeout_multiplier: float | None = None
    extra_instruction: str | None = None


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
    # The job folder to run into: None resumes the latest (the default).
    job_name: str | None
    eval_env_manifest: EnvironmentManifest | None
    eval_config_override: dict | None
    eval_loop_strategy: LoopStrategySpec | None
    parsed_env: dict[str, str]
    include_tasks: set[str]
    exclude_tasks: set[str]
    eval_seeds: list[int] | None = None
    # "acp" (default) or "native"; checked against the agent at planning.
    eval_harness: str = "acp"

    @property
    def eval_budget(self) -> Budget | None:
        """The hard per-job cap from --max-cost-usd/-sandbox-seconds/-tokens."""
        req = self.request
        caps = (req.max_cost_usd, req.max_sandbox_seconds, req.max_tokens)
        if all(c is None for c in caps):
            return None
        return Budget(
            max_cost_usd=req.max_cost_usd,
            max_sandbox_seconds=req.max_sandbox_seconds,
            max_tokens=req.max_tokens,
        )

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
            harness=self.eval_harness,
            environment=self.eval_environment,
            concurrency=self.eval_concurrency,
            build_concurrency=req.build_concurrency,
            prompts=self.eval_prompts,
            agent_idle_timeout=self.eval_agent_idle_timeout,
            checkpoints=req.checkpoints,
            checkpoint_keep=req.checkpoint_keep,
            freeze_workspace=req.freeze_workspace,
            integrity=req.integrity or "off",
            retry_from_checkpoint=req.retry_from_checkpoint,
            retry_prompt=req.retry_prompt,
            retry_resume_session=req.retry_resume_session,
            budget=self.eval_budget,
            agent_env=self.parsed_env,
            reviewer=ReviewerConfig.coerce(req.reviewer),
            codex_apps_policy=req.codex_apps_policy,
            sandbox_user=self.sandbox_user,
            sandbox_setup_timeout=req.sandbox_setup_timeout,
            context_root=str(req.context_root) if req.context_root else None,
            base_image_override=req.base_image_override,
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
            seeds=self.eval_seeds,
            n_tasks=req.n_tasks,
            sample_seed=req.sample_seed,
            timeout_multiplier=req.timeout_multiplier,
            extra_instruction=req.extra_instruction,
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
    _refuse_unknown_bare_agent(canonical_agent)
    return canonical_agent


def _refuse_unknown_bare_agent(name: str) -> None:
    """An unregistered one-word agent name is a usage error.

    It used to fall through to the raw-command path: the run created a job,
    defaulted the model and failed minutes later (or ran a misspelt name as a
    command). A command with arguments (``myagent --acp``) is still accepted
    as a raw ACP command, and namespaced specs (``acp:pi``) resolve as before.
    """
    from benchflow.agents.registry import is_scripted_agent, resolve_agent
    from benchflow.runtime import check_agent_names

    if is_scripted_agent(name) or any(c.isspace() for c in name):
        return
    try:
        # A close misspelling gets the SDK's "did you mean" message.
        check_agent_names([name])
    except ValueError as exc:
        raise EvalPlanError(f"{exc} No job was created.") from None
    try:
        resolve_agent(name)
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else f"Unknown agent: {name!r}"
        raise EvalPlanError(
            f"{message} (`bench agent list` shows the registered agents; to run a "
            "command as the agent, give it with its arguments, e.g. 'myagent --acp')"
        ) from None


FAIL_ON_CHOICES = ("timeout", "error", "verifier-error")


def split_fail_on(values: list[str] | None) -> list[str]:
    """``--fail-on timeout,error`` and repeated ``--fail-on`` flags, flattened."""
    return [v.strip() for raw in values or [] for v in raw.split(",") if v.strip()]


def _fresh_job_name(jobs_dir: Path) -> str:
    """A timestamped job name not yet used under ``jobs_dir``."""
    from datetime import datetime

    base = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    name, n = base, 1
    while (jobs_dir / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


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
    if request.matrix is not None and not request.tasks_dir:
        raise EvalPlanError("--matrix currently requires --tasks-dir")
    if request.trials < 1:
        raise EvalPlanError("--trials must be >= 1")
    if request.n_tasks is not None and request.n_tasks < 1:
        raise EvalPlanError("--n-tasks must be at least 1")
    if request.timeout_multiplier is not None and request.timeout_multiplier <= 0:
        raise EvalPlanError("--timeout-multiplier must be positive")
    eval_seeds: list[int] | None = None
    if request.seeds is not None:
        from benchflow.embodied.rollouts import parse_seeds

        try:
            eval_seeds = parse_seeds(request.seeds)
        except ValueError as exc:
            raise EvalPlanError(f"Invalid --seeds {request.seeds!r}: {exc}") from None
        if not (request.tasks_dir or request.source_repo):
            raise EvalPlanError("--seeds requires --tasks-dir or --source-repo")
        if request.matrix is not None:
            raise EvalPlanError("--seeds cannot be combined with --matrix")
        if request.worker_concurrency is not None:
            raise EvalPlanError("--seeds cannot be combined with --worker-concurrency")
    if request.trials > 1 and request.matrix is None:
        # Only the matrix expansion consumes trials; a plain run would silently
        # do one trial per task while the caller believes it ran N. Per-trial
        # requests created by run_matrix_eval bypass this planner by design.
        raise EvalPlanError(
            "--trials > 1 requires --matrix; to repeat one model, use a "
            'single-entry matrix such as "models: {default: <model>}"'
        )
    if request.expected_tasks is not None and request.expected_tasks < 1:
        raise EvalPlanError("--expected-tasks must be >= 1")
    if request.canonicalize not in {"none", "one-healthy-per-task"}:
        raise EvalPlanError("--canonicalize must be 'none' or 'one-healthy-per-task'")
    if request.canonical_selection_out and request.canonicalize == "none":
        raise EvalPlanError("--canonical-selection-out requires --canonicalize")
    if request.canonical_jobs_dir and not request.canonical_selection_out:
        raise EvalPlanError("--canonical-jobs-dir requires --canonical-selection-out")
    if request.retry_policy not in {"default", "unscored-only"}:
        raise EvalPlanError("--retry-policy must be 'default' or 'unscored-only'")
    if request.retry_attempts is not None and request.retry_attempts < 0:
        raise EvalPlanError("--retry-attempts must be >= 0")
    if request.retry_concurrency is not None and request.retry_concurrency < 1:
        raise EvalPlanError("--retry-concurrency must be >= 1")
    if request.hf_prefix and not (request.publish_hf or request.publish_bucket):
        raise EvalPlanError("--hf-prefix requires --publish-hf or --publish-bucket")
    if request.eval_results_model and not (
        request.eval_results_dataset and request.eval_results_task
    ):
        raise EvalPlanError(
            "--eval-results-model requires --eval-results-dataset and --eval-results-task"
        )
    if request.tasks_dir and not Path(request.tasks_dir).exists():
        raise EvalPlanError(f"--tasks-dir not found: {request.tasks_dir}")
    if request.matrix is not None and not Path(request.matrix).is_file():
        raise EvalPlanError(f"--matrix not found: {request.matrix}")
    if request.context_root and not Path(request.context_root).is_dir():
        raise EvalPlanError(f"--context-root not found: {request.context_root}")
    if request.base_image_override is not None:
        image = request.base_image_override.strip()
        if not image or any(char.isspace() for char in image):
            raise EvalPlanError(
                "--base-image-override must be a non-empty image reference"
            )
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
    if request.source_env and request.reviewer is not None:
        raise EvalPlanError(
            "--reviewer-* options require local tasks; --source-env owns its scoring"
        )
    for flag, value in (
        ("--max-cost-usd", request.max_cost_usd),
        ("--max-sandbox-seconds", request.max_sandbox_seconds),
        ("--max-tokens", request.max_tokens),
    ):
        if value is not None and not value > 0:
            raise EvalPlanError(f"{flag} must be > 0")
    has_budget = any(
        v is not None
        for v in (request.max_cost_usd, request.max_sandbox_seconds, request.max_tokens)
    )
    if has_budget and request.worker_concurrency is not None:
        raise EvalPlanError(
            "--max-cost-usd/--max-sandbox-seconds/--max-tokens cap one job and are "
            "not supported with --worker-concurrency (each worker would get the "
            "whole budget)"
        )
    if has_budget and request.source_env:
        raise EvalPlanError(
            "--max-cost-usd/--max-sandbox-seconds/--max-tokens are not supported "
            "with --source-env (vf-eval runs the rollouts)"
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
                    f"Install it with {extra_install_hint(provider_extra('modal') or 'sandbox-modal')}."
                ) from exc
        elif sandbox_sdk_missing(eval_environment):
            # Same fail-fast for the other extra-backed sandboxes: before this,
            # a missing Daytona SDK created the job, failed every rollout with
            # a chained traceback and recorded 0/N.
            extra = provider_extra(eval_environment) or f"sandbox-{eval_environment}"
            raise EvalPlanError(
                f"Missing optional dependency for {eval_environment!r} sandbox. "
                f"Install it with {extra_install_hint(extra)}."
            )
    eval_prompts = cast("list[str | None] | None", request.prompt)
    sandbox_user = normalize_sandbox_user(request.sandbox_user)
    if request.codex_apps_policy not in (None, "disabled", "inherit"):
        raise EvalPlanError("--codex-apps-policy must be disabled or inherit")
    if request.source_env and request.codex_apps_policy is not None:
        raise EvalPlanError(
            "--codex-apps-policy is not supported for hosted source environments"
        )
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
    from benchflow.native_harness.harnesses import check_harness, normalize_harness

    try:
        eval_harness = normalize_harness(request.harness)
        if request.harness is not None and request.config_file is None:
            check_harness(eval_harness, [eval_agent])
    except ValueError as exc:
        raise EvalPlanError(f"Invalid --harness {request.harness!r}: {exc}") from None
    if eval_harness != "acp" and request.source_env:
        raise EvalPlanError(
            "--harness is for BenchFlow agent rollouts; --source-env runs its own harness"
        )
    output_jobs_dir = request.jobs_dir or "jobs"
    if request.fresh and request.job_name:
        raise EvalPlanError("give --fresh or --job-name, not both")
    if (request.fresh or request.job_name) and request.worker_concurrency:
        raise EvalPlanError(
            "--fresh and --job-name are not supported with --worker-concurrency"
        )
    if request.fail_under is not None and not 0.0 <= request.fail_under <= 1.0:
        raise EvalPlanError("--fail-under is a pass rate between 0 and 1")
    unknown_gates = sorted(set(split_fail_on(request.fail_on)) - set(FAIL_ON_CHOICES))
    if unknown_gates:
        raise EvalPlanError(
            f"--fail-on {', '.join(unknown_gates)}: choose from "
            f"{', '.join(FAIL_ON_CHOICES)}"
        )
    job_name = request.job_name
    if request.fresh:
        job_name = _fresh_job_name(Path(output_jobs_dir))

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
        job_name=job_name,
        eval_env_manifest=eval_env_manifest,
        eval_config_override=eval_config_override,
        eval_loop_strategy=eval_loop_strategy,
        parsed_env=parsed_env,
        include_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
        eval_seeds=eval_seeds,
        eval_harness=eval_harness,
    )
