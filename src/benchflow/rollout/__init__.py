"""Rollout — the single execution path for one agent-on-task evaluation.

The ``Rollout`` class owns the 5-phase lifecycle::

    rollout = await Rollout.create(RolloutConfig(task_path=..., agent=..., model=...))
    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()
    await rollout.connect()
    await rollout.execute()
    result = await rollout.verify()
    await rollout.cleanup()

Or use ``rollout.run()`` for the full lifecycle.

Phases can be composed for multi-agent flows::

    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()

    # Coder turn
    await rollout.connect()
    await rollout.execute(prompts=[coder_prompt])
    await rollout.disconnect()

    # Reviewer turn (same sandbox, new ACP session)
    await rollout.connect()
    await rollout.execute(prompts=[reviewer_prompt])
    await rollout.disconnect()

    result = await rollout.verify()
    await rollout.cleanup()

See also: ``RolloutConfig`` for configuration dataclass.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._types import Role, Scene, Turn

# --- Façade re-exports -------------------------------------------------------
# ``rollout.py`` was split into a package; the cohesive helper groups now live in
# private submodules. Re-export every moved symbol — including the de-facto
# public underscore helpers that ``sdk.py``, ``self_gen.py``,
# ``task/acceptance_live.py`` and the test suite import — so every name that was
# importable from ``benchflow.rollout`` still resolves unchanged. The redundant
# ``import x as x`` aliases mark these as intentional re-exports.
#
# Re-importing these into this module's namespace also preserves patching: tests
# that patch ``benchflow.rollout._verify_rollout`` / ``_scrape_agent_trajectory``
# / ``_capture_session_trajectory`` / ``default_rollout_planes`` keep affecting
# the ``Rollout`` methods that call those names, because those methods stay
# defined in this module.
from benchflow._utils.scoring import classify_error as classify_error
from benchflow.acp.types import McpServerSpec
from benchflow.contracts import (
    AgentProtocolError,
    AskUserRequest,
    BaseUser,
    Environment,
    SandboxStartupFailure,
    default_rollout_planes,
)
from benchflow.contracts import RoundResult as RoundResult
from benchflow.diagnostics import (
    AgentPromptTimeoutError,
    ProviderApiErrorDiagnostic,
    RolloutDiagnostics,
    SuspectedApiErrorDiagnostic,
)
from benchflow.loop_strategies import (
    LoopStrategyUser,
    collect_loop_metadata,
    loop_block,
)
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.rollout._config import GENERATED_SKILLS_ROOT as GENERATED_SKILLS_ROOT
from benchflow.rollout._config import RolloutConfig as RolloutConfig
from benchflow.rollout._results import _DIAG_TRUNCATE as _DIAG_TRUNCATE
from benchflow.rollout._results import _build_rollout_result as _build_rollout_result
from benchflow.rollout._results import (
    _environment_manifest_metadata as _environment_manifest_metadata,
)
from benchflow.rollout._results import _is_secret_env_key as _is_secret_env_key
from benchflow.rollout._results import (
    _least_permissive_option_id as _least_permissive_option_id,
)
from benchflow.rollout._results import (
    _user_confirmation_policy as _user_confirmation_policy,
)
from benchflow.rollout._results import _write_config as _write_config
from benchflow.rollout._results import _write_rewards_jsonl as _write_rewards_jsonl
from benchflow.rollout._results import (
    _write_trainer_artifact as _write_trainer_artifact,
)
from benchflow.rollout._setup import (
    _agent_launch_with_web_policy as _agent_launch_with_web_policy,
)
from benchflow.rollout._setup import (
    _agent_process_kill_pattern as _agent_process_kill_pattern,
)
from benchflow.rollout._setup import _apply_web_policy as _apply_web_policy
from benchflow.rollout._setup import (
    _configured_task_workdir as _configured_task_workdir,
)
from benchflow.rollout._setup import (
    _ensure_canonical_rewards as _ensure_canonical_rewards,
)
from benchflow.rollout._setup import _ensure_sandbox_dir as _ensure_sandbox_dir
from benchflow.rollout._setup import (
    _environment_uses_prebuilt_image as _environment_uses_prebuilt_image,
)
from benchflow.rollout._setup import _init_rollout as _init_rollout
from benchflow.rollout._setup import _install_docker_compat as _install_docker_compat
from benchflow.rollout._setup import (
    _publish_trajectory_for_verifier as _publish_trajectory_for_verifier,
)
from benchflow.rollout._setup import _resolve_agent_cwd as _resolve_agent_cwd
from benchflow.rollout._setup import _resolve_prompts as _resolve_prompts
from benchflow.rollout._setup import _run_oracle as _run_oracle
from benchflow.rollout._setup import _skill_nudge as _skill_nudge
from benchflow.rollout._setup import _start_env_and_upload as _start_env_and_upload
from benchflow.rollout._setup import (
    _task_disallows_internet as _task_disallows_internet,
)
from benchflow.rollout._setup import _verify_rollout as _verify_rollout
from benchflow.rollout._skills import (
    _resolve_skill_creator_root as _resolve_skill_creator_root,
)
from benchflow.rollout._skills import _safe_skill_name as _safe_skill_name
from benchflow.rollout._skills import _self_gen_prompt as _self_gen_prompt
from benchflow.rollout._skills import _skill_frontmatter_name as _skill_frontmatter_name
from benchflow.rollout._usage import (
    _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT as _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT,
)
from benchflow.rollout._usage import (
    ProviderFailure as ProviderFailure,
)
from benchflow.rollout._usage import _as_nonnegative_int as _as_nonnegative_int
from benchflow.rollout._usage import _native_acp_usage_delta as _native_acp_usage_delta
from benchflow.rollout._usage import (
    _provider_api_failure_summary_from_runtime as _provider_api_failure_summary_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_auth_status_from_runtime as _provider_auth_status_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_failure_from_runtime as _provider_failure_from_runtime,
)
from benchflow.rollout._usage import (
    _provider_failure_from_status as _provider_failure_from_status,
)
from benchflow.rollout._usage import (
    _zero_native_acp_usage_metrics as _zero_native_acp_usage_metrics,
)
from benchflow.rollout._usage import classify_api_failure as classify_api_failure

# Step / user-loop drivers live in ``_user_loop`` as free functions taking the
# Rollout; the thin methods below delegate to these engine aliases.
from benchflow.rollout._user_loop import (
    _activate_step_skills as _activate_step_skills_engine,
)
from benchflow.rollout._user_loop import (
    _export_generated_skills as _export_generated_skills_engine,
)
from benchflow.rollout._user_loop import _run_steps as _run_steps_engine
from benchflow.rollout._user_loop import _run_user_loop as _run_user_loop_engine
from benchflow.rollout_branch import ChildRunner
from benchflow.rollout_branch import branch as _branch_engine
from benchflow.sandbox.metadata import persist_sandbox_info
from benchflow.scenes import compile_scenes_to_steps
from benchflow.scenes import scene_step_prompt as scene_step_prompt
from benchflow.scenes import scene_step_role as scene_step_role
from benchflow.skill_policy import SKILL_MODE_NO_SKILL as SKILL_MODE_NO_SKILL
from benchflow.skill_policy import (
    SKILL_MODE_SELF_GEN,
    TaskSkillPolicy,
    resolve_task_skill_policy,
    strip_task_bundled_skills,
    task_bundled_skills_dir,
)
from benchflow.skill_policy import SKILL_MODE_WITH_SKILL as SKILL_MODE_WITH_SKILL
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _scrape_agent_trajectory,
    make_trajectory_sink,
)
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step
from benchflow.usage_tracking import (
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    is_token_usage_available,
)

logger = logging.getLogger(__name__)


_MCP_TRANSPORT_TO_ACP_TYPE = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "http",
}


def _task_mcp_specs(task: Any) -> list[McpServerSpec]:
    """Map the task's ``[[environment.mcp_servers]]`` entries to ACP specs.

    This is the composition seam between the task-config layer
    (``MCPServerConfig``) and the ACP protocol layer (``McpServerSpec``) — kept
    here, in the rollout, so ``acp/`` stays free of any task-config dependency.
    The resulting specs are attached to every ACP session the rollout opens
    (``session/new``), making task-declared MCP servers — e.g. a Playwright MCP
    — reachable by the agent. Returns ``[]`` when the task declares none,
    preserving the historical default of attaching no MCP servers.
    """
    env_config = getattr(getattr(task, "config", None), "environment", None)
    configs = getattr(env_config, "mcp_servers", None) or []
    return [
        McpServerSpec(
            name=config.name,
            type=_MCP_TRANSPORT_TO_ACP_TYPE.get(config.transport, config.transport),
            command=config.command,
            args=list(config.args),
            url=config.url,
        )
        for config in configs
    ]


class Rollout:
    """Decomposed trial lifecycle with independently-callable phases."""

    def __init__(self, config: RolloutConfig) -> None:
        self._planes = config.planes or default_rollout_planes()
        # Activate Docker DinD compatibility shim on first rollout
        # construction (idempotent). Keeps `import benchflow.rollout`
        # side-effect free with respect to sandbox/provider behavior.
        _install_docker_compat(self._planes)

        self._config = config
        self._phase = "created"

        # Populated by setup()
        self._task: Any = None
        self._rollout_dir: Path | None = None
        self._rollout_paths: Any = None
        self._started_at: datetime | None = None
        self._job_name: str | None = None
        self._rollout_name: str | None = None
        self._agent_env: dict[str, str] = {}
        self._resolved_prompts: list[str] = []
        self._agent_launch: str = ""
        self._env: Any = None
        # When True, Rollout treats _env as caller-owned: setup() skips
        # creating a new sandbox and cleanup() skips stopping it. Set via
        # use_prebuilt_env() — see Runtime.execute() for the public path.
        self._env_externally_owned: bool = False
        self._environment: Environment | None = None
        self._timeout: int = 0
        self._timing: dict[str, float] = {}
        self._effective_locked: list[str] = []
        self._disallow_web_tools: bool = False
        self._effective_skills_dir: Path | None = None
        self._effective_skills_sandbox_dir: str | None = None
        # Task dir actually deployed: a temp copy (self._task_tmp) when
        # Dockerfile mutations are needed, otherwise config.task_path.
        # cleanup() removes the temp copy.
        self._effective_task_path: Path = config.task_path
        self._task_tmp: Path | None = None
        self._task_skill_policy: TaskSkillPolicy | None = None
        self._usage_runtime: Any = None
        self._usage_metrics: dict[str, Any] = self._planes.extract_usage(None)
        self._native_usage_metrics: dict[str, Any] = _zero_native_acp_usage_metrics()
        self._native_usage_checkpoint: dict[str, int | None] | None = None
        # Provider failure snapshotted during cleanup, after the usage proxy
        # imports its captures (Daytona's SandboxUsageProxy only fills trajectory
        # on stop()). Read by _provider_failure() so ACP-error classification can
        # expose a provider auth/rate-limit/outage failure (status code only)
        # instead of a generic ACP internal error (#546/#564).
        self._provider_failure_cached: ProviderFailure | None = None
        # Auth-only (401/403) view kept for callers added by PR #564.
        self._provider_auth_status_cached: int | None = None
        # Provider API failure summary (all statuses >= 400), snapshotted in
        # cleanup() alongside the auth status — consumed by the post-rollout
        # silent-API-failure classifier in _build_result().
        self._api_failure_summary_cached: dict[str, Any] | None = None

        # Populated by start()
        self._sandbox_id: str | None = None

        # Populated by install_agent()
        self._agent_cfg: Any = None
        self._agent_cwd: str = "/app"

        # Populated by connect()
        self._acp_client: Any = None
        self._session: Any = None
        # ``_session_adapter`` carries the Agent-plane :class:`Session` contract
        # over the live ACP client (architecture.md, "The four contracts").
        # The kernel registers ``on_ask_user`` handlers through it so the live
        # ``session/request_permission`` path runs the handler instead of the
        # auto-approve fallback (#382 follow-up — instantiating the adapter
        # here is what makes the wire path honour the contract).
        self._session_adapter: Any = None
        # Sticky ``on_ask_user`` handler. Stored on the rollout (not the
        # adapter) because connect()/_reconnect_for_role() rebuild the adapter
        # each time we attach a new agent process; the handler the caller
        # registered before the first connect must follow across reconnects.
        # The "ever set" flag distinguishes "never registered" (default
        # auto-approve policy) from "explicitly cleared" (caller passed
        # ``None`` to roll back to auto-approve).
        self._ask_user_handler: Any = None
        self._ask_user_handler_set: bool = False
        self._agent_name: str = ""
        self._active_role: Role | None = None
        # Cursors into the live ACP session's cumulative trajectory and
        # tool-call totals so execute() and the partial-capture path extend
        # only the delta since the last read. Reset per session in
        # install_agent().
        self._session_traj_count: int = 0
        self._session_tool_count: int = 0

        # Populated by execute()
        self._trajectory: list[dict] = []
        self._n_tool_calls: int = 0
        self._trajectory_source: TrajectorySource | None = None
        self._partial_trajectory: bool = False
        # Set when a clean wall-clock prompt timeout (AgentPromptTimeoutError
        # with no pending tool calls) fired — its captured trajectory is a
        # complete terminal one, not a rerunnable partial (#640).
        self._terminal_timeout: bool = False
        # Every prompt actually sent to the agent across all execute() calls —
        # this is what `n_prompts` and `prompts.json` should reflect for Scene
        # rollouts where each turn issues its own prompt. The original
        # `_resolved_prompts` is only the static base task prompt set.
        self._executed_prompts: list[str] = []
        # The user-loop engine's per-round log (aliased by _run_user_loop).
        # Rounds are appended as soon as their soft verify completes, so
        # _loop_strategy_metadata() can summarize whatever rounds finished
        # even when a later round timed out or crashed.
        self._user_rounds_log: list[dict[str, Any]] = []

        # The tree-native execution model (architecture.md, "tree-native").
        # A linear rollout is a degree-1 tree; execute() grows it one Step at a
        # time, and branch() forks a node into N children. The tree is additive
        # — it never alters linear behaviour or output.
        self._tree: RolloutTree = RolloutTree()
        self._cursor: RolloutNode = self._tree.root

        # Populated by verify()
        self._rewards: dict | None = None
        self._verifier_error: str | None = None
        self._error: str | None = None
        # Populated by _export_generated_skills() on failure (#389 follow-up).
        # Kept separate from self._error so classify_error() does not mis-tag
        # an export-time infra failure ("connection lost") as the agent's own
        # infra_failure category in dashboards.
        self._export_error: str | None = None
        # Single bag for the four parallel diagnostic fields the old code
        # carried as separate attrs (issue #503). Each callsite that used
        # to assign to one of those slots now calls ``self._diagnostics.set(...)``
        # with a typed Diagnostic value.
        self._diagnostics: RolloutDiagnostics = RolloutDiagnostics()

        # Populated by _export_generated_skills() — the skills the agent
        # generated/evolved, captured for a continual-learning LearnerStore.
        self._evolved_skills: dict[str, str] | None = None

    @classmethod
    async def create(cls, config: RolloutConfig) -> Rollout:
        """Create a Rollout instance. Preferred over __init__ for consistency."""
        if config.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.create()."
            )
        return cls(config)

    def use_prebuilt_env(self, inner: Any) -> None:
        """Inject a caller-owned sandbox; skip creation and teardown.

        When the public Runtime API receives a live ``Environment`` (one
        the caller already constructed, possibly started, and may want to
        reuse), the Rollout must evaluate inside that same sandbox rather
        than spinning up a second one. Call this before ``setup()``/
        ``run()``: ``setup()`` will then skip ``_create_environment`` and
        ``cleanup()`` will skip stopping the sandbox — the caller owns the
        lifecycle. Fixes #388.
        """
        if inner is None:
            raise ValueError("use_prebuilt_env() requires a non-None sandbox")
        self._env = inner
        self._env_externally_owned = True

    @property
    def env(self) -> Any:
        return self._env

    @property
    def acp_client(self) -> Any:
        return self._acp_client

    @property
    def trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def tree(self) -> RolloutTree:
        """The RolloutTree this rollout grows as it executes.

        A linear rollout is a degree-1 tree; :meth:`branch` forks a node into
        N children. ``tree.root`` is the start state s₀.
        """
        return self._tree

    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    @property
    def result(self) -> RolloutResult | None:
        if self._phase not in ("verified", "cleaned"):
            return None
        return self._build_result()

    def _require_rollout_dir(self) -> Path:
        if self._rollout_dir is None:
            raise RuntimeError("Rollout.setup() must run before this phase")
        return self._rollout_dir

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise RuntimeError("Rollout.setup() must run before building a result")
        return self._started_at

    # Phase 1: SETUP (host-side, no container yet)

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — oracle files stay visible "
                "to the agent for the entire trial."
            )

        self._effective_locked = self._planes.resolve_locked_paths(
            cfg.sandbox_user, cfg.sandbox_locked_paths
        )

        (
            self._task,
            self._rollout_dir,
            self._rollout_paths,
            self._started_at,
            self._job_name,
            self._rollout_name,
        ) = _init_rollout(cfg.task_path, cfg.job_name, cfg.rollout_name, cfg.jobs_dir)

        self._disallow_web_tools = (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        ) and cfg.primary_agent != "oracle"
        self._agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                cfg.primary_agent, cfg.primary_model, cfg.agent_env
            ),
            disallow=self._disallow_web_tools,
        )
        env_config = getattr(getattr(self._task, "config", None), "environment", None)
        task_skill_policy = resolve_task_skill_policy(
            task_path=cfg.task_path,
            skill_mode=cfg.recorded_skill_mode,
            runtime_skills_dir=cfg.skills_dir,
            declared_sandbox_skills_dir=getattr(env_config, "skills_dir", None),
        )
        self._task_skill_policy = task_skill_policy
        self._resolved_prompts = _resolve_prompts(
            cfg.task_path,
            cfg.prompts,
            skills_dir=task_skill_policy.prompt_dir,
            skill_nudge=_skill_nudge(cfg.agent_env),
            agent=cfg.primary_agent,
            planes=self._planes,
        )
        self._agent_launch = self._planes.agent_launch(
            cfg.primary_agent,
            disallow_web_tools=self._disallow_web_tools,
        )

        # Copy task dir to temp when Dockerfile mutations are needed
        # (_inject_skills writes into environment/_deps/, stage_dockerfile
        # rewrites COPY paths — neither should modify the source tree)
        effective_task_path = cfg.task_path
        if cfg.context_root or task_skill_policy.needs_task_copy:
            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            shutil.copytree(cfg.task_path, tmp / cfg.task_path.name, dirs_exist_ok=True)
            effective_task_path = tmp / cfg.task_path.name
            self._task_tmp = tmp
            if task_skill_policy.strip_bundled_dir_from_copy:
                strip_task_bundled_skills(effective_task_path)

        if cfg.context_root:
            self._planes.stage_dockerfile_deps(
                effective_task_path, Path(cfg.context_root)
            )
        effective_skills_dir = task_skill_policy.host_dir
        if (
            effective_skills_dir is not None
            and task_skill_policy.host_dir_is_bundled
            and effective_task_path != cfg.task_path
        ):
            effective_skills_dir = task_bundled_skills_dir(effective_task_path)
        if effective_skills_dir is not None and not _environment_uses_prebuilt_image(
            env_config, cfg.environment_manifest
        ):
            self._planes.inject_skills_into_dockerfile(
                effective_task_path,
                effective_skills_dir,
                sandbox_dir=task_skill_policy.sandbox_dir or "/skills",
            )

        task_skill_policy = replace(task_skill_policy, host_dir=effective_skills_dir)
        self._task_skill_policy = task_skill_policy
        self._effective_task_path = effective_task_path
        self._effective_skills_dir = effective_skills_dir
        self._effective_skills_sandbox_dir = task_skill_policy.sandbox_dir

        # Honour an externally-supplied sandbox (use_prebuilt_env, set by
        # Runtime.execute() when the caller passes a live Environment).
        # Without this guard, every Runtime.execute() would build a second
        # sandbox and silently discard the caller's prepared one — #388.
        if self._env is None:
            self._env = self._planes.create_environment(
                cfg.environment,
                self._task,
                effective_task_path,
                self._rollout_name,
                self._rollout_paths,
                preserve_agent_network=self._disallow_web_tools,
                environment_manifest=cfg.environment_manifest,
            )
        # Caller-supplied wall-clock budget (e.g. RuntimeConfig.timeout)
        # wins over the task's own default. Without this override there is
        # no way to tighten/loosen the agent budget per run — see #378.
        if cfg.timeout is not None:
            self._timeout = int(cfg.timeout)
        else:
            self._timeout = int(self._task.config.agent.timeout_sec or 0)

        _write_config(
            self._rollout_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            reasoning_effort=cfg.primary_reasoning_effort,
            environment=cfg.environment,
            environment_manifest=cfg.environment_manifest,
            skill_policy=task_skill_policy,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
            usage_tracking=cfg.usage_tracking.with_env_defaults(),
            concurrency=cfg.concurrency,
            agent_idle_timeout=cfg.agent_idle_timeout,
            scenes=cfg.effective_scenes,
            source_provenance=cfg.source_provenance,
            dataset=cfg.dataset,
            task_digest=cfg.task_digest,
            loop_strategy=cfg.loop_strategy_spec,
        )

        self._phase = "setup"

    # Phase 2: START (container comes up)

    async def start(self) -> None:
        """Start the environment and upload task files."""

        def _capture_and_persist_sandbox() -> None:
            # Persist the sandbox id the moment the sandbox exists, before any
            # upload that could fail or be interrupted (#554/#563). Otherwise a
            # mid-upload failure leaves a live Daytona sandbox with no
            # sandbox.json to audit or clean up.
            sid = getattr(self._env, "sandbox_id", None)
            self._sandbox_id = sid if isinstance(sid, str) else None
            persist_sandbox_info(self._env, self._rollout_dir)

        await _start_env_and_upload(
            self._env,
            self._config.task_path,
            self._timing,
            skip_start=self._env_externally_owned,
            on_started=_capture_and_persist_sandbox,
        )

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        # Environment plane: provision the manifest-declared stateful
        # environment and gate on its readiness before the agent runs.
        if self._config.environment_manifest is not None:
            self._environment = self._planes.manifest_environment(
                self._config.environment_manifest, sandbox=self._env
            )
            await self._environment.provision(
                ctx={"task_id": self._config.task_path.name}
            )
            probe = await self._environment.readiness()
            if not probe.ready:
                raise RuntimeError(
                    f"environment plane not ready: {probe.error} "
                    f"(checked: {probe.checked})"
                )
            logger.info(
                "environment '%s' ready (%d probe(s))",
                self._config.environment_manifest.name,
                len(probe.checked),
            )

        self._phase = "started"

    # Phase 3: INSTALL AGENT

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous scene-authored steps (different agents per role),
        each role's agent is installed on-demand in connect_as().
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()

        self._agent_cwd = await _resolve_agent_cwd(self._env, self._task)

        if cfg.primary_agent == "oracle":
            if cfg.sandbox_user:
                await self._planes.setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            await self._planes.snapshot_build_config(
                self._env, workspace=self._agent_cwd
            )
            await self._planes.seed_verifier_workspace(
                self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
            )
            await self._planes.deploy_skills(
                self._env,
                self._effective_task_path,
                self._effective_skills_dir,
                None,
                cfg.sandbox_user,
                self._agent_cwd,
                skills_sandbox_dir=self._effective_skills_sandbox_dir,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await self._planes.lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        self._agent_cfg = await self._planes.install_agent(
            self._env,
            agent_name,
            rollout_dir,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
        )
        if cfg.sandbox_user:
            self._agent_cwd = await self._planes.setup_sandbox_user(
                self._env,
                cfg.sandbox_user,
                workspace=self._agent_cwd,
                timeout_sec=cfg.sandbox_setup_timeout,
            )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await self._planes.write_credential_files(
            self._env,
            agent_name,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await self._planes.upload_subscription_auth(
                self._env, agent_name, cred_home
            )
        await self._planes.apply_web_tool_policy(
            self._env,
            agent_name,
            self._agent_cfg,
            cred_home,
            disallow=self._disallow_web_tools,
        )
        await self._planes.snapshot_build_config(self._env, workspace=self._agent_cwd)
        await self._planes.seed_verifier_workspace(
            self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
        )

        await self._planes.deploy_skills(
            self._env,
            self._effective_task_path,
            self._effective_skills_dir,
            self._agent_cfg,
            cfg.sandbox_user,
            self._agent_cwd,
            skills_sandbox_dir=self._effective_skills_sandbox_dir,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await self._planes.lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # Phase 3b: CONNECT (ACP session — re-entrant)

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        (
            self._agent_env,
            self._usage_runtime,
        ) = await self._planes.ensure_litellm_runtime(
            agent=cfg.primary_agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
        )
        (
            self._acp_client,
            self._session,
            self._session_adapter,
            self._agent_name,
        ) = await self._planes.connect_acp(
            env=self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.primary_model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
            reasoning_effort=cfg.primary_reasoning_effort,
            mcp_servers=_task_mcp_specs(getattr(self, "_task", None)),
        )
        self._native_usage_checkpoint = None
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    def _attach_trajectory_writer(self, rollout_dir: Path) -> None:
        """Wire the current session's ``on_change`` to stream cumulative
        trajectory to ``rollout_dir/trajectory/acp_trajectory.jsonl``.

        The sink prepends ``self._trajectory`` (events from prior scenes,
        captured by value at wire-up time) so multi-scene rollouts don't
        overwrite earlier scenes' events with the current session's
        snapshot.
        """
        if self._session is None or rollout_dir is None:
            return
        prior: list[dict] = getattr(self, "_trajectory", []) or []
        traj_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        self._session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), prior
        )

    async def disconnect(self) -> None:
        """Close the ACP client and clean up agent process, keeping the environment alive."""
        self._capture_partial_acp_trajectory()
        if self._acp_client:
            try:
                await self._acp_client.close()
            except Exception as e:
                logger.warning(f"ACP client close failed: {e}")
            self._acp_client = None
            self._session = None
            self._session_adapter = None
        # Kill any lingering agent processes to prevent context bleed between scenes
        agent_pattern = _agent_process_kill_pattern(self._agent_launch)
        if self._env and agent_pattern:
            with contextlib.suppress(Exception):
                await self._env.exec(
                    f"pkill -f {shlex.quote(agent_pattern)} || true",
                    timeout_sec=10,
                )
        self._active_role = None
        self._session_tool_count = 0
        self._session_traj_count = 0
        self._phase = "installed"

    def on_ask_user(self, handler: Any) -> None:
        """Register the agent-initiated ``session/request_permission`` handler.

        Forwards to :meth:`ACPSessionAdapter.on_ask_user` on the live adapter
        so the handler runs on the wire path; before #382's follow-up the
        adapter was never instantiated in production and the auto-approve
        policy ran unconditionally. The handler is sticky — stored on the
        rollout so reconnects (e.g. ``_reconnect_for_role``) re-register it
        on the freshly bound adapter via :meth:`_reapply_ask_user_handler`.

        Pass ``None`` to clear; the client's most-permissive auto-approve
        policy takes over (preserves the benchmark-mode default).
        """
        self._ask_user_handler = handler
        self._ask_user_handler_set = True
        self._reapply_ask_user_handler()

    def _reapply_ask_user_handler(self) -> None:
        """Re-bind any registered ``on_ask_user`` handler to the live adapter.

        ``getattr`` defaults guard ``Rollout`` instances built via
        ``__new__`` in tests that pre-date the on_ask_user field — they
        skip ``__init__`` and only set the attributes their scenarios use.
        """
        adapter = getattr(self, "_session_adapter", None)
        if adapter is None:
            return
        # No-op when the caller never touched on_ask_user — leaves the
        # default auto-approve path alone and avoids redundant client calls
        # from the connect()/_reconnect_for_role() hot paths.
        if not getattr(self, "_ask_user_handler_set", False):
            return
        handler = getattr(self, "_ask_user_handler", None)
        if handler is None:
            # Explicit clear — drop the bridge closure on the client so
            # the default most-permissive policy takes over.
            client = getattr(self, "_acp_client", None)
            if client is not None:
                client.on_ask_user(None)
            return
        adapter.on_ask_user(handler)

    def _install_document_confirmation_handler(self, user: BaseUser) -> bool:
        """Install a fail-closed permission handler for document human policy.

        ``confirmation_policy: human`` means BenchFlow must not silently fall
        back to ACP's benchmark-mode auto-approve path. If a caller already
        registered an explicit ``on_ask_user`` handler we treat that as the
        human/policy hook and leave it alone; otherwise the non-interactive
        default denies/rejects permission requests when a deny option exists.
        """

        if _user_confirmation_policy(user) != "human":
            return False
        if getattr(self, "_ask_user_handler", None) is not None:
            return False

        async def _deny_without_human(request: AskUserRequest) -> str:
            option = _least_permissive_option_id(
                request.options,
                request.option_kinds,
            )
            logger.info(
                "Document confirmation_policy=human denied ask_user request "
                "%s with option %s",
                request.request_id,
                option,
            )
            return option

        self.on_ask_user(_deny_without_human)
        return True

    def _capture_partial_acp_trajectory(self) -> None:
        """Append the live session's uncaptured tail to ``self._trajectory``.

        Runs on the disconnect / cleanup path when ``execute_prompts`` may
        have raised before the normal extend in :meth:`execute`. Uses
        ``_session_traj_count`` as the pointer to events already extended
        from this session so a partial scene's events are preserved on top
        of any prior scenes' (already-captured) events — see PR #566 review.
        """
        # Defensive lookup tolerates bare ``object()`` stubs used by older
        # rollout tests that pre-date the live-session partial-capture path.
        session = (
            getattr(self._acp_client, "session", None) if self._acp_client else None
        )
        if session is None:
            return
        try:
            captured = _capture_session_trajectory(session)
        except Exception as e:
            logger.warning(f"Partial trajectory capture failed: {e}")
            return
        delta = captured[getattr(self, "_session_traj_count", 0) :]
        if not delta:
            return
        self._trajectory.extend(delta)
        self._session_traj_count = len(captured)
        if getattr(self, "_terminal_timeout", False):
            # Clean wall-clock terminal timeout (#640): the captured tail is the
            # complete trajectory, so leave _partial_trajectory False.
            self._trajectory_source = "acp"
        else:
            self._partial_trajectory = True
            self._trajectory_source = "partial_acp"
        prior_session_tools = getattr(self, "_session_tool_count", 0)
        new_tools = len(session.tool_calls) - prior_session_tools
        if new_tools > 0:
            self._n_tool_calls += new_tools
        self._session_tool_count = len(session.tool_calls)

    # Phase 3c: EXECUTE

    async def execute(
        self, prompts: list[str] | None = None, *, node: RolloutNode | None = None
    ) -> tuple[list[dict], int]:
        """Run prompts through the ACP session. Returns (new trajectory, new tool calls).

        execute_prompts returns cumulative session trajectory. We track
        what we've already captured to avoid duplication when the same
        session is reused across multiple turns.

        ``node`` — when given, a *pending* tree node (no incoming Step yet,
        from :meth:`RolloutTree.attach`) whose Step this call fills in place,
        instead of advancing the tree with a fresh child. The Branch engine
        passes a pre-attached branch-child node here so the child's real
        continuation Step lands on the child node itself.
        """
        effective_prompts = prompts or self._resolved_prompts
        if self._acp_client is None:
            raise RuntimeError("Rollout.connect() must run before execute()")
        prev_session_tools = self._session_tool_count
        t0 = datetime.now()
        active_role = getattr(self, "_active_role", None)
        timeout = (
            active_role.timeout_sec
            if active_role and active_role.timeout_sec is not None
            else self._timeout
        )
        idle_timeout = (
            active_role.idle_timeout_sec
            if active_role and active_role.idle_timeout_sec is not None
            else self._config.agent_idle_timeout
        )

        try:
            trajectory, n_tool_calls = await self._planes.execute_prompts(
                self._acp_client,
                self._session,
                effective_prompts,
                timeout,
                idle_timeout=idle_timeout,
            )
        except AgentPromptTimeoutError as e:
            self._diagnostics.set(e.diagnostic)
            self._commit_acp_execution(
                trajectory=e.trajectory,
                n_tool_calls=e.n_tool_calls,
                prev_session_tools=prev_session_tools,
                effective_prompts=e.executed_prompts or effective_prompts,
                started_at=t0,
                node=node,
                partial_trajectory=not e.terminal_trajectory_complete,
            )
            raise

        self._commit_acp_execution(
            trajectory=trajectory,
            n_tool_calls=n_tool_calls,
            prev_session_tools=prev_session_tools,
            effective_prompts=effective_prompts,
            started_at=t0,
            node=node,
        )
        return trajectory, n_tool_calls

    def _commit_acp_execution(
        self,
        *,
        trajectory: list[dict],
        n_tool_calls: int,
        prev_session_tools: int,
        effective_prompts: list[str],
        started_at: datetime,
        node: RolloutNode | None,
        partial_trajectory: bool = False,
    ) -> None:
        """Commit a finalized ACP snapshot into rollout state."""

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[self._session_traj_count :]
        self._session_tool_count = n_tool_calls
        self._session_traj_count = len(trajectory)

        self._trajectory.extend(new_events)
        self._n_tool_calls += new_tools
        self._executed_prompts.extend(effective_prompts)
        if partial_trajectory:
            self._partial_trajectory = True
            self._trajectory_source = "partial_acp"
        elif not self._partial_trajectory:
            self._trajectory_source = "acp"
        self._collect_native_acp_usage()

        # Grow the tree at Step-level granularity — one Step per ACP event
        # (tool_call, agent_message, agent_thought, user_message). A single
        # execute() call walks the cursor down N nodes when it produced N
        # events. Closes #414: branch/process-reward/value targets the
        # individual action, not a collapsed turn.
        #
        # Empty-event executes still emit one Step so the tree advances at
        # least once per execute() call — the cursor must move, and a branch
        # child's pending node must get populated (see rollout_branch).
        steps = self._build_step_batch(new_events, new_tools)
        first_step, *rest_steps = steps
        if node is not None:
            # Fill a pre-attached pending node (a branch child) in place — the
            # child's real continuation Step lands on the child node itself.
            self._cursor = self._tree.populate(node, first_step)
        else:
            self._cursor = self._tree.advance(self._cursor, first_step)
        for step in rest_steps:
            self._cursor = self._tree.advance(self._cursor, step)

        # Accumulate execution time across all execute() calls — Scene rollouts
        # invoke execute() once per turn, and the previous "set only on first
        # call" behaviour undercounted multi-turn agent time.
        elapsed = (datetime.now() - started_at).total_seconds()
        self._timing["agent_execution"] = (
            self._timing.get("agent_execution", 0.0) + elapsed
        )

        self._phase = "executed"

    def _collect_native_acp_usage(self) -> None:
        """Accumulate ACP PromptResponse.usage deltas for native subscription runs."""
        session = getattr(self, "_session", None)
        latest_fn = getattr(session, "latest_usage_totals", None)
        if not callable(latest_fn):
            return
        latest = latest_fn()
        if not latest:
            return
        previous = getattr(self, "_native_usage_checkpoint", None)
        delta = _native_acp_usage_delta(previous, latest)
        self._native_usage_checkpoint = dict(latest)
        if not any(delta.values()):
            return

        metrics = dict(
            getattr(self, "_native_usage_metrics", _zero_native_acp_usage_metrics())
        )
        for (
            snapshot_field,
            result_field,
        ) in _NATIVE_ACP_USAGE_SNAPSHOT_TO_RESULT.items():
            if result_field == "total_tokens":
                continue
            metrics[result_field] = _as_nonnegative_int(metrics.get(result_field)) + (
                delta.get(snapshot_field) or 0
            )
        metrics["total_tokens"] = _as_nonnegative_int(metrics.get("total_tokens")) + (
            delta.get("total_tokens") or 0
        )
        details = dict(metrics.get("usage_details") or {})
        details["thought_tokens"] = _as_nonnegative_int(
            details.get("thought_tokens")
        ) + (delta.get("thought_tokens") or 0)
        metrics["usage_details"] = details
        metrics["usage_source"] = USAGE_SOURCE_AGENT_NATIVE_ACP
        metrics["cost_usd"] = None
        metrics["price_source"] = None
        self._native_usage_metrics = metrics

    def _build_step_batch(self, new_events: list[dict], new_tools: int) -> list[Step]:
        """Build one Step per ACP event from the events appended this execute.

        Step-level granularity (closes #414) — each ACP event (tool_call,
        agent_message, agent_thought, user_message) becomes a Step the tree
        can address for branching, reward shaping, and value estimation.
        Empty-event executes still produce one Step so the cursor advances
        and any pending branch-child node gets populated.
        """
        base = len(self._trajectory) - len(new_events)
        if not new_events:
            return [
                Step(
                    id=f"step-{base}-empty",
                    data={"event": None, "n_tool_calls": 0},
                )
            ]
        steps: list[Step] = []
        for offset, event in enumerate(new_events):
            traj_index = base + offset
            event_type = (
                event.get("type", "event") if isinstance(event, dict) else "event"
            )
            is_tool_call = event_type == "tool_call"
            steps.append(
                Step(
                    id=f"step-{traj_index}-{event_type}",
                    data={
                        "event": event,
                        "event_type": event_type,
                        "n_tool_calls": 1 if is_tool_call else 0,
                    },
                )
            )
        # n_tool_calls across the batch should equal the new_tools reported
        # by execute_prompts. If they disagree (legacy/non-tool_call events
        # counted as tools by the agent shim) attribute the remainder to the
        # last step so the per-execute total still matches.
        batch_tools = sum(s.data["n_tool_calls"] for s in steps)
        if batch_tools != new_tools and steps:
            steps[-1].data["n_tool_calls"] += new_tools - batch_tools
        return steps

    # Phase 3d: BRANCH

    async def branch(
        self,
        n: int,
        run_child: ChildRunner | None = None,
        *,
        require_sandbox_snapshot: bool = False,
    ) -> float:
        """Branch the rollout at the cursor into ``n`` child continuations.

        Thin entry point — the Branch engine lives in
        :mod:`benchflow.rollout_branch`. It checkpoints the Environment at the
        cursor, runs each forked child as an isolated sub-rollout (its own
        scoped state, a fresh agent session), and aggregates the children's
        returns into V(parent). After this returns, the rollout's linear state
        is exactly what it was before.

        ``run_child`` is the per-child runner — injected for unit tests; the
        default restores the env, connects a fresh agent, runs the continuation,
        and scores it. A caller that needs per-child prompts binds them into the
        ``run_child`` closure.

        ``require_sandbox_snapshot`` gates the branch on the active Sandbox
        implementing container-level snapshot/restore. When True, providers
        without that capability (Modal, Daytona DinD) fail closed with a clear
        diagnostic rather than running with a half-consistent checkpoint
        (#384, Branch lifecycle in docs/architecture.md).
        """
        return await _branch_engine(
            self, n, run_child, require_sandbox_snapshot=require_sandbox_snapshot
        )

    # Phase 4: VERIFY

    async def verify(self) -> dict | None:
        """Run the verifier and return rewards."""
        cfg = self._config

        if not self._trajectory and cfg.primary_agent != "oracle":
            scraped = await _scrape_agent_trajectory(
                self._env, cfg.primary_agent, cfg.sandbox_user
            )
            if scraped:
                self._trajectory = scraped
                self._trajectory_source = "scraped"
                logger.warning(
                    f"Using scraped trajectory ({len(scraped)} events) — UNTRUSTED"
                )

        await _publish_trajectory_for_verifier(
            self._env, self._trajectory, self._rollout_paths.agent_dir
        )

        (
            self._rewards,
            self._verifier_error,
            verifier_timeout_diag,
        ) = await _verify_rollout(
            self._env,
            self._task,
            self._rollout_paths,
            self._timing,
            self._planes,
            sandbox_user=cfg.sandbox_user,
            workspace=self._agent_cwd,
        )
        if verifier_timeout_diag is not None:
            self._diagnostics.set(verifier_timeout_diag)

        self._phase = "verified"
        return self._rewards

    async def soft_verify(self) -> tuple[dict | None, str | None, str | None]:
        """Run the verifier without full hardening — for intermediate feedback.

        Skips process kill and workspace restore/chown (so the sandbox
        stays usable for the next round), but DOES purge agent-injected
        conftest.py / sitecustomize.py / .pth files to prevent the agent
        from gaming intermediate test results.

        Returns (rewards, verifier_output, verifier_error). The final
        verify() still does full hardening.
        """
        self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        try:
            await self._planes.clear_verifier_output_dir(
                self._env,
                "Soft verifier setup failed: clearing verifier output directory",
                user="root",
                timeout_sec=10,
            )
            await self._planes.ensure_legacy_app_dir(
                self._env,
                "Soft verifier setup failed: preparing /app",
                user="root",
                timeout_sec=10,
            )
            # Purge agent-injected conftest/sitecustomize/.pth without
            # killing processes or restoring workspace.
            # Honor per-task [verifier.hardening] opt-outs from task config.
            # No timeout_sec here: the conftest purge walks the rootfs and can be
            # slow on network-backed FS (Daytona), so its budget is owned by
            # lockdown.cleanup_verifier_python_hooks (VERIFIER_SETUP_TIMEOUT_SEC),
            # shared with the scoring path in harden_before_verify. The except
            # below keeps the step fail-closed.
            await self._planes.cleanup_verifier_python_hooks(
                self._env,
                getattr(self._task, "task_dir", None),
                "Soft verifier setup failed: purging Python injection hooks",
                user="root",
            )
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
            return None, None, verifier_error

        rewards = None
        verifier_output = None
        verifier_error = None
        try:
            verifier = self._planes.verifier(
                task=self._task,
                rollout_paths=self._rollout_paths,
                sandbox=self._env,
            )
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = _ensure_canonical_rewards(
                verifier_result.rewards, task=self._task
            )
            # Capture raw verifier output for the user
            cat = await self._env.exec(
                "cat /logs/verifier/*.log 2>/dev/null || "
                "cat /logs/verifier/output.txt 2>/dev/null || true",
                timeout_sec=10,
            )
            verifier_output = (cat.stdout or "").strip() or None
            logger.info(f"[soft_verify] rewards={rewards}")
        except TimeoutError:
            verifier_error = (
                f"soft verifier timed out after "
                f"{self._task.config.verifier.timeout_sec}s"
            )
            logger.error(verifier_error)
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
        return rewards, verifier_output, verifier_error

    # Phase 5: CLEANUP

    async def cleanup(self) -> None:
        """Close ACP client and stop the environment."""
        self._capture_partial_acp_trajectory()
        await self.disconnect()

        if self._env and self._config.export_generated_skills_to:
            try:
                await self._export_generated_skills()
            except Exception as e:
                # Surface export failure on a dedicated sibling channel
                # (#389 follow-up). Routing it through self._error caused
                # classify_error("Skill export failed: ... connection lost")
                # to mis-tag the rollout as agent infra_failure, polluting
                # the agent-error dashboards. Keep the agent/verifier error
                # channels untouched: export runs during cleanup, after the
                # agent already finished.
                export_error = f"Skill export failed: {e}"
                logger.error(export_error)
                if self._export_error is None:
                    self._export_error = export_error
                self._evolved_skills = None

        usage_runtime = getattr(self, "_usage_runtime", None)
        if usage_runtime is not None:
            try:
                await self._planes.stop_provider_runtime(usage_runtime)
                self._usage_metrics = self._planes.extract_usage(usage_runtime)
            except Exception as e:
                logger.warning(f"Usage telemetry runtime stop failed: {e}")
                self._usage_metrics = self._planes.extract_usage(None)
            # Snapshot any provider failure (401/403/429/503) now that captures
            # are imported (stop() populated the trajectory). This must happen
            # before we drop the runtime reference below, and is read later by
            # ACP-error classification — for Daytona the trajectory is empty
            # until here (#546/#564).
            #
            # Coverage gap: only `self._usage_runtime` is scanned here. Bedrock
            # auth failures flow through `self._provider_runtime`, whose server
            # (BedrockProxyServer) exposes no `.trajectory`/`.exchanges`, so a
            # fallback scan of it would always return None — useless, so it's
            # not implemented. The direct-AWS-Bedrock case (remote sandbox,
            # runtime=None) bypasses both proxies entirely and is out of scope.
            self._provider_failure_cached = _provider_failure_from_runtime(
                usage_runtime
            )
            self._provider_auth_status_cached = (
                self._provider_failure_cached.status
                if self._provider_failure_cached is not None
                and self._provider_failure_cached.marker == "provider auth failed"
                else None
            )
            self._api_failure_summary_cached = (
                _provider_api_failure_summary_from_runtime(usage_runtime)
            )
            try:
                self._write_llm_trajectory(usage_runtime)
            except Exception as e:
                logger.warning(f"LLM trajectory write failed: {e}")
            finally:
                self._usage_runtime = None

        self._finalize_usage_metrics()
        self._enforce_required_usage_tracking()

        if self._environment is not None:
            with contextlib.suppress(Exception):
                await self._environment.teardown()
            self._environment = None

        if self._env and not getattr(self, "_env_externally_owned", False):
            # An externally-owned sandbox (use_prebuilt_env) belongs to the
            # caller — leave it running so they can reuse it or stop it
            # themselves. #388. getattr() keeps tests that bypass __init__
            # via Rollout.__new__() working.
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        if hasattr(self, "_task_tmp") and self._task_tmp:
            shutil.rmtree(self._task_tmp, ignore_errors=True)

        self._phase = "cleaned"

    def _finalize_usage_metrics(self) -> None:
        """Prefer LiteLLM usage, otherwise use trusted native ACP usage."""
        current_metrics = getattr(
            self, "_usage_metrics", {"usage_source": "unavailable"}
        )
        if current_metrics.get("usage_source") == USAGE_SOURCE_PROVIDER_RESPONSE:
            return
        native_metrics = getattr(self, "_native_usage_metrics", None)
        if isinstance(native_metrics, dict) and is_token_usage_available(
            native_metrics
        ):
            self._usage_metrics = native_metrics

    def _enforce_required_usage_tracking(self) -> None:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        if usage_cfg.mode != "required" or self._config.primary_agent == "oracle":
            return
        if is_token_usage_available(getattr(self, "_usage_metrics", None)):
            return
        if self._error is not None:
            return
        self._error = (
            "Token usage tracking is required, but no provider token usage was "
            "captured."
        )
        logger.error(self._error)

    # Full run

    def _record_agent_timeout(self, e: TimeoutError) -> None:
        """Record a timed-out agent run on the rollout's error state.

        Shared by run()'s inner per-scene handler and the outer wall-clock
        handler. Preserves the watchdog's diagnostic message ("Agent idle
        for 600s with no new tool call ...") when it raised one, falling
        back to the generic wall-clock message only when there's no detail.

        A BenchFlow-owned wall-clock prompt timeout (``AgentPromptTimeoutError``)
        that fired with no pending tool calls is a *clean terminal* timeout:
        the trajectory is complete, not a rerunnable partial. Record that so
        the partial-capture path leaves ``_partial_trajectory`` False (#640).
        """
        detail = str(e).strip()
        self._error = detail or f"Agent timed out after {self._timeout}s"
        self._diagnostics.capture_idle(e)
        if isinstance(e, AgentPromptTimeoutError) and getattr(
            e, "terminal_trajectory_complete", False
        ):
            self._terminal_timeout = True
        logger.error(self._error)

    async def run(self) -> RolloutResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        agent_timed_out = False
        pending_acp_error: AgentProtocolError | None = None
        if cfg.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.run()."
            )
        try:
            await self.setup()
            await self.start()

            if cfg.primary_agent == "oracle":
                await self.install_agent()
                # git safe.directory needed for SWE-bench tasks with sandbox_user
                await self._env.exec(
                    f"git config --global --add safe.directory "
                    f"{shlex.quote(self._agent_cwd)} 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
                self._trajectory, self._agent_name = await _run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
                )
            else:
                await self.install_agent()
                try:
                    try:
                        if cfg.user is not None:
                            await self._run_user_loop()
                        else:
                            await self._run_steps(
                                compile_scenes_to_steps(
                                    cfg.effective_scenes,
                                    default_prompt=(
                                        self._resolved_prompts[0]
                                        if self._resolved_prompts
                                        else None
                                    ),
                                )
                            )
                    except TimeoutError as e:
                        agent_timed_out = True
                        self._record_agent_timeout(e)
                finally:
                    if cfg.oracle_access:
                        await self._env.exec(
                            "mv /oracle_backup /oracle 2>/dev/null || true; "
                            "mv /solution_oracle_backup /solution 2>/dev/null || true",
                            user="root",
                            timeout_sec=10,
                        )

            if not cfg.skip_verify:
                await self.verify()
                if (
                    agent_timed_out
                    and self._rewards is None
                    and self._verifier_error is None
                ):
                    self._rewards = {"reward": 0.0}
                    self._verifier_error = None

        except TimeoutError as e:
            self._record_agent_timeout(e)
        except ConnectionError as e:
            self._error = str(e)
            self._diagnostics.capture_transport(e)
            await self._probe_sandbox_health()
            logger.error(f"Agent connection lost: {self._error}")
        except SandboxStartupFailure as e:
            self._error = f"Sandbox startup failed: {e}"
            self._diagnostics.set(e.diagnostic)
            logger.error(self._error)
        except AgentProtocolError as e:
            # Defer classification until after cleanup(): the provider 401/403
            # that distinguishes provider_auth from a generic retryable ACP
            # error lives in the usage-proxy trajectory, which Daytona's
            # SandboxUsageProxy only imports on stop() (#546/#564).
            pending_acp_error = e
            # Set a provisional error so cleanup()'s
            # _enforce_required_usage_tracking guard early-returns instead of
            # logging a misleading "no provider token usage was captured"
            # message — the agent failed with an ACP error, not a usage gap.
            # The post-cleanup block below still unconditionally refines
            # self._error to the provider_auth marker, so this is only a
            # placeholder during cleanup.
            self._error = str(e)
            logger.error(str(e))
        except Exception as e:
            self._error = str(e)
            logger.error("Run failed", exc_info=True)
        finally:
            await self.cleanup()

        # cleanup() has now imported usage-proxy captures and snapshotted any
        # provider auth status, so classification can see the real 401/403.
        if pending_acp_error is not None:
            self._error = self._classify_acp_error(pending_acp_error)
            logger.error(self._error)

        if self._rollout_dir is None:
            return RolloutResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return self._build_result()

    # Scene-authored Step execution
    #
    # The step / user-loop drivers and the generated-skill export hook live in
    # ``benchflow.rollout._user_loop`` as free functions taking this Rollout —
    # the same engine convention as ``rollout_branch.py``. These thin methods
    # keep instance-level patching and unbound ``Rollout._export_generated_skills``
    # calls working unchanged.

    async def _export_generated_skills(self) -> None:
        """Download creator-produced skills before sandbox cleanup.

        Also captures the exported skill packs into ``self._evolved_skills``
        — the ``name -> body`` dict a continual-learning Job commits to its
        persistent LearnerStore (capability 5).

        Retries transient download failures up to 3 times (guards ENG-147).
        """
        await _export_generated_skills_engine(self)

    async def _activate_step_skills(self, step: Step) -> None:
        """Activate scene-local skills attached by the Scene desugaring pass."""
        await _activate_step_skills_engine(self, step)

    async def _run_steps(self, steps: list[Step]) -> None:
        """Execute already-compiled rollout Steps in declaration order."""
        await _run_steps_engine(self, steps)

    async def _run_user_loop(self) -> None:
        """Execute a user-driven progressive-disclosure loop.

        Each round: user.run() → connect → agent.execute() → disconnect →
        soft_verify() → build RoundResult → repeat. Stops when user.run()
        returns None or max_user_rounds is reached.
        """
        await _run_user_loop_engine(self)

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role.

        Installs the role's agent binary and credentials if it differs
        from the primary agent (which was set up in install_agent()).
        Updates _agent_launch so disconnect() kills the correct process.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        # Merge cfg.agent_env (config-level) with role.env (role-specific) so
        # provider creds from YAML reach the agent. role.env wins on overlap.
        disallow_web_tools = getattr(self, "_disallow_web_tools", None)
        if disallow_web_tools is None:
            disallow_web_tools = _task_disallows_internet(getattr(self, "_task", None))
        disallow_web_tools = bool(disallow_web_tools and role.agent != "oracle")
        agent_launch = self._planes.agent_launch(
            role.agent,
            disallow_web_tools=disallow_web_tools,
        )
        agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                role.agent,
                role.model,
                {**(cfg.agent_env or {}), **(role.env or {})},
            ),
            disallow=disallow_web_tools,
        )
        agent_env, self._usage_runtime = await self._planes.ensure_litellm_runtime(
            agent=role.agent,
            agent_env=agent_env,
            model=role.model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
        )

        role_agent_differs = role.agent != cfg.primary_agent
        needs_role_credentials = (
            role_agent_differs or role.model != cfg.primary_model or bool(role.env)
        )
        if role_agent_differs:
            agent_cfg = await self._planes.install_agent(
                self._env,
                role.agent,
                rollout_dir,
                sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            )
        else:
            agent_cfg = getattr(self, "_agent_cfg", None)
        if needs_role_credentials:
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await self._planes.write_credential_files(
                self._env,
                role.agent,
                agent_env,
                agent_cfg,
                role.model,
                cred_home,
            )
            if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                await self._planes.upload_subscription_auth(
                    self._env, role.agent, cred_home
                )
            await self._planes.apply_web_tool_policy(
                self._env,
                role.agent,
                agent_cfg,
                cred_home,
                disallow=disallow_web_tools,
            )

        self._agent_launch = agent_launch

        (
            self._acp_client,
            self._session,
            self._session_adapter,
            self._agent_name,
        ) = await self._planes.connect_acp(
            env=self._env,
            agent=role.agent,
            agent_launch=agent_launch,
            agent_env=agent_env,
            sandbox_user=cfg.sandbox_user,
            model=role.model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
            reasoning_effort=role.reasoning_effort,
            mcp_servers=_task_mcp_specs(getattr(self, "_task", None)),
        )
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)
        self._active_role = role

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # Internal helpers

    async def _probe_sandbox_health(self) -> None:
        """Quick health probe after transport death. Enriches transport diagnostic.

        Guards ENG-148: distinguishes Daytona session killed vs agent crash.
        """
        diag = self._diagnostics.transport_closed
        if diag is None or self._env is None:
            return
        try:
            result = await asyncio.wait_for(
                self._env.exec("echo __BENCHFLOW_HEALTH_OK__", timeout_sec=10),
                timeout=15,
            )
            stdout = str(getattr(result, "stdout", "") or "").strip()
            raw_rc = getattr(result, "return_code", None)
            rc = int(raw_rc) if isinstance(raw_rc, (int, float)) else None
            if "__BENCHFLOW_HEALTH_OK__" in stdout:
                diag.sandbox_reachable = True
                diag.sandbox_probe_rc = rc
            else:
                diag.sandbox_reachable = False
                diag.sandbox_probe_rc = rc
                diag.sandbox_probe_stdout = stdout[:200]
        except Exception as probe_err:
            import traceback

            logger.exception("sandbox health probe failed")
            diag.sandbox_reachable = False
            diag.sandbox_probe_error = str(probe_err)[:200]
            diag.sandbox_probe_error_type = type(probe_err).__name__
            diag.sandbox_probe_traceback = traceback.format_exc()[-2000:]

    def _classify_acp_error(self, e: AgentProtocolError) -> str:
        # The base AgentProtocolError only annotates `message: str` without
        # assigning it, so a base instance has no `.message` (AttributeError
        # risk); ACPError subclasses do set it. Fall back to str(e) defensively.
        message = getattr(e, "message", str(e))
        if "Invalid API key" in message:
            from benchflow.agents.env import check_subscription_auth
            from benchflow.agents.registry import infer_env_key_for_model

            key = (
                infer_env_key_for_model(self._config.primary_model)
                if self._config.primary_model
                else None
            )
            if key and check_subscription_auth(self._config.primary_agent, key):
                return (
                    f"{key} was rejected as invalid. "
                    f"Subscription auth credentials exist — unset the env var "
                    f"to use them: env -u {key} <command>"
                )
        # A real provider failure often surfaces only as a generic
        # "ACP error -32603: Internal error" at this layer — the provider's
        # actual 401/403/429/503 is visible only in the proxy-captured
        # trajectory (#546/#564). Surface a sanitized marker (status code only —
        # never the response body or headers) so RetryConfig.should_retry can
        # classify it (auth/rate-limit fail fast; 503 stays retryable infra)
        # instead of burning retries on a generic ACP error.
        provider_failure = self._provider_failure()
        if provider_failure is not None:
            return f"{e} | {provider_failure.error_suffix}"
        return str(e)

    def _provider_failure(self) -> ProviderFailure | None:
        """Return the provider failure snapshotted during cleanup.

        Falls back to the auth-only status cache for partial Rollout doubles in
        tests that set ``_provider_auth_status_cached`` directly (#564).
        """
        failure = getattr(self, "_provider_failure_cached", None)
        if failure is not None:
            return failure
        return _provider_failure_from_status(self._provider_auth_status())

    def _provider_auth_status(self) -> int | None:
        """Return the provider 401/403 status snapshotted during cleanup.

        The snapshot is taken in :meth:`cleanup` after the usage proxy imports
        its captures, so this is valid for both the host proxy (trajectory
        filled as requests complete) and Daytona's SandboxUsageProxy (filled
        only on ``stop()``). Only the integer status code is ever read — never
        response bodies or headers — so no credential material reaches
        ``result.error`` (#546/#564).
        """
        return self._provider_auth_status_cached

    def _write_llm_trajectory(self, usage_runtime: Any) -> None:
        """Persist captured provider HTTP exchanges as JSONL."""
        if self._rollout_dir is None:
            return
        trajectory = getattr(getattr(usage_runtime, "server", None), "trajectory", None)
        if trajectory is None or not trajectory.exchanges:
            return
        traj_dir = self._rollout_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "llm_trajectory.jsonl").write_text(
            trajectory.to_jsonl(redact_keys=True)
        )

    def _usage_tracking_metadata(self) -> dict[str, Any]:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        usage_source = str(self._usage_metrics.get("usage_source", "unavailable"))
        if usage_cfg.mode == "off":
            status = "off"
        elif is_token_usage_available(self._usage_metrics):
            status = "enabled"
        else:
            status = "unavailable"
        return usage_cfg.to_result_metadata(
            environment=self._config.environment,
            status=status,
            usage_source=usage_source,
        )

    def _current_sandbox_id(self) -> str | None:
        sandbox_id = getattr(self, "_sandbox_id", None)
        if isinstance(sandbox_id, str):
            return sandbox_id
        env_sandbox_id = getattr(getattr(self, "_env", None), "sandbox_id", None)
        return env_sandbox_id if isinstance(env_sandbox_id, str) else None

    def _maybe_classify_api_error(self) -> None:
        """Detect a silent provider API failure after the rollout finished.

        Runs only when no other error was recorded. Layer 1 (proxy-proven):
        every captured provider request failed and the agent produced zero
        tokens -> error_category "api_error". Layer 2 (zero-signal): no proxy
        failure evidence, but the agent ended with zero tokens AND zero tool
        calls -> "suspected_api_error" (e.g. the agent rejected the model id
        against its own catalog and never issued a request). Both null the
        reward so the slot is excluded from score denominators instead of
        polluting them as a fake healthy fail; the slot stays rerun-able and
        the batch is never interrupted.
        """
        if self._error is not None:
            return
        # Only judge rollouts where the agent actually ran: when no execute()
        # recorded a prompt, this is a setup/export failure path that owns its
        # own error channels (#389) — zero activity there is expected, not a
        # silent API failure.
        if not getattr(self, "_executed_prompts", None):
            return
        # getattr-defensive: tests construct partial Rollout doubles that
        # bypass __init__ (same pattern as _task_skill_policy below).
        usage_metrics = getattr(self, "_usage_metrics", None) or {}
        total_tokens = _as_nonnegative_int(usage_metrics.get("total_tokens"))
        verdict, info = classify_api_failure(
            getattr(self, "_api_failure_summary_cached", None),
            total_tokens=total_tokens,
            n_tool_calls=getattr(self, "_n_tool_calls", 0),
        )
        if verdict is None:
            return
        if verdict == "api_error":
            subcategory = info.get("subcategory") or "provider_error"
            kind = "transient" if info.get("transient") else "permanent"
            diag = ProviderApiErrorDiagnostic(
                subcategory=subcategory,
                transient=bool(info.get("transient")),
                dominant_status=info.get("dominant_status"),
                status_counts=info.get("status_counts"),
                total_requests=info.get("total_requests") or 0,
                failed_requests=info.get("failed_requests") or 0,
                fingerprint=info.get("fingerprint") or "",
            )
            self._diagnostics.set(diag)
            self._error = (
                f"provider api error [{subcategory}/{kind}] "
                f"HTTP {info.get('dominant_status')} on "
                f"{diag.failed_requests}/{diag.total_requests} requests"
            )
        else:
            diag = SuspectedApiErrorDiagnostic(
                total_tokens=total_tokens,
                n_tool_calls=self._n_tool_calls,
                total_requests=info.get("total_requests") or 0,
                failed_requests=info.get("failed_requests") or 0,
            )
            self._diagnostics.set(diag)
            self._error = (
                "suspected provider api error: agent ended with zero tokens "
                "and zero tool calls (no scoreable model activity)"
            )
        # Unhealthy by definition: drop any verifier reward so the slot is
        # excluded from score denominators (rerun-able, never counted).
        self._rewards = None

    def _loop_strategy_metadata(self) -> dict[str, Any] | None:
        """Loop-strategy run summary for the result.json ``loop`` block.

        Computed at result-build time — after run() has finalized
        ``self._error`` on every path (agent timeout, ACP error, success) —
        from the engine's in-loop round log, so a mid-round crash still
        reports the rounds that completed. getattr() keeps tests that bypass
        __init__ via Rollout.__new__() working.
        """
        user = self._config.user
        if self._config.loop_strategy_spec is None or not isinstance(
            user, LoopStrategyUser
        ):
            return None
        return collect_loop_metadata(
            user,
            getattr(self, "_user_rounds_log", []),
            max_rounds=self._config.max_user_rounds,
            error=getattr(self, "_error", None),
        )

    def _build_result(self) -> RolloutResult:
        rollout_dir = self._require_rollout_dir()
        self._maybe_classify_api_error()
        # For Scene/multi-turn rollouts, each execute() call records the
        # prompt(s) it sent into self._executed_prompts. Use that as the
        # authoritative prompt list so n_prompts and prompts.json reflect
        # every prompt the agent actually received (issue #377). Fall back
        # to the resolved base prompts when no execute() ran (e.g. setup
        # failure paths).
        prompts = self._executed_prompts or self._resolved_prompts
        return _build_rollout_result(
            rollout_dir,
            task_name=self._config.task_path.name,
            rollout_name=self._rollout_name or "",
            agent=self._config.primary_agent,
            agent_name=self._agent_name,
            model=self._config.primary_model,
            n_tool_calls=self._n_tool_calls,
            prompts=prompts,
            error=self._error,
            verifier_error=self._verifier_error,
            export_error=self._export_error,
            trajectory=self._trajectory,
            partial_trajectory=self._partial_trajectory,
            trajectory_source=self._trajectory_source,
            rewards=self._rewards,
            started_at=self._require_started_at(),
            timing=self._timing,
            scenes=self._config.effective_scenes,
            evolved_skills=self._evolved_skills,
            source_provenance=self._config.source_provenance,
            dataset=self._config.dataset,
            task_digest=self._config.task_digest,
            diagnostics=self._diagnostics,
            usage_tracking=self._usage_tracking_metadata(),
            skill_policy=getattr(self, "_task_skill_policy", None)
            or resolve_task_skill_policy(
                task_path=self._config.task_path,
                skill_mode=self._config.recorded_skill_mode,
                runtime_skills_dir=self._config.skills_dir,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_id=self._current_sandbox_id(),
            loop=loop_block(
                self._config.loop_strategy_spec,
                self._loop_strategy_metadata(),
            ),
            **self._usage_metrics,
        )


__all__ = [
    "Role",
    "Scene",
    "Turn",
    "Rollout",
    "RolloutConfig",
]
