"""BenchFlow Runtime — the execution center.

``Runtime.execute()`` is the single execution path for both single-agent
and multi-agent runs. Everything else layers on top:

- ``bf.run(scene, env)`` → convenience sugar
- ``SDK.run(...)`` → backwards-compat shim
- ``Eval.run(...)`` → batch of Runtime.execute()

Architecture:
    Agent  → thin wrapper around registry entry + model + creds
    Environment → wraps Docker/Daytona sandbox, owns lifecycle
    Scene → declarative roles + turns, lowered to rollout Steps
    Runtime → env + rollout execution loop + verify
    RuntimeResult → trajectories + messages + rewards + snapshots
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from benchflow.agents.registry import (
    AgentConfig,
    is_explicit_raw_agent_command,
    resolve_agent,
)
from benchflow.models import RolloutResult
from benchflow.review.options import ReviewerConfig
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

if TYPE_CHECKING:
    from benchflow.review.outcome import ScoringResult
    from benchflow.rollout import RolloutConfig as TrialConfig

logger = logging.getLogger(__name__)


class Environment:
    """Wraps a Docker/Daytona sandbox environment, owns lifecycle.

    Usage::

        env = Environment.from_task("tasks/my-task", sandbox="daytona")
        await env.start()
        # ... run agents ...
        await env.stop()

    Or as a context manager::

        async with Environment.from_task("tasks/X", sandbox="daytona") as env:
            result = await runtime.execute()
    """

    def __init__(self, inner: Any, task_path: Path, sandbox: str) -> None:
        self._inner = inner
        self.task_path = task_path
        self.sandbox = sandbox
        self._started = False

    @classmethod
    def from_task(
        cls,
        task_path: str | Path,
        sandbox: str = "daytona",
        rollout_name: str | None = None,
        planes: Any | None = None,
    ) -> Environment:
        """Create an environment from a task directory."""
        from uuid import uuid4

        from benchflow.contracts import default_rollout_planes
        from benchflow.task import RolloutPaths, Task

        task_path = Path(task_path)
        task = Task(task_path)
        rollout_name = rollout_name or task_path.name
        rollout_paths = RolloutPaths(
            rollout_dir=Path.cwd()
            / "jobs"
            / "environment"
            / f"{rollout_name}__{uuid4().hex[:8]}"
        )
        rollout_paths.mkdir()
        plane_bundle = planes or default_rollout_planes()
        try:
            inner = plane_bundle.create_environment(
                sandbox,
                task=task,
                task_path=task_path,
                rollout_name=rollout_name,
                rollout_paths=rollout_paths,
                preserve_agent_network=False,
                environment_manifest=None,
            )
        except Exception:
            # create_environment failed (e.g. a missing optional sandbox SDK) —
            # don't leave the empty rollout dir we just created littering
            # jobs/environment/. Best-effort cleanup, then re-raise unchanged.
            import shutil

            shutil.rmtree(rollout_paths.rollout_dir, ignore_errors=True)
            raise
        return cls(inner=inner, task_path=task_path, sandbox=sandbox)

    @property
    def inner(self) -> Any:
        """The underlying harbor environment (Docker/Daytona). Use for Scene-based shared sandbox access."""
        return self._inner

    @property
    def task(self) -> Any:
        from benchflow.task import Task

        return Task(self.task_path)

    async def start(self, force_build: bool = False) -> None:
        await self._inner.start(force_build=force_build)
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        if self._started:
            await self._inner.stop(delete=delete)
            self._started = False

    async def exec(self, cmd: str, **kwargs: Any) -> Any:
        """Run a command in the sandbox.

        Pass ``service="<name>"`` to target an additional compose service
        (a vulhub-style target container) instead of the default agent
        container ``"main"`` — see #248.
        """
        return await self._inner.exec(cmd, **kwargs)

    async def exec_in_service(self, service: str, cmd: str, **kwargs: Any) -> Any:
        """Run a command in a named compose service container (#248).

        Ergonomic wrapper for ``exec(cmd, service=service)``. Useful for
        injecting flags into, or verifying state of, a multi-container
        task's target container.
        """
        return await self._inner.exec(cmd, service=service, **kwargs)

    async def upload_file(self, src: str | Path, dst: str) -> None:
        await self._inner.upload_file(src, dst)

    async def upload_dir(
        self, src: str | Path, dst: str, service: str = "main"
    ) -> None:
        """Upload a directory into the sandbox.

        Pass ``service="<name>"`` to target a non-``main`` compose service
        (a vulhub-style target container) — see #248.
        """
        await self._inner.upload_dir(src, dst, service=service)

    async def download_file(self, src: str, dst: str | Path) -> None:
        await self._inner.download_file(src, dst)

    async def download_dir(
        self, src: str, dst: str | Path, service: str = "main"
    ) -> None:
        """Download a directory from the sandbox.

        Pass ``service="<name>"`` to fetch from a non-``main`` compose service
        (a vulhub-style target container) — see #248.
        """
        await self._inner.download_dir(src, dst, service=service)

    async def __aenter__(self) -> Environment:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def __repr__(self) -> str:
        return f"Environment({self.task_path.name!r}, sandbox={self.sandbox!r})"


@dataclass
class Agent:
    """Thin wrapper around a registered agent + model + credentials."""

    name: str
    model: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def config(self) -> AgentConfig | None:
        try:
            return resolve_agent(self.name)
        except KeyError:
            return None

    @property
    def launch_cmd(self) -> str:
        config = self.config
        if config is None:
            if is_explicit_raw_agent_command(self.name):
                return self.name
            raise KeyError(f"Unknown agent: {self.name!r}")
        return config.launch_cmd

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, model={self.model!r})"


# The agent timeout RuntimeConfig applied in earlier releases when none was set.
_OLD_DEFAULT_RUNTIME_TIMEOUT = 900

# RuntimeConfig fields that nothing reads, with their defaults. Kept so
# existing callers do not break; setting one to another value warns.
_UNUSED_RUNTIME_CONFIG_FIELDS: dict[str, Any] = {
    "max_rounds": 10,
    "snapshot_policy": "none",
    "reward_stream": True,
}


@dataclass
class RuntimeConfig:
    """Configuration for ``bf.run(agent, env, config)`` and ``bf.run("agent", ...)``.

    ``timeout`` overrides the task's agent timeout (``[agent] timeout_sec``) in
    seconds; ``None`` (the default) keeps the task's own. ``rollout_name``
    names the rollout directory, and ``jobs_dir`` is where artifacts go. The
    remaining fields mirror :class:`~benchflow.RolloutConfig`. ``max_rounds``,
    ``snapshot_policy`` and ``reward_stream`` are not used; setting them emits a
    ``DeprecationWarning``. Use ``RolloutConfig.max_user_rounds`` for user loops.
    """

    codex_apps_policy: Literal["disabled", "inherit"] | None = field(
        default=None, kw_only=True
    )
    sandbox_user: str | None = "agent"
    sandbox_setup_timeout: int = 120
    max_rounds: int = 10
    snapshot_policy: str = "none"
    reward_stream: bool = True
    timeout: int | None = None
    jobs_dir: str | Path = "jobs"
    rollout_name: str | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    context_root: str | Path | None = None
    base_image_override: str | None = None
    pre_agent_hooks: list | None = None
    sandbox_locked_paths: list[str] | None = None
    usage_tracking: Any = None
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)

    def __post_init__(self) -> None:
        import warnings

        for name, default in _UNUSED_RUNTIME_CONFIG_FIELDS.items():
            if getattr(self, name) != default:
                warnings.warn(
                    f"RuntimeConfig.{name} is not used and will be removed; "
                    "it has no effect on the run.",
                    DeprecationWarning,
                    stacklevel=3,
                )


@dataclass
class RuntimeResult:
    """Deprecated: ``Runtime.execute()`` and every ``bf.run`` form return
    :class:`~benchflow.RolloutResult`. Constructing this class warns.

    The former output of Runtime.execute().

    Artifact-oriented: exposes paths and structured summaries,
    not only in-memory objects.

    Guaranteed artifacts (when run completes):
        rollout_dir/result.json       — reward, timing, error, metadata
        rollout_dir/rewards.jsonl     — terminal + rubric reward events
        rollout_dir/trajectory/       — ACP trajectory JSONL
        rollout_dir/timing.json       — phase-level timing
        rollout_dir/config.json       — run configuration snapshot
        rollout_dir/prompts.json      — prompts sent to agent

    Optional artifacts:
        rollout_dir/snapshots/             — checkpoint refs (if snapshot_policy != "none")
    """

    task_name: str
    rollout_name: str
    reward: float | None
    rewards: dict | None
    n_tool_calls: int
    error: str | None
    verifier_error: str | None
    trajectory: list[dict]
    messages: list[dict] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    rollout_dir: Path | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    scoring: ScoringResult | None = None

    def __post_init__(self) -> None:
        import warnings

        warnings.warn(
            "RuntimeResult is deprecated; Runtime.execute() and bf.run() return "
            "benchflow.RolloutResult, which has reward, passed and rollout_dir.",
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def passed(self) -> bool:
        from benchflow._utils.scoring import classify_result_outcome

        return (
            classify_result_outcome(
                {
                    "rewards": self.rewards,
                    "scoring": self.scoring.to_dict()
                    if self.scoring is not None
                    else None,
                    "error": self.error,
                    "verifier_error": self.verifier_error,
                }
            )
            == "passed"
        )

    @property
    def verified(self) -> bool:
        from benchflow._utils.scoring import classify_result_outcome

        return classify_result_outcome(
            {
                "rewards": self.rewards,
                "scoring": self.scoring.to_dict() if self.scoring is not None else None,
                "error": self.error,
                "verifier_error": self.verifier_error,
            }
        ) in {"passed", "failed"}


def _warn_if_old_default_timeout_mattered(task_path: Path) -> None:
    """Warn when dropping the old 900 s default changes this run's timeout."""
    import warnings

    from benchflow.task import Task

    try:
        task_timeout = Task(task_path).config.agent.timeout_sec
    except Exception:
        return
    if task_timeout is None or int(task_timeout) == _OLD_DEFAULT_RUNTIME_TIMEOUT:
        return
    warnings.warn(
        f"RuntimeConfig no longer applies a {_OLD_DEFAULT_RUNTIME_TIMEOUT} s agent "
        f"timeout by default; this run uses the task's own timeout of "
        f"{int(task_timeout)} s. Pass RuntimeConfig(timeout="
        f"{_OLD_DEFAULT_RUNTIME_TIMEOUT}) to keep the old limit.",
        FutureWarning,
        stacklevel=3,
    )


class Runtime:
    """The 0.3 execution center.

    Single execution path for both single-agent and multi-agent runs.
    Owns: environment lifecycle, agent setup, ACP session, verification,
    reward emission, snapshots, artifact writing.

    Usage::

        agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
        env = Environment.from_task("tasks/X", sandbox="daytona")
        runtime = Runtime(env, agent)
        result = await runtime.execute()
    """

    def __init__(
        self,
        env: Environment,
        agent: Agent,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or RuntimeConfig()

    async def execute(self) -> RolloutResult:
        """Run the full execution loop via Trial.

        Runtime is the stable user-facing surface. Trial owns the
        decomposed lifecycle phases underneath.

        Honours the caller-supplied :class:`Environment`: the live
        ``env.inner`` sandbox is reused instead of creating a second one
        (fixes #388). If the caller has not yet called ``env.start()``
        we start it now so ``env`` stays in a consistent state and the
        underlying sandbox is brought up exactly once.
        """
        from benchflow._types import Scene
        from benchflow.rollout import Rollout, RolloutConfig

        config = self.config
        trial_config = RolloutConfig(
            task_path=self.env.task_path,
            scenes=[
                Scene.single(
                    agent=self.agent.name,
                    model=self.agent.model,
                )
            ],
            environment=self.env.sandbox,
            codex_apps_policy=config.codex_apps_policy,
            sandbox_user=config.sandbox_user,
            sandbox_locked_paths=config.sandbox_locked_paths,
            sandbox_setup_timeout=config.sandbox_setup_timeout,
            jobs_dir=config.jobs_dir,
            rollout_name=config.rollout_name,
            timeout=config.timeout,
            context_root=config.context_root,
            base_image_override=config.base_image_override,
            pre_agent_hooks=config.pre_agent_hooks,
            agent=self.agent.name,
            model=self.agent.model,
            agent_env=self.agent.env,
            skills_dir=config.skills_dir,
            skill_mode=config.skill_mode,
            usage_tracking=config.usage_tracking,
            reviewer=config.reviewer,
        )

        if config.timeout is None:
            _warn_if_old_default_timeout_mattered(self.env.task_path)

        rollout = await Rollout.create(trial_config)

        # If the caller has not started the Environment yet, do it now so
        # the Environment's _started flag stays accurate and the caller's
        # later env.stop() works. Then hand the live sandbox to Rollout —
        # this is what makes Runtime honour the input Environment instead
        # of silently building a second one. #388.
        if not self.env._started:
            await self.env.start()
        rollout.use_prebuilt_env(self.env.inner)

        run_result = await rollout.run()

        # The Rollout owns the on-disk artifact directory; make sure the
        # result points at it so callers can find result.json (#378).
        if run_result.rollout_dir is None:
            run_result.rollout_dir = rollout._rollout_dir
        return run_result


def _suggest(name: str, choices: list[str]) -> str | None:
    import difflib

    close = difflib.get_close_matches(name, choices, n=1, cutoff=0.85)
    return close[0] if close else None


def _transposition_of(name: str, choices: list[str]) -> str | None:
    """The choice that ``name`` equals up to one swap of adjacent letters."""
    for choice in choices:
        if len(choice) != len(name) or choice == name:
            continue
        diff = [i for i, (a, b) in enumerate(zip(name, choice, strict=True)) if a != b]
        if (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and name[diff[0]] == choice[diff[1]]
            and name[diff[1]] == choice[diff[0]]
        ):
            return choice
    return None


def check_sandbox_name(sandbox: str) -> None:
    """Raise ``ValueError`` (with a suggestion) for an unknown sandbox name."""
    from benchflow.sandbox.providers import SANDBOX_PROVIDER_SET, providers_phrase

    if sandbox not in SANDBOX_PROVIDER_SET:
        hint = _suggest(sandbox, sorted(SANDBOX_PROVIDER_SET))
        raise ValueError(
            f"Unknown sandbox {sandbox!r}"
            + (f"; did you mean {hint!r}?" if hint else "")
            + f" Use {providers_phrase()}."
        )


def check_agent_names(names: Iterable[str | None]) -> None:
    """Raise ``ValueError`` for an agent name that closely misspells a registered one.

    Raw commands (a space or ``/``), namespaced specs (``acp:pi``) and names
    unrelated to any registered agent pass, as the registry allows them.
    """
    from benchflow.agents.registry import AGENT_ALIASES, AGENTS

    known = sorted({*AGENTS, *AGENT_ALIASES, "oracle", "nop"})
    for name in sorted({n for n in names if n}):
        if name in known or any(c in name for c in " /:\t"):
            continue
        # Two swapped neighbouring letters in a short name ("oracel") score
        # under the similarity cutoff, so they are matched exactly.
        hint = _suggest(name, known) or _transposition_of(name, known)
        if hint is not None:
            raise ValueError(
                f"Unknown agent {name!r}; did you mean {hint!r}? "
                "`bench agent list` shows the registered agents."
            )


def check_rollout_config(config: TrialConfig) -> None:
    """Refuse caller mistakes before a sandbox starts.

    Raises ``FileNotFoundError`` for a missing task directory, ``ValueError``
    for an unknown sandbox and for an agent name that is a close misspelling
    of a registered agent (``"claud-agent-acp"``). Agent specs that are raw
    commands (a space or ``/``), namespaced (``acp:pi``) or unrelated to any
    registered name pass unchanged, as the registry allows them.
    """
    task_path = Path(config.task_path)
    if not task_path.is_dir():
        raise FileNotFoundError(
            f"Task directory not found: {task_path} (resolved against "
            f"{Path.cwd()}); pass the folder that holds task.md or task.toml."
        )
    check_sandbox_name(config.environment)
    check_agent_names(
        [config.agent, *(role.agent for scene in config.scenes for role in scene.roles)]
    )


# Set while a batch runs, so each rollout does not repeat the batch's
# host checks (and warnings).
_HOST_CHECKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_host_checked", default=False
)


def check_host(configs: Sequence[Any]) -> None:
    """The host checks ``bench eval run`` makes before a job, for SDK callers.

    Reuses ``bench doctor``'s checks: a ``RuntimeError`` naming doctor's fix
    when a config uses the Docker sandbox and Docker is not ready, and a
    ``UserWarning`` with doctor's fix when a Claude agent would fall back to a
    login file whose access token has expired (a warning, since Claude can
    refresh it when the refresh token still works). ``BENCHFLOW_SKIP_PREFLIGHT=1``
    skips both. ``configs`` are ``RolloutConfig`` or ``EvaluationConfig``
    objects (anything with ``environment``, ``agent``, ``model`` and
    ``agent_env``; ``scenes`` when present).
    """
    import os
    import warnings

    if os.environ.get("BENCHFLOW_SKIP_PREFLIGHT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    from benchflow import doctor as doctor_mod
    from benchflow._utils.config import normalize_agent_name
    from benchflow.cli.doctor import _expired_claude_login

    probes = doctor_mod.DoctorProbes.from_host()
    if any(c.environment == "docker" for c in configs):
        failed = [
            c
            for c in doctor_mod.check_docker(probes, required=True)
            if c.status == "fail"
        ]
        if failed:
            lines = "\n".join(
                f"  {c.name}: {c.summary}" + (f" (fix: {c.fix})" if c.fix else "")
                for c in failed
            )
            raise RuntimeError(
                "Docker is not ready for environment='docker'; nothing was "
                f"started.\n{lines}\nRun `bench doctor` for the full report, or "
                "set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check."
            )
    if any(c.environment == "remote-docker" for c in configs):
        from benchflow.sandbox.remote_docker import (
            probe_remote_docker,
            resolve_remote_docker_host,
        )

        try:
            probe_remote_docker(resolve_remote_docker_host())
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "The remote Docker host is not ready for "
                f"environment='remote-docker'; nothing was started.\n  {exc}\n"
                "Set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check."
            ) from exc
    if any(c.environment == "daytona" for c in configs):
        daytona = doctor_mod.check_daytona(probes, required=True, offline=False)
        if daytona.status == "fail":
            raise RuntimeError(
                "Daytona is not ready for environment='daytona'; nothing was "
                f"started.\n  {daytona.name}: {daytona.summary}"
                + (f" (fix: {daytona.fix})" if daytona.fix else "")
                + "\nRun `bench doctor --sandbox daytona` for the full report, or "
                "set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check."
            )
    warned: set[tuple[str, str | None]] = set()
    for config in configs:
        pairs: list[tuple[str, str | None]] = [(config.agent, config.model)] + [
            (role.agent, role.model)
            for scene in getattr(config, "scenes", None) or []
            for role in scene.roles
        ]
        for agent, model in pairs:
            if not agent or (agent, model) in warned:
                continue
            check = _expired_claude_login(
                probes,
                agent=normalize_agent_name(agent),
                model=model,
                agent_env=config.agent_env or {},
            )
            if check is None:
                continue
            warned.add((agent, model))
            warnings.warn(
                f"{agent} will use a Claude login file whose access token has "
                f"expired ({check.summary}); if it cannot be refreshed, the run "
                "fails after the sandbox starts. "
                + (f"Fix: {check.fix}" if check.fix else ""),
                UserWarning,
                stacklevel=4,
            )


async def run(
    subject: Agent | TrialConfig | str,
    env: Environment | str | None = None,
    config: RuntimeConfig | None = None,
    *,
    task_path: str | Path | None = None,
    model: str | None = None,
) -> RolloutResult:
    """Primary user-facing API — multiple calling conventions.

    Usage::

        import benchflow as bf

        # 1. TrialConfig (Scene-based, full control)
        result = await bf.run(TrialConfig(task_path=..., scenes=[...]))

        # 2. Agent + Environment (0.3 style)
        result = await bf.run(Agent("gemini", "flash"), Environment.from_task("tasks/X"))

        # 3. Agent name string (simplest)
        result = await bf.run("gemini", task_path="tasks/X")
    """
    from benchflow._types import Scene
    from benchflow.rollout import SKILL_MODE_SELF_GEN, Rollout, RolloutConfig

    if isinstance(subject, RolloutConfig):
        check_rollout_config(subject)
        if not _HOST_CHECKED.get():
            check_host([subject])
        if subject.skill_mode == SKILL_MODE_SELF_GEN:
            from benchflow.self_gen import run_self_gen

            return await run_self_gen(subject)
        rollout = await Rollout.create(subject)
        return await rollout.run()

    if isinstance(subject, Agent):
        if not isinstance(env, Environment):
            raise TypeError(
                f"When passing an Agent, env must be an Environment, got {type(env).__name__}. "
                f"Use bf.run('agent-name', task_path=...) for the string shortcut."
            )
        runtime = Runtime(env, subject, config)
        return await runtime.execute()

    if isinstance(subject, str):
        if task_path is None and Path(subject).is_dir():
            raise TypeError(
                f"bf.run's first argument is the agent, got the directory "
                f"{subject!r}. Use bf.run(bf.RolloutConfig(task_path={subject!r}, "
                "agent=...)) or bf.run('oracle', task_path=...)."
            )
        if task_path is None:
            raise ValueError(
                "task_path required when passing agent name as string, e.g. "
                f"bf.run({subject!r}, task_path='tasks/my-task')"
            )
        if env is not None and not isinstance(env, str):
            raise TypeError(
                f"With an agent name, env must be a sandbox name such as "
                f"'docker' or 'daytona', got {type(env).__name__}. To run in an "
                f"Environment you created, pass an Agent: "
                f"bf.run(bf.Agent({subject!r}, model=...), env)."
            )
        rc = config or RuntimeConfig()
        rollout_config = RolloutConfig(
            task_path=Path(task_path),
            scenes=[Scene.single(agent=subject, model=model)],
            environment=env if isinstance(env, str) else "docker",
            codex_apps_policy=rc.codex_apps_policy,
            sandbox_user=rc.sandbox_user,
            sandbox_locked_paths=rc.sandbox_locked_paths,
            sandbox_setup_timeout=rc.sandbox_setup_timeout,
            jobs_dir=rc.jobs_dir,
            rollout_name=rc.rollout_name,
            timeout=rc.timeout,
            context_root=rc.context_root,
            base_image_override=rc.base_image_override,
            pre_agent_hooks=rc.pre_agent_hooks,
            skills_dir=rc.skills_dir,
            skill_mode=rc.skill_mode,
            agent=subject,
            model=model,
            usage_tracking=rc.usage_tracking,
            reviewer=rc.reviewer,
        )
        check_rollout_config(rollout_config)
        if not _HOST_CHECKED.get():
            check_host([rollout_config])
        rollout = await Rollout.create(rollout_config)
        return await rollout.run()

    raise TypeError(f"Unsupported subject type: {type(subject).__name__}")
