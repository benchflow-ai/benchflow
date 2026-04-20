"""Trial — decomposed run lifecycle for a single agent-on-task evaluation.

Replaces the monolithic ``SDK.run()`` with independently-callable phases::

    trial = await Trial.create(TrialConfig(task_path=..., agent=..., model=...))
    await trial.setup()
    await trial.start()
    await trial.install_agent()
    await trial.connect()
    await trial.execute()
    result = await trial.verify()
    await trial.cleanup()

Or use ``trial.run()`` for the full lifecycle (equivalent to ``SDK.run()``).

Phases can be composed for multi-agent flows::

    await trial.setup()
    await trial.start()
    await trial.install_agent()

    # Coder turn
    await trial.connect()
    await trial.execute(prompts=[coder_prompt])
    await trial.disconnect()

    # Reviewer turn (same sandbox, new ACP session)
    await trial.connect()
    await trial.execute(prompts=[reviewer_prompt])
    await trial.disconnect()

    result = await trial.verify()
    await trial.cleanup()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_env import resolve_agent_env
from benchflow._agent_setup import deploy_skills, install_agent
from benchflow._credentials import upload_subscription_auth, write_credential_files
from benchflow._env_setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    stage_dockerfile_deps,
)
from benchflow._sandbox import (
    _resolve_locked_paths,
    _seed_verifier_workspace,
    _snapshot_build_config,
    harden_before_verify,
    lockdown_paths,
    setup_sandbox_user,
)
from benchflow._trajectory import (
    _capture_session_trajectory,
    _scrape_agent_trajectory,
)
from benchflow.acp.client import ACPClient, ACPError
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.models import RunResult, TrajectorySource

logger = logging.getLogger(__name__)


@dataclass
class Role:
    """One agent participant in a scene."""

    name: str
    agent: str
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    """One prompt in a scene. role selects which Role acts."""

    role: str
    prompt: str | None = None  # None = expand from instruction.md


@dataclass
class Scene:
    """One interaction region — roles take turns executing prompts."""

    name: str = "default"
    roles: list[Role] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    skills_dir: str | Path | None = None
    # Future (xiangyi li): snapshot_before, snapshot_after for stateful envs
    # Future: scoring config (None = unscored warmup scene)

    @classmethod
    def single(
        cls,
        *,
        agent: str,
        model: str | None = None,
        prompts: list[str | None] | None = None,
        role_name: str = "agent",
        skills_dir: str | Path | None = None,
    ) -> "Scene":
        """Shortcut for single-agent, single-role scene."""
        prompts = prompts or [None]
        return cls(
            roles=[Role(name=role_name, agent=agent, model=model)],
            turns=[Turn(role=role_name, prompt=p) for p in prompts],
            skills_dir=skills_dir,
        )


@dataclass
class TrialConfig:
    """Declarative trial configuration.

    A trial is a sequence of scenes executed in a shared sandbox.
    Single-agent runs are a trial with one scene containing one role.
    """

    task_path: Path
    scenes: list[Scene] = field(default_factory=list)
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    services: list[str] | None = None
    job_name: str | None = None
    trial_name: str | None = None
    jobs_dir: str | Path = "jobs"
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None

    # Legacy compat fields — used by SDK.run() shim. Ignored when scenes is set.
    agent: str = "claude-agent-acp"
    prompts: list[str | None] | None = None
    model: str | None = None
    agent_env: dict[str, str] | None = None
    skills_dir: str | Path | None = None

    @classmethod
    def from_legacy(
        cls,
        *,
        task_path: Path,
        agent: str = "claude-agent-acp",
        model: str | None = None,
        prompts: list[str | None] | None = None,
        skills_dir: str | Path | None = None,
        **kwargs,
    ) -> "TrialConfig":
        """Construct from flat SDK.run()-style args."""
        return cls(
            task_path=task_path,
            scenes=[Scene.single(agent=agent, model=model, prompts=prompts, skills_dir=skills_dir)],
            agent=agent,
            model=model,
            prompts=prompts,
            skills_dir=skills_dir,
            **kwargs,
        )

    @property
    def effective_scenes(self) -> list[Scene]:
        """Scenes to execute — falls back to legacy fields if scenes is empty."""
        if self.scenes:
            return self.scenes
        return [Scene.single(agent=self.agent, model=self.model, prompts=self.prompts,
                             skills_dir=self.skills_dir)]

    @property
    def primary_agent(self) -> str:
        """Agent name for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].agent
        return self.agent

    @property
    def primary_model(self) -> str | None:
        """Model for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].model
        return self.model


class Trial:
    """Decomposed trial lifecycle with independently-callable phases."""

    def __init__(self, config: TrialConfig) -> None:
        self._config = config
        self._phase = "created"

        # Populated by setup()
        self._task: Any = None
        self._trial_dir: Path | None = None
        self._trial_paths: Any = None
        self._started_at: datetime | None = None
        self._job_name: str | None = None
        self._trial_name: str | None = None
        self._agent_env: dict[str, str] = {}
        self._resolved_prompts: list[str] = []
        self._agent_launch: str = ""
        self._env: Any = None
        self._timeout: int = 0
        self._timing: dict[str, float] = {}
        self._effective_locked: list[str] = []

        # Populated by install_agent()
        self._agent_cfg: Any = None
        self._agent_cwd: str = "/app"

        # Populated by connect()
        self._acp_client: ACPClient | None = None
        self._session: Any = None
        self._agent_name: str = ""

        # Populated by execute()
        self._trajectory: list[dict] = []
        self._n_tool_calls: int = 0
        self._trajectory_source: TrajectorySource | None = None
        self._partial_trajectory: bool = False

        # Populated by verify()
        self._rewards: dict | None = None
        self._verifier_error: str | None = None
        self._error: str | None = None

    @classmethod
    async def create(cls, config: TrialConfig) -> Trial:
        """Create a Trial instance. Preferred over __init__ for consistency."""
        return cls(config)

    @property
    def env(self) -> Any:
        return self._env

    @property
    def acp_client(self) -> ACPClient | None:
        return self._acp_client

    @property
    def trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    @property
    def result(self) -> RunResult | None:
        if self._phase not in ("verified", "cleaned"):
            return None
        return self._build_result()

    # ── Phase 1: SETUP (host-side, no container yet) ──

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        from benchflow.sdk import SDK

        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )

        self._effective_locked = _resolve_locked_paths(
            cfg.sandbox_user, cfg.sandbox_locked_paths
        )

        sdk = SDK()
        (
            self._task,
            self._trial_dir,
            self._trial_paths,
            self._started_at,
            self._job_name,
            self._trial_name,
        ) = sdk._init_trial(
            cfg.task_path, cfg.job_name, cfg.trial_name, cfg.jobs_dir
        )

        self._agent_env = resolve_agent_env(cfg.agent, cfg.model, cfg.agent_env)
        self._resolved_prompts = sdk._resolve_prompts(cfg.task_path, cfg.prompts)
        self._agent_launch = AGENT_LAUNCH.get(cfg.agent, cfg.agent)

        if cfg.context_root:
            stage_dockerfile_deps(cfg.task_path, Path(cfg.context_root))
        if cfg.skills_dir:
            _inject_skills_into_dockerfile(cfg.task_path, Path(cfg.skills_dir))

        self._env = _create_environment(
            cfg.environment, self._task, cfg.task_path,
            self._trial_name, self._trial_paths,
        )
        self._timeout = int(self._task.config.agent.timeout_sec or 0)

        sdk._write_config(
            self._trial_dir,
            task_path=cfg.task_path,
            agent=cfg.agent,
            model=cfg.model,
            environment=cfg.environment,
            skills_dir=cfg.skills_dir,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
        )

        self._phase = "setup"

    # ── Phase 2: START (container comes up) ──

    async def start(self) -> None:
        """Start the environment and upload task files."""
        from benchflow.sdk import SDK

        sdk = SDK()
        await sdk._start_env_and_upload(self._env, self._config.task_path, self._timing)

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        self._phase = "started"

    # ── Phase 3: INSTALL AGENT ──

    async def install_agent(self) -> None:
        """Install agent binary, set up credentials, sandbox user, skills, lockdown."""
        cfg = self._config

        cwd_result = await self._env.exec("pwd", timeout_sec=10)
        agent_cwd = (cwd_result.stdout or "").strip() or "/app"
        self._agent_cwd = agent_cwd

        if cfg.agent == "oracle":
            if cfg.sandbox_user:
                await setup_sandbox_user(
                    self._env, cfg.sandbox_user, workspace=self._agent_cwd
                )
            await _snapshot_build_config(self._env, workspace=self._agent_cwd)
            await _seed_verifier_workspace(self._env, workspace=self._agent_cwd)
            await lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        self._agent_cfg = await install_agent(
            self._env, cfg.agent, self._trial_dir
        )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await write_credential_files(
            self._env, cfg.agent, self._agent_env,
            self._agent_cfg, cfg.model, cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await upload_subscription_auth(self._env, cfg.agent, cred_home)

        if cfg.sandbox_user:
            self._agent_cwd = await setup_sandbox_user(
                self._env, cfg.sandbox_user, workspace=self._agent_cwd
            )
        await _snapshot_build_config(self._env, workspace=self._agent_cwd)
        await _seed_verifier_workspace(self._env, workspace=self._agent_cwd)

        await deploy_skills(
            self._env, cfg.task_path, cfg.skills_dir,
            self._agent_cfg, cfg.sandbox_user, self._agent_cwd, self._task,
        )
        await lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # ── Phase 3b: CONNECT (ACP session — re-entrant) ──

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        cfg = self._config
        t0 = datetime.now()

        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=cfg.agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.model,
            trial_dir=self._trial_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
        )

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    async def disconnect(self) -> None:
        """Close the ACP client and clean up agent process, keeping the environment alive."""
        if self._acp_client:
            try:
                await self._acp_client.close()
            except Exception as e:
                logger.warning(f"ACP client close failed: {e}")
            self._acp_client = None
            self._session = None
        # Kill any lingering agent processes to prevent context bleed between scenes
        if self._env:
            agent_cmd = self._agent_launch.split()[0].split("/")[-1]
            try:
                await self._env.exec(f"pkill -f '{agent_cmd}' || true", timeout_sec=10)
            except Exception:
                pass
        self._phase = "installed"

    # ── Phase 3c: EXECUTE ──

    async def execute(self, prompts: list[str] | None = None) -> tuple[list[dict], int]:
        """Run prompts through the ACP session. Returns (trajectory, n_tool_calls)."""
        effective_prompts = prompts or self._resolved_prompts
        t0 = datetime.now()

        trajectory, n_tool_calls = await execute_prompts(
            self._acp_client,
            self._session,
            effective_prompts,
            self._timeout,
        )

        self._trajectory.extend(trajectory)
        self._n_tool_calls += n_tool_calls
        self._trajectory_source = "acp"

        if "agent_execution" not in self._timing:
            self._timing["agent_execution"] = (datetime.now() - t0).total_seconds()

        self._phase = "executed"
        return trajectory, n_tool_calls

    # ── Phase 4: VERIFY ──

    async def verify(self) -> dict | None:
        """Run the verifier and return rewards."""
        cfg = self._config

        if not self._trajectory and cfg.agent != "oracle":
            scraped = await _scrape_agent_trajectory(
                self._env, cfg.agent, cfg.sandbox_user
            )
            if scraped:
                self._trajectory = scraped
                self._trajectory_source = "scraped"
                logger.warning(
                    f"Using scraped trajectory ({len(scraped)} events) — UNTRUSTED"
                )

        from benchflow.sdk import SDK
        sdk = SDK()
        self._rewards, self._verifier_error = await sdk._verify(
            self._env, self._task, self._trial_paths, self._timing,
            sandbox_user=cfg.sandbox_user, workspace=self._agent_cwd,
        )

        self._phase = "verified"
        return self._rewards

    # ── Phase 5: CLEANUP ──

    async def cleanup(self) -> None:
        """Close ACP client and stop the environment."""
        if not self._trajectory and self._acp_client and self._acp_client.session:
            try:
                captured = _capture_session_trajectory(self._acp_client.session)
                if captured:
                    self._trajectory = captured
                    self._partial_trajectory = True
                    self._trajectory_source = "partial_acp"
                    self._n_tool_calls = len(self._acp_client.session.tool_calls)
            except Exception as e:
                logger.warning(f"Partial trajectory capture failed: {e}")

        await self.disconnect()

        if self._env:
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        self._phase = "cleaned"

    # ── Full run ──

    async def run(self) -> RunResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        try:
            await self.setup()
            await self.start()

            if cfg.primary_agent == "oracle":
                await self.install_agent()
                from benchflow.sdk import SDK
                sdk = SDK()
                self._trajectory, self._agent_name = await sdk._run_oracle(
                    self._env, cfg.task_path, self._timeout, cfg.sandbox_user
                )
            else:
                await self.install_agent()
                for scene in cfg.effective_scenes:
                    await self._run_scene(scene)

            await self.verify()

        except TimeoutError:
            self._error = f"Agent timed out after {self._timeout}s"
            logger.error(self._error)
        except ConnectionError as e:
            self._error = str(e)
            logger.error(f"Agent connection lost: {self._error}")
        except ACPError as e:
            self._error = self._classify_acp_error(e)
            logger.error(self._error)
        except Exception as e:
            self._error = str(e)
            logger.error("Run failed", exc_info=True)
        finally:
            await self.cleanup()

        return self._build_result()

    # ── Scene execution ──

    async def _run_scene(self, scene: Scene) -> None:
        """Execute one scene: for each turn, connect as the turn's role, execute, disconnect."""
        cfg = self._config
        logger.info(f"[Scene] {scene.name} — {len(scene.turns)} turns, {len(scene.roles)} roles")

        role_map = {r.name: r for r in scene.roles}
        current_role: str | None = None

        for i, turn in enumerate(scene.turns):
            role = role_map.get(turn.role)
            if not role:
                raise ValueError(f"Turn references unknown role {turn.role!r}")

            # Reconnect if role changed or first turn
            if current_role != turn.role:
                if current_role is not None:
                    await self.disconnect()

                # Override agent/model for this role
                self._agent_launch = AGENT_LAUNCH.get(role.agent, role.agent)
                self._agent_env = resolve_agent_env(role.agent, role.model, role.env or None)

                await self.connect_as(role)
                current_role = turn.role

            prompts = [turn.prompt] if turn.prompt else self._resolved_prompts
            await self.execute(prompts=prompts)

        if current_role is not None:
            await self.disconnect()

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role."""
        cfg = self._config
        t0 = datetime.now()

        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=role.agent,
            agent_launch=AGENT_LAUNCH.get(role.agent, role.agent),
            agent_env=resolve_agent_env(role.agent, role.model, role.env or None),
            sandbox_user=cfg.sandbox_user,
            model=role.model,
            trial_dir=self._trial_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
        )

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # ── Internal helpers ──

    def _classify_acp_error(self, e: ACPError) -> str:
        if "Invalid API key" in e.message:
            from benchflow._agent_env import check_subscription_auth
            from benchflow.agents.registry import infer_env_key_for_model

            key = infer_env_key_for_model(self._config.model) if self._config.model else None
            if key and check_subscription_auth(self._config.agent, key):
                return (
                    f"{key} was rejected as invalid. "
                    f"Subscription auth credentials exist — unset the env var "
                    f"to use them: env -u {key} <command>"
                )
        return str(e)

    def _build_result(self) -> RunResult:
        from benchflow.sdk import SDK, _write_rewards_jsonl

        finished_at = datetime.now()
        _write_rewards_jsonl(self._trial_dir, self._rewards, finished_at)

        return SDK._build_result(
            self._trial_dir,
            task_name=self._config.task_path.name,
            trial_name=self._trial_name or "",
            agent=self._config.agent,
            agent_name=self._agent_name,
            model=self._config.model or "",
            n_tool_calls=self._n_tool_calls,
            prompts=self._resolved_prompts,
            error=self._error,
            verifier_error=self._verifier_error,
            trajectory=self._trajectory,
            partial_trajectory=self._partial_trajectory,
            trajectory_source=self._trajectory_source,
            rewards=self._rewards,
            started_at=self._started_at,
            timing=self._timing,
        )
