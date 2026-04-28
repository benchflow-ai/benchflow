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

Or use ``trial.run()`` for the full lifecycle.

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
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.agents.discovery import BUILTINS_DIR
from benchflow.agents.loader import load_agent_toml
from benchflow.agents.run import connect_acp, execute_prompts
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.install import deploy_skills, install_agent
from benchflow.agents.credentials import upload_subscription_auth, write_credential_files
from benchflow.sandbox.build import (
    _create_environment,
    _inject_skills_into_dockerfile,
    stage_dockerfile_deps,
)
from benchflow.sandbox.lockdown import _resolve_locked_paths, lockdown_paths
from benchflow.sandbox.user import setup_sandbox_user
from benchflow.sandbox.verifier_harden import harden_before_verify
from benchflow.sandbox.verifier_workspace import _seed_verifier_workspace, _snapshot_build_config
from benchflow.trajectories._capture import (
    _capture_session_trajectory,
    _scrape_agent_trajectory,
)
from benchflow.acp.client import ACPClient, ACPError
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.contracts.trial_config import Role, Scene, TrialConfig, Turn
from benchflow.results import TrialResult, TrajectorySource
from benchflow._orchestration import (
    SANDBOX_USAGE_PATH,
    build_result,
    init_trial,
    read_usage_sidecar,
    resolve_prompts,
    run_oracle,
    start_env_and_upload,
    verify_with_harden,
    write_trial_config,
)

logger = logging.getLogger(__name__)


__all__ = ["Trial", "TrialConfig"]


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

        # Populated by cleanup() — agent self-reported token/cost from
        # $BENCHFLOW_USAGE_PATH (PR9). None when the sidecar is absent.
        self._usage: dict[str, int | float] | None = None

        # PR10: optional OTel collector. Spun up in start() when
        # BENCHFLOW_OTEL_ENABLE=1 AND the manifest declares
        # [reporting].otel_enable_env. None otherwise. Falls back path
        # for usage capture when the sidecar isn't written.
        self._otel_collector: Any = None

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
    def result(self) -> TrialResult | None:
        if self._phase not in ("verified", "cleaned"):
            return None
        return self._build_result()

    # ── Phase 1: SETUP (host-side, no container yet) ──

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )

        self._effective_locked = _resolve_locked_paths(
            cfg.sandbox_user, cfg.sandbox_locked_paths
        )

        (
            self._task,
            self._trial_dir,
            self._trial_paths,
            self._started_at,
            self._job_name,
            self._trial_name,
        ) = init_trial(
            cfg.task_path, cfg.job_name, cfg.trial_name, cfg.jobs_dir
        )

        self._agent_env = resolve_agent_env(
            cfg.primary_agent, cfg.primary_model, cfg.agent_env
        )
        # PR9 (PLAN_V2_byoa.md): expose the BYOA usage sidecar path. Shims
        # that compute their own LLM cost write the JSON here; Trial
        # downloads it after agent exit and projects it onto TrialResult.
        # ACP-mode built-ins ignore the var; later PR10 (OTel) populates
        # the same TrialResult fields via a different pipe.
        self._agent_env.setdefault("BENCHFLOW_USAGE_PATH", SANDBOX_USAGE_PATH)
        self._resolved_prompts = resolve_prompts(cfg.task_path, cfg.prompts)
        self._agent_launch = AGENT_LAUNCH.get(cfg.primary_agent, cfg.primary_agent)

        # Copy task dir to temp when Dockerfile mutations are needed
        # (_inject_skills writes into environment/_deps/, stage_dockerfile
        # rewrites COPY paths — neither should modify the source tree)
        effective_task_path = cfg.task_path
        if cfg.context_root or cfg.skills_dir:
            import shutil
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            shutil.copytree(cfg.task_path, tmp / cfg.task_path.name, dirs_exist_ok=True)
            effective_task_path = tmp / cfg.task_path.name
            self._task_tmp = tmp

        if cfg.context_root:
            stage_dockerfile_deps(effective_task_path, Path(cfg.context_root))
        if cfg.skills_dir:
            _inject_skills_into_dockerfile(effective_task_path, Path(cfg.skills_dir))

        self._env = _create_environment(
            cfg.environment, self._task, effective_task_path,
            self._trial_name, self._trial_paths,
        )
        self._timeout = int(self._task.config.agent.timeout_sec or 0)

        write_trial_config(
            self._trial_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
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
        await start_env_and_upload(self._env, self._config.task_path, self._timing)

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        # PR10 — opt-in OTel capture for cost/token. Gated on the
        # BENCHFLOW_OTEL_ENABLE env var; only fires if the agent's manifest
        # declares an otel_enable_env. Endpoint defaults to host.docker.internal
        # (which the user must wire via Docker --add-host on Linux); a custom
        # BENCHFLOW_OTEL_ENDPOINT host overrides if set.
        await self._maybe_start_otel()

        self._phase = "started"

    async def _maybe_start_otel(self) -> None:
        if os.environ.get("BENCHFLOW_OTEL_ENABLE", "").lower() not in {"1", "true", "yes"}:
            return
        try:
            manifest = load_agent_toml(BUILTINS_DIR / self._config.primary_agent)
        except Exception:
            return
        enable_var = manifest.reporting.otel_enable_env
        if not enable_var:
            return
        from benchflow.trajectories.otel import OTelCollector

        self._otel_collector = OTelCollector(
            session_id=self._trial_name or "",
            agent_name=self._config.primary_agent,
            host="0.0.0.0",
            port=0,
        )
        await self._otel_collector.start()
        host = os.environ.get("BENCHFLOW_OTEL_ENDPOINT_HOST", "host.docker.internal")
        endpoint = f"http://{host}:{self._otel_collector.port}"
        self._agent_env[enable_var] = "1"
        self._agent_env.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        self._agent_env.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
        self._agent_env.setdefault("OTEL_SERVICE_NAME", "benchflow-agent")
        logger.info("OTel collector listening on %s for agent %s",
                    endpoint, self._config.primary_agent)

    # ── Phase 3: INSTALL AGENT ──

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous multi-agent scenes (different agents per role),
        each role's agent is installed on-demand in _run_scene/connect_as.
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config

        cwd_result = await self._env.exec("pwd", timeout_sec=10)
        agent_cwd = (cwd_result.stdout or "").strip()
        if not agent_cwd or agent_cwd == "/":
            agent_cwd = "/app"
        self._agent_cwd = agent_cwd

        if cfg.primary_agent == "oracle":
            if cfg.sandbox_user:
                await setup_sandbox_user(
                    self._env, cfg.sandbox_user, workspace=self._agent_cwd
                )
            await _snapshot_build_config(self._env, workspace=self._agent_cwd)
            await _seed_verifier_workspace(self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user)
            await lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        self._agent_cfg = await install_agent(
            self._env, agent_name, self._trial_dir
        )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await write_credential_files(
            self._env, agent_name, self._agent_env,
            self._agent_cfg, cfg.primary_model, cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await upload_subscription_auth(self._env, agent_name, cred_home)

        if cfg.sandbox_user:
            self._agent_cwd = await setup_sandbox_user(
                self._env, cfg.sandbox_user, workspace=self._agent_cwd
            )
        await _snapshot_build_config(self._env, workspace=self._agent_cwd)
        await _seed_verifier_workspace(self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user)

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
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.primary_model,
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
        if self._env and self._agent_launch.strip():
            agent_cmd = self._agent_launch.split()[0].split("/")[-1]
            try:
                await self._env.exec(f"pkill -f '{agent_cmd}' || true", timeout_sec=10)
            except Exception:
                pass
        self._session_tool_count = 0
        self._session_traj_count = 0
        self._phase = "installed"

    # ── Phase 3c: EXECUTE ──

    async def execute(self, prompts: list[str] | None = None) -> tuple[list[dict], int]:
        """Run prompts through the ACP session. Returns (new trajectory, new tool calls).

        execute_prompts returns cumulative session trajectory. We track
        what we've already captured to avoid duplication when the same
        session is reused across multiple turns.
        """
        effective_prompts = prompts or self._resolved_prompts
        prev_session_tools = getattr(self, "_session_tool_count", 0)
        t0 = datetime.now()

        trajectory, n_tool_calls = await execute_prompts(
            self._acp_client,
            self._session,
            effective_prompts,
            self._timeout,
        )

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[getattr(self, "_session_traj_count", 0):]
        self._session_tool_count = n_tool_calls
        self._session_traj_count = len(trajectory)

        self._trajectory.extend(new_events)
        self._n_tool_calls += new_tools
        self._trajectory_source = "acp"

        if "agent_execution" not in self._timing:
            self._timing["agent_execution"] = (datetime.now() - t0).total_seconds()

        self._phase = "executed"
        return trajectory, n_tool_calls

    # ── Phase 4: VERIFY ──

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

        self._rewards, self._verifier_error = await verify_with_harden(
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

        # PR9: download the BYOA usage sidecar before tearing down the env
        # (the file lives in the sandbox FS, not on the host). Best-effort —
        # missing file or download errors leave self._usage as None and
        # TrialResult fields stay null.
        if self._env:
            try:
                local = self._trial_dir / "usage.json"
                await self._env.download_file(SANDBOX_USAGE_PATH, local)
                self._usage = read_usage_sidecar(local)
            except Exception as e:
                logger.debug(f"No usage sidecar (or unreadable): {e}")
                self._usage = None

        # PR10: stop the OTel collector and fall back to its usage_summary
        # if the sidecar didn't populate. Sidecar (agent self-report)
        # always wins because the agent's accounting is more accurate
        # than wire-level observation (sub-agents, retries, batched calls).
        if self._otel_collector is not None:
            try:
                await self._otel_collector.stop()
                if self._usage is None:
                    summary = self._otel_collector.trajectory.usage_summary()
                    if any(v for v in summary.values() if v):
                        self._usage = {k: v for k, v in summary.items() if v is not None}
            except Exception as e:
                logger.warning(f"OTel collector stop failed: {e}")

        await self.disconnect()

        if self._env:
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        if hasattr(self, "_task_tmp") and self._task_tmp:
            import shutil
            shutil.rmtree(self._task_tmp, ignore_errors=True)

        self._phase = "cleaned"

    # ── Full run ──

    async def run(self) -> TrialResult:
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
                # git safe.directory needed for SWE-bench tasks with sandbox_user
                import shlex
                await self._env.exec(
                    f"git config --global --add safe.directory "
                    f"{shlex.quote(self._agent_cwd)} 2>/dev/null || true",
                    user="root", timeout_sec=10,
                )
                self._trajectory, self._agent_name = await run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
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

        if self._trial_dir is None:
            from benchflow.results import TrialResult
            return TrialResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return self._build_result()

    # ── Scene execution ──

    async def _run_scene(self, scene: Scene) -> None:
        """Execute one scene: for each turn, connect as the turn's role, execute, disconnect.

        Single-role multi-turn is supported (the ACP session is preserved across
        same-role turns). Multi-role scenes are rejected: there is no inter-role
        coordination here. Use ``benchflow.experimental.mailbox.MailboxRunner``
        for multi-agent runs.
        """
        if len(scene.roles) > 1:
            raise ValueError(
                f"Trial._run_scene is single-role only; scene {scene.name!r} declares "
                f"{len(scene.roles)} roles. Multi-role execution is not implemented in the "
                "graduated path — use benchflow.experimental.mailbox.MailboxRunner instead."
            )
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
                await self.connect_as(role)
                current_role = turn.role

            if turn.prompt:
                prompts = [turn.prompt]
            elif self._resolved_prompts:
                prompts = [self._resolved_prompts[0]]
            else:
                prompts = ["Solve the task described in /app/instruction.md"]
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
            from benchflow.agents.env import check_subscription_auth
            from benchflow.agents.registry import infer_env_key_for_model

            key = infer_env_key_for_model(self._config.primary_model) if self._config.primary_model else None
            if key and check_subscription_auth(self._config.primary_agent, key):
                return (
                    f"{key} was rejected as invalid. "
                    f"Subscription auth credentials exist — unset the env var "
                    f"to use them: env -u {key} <command>"
                )
        return str(e)

    def _build_result(self) -> TrialResult:
        return build_result(
            self._trial_dir,
            task_name=self._config.task_path.name,
            trial_name=self._trial_name or "",
            agent=self._config.primary_agent,
            agent_name=self._agent_name,
            model=self._config.primary_model or "",
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
            usage=self._usage,
        )
