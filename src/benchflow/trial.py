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
import contextlib
import json
import logging
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_env import resolve_agent_env
from benchflow._agent_setup import (
    _link_skill_paths,
    apply_web_tool_policy,
    deploy_skills,
    install_agent,
)
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
from benchflow.rollouts import _helpers as rollout_helpers
from benchflow.rollouts.config import GENERATED_SKILLS_ROOT as GENERATED_SKILLS_ROOT
from benchflow.rollouts.config import SKILL_MODE_DEFAULT as SKILL_MODE_DEFAULT
from benchflow.rollouts.config import SKILL_MODE_SELF_GEN, Role, RolloutConfig, Scene
from benchflow.rollouts.config import Turn as Turn
from benchflow.user import RoundResult

logger = logging.getLogger(__name__)

_DISALLOW_WEB_TOOLS_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"


def _task_disallows_internet(task: Any) -> bool:
    """Return True when task.toml requests no internet for the agent task."""
    env_config = getattr(getattr(task, "config", None), "environment", None)
    return getattr(env_config, "allow_internet", True) is False


def _apply_web_policy(agent_env: dict[str, str], *, disallow: bool) -> dict[str, str]:
    """Inject BenchFlow's no-web policy marker into agent env when requested."""
    if not disallow:
        return agent_env
    return {**agent_env, _DISALLOW_WEB_TOOLS_ENV: "1"}


def _agent_launch_with_web_policy(agent: str, *, disallow: bool) -> str:
    """Return launch command, appending the agent's no-web launch knob if any."""
    launch = AGENT_LAUNCH.get(agent, agent)
    if not disallow:
        return launch
    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.disallow_web_tools_launch_suffix:
        return launch + agent_cfg.disallow_web_tools_launch_suffix
    return launch


def _skill_nudge(agent_env: dict[str, str] | None) -> str:
    """Read skill nudge from explicit agent env or the host environment."""
    return (agent_env or {}).get("BENCHFLOW_SKILL_NUDGE") or os.environ.get(
        "BENCHFLOW_SKILL_NUDGE", ""
    )


def _safe_skill_name(value: str) -> str:
    """Return a filesystem-safe generated skill directory name."""
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return name or "generated-task"


def _skill_frontmatter_name(skill_dir: Path) -> str:
    """Read a skill's frontmatter name, falling back to the directory name."""
    from benchflow.skills import parse_skill

    info = parse_skill(skill_dir / "SKILL.md")
    return info.name if info and info.name else skill_dir.name


def _resolve_skill_creator_root(path: str | Path | None) -> tuple[Path, str]:
    """Resolve a skills root containing the official skill-creator directory.

    BenchFlow mounts skills as a root directory whose children are individual
    skill directories. If the caller points directly at skill-creator, use its
    parent as the mounted root and leave the skill pack contents unchanged.
    """
    candidates: list[Path] = []
    scan_single_skill_roots: set[Path] = set()
    if path:
        explicit_path = Path(path).expanduser()
        candidates.append(explicit_path)
        scan_single_skill_roots.add(explicit_path)
    env_path = os.environ.get("BENCHFLOW_SKILL_CREATOR_DIR")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append(env_candidate)
        scan_single_skill_roots.add(env_candidate)
    repo_skill_creator = (
        Path(__file__).resolve().parents[2] / ".claude" / "skills" / "skill-creator"
    )
    cwd_skill_creator = Path.cwd() / ".claude" / "skills" / "skill-creator"
    candidates.append(repo_skill_creator)
    if cwd_skill_creator != repo_skill_creator:
        candidates.append(cwd_skill_creator)
    candidates.extend(
        [
            Path.home() / ".claude" / "skills" / "skill-creator",
            Path.home() / ".codex" / "skills" / ".system" / "skill-creator",
            Path.home() / ".agents" / "skills" / "skill-creator",
        ]
    )

    for candidate in candidates:
        if (candidate / "SKILL.md").exists():
            return candidate.parent, _skill_frontmatter_name(candidate)
        if (candidate / "skill-creator" / "SKILL.md").exists():
            skill_dir = candidate / "skill-creator"
            return candidate, _skill_frontmatter_name(skill_dir)
        if candidate in scan_single_skill_roots and candidate.is_dir():
            skill_dirs = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and (child / "SKILL.md").exists()
            ]
            if len(skill_dirs) == 1:
                skill_dir = skill_dirs[0]
                return candidate, _skill_frontmatter_name(skill_dir)

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find skill-creator. Pass --skill-creator-dir or set "
        f"BENCHFLOW_SKILL_CREATOR_DIR. Checked: {checked}"
    )


def _self_gen_prompt(
    task_path: Path, generated_skills_root: str, skill_creator_name: str
) -> str:
    """Prompt the clean creator agent to use the mounted skill-creator skill."""
    skill_dir_name = f"{_safe_skill_name(task_path.name)}-skill"
    target_dir = f"{generated_skills_root}/{skill_dir_name}"
    return f"""Use the {skill_creator_name} skill exactly as provided.

Read /instruction.md and inspect the task environment only as needed to understand the reusable workflow. Do not solve the task directly.

Create one or more complete Anthropic-standard skill packs as immediate child directories under:

{generated_skills_root}

Use this suggested path if one skill is enough:

{target_dir}

Each generated skill pack path must look like {generated_skills_root}/<skill-name>/SKILL.md. It may include scripts/, references/, assets/, examples, or other bundled resources when they help a fresh solver avoid repeated work.

The solver context will start with a clean agent session and only the generated skill packs mounted. Make the skills useful for solving this task type from the same sandbox environment."""


async def _ensure_sandbox_dir(
    env: Any, path: str | Path, sandbox_user: str | None = None
) -> None:
    """Create a sandbox directory and optionally make it writable by the agent."""
    q_path = shlex.quote(str(path))
    cmd = f"mkdir -p {q_path}"
    if sandbox_user:
        q_user = shlex.quote(sandbox_user)
        cmd += f" && chown -R {q_user}:{q_user} {q_path}"
    result = await env.exec(cmd, timeout_sec=10)
    if result.return_code != 0:
        raise RuntimeError(
            f"Failed to create sandbox directory {path}: "
            f"{result.stderr or result.stdout}"
        )


class TrialConfig(RolloutConfig):
    """Compatibility name while the lifecycle class migrates to Rollout."""


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
        self._disallow_web_tools: bool = False

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
        if config.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use SDK.run(), "
                "Job.run(), or bf.run(TrialConfig(...)) instead of Trial.create()."
            )
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

    def _require_trial_dir(self) -> Path:
        if self._trial_dir is None:
            raise RuntimeError("Trial.setup() must run before this phase")
        return self._trial_dir

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise RuntimeError("Trial.setup() must run before building a result")
        return self._started_at

    # ── Phase 1: SETUP (host-side, no container yet) ──

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — /solution stays visible "
                "to the agent for the entire trial."
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
        ) = rollout_helpers.init_rollout(
            cfg.task_path, cfg.job_name, cfg.trial_name, cfg.jobs_dir
        )

        self._disallow_web_tools = (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        ) and cfg.primary_agent != "oracle"
        self._agent_env = _apply_web_policy(
            resolve_agent_env(cfg.primary_agent, cfg.primary_model, cfg.agent_env),
            disallow=self._disallow_web_tools,
        )
        self._resolved_prompts = rollout_helpers.resolve_prompts(
            cfg.task_path,
            cfg.prompts,
            skills_dir=cfg.skills_dir,
            skill_nudge=_skill_nudge(cfg.agent_env),
            agent=cfg.primary_agent,
        )
        self._agent_launch = _agent_launch_with_web_policy(
            cfg.primary_agent,
            disallow=self._disallow_web_tools,
        )

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
            cfg.environment,
            self._task,
            effective_task_path,
            self._trial_name,
            self._trial_paths,
            preserve_agent_network=self._disallow_web_tools,
        )
        self._timeout = int(self._task.config.agent.timeout_sec or 0)

        rollout_helpers.write_config(
            self._trial_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            environment=cfg.environment,
            skills_dir=cfg.skills_dir,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
        )

        self._phase = "setup"

    # ── Phase 2: START (container comes up) ──

    async def start(self) -> None:
        """Start the environment and upload task files."""
        await rollout_helpers.start_env_and_upload(
            self._env, self._config.task_path, self._timing
        )

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        self._phase = "started"

    # ── Phase 3: INSTALL AGENT ──

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous multi-agent scenes (different agents per role),
        each role's agent is installed on-demand in _run_scene/connect_as.
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config
        trial_dir = self._require_trial_dir()

        cwd_result = await self._env.exec("pwd", timeout_sec=10)
        agent_cwd = (cwd_result.stdout or "").strip() or "/app"
        self._agent_cwd = agent_cwd

        if cfg.primary_agent == "oracle":
            if cfg.sandbox_user:
                await setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            await _snapshot_build_config(self._env, workspace=self._agent_cwd)
            await _seed_verifier_workspace(
                self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
            )
            await deploy_skills(
                self._env,
                cfg.task_path,
                cfg.skills_dir,
                None,
                cfg.sandbox_user,
                self._agent_cwd,
                self._task,
                include_task_skills=cfg.include_task_skills,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        self._agent_cfg = await install_agent(self._env, agent_name, trial_dir)
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await write_credential_files(
            self._env,
            agent_name,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await upload_subscription_auth(self._env, agent_name, cred_home)

        if cfg.sandbox_user:
            self._agent_cwd = await setup_sandbox_user(
                self._env,
                cfg.sandbox_user,
                workspace=self._agent_cwd,
                timeout_sec=cfg.sandbox_setup_timeout,
            )
        await apply_web_tool_policy(
            self._env,
            agent_name,
            self._agent_cfg,
            cred_home,
            disallow=self._disallow_web_tools,
        )
        await _snapshot_build_config(self._env, workspace=self._agent_cwd)
        await _seed_verifier_workspace(
            self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
        )

        await deploy_skills(
            self._env,
            cfg.task_path,
            cfg.skills_dir,
            self._agent_cfg,
            cfg.sandbox_user,
            self._agent_cwd,
            self._task,
            include_task_skills=cfg.include_task_skills,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # ── Phase 3b: CONNECT (ACP session — re-entrant) ──

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        cfg = self._config
        trial_dir = self._require_trial_dir()
        t0 = datetime.now()

        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.primary_model,
            trial_dir=trial_dir,
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
            with contextlib.suppress(Exception):
                await self._env.exec(f"pkill -f '{agent_cmd}' || true", timeout_sec=10)
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
        if self._acp_client is None:
            raise RuntimeError("Trial.connect() must run before execute()")
        prev_session_tools = getattr(self, "_session_tool_count", 0)
        t0 = datetime.now()

        trajectory, n_tool_calls = await execute_prompts(
            self._acp_client,
            self._session,
            effective_prompts,
            self._timeout,
            idle_timeout=self._config.agent_idle_timeout,
        )

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[getattr(self, "_session_traj_count", 0) :]
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

        self._rewards, self._verifier_error = await rollout_helpers.verify(
            self._env,
            self._task,
            self._trial_paths,
            self._timing,
            sandbox_user=cfg.sandbox_user,
            workspace=self._agent_cwd,
        )

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
        import benchflow._harbor as harbor_compat
        from benchflow._sandbox import _build_cleanup_cmd, _read_hardening_config

        self._trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        await self._env.exec(
            "rm -rf /logs/verifier && mkdir -p /logs/verifier /app && "
            "chmod 777 /logs/verifier",
            user="root",
            timeout_sec=10,
        )
        # Purge agent-injected conftest/sitecustomize/.pth without
        # killing processes or restoring workspace.
        # Honor per-task [verifier.hardening] opt-outs from task.toml.
        hardening = _read_hardening_config(getattr(self._task, "task_dir", None))
        await self._env.exec(_build_cleanup_cmd(hardening), user="root", timeout_sec=10)

        rewards = None
        verifier_output = None
        verifier_error = None
        try:
            verifier = harbor_compat.make_verifier(
                task=self._task,
                trial_paths=self._trial_paths,
                environment=self._env,
            )
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = verifier_result.rewards
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
            logger.warning(verifier_error)
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.warning(verifier_error)

        return rewards, verifier_output, verifier_error

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

        if self._env and self._config.export_generated_skills_to:
            try:
                await self._export_generated_skills()
            except Exception as e:
                logger.warning(f"Generated skill export failed: {e}")

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

    async def run(self) -> RunResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        agent_timed_out = False
        if cfg.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use SDK.run(), "
                "Job.run(), or bf.run(TrialConfig(...)) instead of Trial.run()."
            )
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
                    user="root",
                    timeout_sec=10,
                )
                self._trajectory, self._agent_name = await rollout_helpers.run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
                )
            else:
                await self.install_agent()
                try:
                    try:
                        if cfg.user is not None:
                            await self._run_user_loop()
                        else:
                            for scene in cfg.effective_scenes:
                                await self._run_scene(scene)
                    except TimeoutError as e:
                        agent_timed_out = True
                        detail = str(e).strip()
                        self._error = (
                            detail or f"Agent timed out after {self._timeout}s"
                        )
                        logger.error(self._error)
                finally:
                    if cfg.oracle_access:
                        await self._env.exec(
                            "mv /solution_oracle_backup /solution 2>/dev/null || true",
                            user="root",
                            timeout_sec=10,
                        )

            if not cfg.skip_verify:
                await self.verify()
                if agent_timed_out and self._rewards is None:
                    self._rewards = {"reward": 0.0}
                    self._verifier_error = None

        except TimeoutError as e:
            # Preserve the watchdog's diagnostic message ("Agent idle for 600s
            # with no new tool call ...") if it raised one. Fall back to the
            # generic wall-clock message only when there's no detail.
            detail = str(e).strip()
            self._error = detail or f"Agent timed out after {self._timeout}s"
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
            from benchflow.models import RunResult

            return RunResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return self._build_result()

    # ── Scene execution ──

    _OUTBOX_DIR = "/app/.outbox"

    async def _export_generated_skills(self) -> None:
        """Download creator-produced skills before sandbox cleanup."""
        export_target = self._config.export_generated_skills_to
        if export_target is None:
            return
        target = Path(export_target)
        target.mkdir(parents=True, exist_ok=True)
        await self._env.download_dir(self._config.generated_skills_root, target)

    async def _activate_scene_skills(self, scene: Scene) -> None:
        """Activate scene-local skills by linking them into role discovery paths."""
        if not scene.skills_dir:
            return
        if self._env is None:
            raise RuntimeError("Environment is not started")

        source = str(scene.skills_dir)
        local_source = Path(source).expanduser()
        if local_source.is_dir():
            remote_source = f"/skills/{_safe_skill_name(scene.name)}"
            await _ensure_sandbox_dir(self._env, Path(remote_source).parent)
            await self._env.upload_dir(local_source, remote_source)
        elif source.startswith("/"):
            remote_source = source
        else:
            raise FileNotFoundError(f"Scene skills_dir not found: {scene.skills_dir}")

        home = (
            f"/home/{self._config.sandbox_user}"
            if self._config.sandbox_user
            else "/root"
        )
        for role in scene.roles:
            agent_cfg = AGENTS.get(role.agent)
            if not agent_cfg or not agent_cfg.skill_paths:
                continue
            await _link_skill_paths(
                self._env,
                remote_source,
                agent_cfg.skill_paths,
                home,
                self._agent_cwd,
                self._config.sandbox_user,
            )

    async def _run_scene(self, scene: Scene) -> None:
        """Execute one scene: for each turn, connect as the turn's role, execute, disconnect.

        For multi-role scenes, agents communicate via outbox files:
        an agent writes ``/app/.outbox/{recipient}.json`` with
        ``{"to": "role_name", "content": "..."}`` and the scheduler
        injects received messages into the next turn's prompt.

        Inter-role messages are persisted to ``trial_dir/scene_messages.jsonl``.
        """
        cfg = self._config
        logger.info(
            f"[Scene] {scene.name} — {len(scene.turns)} turns, {len(scene.roles)} roles"
        )
        await self._activate_scene_skills(scene)

        role_map = {r.name: r for r in scene.roles}
        current_role: str | None = None
        multi_role = len(scene.roles) > 1
        scene_messages: list[dict] = []

        if multi_role:
            setup_cmd = f"rm -rf {self._OUTBOX_DIR} && mkdir -p {self._OUTBOX_DIR}"
            if cfg.sandbox_user:
                user = shlex.quote(cfg.sandbox_user)
                setup_cmd += f" && chown {user}:{user} {self._OUTBOX_DIR}"
            await self._env.exec(setup_cmd, timeout_sec=10)

        inbox: dict[str, list[str]] = {r.name: [] for r in scene.roles}
        turn_counter = 0

        for _i, turn in enumerate(scene.turns):
            role = role_map.get(turn.role)
            if not role:
                raise ValueError(f"Turn references unknown role {turn.role!r}")

            if current_role != turn.role:
                if current_role is not None:
                    await self.disconnect()
                await self.connect_as(role)
                current_role = turn.role

            if turn.prompt:
                base_prompt = turn.prompt
            elif self._resolved_prompts:
                base_prompt = self._resolved_prompts[0]
            else:
                base_prompt = "Solve the task described in /app/instruction.md"

            pending = inbox.get(turn.role, [])
            if pending:
                parts = [base_prompt, "\n---\nMessages from other agents:\n"]
                parts.extend(pending)
                prompts = ["\n".join(parts)]
                inbox[turn.role] = []
            else:
                prompts = [base_prompt]

            await self.execute(prompts=prompts)

            if multi_role:
                if current_role is None:
                    raise RuntimeError("No active role after scene turn execution")
                for recipient, content in await self._read_scene_outbox(current_role):
                    turn_counter += 1
                    inbox.setdefault(recipient, []).append(
                        f"**From {current_role}:** {content}"
                    )
                    scene_messages.append(
                        {
                            "scene": scene.name,
                            "turn": turn_counter,
                            "sender": current_role,
                            "recipient": recipient,
                            "content": content,
                        }
                    )

        if current_role is not None:
            await self.disconnect()

        if scene_messages and self._trial_dir:
            msg_path = self._trial_dir / "scene_messages.jsonl"
            with msg_path.open("a") as f:
                for m in scene_messages:
                    f.write(json.dumps(m) + "\n")
            logger.info(
                f"[Scene] {scene.name}: {len(scene_messages)} messages → {msg_path}"
            )

    async def _read_scene_outbox(self, sender: str) -> list[tuple[str, str]]:
        """Read and clear outbox files left by *sender*. Returns [(recipient, content), ...]."""
        result = await self._env.exec(
            f"ls {self._OUTBOX_DIR}/*.json 2>/dev/null || true",
            timeout_sec=10,
        )
        files = [
            f.strip() for f in (result.stdout or "").strip().splitlines() if f.strip()
        ]
        messages: list[tuple[str, str]] = []
        for fpath in files:
            quoted = shlex.quote(fpath)
            cat = await self._env.exec(f"cat {quoted}", timeout_sec=10)
            try:
                data = json.loads(cat.stdout or "{}")
                recipient = data.get("to", "")
                content = data.get("content", "")
                if recipient and content:
                    messages.append((recipient, content))
                    logger.info(
                        f"[Scene] outbox: {sender} → {recipient}: {content[:80]}"
                    )
            except json.JSONDecodeError:
                logger.warning(f"[Scene] invalid JSON in outbox: {fpath}")
            await self._env.exec(f"rm -f {quoted}", timeout_sec=10)
        return messages

    async def _run_user_loop(self) -> None:
        """Execute a user-driven progressive-disclosure loop.

        Each round: user.run() → connect → agent.execute() → disconnect →
        soft_verify() → build RoundResult → repeat. Stops when user.run()
        returns None or max_user_rounds is reached.
        """
        cfg = self._config
        user = cfg.user
        assert user is not None

        if len(cfg.effective_scenes) > 1:
            raise ValueError(
                "User-driven loops operate on a single scene. "
                f"Got {len(cfg.effective_scenes)} scenes."
            )
        scene = cfg.effective_scenes[0]
        if len(scene.roles) != 1:
            raise ValueError(
                "User-driven loops require a single-role scene. "
                f"Got {len(scene.roles)} roles."
            )
        role = scene.roles[0]

        instruction = (
            self._resolved_prompts[0]
            if self._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        # Oracle access: read /solution before the agent runs, then remove it
        solution: str | None = None
        if cfg.oracle_access:
            cat = await self._env.exec(
                "cat /solution/solve.sh 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )
            solution = (cat.stdout or "").strip() or None

        await user.setup(instruction, solution)

        # Hide oracle files from agent — move rather than delete so the
        # final verify() can still access /solution if the verifier needs it.
        if cfg.oracle_access:
            await self._env.exec(
                "mv /solution /solution_oracle_backup 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )

        round_result: RoundResult | None = None
        rounds_log: list[dict] = []

        for round_num in range(cfg.max_user_rounds):
            try:
                prompt = await user.run(round_num, instruction, round_result)
            except Exception as e:
                self._error = f"user.run() failed at round {round_num}: {e}"
                logger.error(self._error, exc_info=True)
                break

            if prompt is None:
                logger.info(f"[User] stopped at round {round_num}")
                break

            logger.info(
                f"[User] round {round_num}: prompt={prompt[:80]!r}..."
                if len(prompt) > 80
                else f"[User] round {round_num}: prompt={prompt!r}"
            )

            # Fresh ACP session each round — agent starts clean but sees
            # its previous workspace changes in the shared sandbox.
            traj_before = len(self._trajectory)
            try:
                await self.connect_as(role)
                await self.execute(prompts=[prompt])
            finally:
                await self.disconnect()

            round_trajectory = self._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            # Soft verify: run tests after agent disconnected but before
            # next round. Temporarily restore /solution so the verifier can
            # access it, then re-hide before the next agent round.
            if cfg.oracle_access:
                await self._env.exec(
                    "mv /solution_oracle_backup /solution 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
            try:
                rewards, verifier_output, verifier_error = await self.soft_verify()
            finally:
                if cfg.oracle_access:
                    await self._env.exec(
                        "mv /solution /solution_oracle_backup 2>/dev/null || true",
                        user="root",
                        timeout_sec=10,
                    )

            round_result = RoundResult(
                round=round_num,
                trajectory=round_trajectory,
                rewards=rewards,
                verifier_output=verifier_output,
                verifier_error=verifier_error,
                n_tool_calls=round_tools,
            )

            rounds_log.append(
                {
                    "round": round_num,
                    "prompt": prompt,
                    "rewards": rewards,
                    "verifier_error": verifier_error,
                    "n_tool_calls": round_tools,
                    "n_trajectory_events": len(round_trajectory),
                }
            )

            logger.info(
                f"[User] round {round_num} done: rewards={rewards}, tools={round_tools}"
            )

        # Persist round log
        if rounds_log and self._trial_dir:
            log_path = self._trial_dir / "user_rounds.jsonl"
            with log_path.open("w") as f:
                for entry in rounds_log:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"[User] {len(rounds_log)} rounds → {log_path}")

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role.

        Installs the role's agent binary and credentials if it differs
        from the primary agent (which was set up in install_agent()).
        Updates _agent_launch so disconnect() kills the correct process.
        """
        cfg = self._config
        trial_dir = self._require_trial_dir()
        t0 = datetime.now()

        # Merge cfg.agent_env (config-level) with role.env (role-specific) so
        # provider creds from YAML reach the agent. role.env wins on overlap.
        disallow_web_tools = getattr(self, "_disallow_web_tools", None)
        if disallow_web_tools is None:
            disallow_web_tools = _task_disallows_internet(getattr(self, "_task", None))
        disallow_web_tools = bool(disallow_web_tools and role.agent != "oracle")
        agent_launch = _agent_launch_with_web_policy(
            role.agent,
            disallow=disallow_web_tools,
        )
        agent_env = _apply_web_policy(
            resolve_agent_env(
                role.agent,
                role.model,
                {**(cfg.agent_env or {}), **(role.env or {})},
            ),
            disallow=disallow_web_tools,
        )

        if role.agent != cfg.primary_agent:
            agent_cfg = await install_agent(self._env, role.agent, trial_dir)
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await write_credential_files(
                self._env,
                role.agent,
                agent_env,
                agent_cfg,
                role.model,
                cred_home,
            )
            if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                await upload_subscription_auth(self._env, role.agent, cred_home)
            await apply_web_tool_policy(
                self._env,
                role.agent,
                agent_cfg,
                cred_home,
                disallow=disallow_web_tools,
            )

        self._agent_launch = agent_launch

        self._acp_client, self._session, self._agent_name = await connect_acp(
            env=self._env,
            agent=role.agent,
            agent_launch=agent_launch,
            agent_env=agent_env,
            sandbox_user=cfg.sandbox_user,
            model=role.model,
            trial_dir=trial_dir,
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
        return str(e)

    def _build_result(self) -> RunResult:
        trial_dir = self._require_trial_dir()
        return rollout_helpers.build_result(
            trial_dir,
            task_name=self._config.task_path.name,
            rollout_name=self._trial_name or "",
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
            started_at=self._require_started_at(),
            timing=self._timing,
        )
