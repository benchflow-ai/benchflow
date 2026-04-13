"""benchflow SDK — unified run() that uses ACP inside Harbor environments.

``SDK.run()`` is a thin orchestrator that calls into five phase modules.
Each phase module owns one slice of the run loop and is independently
greppable. This docstring is the map: read it before navigating the file.

Run loop (SDK.run, top to bottom)
---------------------------------

::

    ┌─ SETUP (host) ──────────────────────────────────────────────┐
    │  _init_trial            task, trial_dir, paths, names        │
    │  resolve_agent_env      env vars: inherit, mirror, vertex,   │  ← _agent_env
    │                         subscription auth detection          │
    │  _resolve_prompts       prompt list (instruction.md fallback)│
    │  stage_dockerfile_deps  COPY rewrites for context_root       │  ← _env_setup
    │  _inject_skills_…       Dockerfile skill mount               │  ← _env_setup
    │  _create_environment    Docker or Daytona, not yet started   │  ← _env_setup
    │  _write_config          config.json → trial_dir              │
    └──────────────────────────────────────────────────────────────┘
    ┌─ START (sandbox) ───────────────────────────────────────────┐
    │  _start_env_and_upload  spin up container, copy task files   │
    │  pre_agent_hooks        user callbacks (services, etc.)      │
    └──────────────────────────────────────────────────────────────┘
    ┌─ AGENT (oracle: _run_oracle and skip everything below) ─────┐
    │  install_agent             registry-driven install_cmd      │  ← _agent_setup
    │  write_credential_files    AgentConfig.credential_files     │  ← _credentials
    │  upload_subscription_auth  if host login files detected     │  ← _credentials
    │  setup_sandbox_user        non-root user + path lockdown    │  ← _sandbox
    │  deploy_skills             symlink skills into agent paths  │  ← _agent_setup
    │  lockdown_paths            chmod -r on solution / tests     │  ← _sandbox
    │  connect_acp               stdio pipe + ACP initialize/new  │  ← _acp_run
    │  execute_prompts           multi-turn session/prompt loop   │  ← _acp_run
    └──────────────────────────────────────────────────────────────┘
    ┌─ VERIFY ────────────────────────────────────────────────────┐
    │  (fallback) _scrape_agent_trajectory  if ACP captured none   │
    │             — labeled trajectory_source="scraped" UNTRUSTED  │
    │  harden_before_verify  permissions reset for verifier root   │  ← _sandbox
    │  _verify                Harbor verifier → rewards            │
    │  _build_result          RunResult + result.json + timing     │
    └──────────────────────────────────────────────────────────────┘
    finally: env.stop()

Phase modules (extracted from sdk.py — see refactor branch for the arc)
-----------------------------------------------------------------------
- ``_agent_env``    env var resolution: auto-inherit, vertex ADC, provider
                    BENCHFLOW_PROVIDER_*, env_mapping, subscription auth
- ``_credentials``  credential file writing: upload_credential, agent +
                    provider credential_files, gemini vertex settings,
                    upload_subscription_auth
- ``_sandbox``      sandbox user creation, privilege drop (setpriv/su),
                    path lockdown, verifier hardening + VERIFIER_ENV /
                    CLEANUP_CMD constants
- ``_acp_run``      ACP transport bring-up + the multi-turn prompt loop.
                    Imports ``build_priv_drop_cmd`` from ``_sandbox`` —
                    the only allowed horizontal phase-to-phase import.
- ``_agent_setup``  install_agent (registry-driven) + deploy_skills
                    (runtime upload + per-agent distribution)

Support modules
---------------
- ``_env_setup``    Dockerfile staging, skills injection, DinD patching,
                    ``_create_environment``
- ``_trajectory``   ACP-native + agent-scraped trajectory capture
- ``models``        ``RunResult``, ``AgentInstallError``, ``AgentTimeoutError``,
                    ``TrajectorySource`` (public — re-exported from ``benchflow``)
- ``_scoring``      pure functions: ``extract_reward``,
                    ``classify_verifier_error``, pass-rate math
- ``agents/registry``  ``AGENTS``, ``AgentConfig`` — see registry.py docstring
                       for the "add a new agent" recipe
- ``agents/providers`` ``PROVIDERS``, ``ProviderConfig`` — see providers.py
                       docstring for the "add a new provider" recipe

Critical invariants
-------------------
- The phases above run in strict order. Functions in ``_agent_env`` are pure
  and can be called in tests independently; everything from
  ``_start_env_and_upload`` onward assumes a live container and ordered setup.
- Trajectory source is *labeled*, not deleted. ``trajectory_source`` is one of
  ``"acp"`` (trusted) | ``"scraped"`` (UNTRUSTED, agent-writable, forgeable) |
  ``"partial_acp"``. Verifier and metrics consumers decide trust per source.
- ``n_tool_calls`` is set from ACP only and **never** overwritten by scraped
  trajectories — see the security test in ``tests/test_verify.py``.
- Adding a new agent or provider must be a registry-only change. No edits to
  this file should be needed; ``tests/test_registry_invariants.py`` enforces
  the contract.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_env import resolve_agent_env
from benchflow._agent_setup import deploy_skills, install_agent
from benchflow._credentials import (
    upload_subscription_auth,
    write_credential_files,
)
from benchflow._env_setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    _patch_harbor_dind,
    stage_dockerfile_deps,
)
from benchflow._sandbox import (
    _resolve_locked_paths,
    harden_before_verify,
    lockdown_paths,
    setup_sandbox_user,
)
from benchflow._trajectory import (
    _capture_session_trajectory,
    _scrape_agent_trajectory,
)
from benchflow.acp.client import ACPClient, ACPError
from benchflow.agents.registry import AGENT_LAUNCH
from benchflow.models import RunResult, TrajectorySource

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000  # max chars for diagnostic stdout/stderr in logs


# Apply at import time so any Harbor DockerEnvironment in this process
# (SDK.run or otherwise) gets the env-var rewrite, and so we patch exactly
# once without an idempotency guard. Do not move into SDK.run().
_patch_harbor_dind()


class SDK:
    """benchflow SDK.

    Usage:
        sdk = SDK()
        result = await sdk.run(
            task_path="path/to/task",
            agent="claude-agent-acp",
            prompts=["solve the task", "now test your solution"],
            agent_env={"ANTHROPIC_API_KEY": "..."},
        )
        print(result.rewards)
        print(result.trajectory)
    """

    @staticmethod
    def _init_trial(
        task_path: Path,
        job_name: str | None,
        trial_name: str | None,
        jobs_dir: str | Path,
    ) -> tuple["Task", Path, "TrialPaths", datetime, str, str]:
        """Set up trial directory tree and return core trial objects."""
        from uuid import uuid4

        task = Task(task_path)
        job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        trial_name = trial_name or f"{task_path.name}__{uuid4().hex[:8]}"
        trial_dir = Path(jobs_dir) / job_name / trial_name
        trial_paths = TrialPaths(trial_dir)
        started_at = datetime.now()
        # Pre-create trial directory tree so Docker doesn't create them as root.
        trial_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (trial_dir / subdir).mkdir(exist_ok=True)
        return task, trial_dir, trial_paths, started_at, job_name, trial_name

    @staticmethod
    def _write_config(
        trial_dir: Path,
        *,
        task_path: Path,
        agent: str,
        model: str | None,
        environment: str,
        skills_dir: str | Path | None,
        sandbox_user: str | None,
        context_root: str | Path | None,
        sandbox_locked_paths: list[str] | None = None,
        timeout: int,
        started_at: datetime,
        agent_env: dict[str, str],
    ) -> None:
        """Write config.json to trial_dir with secrets filtered out."""
        _secret_substrings = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS")
        recorded_env = {
            k: v
            for k, v in agent_env.items()
            if not any(s in k.upper() for s in _secret_substrings)
        }
        config_data = {
            "task_path": str(task_path),
            "agent": agent,
            "model": model,
            "environment": environment,
            "skills_dir": str(skills_dir) if skills_dir else None,
            "sandbox_user": sandbox_user,
            "sandbox_locked_paths": sandbox_locked_paths,
            "context_root": str(context_root) if context_root else None,
            "timeout_sec": timeout,
            "started_at": str(started_at),
            "agent_env": recorded_env,
        }
        (trial_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    @staticmethod
    def _build_result(
        trial_dir: Path,
        *,
        task_name: str,
        trial_name: str,
        agent: str,
        agent_name: str,
        model: str,
        n_tool_calls: int,
        prompts: list[str],
        error: str | None,
        verifier_error: str | None,
        trajectory: list[dict],
        partial_trajectory: bool,
        trajectory_source: TrajectorySource | None = None,
        rewards: dict | None,
        started_at: datetime,
        timing: dict[str, float],
    ) -> RunResult:
        """Build RunResult and write result.json, timing.json, prompts.json, trajectory."""
        finished_at = datetime.now()
        result = RunResult(
            task_name=task_name,
            trial_name=trial_name,
            rewards=rewards,
            trajectory=trajectory,
            agent=agent,
            agent_name=agent_name,
            model=model,
            n_tool_calls=n_tool_calls,
            n_prompts=len(prompts),
            error=error,
            verifier_error=verifier_error,
            partial_trajectory=partial_trajectory,
            trajectory_source=trajectory_source,
            started_at=started_at,
            finished_at=finished_at,
        )
        # Finalize timing — use the locals (RunResult fields are typed
        # datetime | None and would need narrowing)
        timing["total"] = (finished_at - started_at).total_seconds()
        timing = {k: round(v, 1) for k, v in timing.items()}
        # Save trajectory
        traj_dir = trial_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "acp_trajectory.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in trajectory)
        )
        # Save result.json, prompts.json, timing.json
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": result.task_name,
                    "trial_name": result.trial_name,
                    "rewards": result.rewards,
                    "agent": result.agent,
                    "agent_name": result.agent_name,
                    "model": result.model,
                    "n_tool_calls": result.n_tool_calls,
                    "n_prompts": result.n_prompts,
                    "error": result.error,
                    "verifier_error": result.verifier_error,
                    "partial_trajectory": result.partial_trajectory,
                    "trajectory_source": result.trajectory_source,
                    "started_at": str(result.started_at),
                    "finished_at": str(result.finished_at),
                    "timing": timing,
                },
                indent=2,
            )
        )
        (trial_dir / "timing.json").write_text(json.dumps(timing, indent=2))
        (trial_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
        return result

    @staticmethod
    def _resolve_prompts(
        task_path: Path, prompts: list[str | None] | None
    ) -> list[str]:
        """Read instruction.md and resolve prompt list."""
        instruction_path = task_path / "instruction.md"
        if not instruction_path.exists():
            raise FileNotFoundError(f"Task missing instruction.md: {task_path}")
        instruction = instruction_path.read_text().strip()
        if prompts is None:
            return [instruction]
        return [p if p is not None else instruction for p in prompts]

    async def _start_env_and_upload(self, env, task_path: Path, timing: dict) -> None:
        """Start environment and upload task files."""
        logger.info(f"Starting environment: {task_path.name}")
        t0 = datetime.now()
        await env.start(force_build=False)
        timing["environment_setup"] = (datetime.now() - t0).total_seconds()
        if (task_path / "instruction.md").exists():
            await env.upload_file(task_path / "instruction.md", "/instruction.md")
        if (task_path / "solution").is_dir():
            await env.upload_dir(task_path / "solution", "/solution")

    async def _run_oracle(
        self, env, task_path: Path, timeout: int, sandbox_user: str | None = None
    ) -> tuple[list[dict], str]:
        """Run oracle mode (solution/solve.sh), return (trajectory, agent_name)."""
        logger.info("Oracle mode: running solution/solve.sh")
        if not (task_path / "solution" / "solve.sh").exists():
            raise FileNotFoundError(f"Oracle requires solution/solve.sh: {task_path}")
        if sandbox_user:
            cmd = (
                f"chmod +x /solution/solve.sh && "
                f"su -s /bin/bash {sandbox_user} -c /solution/solve.sh"
                f" 2>&1 | tee /logs/agent/oracle.txt"
            )
        else:
            cmd = (
                "chmod +x /solution/solve.sh && "
                "/solution/solve.sh 2>&1 | tee /logs/agent/oracle.txt"
            )
        result = await env.exec(cmd, timeout_sec=timeout)
        if result.return_code != 0:
            logger.warning(f"Oracle solve.sh exited with rc={result.return_code}")
        trajectory = [
            {
                "type": "oracle",
                "command": "solution/solve.sh",
                "return_code": result.return_code,
                "stdout": (result.stdout or "")[:_DIAG_TRUNCATE],
            }
        ]
        return trajectory, "oracle"

    async def _verify(
        self,
        env,
        task: "Task",
        trial_paths: "TrialPaths",
        timing: dict,
        sandbox_user: str | None = None,
    ) -> tuple[dict | None, str | None]:
        """Run verifier with pre-verification hardening."""
        trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        await harden_before_verify(env, task, sandbox_user)
        logger.info("Running verifier...")
        t0 = datetime.now()
        verifier_error = None
        try:
            verifier = Verifier(task=task, trial_paths=trial_paths, environment=env)
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=task.config.verifier.timeout_sec,
            )
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            rewards = verifier_result.rewards
            logger.info(f"Rewards: {rewards}")
        except TimeoutError:
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
            verifier_error = (
                f"verifier timed out after {task.config.verifier.timeout_sec}s"
            )
            rewards = None
            logger.error(verifier_error)
        except Exception as e:
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
            verifier_error = f"verifier crashed: {e}"
            rewards = None
            logger.error(verifier_error)
        return rewards, verifier_error

    async def run(
        self,
        task_path: str | Path,
        agent: str = "claude-agent-acp",
        prompts: list[str | None] | None = None,
        *,
        model: str | None = None,
        agent_env: dict[str, str] | None = None,
        job_name: str | None = None,
        trial_name: str | None = None,
        jobs_dir: str | Path = "jobs",
        environment: str = "docker",
        skills_dir: str | Path | None = None,
        sandbox_user: str | None = "agent",
        sandbox_locked_paths: list[str] | None = None,
        pre_agent_hooks: list | None = None,
        context_root: str | Path | None = None,
    ) -> RunResult:
        """Run a task with an ACP agent inside a sandbox.

        Args:
            task_path: Path to Harbor-format task directory
            agent: ACP agent name or command (e.g. "claude-agent-acp", "openclaw")
            prompts: List of prompts to send. Default: [instruction.md content]
            model: Model to use (e.g. "claude-haiku-4-5-20251001"). Set via ACP session/set_model.
            agent_env: Environment variables for the agent (API keys etc.)
            job_name: Job name. Auto-generated if not provided.
            trial_name: Custom trial name. Auto-generated if not provided.
            jobs_dir: Directory for job output (Harbor convention).
            environment: Environment type — "docker" or "daytona".
            skills_dir: Path to skills directory. Copied into sandbox and symlinked
                to agent-specific discovery paths (e.g. ~/.claude/skills/).
            sandbox_user: Run agent as this non-root user (e.g. "agent"). Uses
                setpriv (Debian/Ubuntu) or su (Alpine/others) — no external
                dependencies. Setup (install) and verification run as root.
            pre_agent_hooks: List of async callables(env) to run after setup but
                before agent launch. Use for starting background services, etc.
            context_root: Repo root for resolving Dockerfile COPY paths. When set,
                scans environment/Dockerfile for COPY sources relative to this root,
                copies them into environment/_deps/, and rewrites the Dockerfile.

        Returns:
            RunResult with rewards, trajectory, and metadata.
        """
        if sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown. "
                "Root can read solution/test files. "
                "Set sandbox_user='agent' for answer integrity."
            )

        # Resolve effective locked paths
        effective_locked = _resolve_locked_paths(sandbox_user, sandbox_locked_paths)

        task_path = Path(task_path)
        task, trial_dir, trial_paths, started_at, job_name, trial_name = (
            self._init_trial(
                task_path,
                job_name,
                trial_name,
                jobs_dir,
            )
        )
        agent_env = resolve_agent_env(agent, model, agent_env)
        # Use a new local so the type narrows from `list[str | None] | None`
        # (the public API allows None entries to mean "use default") to
        # `list[str]` after _resolve_prompts has substituted them.
        resolved_prompts: list[str] = self._resolve_prompts(task_path, prompts)
        agent_launch = AGENT_LAUNCH.get(agent, agent)

        if context_root:
            stage_dockerfile_deps(task_path, Path(context_root))
        if skills_dir:
            _inject_skills_into_dockerfile(task_path, Path(skills_dir))

        env = _create_environment(environment, task, task_path, trial_name, trial_paths)
        # Harbor returns timeout as int | float | None; SDK helpers expect int.
        timeout = int(task.config.agent.timeout_sec or 0)
        timing: dict[str, float] = {}

        self._write_config(
            trial_dir,
            task_path=task_path,
            agent=agent,
            model=model,
            environment=environment,
            skills_dir=skills_dir,
            sandbox_user=sandbox_user,
            context_root=context_root,
            sandbox_locked_paths=effective_locked,
            timeout=timeout,
            started_at=started_at,
            agent_env=agent_env,
        )

        acp_client: ACPClient | None = None
        trajectory: list[dict] = []
        partial_trajectory = False
        trajectory_source: TrajectorySource | None = None
        agent_name = ""
        n_tool_calls = 0
        error = None
        verifier_error = None
        rewards = None

        try:
            await self._start_env_and_upload(env, task_path, timing)
            t_agent_setup = datetime.now()
            t_agent_exec = t_agent_setup

            for hook in pre_agent_hooks or []:
                await hook(env)

            if agent == "oracle":
                if sandbox_user:
                    await setup_sandbox_user(env, sandbox_user, workspace="/app")
                await lockdown_paths(env, effective_locked)
                trajectory, agent_name = await self._run_oracle(
                    env, task_path, timeout, sandbox_user
                )
            else:
                agent_cfg = await install_agent(env, agent, trial_dir)
                cred_home = f"/home/{sandbox_user}" if sandbox_user else "/root"
                await write_credential_files(
                    env,
                    agent,
                    agent_env,
                    agent_cfg,
                    model,
                    cred_home,
                )
                if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                    await upload_subscription_auth(env, agent, cred_home)

                # Detect working directory (preserved when sandbox user is set)
                cwd_result = await env.exec("pwd", timeout_sec=10)
                agent_cwd = (cwd_result.stdout or "").strip() or "/app"
                if sandbox_user:
                    agent_cwd = await setup_sandbox_user(
                        env, sandbox_user, workspace=agent_cwd
                    )

                await deploy_skills(
                    env,
                    task_path,
                    skills_dir,
                    agent_cfg,
                    sandbox_user,
                    agent_cwd,
                    task,
                )

                await lockdown_paths(env, effective_locked)

                acp_client, session, agent_name = await connect_acp(
                    env,
                    agent,
                    agent_launch,
                    agent_env,
                    sandbox_user,
                    model,
                    trial_dir,
                    environment,
                    agent_cwd,
                )
                timing["agent_setup"] = (datetime.now() - t_agent_setup).total_seconds()
                t_agent_exec = datetime.now()

                trajectory, n_tool_calls = await execute_prompts(
                    acp_client,
                    session,
                    resolved_prompts,
                    timeout,
                )
                trajectory_source = "acp"

            if agent != "oracle" and "agent_setup" not in timing:
                timing["agent_setup"] = (datetime.now() - t_agent_setup).total_seconds()
            if agent == "oracle":
                timing["agent_execution"] = (
                    datetime.now() - t_agent_setup
                ).total_seconds()
            elif "agent_execution" not in timing:
                timing["agent_execution"] = (
                    datetime.now() - t_agent_exec
                ).total_seconds()

            # Fallback: scrape agent-native trajectory if ACP captured nothing
            if not trajectory and agent != "oracle":
                scraped = await _scrape_agent_trajectory(env, agent, sandbox_user)
                if scraped:
                    trajectory = scraped
                    trajectory_source = "scraped"
                    # Do NOT overwrite n_tool_calls — keep ACP-sourced value (trusted).
                    # Scraped trajectory is agent-writable and forgeable.
                    logger.warning(
                        f"Using scraped trajectory ({len(scraped)} events) from "
                        f"agent-writable directory — data is UNTRUSTED"
                    )

            rewards, verifier_error = await self._verify(
                env, task, trial_paths, timing, sandbox_user=sandbox_user
            )

        except TimeoutError:
            error = f"Agent timed out after {timeout}s"
            logger.error(error)
        except ConnectionError as e:
            error = str(e)
            logger.error(f"Agent connection lost: {error}")
        except ACPError as e:
            if "Invalid API key" in e.message:
                from benchflow._agent_env import check_subscription_auth
                from benchflow.agents.registry import infer_env_key_for_model

                key = infer_env_key_for_model(model) if model else None
                if key and check_subscription_auth(agent, key):
                    error = (
                        f"{key} was rejected as invalid. "
                        f"Subscription auth credentials exist — unset the env var "
                        f"to use them: env -u {key} <command>"
                    )
                else:
                    error = str(e)
            else:
                error = str(e)
            logger.error(error)
        except Exception as e:
            error = str(e)
            logger.error("Run failed", exc_info=True)

        finally:
            if not trajectory and acp_client and acp_client.session is not None:
                try:
                    trajectory = _capture_session_trajectory(acp_client.session)
                    if trajectory:
                        partial_trajectory = True
                        trajectory_source = "partial_acp"
                        n_tool_calls = len(acp_client.session.tool_calls)
                        logger.info(
                            f"Captured {len(trajectory)} partial trajectory events"
                        )
                except Exception as e:
                    logger.warning(f"Partial trajectory capture failed: {e}")

            if acp_client:
                try:
                    await acp_client.close()
                except Exception as e:
                    logger.warning(f"ACP client close failed: {e}")
            try:
                await env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        return self._build_result(
            trial_dir,
            task_name=task_path.name,
            trial_name=trial_name,
            agent=agent,
            agent_name=agent_name,
            model=model or "",
            n_tool_calls=n_tool_calls,
            prompts=resolved_prompts,
            error=error,
            verifier_error=verifier_error,
            trajectory=trajectory,
            partial_trajectory=partial_trajectory,
            trajectory_source=trajectory_source,
            rewards=rewards,
            started_at=started_at,
            timing=timing,
        )
