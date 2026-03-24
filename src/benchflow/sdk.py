"""benchflow SDK — unified run() that uses ACP inside Harbor environments.

One execution path:
1. Start Harbor environment (Docker or Daytona)
2. Install ACP agent in sandbox
3. Connect via live stdio pipe (ContainerTransport)
4. ACP: initialize → session/new → session/prompt (multi-turn)
5. Capture trajectory from session/update notifications
6. Run Harbor verifier
7. Stop environment
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.agents.registry import AGENTS, get_agent
from benchflow.process import DockerProcess, DaytonaProcess, LiveProcess

logger = logging.getLogger(__name__)

# Backwards compat — expose install/launch dicts from registry
AGENT_INSTALLERS = {name: a.install_cmd for name, a in AGENTS.items()}
AGENT_LAUNCH = {name: a.launch_cmd for name, a in AGENTS.items()}


def _create_environment(
    environment_type: str,
    task: Task,
    task_path: Path,
    trial_name: str,
    trial_paths: TrialPaths,
) -> Any:
    """Create a Harbor environment (Docker or Daytona)."""
    if environment_type == "docker":
        from harbor.environments.docker.docker import DockerEnvironment
        return DockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
        )
    elif environment_type == "daytona":
        from harbor.environments.daytona import DaytonaEnvironment
        return DaytonaEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
        )
    else:
        raise ValueError(f"Unknown environment_type: {environment_type!r} (use 'docker' or 'daytona')")


def _create_live_process(environment_type: str, env: Any) -> LiveProcess:
    """Create the appropriate LiveProcess for ACP communication."""
    if environment_type == "docker":
        return DockerProcess.from_harbor_env(env)
    elif environment_type == "daytona":
        # DaytonaProcess.from_harbor_env is async, handle in caller
        raise ValueError("Use await DaytonaProcess.from_harbor_env(env) directly")
    else:
        raise ValueError(f"Unknown environment_type: {environment_type!r}")


class RunResult:
    """Result of a benchflow run."""

    def __init__(
        self,
        task_name: str,
        trial_name: str,
        rewards: dict[str, float | int] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        agent_name: str = "",
        n_tool_calls: int = 0,
        n_prompts: int = 0,
        error: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ):
        self.task_name = task_name
        self.trial_name = trial_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent_name = agent_name
        self.n_tool_calls = n_tool_calls
        self.n_prompts = n_prompts
        self.error = error
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return (
            f"RunResult(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )


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

    async def run(
        self,
        task_path: str | Path,
        agent: str = "claude-agent-acp",
        prompts: list[str] | None = None,
        *,
        model: str | None = None,
        agent_env: dict[str, str] | None = None,
        job_name: str | None = None,
        trial_name: str | None = None,
        jobs_dir: str | Path = "jobs",
        environment: str = "docker",
    ) -> RunResult:
        """Run a task with an ACP agent inside a sandbox.

        Args:
            task_path: Path to Harbor-format task directory
            agent: ACP agent name or command (e.g. "claude-agent-acp", "openclaw")
            prompts: List of prompts to send. Default: [instruction.md content]
            model: Model to use (e.g. "claude-haiku-4-5-20251001"). Passed as ANTHROPIC_MODEL.
            agent_env: Environment variables for the agent (API keys etc.)
            job_name: Job name. Auto-generated if not provided.
            trial_name: Custom trial name. Auto-generated if not provided.
            jobs_dir: Directory for job output (Harbor convention).
            environment: Environment type — "docker" or "daytona".

        Returns:
            RunResult with rewards, trajectory, and metadata.
        """
        from uuid import uuid4

        task_path = Path(task_path)
        task = Task(task_path)
        job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        trial_name = trial_name or f"{task_path.name}__{uuid4().hex[:8]}"
        job_dir = Path(jobs_dir) / job_name
        trial_dir = job_dir / trial_name
        trial_paths = TrialPaths(trial_dir)
        started_at = datetime.now()

        # Resolve agent env — auto-inherit API keys from os.environ
        agent_env = dict(agent_env or {})
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            if key in os.environ:
                agent_env.setdefault(key, os.environ[key])
        if model:
            agent_env.setdefault("ANTHROPIC_MODEL", model)
        # Increase output token limit to avoid truncation errors
        agent_env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000")
        # Disable telemetry/non-essential traffic in container
        agent_env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")

        # Resolve agent launch command
        agent_launch = AGENT_LAUNCH.get(agent, agent)

        # Default prompts: task instruction
        instruction = (task_path / "instruction.md").read_text().strip()
        if prompts is None:
            prompts = [instruction]
        else:
            # Replace None entries with instruction
            prompts = [p if p is not None else instruction for p in prompts]

        # Create Harbor environment
        env = _create_environment(environment, task, task_path, trial_name, trial_paths)

        acp_client: ACPClient | None = None
        trajectory: list[dict] = []
        agent_name = ""
        n_tool_calls = 0
        error = None
        rewards = None

        try:
            # 1. Start environment
            logger.info(f"Starting {environment} environment: {task_path.name}")
            await env.start(force_build=False)

            # Upload task files
            if (task_path / "instruction.md").exists():
                await env.upload_file(task_path / "instruction.md", "/instruction.md")
            if (task_path / "solution").is_dir():
                await env.upload_dir(task_path / "solution", "/solution")

            # 2. Install agent in sandbox
            agent_base = agent.split()[0]
            if agent_base in AGENT_INSTALLERS:
                logger.info(f"Installing {agent_base} in sandbox...")
                install_result = await env.exec(
                    AGENT_INSTALLERS[agent_base],
                    timeout_sec=900,
                )
                if install_result.return_code != 0:
                    diag = await env.exec(
                        "echo 'OS:' && cat /etc/os-release 2>/dev/null | head -2; "
                        "echo 'Node:' && node --version 2>&1; "
                        f"echo 'Agent:' && which {agent_base} 2>&1",
                        timeout_sec=10,
                    )
                    raise RuntimeError(
                        f"Agent install failed (rc={install_result.return_code}): "
                        f"{install_result.stdout}\n"
                        f"Diagnostics: {diag.stdout}"
                    )

            # Detect sandbox working directory (from Dockerfile WORKDIR)
            cwd_result = await env.exec("pwd", timeout_sec=10)
            agent_cwd = cwd_result.stdout.strip() if cwd_result.return_code == 0 else "/app"
            logger.info(f"Agent cwd: {agent_cwd}")

            # 3. Connect ACP via live stdio pipe
            # For non-Docker envs, resolve full path to agent binary
            # since SSH sessions may have different PATH
            if environment != "docker":
                which_result = await env.exec(f"which {agent_launch.split()[0]}", timeout_sec=10)
                if which_result.return_code == 0 and which_result.stdout.strip():
                    full_path = which_result.stdout.strip()
                    parts = agent_launch.split()
                    parts[0] = full_path
                    agent_launch = " ".join(parts)
                    logger.info(f"Resolved agent path: {agent_launch}")

            if environment == "docker":
                live_proc = DockerProcess.from_harbor_env(env)
            else:
                live_proc = await DaytonaProcess.from_harbor_env(env)

            transport = ContainerTransport(
                container_process=live_proc,
                command=agent_launch,
                env=agent_env,
                cwd=agent_cwd,
            )
            acp_client = ACPClient(transport)
            await acp_client.connect()

            init_result = await acp_client.initialize()
            agent_name = (
                init_result.agent_info.name if init_result.agent_info else agent
            )
            logger.info(f"ACP agent: {agent_name}")

            session = await acp_client.session_new(cwd=agent_cwd)
            logger.info(f"Session: {session.session_id}")

            # Set model via ACP (env var ANTHROPIC_MODEL is ignored by claude-agent-acp)
            if model:
                try:
                    await acp_client.set_model(model)
                    logger.info(f"Model set to: {model}")
                except Exception as e:
                    logger.warning(f"Failed to set model via ACP: {e}")

            # 4. Send prompts (multi-turn)
            timeout = task.config.agent.timeout_sec
            for i, prompt in enumerate(prompts):
                logger.info(f"Prompt {i + 1}/{len(prompts)}: {prompt[:80]}...")
                prompt_result = await asyncio.wait_for(
                    acp_client.prompt(prompt),
                    timeout=timeout,
                )
                logger.info(
                    f"  → {prompt_result.stop_reason.value}, "
                    f"{len(session.tool_calls)} total tool calls"
                )

            n_tool_calls = len(session.tool_calls)

            # 5. Capture trajectory
            for tc in session.tool_calls:
                trajectory.append(
                    {
                        "type": "tool_call",
                        "tool_call_id": tc.tool_call_id,
                        "kind": tc.kind,
                        "title": tc.title,
                        "status": tc.status.value,
                        "content": tc.content,
                    }
                )
            if session.full_message:
                trajectory.append(
                    {
                        "type": "agent_message",
                        "text": session.full_message,
                    }
                )
            if session.full_thought:
                trajectory.append(
                    {
                        "type": "agent_thought",
                        "text": session.full_thought,
                    }
                )

            # Save trajectory
            traj_dir = trial_dir / "trajectory"
            traj_dir.mkdir(parents=True, exist_ok=True)
            (traj_dir / "acp_trajectory.jsonl").write_text(
                "\n".join(json.dumps(e, default=str) for e in trajectory)
            )

            # 6. Verify
            # Ensure verifier output directory exists locally (Daytona doesn't mount it)
            trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Running verifier...")
            verifier = Verifier(
                task=task,
                trial_paths=trial_paths,
                environment=env,
            )
            verifier_result = await verifier.verify()
            rewards = verifier_result.rewards
            logger.info(f"Rewards: {rewards}")

        except asyncio.TimeoutError:
            error = f"Agent timed out after {timeout}s"
            logger.error(error)
        except ConnectionError as e:
            error = str(e)
            logger.error(f"Agent connection lost: {error}")
        except Exception as e:
            error = str(e)
            logger.error(f"Run failed: {e}")

        finally:
            if acp_client:
                try:
                    await acp_client.close()
                except Exception:
                    pass
            try:
                await env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        result = RunResult(
            task_name=task_path.name,
            trial_name=trial_name,
            rewards=rewards,
            trajectory=trajectory,
            agent_name=agent_name,
            n_tool_calls=n_tool_calls,
            n_prompts=len(prompts),
            error=error,
            started_at=started_at,
            finished_at=datetime.now(),
        )

        # Save result.json and prompts.json
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": result.task_name,
                    "trial_name": result.trial_name,
                    "rewards": result.rewards,
                    "agent_name": result.agent_name,
                    "n_tool_calls": result.n_tool_calls,
                    "n_prompts": result.n_prompts,
                    "error": result.error,
                    "started_at": str(result.started_at),
                    "finished_at": str(result.finished_at),
                },
                indent=2,
            )
        )
        (trial_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

        return result
