"""benchflow SDK — unified run() that uses ACP inside Harbor containers.

One execution path:
1. Start Harbor Docker environment
2. Install ACP agent in container
3. Connect via live stdio pipe (ContainerTransport)
4. ACP: initialize → session/new → session/prompt (multi-turn)
5. Capture trajectory from session/update notifications
6. Run Harbor verifier
7. Stop container
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.container import ContainerProcess

logger = logging.getLogger(__name__)

# Node.js install prefix — shared by all npm-based agents
_NODE_INSTALL = (
    "command -v node >/dev/null 2>&1 || ("
    "  apt-get update -qq >/dev/null 2>&1 && "
    "  apt-get install -y -qq curl >/dev/null 2>&1 && "
    "  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1 && "
    "  apt-get install -y -qq nodejs >/dev/null 2>&1"
    ")"
)

# Agent install commands — must handle bare containers
AGENT_INSTALLERS: dict[str, str] = {
    "claude-agent-acp": (
        f"{_NODE_INSTALL} && "
        "command -v claude-agent-acp >/dev/null 2>&1 || "
        "npm install -g @zed-industries/claude-agent-acp@latest 2>&1 | tail -3"
    ),
    "pi-acp": (
        f"{_NODE_INSTALL} && "
        "command -v pi-acp >/dev/null 2>&1 || "
        "npm install -g pi-acp@latest 2>&1 | tail -3"
    ),
    "openclaw": (
        f"{_NODE_INSTALL} && "
        "command -v openclaw >/dev/null 2>&1 || "
        "npm install -g openclaw@latest 2>&1 | tail -3"
    ),
    "codex-acp": (
        f"{_NODE_INSTALL} && "
        "command -v codex-acp >/dev/null 2>&1 || "
        "npm install -g @zed-industries/codex-acp@latest 2>&1 | tail -3"
    ),
    "gemini": (
        f"{_NODE_INSTALL} && "
        "command -v gemini >/dev/null 2>&1 || "
        "npm install -g @google/gemini-cli@latest 2>&1 | tail -3"
    ),
}

# ACP launch commands — how to start each agent after install
AGENT_LAUNCH: dict[str, str] = {
    "claude-agent-acp": "claude-agent-acp",
    "pi-acp": "pi-acp",
    "openclaw": "openclaw acp",
    "codex-acp": "codex-acp",
    "gemini": "gemini --acp",
}


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
    ) -> RunResult:
        """Run a task with an ACP agent inside a Docker container.

        Args:
            task_path: Path to Harbor-format task directory
            agent: ACP agent name or command (e.g. "claude-agent-acp", "openclaw")
            prompts: List of prompts to send. Default: [instruction.md content]
            model: Model to use (e.g. "claude-haiku-4-5-20251001"). Passed as ANTHROPIC_MODEL.
            agent_env: Environment variables for the agent (API keys etc.)
            job_name: Job name. Auto-generated if not provided.
            trial_name: Custom trial name. Auto-generated if not provided.
            jobs_dir: Directory for job output (Harbor convention).

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

        # Resolve agent env — add model if specified
        agent_env = dict(agent_env or {})
        if model:
            agent_env.setdefault("ANTHROPIC_MODEL", model)

        # Resolve agent launch command
        agent_launch = AGENT_LAUNCH.get(agent, agent)

        # Default prompts: task instruction
        instruction = (task_path / "instruction.md").read_text().strip()
        if prompts is None:
            prompts = [instruction]
        else:
            # Replace None entries with instruction
            prompts = [p if p is not None else instruction for p in prompts]

        # Create Harbor Docker environment
        env = DockerEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task_path.name,
            session_id=trial_name,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
        )

        acp_client: ACPClient | None = None
        trajectory: list[dict] = []
        agent_name = ""
        n_tool_calls = 0
        error = None
        rewards = None

        try:
            # 1. Start container
            logger.info(f"Starting environment: {task_path.name}")
            await env.start(force_build=False)

            # Upload task files
            if (task_path / "instruction.md").exists():
                await env.upload_file(task_path / "instruction.md", "/instruction.md")
            if (task_path / "solution").is_dir():
                await env.upload_dir(task_path / "solution", "/solution")

            # 2. Install agent in container
            agent_base = agent.split()[
                0
            ]  # "claude-agent-acp" from "claude-agent-acp --flag"
            if agent_base in AGENT_INSTALLERS:
                logger.info(f"Installing {agent_base} in container...")
                install_result = await env.exec(
                    AGENT_INSTALLERS[agent_base],
                    timeout_sec=300,
                )
                if install_result.return_code != 0:
                    raise RuntimeError(
                        f"Agent install failed (rc={install_result.return_code}): "
                        f"{install_result.stdout}"
                    )

            # 3. Connect ACP via live container pipe
            cp = ContainerProcess.from_harbor_env(env)
            transport = ContainerTransport(
                container_process=cp,
                command=agent_launch,
                env=agent_env,
                cwd="/app",
            )
            acp_client = ACPClient(transport)
            await acp_client.connect()

            init_result = await acp_client.initialize()
            agent_name = (
                init_result.agent_info.name if init_result.agent_info else agent
            )
            logger.info(f"ACP agent: {agent_name}")

            session = await acp_client.session_new(cwd="/app")
            logger.info(f"Session: {session.session_id}")

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
