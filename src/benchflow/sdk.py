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
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.agents.registry import AGENTS, get_agent
from benchflow.process import DockerProcess, DaytonaProcess, LiveProcess

logger = logging.getLogger(__name__)

# Directories to ignore when copying deps
_IGNORE_DIRS = {".venv", "__pycache__", ".pytest_cache", "node_modules", ".git", ".mypy_cache", ".ruff_cache"}


def _dep_local_name(src_path: str) -> str:
    """Compute a short unique local name for a dependency path.

    packages/environments/claw-gmail  -> claw-gmail
    tasks/email-foo/environment/skills -> skills
    tasks/email-foo/data              -> email-foo__data
    """
    parts = Path(src_path).parts
    if len(parts) == 1:
        return parts[0]
    basename = parts[-1]
    if basename in ("data", "config", "src", "lib", "skills", "environment"):
        return f"{parts[-2]}__{basename}"
    return basename


def stage_dockerfile_deps(
    task_path: Path,
    context_root: Path,
) -> None:
    """Copy Dockerfile COPY sources into environment/_deps/ and rewrite paths.

    When a Dockerfile references files relative to the repo root (e.g.
    `COPY packages/environments/claw-gmail /app`), the Docker build context
    (set to environment/) won't find them. This function:

    1. Scans the Dockerfile for COPY instructions
    2. Copies each source from context_root into environment/_deps/
    3. Rewrites the COPY instruction to use the local _deps/ path

    Args:
        task_path: Path to the task directory (contains environment/Dockerfile)
        context_root: Path to the repo root where COPY sources are relative to
    """
    env_dir = task_path / "environment"
    dockerfile_path = env_dir / "Dockerfile"
    if not dockerfile_path.exists():
        return

    content = dockerfile_path.read_text()
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        copy_match = re.match(
            r"^(\s*COPY\s+(?:--\S+\s+)*)(\S+)\s+(\S+)\s*$", line
        )
        if copy_match:
            prefix = copy_match.group(1)
            src_path = copy_match.group(2)
            dst_path = copy_match.group(3)

            # Skip sources already relative to env dir, absolute, or using build args
            if src_path.startswith("/") or src_path.startswith("$") or src_path == ".":
                new_lines.append(line)
                continue

            abs_src = context_root / src_path
            if abs_src.exists():
                dep_name = _dep_local_name(src_path)
                local_dest = env_dir / "_deps" / dep_name

                if abs_src.is_dir():
                    if local_dest.exists():
                        shutil.rmtree(local_dest)
                    shutil.copytree(abs_src, local_dest, ignore=shutil.ignore_patterns(*_IGNORE_DIRS))
                else:
                    local_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(abs_src, local_dest)

                new_lines.append(f"{prefix}_deps/{dep_name} {dst_path}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    dockerfile_path.write_text("\n".join(new_lines))


_AGENT_SKILL_PATHS = [
    "/root/.gemini/skills",
    "/root/.claude/skills",
]


def _inject_skills_into_dockerfile(task_path: Path, skills_dir: Path) -> None:
    """Inject skills into the task's Dockerfile (baked into image).

    Copies skills_dir into environment/_deps/skills/ and appends COPY + symlink
    lines to the Dockerfile. This is more reliable than runtime upload since
    skills are part of the image.
    """
    env_dir = task_path / "environment"
    dockerfile_path = env_dir / "Dockerfile"
    if not dockerfile_path.exists() or not skills_dir.is_dir():
        return

    dest = env_dir / "_deps" / "skills"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(skills_dir, dest, ignore=shutil.ignore_patterns(*_IGNORE_DIRS))

    lines = [
        "",
        "# Skills directory (injected by benchflow --skills-dir)",
        "COPY _deps/skills /skills/",
    ]
    for agent_path in _AGENT_SKILL_PATHS:
        parent = str(Path(agent_path).parent)
        lines.append(f"RUN mkdir -p {parent} && ln -sf /skills {agent_path}")

    content = dockerfile_path.read_text()
    dockerfile_path.write_text(content + "\n".join(lines) + "\n")
    logger.info(f"Skills injected into Dockerfile: {len(list(skills_dir.iterdir()))} items")


def _detect_dind_mount() -> tuple[str, str] | None:
    """Detect Docker-in-Docker host path translation.

    When running inside a devcontainer that shares the host Docker socket,
    bind mount paths must be translated from container paths to host paths.

    Returns (host_source, container_dest) tuple, or None if not in DinD.
    """
    if not Path("/.dockerenv").exists():
        return None
    import subprocess as _sp
    try:
        hostname = _sp.check_output(["hostname"], text=True).strip()
        result = _sp.run(
            ["docker", "inspect", hostname],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        cwd = str(Path.cwd())
        best = None
        for mount in data[0].get("Mounts", []):
            if mount.get("Type") != "bind":
                continue
            dest = mount.get("Destination", "")
            if cwd.startswith(dest) and (best is None or len(dest) > len(best[1])):
                best = (mount["Source"], dest)
        return best
    except Exception:
        return None


def _patch_harbor_dind() -> None:
    """Monkey-patch Harbor's DockerEnvironmentEnvVars for DinD path translation.

    When running inside a devcontainer, HOST_*_PATH env vars need to use
    host filesystem paths, not container paths. Applied once at import time.
    """
    dind_mount = _detect_dind_mount()
    if not dind_mount:
        return

    host_source, container_dest = dind_mount
    logger.info(f"DinD detected: {container_dest} → {host_source}")

    try:
        from harbor.environments.docker.docker import DockerEnvironmentEnvVars
    except ImportError:
        return

    _original = DockerEnvironmentEnvVars.to_env_dict

    def _patched(self, include_os_env=True):
        env = _original(self, include_os_env=include_os_env)
        for key in ("HOST_VERIFIER_LOGS_PATH", "HOST_AGENT_LOGS_PATH", "HOST_ARTIFACTS_PATH"):
            val = env.get(key, "")
            if val.startswith(container_dest):
                env[key] = host_source + val[len(container_dest):]
        return env

    DockerEnvironmentEnvVars.to_env_dict = _patched


# Apply DinD patch once at import time
_patch_harbor_dind()


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



class RunResult:
    """Result of a benchflow run."""

    def __init__(
        self,
        task_name: str,
        trial_name: str = "",
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
        prompts: list[str | None] | None = None,
        *,
        model: str | None = None,
        agent_env: dict[str, str] | None = None,
        job_name: str | None = None,
        trial_name: str | None = None,
        jobs_dir: str | Path = "jobs",
        environment: str = "docker",
        skills_dir: str | Path | None = None,
        sandbox_user: str | None = None,
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
            sandbox_user: Run agent as this non-root user (e.g. "agent"). Requires
                gosu in the container. Setup (install) and verification run as root.
            pre_agent_hooks: List of async callables(env) to run after setup but
                before agent launch. Use for starting background services, etc.
            context_root: Repo root for resolving Dockerfile COPY paths. When set,
                scans environment/Dockerfile for COPY sources relative to this root,
                copies them into environment/_deps/, and rewrites the Dockerfile.

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

        # Pre-create trial directory tree so Docker doesn't create them as root.
        # Harbor's DockerEnvironment bind-mounts these subdirs into the container;
        # if they don't exist, Docker creates them owned by root, causing
        # PermissionError when SDK.run() writes artifacts after env.stop().
        trial_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (trial_dir / subdir).mkdir(exist_ok=True)

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

        # Stage Dockerfile deps if context_root is provided
        if context_root:
            stage_dockerfile_deps(task_path, Path(context_root))

        # Inject skills into Dockerfile (bakes into image, more reliable than runtime upload)
        if skills_dir:
            _inject_skills_into_dockerfile(task_path, Path(skills_dir))

        # Create Harbor environment
        env = _create_environment(environment, task, task_path, trial_name, trial_paths)

        acp_client: ACPClient | None = None
        trajectory: list[dict] = []
        agent_name = ""
        n_tool_calls = 0
        error = None
        rewards = None
        timeout = task.config.agent.timeout_sec  # Define before try for except block

        try:
            # Default prompts: task instruction
            instruction_path = task_path / "instruction.md"
            if not instruction_path.exists():
                raise FileNotFoundError(f"Task missing instruction.md: {task_path}")
            instruction = instruction_path.read_text().strip()
            if prompts is None:
                prompts = [instruction]
            else:
                # Replace None entries with instruction
                prompts = [p if p is not None else instruction for p in prompts]

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

            # 2b. Deploy skills into sandbox (runtime fallback if no Dockerfile injection)
            if skills_dir:
                # Check if Dockerfile injection already handled it
                env_dir = task_path / "environment"
                dockerfile = env_dir / "Dockerfile"
                already_injected = (
                    dockerfile.exists()
                    and "COPY _deps/skills /skills/" in dockerfile.read_text()
                )
                if not already_injected:
                    skills_path = Path(skills_dir)
                    if skills_path.is_dir():
                        logger.info(f"Deploying skills via runtime upload from {skills_path}")
                        await env.upload_dir(skills_path, "/skills")
                        await env.exec(
                            "mkdir -p /root/.claude /root/.gemini && "
                            "ln -sf /skills /root/.claude/skills && "
                            "ln -sf /skills /root/.gemini/skills",
                            timeout_sec=10,
                        )
                        logger.info("Skills deployed to /skills and symlinked")
                    else:
                        logger.warning(f"Skills dir not found: {skills_path}")
                else:
                    logger.info("Skills already injected via Dockerfile")

            # 2c. Run pre-agent hooks (e.g. start background services)
            for hook in (pre_agent_hooks or []):
                await hook(env)

            # 2d. Set up sandbox user (non-root agent execution)
            if sandbox_user:
                logger.info(f"Setting up sandbox user: {sandbox_user}")
                await env.exec(
                    f"id -u {sandbox_user} >/dev/null 2>&1 || "
                    f"useradd -m -s /bin/bash {sandbox_user} && "
                    f"mkdir -p /home/{sandbox_user}/.local/bin "
                    f"/home/{sandbox_user}/.claude /home/{sandbox_user}/.gemini && "
                    # Copy agent binaries to user home
                    "if [ -d /root/.local/bin ]; then "
                    f"cp -aL /root/.local/bin/. /home/{sandbox_user}/.local/bin/ 2>/dev/null || true; fi && "
                    # Copy nvm if present
                    "if [ -d /root/.nvm ]; then "
                    f"cp -a /root/.nvm/. /home/{sandbox_user}/.nvm/ 2>/dev/null || true; fi && "
                    # Symlink skills into user home
                    "if [ -d /skills ]; then "
                    f"chmod -R a+rX /skills && "
                    f"ln -sf /skills /home/{sandbox_user}/.claude/skills && "
                    f"ln -sf /skills /home/{sandbox_user}/.gemini/skills; fi && "
                    f"chown -R {sandbox_user}:{sandbox_user} /home/{sandbox_user}",
                    timeout_sec=30,
                )
                logger.info(f"Sandbox user {sandbox_user} ready")

            # Detect sandbox working directory (from Dockerfile WORKDIR)
            cwd_result = await env.exec("pwd", timeout_sec=10)
            agent_cwd = cwd_result.stdout.strip() if cwd_result.return_code == 0 else "/app"
            if sandbox_user:
                agent_cwd = f"/home/{sandbox_user}"
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

            # Wrap agent command with gosu for sandbox user
            if sandbox_user:
                import shlex
                inner = f"export HOME=/home/{sandbox_user} && cd /home/{sandbox_user} && {agent_launch}"
                agent_launch = f"gosu {sandbox_user} bash -c {shlex.quote(inner)}"
                logger.info(f"Agent sandboxed as: {sandbox_user}")

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

            init_result = await asyncio.wait_for(
                acp_client.initialize(), timeout=60,
            )
            agent_name = (
                init_result.agent_info.name if init_result.agent_info else agent
            )
            logger.info(f"ACP agent: {agent_name}")

            session = await asyncio.wait_for(
                acp_client.session_new(cwd=agent_cwd), timeout=60,
            )
            logger.info(f"Session: {session.session_id}")

            # Set model via ACP (env var ANTHROPIC_MODEL is ignored by claude-agent-acp)
            if model:
                try:
                    await acp_client.set_model(model)
                    logger.info(f"Model set to: {model}")
                except Exception as e:
                    logger.warning(f"Failed to set model via ACP: {e}")

            # 4. Send prompts (multi-turn)
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
            logger.error("Run failed", exc_info=True)

        finally:
            if acp_client:
                try:
                    await acp_client.close()
                except Exception as e:
                    logger.warning(f"ACP client close failed: {e}")
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
