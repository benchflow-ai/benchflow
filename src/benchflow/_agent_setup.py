"""Agent provisioning inside the sandbox: install + skill deployment.

Owns the "prepare the sandbox for the agent" phase that runs once,
sequentially, before ACP connection:

    install_agent  → registry-driven install_cmd, captures stdout, raises
                     AgentInstallError on non-zero return code
    deploy_skills  → runtime skill upload (Dockerfile-mount fallback) and
                     distribution into agent-specific discovery paths

Together they form the install → distribute lifecycle that the SDK loop
runs as: install_agent → deploy_skills → connect_acp → execute_prompts.

Does not own:
    - Agent / provider env vars — see _agent_env.py
    - Credential file writing — see _credentials.py
"""

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow.agents.registry import AGENT_INSTALLERS, AGENTS, AgentConfig
from benchflow.models import AgentInstallError

if TYPE_CHECKING:
    from harbor.models.task.task import Task

logger = logging.getLogger(__name__)


async def install_agent(env, agent: str, trial_dir: Path) -> AgentConfig | None:
    """Install agent in sandbox and return its config."""
    agent_base = agent.split()[0]
    agent_cfg = AGENTS.get(agent_base)
    if agent_base not in AGENT_INSTALLERS:
        return agent_cfg
    install_timeout = agent_cfg.install_timeout if agent_cfg else 900
    logger.info(f"Installing {agent_base} in sandbox (timeout={install_timeout}s)...")
    install_result = await env.exec(
        AGENT_INSTALLERS[agent_base],
        timeout_sec=install_timeout,
    )
    install_log = trial_dir / "agent" / "install-stdout.txt"
    install_log.parent.mkdir(parents=True, exist_ok=True)
    install_log.write_text(install_result.stdout or "")
    if install_result.return_code != 0:
        diag = await env.exec(
            "echo 'OS:' && cat /etc/os-release 2>/dev/null | head -2; "
            "echo 'Node:' && node --version 2>&1; "
            f"echo 'Agent:' && which {agent_base} 2>&1",
            timeout_sec=10,
        )
        raise AgentInstallError(
            agent=agent_base,
            return_code=install_result.return_code,
            stdout=install_result.stdout or "",
            diagnostics=diag.stdout or "",
            log_path=str(install_log),
        )
    return agent_cfg


async def deploy_skills(
    env,
    task_path: Path,
    skills_dir: str | Path | None,
    agent_cfg,
    sandbox_user: str | None,
    agent_cwd: str,
    task: "Task",
) -> None:
    """Deploy and distribute skills into sandbox."""
    # Runtime upload (fallback if not baked into Dockerfile)
    if skills_dir:
        dockerfile = task_path / "environment" / "Dockerfile"
        already_injected = (
            dockerfile.exists()
            and "COPY _deps/skills /skills/" in dockerfile.read_text()
        )
        if not already_injected:
            skills_path = Path(skills_dir)
            if skills_path.is_dir():
                logger.info(f"Deploying skills via runtime upload from {skills_path}")
                await env.upload_dir(skills_path, "/skills")
                if agent_cfg and agent_cfg.skill_paths:
                    parts = []
                    for sp in agent_cfg.skill_paths:
                        expanded = sp.replace("$HOME", "/root").replace(
                            "$WORKSPACE", "/app"
                        )
                        parent = str(Path(expanded).parent)
                        parts.append(
                            f"mkdir -p '{parent}' && ln -sf /skills '{expanded}'"
                        )
                    await env.exec(" && ".join(parts), timeout_sec=10)
                logger.info("Skills deployed to /skills and symlinked")
            else:
                logger.warning(f"Skills dir not found: {skills_path}")
        else:
            logger.info("Skills already injected via Dockerfile")

    # Distribute to agent-specific discovery paths
    task_skills_dir = task.config.environment.skills_dir
    effective_skills = "/skills" if skills_dir else task_skills_dir
    if effective_skills and agent_cfg and agent_cfg.skill_paths:
        home = f"/home/{sandbox_user}" if sandbox_user else "/root"
        parts = []
        for sp in agent_cfg.skill_paths:
            expanded = sp.replace("$HOME", home).replace("$WORKSPACE", agent_cwd)
            q_expanded = shlex.quote(expanded)
            q_skills = shlex.quote(effective_skills)
            parts.append(
                f"mkdir -p {q_expanded} && cp -r {q_skills}/. {q_expanded}/ 2>/dev/null"
            )
        if parts:
            await env.exec("; ".join(parts), timeout_sec=15)
            logger.info(
                f"Skills distributed to {len(parts)} paths for {agent_cfg.name}"
            )
