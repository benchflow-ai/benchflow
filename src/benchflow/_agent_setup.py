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


def _skill_link_cmd(source: str, dest: str, sandbox_user: str | None) -> str:
    """Link a shared skills tree into an agent discovery path.

    When sandbox_user is set, mkdir runs as root but the resulting parent
    directory is chowned so the agent (running as sandbox_user) can later
    write into it. Guards against PR #208 / issue #7 — root-owned `.pi/agent`
    blocking pi-acp's models.json write.
    """
    if source == dest:
        q_dest = shlex.quote(dest)
        if sandbox_user:
            q_user = shlex.quote(sandbox_user)
            return f"mkdir -p {q_dest} && chown -R {q_user}:{q_user} {q_dest}"
        return f"mkdir -p {q_dest}"

    parent = shlex.quote(str(Path(dest).parent))
    q_source = shlex.quote(source)
    q_dest = shlex.quote(dest)
    chown = ""
    if sandbox_user:
        q_user = shlex.quote(sandbox_user)
        chown = f"chown -R {q_user}:{q_user} {parent} && "
    return (
        f"mkdir -p {parent} && {chown}rm -rf {q_dest} && ln -sfn {q_source} {q_dest}"
    )


async def _link_skill_paths(
    env,
    source: str,
    skill_paths: list[str],
    home: str,
    cwd: str,
    sandbox_user: str | None,
) -> int:
    """Link one shared skills tree into each configured discovery path."""
    parts = []
    for sp in skill_paths:
        expanded = sp.replace("$HOME", home).replace("$WORKSPACE", cwd)
        parts.append(_skill_link_cmd(source, expanded, sandbox_user))
    if parts:
        cmd = " && ".join(parts)
        result = await env.exec(cmd, timeout_sec=15)
        if result.return_code != 0:
            stdout = (getattr(result, "stdout", "") or "").strip()
            stderr = (getattr(result, "stderr", "") or "").strip()
            details = [
                f"exit code {result.return_code}",
                f"command: {cmd}",
            ]
            if stdout:
                details.append(f"stdout: {stdout}")
            if stderr:
                details.append(f"stderr: {stderr}")
            raise RuntimeError(
                f"Failed to link skills from {source}: {'; '.join(details)}"
            )
    return len(parts)


async def install_agent(env, agent: str, trial_dir: Path) -> AgentConfig | None:
    """Install agent in sandbox and return its config."""
    agent_base = agent.split()[0]
    agent_cfg = AGENTS.get(agent_base)
    if agent_base not in AGENT_INSTALLERS:
        return agent_cfg
    install_cmd = AGENT_INSTALLERS[agent_base]
    install_timeout = agent_cfg.install_timeout if agent_cfg else 900
    logger.info(f"Installing {agent_base} in sandbox (timeout={install_timeout}s)...")
    install_result = await env.exec(
        install_cmd,
        timeout_sec=install_timeout,
    )
    install_log = trial_dir / "agent" / "install-stdout.txt"
    install_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = install_result.stdout or ""
    stderr = install_result.stderr or ""
    parts = [f"$ {install_cmd}\n"]
    if stdout:
        parts.append("=== stdout ===\n")
        parts.append(stdout)
        if not stdout.endswith("\n"):
            parts.append("\n")
    if stderr:
        parts.append("=== stderr ===\n")
        parts.append(stderr)
        if not stderr.endswith("\n"):
            parts.append("\n")
    install_log.write_text("".join(parts))
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
            stdout="".join(parts),
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
    task_skills_dir = task.config.environment.skills_dir
    effective_skills = task_skills_dir

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
                logger.info("Skills deployed to /skills")
                effective_skills = "/skills"
            else:
                logger.warning(f"Skills dir not found: {skills_path}")
        else:
            logger.info("Skills already injected via Dockerfile")

    # Distribute to agent-specific discovery paths
    if effective_skills and agent_cfg and agent_cfg.skill_paths:
        home = f"/home/{sandbox_user}" if sandbox_user else "/root"
        count = await _link_skill_paths(
            env,
            effective_skills,
            agent_cfg.skill_paths,
            home,
            agent_cwd,
            sandbox_user,
        )
        if count:
            logger.info(f"Skills distributed to {count} paths for {agent_cfg.name}")
