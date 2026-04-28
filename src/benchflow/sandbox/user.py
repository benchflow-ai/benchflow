"""Sandbox user creation + privilege-drop wrapper.

Owns the "agent runs as non-root" lifecycle:
    - Creating the sandbox user and copying root's tooling into its home
    - Building the privilege-drop wrapper (setpriv / su) for agent launch

Does not own:
    - Spawning the agent process — see benchflow.agents.run
    - Path lockdown — see benchflow.sandbox.lockdown
"""

from __future__ import annotations

import logging
import re
import shlex

from benchflow.agents.registry import get_sandbox_home_dirs

logger = logging.getLogger(__name__)


def build_priv_drop_cmd(agent_launch: str, sandbox_user: str) -> str:
    """Build a shell command that drops to sandbox_user via setpriv or su.

    setpriv (util-linux) execs directly; su -l is the fallback for Alpine/BusyBox.
    No outer sh -c wrapper — DockerProcess wraps in bash -c already.
    """
    inner = (
        f"export HOME=/home/{sandbox_user} && cd /home/{sandbox_user} && {agent_launch}"
    )
    quoted = shlex.quote(inner)
    return (
        f"if setpriv --help 2>&1 | grep -q reuid; then"
        f" exec setpriv --reuid={sandbox_user} --regid={sandbox_user}"
        f" --init-groups -- bash -c {quoted};"
        f" else exec su -l {sandbox_user} -c {quoted};"
        f" fi"
    )


async def setup_sandbox_user(
    env, sandbox_user: str, workspace: str, *, timeout_sec: int = 120
) -> str:
    """Create non-root sandbox user, grant workspace access. Return agent_cwd."""
    if not re.match(r"^[a-z_][a-z0-9_-]*$", sandbox_user):
        raise ValueError(
            f"Invalid sandbox_user: {sandbox_user!r} (must be alphanumeric)"
        )
    logger.info(f"Setting up sandbox user: {sandbox_user}")
    await env.exec(
        f"id -u {sandbox_user} >/dev/null 2>&1 || "
        f"useradd -m -s /bin/bash {sandbox_user} && "
        f"mkdir -p /home/{sandbox_user}/.local/bin && "
        "if [ -d /root/.local/bin ]; then "
        f"cp -aL /root/.local/bin/. /home/{sandbox_user}/.local/bin/ 2>/dev/null || true; fi && "
        "if [ -d /root/.nvm ]; then "
        f"cp -a /root/.nvm/. /home/{sandbox_user}/.nvm/ 2>/dev/null || true; fi && "
        f"for d in {' '.join(sorted(get_sandbox_home_dirs()))}; do "
        f"if [ -d /root/$d ]; then mkdir -p /home/{sandbox_user}/$d && "
        f"cp -a /root/$d/. /home/{sandbox_user}/$d/ 2>/dev/null || true; fi; done && "
        f"chown -R {sandbox_user}:{sandbox_user} /home/{sandbox_user} && "
        f"chown -R {sandbox_user}:{sandbox_user} {shlex.quote(workspace)}",
        timeout_sec=timeout_sec,
    )
    logger.info(f"Sandbox user {sandbox_user} ready (workspace={workspace})")
    return workspace


__all__ = ["build_priv_drop_cmd", "setup_sandbox_user"]
