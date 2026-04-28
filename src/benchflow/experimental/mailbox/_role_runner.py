"""EXPERIMENTAL SURFACE — may change or be removed in any minor version.

``default_role_runner`` — factory that returns the MailboxRunner role_runner
callback most callers want: install + connect ACP + execute + close.

Captures per-agent install state in the returned closure so repeat invocations
in the same MailboxRunner.run skip the install. Does NOT handle credential
file uploads or sandbox-user setup — those are once-per-sandbox concerns; do
them before calling MailboxRunner.run, or write a custom role_runner that
wraps this one.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from benchflow.agents.install import install_agent
from benchflow.agents.registry import AGENT_LAUNCH
from benchflow.agents.run import connect_acp, execute_prompts
from benchflow.experimental.mailbox._runner import MailboxRole

logger = logging.getLogger(__name__)


def default_role_runner(
    *,
    sandbox_user: str = "agent",
    agent_cwd: str = "/app",
    environment: str = "daytona",
    timeout_sec: int = 600,
    agent_env: dict[str, str] | None = None,
) -> Callable[[Any, MailboxRole, str], Awaitable[None]]:
    """Build a role_runner callback for ``MailboxRunner.run``.

    Each invocation:
      1. Creates a fresh tempdir for the role's trial_dir.
      2. Calls ``install_agent`` if the agent hasn't been installed in this
         closure yet (cached per-agent for the closure's lifetime).
      3. Connects ACP, executes the prompt, closes the client.

    The closure captures install state, so reusing the same returned callback
    across MailboxRunner roles avoids redundant installs.
    """
    installed: set[str] = set()
    static_env = dict(agent_env or {})

    async def _runner(env: Any, role: MailboxRole, prompt: str) -> None:
        trial_dir = Path(tempfile.mkdtemp(prefix=f"mailbox-{role.name}-"))

        if role.agent not in installed:
            await install_agent(env, role.agent, trial_dir)
            installed.add(role.agent)

        launch_cmd = AGENT_LAUNCH.get(role.agent, role.agent)
        client, session, _ = await connect_acp(
            env=env,
            agent=role.agent,
            agent_launch=launch_cmd,
            agent_env=static_env,
            sandbox_user=sandbox_user,
            model=role.model,
            trial_dir=trial_dir,
            environment=environment,
            agent_cwd=agent_cwd,
        )
        try:
            await execute_prompts(client, session, [prompt], timeout=timeout_sec)
        finally:
            with contextlib.suppress(Exception):
                await client.close()

    return _runner
