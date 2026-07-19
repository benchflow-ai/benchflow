"""Write a seat's per-agent instruction file into its sandbox folder.

Each agent reads a conventional instruction file from its cwd — claude-agent-acp
reads ``CLAUDE.md``, gemini reads ``GEMINI.md``, everything else reads ``AGENTS.md``
(the filename is :attr:`AgentConfig.instruction_filename`). The concurrent-floor
runner calls this BEFORE launching the agent so the file is in place when the
agent starts in ``/work/<seat>``.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from benchflow.agents.registry import AgentConfig

__all__ = ["instruction_target", "write_agent_instructions"]


def instruction_target(agent_cwd: str, cfg: AgentConfig) -> str:
    """The in-sandbox path the agent will read instructions from."""
    return f"{agent_cwd.rstrip('/')}/{cfg.instruction_filename}"


async def write_agent_instructions(
    sandbox,
    agent_cwd: str,
    cfg: AgentConfig,
    instructions_path: str | Path | None,
) -> str | None:
    """Upload ``instructions_path`` into ``<agent_cwd>/<cfg.instruction_filename>``.

    No-op (returns None) when the seat declares no ``instructions:``. Returns the
    target path on success.
    """
    if instructions_path is None:
        return None
    target = instruction_target(agent_cwd, cfg)
    await sandbox.exec(f"mkdir -p {shlex.quote(agent_cwd)}", timeout_sec=20)
    await sandbox.upload_file(Path(instructions_path), target)
    return target
