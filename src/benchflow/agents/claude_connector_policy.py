"""Root-owned Claude Code connector restrictions, independent of web access."""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from typing import Any

from benchflow._runtime_tree import NODE_RUNTIME_PATHS, runtime_custody_check
from benchflow.agents.registry import (
    _BENCHFLOW_NODE_PREFIX,
    AGENTS,
    CLAUDE_CODE_EXECUTABLE_PATH,
    pinned_npm_package,
)
from benchflow.sandbox._exec_output import exec_raw

#: Why a task with MCP servers is refused for Claude Code, on either harness.
MCP_REFUSAL = (
    "Claude Code's managed connector policy disables MCP servers; "
    "this task requires an agent with task MCP support."
)
# Native managed settings cannot be relaxed by user/project/CLI settings.
CLAUDE_CONNECTOR_SETTINGS = {
    "disableClaudeAiConnectors": True,
    "allowManagedMcpServersOnly": True,
    "allowedMcpServers": [],
}
_INSTALL = r"""
const fs = require('node:fs');
function trusted(p, dir) {
  const s = fs.lstatSync(p);
  if (s.isSymbolicLink() || s.uid !== 0 || (s.mode & 0o022) ||
      (dir ? !s.isDirectory() : !s.isFile())) throw Error('unsafe managed policy path');
}
trusted('/', true); trusted('/etc', true);
const folder = '/etc/claude-code';
try { fs.mkdirSync(folder, {mode:0o755}); }
catch(e) { if(e.code !== 'EEXIST') throw e; }
trusted(folder, true);
const path = folder + '/managed-settings.json';
const required = JSON.parse(process.argv[1]);
try {
  const fd = fs.openSync(path, fs.constants.O_WRONLY | fs.constants.O_CREAT |
    fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, 0o644);
  try { fs.writeFileSync(fd, JSON.stringify(required)); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
} catch(e) { if(e.code !== 'EEXIST') throw e; }
trusted(path, false);
const data = JSON.parse(fs.readFileSync(path, 'utf8'));
for (const [key, value] of Object.entries(required)) {
  if (JSON.stringify(data[key]) !== JSON.stringify(value))
    throw Error('existing managed settings do not enforce connector restriction');
}
if (data.mcpServers && Object.keys(data.mcpServers).length)
  throw Error('managed MCP servers conflict with connector restriction');
console.log('connector-policy-installed');
"""


async def enforce_claude_connector_policy(
    env: Any,
    *,
    agent: str,
    agent_launch: str,
    agent_env: dict[str, str],
    sandbox_user: str | None,
    mcp_servers: Sequence[Any] = (),
    timeout_sec: int = 60,
) -> dict[str, str]:
    """Require native managed restrictions before a supported Claude launch.

    The managed settings allow no MCP server, the task's own included, and
    the CLI drops a blocked server with a warning and goes on. So a task that
    declares *mcp_servers* is refused here, before either harness launches
    Claude Code (ACP's ``session/new`` or the native harness's
    ``--mcp-config``), instead of running without its tools.
    """
    if agent != "claude-agent-acp":
        return agent_env
    if mcp_servers:
        raise RuntimeError(MCP_REFUSAL)
    if not sandbox_user or sandbox_user in {"root", "0"}:
        raise RuntimeError(
            "Claude connector restrictions require a non-root sandbox user"
        )
    if agent_launch != AGENTS[agent].launch_cmd:
        raise RuntimeError("Claude connector restrictions require the managed launcher")
    if any(agent_env.get(key) for key in ("NODE_OPTIONS", "NODE_PATH")):
        raise RuntimeError(
            "Claude connector restrictions refuse Node injection overrides"
        )
    _, pin = pinned_npm_package("claude-code")
    command = (
        runtime_custody_check(NODE_RUNTIME_PATHS, "managed Node runtime")
        + f' && uid="$(/usr/bin/id -u {shlex.quote(sandbox_user)})" && [ "$uid" -gt 0 ]'
        + f' && [ "$(env -u NODE_OPTIONS -u NODE_PATH {CLAUDE_CODE_EXECUTABLE_PATH} --version)" = {shlex.quote(pin + " (Claude Code)")} ]'
        + f" && env -u NODE_OPTIONS -u NODE_PATH {_BENCHFLOW_NODE_PREFIX}/bin/node -e "
        + shlex.quote(_INSTALL)
        + " "
        + shlex.quote(json.dumps(CLAUDE_CONNECTOR_SETTINGS))
    )
    result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout_sec)
    if (
        result.return_code != 0
        or (result.stdout or "").strip() != "connector-policy-installed"
    ):
        raise RuntimeError("Claude managed connector restriction could not be verified")
    return agent_env
