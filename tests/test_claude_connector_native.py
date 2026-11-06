"""Guards the Claude connector policy with the real pinned CLI; opt in with native binary paths."""

import json
import os
import subprocess

import pytest

from benchflow.agents.claude_connector_policy import _INSTALL, CLAUDE_CONNECTOR_SETTINGS

pytestmark = pytest.mark.claude_connector_policy


@pytest.mark.skipif(
    not (os.environ.get("TEST_CLAUDE_BINARY") and os.environ.get("TEST_NODE_BINARY")),
    reason="requires Docker and TEST_CLAUDE_BINARY / TEST_NODE_BINARY",
)
def test_managed_policy_blocks_inherited_mcp_despite_cli_settings():
    for managed in (False, True):
        script = r"""set -eu
mkdir -p /home/user /app
chmod 777 /home/user /app
printf '%s' '{"mcpServers":{"probe":{"command":"sh","args":["-c","touch /app/connected"]}}}' > /home/user/.claude.json
chmod 644 /home/user/.claude.json
"""
        if managed:
            script += 'node -e "$INSTALL" "$POLICY"\n'
        script += r"""
cd /app
setpriv --reuid=65534 --regid=65534 --clear-groups env HOME=/home/user CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 /cli --settings '{"allowManagedMcpServersOnly":false,"allowedMcpServers":[{"serverName":"probe"}],"disableClaudeAiConnectors":false}' mcp list
if [ -f /app/connected ]; then echo CONNECTED; else echo BLOCKED; fi
"""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{os.environ['TEST_CLAUDE_BINARY']}:/cli:ro",
                "-v",
                f"{os.environ['TEST_NODE_BINARY']}:/usr/local/bin/node:ro",
                "-e",
                f"INSTALL={_INSTALL}",
                "-e",
                f"POLICY={json.dumps(CLAUDE_CONNECTOR_SETTINGS)}",
                "ubuntu:24.04",
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("BLOCKED" if managed else "CONNECTED") in result.stdout
