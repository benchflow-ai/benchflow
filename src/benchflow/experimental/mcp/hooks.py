"""MCP service hooks for Trial lifecycle.

Starts MCP servers (reviewer, tools, etc.) as background processes
in the sandbox before agent execution begins. Declared in TrialConfig.services.

Usage in trial YAML:
    services:
      - "benchflow-reviewer:8100"

Or programmatically:
    config = TrialConfig(
        ...,
        services=["benchflow-reviewer:8100"],
        pre_agent_hooks=[mcp_reviewer_hook(port=8100, model="gemini-3.1-flash-lite")],
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def mcp_reviewer_hook(
    port: int = 8100,
    model: str = "gemini-3.1-flash-lite",
    host: str = "0.0.0.0",
):
    """Create a pre_agent_hook that starts the MCP reviewer server in the sandbox."""

    async def _start_reviewer(env: Any) -> None:
        logger.info(f"Starting MCP reviewer server on port {port} (model={model})")
        await env.exec(
            f"python -m benchflow.experimental.mcp.reviewer_server "
            f"--port {port} --model {model} --host {host} &",
            timeout_sec=10,
        )
        # Wait for server to respond
        result = await env.exec(
            f"for i in $(seq 1 15); do "
            f"curl -sf http://localhost:{port}/mcp > /dev/null 2>&1 && echo ok && exit 0; "
            f"sleep 1; done; echo fail",
            timeout_sec=20,
        )
        if "ok" in (result.stdout or ""):
            logger.info(f"MCP reviewer server ready on port {port}")
        else:
            logger.warning(f"MCP reviewer server may not be ready on port {port}")

    return _start_reviewer
