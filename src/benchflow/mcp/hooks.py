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
            f"python -m benchflow.mcp.reviewer_server "
            f"--port {port} --model {model} --host {host} &",
            timeout_sec=10,
        )
        # Wait for server to respond (FastMCP uses /mcp endpoint, not /health)
        await env.exec(
            f"for i in $(seq 1 15); do "
            f"curl -sf http://localhost:{port}/mcp > /dev/null 2>&1 && break; "
            f"curl -sf http://localhost:{port}/ > /dev/null 2>&1 && break; "
            f"sleep 1; done",
            timeout_sec=20,
        )
        logger.info(f"MCP reviewer server ready on port {port}")

    return _start_reviewer


def mcp_service_hooks_from_config(services: list[str] | None) -> list:
    """Parse service declarations into pre_agent_hooks.

    Format: "service-name:port" where service-name maps to a known MCP server.
    """
    if not services:
        return []

    hooks = []
    for svc in services:
        parts = svc.split(":")
        name = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8100

        if name == "benchflow-reviewer":
            hooks.append(mcp_reviewer_hook(port=port))
        else:
            logger.warning(f"Unknown MCP service: {name}")

    return hooks
