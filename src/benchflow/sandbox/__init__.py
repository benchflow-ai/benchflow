"""Sandbox registry and base contracts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.sandbox._base import (
    BaseSandbox,
    ExecResult,
    SandboxClosed,
    SandboxState,
    SandboxType,
)
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.docker import DockerSandbox, DockerSandboxEnvVars

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a sandbox service (e.g. claw-gmail)."""

    name: str
    cli_name: str
    port: int
    db_path: str
    health_path: str = "/health"
    description: str = ""


SERVICES: dict[str, ServiceConfig] = {
    "gmail": ServiceConfig(
        name="gmail",
        cli_name="claw-gmail",
        port=9001,
        db_path="/data/gmail.db",
        description="Mock Gmail REST API (FastAPI + SQLite)",
    ),
    "gcal": ServiceConfig(
        name="gcal",
        cli_name="claw-gcal",
        port=9003,
        db_path="/data/gcal.db",
        description="Mock Google Calendar API",
    ),
    "gdoc": ServiceConfig(
        name="gdoc",
        cli_name="claw-gdoc",
        port=9004,
        db_path="/data/gdoc.db",
        description="Mock Google Docs API",
    ),
    "gdrive": ServiceConfig(
        name="gdrive",
        cli_name="claw-gdrive",
        port=9005,
        db_path="/data/gdrive.db",
        description="Mock Google Drive API",
    ),
    "slack": ServiceConfig(
        name="slack",
        cli_name="claw-slack",
        port=9002,
        db_path="/data/slack.db",
        description="Mock Slack API",
    ),
}


def register_service(
    name: str,
    cli_name: str,
    port: int,
    db_path: str,
    **kwargs: Any,
) -> ServiceConfig:
    """Register a custom service at runtime."""
    config = ServiceConfig(
        name=name, cli_name=cli_name, port=port, db_path=db_path, **kwargs
    )
    SERVICES[name] = config
    return config


def detect_services_from_dockerfile(task_path: Path | str) -> list[ServiceConfig]:
    """Auto-detect which services a task needs from its Dockerfile."""
    dockerfile = Path(task_path) / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return []
    text = dockerfile.read_text()
    return [svc for svc in SERVICES.values() if svc.cli_name in text]


def build_service_hooks(
    services: list[ServiceConfig],
) -> list[Callable[[Any], Coroutine[Any, Any, None]]]:
    """Build pre_agent_hooks that start the given services."""
    if not services:
        return []

    async def _start_services(env: Any) -> None:
        for svc in services:
            await env.exec(
                f"{svc.cli_name} --db {svc.db_path} serve "
                f"--host 0.0.0.0 --port {svc.port} --no-mcp &",
                timeout_sec=10,
            )
        for svc in services:
            await env.exec(
                f"for i in $(seq 1 30); do "
                f"curl -sf http://localhost:{svc.port}{svc.health_path} > /dev/null && break; "
                f"sleep 1; done",
                timeout_sec=60,
            )
        logger.info("Started services: %s", [s.name for s in services])

    return [_start_services]


BaseEnvironment = BaseSandbox

DockerEnvironment = DockerSandbox
DockerEnvironmentEnvVars = DockerSandboxEnvVars

__all__ = [
    "BaseEnvironment",
    "BaseSandbox",
    "DockerEnvironment",
    "DockerEnvironmentEnvVars",
    "DockerSandbox",
    "DockerSandboxEnvVars",
    "DaytonaSandbox",
    "ExecResult",
    "SandboxClosed",
    "SandboxState",
    "SandboxType",
    "SERVICES",
    "ServiceConfig",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
]
