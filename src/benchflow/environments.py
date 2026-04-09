"""Environment registry — declare and manage sandbox services.

Maps service names to their startup commands, ports, and health endpoints.
Used by SDK.run()'s pre_agent_hooks to auto-start services in the container.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a sandbox service (e.g. claw-gmail)."""
    name: str
    cli_name: str            # Binary name (e.g. "claw-gmail")
    port: int                # Default port
    db_path: str             # Default database path
    health_path: str = "/health"
    description: str = ""


# Built-in smolclaws environment services
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
    **kwargs,
) -> ServiceConfig:
    """Register a custom service at runtime."""
    config = ServiceConfig(name=name, cli_name=cli_name, port=port, db_path=db_path, **kwargs)
    SERVICES[name] = config
    return config


def detect_services_from_dockerfile(task_path: Path | str) -> list[ServiceConfig]:
    """Auto-detect which services a task needs from its Dockerfile."""
    dockerfile = Path(task_path) / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return []
    text = dockerfile.read_text()
    return [svc for svc in SERVICES.values() if svc.cli_name in text]


def build_service_hooks(services: list[ServiceConfig]) -> list[Callable[[Any], Coroutine[Any, Any, None]]]:
    """Build pre_agent_hooks that start the given services.

    Returns a list of async callables compatible with SDK.run(pre_agent_hooks=...).
    """
    if not services:
        return []

    async def _start_services(env: Any):
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
        names = [s.name for s in services]
        logger.info(f"Started services: {names}")

    return [_start_services]
