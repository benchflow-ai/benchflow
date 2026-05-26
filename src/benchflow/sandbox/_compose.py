"""Compose helpers for Docker and Daytona DinD backends."""

import shlex
from pathlib import Path, PurePosixPath

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"


def compose_cp_destination(service: str, container_path: str) -> str:
    """Return the service-qualified destination used by compose cp."""
    return f"{service}:{container_path}"


def compose_mkdir_p_command(container_path: str) -> str:
    """Return a POSIX shell command that creates a container path."""
    return f"mkdir -p {shlex.quote(container_path)}"


def compose_parent_mkdir_p_command(container_path: str) -> str | None:
    """Return a mkdir command for a container path parent, if it has one."""
    parent = str(PurePosixPath(container_path).parent)
    if parent in {"", "."}:
        return None
    return compose_mkdir_p_command(parent)
