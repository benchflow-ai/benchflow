"""Compose helpers for Docker and Daytona DinD backends."""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.task.config import HealthcheckConfig

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"

# Back-off delays for retrying a `compose up` that hit a daemon-side network
# create/attach race. Shared by the host docker.py path and the Daytona DinD
# path so a fresh-daemon race is retried identically on both.
COMPOSE_UP_RETRY_DELAYS_SEC = (2.0, 5.0)
# Daemon-side create/attach race seen on Docker 29.x: `compose up` prints
# "Network ... Created" but the container create/start that follows fails with
# "network <project>_default not found". Older daemons emit the same race
# without the "failed to set up container networking" wrapper.
_COMPOSE_UP_NETWORK_RACE_ERROR = re.compile(
    r"error response from daemon: "
    r"(?:failed to set up container networking: )?network \S+ not found",
    re.IGNORECASE,
)


def is_compose_up_network_race_error(message: str) -> bool:
    """Return whether *message* is a retryable compose-up network create race."""
    return bool(_COMPOSE_UP_NETWORK_RACE_ERROR.search(message))


def _compose_duration(seconds: float) -> str:
    """Serialize a duration in seconds as a Compose duration string (e.g. "5s")."""
    if seconds == int(seconds):
        return f"{int(seconds)}s"
    return f"{seconds}s"


def healthcheck_to_compose_block(healthcheck: HealthcheckConfig) -> dict[str, Any]:
    """Render a task ``HealthcheckConfig`` as a Compose service ``healthcheck`` block.

    Maps the task-declared healthcheck onto Compose healthcheck keys so that
    ``compose up --wait`` blocks until the in-container service reports healthy,
    closing the boot race where a container is treated "up" before its service
    is ready. Field mapping:

    - ``command``            -> ``test`` as ``["CMD-SHELL", command]``
    - ``interval_sec``       -> ``interval``
    - ``timeout_sec``        -> ``timeout``
    - ``retries``            -> ``retries``
    - ``start_period_sec``   -> ``start_period``
    - ``start_interval_sec`` -> ``start_interval``
    """
    return {
        "test": ["CMD-SHELL", healthcheck.command],
        "interval": _compose_duration(healthcheck.interval_sec),
        "timeout": _compose_duration(healthcheck.timeout_sec),
        "retries": healthcheck.retries,
        "start_period": _compose_duration(healthcheck.start_period_sec),
        "start_interval": _compose_duration(healthcheck.start_interval_sec),
    }


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
