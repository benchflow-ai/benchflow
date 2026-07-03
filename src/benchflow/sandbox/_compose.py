"""Compose helpers for Docker and Daytona DinD backends."""

import re
import shlex
from pathlib import Path, PurePosixPath

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"

# Back-off delays for retrying a `compose up` that hit a daemon-side network
# create/attach race. Shared by the host docker.py path and the Daytona DinD
# path so a fresh-daemon race is retried identically on both. Extended past the
# original (2.0, 5.0): under max-parallel sweeps many `compose up` calls race on
# the daemon's network create/attach at once, and two short retries were not
# enough (observed "network <project>_default not found" surviving both).
COMPOSE_UP_RETRY_DELAYS_SEC = (2.0, 5.0, 10.0, 20.0)
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
