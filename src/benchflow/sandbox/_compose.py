"""Compose helpers for Docker and Daytona DinD backends."""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from benchflow.task.config import NetworkMode

if TYPE_CHECKING:
    from benchflow.task.config import SandboxConfig

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
COMPOSE_NET_ADMIN_PATH = COMPOSE_DIR / "docker-compose-net-admin.yaml"

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


#: Filenames Docker Compose recognizes for a task-supplied topology.
COMPOSE_DEFINITION_NAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)


def compose_definition_path(environment_dir: Path) -> Path | None:
    """The task's own compose file in *environment_dir*, if it has one.

    Single-container backends use this to refuse a multi-service task up front
    rather than silently building only the agent's container.
    """
    for name in COMPOSE_DEFINITION_NAMES:
        candidate = environment_dir / name
        if candidate.is_file():
            return candidate
    return None


def compose_needs_net_admin(config: SandboxConfig) -> bool:
    """Whether ``main`` needs NET_ADMIN for the sandbox-user iptables firewall.

    Three runs arm that firewall inside ``main``: ``network_mode: denylist``,
    ``network_mode: allowlist`` (which also needs the nat table for its DNS
    redirect), and a no-network task whose container stays open for its LLM agent.
    ``_create_sandbox_environment`` (preserve_agent_network) sets
    ``allow_internet`` on a copy whose ``network_mode`` stays no-network, a
    pair task validation never produces; BenchFlow then enforces no-web at the
    agent layer. Oracle runs keep the container network block and get no
    capability. Docker and Daytona DinD both stack
    ``docker-compose-net-admin.yaml`` when this holds.
    """
    mode = config.network_mode
    return mode in (NetworkMode.DENYLIST, NetworkMode.ALLOWLIST) or (
        mode == NetworkMode.NO_NETWORK and config.allow_internet
    )


def is_compose_up_network_race_error(message: str) -> bool:
    """Return whether *message* is a retryable compose-up network create race."""
    return bool(_COMPOSE_UP_NETWORK_RACE_ERROR.search(message))


# What the docker CLI prints when it cannot reach the daemon: current CLIs say
# "failed to connect to the docker API at ...", older ones "Cannot connect to
# the Docker daemon at ... Is the docker daemon running?".
_DOCKER_DAEMON_UNREACHABLE = (
    "failed to connect to the docker api",
    "cannot connect to the docker daemon",
)
_DAEMON_REASON_LIMIT = 300


def docker_daemon_unreachable_reason(message: str) -> str | None:
    """The docker CLI's "cannot reach the daemon" line in *message*, or None.

    A compose failure with this line is an environment problem (Docker is not
    running), not a task problem; callers report it as a sandbox startup
    failure instead of a traceback that embeds the compose command.
    """
    for line in message.splitlines():
        lowered = line.lower()
        for marker in _DOCKER_DAEMON_UNREACHABLE:
            start = lowered.find(marker)
            if start < 0:
                continue
            # The CLI line may follow the compose command on the same line
            # ("... Stdout: failed to connect ..."); keep only the CLI's words.
            line = line[start:].strip()
            if len(line) > _DAEMON_REASON_LIMIT:
                line = line[: _DAEMON_REASON_LIMIT - 3] + "..."
            return line
    return None


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
