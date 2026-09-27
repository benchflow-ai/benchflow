"""Compose helpers for Docker and Daytona DinD backends."""

import json
import re
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

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


def stage_compose_context(environment_dir: Path, compose: Mapping[str, Any]) -> Path:
    """A private copy of *environment_dir* whose docker-compose.yaml is *compose*.

    The compose backends read a task's services only from docker-compose.yaml
    in its build context, so services a task declares another way (task.md's
    ``[[sandbox.services]]``) run from a copy, and the task package is never
    written to. Service ``environment`` values are resolved from the host as
    ``[sandbox] env`` values are, and every ``$`` is escaped from Compose's
    interpolation, so containers get values and commands as written. The file
    is kept out of the build itself, so a Dockerfile that copies its whole
    context does not put the services' settings in the agent's image. The
    caller removes the copy's temporary parent directory.
    """
    from benchflow.task.env import resolve_env_vars

    services = {
        name: (
            {**service, "environment": resolve_env_vars(service["environment"])}
            if "environment" in service
            else service
        )
        for name, service in compose["services"].items()
    }
    text = json.dumps(
        _escape_interpolation({**compose, "services": services}), indent=2
    )
    staged = Path(tempfile.mkdtemp(prefix="benchflow-services-")) / environment_dir.name
    shutil.copytree(environment_dir, staged, symlinks=True)
    # JSON is YAML, so Compose reads the file under the name the backends expect.
    (staged / "docker-compose.yaml").write_text(text + "\n")
    # BuildKit reads Dockerfile.dockerignore instead of .dockerignore when both exist.
    for name in (".dockerignore", "Dockerfile.dockerignore"):
        ignore = staged / name
        if name != ".dockerignore" and not ignore.exists():
            continue
        kept = ignore.read_text() if ignore.exists() else ""
        ignore.unlink(missing_ok=True)  # never write through a symlink
        separator = "\n" if kept and not kept.endswith("\n") else ""
        ignore.write_text(f"{kept}{separator}docker-compose.yaml\n")
    return staged


def _escape_interpolation(value: Any) -> Any:
    """*value* with each ``$`` in its strings doubled, which Compose reads as ``$``."""
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, Mapping):
        return {key: _escape_interpolation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_escape_interpolation(item) for item in value]
    return value


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
