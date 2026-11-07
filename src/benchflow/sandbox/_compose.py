"""Compose helpers for Docker and Daytona DinD backends."""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
from benchflow.task.config import NetworkMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from benchflow.sandbox.protocol import ExecResult
    from benchflow.task.config import SandboxConfig

COMPOSE_DIR = Path(__file__).parent / "_compose_files"
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_REMOTE_BASE_PATH = COMPOSE_DIR / "docker-compose-remote-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
COMPOSE_NET_ADMIN_PATH = COMPOSE_DIR / "docker-compose-net-admin.yaml"
COMPOSE_NO_NET_RAW_PATH = COMPOSE_DIR / "docker-compose-no-net-raw.yaml"
COMPOSE_NO_RESTART_PATH = COMPOSE_DIR / "docker-compose-no-restart.yaml"
COMPOSE_KEPT_FREEZER_PATH = COMPOSE_DIR / "docker-compose-kept-freezer.yaml"
COMPOSE_NO_NEW_PRIVILEGES_PATH = COMPOSE_DIR / "docker-compose-no-new-privileges.yaml"

# Keep DinD's upload and preflight in the same VM coordinate space.
DIND_TASK_DIR = "/runtime/task"
DIND_ENVIRONMENT_DIR = f"{DIND_TASK_DIR}/environment"
DIND_LOGS_DIR = "/runtime/logs"

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


async def compose_container_ids(
    compose: Any, *, service: str | None = None, all_containers: bool = True
) -> list[str]:
    """Read IDs on stdout only; Compose warnings belong to its diagnostic stream.

    Every discovery path, including stopped prototypes and checkpoint peers,
    uses the same framing and refuses non-ID records before docker inspect.
    """
    from benchflow.sandbox._exec_output import exec_raw

    result: ExecResult = await exec_raw(
        compose,
        [
            "ps",
            *(["--all"] if all_containers else []),
            "-q",
            *([service] if service else []),
        ],
        merge_stderr=False,
    )
    output = result.stdout
    if not isinstance(output, str):
        raise RuntimeError("Cannot read Compose container identities; retry startup")
    ids: list[str] = output.splitlines()
    if result.return_code or any(
        not re.fullmatch(r"[0-9a-f]{12,64}", cid) for cid in ids
    ):
        raise RuntimeError("Cannot identify Compose containers; retry startup")
    return ids


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


def compose_file_paths(
    environment_dir: Path,
    config: SandboxConfig,
    *,
    base: Path = COMPOSE_BASE_PATH,
    prebuilt: bool = False,
    mounts: Path | None = None,
    no_network: bool = True,
) -> list[Path]:
    """The ``-f`` files of a Docker sandbox's ``docker compose`` calls, in
    order: BenchFlow's base and build (or prebuilt) files, the task's own
    ``docker-compose.yaml`` (without one, the no-new-privileges overlay),
    the operator's mounts, and the network overlays.

    The launch gate's ``docker compose config`` passes ``no_network=False``:
    that overlay only takes the network away (``network_mode: none``), and a
    run that keeps the agent's network leaves it out.

    A task's own project gets no-new-privileges on every service after
    Compose resolves it (``freeze_compose``, ``private_build_override``),
    never from a file stacked before the task's, which Compose 2.40 merges
    into a list it refuses when the task sets the same option.
    """
    paths = [base, COMPOSE_PREBUILT_PATH if prebuilt else COMPOSE_BUILD_PATH]
    task_compose = environment_dir / COMPOSE_DEFINITION_NAMES[0]
    if task_compose.exists():
        paths.append(task_compose)
    else:
        paths.append(COMPOSE_NO_NEW_PRIVILEGES_PATH)
    if mounts is not None:
        paths.append(mounts)
    if no_network and not config.allow_internet:
        paths.append(COMPOSE_NO_NETWORK_PATH)
    if compose_needs_net_admin(config):
        paths.append(COMPOSE_NET_ADMIN_PATH)
    if compose_drops_net_raw(config):
        paths.append(COMPOSE_NO_NET_RAW_PATH)
    if compose_stops_main_restart(config):
        paths.append(COMPOSE_NO_RESTART_PATH)
    return paths


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

    A fourth: an open network whose run refuses the cloud metadata endpoints
    to its agent (``lockdown.cloud_metadata_plan``), which the sandbox
    factory marks on its copy of the config, unless the Docker sandbox's
    metadata helper adds those rules from outside ``main``
    (``cloud_metadata_by_helper``). An open run without the block (the run
    opts out, or no agent runs) gets no capability, as before.
    """
    mode = config.network_mode
    return (
        mode in (NetworkMode.DENYLIST, NetworkMode.ALLOWLIST)
        or (mode == NetworkMode.NO_NETWORK and config.allow_internet)
        or (
            mode == NetworkMode.PUBLIC
            and config.allow_internet
            and getattr(config, "cloud_metadata_block", False)
            and not getattr(config, "cloud_metadata_by_helper", False)
            # A root agent's main never gets it for the block: the block holds
            # root only where root has none.
            and not getattr(config, "cloud_metadata_root_agent", False)
        )
    )


def compose_drops_net_raw(config: SandboxConfig) -> bool:
    """Whether ``main`` starts without NET_RAW: the agent runs as root under
    the cloud metadata block, whose rules BenchFlow's helper adds from
    outside ``main``. A packet socket sends frames the OUTPUT
    chain never sees, so root with NET_RAW could go around the rules; the
    block reads root's sets before it holds a root agent."""
    return bool(
        getattr(config, "cloud_metadata_root_agent", False)
        and getattr(config, "cloud_metadata_by_helper", False)
    )


def compose_stops_main_restart(config: SandboxConfig) -> bool:
    """Whether ``main`` gets ``docker-compose-no-restart.yaml``: the cloud
    metadata block applies to it. The block's rules live in main's network
    namespace; a restarted ``main`` gets a new namespace without them, while
    the record would still say ``on``, so ``main`` must never restart."""
    return bool(getattr(config, "cloud_metadata_block", False))


def main_restart_refusal(policy: str | None) -> str:
    """Why the cloud metadata block cannot hold a ``main`` whose restart
    policy the daemon reports as ``policy`` ("" when it cannot restart;
    None when the policy could not be read)."""
    if policy is None:
        return "main's restart policy could not be read"
    if policy in ("", "no"):
        return ""
    return (
        f"main restarts (restart policy {policy!r}), and a restarted main "
        "gets a new network namespace without the rules"
    )


def compose_needs_kept_freezer(env: object) -> bool:
    """Whether ``main`` gets ``docker-compose-kept-freezer.yaml``: what
    :func:`decide_kept_service_freezer` set for this start. The one test
    every Compose provider (Docker, remote Docker, Daytona DinD) uses."""
    return bool(getattr(env, "kept_service_freezer", False))


# Which sandboxes ask for the kept-service freezer (writable-cgroups on main),
# decided once per start, before Compose creates anything, for ordinary
# startup and recovery alike. Only a sandbox whose rollout runs
# agent code of a task that may keep services asks (keep_services.may_keep,
# decided with no identity): the rollout sets
# ``kept_service_runner`` to "oracle" or "agent". nop, verifier-only and every
# other sandbox have none and start exactly as without K1. The Engine's own
# report (``docker info``) says whether it can give the freezer: Docker
# Engine 28.0 added writable-cgroups (27.5.1 refuses the option), and it is
# refused under rootless mode or user-namespace remapping; the freezer also
# needs cgroup v2. Where it cannot, the sandbox starts without the freezer,
# and the trial is judged once its identities are measured, before any agent
# code runs: a trial that keeps nothing goes on; a non-root oracle that keeps
# services meets the freezer probe, which refuses the trial; a non-root agent
# that keeps services is refused with the Engine's reason. Never decided from
# a failed startup.
WRITABLE_CGROUPS_ENGINE = (28, 0)


def engine_freezer_refusal(info_json: str) -> str | None:
    """Why the Engine whose ``docker info --format '{{json .}}'`` is
    *info_json* cannot give the kept-service freezer, or None when it can."""
    import json

    try:
        info = json.loads(info_json)
    except ValueError:
        return "the Docker Engine's info could not be read"
    if not isinstance(info, dict):
        return "the Docker Engine's info could not be read"
    version = str(info.get("ServerVersion") or "")
    parts = re.match(r"(\d+)\.(\d+)", version)
    if parts is None:
        return f"the Docker Engine reports no version ({version or 'none'})"
    if (int(parts[1]), int(parts[2])) < WRITABLE_CGROUPS_ENGINE:
        return f"Docker Engine {version} predates writable-cgroups (28.0)"
    if str(info.get("CgroupVersion") or "") != "2":
        return (
            f"the Docker Engine uses cgroup v{info.get('CgroupVersion') or '?'}, not v2"
        )
    for option in info.get("SecurityOptions") or ():
        name = dict(
            field.split("=", 1) for field in str(option).split(",") if "=" in field
        ).get("name")
        if name in ("rootless", "userns"):
            return (
                f"the Docker Engine runs in {name} mode, which refuses writable-cgroups"
            )
    return None


async def decide_kept_service_freezer(
    env: Any, read_engine_info: Callable[[], Awaitable[str]]
) -> None:
    """Set ``env.kept_service_freezer`` for this start (see above), and
    ``env.kept_service_freezer_refusal`` to the Engine's reason when the
    runner asked and the Engine cannot give it. Nothing is refused here:
    whether the trial keeps anything is known only once its identities are
    measured, and a non-root agent that keeps services on such an Engine is
    refused then, before any agent code runs."""
    env.kept_service_freezer = False
    env.kept_service_freezer_refusal = None
    runner = getattr(env, "kept_service_runner", None)
    if runner not in ("oracle", "agent"):
        return
    why = engine_freezer_refusal(await read_engine_info())
    if why is None:
        env.kept_service_freezer = True
        return
    env.kept_service_freezer_refusal = why
    logger.info("Starting the sandbox without the kept-service freezer: %s", why)


def engine_freezer_unavailable(why: str) -> KeptServiceFreezerUnavailable:
    """The refusal of a non-root agent that keeps services on an Engine that
    cannot give the freezer (``why``, :func:`engine_freezer_refusal`)."""
    return KeptServiceFreezerUnavailable(
        "This trial keeps services (x-keep-services) for an agent that "
        "runs as a non-root user, which BenchFlow holds in a frozen cgroup "
        f"from the kill after the agent's turn until the verifier starts; "
        f"{why}. No agent code ran. Use Docker Engine 28 or later "
        "with cgroup v2, not rootless and without user-namespace remapping."
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
