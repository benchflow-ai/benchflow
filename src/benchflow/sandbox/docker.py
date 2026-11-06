"""Native DockerSandbox, in RL-first terminology.

Uses docker-compose for container orchestration on local Docker.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import base64
import contextlib
import ipaddress
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from pydantic import BaseModel

from benchflow._bfx import client as bfx_client
from benchflow._bfx import delivery as bfx_delivery
from benchflow._bfx.locate import SANDBOX_DIR as BFX_DIR
from benchflow._bfx.locate import SANDBOX_PATH as BFX_PATH
from benchflow._bfx.locate import BfxUnavailable, normalize_arch
from benchflow._runtime_tree import KIT_BIN, KIT_PYTHON, ROOT_HOME, RUN_DIR
from benchflow._utils.reachable_write import (
    ReachablePath,
    StagedTar,
    UnsafeDownloadEntryError,
    mount_environment,
)
from benchflow._utils.text import describe_exception
from benchflow.sandbox import (
    _bind_targets,
    _direct_docker,
    _inert_bootstrap,
    _metadata_helper,
    _transport,
    _trial_volumes,
)
from benchflow.sandbox._base import (
    IMAGE_ENV_PREFIX,
    SYSTEM_PATH,
    SYSTEM_SHELL,
    SYSTEM_SHELL_EMPTY_VARS,
    BaseSandbox,
    ExecResult,
    ExecRoute,
    StartupNamesRefused,
    _filter_compose_service_names,
    _root_exec_command,
    capture_pid1_image_env,
    env_file_body,
    env_stdin_body,
    loader_switches,
    loader_variable,
    refused_startup_names,
    restore_image_env,
    startup_refusal,
    system_shell_env_flags,
    system_shell_route,
    with_system_path,
    wrap_command_with_env_stdin,
)
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_KEPT_FREEZER_PATH,
    COMPOSE_NET_ADMIN_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_UP_RETRY_DELAYS_SEC,
    compose_container_ids,
    compose_file_paths,
    compose_needs_kept_freezer,
    decide_kept_service_freezer,
    docker_daemon_unreachable_reason,
    is_compose_up_network_race_error,
)
from benchflow.sandbox._compose_build import ComposeBuild, resolve_compose_build
from benchflow.sandbox._compose_runtime import private_image_repository
from benchflow.sandbox._compose_teardown import (
    COMPOSE_KILL_COMMAND,
    built_image_refs,
    compose_down_command,
    image_ref_identity,
)
from benchflow.sandbox._docker_subnets import (
    DockerDaemon,
    SubnetAllocator,
    SubnetConfigError,
    SubnetRangeExhausted,
    allocator_for,
    daemon_key,
    is_pool_overlap_error,
    is_pools_exhausted_error,
    override_document,
    pin_subnets,
    pinned_overlap_message,
    plan_networks,
    pools_exhausted_message,
    read_taken,
    refused_network,
)
from benchflow.sandbox._docker_sweep import (
    OWNED_LABEL,
    OWNED_VALUE,
    PROCESS_ENV,
    PROCESS_LABEL,
    claim_project,
    process_token,
    release_project,
)
from benchflow.sandbox._download import RemoteFileAbsentError, drain, private_download
from benchflow.sandbox._exec_channel import (
    HELPER_SOURCE,
    VERDICT_SOURCE,
    ChannelDeadline,
    ChannelLost,
    ChannelUnavailable,
    ChannelUnconfirmed,
    ExecChannel,
    WarmExecPool,
    helper_argv,
    helper_python_argv,
    parse_journal,
    verdict_argv,
)
from benchflow.sandbox._exec_merge import start_with_identity_probe
from benchflow.sandbox._exec_output import exec_raw, raw_output_requested
from benchflow.sandbox._recovery_baseline import (
    DockerRecoveryBaseline,
    capture_baseline,
    validate_baseline,
)
from benchflow.sandbox._snapshot_credentials import (
    AsUserResult,
    StashedCredential,
    put_back_credentials,
    scrub_credentials,
)
from benchflow.sandbox._transport import fit, reader_cat
from benchflow.sandbox.env_layers import (
    AGENT,
    ALIAS_PREFIX,
    LAYER_LABELS,
    PROXY,
    VERIFIER,
    AgentLayerRecipe,
    BaseImage,
    DockerLayers,
    LayerError,
    LayerLockFallback,
    LayerResult,
    LayerSkipped,
    ProxyLayerRecipe,
    alias_owner_tag,
    fresh_record,
)
from benchflow.sandbox.image_build import image_build_error
from benchflow.sandbox.protocol import (
    ExecTimeoutError,
    SandboxImage,
    SandboxRestoreHostConfigUnavailable,
    SandboxSnapshotNotSupported,
    SandboxStartupError,
)
from benchflow.sandbox.verifier_layers import (
    VerifierLayer,
    VerifierLayerRecipe,
    ensure_verifier_layer,
)
from benchflow.task.config import NetworkMode, SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.trust import state as trust_state

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess
    from benchflow.task.compose_env import ComposeLaunch
    from benchflow.task.staging import StagedEnvironment

logger = logging.getLogger("benchflow")

_DOCKER_BUILD_RETRY_DELAYS_SEC = (2.0, 5.0)
# A build still running after this many seconds is announced, then a line
# every _BUILD_HEARTBEAT_SEC: a cached rebuild (about a second) stays quiet,
# and a first build (45 s for SkillsBench citation-check) no longer looks
# like a hang after "Starting environment" (DF-p1-first-run-6).
_BUILD_ANNOUNCE_AFTER_SEC = 5.0
_BUILD_HEARTBEAT_SEC = 30.0
# Bound at import: tests replace asyncio.sleep to skip the build retry delays
# and count them; the heartbeat's waits are not retry delays.
_heartbeat_sleep = asyncio.sleep


def prefetch_local_subnet_allocator() -> asyncio.Task | None:
    """Begin reading the local daemon's networks for the subnet allocator
    as a job starts (PERF-SBX), before its first sandbox exists.

    The same read a local Docker sandbox's ``_prefetch_subnet_allocator``
    starts with its own start(), and ``_choose_subnets`` then waited for
    (0.3-0.4 s): ``allocator_for`` shares one pending build per daemon
    connection on this loop and publishes the allocator under the daemon's
    identity, so the sandbox, which still identifies its daemon itself,
    takes this one. Nothing is chosen or reserved; every failure is left to
    the sandbox's own call, as before. None when no loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    async def cli(args: list[str]) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return ExecResult(
            stdout=(out or b"").decode(errors="replace"),
            stderr=(err or b"").decode(errors="replace"),
            return_code=proc.returncode,
        )

    from benchflow.task.compose_env import docker_connection_args

    try:
        socket = _direct_docker.endpoint_socket(docker_connection_args())
    except _direct_docker.Fallback:
        socket = None
    engine = _EngineReads(socket) if socket else None

    async def docker(args: list[str]) -> ExecResult:
        if engine is not None:
            answer = await engine.answer(args)
            if answer is not None:
                return answer
        return await cli(args)

    async def prefetch() -> None:
        with contextlib.suppress(Exception):
            key = daemon_key(None)
            await key.identify()
            await allocator_for(key, docker, daemon_display="Docker daemon")

    return loop.create_task(prefetch())


class _EngineReads:
    """The subnet allocator's three daemon reads, answered by the daemon's
    Engine API on the CLI's own Unix socket, in the CLI's output format
    (PERF-SBX): ``docker info`` with the allocator's format, ``network ls
    --quiet --no-trunc`` and ``network inspect --format '{{json
    .IPAM.Config}}'``; one request for the networks instead of two CLI
    processes (0.17 s). Any other command, and any error, is None: the CLI
    answers instead."""

    _INFO = '{"name":{{json .Name}},"pools":{{json .DefaultAddressPools}}}'

    def __init__(self, socket: str) -> None:
        from benchflow.sandbox._engine_api import EngineClient

        self.client = EngineClient(socket, timeout=30.0)
        self.networks: dict[str, Any] | None = None

    async def _get(self, path: str) -> Any:
        return (await self.client.call("GET", path)).json()

    async def answer(self, args: list[str]) -> ExecResult | None:
        try:
            if args == ["info", "--format", self._INFO]:
                info = await self._get("/info")
                if not isinstance(info, dict) or not isinstance(info.get("Name"), str):
                    return None
                facts = {"name": info["Name"], "pools": info.get("DefaultAddressPools")}
                return ExecResult(stdout=json.dumps(facts) + "\n", return_code=0)
            if args == ["network", "ls", "--quiet", "--no-trunc"]:
                rows = await self._get("/networks")
                if not isinstance(rows, list) or not all(
                    isinstance(row, dict) and isinstance(row.get("Id"), str)
                    for row in rows
                ):
                    return None
                self.networks = {row["Id"]: row for row in rows}
                return ExecResult(
                    stdout="".join(f"{nid}\n" for nid in self.networks),
                    return_code=0,
                )
            if (
                args[:4] == ["network", "inspect", "--format", "{{json .IPAM.Config}}"]
                and self.networks is not None
                and all(nid in self.networks for nid in args[4:])
            ):
                lines = []
                for nid in args[4:]:
                    ipam = self.networks[nid].get("IPAM")
                    if not isinstance(ipam, dict):
                        return None
                    lines.append(json.dumps(ipam.get("Config")) + "\n")
                return ExecResult(stdout="".join(lines), return_code=0)
        except Exception:
            return None
        return None


class _HostFlusher:
    """One thread that runs ``sync()`` about once a second while any
    sandbox of this process is in its tests (PERF-SBX), so its pages are on
    disk before its overlay's unmount has to write them. A daemon thread:
    the process never waits for it."""

    INTERVAL = 1.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users = 0
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def join(self) -> None:
        with self._lock:
            self._users += 1
            self._wake.set()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="benchflow-flush", daemon=True
                )
                self._thread.start()

    def leave(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._users == 0:
                    self._thread = None
                    return
                self._wake.clear()
            os.sync()
            self._wake.wait(self.INTERVAL)


_HOST_FLUSHER = _HostFlusher()


def _unlink_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}" if minutes else f"{secs} s"


_DOCKER_BUILD_RETRYABLE_ERRORS = (
    re.compile(r"at least one invalid signature was encountered", re.IGNORECASE),
    re.compile(r"the repository '.+' is not signed", re.IGNORECASE),
    re.compile(r"no space left on device", re.IGNORECASE),
    re.compile(r"readtimeouterror", re.IGNORECASE),
    re.compile(r"read timed out", re.IGNORECASE),
    re.compile(r"connection (?:timed out|reset by peer)", re.IGNORECASE),
)

# Two jobs building the same task at once (every main image is tagged
# bf__<task>, and `pull_policy: build` makes `compose up` build too): BuildKit
# fails the second export with `image "...": already exists`. The image is
# then there, so building again succeeds.
_IMAGE_EXPORT_RACE = re.compile(r"image \"[^\"]+\": already exists", re.IGNORECASE)

# Compose-up network-race retry config lives in _compose so the host docker
# path and the Daytona DinD path share the exact same race detection + back-off.
_COMPOSE_UP_RETRY_DELAYS_SEC = COMPOSE_UP_RETRY_DELAYS_SEC

# How many times start() gives a trial new subnets after the daemon refused
# one that overlaps a network made meanwhile (another process's, or the
# user's): each refusal re-reads the daemon's networks first, so a second one
# needs yet another network made in the same moment.
_SUBNET_ATTEMPTS = 5

# On macOS the daemon always runs in a Linux VM (Colima, Docker Desktop, ...).
_DOCKER_RUNS_IN_VM = sys.platform == "darwin"

# Compose and docker subcommands that never create, replace or remove a
# container. Any other one (up, down, stop, run, rm, ...) makes the sandbox
# forget the main container's id, so exec() and the copies go back through
# ``docker compose`` until start() looks the container up again.
_COMPOSE_KEEPS_CONTAINERS = frozenset(
    {"exec", "cp", "ps", "config", "logs", "top", "images", "port", "version", "ls"}
)
#: The docker CLI's connection variables; a call given connection flags
#: carries none of them beside the flags.
_DOCKER_CONNECTION_ENV = frozenset(
    {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"}
    | {"DOCKER_TLS"}
)

_DOCKER_KEEPS_CONTAINERS = frozenset(
    {"exec", "cp", "inspect", "ps", "logs", "top", "port", "diff", "info"}
    | {"version", "image", "images", "commit", "events", "stats"}
)
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")
# A stream cut short by its producer: the exit status is the real error.
_TRANSPORT_EOF = (tarfile.TarError, EOFError)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
# Each docker volume call of a teardown (as _remove_project_volumes's).
_VOLUME_CALL_TIMEOUT_SEC = 30
# How many times a sandbox restarts an exec channel whose helper died.
_EXEC_CHANNEL_RESTARTS = 3


def _opened(opening: asyncio.Future) -> bool:
    """Whether a drained ``StagedTar.__enter__`` left staging to clean up."""
    return opening.done() and not opening.cancelled() and opening.exception() is None


async def _settle(*steps) -> None:
    """Drain each step in order; errors are dropped, cancellation deferred.

    Cleanup must finish before its responsibility is given up: a
    cancellation that arrives during any step is re-raised only once every
    step has completed (TASKMD-R15 B-R14-M1). Cancellation then wins over an
    exception already propagating, which stays chained as its context.
    """
    cancelled: asyncio.CancelledError | None = None
    for step in steps:
        if step is None:
            continue
        try:
            await drain(step)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            pass
    if cancelled is not None:
        raise cancelled


def route_exec_argv(
    route: ExecRoute,
    command: str,
    *,
    container: str,
    user: str,
    stdin: bool = False,
    tty: bool = False,
    cwd: str | None = None,
    shell_argv: list[str] | None = None,
    remove: Sequence[str] = (),
) -> list[str]:
    """``docker exec`` argv (after ``docker``, or after ``docker compose``
    with ``-T`` added by the caller) for one root step on *route* (C8).

    The one place a Docker root step outside :meth:`DockerSandbox.exec`
    gets its process from (the credential scrub, the live launcher's env
    staging, ``attach``): the route's absolute shell, the loader and
    start-up variables set empty where Docker starts it, then PATH pinned
    to the route's and every other loader, start-up and ``PYTHON*`` name
    unset before *command*. *shell_argv* replaces ``[shell, "-c", ...]``
    for an interactive shell, which gets the route's PATH as ``-e``.
    *remove* (:func:`~benchflow.sandbox._base.loader_switches` of the
    container's Config.Env) become bare ``-e NAME``: Docker removes them, so
    a service S1 did not bootstrap starts its shell without its image's
    valued loader switches (LD_TRACE_LOADED_OBJECTS=1 listed the shell's
    libraries instead of running it, SCRIPT-AGENT-DOCKER-F1 round 2); the
    caller runs the client without them (they would be filled in from it).
    """
    if route.pin is None:
        raise ValueError("route_exec_argv is for root's plumbing routes")
    argv = ["exec"]
    if stdin:
        argv.append("-i")
    if tty:
        argv.append("-t")
    if cwd:
        argv.extend(["-w", cwd])
    argv.extend(["-u", user, *system_shell_env_flags()])
    argv.extend(arg for name in remove for arg in ("-e", name))
    if shell_argv is not None:
        if route.kit:
            # On the kit an interactive root shell reads nothing under a
            # HOME another user may write (readline, history; PY8-PT-R8-8).
            for name, value in (
                ("HOME", ROOT_HOME),
                ("INPUTRC", "/dev/null"),
                ("HISTFILE", "/dev/null"),
            ):
                argv.extend(["-e", f"{name}={value}"])
        argv.extend(["-e", f"PATH={route.pin}", container, *shell_argv])
        _transport.sent("Docker", argv)
        return argv
    argv.extend([container, route.shell, "-c", with_system_path(command, route.pin)])
    # Its commands are constant scripts; the bound still holds here.
    _transport.sent("Docker", argv)
    return argv


def _without_loader_names_if_removed(
    args: Sequence[str], environment: dict[str, str] | None
) -> dict[str, str] | None:
    """The docker client's *environment* for *args*: without any loader name
    when *args* removes one with a bare ``-e NAME`` (route_exec_argv's
    *remove*), which Docker would fill in from it; unchanged otherwise."""
    if not any(
        args[i] == "-e" and "=" not in args[i + 1] and loader_variable(args[i + 1])
        for i in range(len(args) - 1)
    ):
        return environment
    return {
        k: v
        for k, v in (os.environ if environment is None else environment).items()
        if not loader_variable(k)
    }


# SandboxImage.meta key: the snapshot holds the protected runtime kit.
_SNAPSHOT_KIT_META = "benchflow_kit"


def _container_env(inspected: dict[str, Any]) -> dict[str, str]:
    """The environment ``docker inspect`` reports the container started with."""
    out: dict[str, str] = {}
    for item in (inspected.get("Config") or {}).get("Env") or []:
        if isinstance(item, str) and "=" in item:
            name, _, value = item.partition("=")
            out[name] = value
    return out


def interactive_root_shell(route: ExecRoute) -> list[str]:
    """argv of root's interactive shell on *route*, started from nothing
    (PT-C8-R9-5 sol / R9-7 astra): ``env -i`` with only terminal data,
    the route's PATH and a HOME (the protected one on the kit), so no
    image value (a ``PROMPT_COMMAND``, a ``PS1`` that runs a command)
    reaches it; then its bash with no profile or rc file, or its POSIX
    shell."""
    env = f"{KIT_BIN}/env" if route.kit else "/usr/bin/env"
    home = ROOT_HOME if route.kit else "/root"
    data = [
        "TERM=xterm-256color",
        f"HOME={home}",
        f"PATH={route.pin or SYSTEM_PATH}",
        "INPUTRC=/dev/null",
        "HISTFILE=/dev/null",
        "PS1=# ",
    ]
    if route.bash.endswith("/bash"):
        return [env, "-i", *data, route.bash, "--noprofile", "--norc", "-i"]
    return [env, "-i", *data, route.shell, "-i"]


def _docker_unavailable_reason(exc: BaseException) -> str | None:
    """Why a docker command could not reach Docker at all, or None.

    Either the ``docker`` CLI is missing or the daemon is unreachable; any
    other failure (a timeout, a daemon error) returns None.
    """
    if isinstance(exc, FileNotFoundError) and exc.filename == "docker":
        return "docker CLI not found on PATH"
    return docker_daemon_unreachable_reason(str(exc))


def _vm_does_not_share(host_path: Path | ReachablePath) -> bool:
    """Whether the Docker VM is expected not to see ``host_path``.

    A Docker VM sees only the host directories it shares. Colima, and Lima
    under it, share only ``$HOME`` by default, so a rollout directory elsewhere
    (for example under ``/tmp``) is bind-mounted as an empty VM-local
    directory. Under ``$HOME``, or on a host where the daemon runs natively,
    the daemon should see the path.
    """
    if not _DOCKER_RUNS_IN_VM:
        return False
    try:
        if isinstance(host_path, ReachablePath):
            return not host_path.is_under(Path.home())
        host_path.resolve().relative_to(Path.home().resolve())
    except ValueError:
        return True
    return False


def _sanitize_docker_image_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    name = re.sub(r"[^a-z0-9._-]", "-", name)
    return name


def _sanitize_docker_compose_project_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    return name


def _daemon_sees_this_filesystem(env: Any) -> bool:
    """Whether the daemon this sandbox talks to sees this machine's files, so
    a bind source can be resolved here (DET-R2b round 3, M-R2-A1).

    Both the provider (``_HOST_MOUNTS_ARE_LOCAL``, false for remote Docker)
    and the effective connection must say so: no remote daemon, no client
    environment of its own, and the endpoint proof the launch gate uses
    (``daemon_filesystem_local``: no ``DOCKER_HOST``, ``DOCKER_CONTEXT`` or
    saved context, a Linux controller, only known global flags) on the
    connection the sandbox actually uses: ``_docker_args()``, the frozen
    launch's arguments once a launch exists, not ones computed again from
    the environment now (round 4, M-R3-A1). Anything else, or anything
    unknown, is not local.
    """
    from benchflow.task.compose_env import daemon_filesystem_local

    try:
        return bool(
            getattr(env, "_HOST_MOUNTS_ARE_LOCAL", False)
            and not getattr(env, "_remote_daemon", True)
            and env._docker_client_env() is None
            and daemon_filesystem_local(os.environ, env._docker_args())
        )
    except Exception:
        return False


def _unique_compose_project_name(session_id: str) -> str:
    """A Compose project name no other sandbox on the daemon uses.

    The session id is the rollout name, and several callers fix it: branch
    children are ``n<k>``, regrade's verifier ``verifier``, and SDK callers
    pass their own. Two sandboxes with one project
    name share containers and networks: ``start()``'s ``compose down
    --remove-orphans`` and teardown's ``compose down`` of one delete the
    other's. The random suffix keeps concurrent sandboxes (any job, process
    or host on the daemon) apart; the session id stays readable in front.
    """
    return f"{_sanitize_docker_compose_project_name(session_id)}-{uuid.uuid4().hex[:8]}"


def _is_retryable_docker_build_error(message: str) -> bool:
    return bool(_IMAGE_EXPORT_RACE.search(message)) or any(
        pattern.search(message) for pattern in _DOCKER_BUILD_RETRYABLE_ERRORS
    )


def _is_compose_up_network_race_error(message: str) -> bool:
    return is_compose_up_network_race_error(message)


# The labels the leftover sweep (_docker_sweep.py) filters and judges on.
_BENCHFLOW_OWNED_LABEL = OWNED_LABEL
_BENCHFLOW_PROCESS_LABEL = PROCESS_LABEL

# HostConfig keys restore reproduces below. Anything else must hold an unset
# value, so a setting restore cannot rebuild -- including one a newer Docker
# adds -- fails closed instead of vanishing (e.g. a gVisor runtime silently
# restored onto runc).
_REPLAYED_HOST_CONFIG = frozenset(
    {
        "NetworkMode",
        "Tmpfs",
        "RestartPolicy",
        "IpcMode",
        "CapAdd",
        "CapDrop",
        "SecurityOpt",
        "ReadonlyRootfs",
        "NanoCpus",
        "Memory",
        "MemorySwap",
        "MemoryReservation",
        "MemorySwappiness",
        "PidsLimit",
        "ShmSize",
        "Dns",
        "DnsSearch",
        "DnsOptions",
        "Runtime",
        "Init",
        "GroupAdd",
        "OomScoreAdj",
        "CgroupnsMode",
        "LogConfig",
    }
)
# Keys with no setting of their own: mounts are replayed from the container's
# Mounts list, the daemon recomputes masked paths from Privileged/SecurityOpt,
# and the rest only affect the client that created the container.
_DERIVED_HOST_CONFIG = frozenset(
    {
        "Binds",
        "Mounts",
        "MaskedPaths",
        "ReadonlyPaths",
        "ContainerIDFile",
        "ConsoleSize",
    }
)
_UNSET_VALUES: tuple[Any, ...] = (None, False, 0, "", [], {})


def _replayed_run_args(container: dict[str, Any], *, default_network: str) -> list[str]:
    """Replay supported host configuration, rejecting lossy restore before removal.

    Adapted from JeremyJC67's PR #1046. Docker commit preserves
    image configuration, but not mounts, networking, security or resource caps.
    Mounted contents and running processes are not captured by this operation.
    """

    def reject(detail: str) -> NoReturn:
        raise SandboxRestoreHostConfigUnavailable(
            f"Docker snapshot restore cannot preserve {detail}; container left intact"
        )

    host = container.get("HostConfig")
    mounts = container.get("Mounts")
    if not isinstance(host, dict) or not isinstance(mounts, list):
        reject("missing HostConfig or Mounts")
    for key, value in host.items():
        if (
            key not in _REPLAYED_HOST_CONFIG
            and key not in _DERIVED_HOST_CONFIG
            and value not in _UNSET_VALUES
        ):
            reject(f"HostConfig.{key}")
    for mount in host.get("Mounts") or []:
        # Only bind propagation is replayed from the container's Mounts list.
        options = {
            **(mount.get("BindOptions") or {}),
            **(mount.get("VolumeOptions") or {}),
            **(mount.get("TmpfsOptions") or {}),
        }
        options.pop("Propagation", None)
        if mount.get("Target") == BFX_DIR:
            # bfx's volume is never filled from the image (volume-nocopy).
            options.pop("NoCopy", None)
        if any(value not in _UNSET_VALUES for value in options.values()):
            reject(f"mount options for {mount.get('Target')!r}")
    if (host.get("RestartPolicy") or {}).get("Name") not in (None, "", "no"):
        reject("RestartPolicy")
    network = host.get("NetworkMode")
    if (
        not network
        or network in {"host", "default", "bridge"}
        or network.startswith("container:")
    ):
        reject(f"network mode {network!r}")
    networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
    if len(networks) > 1:
        reject("multiple network attachments")
    if network != "none" and network != default_network:
        reject(f"network {network!r} outside the compose project")
    args = ["--network", network]
    for settings in networks.values():
        if settings.get("IPAMConfig"):
            reject("static network addressing")
        for alias in settings.get("Aliases") or []:
            args += ["--network-alias", alias]
    # Only BenchFlow's bfx sits at /.bfx (checked again on the replacement);
    # nothing is replayed beneath it.
    beneath = [
        m.get("Destination")
        for m in mounts
        if isinstance(m, dict)
        and bfx_delivery.at_or_beneath(m.get("Destination"))
        and str(m.get("Destination")).rstrip("/") != BFX_DIR
    ] + [d for d in host.get("Tmpfs") or {} if bfx_delivery.at_or_beneath(d)]
    if beneath:
        reject(f"mounts at or beneath {BFX_DIR} ({', '.join(map(str, beneath))})")
    # Docker lists --tmpfs mounts only in HostConfig.Tmpfs, not in Mounts.
    tmpfs = host.get("Tmpfs") or {}
    for destination, options in tmpfs.items():
        args += ["--tmpfs", destination + (f":{options}" if options else "")]
    for mount in mounts:
        kind, destination = mount.get("Type"), mount.get("Destination")
        if not destination:
            reject("a mount without a destination")
        if kind == "tmpfs":
            if destination not in tmpfs:
                reject("tmpfs options absent from HostConfig")
            continue
        source = mount.get("Name") if kind == "volume" else mount.get("Source")
        if kind not in {"bind", "volume"} or not source:
            reject(f"mount type {kind!r} or missing source")
        if "," in source or "," in destination:
            reject("mount paths containing commas")
        spec = f"type={kind},src={source},dst={destination}"
        if mount.get("RW") is False:
            spec += ",readonly"
        if kind == "volume" and destination == BFX_DIR:
            spec += ",volume-nocopy"
        if kind == "bind" and mount.get("Propagation"):
            spec += f",bind-propagation={mount['Propagation']}"
        args += ["--mount", spec]
    for key, flag in (
        ("CapAdd", "--cap-add"),
        ("CapDrop", "--cap-drop"),
        ("SecurityOpt", "--security-opt"),
        ("GroupAdd", "--group-add"),
    ):
        for value in host.get(key) or []:
            args += [flag, value]
    if host.get("ReadonlyRootfs"):
        args += ["--read-only"]
    if host.get("Init") is not None:
        args += ["--init" if host["Init"] else "--init=false"]
    if host.get("NanoCpus"):
        args += ["--cpus", f"{host['NanoCpus'] / 1_000_000_000:g}"]
    for key, flag in (
        ("Runtime", "--runtime"),
        ("CgroupnsMode", "--cgroupns"),
        ("OomScoreAdj", "--oom-score-adj"),
        ("Memory", "--memory"),
        ("MemorySwap", "--memory-swap"),
        ("MemoryReservation", "--memory-reservation"),
        ("PidsLimit", "--pids-limit"),
        ("ShmSize", "--shm-size"),
    ):
        if host.get(key):
            args += [flag, str(host[key])]
    if host.get("MemorySwappiness") is not None:
        args += ["--memory-swappiness", str(host["MemorySwappiness"])]
    log_config = host.get("LogConfig") or {}
    if log_config.get("Type"):
        args += ["--log-driver", log_config["Type"]]
    for name, value in (log_config.get("Config") or {}).items():
        args += ["--log-opt", f"{name}={value}"]
    if host.get("IpcMode") not in (None, "", "private"):
        reject("non-private IPC")
    for key, flag in (
        ("Dns", "--dns"),
        ("DnsSearch", "--dns-search"),
        ("DnsOptions", "--dns-option"),
    ):
        for value in host.get(key) or []:
            args += [flag, value]
    return args


_TIMEOUT_TEARDOWN_GRACE_SEC = 5


async def _drain_timed_out_process(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool = False,
) -> tuple[bytes | None, bytes | None]:
    """Stop a child that overran its timeout and return whatever it emitted.

    Signalling races the child's own exit: ``asyncio`` reaps as soon as the
    process ends, and ``terminate``/``kill`` on a reaped child raise
    ``ProcessLookupError`` — unlike ``subprocess.Popen``, which polls first and
    swallows the same race (CPython bpo-38630, bpo-40550). Callers here are in
    ``except TimeoutError`` blocks about to raise a description of the timeout,
    so an escaping ``ProcessLookupError`` would replace that description with an
    exception carrying no args at all, and ``_verify_rollout`` would record the
    rollout as ``verifier crashed:`` with nothing after the colon (#1065).
    """

    group_pid: int | None = None
    if process_group and os.name == "posix":
        group_pid = process.pid
        # killpg(0) addresses our own group; libc can implement killpg(1)
        # as the broadcast kill(-1). Never coerce a proxy/mock into either.
        # A new-session child has a concrete PID distinct from our group.
        if type(group_pid) is not int or group_pid <= 1 or group_pid == os.getpgrp():
            raise RuntimeError("invalid Docker client process group identity")
        # Pipes cannot prove child lifetime: buildx may close them before the
        # launcher exits. Stop the entire owned session even in that case.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group_pid, signal.SIGKILL)
    else:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        return await asyncio.wait_for(
            process.communicate(), timeout=_TIMEOUT_TEARDOWN_GRACE_SEC
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            if group_pid is not None:
                os.killpg(group_pid, signal.SIGKILL)
            else:
                process.kill()
        return await process.communicate()


class DockerSandboxEnvVars(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    main_image_name: str
    context_dir: str
    host_verifier_logs_path: ReachablePath | str
    host_agent_logs_path: ReachablePath | str
    host_artifacts_path: ReachablePath | str
    env_verifier_logs_path: str
    env_agent_logs_path: str
    env_artifacts_path: str
    prebuilt_image_name: str | None = None
    cpus: int = 1
    memory: str = "1G"

    def to_env_dict(self, include_os_env: bool = True) -> dict[str, str]:
        return mount_environment(
            self.model_dump(exclude_none=True), include_os_env=include_os_env
        )


class DockerSandbox(BaseSandbox):
    agent_container_no_new_privileges = True
    _DOCKER_COMPOSE_BASE_PATH = COMPOSE_BASE_PATH
    # Whether this daemon's bind sources are paths on this controller: the
    # provider half of the DET-R2b locality proof (_daemon_sees_this_filesystem).
    _HOST_MOUNTS_ARE_LOCAL: ClassVar[bool] = True
    _DOCKER_COMPOSE_BUILD_PATH = COMPOSE_BUILD_PATH
    _DOCKER_COMPOSE_PREBUILT_PATH = COMPOSE_PREBUILT_PATH
    _DOCKER_COMPOSE_NO_NETWORK_PATH = COMPOSE_NO_NETWORK_PATH
    _DOCKER_COMPOSE_NET_ADMIN_PATH = COMPOSE_NET_ADMIN_PATH
    _DOCKER_COMPOSE_KEPT_FREEZER_PATH = COMPOSE_KEPT_FREEZER_PATH

    _image_build_locks: ClassVar[
        weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[tuple[str, str], asyncio.Lock]
        ]
    ] = weakref.WeakKeyDictionary()
    _build_concurrency: ClassVar[int | None] = None
    _build_semaphores: ClassVar[
        weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]
    ] = weakref.WeakKeyDictionary()
    # "building" while a slow image build runs (_build_progress), "built"
    # after it; the rollout's activity cell reads it.
    image_build: str = ""

    # The ``main`` service's container while start() knows it is the one
    # compose would pick: exec() and the copies then call ``docker exec`` and
    # ``docker cp`` on it directly. None sends them through ``docker compose``.
    _main_container: str | None = None

    # Root's steps on a sandbox with the protected kit go through one
    # long-lived command channel per (container, generation, exec user)
    # instead of a docker exec each (_exec_channel). The tests that script
    # Docker's answers turn it off (tests/conftest.py); BENCHFLOW_EXEC_CHANNEL=0
    # turns it off for a run.
    exec_channel: ClassVar[bool] = True
    # Inside a job this sandbox's stop runs in the job's background queue
    # (rollout/_teardown_queue.py). Docker and remote Docker only: a
    # provider sandbox (Daytona, Modal, AgentCore) stops inline, since an
    # extra live one costs quota and money.
    background_stop: ClassVar[bool] = True
    # key -> the channel, the task starting it, or None (it could not start).
    _exec_channels: dict[tuple[str, int, str | None], Any] | None = None
    # The closes of channels of an earlier container or generation.
    _exec_channels_closing: list[asyncio.Future[None]] | None = None
    # Helpers restarted after they died, this sandbox.
    _exec_channel_restarts: int | None = None

    # Batch verifier preparation between the host-side trust decisions.
    verifier_prep_script: ClassVar[bool] = True
    # Independent read steps before the verifier may run side by side
    # (sandbox/_in_order.py): each is its own docker exec client or channel
    # request. The tests that script Docker's answers turn it off.
    concurrent_steps: ClassVar[bool] = True

    # A single-service task on a local daemon starts and stops its main
    # container through the Engine API, with the bodies Compose sends
    # (_direct_docker, PERF-E). BENCHFLOW_DIRECT_DOCKER=0 keeps
    # every trial on Compose; the tests that script Docker's answers turn it
    # off (tests/conftest.py).
    direct_docker: ClassVar[bool] = True
    # The project made through the API, from the prototype until teardown.
    _direct: _direct_docker.DirectProject | None = None
    # The plan the start decided on, and the printed project it came from.
    _direct_pending: tuple[_direct_docker.DirectProject, dict] | None = None
    # How this sandbox's last start made its containers, for config.json:
    # {"path": "direct"} or {"path": "compose", "reason": ...}.
    direct_docker_record: dict[str, str] | None = None
    # The bootstrap overlay of the current direct start.
    _direct_overlay: dict | None = None
    # Compose's version (its core, as Compose labels what it makes) per
    # client flags and PATH.
    _compose_versions: ClassVar[dict[tuple[str, ...], str]] = {}

    @classmethod
    def set_build_concurrency(cls, n: int) -> None:
        """Limit how many sandboxes go through the docker startup phase
        (build + compose down --remove-orphans + compose up --wait) in parallel.

        Default is unlimited. Setting this is critical when --concurrency is
        high (e.g. 60): otherwise N tasks all hammer the docker daemon at once,
        causing build/network creation races and `docker container prune`
        timeouts. Agent execution after the container is up is NOT gated.
        """
        cls._build_concurrency = n
        cls._build_semaphores.clear()

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is not installed or not on PATH. "
                "Please install Docker and try again."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(
                "Docker daemon is not running. Please start Docker and try again."
            ) from exc

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        keep_containers: bool = False,
        mounts_json: list[dict[str, str]] | None = None,
        *args: Any,
        staged_environment: StagedEnvironment | None = None,
        verifier_service: str = "main",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._keep_containers = keep_containers
        # --mount sources are operator paths. Pin their host realpaths before
        # writing the Compose override; keep named volumes as volume names.
        self._mounts_json = (
            [
                {**mount, "source": str(Path(mount["source"]).resolve())}
                if mount.get("type") == "bind" and mount.get("source")
                else dict(mount)
                for mount in mounts_json
            ]
            if mounts_json is not None
            else None
        )
        # The private copy of the task's environment folder this sandbox
        # builds and launches from (environment_dir is its folder), and the
        # checked Compose project every Compose call uses once frozen
        # (freeze_launch).
        self._staged = staged_environment
        self._launch: ComposeLaunch | None = None
        self._bind_baseline: _bind_targets.BindTargets | None = None
        # Per service, the targets whose source is this sandbox's private
        # task copy, read at the same first start (DET-R2b).
        self._staged_bind_targets: dict[str, frozenset[str]] | None = None
        self._mounts_compose_path: Path | None = None
        self._container_env_compose_path: Path | None = None
        self._runtime_compose_path: Path | None = None
        self._build_compose_path: Path | None = None
        # bfx at /.bfx (TRUSTED-RUNTIME.md §6): how it reaches the container
        # (``bind`` or ``volume``; None delivers nothing and the kit path
        # stays), for which architecture, and the overlay that mounts it
        # read-only, decided at start before the launch is frozen.
        self.bfx_delivery: str | None = None
        self._bfx_arch: str | None = None
        self._bfx_compose_path: Path | None = None
        self._bfx_source: str | None = None
        # The subnets BenchFlow chose for this sandbox's networks
        # (_docker_subnets.py), held from start() until stop() removed them,
        # and the compose override that names them.
        self._subnet_allocator: SubnetAllocator | None = None
        self._subnets: dict[str, ipaddress.IPv4Network] = {}
        self._subnets_compose_path: Path | None = None
        # The task's compose file could not be read for networks
        # (``include``, or it does not parse): the daemon chose them all.
        self._subnet_plan_unreadable = False
        # The IPv4 subnets the task's own networks pin (their ``ipam``).
        self._subnet_pinned: list[ipaddress.IPv4Network] = []
        self._logs_are_mounted = True
        # The Compose service the task's verifier runs in. A task whose
        # verifier runs elsewhere (robohub's `verifier.service: simulator`)
        # may drop the verifier-log mount from main on purpose.
        self._verifier_service = verifier_service

        verifier_dir = (
            rollout_paths.verifier_dir
            if rollout_paths
            else str(Path("/tmp/verifier").resolve())
        )
        agent_dir = (
            rollout_paths.agent_dir
            if rollout_paths
            else str(Path("/tmp/agent").resolve())
        )
        artifacts_dir = (
            rollout_paths.artifacts_dir
            if rollout_paths
            else str(Path("/tmp/artifacts").resolve())
        )

        self._env_vars = DockerSandboxEnvVars(
            # A basename is shared by unrelated checkouts and processes. Keep
            # the build's tag private through layer selection and container up.
            main_image_name=private_image_repository(self.compose_project_name),
            context_dir=str(self.environment_dir.resolve().absolute()),
            host_verifier_logs_path=verifier_dir,
            host_agent_logs_path=agent_dir,
            host_artifacts_path=artifacts_dir,
            env_verifier_logs_path=str(SandboxPaths.verifier_dir),
            env_agent_logs_path=str(SandboxPaths.agent_dir),
            env_artifacts_path=str(SandboxPaths.artifacts_dir),
            prebuilt_image_name=task_env_config.docker_image,
            cpus=task_env_config.cpus,
            memory=f"{task_env_config.memory_mb}M",
        )
        self._use_prebuilt = False
        # Set just before `compose up`: from then on containers may exist and
        # stop() must tear them down; before it, nothing was started.
        self._compose_up_attempted = False
        # Custom Compose builds record their exact resolved image/build.tags
        # before startup; a prefix or "only tag" guess cannot prove ownership
        # of an operator's alias. The built-in main image is known exactly.
        self._built_image_tags: set[str] = (
            set()
            if self._uses_compose
            else {f"{self._env_vars.main_image_name}:latest"}
        )
        # Only Compose's generated names belong to the unique trial project.
        # Explicit image/build.tags names can be shared by other trials.
        self._private_built_image_tags: set[str] = set(self._built_image_tags)
        self._recovery_baseline: DockerRecoveryBaseline | None = None
        # Branch snapshot images still used by a container when their fork
        # finished; stop() removes them once the containers are gone.
        self._deferred_snapshot_refs: set[str] = set()
        # Credential files scrubbed out of each snapshot image, held in host
        # memory so restore() can put them back (see _snapshot_credentials).
        self._snapshot_credentials: dict[str, list[StashedCredential]] = {}
        # The cached layers (env_layers): use_agent_layer() and
        # use_proxy_layer() name the recipes before start(), which starts
        # ``main`` from them (the agent's on top of the proxy's).
        self._agent_layer_recipe: AgentLayerRecipe | None = None
        self._agent_layer: LayerResult | None = None
        self._proxy_layer_recipe: ProxyLayerRecipe | None = None
        self._proxy_layer: LayerResult | None = None
        self._layers_base_ref: str | None = None
        # The opt-in separate verifier image cache (verifier_layers).
        # Shared grading never requests it; no tool state is baked or seeded.
        self._verifier_layer_recipe: VerifierLayerRecipe | None = None
        # The cloud metadata block's rules come from this helper container in
        # main's network namespace when the run arms the block (_metadata_helper).
        self._metadata_helper: _metadata_helper.MetadataHelper | None = None
        self._retired_metadata_helpers: list[_metadata_helper.MetadataHelper] = []
        self._verifier_layer: VerifierLayer | None = None
        # This trial's ``layers`` record by kind, for config.json.
        self.layer_records: dict[str, dict[str, Any]] = {}

        self._compose_task_env: dict[str, str] = {}
        if task_env_config.env and self._uses_compose:
            self._compose_task_env = resolve_env_vars(task_env_config.env)

        resolved_task_keys = set(self._compose_task_env.keys()) | set(
            self._persistent_env.keys()
        )
        if resolved_task_keys:
            benchflow_keys = set(
                self._env_vars.to_env_dict(include_os_env=False).keys()
            )
            collisions = benchflow_keys & resolved_task_keys
            if collisions:
                self.logger.warning(
                    "BenchFlow sets these Compose variables itself; the task's "
                    "value is not used for Compose: %s",
                    ", ".join(sorted(collisions)),
                )

    @property
    def compose_project_name(self) -> str:
        """This sandbox's own Compose project (:func:`_unique_compose_project_name`).

        Fixed on first use, so every compose, ``docker`` and live-process
        call of the sandbox names the same project.
        """
        project = getattr(self, "_compose_project", None)
        if project is None:
            project = _unique_compose_project_name(self.session_id)
            self._compose_project = project
        return project

    @property
    def _uses_compose(self) -> bool:
        return self._environment_docker_compose_path.exists()

    @property
    def is_mounted(self) -> bool:
        return self._logs_are_mounted

    async def _probe_verifier_log_mount(self) -> None:
        """Confirm that the daemon can see the host verifier-log bind mount.

        Docker Desktop normally translates WSL paths before they reach its VM.
        A client connected directly to the underlying daemon can bypass that
        translation: Compose still accepts the bind mount, but the container
        and Benchflow then write to different directories.  Treat that case as
        non-mounted so the verifier clears and downloads its remote outputs.

        The gap is logged at debug level where it is expected: a task whose
        verifier runs in another Compose service (``verifier.service``), and a
        Docker VM that does not share the directory (Colima on a Mac with the
        job outside ``$HOME``). Anywhere else it stays a warning.
        """
        if self.rollout_paths is None:
            return

        probe_name = f".benchflow-mount-probe-{uuid.uuid4().hex}"
        host_probe = self.rollout_paths.verifier_dir / probe_name
        sandbox_probe = SandboxPaths.verifier_dir / probe_name
        host_probe.parent.mkdir(parents=True, exist_ok=True)
        host_probe.write(b"")
        try:
            result = await exec_raw(
                self.exec,
                f"test -f {shlex.quote(str(sandbox_probe))}",
                user="root",
                timeout_sec=10,
            )
            self._logs_are_mounted = result.return_code == 0
        finally:
            host_probe.unlink(missing_ok=True)

        if self._logs_are_mounted:
            return
        verifier_dir = self.rollout_paths.verifier_dir
        verifier_service = getattr(self, "_verifier_service", "main")
        if verifier_service != "main":
            # Expected: the task runs its verifier in another service and
            # leaves main without the mount (robohub drops it with Compose's
            # `!override`), so the DOCKER_HOST hint would be wrong (REPORT B14).
            self.logger.debug(
                "Verifier-log directory %s is not mounted in main; the task runs "
                "its verifier in service %r. Verifier outputs will be copied back "
                "from the container instead.",
                verifier_dir,
                verifier_service,
            )
            return
        if _vm_does_not_share(verifier_dir):
            # Expected (Colima on a Mac, jobs outside $HOME) and harmless.
            self.logger.debug(
                "Verifier-log directory %s is outside $HOME, which the Docker VM "
                "(Colima by default) does not share; verifier outputs will be "
                "copied back from the container instead.",
                verifier_dir,
            )
            return
        self.logger.warning(
            "Docker verifier-log bind mount is not visible inside the "
            "container; verifier outputs will be copied back explicitly. The "
            "daemon cannot see %s, so it may be remote or reached without path "
            "translation (check DOCKER_HOST and `docker context ls`).",
            verifier_dir,
        )

    def _docker_client_env(self) -> dict[str, str] | None:
        """Environment for raw ``docker`` subprocesses; None inherits the caller's.

        The remote provider returns an environment pointed at its host.
        """
        return None

    def _docker_layers(self) -> DockerLayers:
        """One provider boundary for every layer and pull-provenance call."""
        return DockerLayers(daemon=self._docker_subnet_key())

    def _docker_daemon_display(self) -> str:
        """A diagnostic label; docker info's machine data stays internal."""
        return "Docker daemon"

    def _is_retryable_build_error(self, message: str) -> bool:
        return _is_retryable_docker_build_error(message)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _pre_compose_hook_path(self) -> Path:
        return self.environment_dir / "benchflow-pre-compose.sh"

    @property
    def _docker_compose_paths(self) -> list[Path]:
        launch = getattr(self, "_launch", None)
        if launch is not None:
            paths = [launch.path]
        else:
            paths = compose_file_paths(
                self.environment_dir,
                self.task_env_config,
                base=self._DOCKER_COMPOSE_BASE_PATH,
                prebuilt=self._use_prebuilt,
                mounts=self._mounts_compose_path,
            )
            paths.extend(self._runtime_compose_inputs())
        # This private BenchFlow overlay contains only allocated IPAM subnets;
        # the task's services and host sources remain in the frozen project.
        subnets_path = getattr(self, "_subnets_compose_path", None)
        if subnets_path:
            paths.append(subnets_path)
        # Names the volumes Docker would make anonymous (_trial_volumes).
        volumes_path = getattr(self, "_trial_volumes_compose_path", None)
        if volumes_path:
            paths.append(volumes_path)
        bootstrap = getattr(self, "_bootstrap_override", None)
        if bootstrap is not None:
            paths.append(bootstrap)
        return paths

    def _runtime_compose_inputs(self) -> list[Path]:
        paths = [
            path
            for name in ("_container_env_compose_path",)
            if (path := getattr(self, name, None)) is not None
        ]
        # Decided at start, before the launch is frozen; part of
        # the frozen project like every other input. Before the build
        # overlay: that overlay's security_opt (!override, read from a
        # project that already holds this file's writable-cgroups) is the
        # last word, so no later file appends that option a second time,
        # which Compose 2.40 refuses ("security_opt items ... are equal").
        if compose_needs_kept_freezer(self):
            paths.append(self._DOCKER_COMPOSE_KEPT_FREEZER_PATH)
        paths.extend(
            path
            for name in (
                "_build_compose_path",
                "_runtime_compose_path",
                "_bfx_compose_path",
            )
            if (path := getattr(self, name, None)) is not None
        )
        return paths

    def _write_prebuilt_compose_file(self) -> None:
        # start() owns the overlay from here (see freeze_launch's early one).
        self._prebuilt_overlay_early = False
        image = None
        if self._use_prebuilt and self._uses_compose:
            image = self._env_vars.prebuilt_image_name
            assert image
        self._write_prebuilt_overlay(image)

    def _write_prebuilt_overlay(self, image: str | None) -> None:
        """Main's prebuilt overlay for *image* (none for None), kept when it
        already holds the same bytes."""
        from benchflow.sandbox._compose_runtime import prebuilt_main_override

        path = getattr(self, "_runtime_compose_path", None)
        override = None if image is None else prebuilt_main_override(image)
        if path is not None:
            # The same overlay again (start() writes it twice, before and
            # after the cached layers, which often leave main's image as it
            # was): keep the file, so the launch key and the frozen project
            # stay the same and freeze_launch does not resolve and check the
            # same project a second time (PERF-D). Any change writes a new one.
            with contextlib.suppress(OSError):
                if override is not None and path.read_text() == override:
                    return
            self._drop_runtime_compose_file()
        if override is None:
            return
        fd, name = tempfile.mkstemp(prefix="benchflow-prebuilt-", suffix=".yaml")
        with os.fdopen(fd, "w") as stream:
            stream.write(override)
        self._runtime_compose_path = Path(name)
        # The trial factory writes it before start() (freeze_launch's early
        # overlay): a sandbox dropped without stop() still removes it.
        self._runtime_compose_finalizer = weakref.finalize(self, _unlink_quietly, name)

    def _drop_runtime_compose_file(self) -> None:
        path = getattr(self, "_runtime_compose_path", None)
        if path is not None:
            path.unlink(missing_ok=True)
            self._runtime_compose_path = None
        finalizer = getattr(self, "_runtime_compose_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
            self._runtime_compose_finalizer = None

    def _write_container_env_compose_file(self) -> None:
        """Pass explicit task/manifest values to PID 1, before it initializes.

        The Compose client's ambient environment is interpolation input, not
        container configuration. Only the task's explicit values cross here;
        dollar signs are escaped so Compose preserves their literal bytes.
        """
        values = {**self._compose_task_env, **self._persistent_env}
        if not values:
            path = self._container_env_compose_path
            if path is not None:
                path.unlink(missing_ok=True)
                self._container_env_compose_path = None
            return
        path = self._container_env_compose_path
        if path is None:
            fd, name = tempfile.mkstemp(
                prefix="benchflow-container-env-", suffix=".json"
            )
            os.close(fd)
            path = self._container_env_compose_path = Path(name)
        path.write_text(
            json.dumps(
                {
                    "services": {
                        "main": {
                            "environment": {
                                key: value.replace("$", "$$")
                                for key, value in values.items()
                            }
                        }
                    }
                }
            )
        )

    def _docker_compose_env(self) -> dict[str, str]:
        from benchflow.task.compose_env import compose_process_env

        # The hermetic environment the launch gate's `docker compose config`
        # checked the task with (compose_process_env): PATH, HOME and the
        # docker connection variables of the operator's, the variables they
        # named, the task's values (never one that steers a program Compose
        # runs here, such as LD_PRELOAD or PASSWORD_STORE_GPG_OPTS), then
        # BenchFlow's own. No operator credential is in it, so no file
        # Compose reads (include, extends, .env, env_file) can copy one.
        # BenchFlow's after the task's: a task's HOST_ARTIFACTS_PATH or
        # CONTEXT_DIR would have docker-compose-base.yaml bind-mount, or the
        # build copy, a host folder the task chose. The process token last,
        # so no task variable can hide which process owns the containers and
        # networks (the leftover sweep, _docker_sweep.py).
        return compose_process_env(
            {**self._compose_task_env, **self._persistent_env},
            {
                **self._env_vars.to_env_dict(include_os_env=False),
                PROCESS_ENV: process_token(),
            },
        )

    @property
    def compose_project_dir(self) -> Path:
        """The ``--project-directory`` of every Compose call: the frozen
        launch's own empty folder once frozen (no task ``.env`` is read
        again), else the environment folder."""
        launch = getattr(self, "_launch", None)
        if launch is not None:
            return launch.project_dir
        return self.environment_dir.resolve().absolute()

    def _docker_compose_global_args(self) -> list[str]:
        """The docker CLI's flags for the daemon (``docker <flags> compose``):
        the operator's DOCKER_HOST and DOCKER_CONTEXT, given on the command
        line so no Compose file can read them, and the client configuration
        without the operator's proxies (``--config``)."""
        from benchflow.task.compose_env import (
            docker_connection_args,
            hermetic_docker_config,
        )

        return docker_connection_args(
            {**os.environ, "DOCKER_CONFIG": hermetic_docker_config()}
        )

    #: Whether the daemon runs on another machine (remote-docker): every
    #: bind is a path there.
    _remote_daemon: ClassVar[bool] = False
    #: Its agents reach the model through a host proxy, which the rollout may
    #: start while this sandbox starts (Rollout._reserve_host_proxy).
    starts_host_proxy_ahead: ClassVar[bool] = True

    def _stage(self) -> StagedEnvironment:
        """The private copy of the environment folder, made now for a
        sandbox constructed without one (``_create_sandbox_environment``
        makes it before the sandbox)."""
        if self._staged is None:
            from benchflow.task.compose_env import host_access_allowed
            from benchflow.task.env import TaskEnvNotAllowedError
            from benchflow.task.staging import StagingRefused, stage_environment

            try:
                self._staged = stage_environment(
                    self.environment_dir, allow_outside=host_access_allowed()
                )
            except StagingRefused as exc:
                raise TaskEnvNotAllowedError(
                    {self.environment_name: exc.refused}
                ) from None
            self.environment_dir = self._staged.environment_dir
            self._env_vars.context_dir = str(self.environment_dir)
        return self._staged

    def trust_host_mount_sources(self) -> list[str]:
        """Every host path this sandbox binds into a container, for the trust
        ledger's isolation check: the operator's ``--mount`` bind sources and
        the frozen launch's binds. Raises when the frozen launch cannot be
        read."""
        sources = [
            str(mount["source"])
            for mount in getattr(self, "_mounts_json", None) or ()
            if isinstance(mount, dict)
            and mount.get("source")
            and (
                mount.get("type") == "bind"
                or (not mount.get("type") and str(mount["source"]).startswith("/"))
            )
        ]
        launch = getattr(self, "_launch", None)
        if launch is not None:
            sources.extend(launch.bind_sources())
        return sources

    def freeze_launch(self, force_build: bool = False) -> None:
        """Check the task's Compose project as this sandbox will run it and
        freeze it: the run's files, project name, values and
        docker flags, resolved once on the private copy, checked, written to
        a private file that every later Compose call of the sandbox uses
        with the same environment. Raises
        :class:`~benchflow.task.env.TaskEnvNotAllowedError` before anything
        is built or started. A second call with the same inputs keeps the
        frozen project; other inputs (``force_build`` changes the files)
        make and check another."""
        if not self._uses_compose:
            return
        if (
            not force_build
            and self._recovery_baseline is None
            and self.task_env_config.docker_image
            and getattr(self, "_runtime_compose_path", None) is None
            and self._env_vars.prebuilt_image_name
        ):
            # A check before start() (the trial factory's): give it main's
            # prebuilt overlay now, the same bytes start() writes before its
            # first freeze in this (the usual) case, so that freeze keeps the
            # project checked here instead of resolving and checking it again
            # with only the overlay added (PERF-D). A forced build or a
            # cached layer changes the overlay, and start() checks again.
            self._write_prebuilt_overlay(self._env_vars.prebuilt_image_name)
            self._prebuilt_overlay_early = True
        elif force_build and getattr(self, "_prebuilt_overlay_early", False):
            # The early overlay was for a prebuilt main; a forced build
            # launches none (start() drops it the same way before its freezes).
            self._write_prebuilt_overlay(None)
            self._prebuilt_overlay_early = False
        from benchflow.task.compose_env import (
            _BENCHFLOW_REMEDY,
            BENCHFLOW_COMPOSE_VARIABLES,
            daemon_filesystem_local,
            freeze_compose,
            host_access_allowed,
            launch_key,
            project_check,
        )
        from benchflow.task.env import (
            TaskEnvNotAllowedError,
            TaskEnvRequest,
            allowed_task_env,
        )

        staged = self._stage()
        prebuilt = (
            self._recovery_baseline is not None
            or self._use_prebuilt
            or (not force_build and bool(self.task_env_config.docker_image))
        )
        if self._mounts_json and self._mounts_compose_path is None:
            self._mounts_compose_path = self._write_mounts_compose_file()
        files = [
            path.resolve().absolute()
            for path in compose_file_paths(
                self.environment_dir,
                self.task_env_config,
                base=self._DOCKER_COMPOSE_BASE_PATH,
                prebuilt=prebuilt,
                mounts=self._mounts_compose_path,
            )
        ]
        files.extend(
            path.resolve().absolute() for path in self._runtime_compose_inputs()
        )
        transport = self._compose_transport_env()
        env = {**self._docker_compose_env(), **transport}
        docker_args = self._docker_compose_global_args()
        project = self.compose_project_name
        key = launch_key(files, env, docker_args, project, self.environment_dir)
        if self._launch is not None and self._launch.key == key:
            return
        self._launch = None
        provided = {**self._compose_task_env, **self._persistent_env}
        names = allowed_task_env()
        host_access = host_access_allowed()
        refused = [
            TaskEnvRequest(
                "sandbox.env",
                "",
                None,
                f"sandbox.env sets {name}, which BenchFlow sets for Compose itself",
                remedy=_BENCHFLOW_REMEDY,
            )
            for name in sorted(BENCHFLOW_COMPOSE_VARIABLES & set(provided))
        ]
        folders = [
            # ReachablePath keeps host I/O on held handles. Admission needs
            # the same checked mount strings serialized for Compose above,
            # not the handles' deliberately non-path diagnostic spelling.
            env["HOST_VERIFIER_LOGS_PATH"],
            env["HOST_AGENT_LOGS_PATH"],
            env["HOST_ARTIFACTS_PATH"],
            # The operator's own --mount sources.
            *(
                str(m.get("source"))
                for m in self._mounts_json or ()
                if isinstance(m, dict) and m.get("source")
            ),
        ]
        result = freeze_compose(
            staged=staged,
            files=files,
            project_name=project,
            env=env,
            docker_args=docker_args,
            check=None
            if host_access
            else project_check(
                staged,
                folders,
                names=names,
                environ=os.environ,
                remote=self._remote_daemon
                or not daemon_filesystem_local(os.environ, docker_args),
                agent="SSH_AUTH_SOCK" in transport and "SSH_AUTH_SOCK" not in names,
                project=project,
                read_only_binds=self._bfx_read_only_binds(),
                read_only_volumes=self._bfx_read_only_volumes(),
            ),
            host_access=host_access,
            provided=provided,
            names=names,
            transport=transport,
            operator_environ=os.environ,
            # Every service the launch creates can run agent code: each gets
            # no-new-privileges in the frozen project, in ordinary, cached and
            # recovery starts alike (the base files no longer set it).
            agent_security=True,
        )
        refused += result.refused
        if refused or result.launch is None:
            raise TaskEnvNotAllowedError({self.environment_name: refused})
        self._launch = result.launch

    def _compose_transport_env(self) -> dict[str, str]:
        """What the transport to the daemon needs in Compose's environment
        beyond :meth:`_docker_compose_env` (remote-docker's ssh agent);
        never given to the pre-compose hook."""
        return {}

    async def _prepare_bfx(self) -> None:
        """Decide how bfx reaches ``/.bfx`` and write the overlay that mounts
        it read-only into ``main`` when the container is created (a bind of
        a host folder holding only the verified binary on a daemon that sees
        this machine's files, else the harness-owned volume). Without a
        pinned binary for the daemon's architecture nothing is delivered and
        the kit path stays."""
        self.bfx_delivery = None
        previous, self._bfx_compose_path = (
            getattr(self, "_bfx_compose_path", None),
            None,
        )
        if previous is not None:
            previous.unlink(missing_ok=True)
        local = _daemon_sees_this_filesystem(self)
        mode = bfx_delivery.docker_mode(local_daemon=local)
        if mode is None:
            return
        # No Docker call: a daemon that sees this machine's files runs on
        # this machine's kernel; a remote one's architecture came with the
        # host check (``docker info``) before start.
        machine = (
            platform.machine()
            if local
            else getattr(self, "_daemon_architecture", "") or "unknown"
        )
        try:
            arch = normalize_arch(machine)
            folder = await asyncio.to_thread(bfx_delivery.host_dir, arch)
        except BfxUnavailable as exc:
            self.logger.debug("bfx: not delivered (%s)", exc)
            return
        source = (
            str(folder) if mode == bfx_delivery.BIND else bfx_delivery.volume_name(arch)
        )
        fd, name = tempfile.mkstemp(prefix="benchflow-bfx-", suffix=".json")
        with os.fdopen(fd, "w") as stream:
            stream.write(bfx_delivery.compose_overlay(mode, source=source))
        self._bfx_compose_path = Path(name)
        weakref.finalize(self, _unlink_quietly, name)
        self._bfx_arch = arch
        self._bfx_source = source
        self.bfx_delivery = mode

    def _bfx_read_only_binds(self) -> list[tuple[str, str]]:
        """bfx's read-only bind, for the launch check to admit (only that
        source, only at /.bfx, only read-only)."""
        source = getattr(self, "_bfx_source", None)
        if getattr(self, "bfx_delivery", None) == bfx_delivery.BIND and source:
            return [(source, BFX_DIR)]
        return []

    def _bfx_read_only_volumes(self) -> list[tuple[str, str]]:
        """bfx's harness volume, for the launch check to admit (only that
        name, only read-only and uncopied at /.bfx)."""
        source = getattr(self, "_bfx_source", None)
        if getattr(self, "bfx_delivery", None) == bfx_delivery.VOLUME and source:
            return [(source, BFX_DIR)]
        return []

    def _check_bfx_mounts(self, inspected: dict[str, Any]) -> None:
        """main as created (its inspect, from the bind proof every start and
        restore takes before any task process or bfx runs): nothing but
        BenchFlow's own read-only helper mount sits at or beneath /.bfx,
        whatever put it there (bfx_delivery.mount_problems)."""
        mode = getattr(self, "bfx_delivery", None)
        if mode is None:
            # No bfx runs here; the launch check still refuses such mounts.
            return
        problems = bfx_delivery.mount_problems(
            inspected, mode=mode, source=getattr(self, "_bfx_source", None)
        )
        if problems:
            raise RuntimeError(
                f"main has mounts at or beneath {BFX_DIR}, where only "
                f"BenchFlow's read-only bfx may be: {'; '.join(problems)}"
            )

    async def _fill_bfx_volume(self) -> None:
        """Before any container starts: the volume the overlay names holds
        the pinned bfx (filled and checked once per daemon)."""
        if (
            getattr(self, "bfx_delivery", None) != bfx_delivery.VOLUME
            or self._bfx_arch is None
        ):
            return
        try:
            await bfx_delivery.ensure_volume(
                ["docker", *self._docker_args()],
                self._docker_process_env(),
                arch=self._bfx_arch,
                daemon_key=repr(self._docker_subnet_key()),
            )
        except BfxUnavailable as exc:
            raise RuntimeError(
                f"bfx: {exc} (set {bfx_delivery.DELIVERY_ENV}=off to start without it)"
            ) from exc

    def _write_mounts_compose_file(self) -> Path:
        compose = {"services": {"main": {"volumes": self._mounts_json}}}
        assert self.rollout_paths is not None
        path = self.rollout_paths.rollout_dir / "docker-compose-mounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compose, indent=2))
        return path

    async def _docker_cli_unchecked(self, args: list[str]) -> ExecResult:
        result = await self._docker_cli(args, check=False, raw_output=True)
        # Subnet discovery parses stdout only. Its error formatter truncates
        # stderr, so sanitize that diagnostic copy at this boundary first.
        return ExecResult(
            stdout=result.stdout,
            stderr=self._diagnostic_text(result.stderr or ""),
            return_code=result.return_code,
        )

    def _diagnostic_text(self, text: str) -> str:
        """Format a copy for display; machine-output consumers keep the original."""
        return text

    def _docker_subnet_key(self) -> DockerDaemon:
        if not hasattr(self, "_daemon_connection"):
            self._daemon_connection = daemon_key(self._docker_client_env())
        return self._daemon_connection

    def _prefetch_subnet_allocator(self) -> None:
        """Start building this daemon's subnet allocator now, in the
        background, when the task's networks will want subnets of their own.

        allocator_for shares one build per daemon connection and loop, so the
        allocate in _choose_subnets waits for this same read instead of
        starting another; nothing is chosen or reserved here. The read is the
        one _choose_subnets would make a moment later (the allocator already
        keeps one view of the daemon for the whole process, and a refused
        subnet refreshes it). Any failure is left for _choose_subnets to
        report, as before.
        """

        async def prefetch() -> None:
            with contextlib.suppress(Exception):
                plan = plan_networks(
                    self._environment_docker_compose_path
                    if self._uses_compose
                    else None,
                    main_networked=bool(self.task_env_config.allow_internet),
                    compose_env=self._docker_compose_env(),
                )
                if plan is None or not plan.needs:
                    return
                with contextlib.suppress(SubnetConfigError):
                    await self._docker_subnet_key().identify()
                await allocator_for(
                    self._docker_subnet_key(),
                    self._docker_cli_unchecked,
                    daemon_display=self._docker_daemon_display(),
                )

        self._subnet_prefetch = asyncio.get_running_loop().create_task(prefetch())

    async def _choose_subnets(self, *, refresh: bool = False) -> None:
        """Give this sandbox's networks small subnets of their own.

        BenchFlow picks a /28 per network (``_docker_subnets.py``) instead of
        the daemon's whole pool chunk, so a stock daemon holds thousands of
        trial networks rather than about 30. Nothing is chosen when the
        daemon chooses itself (it is not on this host and
        ``BENCHFLOW_DOCKER_SUBNETS`` is unset) or the task's compose file
        cannot be read. ``refresh`` re-reads the daemon's networks first
        (after it refused a subnet).
        """
        self._release_subnets()
        prefetch = getattr(self, "_subnet_prefetch", None)
        if prefetch is not None:
            # The read start() began (_prefetch_subnet_allocator); its
            # failures are this method's to report, below.
            self._subnet_prefetch = None
            with contextlib.suppress(Exception):
                await prefetch
        plan = plan_networks(
            self._environment_docker_compose_path if self._uses_compose else None,
            main_networked=bool(self.task_env_config.allow_internet),
            compose_env=self._docker_compose_env(),
        )
        self._subnet_plan_unreadable = plan is None
        self._subnet_pinned = list(plan.pinned) if plan is not None else []
        # Resolve once through the originating client before synchronous subnet
        # consumers use the shared identity. The endpoint never selects a backend.
        if self._subnet_pinned or (plan is not None and plan.needs):
            # The allocator refuses if any pinned reservations exist.
            with contextlib.suppress(SubnetConfigError):
                await self._docker_subnet_key().identify()
        # Kept clear of every subnet this process chooses, for any trial.
        pin_subnets(self._subnet_pinned, key=self._docker_subnet_key())
        if plan is None or not plan.needs:
            return
        needs, pinned = plan.needs, plan.pinned
        try:
            allocator = await allocator_for(
                self._docker_subnet_key(),
                self._docker_cli_unchecked,
                daemon_display=self._docker_daemon_display(),
            )
        except ValueError as exc:
            raise SandboxStartupError(str(exc)) from None
        if allocator is None:
            return
        if refresh:
            allocator.set_taken(await read_taken(self._docker_cli_unchecked))
            allocator.step_aside()
        try:
            subnets = allocator.allocate(needs, avoid=pinned)
        except SubnetRangeExhausted:
            # This process may hold a stale list naming networks gone since.
            allocator.set_taken(await read_taken(self._docker_cli_unchecked))
            try:
                subnets = allocator.allocate(needs, avoid=pinned)
            except SubnetRangeExhausted as exc:
                raise SandboxStartupError(str(exc)) from None
        self._subnet_allocator = allocator
        self._subnets = dict(subnets)
        # Outside the rollout folder: a separate verifier sandbox shares it.
        fd, name = tempfile.mkstemp(
            prefix=f"benchflow-{self.compose_project_name}-networks-",
            suffix=".json",
        )
        with os.fdopen(fd, "w") as f:
            json.dump(override_document(subnets), f, indent=2)
        self._subnets_compose_path = Path(name)
        self.logger.debug(
            "Docker networks of %s: %s",
            self.compose_project_name,
            ", ".join(f"{k} {v}" for k, v in subnets.items()),
        )

    def _release_subnets(self) -> None:
        """Hand this sandbox's subnets back once its networks are gone."""
        allocator = getattr(self, "_subnet_allocator", None)
        subnets = getattr(self, "_subnets", None)
        if allocator is not None and subnets:
            allocator.release(subnets.values())
        self._subnets = {}
        path = getattr(self, "_subnets_compose_path", None)
        if path is not None:
            path.unlink(missing_ok=True)
        self._subnets_compose_path = None

    async def _name_trial_volumes(self) -> None:
        """Before Compose creates any container: read what gives the trial's
        containers a volume (the project's services, the ``VOLUME``s of their
        images, the mounts of an outside ``volumes_from`` source) and write
        the overlay that names and labels each volume Docker would otherwise
        make anonymous (:mod:`_trial_volumes`). One ``image inspect`` per
        image; an image not on the daemon yet, or one a service pulls
        ``always``, is pulled first, as Compose's create would. Anything that
        cannot be read stops the start."""
        self._drop_trial_volumes_overlay()
        self._trial_volume_inputs = None
        launch = getattr(self, "_launch", None)
        try:
            if launch is not None:
                project = json.loads(Path(launch.path).read_text())
            else:
                printed = await self._run_docker_compose_command(
                    ["config", "--format", "json"], raw_output=True, merge_stderr=False
                )
                project = json.loads(printed.stdout or "")
            services = project["services"]
            images = {name: _trial_volumes.image_of(project, name) for name in services}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SandboxStartupError(
                f"Cannot read the Compose project to name its volumes: {exc}"
            ) from None

        async def declared(image: str) -> list[str] | None:
            read = await self._docker_cli(
                ["image", "inspect", "--format", "{{json .Config.Volumes}}", image],
                check=False,
                raw_output=True,
            )
            if read.return_code != 0:
                return None
            with contextlib.suppress(ValueError, AttributeError):
                return sorted(json.loads(read.stdout or "null") or {})
            return None

        # A policy that may pull a newer image at create (always, daily, ...).
        pull = sorted(
            name
            for name, spec in services.items()
            if (spec or {}).get("pull_policy")
            not in (None, "missing", "if_not_present", "never", "build")
        )
        distinct = sorted(set(images.values()))
        found = dict(
            zip(distinct, await asyncio.gather(*map(declared, distinct)), strict=True)
        )
        pull += [
            name
            for name, image in images.items()
            if found[image] is None and name not in pull
        ]
        pulled = None
        if pull:
            pulled = await self._run_docker_compose_command(
                ["pull", "--quiet", "--ignore-buildable", *sorted(pull)], check=False
            )
            again = sorted({images[name] for name in pull})
            found.update(
                zip(again, await asyncio.gather(*map(declared, again)), strict=True)
            )
        missing = sorted(image for image, paths in found.items() if paths is None)
        if missing and pulled is not None and pulled.return_code != 0:
            # The registry's reason (a typo, a login, a rate limit), as the
            # create's own pull would have reported it.
            reason = (pulled.stdout or "").strip()[-1000:] or "no output"
            raise SandboxStartupError(f"Cannot pull image {missing[0]!r}: {reason}")
        if missing:
            raise SandboxStartupError(
                f"Cannot read the VOLUMEs of image {missing[0]!r} (not on the "
                "daemon after a pull); BenchFlow names each volume a trial "
                "makes before Compose creates it"
            )
        outside: dict[str, list[str]] = {}
        for source in sorted(
            {
                name
                for spec in services.values()
                for name, external in _trial_volumes.volumes_from(spec or {})
                if external
            }
        ):
            read = await self._docker_cli(
                [
                    "inspect",
                    "--type",
                    "container",
                    "--format",
                    "{{json .Mounts}}",
                    source,
                ],
                check=False,
                raw_output=True,
            )
            try:
                if read.return_code != 0:
                    raise ValueError(read.stderr)
                outside[source] = [
                    str(mount["Destination"]) for mount in json.loads(read.stdout or "")
                ]
            except (ValueError, KeyError, TypeError):
                raise SandboxStartupError(
                    f"Cannot read the mounts of container {source!r}, which "
                    "the task names in volumes_from"
                ) from None
        self._trial_volume_inputs = (
            project,
            {image: paths or [] for image, paths in found.items()},
            outside,
        )
        self._write_trial_volumes_overlay()

    def _write_trial_volumes_overlay(self) -> None:
        """Write the overlay for what :meth:`_name_trial_volumes` read, with
        names no earlier start or attempt of this sandbox used."""
        self._drop_trial_volumes_overlay()
        inputs = getattr(self, "_trial_volume_inputs", None)
        if inputs is None:
            return
        self._trial_volume_generation = getattr(self, "_trial_volume_generation", 0) + 1
        try:
            document = _trial_volumes.overlay(*inputs, self._trial_volume_generation)
        except _trial_volumes.Unreadable as exc:
            raise SandboxStartupError(
                f"Cannot name the volumes of {self.compose_project_name}: {exc}"
            ) from None
        if not document:
            return
        fd, name = tempfile.mkstemp(
            prefix=f"benchflow-{self.compose_project_name}-volumes-",
            suffix=".json",
        )
        with os.fdopen(fd, "w") as f:
            json.dump(document, f, indent=2)
        self._trial_volumes_compose_path = Path(name)

    def _drop_trial_volumes_overlay(self) -> None:
        path = getattr(self, "_trial_volumes_compose_path", None)
        if path is not None:
            path.unlink(missing_ok=True)
        self._trial_volumes_compose_path = None

    async def _up_on_own_subnets(
        self,
        up: Callable[[], Awaitable[Any]],
        *,
        prepare: Callable[[], Awaitable[Any]] | None = None,
        down: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        """Run ``up``; when the daemon refuses a chosen subnet because a
        network made meanwhile overlaps it, take the project down, choose
        again and retry. The daemon running out of pools for a network it
        chooses itself ends the trial with what to do instead."""
        for attempt in range(1, _SUBNET_ATTEMPTS + 1):
            try:
                # Every attempt owns a complete set of stopped prototypes.
                # A subnet retry's down removes native sidecars too; guarded
                # up alone cannot recreate the set needed by proof/release.
                # Keep create here as it can itself encounter an overlap.
                await (prepare or self._prepare_inert_bootstrap)()
                await up()
                return
            except RuntimeError as exc:
                message = str(exc)
                if is_pools_exhausted_error(message):
                    raise SandboxStartupError(
                        pools_exhausted_message(
                            chooses=bool(self._subnets),
                            unreadable=self._subnet_plan_unreadable,
                        )
                    ) from exc
                if self._subnet_pinned and is_pool_overlap_error(message):
                    # A network the task pins was refused: BenchFlow's own
                    # subnets stay out of it, so another try cannot help.
                    # Compose names it <project>_<key>; one BenchFlow did
                    # not choose is the task's own.
                    refused = (refused_network(message) or "").lower()
                    project = self.compose_project_name.lower()
                    ours = {f"{project}_{key}".lower() for key in self._subnets}
                    if refused.startswith(f"{project}_") and refused not in ours:
                        raise SandboxStartupError(
                            pinned_overlap_message(refused, self._subnet_pinned)
                            + f" {message}"
                        ) from exc
                if not self._subnets or not is_pool_overlap_error(message):
                    raise
                if attempt == _SUBNET_ATTEMPTS:
                    raise SandboxStartupError(
                        f"Docker refused {attempt} sets of subnets for this "
                        "trial's networks, each overlapping a network made "
                        "meanwhile; lower --concurrency, or give BenchFlow a "
                        "range of its own with BENCHFLOW_DOCKER_SUBNETS. "
                        f"{message}"
                    ) from exc
                self.logger.info(
                    "Docker refused subnets %s for %s (a network made meanwhile "
                    "overlaps; attempt %s/%s); choosing others",
                    ", ".join(str(n) for n in self._subnets.values()),
                    self.compose_project_name,
                    attempt,
                    _SUBNET_ATTEMPTS,
                )
                with contextlib.suppress(RuntimeError):
                    if down is not None:
                        await down()
                    else:
                        await self._run_docker_compose_command(
                            ["down", "--remove-orphans"]
                        )
                await self._choose_subnets(refresh=True)
                # Fresh volumes for the next attempt, as anonymous ones were;
                # teardown removes these by their label.
                self._write_trial_volumes_overlay()

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image:
            # Prebuilt-image task: compose references the image directly and
            # the build step is skipped (matches modal/agentcore/apple).
            return
        if (
            not self._dockerfile_path.exists()
            and not self._environment_docker_compose_path.exists()
        ):
            raise FileNotFoundError(
                f"{self._dockerfile_path} and {self._environment_docker_compose_path} "
                "not found. Please ensure at least one of these files exist."
            )

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        *,
        merge_stderr: bool = True,
        raw_output: bool = False,
        stdin_data: bytes | None = None,
        unset_env: Sequence[str] = (),
    ) -> ExecResult:
        """Run ``docker compose <command>`` on this sandbox's project.

        Compose's stderr, its warnings included, goes into ``stdout`` unless
        ``merge_stderr`` is False (for a caller that parses the output) or the
        command supplies machine-readable plans or identities. Parsers must
        select ``raw_output``: transport redaction is only for diagnostics,
        never for data (even a failed command can supply a partial record).
        With *unset_env*, those names, and every loader name, are left out
        of Compose's own environment: a bare ``-e NAME`` in *command* then
        removes NAME rather than copying it in, and no loader setting
        prints into the output a caller parses.
        """
        raw_output = raw_output or raw_output_requested()
        if command[:1] and command[0] not in _COMPOSE_KEEPS_CONTAINERS:
            # up, down, stop, restart, ... can replace or remove the main
            # container: exec() goes back through compose until start()
            # learns the new one.
            self._main_container = None
        full_command = self._compose_command(command)
        scrubbed = {}
        if unset_env:
            scrubbed["environment"] = {
                k: v
                for k, v in self._docker_process_env().items()
                if k not in unset_env and not loader_variable(k)
            }
        return await self._run_docker_process(
            full_command,
            check=check,
            timeout_sec=timeout_sec,
            what="Docker compose command",
            # Preserve machine-readable plans and identities through the same
            # command assembly used to certify verifier cache invocations.
            separate_stderr=raw_output
            or not merge_stderr
            or (bool(command) and command[0] in {"config", "build", "ps", "images"}),
            stdin_data=stdin_data,
            **scrubbed,
        )

    def _compose_command(self, command: list[str]) -> list[str]:
        launch = getattr(self, "_launch", None)
        full_command = [
            "docker",
            *(
                launch.docker_args
                if launch is not None
                else self._docker_compose_global_args()
            ),
            "compose",
            "--project-name",
            self.compose_project_name,
            "--project-directory",
            str(self.compose_project_dir),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)
        return full_command

    async def _resolve_compose_build(self) -> ComposeBuild:
        return await resolve_compose_build(
            self._compose_command(["build"]),
            {**self._docker_compose_env(), **self._compose_transport_env()},
            list(self._docker_compose_paths),
            self.environment_dir,
        )

    async def _run_docker_container_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        raw_output: bool = False,
        stdin_data: bytes | None = None,
    ) -> ExecResult:
        """Run ``docker <command>`` on the main container, with compose's I/O.

        Same daemon flags, environment, stdin, merged stdout/stderr, timeout
        handling and result as :meth:`_run_docker_compose_command`, without
        the compose CLI: ``docker compose exec`` and ``cp`` start the compose
        plugin and load the project for every call (about 0.1 s against
        0.05 s for ``docker exec`` on the engine bench's VM), and a trial
        runs dozens. The daemon flags and transport environment are
        Compose's (remote-docker's host), so this reaches the same daemon.
        """
        launch = getattr(self, "_launch", None)
        client = (
            launch.docker_args
            if launch is not None
            else self._docker_compose_global_args()
        )
        raw_output = raw_output or raw_output_requested()
        return await self._run_docker_process(
            ["docker", *client, *command],
            check=check,
            timeout_sec=timeout_sec,
            what="Docker command",
            separate_stderr=raw_output,
            stdin_data=stdin_data,
        )

    async def _run_docker_process(
        self,
        full_command: list[str],
        *,
        check: bool,
        timeout_sec: float | None,
        what: str,
        separate_stderr: bool = False,
        environment: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin_data: bytes | None = None,
    ) -> ExecResult:
        """Run *full_command*; *stdin_data*, if any, is its whole standard input."""
        env = self._docker_process_env(environment)

        cancelled_during_cleanup = False

        async def settle(task: asyncio.Task[Any]) -> Any:
            nonlocal cancelled_during_cleanup
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
                    if task.done():
                        return task.result()

        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *full_command,
                env=env,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL
                if stdin_data is None
                else asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
                if separate_stderr
                else asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        )

        async def stop() -> None:
            await _drain_timed_out_process(process, process_group=True)

        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            process = await settle(spawning)
            await settle(asyncio.create_task(stop()))
            raise

        def communicate() -> Any:
            if stdin_data is None:
                return process.communicate()
            return process.communicate(stdin_data)

        try:
            if timeout_sec is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await communicate()
        except (TimeoutError, asyncio.CancelledError) as error:
            await settle(asyncio.create_task(stop()))
            if isinstance(error, asyncio.CancelledError):
                raise
            if cancelled_during_cleanup:
                raise asyncio.CancelledError from None
            raise ExecTimeoutError(timeout_sec) from None

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

        if check and result.return_code != 0:
            raise RuntimeError(
                f"{what} failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}. "
            )

        return result

    def _docker_process_env(
        self, environment: dict[str, str] | None = None
    ) -> dict[str, str]:
        """The docker client's environment for :meth:`_run_docker_process`
        and the exec channel."""
        # Keep the frozen launch environment and integration's explicit build env.
        launch = getattr(self, "_launch", None)
        env = (
            environment
            if environment is not None
            else (
                dict(launch.env)
                if launch is not None
                else {**self._docker_compose_env(), **self._compose_transport_env()}
            )
        )
        if getattr(self, "_bootstrap_override", None) is not None:
            env = _inert_bootstrap.client_env(env)
        return env

    async def _run_pre_compose_hook(self) -> None:
        hook = self._pre_compose_hook_path
        if not hook.is_file():
            return
        from benchflow.task.compose_env import (
            ALLOW_HOST_ACCESS_VAR,
            PRE_COMPOSE_HOOK,
            host_access_allowed,
        )

        # The hook is the task's code, run here as the operator: only when
        # they opted in (the launch gate refuses the task otherwise), and
        # with Compose's allowlisted environment.
        if not host_access_allowed():
            raise RuntimeError(
                f"refused: environment/{PRE_COMPOSE_HOOK} of "
                f"{self.environment_name} would run on this machine as you; set "
                f"{ALLOW_HOST_ACCESS_VAR}=1 if you trust the task with it"
            )

        timeout_sec = max(120, round(self.task_env_config.build_timeout_sec))
        process = await asyncio.create_subprocess_exec(
            "sh",
            str(hook.resolve().absolute()),
            cwd=str(self.environment_dir.resolve().absolute()),
            env=self._docker_compose_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            stdout_bytes, _ = await _drain_timed_out_process(process)
            output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
            raise RuntimeError(
                f"Pre-compose hook timed out after {timeout_sec} seconds for "
                f"environment {self.environment_name}. Output: {output}"
            ) from None

        output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        if process.returncode:
            raise RuntimeError(
                f"Pre-compose hook failed for environment {self.environment_name}. "
                f"Return code: {process.returncode}. Output: {output}"
            )

    @contextlib.asynccontextmanager
    async def _build_progress(self) -> Any:
        """Announce a slow image build in the log and the live display.

        While it runs, ``image_build`` is "building" and the rollout's
        activity cell shows "building image…" instead of "creating
        sandbox…"; then "built" ("starting sandbox…"), so the label never
        steps back.
        """
        started = time.monotonic()

        async def heartbeat() -> None:
            await _heartbeat_sleep(_BUILD_ANNOUNCE_AFTER_SEC)
            self.image_build = "building"
            self.logger.info(
                "Building the task image for %s: the first run of a task builds "
                "it, later runs reuse it (this can take a few minutes)",
                self.environment_name,
            )
            while True:
                await _heartbeat_sleep(_BUILD_HEARTBEAT_SEC)
                self.logger.info(
                    "Still building the task image for %s (%s so far)",
                    self.environment_name,
                    _elapsed(time.monotonic() - started),
                )

        ticker = asyncio.create_task(heartbeat())
        try:
            yield
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            if self.image_build == "building":
                self.image_build = "built"
                self.logger.info(
                    "Built the task image for %s in %s",
                    self.environment_name,
                    _elapsed(time.monotonic() - started),
                )

    async def _run_docker_compose_build(
        self, *, invocation: ComposeBuild | None = None
    ) -> None:
        async with self._build_progress():
            await self._run_docker_compose_build_attempts(invocation=invocation)

    async def _run_docker_compose_build_attempts(
        self, *, invocation: ComposeBuild | None = None
    ) -> None:
        # An explicit, successful cache resolution is the only way to bypass
        # Docker's launcher. Ordinary/fallback calls never consult saved state.
        if invocation is not None and invocation.uncertainty is not None:
            raise LayerError(invocation.uncertainty)
        # Retries and their backoff spend the same budget on both build paths.
        budget = self.task_env_config.build_timeout_sec
        deadline = time.monotonic() + budget
        max_attempts = len(_DOCKER_BUILD_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Docker build timed out after {budget} seconds")
            try:
                if invocation is None:
                    await self._run_docker_compose_command(
                        ["build"], timeout_sec=remaining
                    )
                else:
                    await self._run_docker_process(
                        list(invocation.argv),
                        check=True,
                        timeout_sec=remaining,
                        what="Docker compose build",
                        environment=dict(invocation.environment),
                        cwd=invocation.cwd,
                    )
                return
            except RuntimeError as exc:
                if attempt == max_attempts or not self._is_retryable_build_error(
                    str(exc)
                ):
                    # A build another attempt would repeat is the task's, named
                    # at its Dockerfile line (image_build); the rest stay as is.
                    typed = image_build_error(str(exc))
                    if typed is not None:
                        self.logger.debug("Docker build output:\n%s", typed.log)
                        raise typed from None
                    raise

                delay = _DOCKER_BUILD_RETRY_DELAYS_SEC[attempt - 1]
                if delay >= deadline - time.monotonic():
                    raise RuntimeError(
                        f"Docker build timed out after {budget} seconds"
                    ) from exc
                self.logger.warning(
                    "Retrying Docker build for %s after transient failure "
                    "(attempt %s/%s, retrying in %.1fs): %s",
                    self.environment_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    async def _run_docker_compose_up(self) -> None:
        """Start once and preserve Compose's readiness failure.

        A detached up followed by another up to wait can restart an exited
        entrypoint on its partially initialized filesystem, hiding its first
        failure (U11-C1). The bootstrap uses a single detached up and observes
        readiness after release, without another mutating up. Calls without
        a bootstrap retain Compose's wait. Only the existing network and
        image-export races below are retryable.
        """
        max_attempts = len(_COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                command = _inert_bootstrap.up_args(
                    self,
                    *([] if getattr(self, "_bootstrap_override", None) else ["--wait"]),
                    "--no-build",
                )
                if command is not None:
                    await self._run_docker_compose_command(command)
                return
            except RuntimeError as exc:
                if attempt == max_attempts or not (
                    _is_compose_up_network_race_error(str(exc))
                    or _IMAGE_EXPORT_RACE.search(str(exc))
                ):
                    raise

                delay = _COMPOSE_UP_RETRY_DELAYS_SEC[attempt - 1]
                self.logger.warning(
                    "Retrying docker compose up for %s after network "
                    "create/attach race (attempt %s/%s, retrying in %.1fs): %s",
                    self.environment_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    async def capture_recovery_baseline(
        self, task_digest: str, effective_config_digest: str
    ) -> DockerRecoveryBaseline:
        return await capture_baseline(self, task_digest, effective_config_digest)

    def use_recovery_baseline(self, baseline: DockerRecoveryBaseline) -> None:
        if (
            self._uses_compose
            or self._mounts_json
            or self._pre_compose_hook_path.exists()
        ):
            raise ValueError(
                "Custom runtime topology cannot use a recovery image lease"
            )
        if baseline.bind_targets is None:
            raise ValueError(
                "Recovery has no first-start bind proof; start a fresh trial"
            )
        self._recovery_baseline = baseline
        self._bind_baseline = baseline.bind_targets

    async def start(self, force_build: bool) -> None:
        # The metadata helper's image builds beside the whole start. Where
        # this process has built it already (for a root agent, once it is
        # built), main starts without NET_ADMIN, decided before the first
        # launch freeze.
        self._prebuild_metadata_helper()
        await self._decide_metadata_net_admin()
        # Ordinary startup and recovery both decide the kept-service freezer
        # here, from the Engine's report, before Compose creates anything
        # .
        await decide_kept_service_freezer(self, self._engine_info_json)
        self._context_image_built = False
        # A new container: nothing a previous start's probe kept holds.
        self._forget_kept_identity()
        self._runtime_kit_fresh()
        trust_state.begin_fresh(self)
        # Before Compose creates anything: from here until stop() the
        # leftover sweep of any Evaluation in this process keeps the project.
        claim_project(self.compose_project_name)
        # The subnet allocator's read of the daemon's networks (one inspect of
        # every network, about 1 s on a daemon with many) runs beside the
        # Compose checks and builds below; _choose_subnets joins it (PERF-D).
        self._prefetch_subnet_allocator()
        self._write_container_env_compose_file()
        if self._mounts_json:
            self._mounts_compose_path = self._write_mounts_compose_file()
        await self._prepare_bfx()

        if self._recovery_baseline is not None:
            if force_build:
                raise ValueError("Recovery must never rebuild its original baseline")
            effective = self.task_env_config.model_copy(deep=True)
            effective.allow_internet = self._recovery_baseline.effective_allow_internet
            await validate_baseline(
                self, self._recovery_baseline, sandbox_config=effective
            )
            self.task_env_config = effective
            self._env_vars.prebuilt_image_name = self._recovery_baseline.image_id
        self._use_prebuilt = self._recovery_baseline is not None or (
            not force_build and bool(self.task_env_config.docker_image)
        )
        self._write_prebuilt_compose_file()

        # Gate the entire startup phase (build + down + up) — not just build.
        # When images are cached, build is a no-op so a build-only semaphore
        # has no effect, but the simultaneous `compose up` calls still flood
        # the docker daemon (network creation races, prune timeouts).
        # asyncio primitives bind on contention. SDK batches may each use
        # asyncio.run(), so neither a lock nor a semaphore crosses loops.
        loop = asyncio.get_running_loop()
        build_sem = None
        if self._build_concurrency is not None:
            build_sem = self._build_semaphores.setdefault(
                loop, asyncio.Semaphore(self._build_concurrency)
            )
        if build_sem is not None:
            await build_sem.acquire()
        try:
            if self._recovery_baseline is None:
                if self._pre_compose_hook_path.is_file():
                    # The hook may rewrite the copy's files: check again.
                    self._launch = None
                await self._run_pre_compose_hook()
            # Check the original task before deriving private build references.
            await asyncio.to_thread(self.freeze_launch, force_build)
            if self._recovery_baseline is None and self._uses_compose:
                await self._prepare_compose_builds()
                await asyncio.to_thread(self.freeze_launch, force_build)

            if not self._use_prebuilt and await self._start_verifier_from_context():
                # A cache hit already supplies the single-service verifier.
                pass
            elif not self._use_prebuilt or self._uses_compose:
                # A prebuilt main can still have Dockerfile-based sidecars.
                # They share the explicit build deadline; up never builds.
                await self._build_task_image()

            # Whether BenchFlow is about to pull the task image.
            pull_started = datetime.now(UTC)
            absent_ref = await self._absent_task_image()
            await self._start_from_layers()
            self._write_prebuilt_compose_file()
            # Asked again before the freeze main starts by: the helper image
            # may have been built during this start.
            await self._decide_metadata_net_admin()
            # A cached layer changes main's image and the prebuilt overlay.
            # Re-freeze those inputs before starting any container.
            await asyncio.to_thread(self.freeze_launch, force_build)
            await self._fill_bfx_volume()
            await self._choose_subnets()

            self._bootstrap_override = None
            if self._compose_up_attempted:
                # A start after an earlier one: clear what that left. Before
                # the first `up` the project (a name no other sandbox uses,
                # _unique_compose_project_name) has nothing to remove.
                with contextlib.suppress(RuntimeError):
                    await self._run_docker_compose_command(["down", "--remove-orphans"])

            # Every volume Docker would make anonymous gets a name and the
            # trial's label first, so teardown removes it by that label.
            await self._name_trial_volumes()

            # Prototypes are created stopped. Their effective Docker Config is
            # the sole source for the post-proof command; no image/task process
            # or healthcheck starts while any bind identity is still unknown.
            direct = await self._decide_direct_docker()
            self._compose_up_attempted = True
            if self._recovery_baseline is None:
                if direct:
                    await self._up_on_own_subnets(
                        self._direct_up,
                        prepare=self._direct_prepare,
                        down=self._direct_down,
                    )
                else:
                    await self._up_on_own_subnets(self._run_docker_compose_up)
                await self._note_task_image_pull(absent_ref, pull_started)
            else:

                async def recovery_up():
                    command = _inert_bootstrap.up_args(
                        self,
                        *(
                            []
                            if getattr(self, "_bootstrap_override", None)
                            else ["--wait"]
                        ),
                        "--no-build",
                        "--pull",
                        "never",
                    )
                    if command is not None:
                        await self._run_docker_compose_command(command)

                await self._up_on_own_subnets(recovery_up)
                container_id = await self._main_container_id()
                if (
                    not container_id
                    or (await self._inspect_container(container_id)).get("Image")
                    != self._recovery_baseline.image_id
                ):
                    raise RuntimeError(
                        "Recovery container did not use the original image"
                    )
        finally:
            if build_sem is not None:
                build_sem.release()

        if self._main_container is None:
            await self._learn_main_container()
        # The kit cache's checks (the daemon's diff of main against its image,
        # 0.3-0.7 s) run beside the rest of the start, from before the inert
        # bootstrap is released; the kit build takes them (PERF-SBX).
        self._kit_lookup = (
            asyncio.ensure_future(self._lookup_cached_kit())
            if self._direct is not None and self._main_container is not None
            else None
        )
        # Before any task process runs, while the rest of the start goes on.
        self._start_metadata_helper()
        try:
            await self._pin_or_verify_bind_targets()
            await self._release_inert_bootstrap()
        except BaseException:
            await self._close_metadata_helper()
            if self._direct is not None:
                with contextlib.suppress(Exception):
                    await self._direct_down()
            else:
                await self._run_docker_compose_command(
                    ["down", "--remove-orphans"], check=False
                )
            raise
        if self._main_container is None:
            # The release signals PID 1 (`docker kill --signal USR1`) and may
            # start other services (`up --no-recreate`); like any kill or up,
            # that forgets the main container, so every later exec() would go
            # through `docker compose exec` (about 0.1 s against 0.05 s, for
            # each of a trial's dozens of commands). Ask Compose again.
            await self._learn_main_container()
        await self._prepare_log_dirs_after_up()

    # -- The built kit, kept per image on this host (_kit_cache, PERF-SBX) --

    async def kit_cache_lookup(self) -> Any:
        """The kit cache's entry for main as it is now, or None when main may
        not use the cache (:func:`benchflow.sandbox._kit_cache.eligible`,
        :func:`~benchflow.sandbox._kit_cache.unchanged`)."""
        pending = getattr(self, "_kit_lookup", None)
        self._kit_lookup = None
        if pending is not None:
            return await pending
        return await self._lookup_cached_kit()

    async def _lookup_cached_kit(self) -> Any:
        from benchflow._runtime_tree import kit_prepare_command
        from benchflow.sandbox import _kit_cache

        direct = getattr(self, "_direct", None)
        cid = getattr(self, "_main_container", None)
        if (
            direct is None
            or cid is None
            or self._recovery_baseline is not None
            or not _kit_cache.ENABLED
            or os.environ.get("BENCHFLOW_KIT_CACHE", "").strip() == "0"
        ):
            return None
        try:
            inspected = await direct.inspect(cid)
            why = _kit_cache.eligible(inspected, direct.image_id)
            if why is None:
                changes = (
                    await direct.client.call("GET", f"/containers/{cid}/changes")
                ).json()
                why = _kit_cache.unchanged(changes)
            if why is not None:
                self.logger.debug("Kit cache not used: %s", why)
                return None
            info = (await direct.client.call("GET", "/info")).json() or {}
            key = _kit_cache.kit_key(inspected, info, kit_prepare_command("build"))
            entry = await asyncio.to_thread(_kit_cache.load, key)
            if entry.tar is not None:
                # Its stage goes in now, beside the rest of the start: one
                # archive upload to /, of a fresh folder only root may enter.
                name = f".benchflow-kit-{uuid.uuid4().hex}"
                data = await asyncio.to_thread(_kit_cache.staged_tar, entry.tar, name)
                await direct.client.call(
                    "PUT",
                    f"/containers/{cid}/archive",
                    query={"path": "/"},
                    raw=data,
                    content_type="application/x-tar",
                )
                self._kit_staged = (entry.key, f"/{name}")
            return entry
        except Exception as exc:
            self.logger.debug("Kit cache not used: %s", describe_exception(exc))
            return None

    async def kit_cache_stage(self, entry: Any) -> str | None:
        """Upload *entry*'s kit into a fresh folder only root may enter;
        its path, or None (the build then runs as before)."""
        direct = getattr(self, "_direct", None)
        cid = getattr(self, "_main_container", None)
        if direct is None or cid is None or entry is None or entry.tar is None:
            return None
        staged = getattr(self, "_kit_staged", None)
        self._kit_staged = None
        if staged is not None and staged[0] == entry.key:
            return staged[1]
        stage = f"/.benchflow-kit-{uuid.uuid4().hex}"
        try:
            made = await exec_raw(
                self.exec, f"mkdir -m 700 {stage}", user="root", timeout_sec=30
            )
            if made.return_code != 0:
                return None
            data = await asyncio.to_thread(entry.tar.read_bytes)
            await direct.client.call(
                "PUT",
                f"/containers/{cid}/archive",
                query={"path": stage},
                raw=data,
                content_type="application/x-tar",
            )
            return stage
        except Exception as exc:
            self.logger.debug("Kit cache upload failed: %s", describe_exception(exc))
            with contextlib.suppress(Exception):
                await exec_raw(
                    self.exec, f"rm -rf {stage}", user="root", timeout_sec=30
                )
            return None

    async def kit_cache_store(self, entry: Any) -> None:
        """Keep main's kit, as the build just left it, for *entry*'s key."""
        from benchflow._runtime_tree import KIT
        from benchflow.sandbox import _kit_cache

        direct = getattr(self, "_direct", None)
        cid = getattr(self, "_main_container", None)
        if direct is None or cid is None or entry is None:
            return
        try:
            reply = await direct.client.call(
                "GET", f"/containers/{cid}/archive", query={"path": KIT}
            )
            await asyncio.to_thread(_kit_cache.store, entry.key, reply.body)
        except Exception as exc:
            self.logger.debug("Kit cache store failed: %s", describe_exception(exc))

    # The CPUs main may use while only BenchFlow's own steps run in it.
    _HARNESS_CPUS: ClassVar[int] = 4

    @contextlib.asynccontextmanager
    async def harness_cpus(self) -> Any:
        """Lift main's CPU limit to :data:`_HARNESS_CPUS` while the body runs
        BenchFlow's own steps (the kit, the user and workspace grant, the
        verifier's runtime and the hardening before it), and put the task's
        limit back before any agent, solution or test command (PERF-SBX).

        Those steps ran at the task's limit (one CPU on Terminal-Bench 2)
        although nothing of the task's is measured there. Only a plain
        ``NanoCpus`` limit on the direct path is lifted, never a quota,
        period or CPU set, nor for kept containers. The limit put back is
        read back from the daemon; a different answer raises, so no task
        command runs with more than the task asked for.
        """
        direct = getattr(self, "_direct", None)
        cid = getattr(self, "_main_container", None)
        if direct is None or cid is None or self._keep_containers:
            yield
            return
        host = (await direct.inspect(cid)).get("HostConfig") or {}
        task_nano = host.get("NanoCpus") or 0
        lifted = min(os.cpu_count() or 1, self._HARNESS_CPUS) * 1_000_000_000
        if (
            not isinstance(task_nano, int)
            or task_nano <= 0
            or task_nano >= lifted
            or host.get("CpuQuota")
            or host.get("CpuPeriod")
            or host.get("CpusetCpus")
        ):
            yield
            return
        update = f"/containers/{cid}/update"
        try:
            await direct.client.call("POST", update, body={"NanoCpus": lifted})
            yield
        finally:
            await direct.client.call("POST", update, body={"NanoCpus": task_nano})
            after = (await direct.inspect(cid)).get("HostConfig") or {}
            if after.get("NanoCpus") != task_nano:
                raise RuntimeError(
                    "Docker did not put main's CPU limit back to the task's "
                    f"({task_nano} nano-CPUs; it reports {after.get('NanoCpus')!r}), "
                    "so BenchFlow will not run the task's commands"
                )

    def flush_writes_in_background(self) -> None:
        """Keep this host's dirty pages going to disk while this sandbox's
        tests run, in one thread for the process that no one waits for
        (PERF-SBX); :meth:`stop` ends this sandbox's share.

        A container's removal waits for its overlay's unmount, which writes
        back every dirty page of the filesystem under it first: 60 MB still
        in memory added 0.25 s to the kill. Written while the tests run (the
        kit, the verifier's runtime, the solver's and the tests' own files),
        they no longer wait in the teardown. Only for a daemon on this Linux
        host; nothing is read or changed.
        """
        if (
            getattr(self, "_flushing", False)
            or not self._HOST_MOUNTS_ARE_LOCAL
            or getattr(self, "_remote_daemon", False)
            or self._docker_client_env() is not None
            or not sys.platform.startswith("linux")
            or not hasattr(os, "sync")
        ):
            return
        self._flushing = True
        _HOST_FLUSHER.join()

    def _stop_flushing(self) -> None:
        if getattr(self, "_flushing", False):
            self._flushing = False
            _HOST_FLUSHER.leave()

    async def _prepare_inert_bootstrap(self) -> None:
        # Reprepare the original effective project, never the prior overlay's
        # weakened readiness dependencies or cleared startup environment.
        self._bootstrap_override = None
        document = await _inert_bootstrap.prepare(
            self, self._run_docker_compose_command
        )
        self._write_bootstrap_override(document)

    async def _release_inert_bootstrap(self) -> None:
        direct = getattr(self, "_direct", None)
        if direct is not None:
            await _inert_bootstrap.release(
                self,
                self._run_docker_compose_command,
                ids=await direct.container_ids(),
                start_native=direct.start,
            )
            return
        await _inert_bootstrap.release(self, self._run_docker_compose_command)

    # -- The direct Docker path (_direct_docker, PERF-E) ---------------------

    def _direct_unavailable(self) -> str | None:
        """Why this start stays on Compose before Compose is asked, or None."""
        if not self.direct_docker or not _direct_docker.enabled():
            return f"off ({_direct_docker.SETTING_ENV}=0)"
        if self._remote_daemon or self._docker_client_env() is not None:
            return "remote Docker"
        if self._recovery_baseline is not None:
            return "a recovery start"
        if self._pre_compose_hook_path.is_file():
            return "a pre-compose hook"
        if (self.compose_project_dir / ".env").exists():
            # Compose reads it and labels the container with it.
            return "an .env file in the Compose project folder"
        if "DOCKER_DEFAULT_PLATFORM" in self._docker_compose_env():
            return "DOCKER_DEFAULT_PLATFORM"
        return None

    def _docker_args(self) -> list[str]:
        launch = getattr(self, "_launch", None)
        if launch is not None:
            return list(launch.docker_args)
        return self._docker_compose_global_args()

    async def _decide_direct_docker(self) -> bool:
        """Whether this start makes its containers through the Engine API.

        The plan asks Compose for the project once and checks every key
        (_direct_docker.check_project); anything it cannot reproduce keeps
        the trial on Compose with the reason in ``direct_docker_record``."""
        self._direct = None
        self._direct_pending = None
        reason = self._direct_unavailable()
        if reason is None:
            try:
                self._direct_pending = await self._direct_plan()
            except _direct_docker.Fallback as exc:
                reason = str(exc)
        if reason is not None:
            self.direct_docker_record = {"path": "compose", "reason": reason}
            self.logger.debug(
                "%s starts on Compose: %s", self.compose_project_name, reason
            )
            return False
        assert self._direct_pending is not None
        # The versions it was proven on, and the Engine release (recorded,
        # not gated).
        self.direct_docker_record = {
            "path": "direct",
            **self._direct_pending[0].versions,
        }
        return True

    async def _direct_plan(self) -> tuple[_direct_docker.DirectProject, dict]:
        """Compose's own project (``config --format json``), checked, with
        the image's ID and the Compose version its labels carry."""
        # The overlay's named volumes are top-level volumes, which this path
        # does not make; say why rather than which key refused.
        if getattr(self, "_trial_volumes_compose_path", None) is not None:
            raise _direct_docker.Fallback(
                "BenchFlow names this trial's volumes (an image VOLUME or a "
                "Compose anonymous volume); the direct path makes binds only"
            )
        socket = _direct_docker.endpoint_socket(self._docker_args())
        # The Compose version (a process of its own, 50 ms) is read beside the
        # config and the image's inspect; it is still checked after them, so
        # the first refusal is the one it was (PERF-SBX).
        version = asyncio.ensure_future(self._compose_version())
        version.add_done_callback(lambda task: task.cancelled() or task.exception())
        try:
            return await self._direct_plan_with(socket, version)
        finally:
            if not version.done():
                version.cancel()

    async def _direct_plan_with(
        self, socket: str, version: asyncio.Future[str]
    ) -> tuple[_direct_docker.DirectProject, dict]:
        from benchflow.sandbox._engine_api import (
            EngineClient,
            EngineError,
            EngineUnavailable,
        )
        from benchflow.task.compose_env import _Undecodable, decode_dollars

        try:
            result = await self._run_docker_compose_command(
                ["config", "--format", "json"], raw_output=True, merge_stderr=False
            )
            printed = json.loads(result.stdout or "")
        except (RuntimeError, ValueError) as exc:
            raise _direct_docker.Fallback(
                f"compose config did not answer ({describe_exception(exc)[:200]})"
            ) from None
        try:
            project = decode_dollars(printed)
        except _Undecodable:
            raise _direct_docker.Fallback("compose config printed a lone $") from None
        if not isinstance(project, dict):
            raise _direct_docker.Fallback("compose config printed no project")
        _direct_docker.check_project(project, self.compose_project_name)
        client = EngineClient(socket)
        image = project["services"]["main"]["image"]
        try:
            await client.negotiate()
            reply = await client.call(
                "GET", f"/images/{urllib.parse.quote(image, safe='/:@')}/json"
            )
        except EngineError as exc:
            if exc.status == 404:
                raise _direct_docker.Fallback(
                    "the image is not on the daemon (Compose pulls it)"
                ) from None
            raise _direct_docker.Fallback(f"image inspect: {exc}") from None
        except (EngineUnavailable, TimeoutError) as exc:
            raise _direct_docker.Fallback(str(exc)) from None
        inspected = reply.json() or {}
        image_id = inspected.get("Id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise _direct_docker.Fallback("the daemon gave the image no ID")
        # Compose's recreate carries the prototype's anonymous volumes over to
        # main (HostConfig.Mounts); this path does not reproduce that
        # (PERF-E-A-1), so an image that declares a VOLUME runs on Compose.
        if (inspected.get("Config") or {}).get("Volumes"):
            raise _direct_docker.Fallback(
                "the image declares volumes (Compose carries them over on recreate)"
            )
        compose_build = await version
        api_version = await client.negotiate()
        # The bodies and hashes are proven for these versions only (PERF-E-A-2).
        _direct_docker.check_versions(compose_build, api_version, client.server_version)
        compose_version = _direct_docker.compose_core(compose_build)
        assert compose_version is not None
        plan = _direct_docker.DirectProject(
            client,
            project,
            image_id=image_id,
            compose_version=compose_version,
            working_dir=str(self.compose_project_dir),
        )
        plan.versions = {
            "compose": compose_build,
            "api": api_version,
            "daemon_api": client.server_version or "",
            "engine": client.engine_version or "",
        }
        return plan, printed

    async def _compose_version(self) -> str:
        """``docker compose version --short`` as printed (``2.40.3+ds1-…``):
        the build the version gate matches exactly (PERF-E-r3-A-L2). Compose
        labels what it makes with its core (_direct_docker.compose_core)."""
        args = self._docker_args()
        key = (*args, os.environ.get("PATH", ""))
        version = self._compose_versions.get(key)
        if version is None:
            try:
                result = await self._run_docker_process(
                    ["docker", *args, "compose", "version", "--short"],
                    check=False,
                    timeout_sec=30,
                    what="Docker compose version",
                    separate_stderr=True,
                )
            except (RuntimeError, OSError) as exc:  # ExecTimeoutError included
                raise _direct_docker.Fallback(
                    f"docker compose version did not answer ({describe_exception(exc)[:120]})"
                ) from None
            printed = (result.stdout or "").strip()
            if result.return_code or _direct_docker.compose_core(printed) is None:
                raise _direct_docker.Fallback("docker compose version did not answer")
            version = self._compose_versions[key] = printed
        return version

    def _compose_file_list(self) -> list[str]:
        """The ``-f`` files Compose would get now, as its label lists them."""
        return [str(path.resolve().absolute()) for path in self._docker_compose_paths]

    async def _direct_prepare(self) -> None:
        """Compose's ``create --force-recreate``: the network and a stopped
        prototype of main, then the same inert-bootstrap overlay
        (_inert_bootstrap.overlay) from what the daemon made of it."""
        self._bootstrap_override = None
        pending, self._direct_pending = self._direct_pending, None
        if pending is None:
            # A retry on other subnets: Compose resolves the project again.
            try:
                pending = await self._direct_plan()
            except _direct_docker.Fallback as exc:
                raise RuntimeError(
                    f"the direct Docker start cannot retry ({exc}); start a fresh trial"
                ) from None
        plan, printed = pending
        self._direct = plan
        await plan.ensure_network()
        prototype = await plan.create(
            plan.project["services"]["main"], self._compose_file_list(), replace=False
        )
        document = await _inert_bootstrap.overlay(self, printed, [prototype])
        self._direct_overlay = document
        self._write_bootstrap_override(document)

    async def _direct_up(self) -> None:
        """Compose's ``up --detach --no-deps main`` with the overlay: the
        prototype replaced by main with the bootstrap, then started. With no
        guarded service, nothing starts before the bind proof (as up_args)."""
        plan = self._direct
        assert plan is not None and plan.container is not None
        if not getattr(self, "_bootstrap_services", None):
            return
        overlay = self._direct_overlay
        assert overlay is not None
        service = _direct_docker.merge_bootstrap(
            plan.project["services"]["main"],
            # Its values as Compose reads the file back (written with $$).
            overlay["services"]["main"],
        )
        await plan.remove(plan.container)
        plan.container = None
        files = self._compose_file_list()
        max_attempts = len(_COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if plan.container is not None:
                    await plan.remove(plan.container, force=True)
                    plan.container = None
                cid = await plan.create(service, files, replace=True)
                await plan.start(cid)
                break
            except RuntimeError as exc:
                if attempt == max_attempts or not _is_compose_up_network_race_error(
                    str(exc)
                ):
                    raise
                await asyncio.sleep(_COMPOSE_UP_RETRY_DELAYS_SEC[attempt - 1])
        self._main_container = plan.container

    async def _direct_down(self) -> None:
        if self._direct is not None:
            await self._direct.teardown(volumes=False)

    def _write_bootstrap_override(self, document: dict) -> None:
        folder = getattr(self, "_bootstrap_folder", None)
        if folder is None:
            self._bootstrap_folder = tempfile.TemporaryDirectory(
                prefix="benchflow-bootstrap-"
            )
        path = Path(self._bootstrap_folder.name) / "bootstrap.json"
        # Compose interpolation must never interpret argv or literal env text.
        path.write_text(json.dumps(document).replace("$", "$$"))
        self._bootstrap_override = path

    async def _learn_main_container(self) -> None:
        """Remember the ``main`` container compose started (``_main_container``).

        Only an answer naming exactly one container is kept;
        anything else leaves exec() on ``docker compose``. The direct path
        made main itself: the same rule, over the Engine API.
        """
        direct = getattr(self, "_direct", None)
        if direct is not None:
            ids = await direct.container_ids(running_only=True)
            if len(ids) == 1 and ids[0] == direct.container:
                self._main_container = ids[0]
            return
        try:
            ids = await compose_container_ids(
                self._run_docker_compose_command, service="main", all_containers=False
            )
        except RuntimeError:
            return
        if len(ids) == 1:
            self._main_container = ids[0]

    async def _decide_metadata_net_admin(self) -> None:
        """Start main without NET_ADMIN where the metadata block is its one
        reason for it and the helper adds the block's rules from outside main:
        an open network, no agent-layer no-web firewall,
        the helper on and this process's helper image already built for this
        daemon. The start asks before its first launch freeze, so that every
        freeze reads the same files, and again before the last (a build that
        began with this start may be done by then); it never waits for the
        build. Nothing in main can then change the firewall, root included,
        and the firewall-power check ends at its first read. Otherwise, and
        on a recovery start (its baseline kept NET_ADMIN), main keeps it, so
        root's script can still add the rules in main.

        For an agent that runs as root the start does wait for the build: the
        block holds such an agent only in a main without NET_ADMIN (and then
        without NET_RAW, compose_drops_net_raw). A build that fails
        leaves main as before, and the block's record says why it cannot hold
        the agent."""
        from benchflow.task.config import NetworkMode

        config = getattr(self, "task_env_config", None)
        if (
            config is None
            or getattr(self, "_recovery_baseline", None) is not None
            or not getattr(config, "cloud_metadata_block", False)
            or getattr(config, "cloud_metadata_by_helper", False)
            or getattr(config, "agent_uid_firewall", False)
            or config.network_mode != NetworkMode.PUBLIC
            or not config.allow_internet
            or not _metadata_helper.helper_enabled()
            or _metadata_helper.main_keeps_net_admin()
        ):
            return
        daemon = self._helper_daemon()
        if getattr(config, "cloud_metadata_root_agent", False):
            # Its failure is the block's to report (the helper says why).
            with contextlib.suppress(Exception):
                await _metadata_helper.ensure_image(self._helper_docker, daemon)
        if _metadata_helper.image_ready(daemon):
            config.serve_cloud_metadata_from_helper()

    def _helper_daemon(self) -> str:
        """The daemon key of this sandbox's helper image (its client flags)."""
        return " ".join(self._docker_compose_global_args())

    def _prebuild_metadata_helper(self) -> None:
        """When this run arms the cloud metadata block, start this process's
        build of the helper image (no build cache) now, beside the sandbox's
        start, so the helper can run as soon as main exists
        (_metadata_helper.prebuild)."""
        config = getattr(self, "task_env_config", None)
        if (
            not getattr(config, "cloud_metadata_block", False)
            or not _metadata_helper.helper_enabled()
        ):
            return
        _metadata_helper.prebuild(self._helper_docker, self._helper_daemon())

    def _start_metadata_helper(self) -> None:
        """When this run arms the cloud metadata block, start its rules'
        install from a helper container in main's network namespace as soon
        as main exists, alongside the rest of the start (_metadata_helper)."""
        main = self._main_container
        if (
            not main
            or not getattr(self.task_env_config, "cloud_metadata_block", False)
            or not _metadata_helper.helper_enabled()
        ):
            return
        earlier = getattr(self, "_metadata_helper", None)
        if earlier is not None:
            if earlier.target == main:
                return
            # An earlier start's, in a namespace that is gone.
            earlier.close()
            self._retired_metadata_helpers = [
                *getattr(self, "_retired_metadata_helpers", ()),
                earlier,
            ]
        from benchflow.sandbox.lockdown import _cloud_metadata_block_cmd

        self._metadata_helper = self._new_metadata_helper()
        # The block runs the same script; for a root agent it refuses root too.
        root = getattr(self.task_env_config, "cloud_metadata_root_agent", False)
        self._metadata_helper.install_early(
            main, _cloud_metadata_block_cmd(cover_root=root)
        )

    def _new_metadata_helper(self) -> _metadata_helper.MetadataHelper:
        return _metadata_helper.MetadataHelper(
            self._helper_docker,
            # The image is built once per daemon this process reaches.
            daemon=self._helper_daemon(),
            name=_sanitize_docker_image_name(
                f"{self.compose_project_name}-bf-metadata"
            ),
            labels={OWNED_LABEL: OWNED_VALUE, PROCESS_LABEL: process_token()},
        )

    async def _helper_docker(
        self, args: list[str], *, timeout: float, stdin: bytes | None = None
    ) -> ExecResult:
        """A docker command for the metadata helper, on Compose's daemon with
        its flags and environment (the direct calls' connection). Unlike
        _docker_cli it never forgets main: the helper's run and rm do not
        touch it."""
        return await self._run_docker_container_command(
            args,
            check=False,
            timeout_sec=int(timeout),
            raw_output=True,
            stdin_data=stdin,
        )

    async def metadata_rules_install(self, command: str) -> tuple[Any, str] | None:
        """The cloud metadata block's install (root's ``command``) from the
        helper: the early one when main is the container it joined, else one
        now; None when the helper cannot serve this sandbox (the block then
        installs in main, and ``metadata_rules_unavailable`` says why)."""
        if not _metadata_helper.helper_enabled():
            return None
        if self._metadata_helper is None:
            self._metadata_helper = self._new_metadata_helper()
        main = self._main_container or await self._main_container_id()
        # Main without NET_ADMIN of its own: a new helper gets one more try.
        retry = getattr(self.task_env_config, "cloud_metadata_by_helper", False)
        return await self._metadata_helper.installed(main, command, retry=retry)

    async def metadata_rules_run(
        self, command: str, *, timeout_sec: int, user: str | None = None
    ) -> tuple[Any, str]:
        """Root's ``command`` in the helper that installed the rules (or
        ``user``'s, numeric ``<uid>:<gid>``: the block's test connect)."""
        helper = self._metadata_helper
        if helper is None:
            return None, "the metadata firewall helper is gone"
        return await helper.run(command, timeout_sec=timeout_sec, user=user)

    async def metadata_main_restarts(self) -> str:
        """Why ``main`` could restart ("" when it cannot): its restart policy
        as the daemon holds it. A restarted ``main`` gets a new network
        namespace without the block's rules, so the block is on only for a
        ``main`` whose policy is ``no`` (docker-compose-no-restart.yaml)."""
        main = self._main_container or await self._main_container_id()
        if not main:
            return "main's restart policy could not be read (main is not known)"
        try:
            result = await self._helper_docker(
                ["inspect", "--format", "{{.HostConfig.RestartPolicy.Name}}", main],
                timeout=60,
            )
        except Exception as exc:
            return f"main's restart policy could not be read ({type(exc).__name__})"
        from benchflow.sandbox._compose import main_restart_refusal

        policy = (result.stdout or "").strip()
        return main_restart_refusal(policy if result.return_code == 0 else None)

    @property
    def metadata_rules_unavailable(self) -> str:
        """Why the helper did not serve the block, or why this process could
        not build its image ("" when neither)."""
        helper = getattr(self, "_metadata_helper", None)
        if helper is not None and helper.unavailable:
            return helper.unavailable
        return _metadata_helper.build_failure(self._helper_daemon())

    def metadata_rules_close(self) -> None:
        """The block has its proof: remove the helper in the background."""
        if self._metadata_helper is not None:
            self._metadata_helper.close()

    async def _close_metadata_helper(self) -> None:
        helpers = [
            *getattr(self, "_retired_metadata_helpers", ()),
            getattr(self, "_metadata_helper", None),
        ]
        self._retired_metadata_helpers = []
        self._metadata_helper = None
        for helper in helpers:
            if helper is not None:
                await helper.aclose()

    async def _absent_task_image(self) -> str | None:
        """The prebuilt task image's ref when the daemon does not have it yet.

        Then this start pulls it (a layer build or `compose up`), and
        :meth:`_note_task_image_pull` records that BenchFlow did, so `bench
        cache clean --only pulled-images` may remove it later; an image that
        was there before is never recorded (_pull_provenance).
        """
        ref = self.task_env_config.docker_image
        if not ref or not self._use_prebuilt or self._recovery_baseline is not None:
            return None
        try:
            absent = await self._docker_layers().absent(ref)
        except Exception:
            return None
        # Only Docker's "No such image": an inspect that failed for another
        # reason may hide the user's own image, which is never recorded.
        return ref if absent else None

    async def _note_task_image_pull(
        self, ref: str | None, since: datetime | None = None
    ) -> None:
        if ref is None:
            return
        layers = self._docker_layers()
        with contextlib.suppress(Exception):
            # A layer build pulled it and recorded it right then; its pin of
            # the task image may have changed the tag time since.
            await layers.note_pulled(ref, unless_recorded_since=since)

    def use_agent_layer(self, recipe: AgentLayerRecipe | None) -> None:
        """Start ``main`` from ``recipe``'s cached agent layer (env_layers).

        Call before :meth:`start`. The layer is the task image plus the
        agent's pinned install, built once per key and reused; what stays per
        trial (the agent user, credentials, skills, lockdown) is the caller's.
        ``None`` starts from the task image as before.
        """
        self._agent_layer_recipe = recipe

    def use_proxy_layer(self, recipe: ProxyLayerRecipe | None) -> None:
        """Start ``main`` from ``recipe``'s cached model-proxy layer (env_layers).

        Call before :meth:`start`. The layer is the task image plus the
        LiteLLM venv of the proxy's locked set; the proxy's config, keys and
        logs are written in the trial. With an agent layer too, the agent's
        is built on top of this one. ``None`` installs the proxy in the trial.
        """
        self._proxy_layer_recipe = recipe

    def use_verifier_layer(self, recipe: VerifierLayerRecipe | None) -> None:
        """Start ``main`` on ``recipe``'s verifier layer (verifier_layers).

        Call before :meth:`start`. This reuses the separate verifier image
        with cache labels, preserving its filesystem and execution config.
        It never installs tools. ``None`` starts as before.
        """
        self._verifier_layer_recipe = recipe
        self._verifier_layer = None

    @property
    def verifier_layer(self) -> VerifierLayer | None:
        """The verifier layer ``main`` started from, or None."""
        return self._verifier_layer

    async def _start_verifier_from_context(self) -> bool:
        """A separate verifier's image from its Dockerfile: once per context key.

        True when ``main`` now starts from the verifier layer, built (or
        found) under a key that holds the build context's digest, so a hit
        builds nothing. False leaves the per-trial build as before, with the
        reason recorded.
        """
        recipe = self._verifier_layer_recipe
        if recipe is None or not recipe.separate or self._recovery_baseline is not None:
            return False
        reason = self._agent_layer_unsupported()
        if reason is None and self._uses_compose:
            reason = (
                "verifier context caching needs the stock single-service Compose build"
            )
        if reason is None and not self._dockerfile_path.is_file():
            reason = "the verifier image's context has no Dockerfile"
        if reason is not None:
            self.layer_records[VERIFIER] = fresh_record("unsupported", reason)
            self._verifier_layer_recipe = None
            return False
        alias = _sanitize_docker_image_name(
            f"{ALIAS_PREFIX}{self.compose_project_name}"
        )
        try:
            layers = self._docker_layers()

            # Only a fully resolved plain plugin may build for the cache.
            # Every ordinary/fallback build retains the original Docker call.
            async def build_context() -> BaseImage:
                image = await self._build_task_image(capture_image=True)
                assert image is not None
                return image

            async def build_identity() -> str:
                return await self._verifier_build_identity(layers)

            layer = await ensure_verifier_layer(
                layers,
                recipe,
                None,
                alias=f"{alias}:{alias_owner_tag()}",
                context=self.environment_dir,
                compose_env=self._docker_compose_env(),
                compose_dir=self.environment_dir,
                build_context=build_context,
                build_identity=build_identity,
            )
        except LayerError as e:
            self._layer_failed(VERIFIER, "verifier", e)
            self._verifier_layer_recipe = None
            return False
        self._verifier_layer = layer
        self.layer_records[VERIFIER] = layer.record()
        self._env_vars.prebuilt_image_name = layer.result.image
        self._use_prebuilt = True
        return True

    async def _build_task_image(
        self, *, capture_image: bool = False
    ) -> BaseImage | None:
        """The sole context build path, with or without verifier image reuse.

        Cache publication uses only the invocation authenticated by its key.
        Ordinary starts never resolve or substitute the original launcher.
        """
        daemon = None
        with contextlib.suppress(SubnetConfigError):
            daemon = (await self._docker_subnet_key().identify()).server_id
        locks = self._image_build_locks.setdefault(asyncio.get_running_loop(), {})
        lock = (
            locks.setdefault((daemon, self.environment_name), asyncio.Lock())
            if daemon is not None
            else asyncio.Lock()
        )
        async with lock:
            if capture_image:
                invocation = getattr(self, "_resolved_context_build", None)
                if invocation is None:
                    raise LayerError("verifier cache build has no resolved invocation")
                await self._run_docker_compose_build(invocation=invocation)
            else:
                await self._run_docker_compose_build()
            self._context_image_built = True
            if capture_image:
                layers = self._docker_layers()
                info = await layers._inspect(self._env_vars.main_image_name)
                labels = (info or {}).get("Config", {}).get("Labels") or {}
                if (
                    not info
                    or labels.get("com.docker.compose.project")
                    != self.compose_project_name
                    or labels.get("com.docker.compose.service") != "main"
                ):
                    # Another process can rebuild the ordinary mutable tag.
                    # Never publish its result under this context's key.
                    raise LayerError(
                        "the shared Compose build result changed before capture"
                    )
                return await layers.inspect_base(str(info["Id"]))
        return None

    async def _verifier_build_identity(self, _layers: DockerLayers) -> str:
        """Hash the actual build's frozen invocation, never a second probe.

        Only the cache build consumes this record. Fallback uses the original
        launcher. Keep every environment/argument value: differences can reduce
        reuse, but can never be normalised into a false cache hit.
        """
        self._resolved_context_build = await self._resolve_compose_build()
        env = dict(self._resolved_context_build.environment)
        generated = self._env_vars.to_env_dict(include_os_env=False)
        for field in DockerSandboxEnvVars.model_fields:
            name = field.upper()
            if env.get(name) != generated.get(name):
                raise LayerError(
                    "verifier context caching needs the stock Compose build inputs"
                )
        return self._resolved_context_build.identity()

    @property
    def agent_layer(self) -> LayerResult | None:
        """The agent layer ``main`` started from, or None."""
        return self._agent_layer

    @property
    def proxy_layer(self) -> LayerResult | None:
        """The model-proxy layer ``main`` started from (under the agent's), or None."""
        return self._proxy_layer

    def _agent_layer_unsupported(self) -> str | None:
        """Why this sandbox cannot start from a cached layer, or None."""
        if self._recovery_baseline is not None:
            return "a recovered trial starts from its original container image"
        if self._mounts_json:
            return "runtime mounts can hide the cached install"
        if self._uses_compose:
            import yaml

            try:
                compose = yaml.safe_load(
                    self._environment_docker_compose_path.read_text()
                )
            except (OSError, yaml.YAMLError) as e:
                return f"the task's docker-compose.yaml could not be read: {e}"
            if not isinstance(compose, dict) or compose.get("include"):
                return (
                    "the task's docker-compose.yaml may override main through includes"
                )
            services = compose.get("services") if isinstance(compose, dict) else None
            main = services.get("main") if isinstance(services, dict) else None
            # A stamp survives mounts over the launcher, and arbitrary startup
            # commands can rewrite any installed dependency. Do not try to
            # enumerate safe paths or Compose keys (extends can add them too).
            # Sidecar-only overlays keep the normal main service unchanged.
            if main:
                return "the task's docker-compose.yaml overrides the main service"
        return None

    def _layer_failed(self, kind: str, what: str, error: LayerError) -> None:
        self.logger.warning(
            "No cached %s layer for %s; it installs in the trial: %s",
            what,
            self.environment_name,
            error,
        )
        record = (
            fresh_record("skipped", str(error))
            if isinstance(error, LayerSkipped)
            else fresh_record("fresh", f"layer failed: {error}"[:2000])
        )
        if isinstance(error, LayerLockFallback):
            # Installed fresh without touching the shared layer: say why
            # (the layer lock, _layer_lock) and how long the trial waited.
            record.update(error.record_fields())
        self.layer_records[kind] = record

    async def _start_from_layers(self) -> None:
        """Point ``main`` at the cached layers asked for, if any.

        The proxy's layer is built on the task image, the agent's on top of
        it and the verifier's on top of that, so one image holds them all. A
        layer that cannot be built or used
        is recorded and the trial goes on from what is below it, where that
        install runs in the trial as before.
        """
        agent_recipe = self._agent_layer_recipe
        proxy_recipe = self._proxy_layer_recipe
        # The separate verifier built from its context has its layer already
        # (_start_verifier_from_context); nothing goes on top of it.
        verifier_recipe = (
            self._verifier_layer_recipe if self._verifier_layer is None else None
        )
        self._agent_layer = None
        self._proxy_layer = None
        asked = [
            kind
            for kind, recipe in (
                (VERIFIER, verifier_recipe),
                (PROXY, proxy_recipe),
                (AGENT, agent_recipe),
            )
            if recipe is not None
        ]
        if not asked:
            return
        reason = self._agent_layer_unsupported()
        if reason is not None:
            for kind in asked:
                self.layer_records[kind] = fresh_record("unsupported", reason)
            self._verifier_layer_recipe = None
            return
        # An earlier start() of this sandbox may have pointed main at its layer.
        self._env_vars.prebuilt_image_name = self.task_env_config.docker_image
        base_ref = (
            self.task_env_config.docker_image
            if self._use_prebuilt
            else self._env_vars.main_image_name
        )
        if not base_ref:
            for kind in asked:
                self.layer_records[kind] = fresh_record("unsupported", "no task image")
            return
        # The trial's own tags on its layers: teardown removes these, never
        # the cached tags other trials reuse, and while they exist no prune
        # (of any process) removes the layer under them. Their tag names this
        # process, so a prune drops them if the process dies first.
        alias_repository = _sanitize_docker_image_name(
            f"{ALIAS_PREFIX}{self.compose_project_name}"
        )
        owner = alias_owner_tag()
        alias = f"{alias_repository}:{owner}"
        proxy_alias = f"{alias_repository}__proxy:{owner}"
        layers = self._docker_layers()
        pull = self._use_prebuilt
        top: LayerResult | None = None
        # The proxy goes on the task image and the agent on the proxy.
        # A separate verifier can instead request its image-only cache.
        # Shared grading does not request a verifier layer.
        task_ref = base_ref
        below: BaseImage | None = None
        if proxy_recipe is not None:
            try:
                proxy = await layers.ensure_proxy_layer(
                    proxy_recipe,
                    base_ref,
                    alias=(
                        alias
                        if agent_recipe is None and verifier_recipe is None
                        else proxy_alias
                    ),
                    pull=pull,
                )
            except LayerError as e:
                self._layer_failed(PROXY, "model proxy", e)
            else:
                self._proxy_layer = top = proxy
                self.layer_records[PROXY] = proxy.record()
                base_ref, below, pull = proxy.tag, proxy.as_base, False
        if agent_recipe is not None:
            try:
                # On the proxy's layer: already inspected, never pulled.
                on_proxy = {} if below is None else {"base": below}
                agent = await layers.ensure_agent_layer(
                    agent_recipe,
                    base_ref,
                    alias=(
                        alias
                        if verifier_recipe is None
                        else f"{alias_repository}__agent:{owner}"
                    ),
                    pull=pull,
                    **on_proxy,
                )
            except LayerError as e:
                self._layer_failed(AGENT, agent_recipe.agent, e)
            else:
                self._agent_layer = top = agent
                self.layer_records[AGENT] = agent.record()
                base_ref, below, pull = agent.tag, agent.as_base, False
        if verifier_recipe is not None:
            try:
                verifier = await ensure_verifier_layer(
                    layers,
                    verifier_recipe,
                    base_ref,
                    alias=alias,
                    pull=pull,
                    base=below,
                    compose_env=self._docker_compose_env(),
                    compose_dir=self.environment_dir,
                )
            except LayerError as e:
                self._layer_failed(VERIFIER, "verifier", e)
                self._verifier_layer_recipe = None
            else:
                self._verifier_layer = verifier
                top = verifier.result
                self.layer_records[VERIFIER] = verifier.record()
        if top is None:
            return
        separate = verifier_recipe is not None and verifier_recipe.separate
        # A separate verifier keeps its base: the task's next trial needs it.
        self._layers_base_ref = None if separate else task_ref
        self._env_vars.prebuilt_image_name = top.image
        self._use_prebuilt = True

    async def _release_layers(self, delete: bool) -> None:
        """After teardown: drop the trial's tags on its layers, and their claims.

        The base ref is borrowed: it may be a prebuilt image or a shared build
        name another trial has reassigned. Only project-filtered cleanup may
        collect the original build, never this mutable name.
        """
        held = [
            layer
            for layer in (
                self._agent_layer,
                self._proxy_layer,
                self._verifier_layer.result if self._verifier_layer else None,
            )
            if layer is not None
        ]
        self._agent_layer = None
        self._proxy_layer = None
        self._verifier_layer = None
        if not held:
            return
        refs = []
        aliases = set()
        for layer in held:
            DockerLayers.release(layer.key, daemon=layer.daemon_id)
            if layer.image != layer.tag:
                refs.append(layer.image)
                aliases.add(layer.image)
        for ref in refs:
            # Without --force: an image a kept container uses stays.
            try:
                result = await self._docker_cli(
                    ["image", "rm", "--no-prune", ref], check=False, raw_output=True
                )
            except Exception as e:
                result = ExecResult(stdout="", stderr=str(e), return_code=1)
            text = (result.stderr or result.stdout).strip()
            if (
                result.return_code != 0
                and ref in aliases
                # `down --rmi all` removed it already.
                and "no such image" not in text.lower()
            ):
                # Left behind, the tag holds the layer until this process ends
                # and a later prune finds it.
                self.logger.warning(
                    "Removing the trial's layer tag %s failed: %s",
                    ref,
                    self._diagnostic_text(text)[-300:],
                )

    async def _prepare_log_dirs_after_up(self) -> None:
        """Make the mounted log folders writable and check the mount is shared.

        The chmod runs as the default user, in one exec with that user's
        ``pwd`` and ``id -u``, which the sandbox keeps until it stops or is
        restored (``start_with_identity_probe``, PY8 U12).
        """
        await start_with_identity_probe(
            self,
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}",
            partial(exec_raw, self.exec),
            user=self._resolve_user(None),
            env=self._merge_env(None),
        )
        await self._probe_verifier_log_mount()

    async def _pin_or_verify_bind_targets(self) -> None:
        """Before setup/agent code, pin every service's actual bind targets.

        Keep the first receipt on this trusted object across start retries.
        Snapshot labels and recovery records carry it across new objects.
        """
        first = self._bind_baseline is None
        direct = getattr(self, "_direct", None)
        if direct is not None:
            ids = await direct.container_ids()
        else:
            ids = await compose_container_ids(self._run_docker_compose_command)
        self._bind_baseline = await _bind_targets.pin_project(
            self, "\n".join(ids), self._bind_baseline
        )
        # Only where the daemon sees this machine's files can a source be
        # resolved here; elsewhere nothing is own and every bind stays strict.
        staged = getattr(self, "_staged", None)
        if first and staged is not None and _daemon_sees_this_filesystem(self):
            self._staged_bind_targets = await _bind_targets.staged_project_targets(
                self, ids, str(staged.task_dir), self._bind_baseline
            )

    def _down_command(self, delete: bool) -> list[str]:
        """The ``compose down`` arguments teardown runs (unless containers are kept).

        A recovery sandbox runs on its parent's leased image and keeps it.
        """
        return compose_down_command()

    def _removes_volumes(self, delete: bool) -> bool:
        # A recovery sandbox keeps its parent's leased image, not volumes: its
        # project is its own, and it may not run a Compose file or mounts
        # (use_recovery_baseline), so every volume it has its start created.
        return delete

    async def _remove_project_volumes(self) -> None:
        """Remove only named volumes Compose created for this project.

        Compose's ``down --volumes`` also deletes adopted, unlabelled volumes.
        The volumes BenchFlow named for the trial (_trial_volumes) carry
        this label from their creation, so they go here too.
        No force: a volume another container still uses must stay.
        """
        label = f"label=com.docker.compose.project={self.compose_project_name}"
        listed = await asyncio.wait_for(
            self._docker_cli(
                ["volume", "ls", "-q", "--filter", label],
                check=False,
                raw_output=True,
            ),
            timeout=30,
        )
        volumes = (listed.stdout or "").split() if listed.return_code == 0 else []
        if volumes:
            await asyncio.wait_for(
                self._docker_cli(["volume", "rm", *volumes], check=False), timeout=30
            )

    def _removes_built_images(self, delete: bool) -> bool:
        """Whether teardown removes the images this sandbox's project built."""
        return delete and self._recovery_baseline is None

    async def _prepare_compose_builds(self) -> None:
        """Remember exact build references from the effective Compose config.

        Freeze private output references after resolving the task topology.
        An unreadable plan refuses startup: shared tags cannot be isolated by
        guessing names, and additional build.tags must not survive the merge.
        """
        from benchflow.sandbox._compose_runtime import (
            image_mount_daemon_version,
            implicit_build_aliases,
            private_build_override,
        )

        previous = self._build_compose_path
        self._build_compose_path = None
        if previous is not None:
            previous.unlink(missing_ok=True)
        self._built_image_tags.clear()
        self._private_built_image_tags.clear()
        try:
            result = await self._run_docker_compose_command(
                ["config", "--format", "json"], timeout_sec=30, raw_output=True
            )
            config = json.loads(result.stdout or "{}")
            if not isinstance(config.get("services"), dict):
                raise ValueError("Compose did not resolve services")
            document, refs = private_build_override(
                config,
                self._env_vars.main_image_name,
                prebuilt=self._use_prebuilt,
                agent_security=True,
                daemon_version=await image_mount_daemon_version(
                    config,
                    lambda: self._run_docker_process(
                        [
                            "docker",
                            *self._docker_compose_global_args(),
                            "version",
                            "--format",
                            "{{.Server.Version}}",
                        ],
                        check=False,
                        timeout_sec=30,
                        what="Docker daemon version",
                        separate_stderr=True,
                    ),
                ),
                implicit_aliases=await implicit_build_aliases(
                    config,
                    prebuilt=self._use_prebuilt,
                    compose=partial(self._run_docker_compose_command, raw_output=True),
                ),
            )
            if config["services"]:
                fd, name = tempfile.mkstemp(prefix="benchflow-build-", suffix=".yaml")
                with os.fdopen(fd, "w") as stream:
                    stream.write(document)
                self._build_compose_path = Path(name)
            await self._record_built_image_tags(known_private_refs=refs)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            detail = self._diagnostic_text(str(exc))
            raise RuntimeError(
                f"Could not isolate Compose build output references: {detail}"
            ) from (exc if detail == str(exc) else None)

    async def _record_built_image_tags(
        self, *, known_private_refs: set[str] | None = None
    ) -> None:
        """Remember exact build references from the effective Compose config.

        Pulled services are excluded. If config cannot be read, retain all
        tagged builds rather than infer ownership from a repository prefix.
        """
        if known_private_refs is not None:
            # Startup constructed these exact references, so no second
            # ownership inference through mutable aliases is needed.
            self._built_image_tags = set(known_private_refs)
            self._private_built_image_tags = set(known_private_refs)
            return
        self._built_image_tags.clear()
        self._private_built_image_tags.clear()
        try:
            result = await self._run_docker_compose_command(
                ["config", "--format", "json"], timeout_sec=30, raw_output=True
            )
            config = json.loads(result.stdout or "{}")
            tags: set[str] = set()
            explicit_tags: set[str] = set()
            generated_builds: list[str] = []

            for service, spec in config.get("services", {}).items():
                if not isinstance(spec, dict):
                    continue
                refs = [spec.get("image")]
                if spec.get("build") and not spec.get("image"):
                    generated_builds.append(service)
                if isinstance(spec.get("build"), dict):
                    refs.extend(spec["build"].get("tags") or [])
                for ref in refs:
                    if isinstance(ref, str) and ref:
                        explicit_tags.add(image_ref_identity(ref))
                        if spec.get("build") and "@" not in ref:
                            tags.add(image_ref_identity(ref))
            private_tags: set[str] = set()
            # The built-in main tag remains ours when a task adds sidecars.
            private_tags.update(
                tags & {image_ref_identity(self._env_vars.main_image_name)}
            )
            if generated_builds:
                # Compose decides the separator (COMPOSE_COMPATIBILITY uses
                # '_'). Its list also includes defaults for provider-only
                # services, and service-scoped queries include dependencies.
                # Match each build's exact candidates against the resolved list,
                # excluding every explicit image/build.tags reference. Neither
                # unrelated services nor a matching prefix establish ownership.
                resolved = await self._run_docker_compose_command(
                    ["config", "--images"], timeout_sec=30, raw_output=True
                )
                names = (resolved.stdout or "").splitlines()
                if not names or any(len(name.split()) != 1 for name in names):
                    raise ValueError("Compose did not resolve build images")
                resolved_tags = {
                    image_ref_identity(name.strip()) for name in names
                } - explicit_tags
                for service in generated_builds:
                    candidates = {
                        image_ref_identity(
                            f"{self.compose_project_name}{separator}{service}:latest"
                        )
                        for separator in ("-", "_")
                    }
                    matches = resolved_tags.intersection(candidates)
                    if len(matches) != 1:
                        self.logger.debug(
                            "Could not resolve build image for %s; keeping its tags",
                            service,
                        )
                        continue
                    # A shared or unresolved build says nothing about the
                    # ownership of another service's independently resolved tag.
                    private_tags.update(matches)
                tags.update(private_tags)
            self._built_image_tags = tags
            self._private_built_image_tags = private_tags
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            self.logger.debug("Could not resolve build tags; keeping tagged images")

    async def _kill_project_containers(self) -> None:
        """SIGKILL every container of the project at once (teardown, step 1).

        A failed kill is left to ``down -t 0``, which kills what is still
        running; a timeout means the daemon is wedged, and the caller falls
        through to :meth:`_force_kill_project`.
        """
        result = await self._run_docker_compose_command(
            list(COMPOSE_KILL_COMMAND), check=False, timeout_sec=30
        )
        if result.return_code != 0:
            self.logger.debug(
                "compose kill of %s exited %s: %s",
                self.compose_project_name,
                result.return_code,
                (result.stdout or "")[-300:],
            )

    async def _list_built_images(self) -> ExecResult:
        """``docker image ls`` of the images labelled with this project.

        On the admitted direct path the same daemon answers through its
        Engine API (one request instead of a CLI process; PERF-SBX), as
        ``image ls --quiet --no-trunc`` prints: one full image ID a line.
        """
        direct = getattr(self, "_direct", None)
        if direct is not None:
            label = f"com.docker.compose.project={self.compose_project_name}"
            try:
                rows = (
                    await direct.client.call(
                        "GET",
                        "/images/json",
                        query={"filters": {"label": {label: True}}},
                    )
                ).json()
            except Exception as exc:
                return ExecResult(stderr=describe_exception(exc), return_code=1)
            if not isinstance(rows, list) or not all(
                isinstance(row, dict)
                and isinstance(row.get("Id"), str)
                and _IMAGE_ID.fullmatch(row["Id"])
                for row in rows
            ):
                return ExecResult(
                    stderr="Docker's image list was not a list of images",
                    return_code=1,
                )
            return ExecResult(
                stdout="".join(row["Id"] + "\n" for row in rows), return_code=0
            )
        return await self._docker_cli(
            [
                "image",
                "ls",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label=com.docker.compose.project={self.compose_project_name}",
            ],
            check=False,
            raw_output=True,
        )

    async def _remove_built_images(
        self, listing: asyncio.Future[ExecResult] | None = None
    ) -> None:
        """Remove the images Compose built for this sandbox's project, and only those.

        Pulled images (a task's ``docker_image``, a service's ``image:``) never
        carry the project's label and stay; so do snapshots (their labels are
        blanked) and any image another tag still holds (see
        :func:`~benchflow.sandbox._compose_teardown.built_image_refs`).
        Generated project-private tags are released; shared build names stay.
        Unreferenced builds are deleted through a private alias so concurrent
        new references stay.
        Without ``--force``: an image a container still uses stays too.

        *listing*: :meth:`_list_built_images` already started (stop() lists
        while the containers go; only reading, and the project builds nothing
        then); the removals still come after the containers are gone.
        """
        project = self.compose_project_name
        try:
            listed = await asyncio.wait_for(
                listing if listing is not None else self._list_built_images(),
                timeout=30,
            )
        except (TimeoutError, OSError) as e:
            self.logger.warning(f"Listing the images {project} built failed: {e}")
            return
        if listed.return_code != 0:
            self.logger.warning(
                f"Listing the images {project} built failed: "
                f"{self._diagnostic_text(listed.stderr or listed.stdout or '').strip()[-300:]}"
            )
            return
        ids = sorted(set((listed.stdout or "").split()))
        if not ids:
            return
        try:
            inspected = await asyncio.wait_for(
                self._docker_cli(
                    ["image", "inspect", *ids], check=False, raw_output=True
                ),
                timeout=30,
            )
        except (TimeoutError, OSError) as e:
            self.logger.warning(f"Inspecting the images {project} built failed: {e}")
            return
        if inspected.return_code != 0:
            return
        # These exact names were generated for this unique project, never
        # supplied by the task. Untagging them preserves all other tags and
        # digests, and collects the build if this was its last reference.
        # Do not infer private ownership from a prefix or from build.tags.
        try:
            images = json.loads(inspected.stdout or "[]")
        except (ValueError, TypeError):
            return
        if not isinstance(images, list):
            return
        private_ids: set[str] = set()
        private_tags = {
            image_ref_identity(tag) for tag in self._private_built_image_tags
        }
        for image in images:
            if not isinstance(image, dict) or image.get("Id") not in ids:
                continue
            tags = image.get("RepoTags")
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                continue
            private_refs = {
                tag for tag in tags if image_ref_identity(tag) in private_tags
            }
            if private_refs:
                private_ids.add(image["Id"])
            for ref in sorted(private_refs):
                try:
                    await asyncio.wait_for(
                        self._docker_cli(
                            ["image", "rm", "--no-prune", ref], check=False
                        ),
                        timeout=30,
                    )
                except (TimeoutError, OSError) as e:
                    self.logger.debug("Could not release private build %s: %s", ref, e)
        for image_id in built_image_refs(
            inspected.stdout or "", build_tags=self._built_image_tags
        ):
            if image_id in private_ids:
                continue
            # Deletion by ID can erase a sole tag added since inventory, even
            # without --force. Deleting our random alias can only untag that
            # alias when any other reference exists. Never remove shared build
            # names: another process may have reassigned them in the meantime.
            alias = f"benchflow-cleanup:{uuid.uuid4().hex}"
            try:
                await asyncio.wait_for(
                    self._docker_cli(["image", "tag", image_id, alias]), timeout=30
                )
            except (TimeoutError, OSError, RuntimeError) as e:
                self.logger.debug("Could not pin built image %s: %s", image_id, e)
            finally:
                try:
                    await asyncio.wait_for(
                        self._docker_cli(
                            ["image", "rm", "--no-prune", alias], check=False
                        ),
                        timeout=30,
                    )
                except (TimeoutError, OSError) as e:
                    self.logger.debug(
                        "Could not release cleanup alias %s: %s", alias, e
                    )

    async def stop(self, delete: bool) -> None:
        self._stop_flushing()
        self._forget_kept_identity()
        pending = getattr(self, "_kit_lookup", None)
        if pending is not None:
            self._kit_lookup = None
            pending.cancel()
        # Before main goes: the helper shares its network namespace.
        await self._close_metadata_helper()
        # Bounded chown: a hung agent container will make `docker exec` block
        # forever. We don't need the chown to succeed for correctness — it just
        # makes host-side log reading nicer. Time out fast and continue to the
        # actual teardown. Before `compose up` there is no container to chown.
        try:
            if self._compose_up_attempted:
                await asyncio.wait_for(
                    self._chown_to_host_user(
                        str(SandboxPaths.logs_dir), recursive=True
                    ),
                    timeout=30,
                )
        except TimeoutError:
            self.logger.warning("Chown logs directory timed out; continuing teardown.")
        except Exception as e:
            self.logger.warning(f"Failed to chown logs directory: {e}")
        await self._close_exec_channels()

        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        # Kept containers stop with `-t 5` (SIGKILL after 5 s instead of the
        # default 10 s). Otherwise teardown kills every container at once and
        # removes them with `down -t 0` (_compose_teardown), then removes the
        # images this project built. Each call has a hard deadline. If the
        # daemon is wedged we fall through to a force-kill by compose project
        # label so the rollout's gather() can advance instead of stalling the
        # entire batch.
        listing: asyncio.Future[ExecResult] | None = None
        try:
            if self._keep_containers:
                await self._run_docker_compose_command(
                    ["stop", "-t", "5"], timeout_sec=90
                )
            else:
                # The image listing (0.3-1 s on a busy daemon) runs beside
                # the kill, rm and down; the removals wait for them (PERF-B).
                if self._removes_built_images(delete):
                    listing = asyncio.ensure_future(self._list_built_images())
                direct = getattr(self, "_direct", None)
                if direct is not None:
                    # compose kill, rm --force and down, through the Engine
                    # API (DirectProject.teardown); volumes as below.
                    await asyncio.wait_for(direct.teardown(volumes=False), timeout=90)
                    self._direct = None
                else:
                    if self._compose_up_attempted:
                        await self._kill_project_containers()
                    # No `rm --volumes`: Docker would also remove a volume a
                    # container inherited through volumes_from whose source
                    # is gone. The trial's volumes go below, by their label.
                    await self._run_docker_compose_command(
                        self._down_command(delete), timeout_sec=90
                    )
                if self._removes_volumes(delete):
                    await self._remove_project_volumes()
                if listing is not None:
                    await self._remove_built_images(listing)
        except Exception as e:
            unavailable = _docker_unavailable_reason(e)
            if unavailable is not None and not self._compose_up_attempted:
                # Startup failed because Docker itself is unavailable, and the
                # rollout already reported that; no container was started.
                self.logger.debug(
                    "Skipped teardown of %s: Docker is unavailable (%s) and no "
                    "container was started.",
                    self.compose_project_name,
                    unavailable,
                )
            else:
                self.logger.warning(
                    f"Docker compose down hung/failed ({e}); force-killing project."
                )
                await self._force_kill_project()
                if not self._keep_containers and self._removes_volumes(delete):
                    # The force-kill removes containers only; the volumes the
                    # trial made go as on the normal path, by their label,
                    # within one limit (DOCKER-ANON-VOLUME-LEAK).
                    try:
                        await asyncio.wait_for(
                            self._remove_project_volumes(),
                            timeout=_VOLUME_CALL_TIMEOUT_SEC,
                        )
                    except Exception as volumes_error:
                        self.logger.warning(
                            f"Removing the volumes {self.compose_project_name} "
                            f"made failed: {describe_exception(volumes_error)}"
                        )
                if not self._keep_containers and self._removes_built_images(delete):
                    # The images keep their project label after the force-kill;
                    # without this every failed down left one behind under a
                    # random name (DF-2). Raw docker calls, each bounded.
                    try:
                        await self._remove_built_images()
                    except Exception as images_error:
                        self.logger.warning(
                            f"Removing the images {self.compose_project_name} "
                            f"built failed: {images_error}"
                        )
        finally:
            if listing is not None and not listing.done():
                listing.cancel()
            if listing is not None:
                await asyncio.wait([listing])
                if not listing.cancelled():
                    listing.exception()  # retrieved: the except above relists
            if not self._keep_containers:
                # Whatever teardown left is now garbage the sweep may remove.
                release_project(self.compose_project_name)
                self._release_subnets()
                self._drop_trial_volumes_overlay()
                # Even when the force-kill raised: the trial is over either way.
                env_path = getattr(self, "_container_env_compose_path", None)
                if env_path is not None:
                    env_path.unlink(missing_ok=True)
                    self._container_env_compose_path = None
                self._drop_runtime_compose_file()
                build_path = getattr(self, "_build_compose_path", None)
                if build_path is not None:
                    build_path.unlink(missing_ok=True)
                    self._build_compose_path = None
                await self._release_layers(delete)
            else:
                # Kept containers keep their Compose files, the runtime
                # overlay too: a later `docker compose -f ...` reads them.
                finalizer = getattr(self, "_runtime_compose_finalizer", None)
                if finalizer is not None:
                    finalizer.detach()
                    self._runtime_compose_finalizer = None
        self._snapshot_credentials.clear()
        folder = getattr(self, "_bootstrap_folder", None)
        if folder is not None:
            folder.cleanup()
            self._bootstrap_folder = None
            self._bootstrap_override = None
        await self._delete_deferred_snapshots()
        if delete and not self._keep_containers and self._staged is not None:
            self._staged.remove()

    async def _delete_deferred_snapshots(self) -> None:
        """Remove branch snapshot images that were in use when their fork ended."""
        for ref in sorted(self._deferred_snapshot_refs):
            try:
                gone = await self._remove_snapshot_image(ref)
            except Exception as e:
                gone = False
                self.logger.warning(f"Snapshot image {ref} removal failed: {e}")
            if gone:
                self._deferred_snapshot_refs.discard(ref)
            else:
                self.logger.warning(
                    f"Snapshot image {ref} is still in use (kept container?); "
                    f"remove it with `docker image rm {ref}` when done"
                )

    async def _remove_snapshot_image(self, ref: str) -> bool:
        """Remove only the snapshot ref, without force or recursive parent pruning."""
        result = await self._docker_cli(
            ["image", "rm", "--no-prune", ref], check=False, raw_output=True
        )
        if result.return_code == 0:
            return True
        return "no such image" in f"{result.stderr} {result.stdout}".lower()

    async def _force_kill_project(self) -> None:
        """Last-resort cleanup when `compose down` hangs or fails.

        Lists containers by ``com.docker.compose.project`` label and `docker
        rm -f`s them, then prunes the matching network. We don't propagate
        errors — by the time we're here, the batch just needs to move on.
        """
        self._main_container = None
        project = self.compose_project_name
        label = f"label=com.docker.compose.project={project}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-aq",
                "--filter",
                label,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._docker_client_env(),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            cids = stdout.decode().split()
            for cid in cids:
                rm_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    cid,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=self._docker_client_env(),
                )
                await asyncio.wait_for(rm_proc.wait(), timeout=10)
            net_proc = await asyncio.create_subprocess_exec(
                "docker",
                "network",
                "prune",
                "-f",
                "--filter",
                label,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._docker_client_env(),
            )
            await asyncio.wait_for(net_proc.wait(), timeout=10)
        except Exception as e:
            self.logger.warning(f"Force-kill of compose project {project} failed: {e}")

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        target_parent = str(Path(target_path).parent)
        # ``/`` always exists: ``mkdir -p /`` would cost an exec and do nothing.
        if target_parent not in {"", ".", "/"}:
            await exec_raw(
                self.exec,
                f"mkdir -p {shlex.quote(target_parent)}",
                user="root",
                system_shell=True,
            )
        await self._copy(str(source_path), target_path, into=True)
        await self._apply_upload_mode(target_path, mode)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service container.

        ``service`` defaults to ``"main"``; pass a target service to land the
        directory in an additional vulhub-style container (#248).
        """
        await exec_raw(
            self.exec,
            f"mkdir -p {shlex.quote(target_dir)}",
            user="root",
            service=service,
            system_shell=True,
        )
        await self._copy(f"{source_dir}/.", target_dir, into=True, service=service)
        if sys.platform == "win32":
            # Root's plumbing: the system shell, find and sed (C8).
            await exec_raw(
                self.exec,
                f"find {shlex.quote(target_dir)} -type f "
                "\\( -name '*.sh' -o -name '*.py' \\) "
                "-exec sed -i 's/\\r$//' {} \\;",
                user="root",
                service=service,
                system_shell=True,
            )

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False, service: str = "main"
    ) -> None:
        # Root's own plumbing: /bin/sh by absolute path, PATH the system
        # folders (system_shell), never a sh or chown on the image's PATH,
        # which can lead into the folder being downloaded (/app/.venv/bin).
        if not hasattr(os, "getuid"):
            return
        if self._services_are_kept:
            # A kept service may hold /logs/verifier open (a descriptor in
            # flight on a socket): handing the folder to the host user, whose
            # uid can be the agent user's, would let it write the reward
            # before the copy. ``docker cp`` writes what it copies as the
            # caller anyway.
            return
        flag = "-R " if recursive else ""
        await exec_raw(
            self.exec,
            f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}",
            user="root",
            service=service,
            system_shell=True,
        )

    async def _download_path_stat(self, container: str, source: str) -> dict | None:
        """Read daemon metadata, using the same local/remote CLI transport as cp.

        No program from the task image runs. A 404 is provisional: archive HEAD
        uses that status for both a missing path and a missing container.
        """
        endpoint = (
            f"/containers/{urllib.parse.quote(container, safe='')}/archive?"
            + urllib.parse.urlencode({"path": source})
        )
        response = await self._run_docker_container_command(
            ["system", "dial-stdio"],
            stdin_data=(
                f"HEAD {endpoint} HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"
            ).encode("ascii"),
            check=True,
            timeout_sec=20,
            raw_output=True,
        )
        header, separator, _ = (response.stdout or "").partition("\r\n\r\n")
        lines = header.split("\r\n")
        status = lines[0].split()
        if not separator or len(status) < 2 or status[0] != "HTTP/1.1":
            raise RuntimeError("Docker archive metadata response is not HTTP/1.1")
        if status[1] == "404":
            return None
        if status[1] != "200":
            raise RuntimeError(f"Docker archive metadata failed (HTTP {status[1]})")
        headers = {
            key.lower(): value.strip()
            for line in lines[1:]
            for key, colon, value in [line.partition(":")]
            if colon
        }
        try:
            metadata = json.loads(
                base64.b64decode(headers["x-docker-container-path-stat"], validate=True)
            )
            if (
                not isinstance(metadata, dict)
                or type(metadata.get("mode")) is not int
                or not 0 <= metadata["mode"] <= 0xFFFFFFFF
                or not isinstance(metadata.get("linkTarget", ""), str)
            ):
                raise ValueError("missing file mode")
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Docker archive metadata is invalid") from exc
        return metadata

    async def _check_download_file(self, container: str, source: str) -> None:
        """Prove absence before capture; a failed capture of an existing file is not absence."""
        path = PurePosixPath(source)
        metadata = await self._download_path_stat(container, source)
        # Go's FileMode type bits, as returned by Docker's archive API. ModeDir
        # is the high bit; permissions and setuid/sticky bits are not file types.
        type_mask, directory_mode = 0x8F280000, 0x80000000
        if metadata is not None:
            if metadata["mode"] & type_mask or metadata.get("linkTarget"):
                raise UnsafeDownloadEntryError(f"not a regular remote file: {source}")
            return
        # A successful stat of a real ancestor distinguishes a missing path
        # from a lost container. A dangling link or non-folder ancestor is
        # unsafe evidence, even when Docker reports the descendant as missing.
        for parent in path.parents:
            metadata = await self._download_path_stat(container, str(parent))
            if metadata is None:
                continue
            if metadata["mode"] & type_mask != directory_mode or metadata.get(
                "linkTarget"
            ):
                raise UnsafeDownloadEntryError(f"unsafe remote file parent: {source}")
            raise RemoteFileAbsentError(f"remote file is absent: {source}")
        raise RuntimeError("Docker could not confirm the capture container exists")

    async def _stream_download(
        self,
        source: str,
        target: ReachablePath,
        *,
        directory: bool,
        service: str = "main",
    ) -> None:
        """Publish ``docker cp CONTAINER:source -`` straight into ``target``.

        A Compose task can bind any host folder, the controller's temporary
        folder included, so a Docker download never stages by pathname: the
        tar stream is read here and each member published through held
        descriptors that follow no link and cross no mount. Links
        and special entries in the stream are refused, as for every download.

        Cancellation returns only after every started operation finished and
        staging is gone (TASKMD-R15). Before the commit starts it keeps the
        old destination; once the commit started it is joined, so a caller
        cancelled then returns with the new output in place.
        """
        container = self._main_container if service == "main" else None
        if container is None:
            listed = await self._run_docker_compose_command(
                ["ps", "-q", service], check=False, merge_stderr=False, raw_output=True
            )
            ids = [
                line.strip()
                for line in (listed.stdout or "").splitlines()
                if _CONTAINER_ID.fullmatch(line.strip())
            ]
            if listed.return_code != 0 or len(ids) != 1:
                raise RuntimeError(f"no running container for service {service!r}")
            container = ids[0]
        if not directory:
            await self._check_download_file(container, source)
        read_fd, write_fd = os.pipe()
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                *self._docker_compose_global_args(),
                "cp",
                f"{container}:{source}",
                "-",
                env={**self._docker_compose_env(), **self._compose_transport_env()},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=write_fd,
                stderr=asyncio.subprocess.PIPE,
            )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        reader = os.fdopen(read_fd, "rb")
        errors = asyncio.create_task(process.stderr.read()) if process.stderr else None
        staged = StagedTar(target, directory=directory)
        # Every submitted operation is kept and drained to completion, even
        # through repeated cancellation (TASKMD-R15 B-R14-M1): a thread whose
        # await was cancelled still runs, and returning before it finishes
        # would leave its staging and descriptors to nobody.
        opening = asyncio.ensure_future(asyncio.to_thread(staged.__enter__))
        worker: asyncio.Task | None = None
        try:
            try:
                await drain(opening)
                # The extraction reads a pipe that ends only when the producer
                # does: these awaits pass cancellation on at once (shielded,
                # kept), so the producer is killed below before they are
                # drained. Drain stderr concurrently so a full error pipe
                # cannot stall stdout.
                worker = asyncio.ensure_future(
                    asyncio.to_thread(staged.extract, reader)
                )
                failure: BaseException | None = None
                try:
                    await asyncio.shield(worker)
                except Exception as exc:
                    failure = exc
                    # A refused archive may leave the producer blocked on a full pipe.
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                code = await process.wait()
                stderr = await asyncio.shield(errors) if errors is not None else b""
            except BaseException:
                # Kill first, releasing the pipe read before joining its thread.
                # Never close a BufferedReader on this loop while the worker
                # owns its lock.
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await _settle(process.wait(), worker, errors)
                raise
            finally:
                await _settle(asyncio.to_thread(reader.close))
            # TASKMD-R14 B-M1: commit only once the archive validated AND the
            # producer exited 0. A tar stream may end cleanly at a member
            # boundary while ``docker cp`` reports failure.
            if failure is not None and not isinstance(failure, _TRANSPORT_EOF):
                # The publication's own error (ENOSPC, a refused link) caused
                # the kill; never relabel it as the induced exit status.
                raise failure
            if code != 0:
                detail = stderr.decode(errors="replace").strip()[:500]
                raise RuntimeError(
                    f"docker cp {source!r} failed (exit {code}): {detail}"
                ) from failure
            if failure is not None:
                raise failure
            # A cancelled caller returns only after the commit has finished.
            await drain(asyncio.to_thread(staged.commit))
        except BaseException as exc:
            # The opener was drained above, so whether staging exists is
            # known: it is discarded here before cancellation is passed on.
            if _opened(opening):
                await _settle(asyncio.to_thread(staged.__exit__, type(exc), exc, None))
            raise
        await drain(asyncio.to_thread(staged.__exit__, None, None, None))

    @private_download(directory=False)
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from ``main``; ``private_download`` streams it
        through ``_stream_download``, so this body never runs."""
        raise AssertionError("Docker downloads stream through _stream_download")

    @private_download(directory=True)
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service container.

        ``service`` defaults to ``"main"``; pass a target service to fetch
        target-side verifier output from a vulhub-style container (#248).
        ``private_download`` streams it through ``_stream_download``, so this
        body never runs; the controller creates every received file itself,
        so no chown runs in the container first.
        """
        raise AssertionError("Docker downloads stream through _stream_download")

    # Container snapshot/restore (Branch substrate)
    #
    # ``docker commit`` captures the ``main`` container's filesystem into a
    # local image; restore re-creates ``main`` from that image. Snapshots
    # are container-level only — they include the filesystem the agent has
    # mutated, but **not** mounted host volumes (rollout dir, verifier dir)
    # and **not** sibling compose services. The Branch lifecycle composes
    # this with the Environment-state snapshot (DB dump) that captures the
    # mounted/sidecar state separately (#384).

    @property
    def supports_system_shell(self) -> bool:
        return True

    @property
    def supports_runtime_kit(self) -> bool:
        return True

    @property
    def supports_snapshot(self) -> bool:
        return True

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Commit the current ``main`` container into a re-usable image.

        Uses ``docker commit`` so the snapshot lives in the local Docker
        image store. Returns a :class:`SandboxImage` whose ``ref`` is the
        committed image tag — pass it back to :meth:`restore` to roll the
        ``main`` container back to this checkpoint.
        """
        container_id = await self._main_container_id()
        if not container_id:
            raise SandboxSnapshotNotSupported(
                "DockerSandbox.snapshot requires the main container to be "
                "running; call start() before snapshot()."
            )
        suffix = name or uuid.uuid4().hex[:12]
        tag = _sanitize_docker_image_name(f"bf-snap-{self.environment_name}-{suffix}")
        # Agent credential files stay out of the image (_snapshot_credentials).
        ops = self._credential_ops(container_id)
        stash = await scrub_credentials(ops)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "commit",
                "--change",
                f"LABEL {_bind_targets.LABEL}="
                + json.dumps(_bind_targets.encode(self._snapshot_bind_baseline())),
                # docker commit copies the container's compose labels; with
                # them, teardown takes the snapshot for one of the images the
                # project built (_remove_built_images; `compose down --rmi
                # all` did the same, Compose 5.5), and a kept checkpoint is
                # gone before a retry can use it.
                "--change",
                "LABEL com.docker.compose.project=",
                "--change",
                "LABEL com.docker.compose.service=",
                # Likewise a container of an agent layer passes on the layer's
                # labels: blanked, the snapshot is never taken for the layer.
                *(
                    arg
                    for label in LAYER_LABELS
                    for arg in ("--change", f"LABEL {label}=")
                ),
                container_id,
                tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._docker_client_env(),
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
        finally:
            if stash:
                await put_back_credentials(ops, stash)
        if proc.returncode != 0:
            raise RuntimeError(
                "docker commit failed: "
                f"{(stderr_bytes or stdout_bytes or b'').decode(errors='replace')}"
            )
        digest = (stdout_bytes or b"").decode(errors="replace").strip()
        if stash:
            self._snapshot_credentials[tag] = stash
        self.logger.info(
            f"Snapshot created: {tag} ({digest}); "
            f"{len(stash)} credential file(s) kept out of it"
        )
        from benchflow.sandbox import _path_inspector_state

        meta = {"container_id": container_id, "digest": digest}
        if await self.runtime_kit_for_route() is not None:
            # restore() starts the replacement on the kit's own sleep.
            meta[_SNAPSHOT_KIT_META] = "1"
        # V3's pre-agent inspector state travels with the snapshot, and so
        # does the trust state (its lineage).
        return trust_state.stamp_snapshot(
            self,
            _path_inspector_state.checkpoint(
                self, SandboxImage(provider="docker", ref=tag, meta=meta)
            ),
        )

    def _snapshot_bind_baseline(self) -> _bind_targets.BindTargets:
        if self._bind_baseline is None:
            raise RuntimeError(
                "No first-start bind proof; start a fresh trial before snapshotting"
            )
        return self._bind_baseline

    async def _snapshot_image_bind_baseline(
        self, ref: str
    ) -> _bind_targets.BindTargets:
        receipt = await exec_raw(
            self._docker_cli, ["image", "inspect", "--format", "{{json .Config}}", ref]
        )
        try:
            config = json.loads(receipt.stdout or "null") or {}
            if not isinstance(config, dict):
                raise ValueError("missing image configuration")
            labels = config.get("Labels") or {}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Snapshot has no complete bind proof; create a fresh checkpoint"
            ) from exc
        if not isinstance(labels, dict) or _bind_targets.LABEL not in labels:
            raise RuntimeError(
                "Snapshot has no bind identity proof; create a fresh checkpoint"
            )
        baseline = _bind_targets.decode(labels[_bind_targets.LABEL])
        self._restore_config = _inert_bootstrap.original_config(config)
        # The replacement's execs get the snapshot's Config.Env, as a fresh
        # start's get the image's (startup_name; _inert_bootstrap.overlay).
        refused = refused_startup_names(
            [*(config.get("Env") or ()), *(self._restore_config.get("Env") or ())]
        )
        if refused:
            raise StartupNamesRefused(
                startup_refusal(refused, "the checkpoint's image")
            )
        return baseline

    async def _verify_replacement_bind_targets(
        self, container: str, expected: _bind_targets.BindTargets
    ) -> None:
        if not _CONTAINER_ID.fullmatch(container):
            raise RuntimeError("Cannot identify restored container for bind proof")
        try:
            targets = await _bind_targets.inspect_targets(
                self, container, await self._inspect_container(container)
            )
            _bind_targets.verify({"main": targets}, {"main": expected.get("main", {})})
            # Include sidecars: isolated branches can start a new project.
            ids = await compose_container_ids(self._run_docker_compose_command)
            actual = await _bind_targets.pin_project(self, "\n".join(ids), expected)
        except Exception:
            await self._docker_cli(["rm", "-f", container], check=False)
            raise
        self._bind_baseline = actual

    async def restore(self, image: SandboxImage) -> None:
        """Restore the ``main`` container from a previously committed image.

        Inspects the live ``main`` container, removes it, then ``docker run``s
        a replacement from ``image.ref`` with the inspected host
        configuration. The bind mounts matter most: without them the rollout's
        ``verifier``/``agent``/``artifacts`` output stays inside the container
        and a verifier reward is silently lost. Sibling compose services are
        untouched, matching the container-only scope of the Sandbox layer.

        The old container is removed with ``docker rm -f`` straight from
        running: a ``docker stop`` first only waited out the 10 s grace of a
        ``sleep`` that ignores SIGTERM, and left a stopped container that an
        older BenchFlow's prune on the same daemon could start removing
        ("removal of container ... is already in progress").

        Raises :class:`~benchflow.sandbox.protocol.SandboxRestoreHostConfigUnavailable`
        before removing anything when the live container cannot be resolved or
        inspected, or its host configuration cannot be replayed.
        """
        # The container is replaced: what the start probe kept no longer holds.
        self._forget_kept_identity()
        self._runtime_kit_unknown()
        # The replacement inherits the snapshot's trust state before any of
        # its processes start.
        trust_state.restore_from(self, image)
        if image.provider != "docker":
            raise SandboxSnapshotNotSupported(
                f"DockerSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )

        project_name = self.compose_project_name
        default_network = f"{project_name}_default"

        container_id = await self._main_container_id()
        if not container_id:
            raise SandboxRestoreHostConfigUnavailable(
                f"DockerSandbox.restore({image.ref!r}) cannot resolve the "
                f"'main' container of compose project {project_name!r}, so its "
                "bind mounts, network and resource limits cannot be replayed"
            )
        # Inspected *before* removal — the host config only exists while
        # the container does.
        try:
            inspected = await self._inspect_container(container_id)
        except RuntimeError as exc:
            raise SandboxRestoreHostConfigUnavailable(
                f"DockerSandbox.restore({image.ref!r}) cannot read the host "
                f"config of the 'main' container {container_id!r}, so the "
                f"replacement cannot be made equivalent to it: {exc}"
            ) from exc
        if self.task_env_config.network_mode in (
            NetworkMode.DENYLIST,
            NetworkMode.ALLOWLIST,
        ):
            raise SandboxRestoreHostConfigUnavailable(
                "Docker snapshot restore cannot preserve live "
                f"{self.task_env_config.network_mode.value.upper()} firewall rules; "
                "container left intact"
            )
        from benchflow.sandbox import _path_inspector_state

        path_inspector = _path_inspector_state.read(image)
        replayed = _replayed_run_args(inspected, default_network=default_network)
        # Read the trusted receipt from the image, never the agent filesystem
        # or a new stat of the old container after the agent has run.
        self._restore_config = inspected.get("Config") or {}
        # The replacement is a new container with this sandbox's log binds
        # and its own task copy's binds: those are proved against this
        # sandbox's own first-start receipt, every bind carried over from
        # the checkpoint against its label (DET-R2, DET-R2b; an isolated
        # branch child, a retry from a checkpoint).
        expected = _bind_targets.with_own_log_binds(
            await self._snapshot_image_bind_baseline(image.ref),
            getattr(self, "_bind_baseline", None),
            getattr(self, "_staged_bind_targets", None),
        )
        original = self._restore_config
        self._bootstrap_configs = {
            **getattr(self, "_bootstrap_configs", {}),
            "main": original,
        }
        owned = ((inspected.get("Config") or {}).get("Labels") or {}).get(
            _BENCHFLOW_OWNED_LABEL
        )
        owned_label = (
            ["--label", f"{_BENCHFLOW_OWNED_LABEL}={owned}"]
            if owned is not None
            else []
        )
        await self._remove_replaced_container(container_id)

        new_name = f"{project_name}-main-restored-{uuid.uuid4().hex[:8]}"
        # PID 1 of the replacement is BenchFlow's (C8, PT-C8-R3-6): the
        # snapshot's entrypoint and its PATH's ``sleep`` are whatever the
        # agent left in the image, so neither runs. An absolute ``sleep``
        # (the protected kit's when the snapshot holds it) starts with the
        # loader and start-up variables empty; the values the container had
        # stay as data under their _BF_IMAGE_ names, where the task's own
        # commands get them back (capture_pid1_image_env).
        kit_snapshot = image.meta.get(_SNAPSHOT_KIT_META) == "1"
        entrypoint = f"{KIT_BIN}/sleep" if kit_snapshot else "/bin/sleep"
        had = _container_env(inspected)
        pid1_env: list[str] = []
        for env_name in SYSTEM_SHELL_EMPTY_VARS:
            pid1_env += ["--env", f"{env_name}="]
            if had.get(env_name):
                pid1_env += ["--env", f"{IMAGE_ENV_PREFIX}{env_name}={had[env_name]}"]

        run_cmd = [
            "run",
            "--detach",
            "--name",
            new_name,
            "--label",
            f"com.docker.compose.project={project_name}",
            "--label",
            "com.docker.compose.service=main",
            *owned_label,
            # This process owns the replacement, whatever the snapshot image
            # carried over from the container it was committed from.
            "--label",
            f"{_BENCHFLOW_PROCESS_LABEL}={process_token()}",
            *replayed,
            "--security-opt",
            "no-new-privileges:true",
            *_inert_bootstrap.run_args(
                shell=f"{KIT_BIN}/sh" if kit_snapshot else SYSTEM_SHELL
            ),
            *pid1_env,
            "--label",
            f"{_inert_bootstrap.LABEL}={_inert_bootstrap.config_receipt(original)}",
            image.ref,
            "-c",
            _inert_bootstrap.restore_bootstrap(kit_snapshot),
            "benchflow-restore",
            entrypoint,
            "infinity",
        ]
        result = await exec_raw(self._docker_cli, run_cmd, check=False)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker run from snapshot {image.ref!r} failed: "
                f"{self._diagnostic_text(result.stderr or result.stdout)}"
            )
        replacement = result.stdout.strip()
        try:
            await _inert_bootstrap.wait_ready(self, replacement)
            await self._verify_replacement_bind_targets(replacement, expected)
            await self._docker_cli(["kill", "--signal", "USR1", replacement])
        except BaseException:
            await self._docker_cli(["rm", "-f", replacement], check=False)
            raise
        self.logger.info(
            "Snapshot restored: %s -> %s (replayed host config: %s)",
            image.ref,
            new_name,
            " ".join(replayed) or "none",
        )
        _path_inspector_state.restored(self, path_inspector)
        stash = self._snapshot_credentials.get(image.ref)
        if stash:
            await put_back_credentials(self._credential_ops(new_name), stash)

    async def _remove_replaced_container(self, container_id: str) -> None:
        """``docker rm -f`` the container restore replaces.

        Gone already, or already being removed by someone else, is the same
        outcome; any other failure raises before the replacement is made.
        """
        result = await self._docker_cli(
            ["rm", "-f", container_id], check=False, raw_output=True
        )
        if result.return_code == 0:
            return
        message = f"{result.stderr} {result.stdout}".lower()
        if "no such container" in message or "already in progress" in message:
            self.logger.info(
                "Container %s was already gone or being removed: %s",
                container_id,
                self._diagnostic_text(result.stderr or result.stdout).strip(),
            )
            return
        raise RuntimeError(
            f"docker rm -f {container_id} failed: "
            f"{self._diagnostic_text(result.stderr or result.stdout)}"
        )

    async def adopt_snapshot(self, image: SandboxImage) -> None:
        """Remember the live container's credential files for ``image``, a
        snapshot of this state taken elsewhere (a kept checkpoint), so a
        restore from it puts them back. The live files are left as they are."""
        trust_state.admit(self, "main", trust_state.IngressKind.SNAPSHOT_RESTORE)
        ops = self._credential_ops(await self._main_container_id())
        stash = await scrub_credentials(ops)
        if stash:
            await put_back_credentials(ops, stash)
            self._snapshot_credentials[image.ref] = stash

    def _credential_ops(self, container: str) -> _DockerCredentialOps:
        return _DockerCredentialOps(self, container)

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        """Remove a committed snapshot image.

        The image a restored container runs from cannot be removed without
        force (and forcing only untags it, leaving the layers behind), so an
        image still in use is removed by :meth:`stop` after the containers
        are gone. Returns True when the image is gone now.
        """
        if image.provider != "docker":
            raise SandboxSnapshotNotSupported(
                f"DockerSandbox.delete_snapshot cannot delete a {image.provider!r} "
                f"snapshot (got ref={image.ref!r})"
            )
        self._snapshot_credentials.pop(image.ref, None)
        if await self._remove_snapshot_image(image.ref):
            self._deferred_snapshot_refs.discard(image.ref)
            return True
        self._deferred_snapshot_refs.add(image.ref)
        self.logger.info(
            f"Snapshot image {image.ref} is still in use; removing it at stop()"
        )
        return False

    async def script_container_config(self) -> dict[str, Any]:
        """Use Docker's exec environment even when PID 1's uid differs."""
        container_id = await self._main_container_id()
        if not container_id:
            raise RuntimeError("script: could not locate the Docker task container")
        config = (await self._inspect_container(container_id)).get("Config")
        if not isinstance(config, dict):
            raise RuntimeError(
                "script: could not read the Docker container environment"
            )
        return config

    def _service_loader_switches(self, service: str) -> list[str]:
        """The valued loader switches *service* runs with: a service S1 did
        not bootstrap keeps its image's and the task's environment in
        Config.Env (the prototype's, as Docker created it), so root's shell
        there removes them before it loads (route_exec_argv's *remove*).
        Main and every bootstrapped service were created with them cleared."""
        if service == "main" or service in (
            getattr(self, "_bootstrap_services", None) or ()
        ):
            return []
        configs = getattr(self, "_bootstrap_configs", None)
        if not isinstance(configs, dict):
            return []
        return loader_switches((configs.get(service) or {}).get("Env"))

    async def identity_container_config(self, service: str = "main") -> dict[str, Any]:
        if service == "main":
            return await self.script_container_config()
        container_id = await self._service_container_id(service)
        if not container_id:
            raise RuntimeError(
                "sandbox identity: could not locate the verifier service; "
                "start the service named by verifier.service before agent setup"
            )
        config = (await self._inspect_container(container_id)).get("Config")
        if not isinstance(config, dict):
            raise RuntimeError(
                "sandbox identity: cannot read verifier service configuration; "
                "check verifier.service and its running container"
            )
        return config

    async def _inspect_container(self, container_id: str) -> dict[str, Any]:
        """``docker inspect`` one container as a dict — raises if it cannot.

        On the direct path the same object comes from the Engine API."""
        direct = getattr(self, "_direct", None)
        if direct is not None:
            try:
                return await direct.inspect(container_id)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"docker inspect {container_id!r} failed: "
                    f"{_direct_docker.daemon_message(exc)}"
                ) from exc
        result = await self._docker_cli(
            ["inspect", container_id], check=False, raw_output=True
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker inspect {container_id!r} failed: "
                f"{self._diagnostic_text(result.stderr or result.stdout)}"
            )
        try:
            payload = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"docker inspect {container_id!r} returned unparseable JSON: {exc}"
            ) from exc
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
        ):
            raise RuntimeError(
                f"docker inspect {container_id!r} returned no container object"
            )
        return payload[0]

    async def _main_container_id(self) -> str | None:
        """Return the container id of the ``main`` compose service, or None."""
        return await self._service_container_id("main")

    async def _service_container_id(self, service: str) -> str | None:
        """Resolve a service in this trial's project on the selected daemon."""
        try:
            # Avoid re-reading task compose files: their warnings can precede
            # the id. The raw CLI keeps stderr separate and routes to the same
            # daemon on remote Docker; restored containers retain these labels.
            result = await self._docker_cli(
                [
                    "ps",
                    "-q",
                    "--no-trunc",
                    "--filter",
                    f"label=com.docker.compose.project={self.compose_project_name}",
                    "--filter",
                    f"label=com.docker.compose.service={service}",
                ],
                check=False,
                raw_output=True,
            )
        except RuntimeError:
            return None
        if result.return_code != 0:
            return None
        ids = [
            line.strip() for line in (result.stdout or "").splitlines() if line.strip()
        ]
        if len(ids) != 1 or not _CONTAINER_ID.fullmatch(ids[0]):
            return None
        return ids[0]

    async def _engine_info_json(self) -> str:
        """The daemon's ``docker info`` as JSON (empty when it cannot say)."""
        result = await self._docker_cli(
            ["info", "--format", "{{json .}}"], check=False, raw_output=True
        )
        return result.stdout if result.return_code == 0 else ""

    async def _docker_cli(
        self, args: list[str], check: bool = True, *, raw_output: bool = False
    ) -> ExecResult:
        """Run a raw ``docker`` CLI command — bypasses compose.

        A ``docker run`` gets the client configuration without the
        operator's proxies, which it would copy into the container's
        environment, as Compose's calls do."""
        client = []
        connection = False
        if args and args[0] == "run":
            from benchflow.task.compose_env import hermetic_docker_config

            client = ["--config", hermetic_docker_config()]
        elif args[:1] == ["inspect"] and not self._remote_daemon:
            # The daemon the frozen launch's Compose calls made the containers
            # on (the bind proof reads these mounts; DET-R2b round 4,
            # M-R3-A1): its flags, and no connection value of the
            # environment's beside them. Remote Docker routes its own calls.
            client = self._docker_args()
            connection = True
        if args[:1] and args[0] not in _DOCKER_KEEPS_CONTAINERS:
            # run, rm, ... (restore replaces the main container this way).
            self._main_container = None
        client_env = self._docker_client_env()
        if args[:1] == ["run"]:
            client_env = _inert_bootstrap.client_env(client_env)
        client_env = _without_loader_names_if_removed(args, client_env)
        if connection:
            client_env = {
                key: value
                for key, value in (
                    os.environ if client_env is None else client_env
                ).items()
                if key not in _DOCKER_CONNECTION_ENV
            }
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *client,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=client_env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        result = ExecResult(
            stdout=(stdout_bytes or b"").decode(errors="replace"),
            stderr=(stderr_bytes or b"").decode(errors="replace"),
            return_code=proc.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"docker {' '.join(args)} failed: {result.stderr or result.stdout}"
            )
        return result

    async def services(self) -> list[str]:
        """List compose service names defined for this sandbox.

        Includes BenchFlow's own ``main`` service plus any additional
        services the task declares in its ``docker-compose.yaml``
        (vulhub-style target/database containers — see #248).

        Config diagnostics are captured separately. The stdout record is
        also filtered to the Docker Compose service naming grammar.
        """
        result = await self._run_docker_compose_command(
            ["config", "--services"], check=True, raw_output=True
        )
        return _filter_compose_service_names(result.stdout or "")

    async def script_runner_exec(self, command: str, **kwargs: Any) -> ExecResult:
        """Bootstrap root supervision before any image-PATH shell can run."""
        from benchflow.sandbox._identities import (
            protected_exec_argv,
            removed_env_names,
        )

        config = await self.script_container_config()
        # Clear every configured variable before /usr/bin/env itself loads.
        # A list of loader hooks would miss other runtimes and future hooks.
        # Docker's implicit defaults contain no startup hooks; env -i drops
        # those too, and only this fixed PATH reaches the root shell.
        route = await self.exec_route("root", system_shell=True)
        argv = protected_exec_argv(
            config,
            command,
            shell=(route.bash, "--noprofile", "--norc", "-p", "-c"),
            route=route,
            cwd=kwargs.get("cwd"),
        )
        return await self._run_docker_compose_command(
            argv,
            check=False,
            timeout_sec=kwargs.get("timeout_sec"),
            merge_stderr=False,
            raw_output=True,
            unset_env=removed_env_names(argv),
        )

    async def identity_exec(
        self, command: str, *, service: str = "main", **kwargs: Any
    ) -> ExecResult:
        from benchflow.sandbox._identities import (
            protected_exec_argv,
            removed_env_names,
        )

        config = await self.identity_container_config(service)
        argv = protected_exec_argv(
            config,
            command,
            service=service,
            route=await self.exec_route("root", system_shell=True, service=service),
        )
        return await self._run_docker_compose_command(
            argv,
            check=False,
            timeout_sec=kwargs.get("timeout_sec"),
            merge_stderr=False,
            raw_output=True,
            unset_env=removed_env_names(argv),
        )

    async def script_runner_upload(self, source: str, target: str) -> None:
        # The parent already exists. Avoid upload_file's root mkdir through
        # the ordinary exec transport, which preserves the image's PATH.
        await self._run_docker_compose_command(["cp", source, f"main:{target}"])

    # The env goes on the exec's stdin (exec below).
    exec_env_is_private = True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        system_shell: bool = False,
        raw_output: bool = False,
    ) -> ExecResult:
        """Run a command in a compose service container.

        ``service`` defaults to ``"main"`` (the agent container). Pass a
        different service name to target an additional container declared
        in the task's ``docker-compose.yaml`` — e.g. to inject a flag into
        a vulnerable target before the agent runs, or to verify exploit
        success by inspecting target-side state afterwards (#248).
        ``system_shell`` starts ``/bin/sh`` by absolute path with the image's
        loader variables set empty (``-e LD_LIBRARY_PATH=`` and the rest of
        :data:`~benchflow.sandbox._base.SYSTEM_SHELL_EMPTY_VARS`) and PATH set
        to the system folders first, so neither the image's PATH nor its
        loader variables pick the shell, the libraries it loads, or the
        env-file wrapper's tools.
        ``exec_raw`` (or ``raw_output``) preserves both streams for harness parsers; only
        copies shown in diagnostics may be redacted by the transport.
        """
        requested_user = user
        user = self._resolve_user(user)
        # C8: plumbing (uid 0 or the image's user, outside task_command) pins
        # PATH and never sees the loader variables; the task's own command as
        # root starts without them too and gets them back before it runs.
        route = await self.exec_route(
            requested_user, system_shell=system_shell, service=service
        )
        pin, task_root = route.pin, route.task_root
        env = self._merge_env(
            env,
            requested_user,
            plumbing=pin is not None,
            explicit_root=route.explicit_root,
        )
        if not pin and not self._services_are_kept:
            restored = _inert_bootstrap.restored_env(self, service)
            if task_root:
                # The receipt holds image defaults, not an explicit task override.
                # Keep them inert until PT's provenance-aware restore admits them.
                restored = {
                    f"_BF_IMAGE_{name}": value for name, value in restored.items()
                }
            env = {**restored, **(env or {})}

        raw_output = raw_output or raw_output_requested()
        # ``docker exec`` on the main container when start() knows it: the
        # same process as ``docker compose exec -T`` (no TTY, no stdin, the
        # container's own user, directory and environment unless set here),
        # without starting the compose CLI for every command.
        container = self._main_container if service == "main" else None
        exec_command: list[str] = ["exec"] if container else ["exec", "-T"]
        removed: list[str] = []

        if cwd:
            exec_command.extend(["-w", cwd])

        if user is not None:
            exec_command.extend(["-u", str(user)])

        if pin is not None or task_root:
            # Empty values only: nothing secret reaches the process list.
            exec_command.extend(system_shell_env_flags())
            if container is None:
                # Bare removals of the service's own valued loader switches
                # (route_exec_argv's *remove*); main is always bootstrapped.
                removed = self._service_loader_switches(service)
                exec_command.extend(arg for name in removed for arg in ("-e", name))

        exec_command.append(container or service)

        # Env vars are read from stdin and sourced, rather than passed as
        # `-e KEY=VALUE` flags or written to a file: the flags are visible in
        # `ps aux` on the host, which would leak secrets — e.g. the verifier's
        # [verifier.env] LLM-judge API keys. Inside PT's
        # route (C8): plumbing pins its PATH; a task command gets the image's
        # kept-aside values back before its own (2).
        if pin is None and task_root:
            command = (
                restore_image_env(
                    env or (),
                    tools=route.tools if route.kit and route.root else None,
                )
                + command
            )

        def routed(inner: str) -> list[str]:
            if pin is not None:
                inner = with_system_path(inner, pin)
            else:
                inner = _root_exec_command(inner, route.explicit_root, route.root_path)
                if task_root:
                    inner = capture_pid1_image_env(route.tools) + inner
            # Use POSIX ``sh`` rather than ``bash``: with multi-service support
            # (#248), ``exec(..., service=...)`` can target arbitrary task
            # containers — Alpine/distroless/minimal DB images frequently ship
            # no ``/bin/bash``. The wrapped command (env-file sourcing,
            # ``trap``, ``base64 -d``, ``set -a``/``. file``) uses only POSIX
            # constructs. The shell is the route's: the protected kit's for
            # root once the sandbox has one (C8).
            return [*exec_command, route.shell, "-c", inner]

        # The env goes on the exec's stdin (wrap_command_with_env_stdin): no
        # value on the host's or the container's command lines, no size limit
        # of one argument, no file (b156 PY8-MAIN-30/32/33). A command over
        # the transport bound (benchflow.sandbox._transport) goes on the same
        # stdin, after its env lines, and a constant reader runs it.
        stdin_data = env_stdin_body(env) if env else None
        fitted = fit(
            routed,
            wrap_command_with_env_stdin(command) if env else command,
            cat=reader_cat(route),
        )
        exec_command = fitted.sent
        if fitted.payload is not None:
            stdin_data = fitted.frame(
                (env_file_body(env) if env else "").encode() + command.encode()
            )
        if stdin_data is not None and container:
            # `docker exec` needs -i to pass stdin; `docker compose exec`
            # passes it by default.
            exec_command.insert(1, "-i")
        _transport.sent("Docker", exec_command)

        if container:
            channel_began = time.monotonic()
            on_channel = await self._exec_on_channel(
                route,
                container=container,
                user=user,
                cwd=cwd,
                argv=exec_command[-3:],
                stdin_data=stdin_data,
                timeout_sec=timeout_sec,
                raw_output=raw_output,
            )
            if on_channel is not None:
                return on_channel
            if timeout_sec is not None:
                # A step that falls back after a channel death gets the
                # time it has left, not a second full timeout (review R3-L1).
                spent = time.monotonic() - channel_began
                if spent >= 1:
                    if spent >= timeout_sec:
                        raise RuntimeError(
                            f"Command timed out after {timeout_sec} seconds"
                        )
                    timeout_sec = max(1, math.ceil(timeout_sec - spent))

        extra: dict[str, Any] = {"raw_output": True} if raw_output else {}
        if stdin_data is not None:
            extra["stdin_data"] = stdin_data
        if container:
            return await self._run_docker_container_command(
                exec_command, check=False, timeout_sec=timeout_sec, **extra
            )
        if removed:
            # The bare names must not be filled in from Compose's environment.
            extra["unset_env"] = removed
        return await self._run_docker_compose_command(
            exec_command, check=False, timeout_sec=timeout_sec, **extra
        )

    async def _exec_on_channel(
        self,
        route: ExecRoute,
        *,
        container: str,
        user: str | int | None,
        cwd: str | None,
        argv: list[str],
        stdin_data: bytes | None,
        timeout_sec: int | None,
        raw_output: bool,
    ) -> ExecResult | None:
        """Run *argv* through the main container's command channel, as the
        ``docker exec`` exec() built would; None when this command takes its
        own ``docker exec`` (see :mod:`~benchflow.sandbox._exec_channel`).

        Only BenchFlow's own root plumbing on the protected kit qualifies
        (uid 0, a pinned route, started with the empty loader values the
        channel was started with). The task's own commands as root keep
        their own ``docker exec``: one may kill processes by pattern, and
        the helper is a process in the container.
        """
        if not (
            self.exec_channel
            and route.kit
            and route.root
            and route.pin is not None
            and (cwd is None or cwd.startswith("/"))
            and os.environ.get("BENCHFLOW_EXEC_CHANNEL", "1") != "0"
        ):
            return None
        key = (container, self._exec_generation, None if user is None else str(user))
        channel = self._ready_exec_channel(key)
        if channel is None:
            return None
        started = time.monotonic()
        try:
            if isinstance(channel, WarmExecPool):
                run = channel.run(argv, cwd=cwd, stdin_data=stdin_data)
            else:
                run = channel.run(
                    argv,
                    cwd=cwd,
                    stdin_data=stdin_data,
                    deadline=(
                        None if timeout_sec is None else time.monotonic() + timeout_sec
                    ),
                )
            if timeout_sec is not None:
                output = await asyncio.wait_for(run, timeout=timeout_sec)
            else:
                output = await run
        except (TimeoutError, ChannelDeadline):
            # As when the docker exec client is killed: the command goes on;
            # one the helper got after its deadline never starts.
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None
        except ChannelUnavailable as exc:
            self.logger.debug("exec channel: not started (%s); docker exec", exc)
            return None
        except ChannelUnconfirmed as exc:
            # Only the dead helper's journal can tell whether the step
            # started; one it does not list takes its own docker exec
            # .
            assert isinstance(channel, ExecChannel)
            started_ids = await self._dead_channel_starts(channel)
            if started_ids is not None and exc.ident not in started_ids:
                self.logger.debug(
                    "exec channel: died before step %s started; docker exec",
                    exc.ident,
                )
                return None
            raise RuntimeError(
                f"{exc}; the command may or may not have run, and BenchFlow does "
                "not start it again"
            ) from None
        except ChannelLost as exc:
            raise RuntimeError(
                f"{exc}; the command may or may not have run, and BenchFlow does "
                "not start it again"
            ) from None
        self.logger.debug(
            "exec channel: rc=%s in %.3fs",
            output.return_code,
            time.monotonic() - started,
        )
        if raw_output:
            stdout_bytes, stderr_bytes = output.stream(1), output.stream(2)
        else:
            stdout_bytes, stderr_bytes = output.merged(), b""
        return ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=output.return_code,
        )

    def _ready_exec_channel(
        self, key: tuple[str, int, str | None]
    ) -> ExecChannel | WarmExecPool | None:
        """The channel for *key* once it runs. The first ask starts it in the
        background; that command and the next ones take docker exec until
        it is ready. Channels of another container or generation close."""
        channels = self._exec_channels
        if channels is None:
            channels = self._exec_channels = {}
        for other in [k for k in channels if k[:2] != key[:2]]:
            # Kept, so stop() still waits for the close.
            self._exec_channels_closing = [
                *(self._exec_channels_closing or []),
                asyncio.ensure_future(self._close_exec_channel(channels.pop(other))),
            ]
        state = channels.get(key, False)
        if state is False:
            channels[key] = asyncio.ensure_future(self._start_exec_channel(key))
            return None
        if isinstance(state, asyncio.Future):
            if not state.done():
                return None
            state = None if state.cancelled() or state.exception() else state.result()
            channels[key] = state
        if isinstance(state, (ExecChannel, WarmExecPool)) and state.alive:
            return state
        if isinstance(state, ExecChannel):
            # Its helper died (a task step that killed every python
            # process, say). Start a new one, a few times per sandbox
            # generation; meanwhile steps take docker exec.
            restarts = self._exec_channel_restarts = (
                self._exec_channel_restarts or 0
            ) + 1
            if restarts <= _EXEC_CHANNEL_RESTARTS:
                self._exec_channels_closing = [
                    *(self._exec_channels_closing or []),
                    asyncio.ensure_future(self._close_exec_channel(state)),
                ]
                channels[key] = asyncio.ensure_future(
                    self._restart_exec_channel(key, state)
                )
            else:
                channels[key] = None
        return None

    async def _restart_exec_channel(
        self, key: tuple[str, int, str | None], dead: ExecChannel
    ) -> ExecChannel | WarmExecPool | None:
        """A new channel for *key*, once *dead*'s helper is gone: a helper
        whose transport died may live on, and must never run a step beside
        the new one. When that is not proven, steps keep
        their own docker exec."""
        if await self._dead_channel_starts(dead) is None:
            self.logger.debug("exec channel: old helper not proven gone; docker exec")
            return None
        return await self._start_exec_channel(key)

    def _dead_channel_starts(self, channel: ExecChannel) -> asyncio.Future[Any]:
        """The step ids *channel*'s dead helper had started, from its journal,
        or None when that could not be read; once per channel. Its own
        docker exec first kills the helper (still running when only the
        transport died) and waits until it is gone, so the journal is final."""
        if channel.death_check is None:
            channel.death_check = asyncio.ensure_future(
                self._read_start_journal(channel)
            )
        return channel.death_check

    async def _read_start_journal(self, channel: ExecChannel) -> frozenset[int] | None:
        if not channel.verify_argv or not channel.journal or not channel.helper_pid:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                *channel.verify_argv,
                channel.journal,
                str(channel.helper_pid),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._docker_process_env(),
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            self.logger.debug("exec channel: journal read did not start (%s)", exc)
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None
        if process.returncode != 0:
            return None
        return parse_journal(stdout)

    async def bfx_call(
        self, cmd: str, args: Mapping[str, Any], *, timeout_sec: int
    ) -> dict[str, Any]:
        """Run one bfx command as root in ``main`` and return its result.

        This is I1's bfx route (TRUSTED-RUNTIME v2): bfx is the first
        program the exec starts, by absolute path from the read-only
        ``/.bfx``, with no shell or PATH; its arguments travel on stdin.
        Raises :class:`BfxUnavailable` where bfx is not delivered (the caller
        keeps its other path), and bfx's own errors as
        :class:`~benchflow._bfx.client.BfxError`.
        """
        if getattr(self, "bfx_delivery", None) is None:
            raise BfxUnavailable("bfx is not delivered to this sandbox")
        container = (
            getattr(self, "_main_container", None) or await self._main_container_id()
        )
        if not container:
            raise BfxUnavailable("the main container is not running")
        argv = [
            "exec",
            "-i",
            "-u",
            "0",
            *system_shell_env_flags(),
            container,
            *bfx_client.one_shot_argv(cmd, args_on_stdin=True),
        ]
        _transport.sent("Docker", argv)
        result = await self._run_docker_container_command(
            argv,
            check=False,
            timeout_sec=timeout_sec,
            raw_output=True,
            stdin_data=bfx_client.stdin_args_line(args),
        )
        return bfx_client.parse_answer(result.stdout)

    @contextlib.asynccontextmanager
    async def bfx_session(
        self, *, timeout_sec: float = 15.0
    ) -> AsyncIterator[bfx_client.BfxServe]:
        """``bfx serve`` as root in ``main`` over one exec, for host-driven
        steps that send several requests (facts, then the changes the host
        decides). I1's bfx route, as :meth:`bfx_call`. Raises
        :class:`BfxUnavailable` where bfx is not delivered or did not start."""
        if getattr(self, "bfx_delivery", None) is None:
            raise BfxUnavailable("bfx is not delivered to this sandbox")
        container = (
            getattr(self, "_main_container", None) or await self._main_container_id()
        )
        if not container:
            raise BfxUnavailable("the main container is not running")
        launch = getattr(self, "_launch", None)
        client = (
            launch.docker_args
            if launch is not None
            else self._docker_compose_global_args()
        )
        argv = [
            "docker",
            *client,
            "exec",
            "-i",
            "-u",
            "0",
            *system_shell_env_flags(),
            container,
            BFX_PATH,
            "serve",
        ]
        _transport.sent("Docker", argv)
        serve = bfx_client.BfxServe(argv, self._docker_process_env())
        try:
            await serve.start(timeout_sec)
        except (
            bfx_client.BfxError,
            bfx_client.BfxProtocolError,
            OSError,
            TimeoutError,
        ) as exc:
            await serve.close()
            raise BfxUnavailable(f"bfx serve did not start: {exc}") from exc
        try:
            yield serve
        finally:
            await serve.close()

    async def bfx_tar_out(
        self, args: Mapping[str, Any], destination: Path, *, timeout_sec: int
    ) -> dict[str, Any]:
        """``bfx tar-out`` as root in ``main``, its archive written straight to
        the new host file *destination*; returns bfx's report.

        I1's bfx route, as :meth:`bfx_call`. bfx reads beneath
        ``args["root"]`` through descriptors it opened: it follows no link and
        enters no mount. The archive is data on the host
        and nothing in the sandbox reads it back. Raises as :meth:`bfx_call`.
        On a timeout the exec is killed and :class:`BfxAmbiguous` raised.
        """
        if getattr(self, "bfx_delivery", None) is None:
            raise BfxUnavailable("bfx is not delivered to this sandbox")
        container = (
            getattr(self, "_main_container", None) or await self._main_container_id()
        )
        if not container:
            raise BfxUnavailable("the main container is not running")
        launch = getattr(self, "_launch", None)
        client = (
            launch.docker_args
            if launch is not None
            else self._docker_compose_global_args()
        )
        argv = [
            "docker",
            *client,
            "exec",
            "-i",
            "-u",
            "0",
            *system_shell_env_flags(),
            container,
            *bfx_client.one_shot_argv("tar-out", args_on_stdin=True),
        ]
        _transport.sent("Docker", argv)
        with open(destination, "xb") as archive:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=archive,
                stderr=asyncio.subprocess.PIPE,
                env=self._docker_process_env(),
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(bfx_client.stdin_args_line(args)),
                    timeout_sec,
                )
            except BaseException as exc:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await asyncio.shield(process.wait())
                if isinstance(exc, TimeoutError):
                    raise bfx_client.BfxAmbiguous(
                        "bfx tar-out did not finish in time", 0, True
                    ) from None
                raise
        answer = [
            line
            for line in (stderr or b"").decode("utf-8", "replace").splitlines()
            if line.startswith("{")
        ]
        return bfx_client.parse_answer(answer[-1] if answer else "")

    async def _start_exec_channel(
        self, key: tuple[str, int, str | None]
    ) -> ExecChannel | WarmExecPool | None:
        """Start the channel for *key*: ``docker exec -i`` with the key's
        ``-u`` and the empty loader values, in the image's working folder.
        The helper is bfx (``/.bfx/bfx serve --frames``) where it is
        delivered. Without bfx the route is today's (TRUSTED-RUNTIME.md v3
        scope): the kit's Python by its loader line, and a kit without a
        Python but with GNU ``head`` gets pre-started execs instead
        (:class:`~benchflow.sandbox._exec_channel.WarmExecPool`). Both
        are read on root's route."""
        container, generation, user = key
        channel: ExecChannel | WarmExecPool | None = None
        try:
            route = await self.exec_route("root", system_shell=True)
            if not route.kit:
                return None
            bfx = getattr(self, "bfx_delivery", None) is not None
            python: list[str] | None = None
            if not bfx:
                marker = "@@benchflow-exec-channel-head@@"
                read = await self._run_docker_container_command(
                    route_exec_argv(
                        route,
                        f"{KIT_BIN}/cat {KIT_PYTHON} 2>/dev/null; "
                        f"printf '\\n%s\\n' {marker}; "
                        f"{KIT_BIN}/head --version 2>/dev/null | {KIT_BIN}/head -n 1",
                        container=container,
                        user="0",
                    ),
                    check=False,
                    timeout_sec=30,
                    raw_output=True,
                )
                wrapper, _, head = (read.stdout or "").partition(f"\n{marker}\n")
                python = helper_python_argv(wrapper) if read.return_code == 0 else None
                gnu_head = read.return_code == 0 and head.startswith(
                    "head (GNU coreutils)"
                )
                if python is None and not gnu_head:
                    self.logger.debug(
                        "exec channel: no bfx, and the kit has no Python or GNU head"
                    )
                    return None
            config = (await self._inspect_container(container)).get("Config") or {}
            launch = getattr(self, "_launch", None)
            client = (
                launch.docker_args
                if launch is not None
                else self._docker_compose_global_args()
            )
            prefix = [
                "docker",
                *client,
                "exec",
                "-i",
                *(["-u", user] if user is not None else []),
                *system_shell_env_flags(),
                container,
            ]
            if not bfx and python is None:
                channel = WarmExecPool(
                    prefix,
                    self._docker_process_env(),
                    shell=route.shell,
                    head=f"{KIT_BIN}/head",
                )
                channel.fill()
            else:
                # One journal per helper, in root's own run folder.
                journal = f"{RUN_DIR}/exec-channel-{uuid.uuid4().hex}.started"
                if python is not None:
                    helper = [*python, "-c", HELPER_SOURCE, journal]
                    verdict = [*python, "-c", VERDICT_SOURCE]
                else:
                    helper = helper_argv(BFX_PATH, journal)
                    verdict = verdict_argv(BFX_PATH)
                channel = ExecChannel(
                    [*prefix, *helper],
                    self._docker_process_env(),
                    workdir=config.get("WorkingDir") or "/",
                    logger=self.logger,
                    journal=journal,
                )
                channel.verify_argv = [*prefix, *verdict]
                await channel.start()
        except asyncio.CancelledError:
            if channel is not None:
                await channel.close()
            raise
        except Exception as exc:
            self.logger.debug("exec channel: did not start (%s); docker exec", exc)
            return None
        if self._exec_generation != generation or self._main_container != container:
            await channel.close()
            return None
        self.logger.debug("exec channel: ready")
        return channel

    async def _close_exec_channel(self, state: Any) -> None:
        if isinstance(state, asyncio.Future):
            if not state.done():
                state.cancel()
                await asyncio.wait([state])
                return
            state = None if state.cancelled() or state.exception() else state.result()
        if isinstance(state, (ExecChannel, WarmExecPool)):
            await state.close()

    async def _close_exec_channels(self) -> None:
        """Close every channel (the sandbox stops)."""
        channels, self._exec_channels = self._exec_channels or {}, None
        closing, self._exec_channels_closing = self._exec_channels_closing or [], None
        for state in channels.values():
            with contextlib.suppress(Exception):
                await self._close_exec_channel(state)
        if closing:
            await asyncio.wait(closing)

    async def _copy(
        self, host_path: str, sandbox_path: str, *, into: bool, service: str = "main"
    ) -> None:
        """Copy ``host_path`` into the service (``into``) or ``sandbox_path`` out.

        Through ``docker cp`` on the main container when start() knows it,
        else ``docker compose cp`` on the service.
        """
        container = self._main_container if service == "main" else None
        side = f"{container or service}:{sandbox_path}"
        command = ["cp", host_path, side] if into else ["cp", side, host_path]
        if container:
            await self._run_docker_container_command(command, check=True)
        else:
            await self._run_docker_compose_command(command, check=True)

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        from benchflow.sandbox.process import DockerProcess

        return DockerProcess.from_sandbox_env(self)

    async def attach(self) -> None:
        variables = " ".join(
            f"export {k}={shlex.quote(str(v))}"
            for k, v in self._env_vars.to_env_dict(include_os_env=False).items()
        )

        compose_file_args: list[str] = []
        for path in self._docker_compose_paths:
            compose_file_args.extend(
                ["-f", shlex.quote(str(path.resolve().absolute()))]
            )

        project_name = self.compose_project_name
        compose_base = [
            "docker",
            "compose",
            "--project-name",
            project_name,
            *compose_file_args,
        ]

        client_env = self._docker_client_env()
        # Root's interactive shell is the route's (C8): an absolute shell
        # (the protected kit's once the sandbox has one) with no profile, rc
        # or ENV file, the loader variables empty where Docker starts it.
        route = await self.exec_route("root", system_shell=True)
        shell = interactive_root_shell(route)
        exec_argv = route_exec_argv(
            route,
            "",
            container="main",
            user="0",
            stdin=True,
            tty=True,
            shell_argv=shell,
        )
        argv = [
            "bash",
            "-c",
            f"{variables}; "
            + " ".join([*compose_base, *(shlex.quote(a) for a in exec_argv)])
            + "; "
            + " ".join([*compose_base, "down"]),
        ]
        if client_env is None:
            os.execvp("bash", argv)
        else:
            os.execvpe("bash", argv, client_env)


class _DockerCredentialOps:
    """Access to one container for the snapshot credential scrub.

    :meth:`run_as` drops to the credential home owner's uid:gid with
    ``docker exec -u`` (see :mod:`benchflow.sandbox._snapshot_credentials`).
    File contents travel over ``docker exec`` stdin/stdout, never as
    command-line arguments; Docker keeps no copy of exec output in the
    container filesystem.
    """

    def __init__(self, sandbox: DockerSandbox, container: str) -> None:
        self._sandbox = sandbox
        self._container = container

    async def run(self, command: str) -> ExecResult:
        route = await system_shell_route(self._sandbox)
        result = await self._sandbox._docker_cli(
            route_exec_argv(route, command, container=self._container, user="0"),
            check=False,
            raw_output=True,
        )
        # Discovery parses stdout; stderr is only displayed in scrub errors.
        # Redact that diagnostic before its consumer truncates it.
        return ExecResult(
            stdout=result.stdout,
            stderr=self._sandbox._diagnostic_text(result.stderr or ""),
            return_code=result.return_code,
        )

    async def run_as(
        self, uid: int, gid: int, script: str, *, stdin: bytes | None = None
    ) -> AsUserResult:
        # An absolute shell: Docker resolves a bare ``sh`` through the
        # image's PATH, where the agent may have planted one (PT-CONF-2);
        # as root, the route's shell and tools (the protected kit's once the
        # sandbox has one, C8).
        route = await system_shell_route(self._sandbox) if int(uid) == 0 else None
        argv = (
            route_exec_argv(
                route,
                script,
                container=self._container,
                user=f"{int(uid)}:{int(gid)}",
                stdin=stdin is not None,
            )
            if route is not None
            else [
                "exec",
                *(["-i"] if stdin is not None else []),
                "-u",
                f"{int(uid)}:{int(gid)}",
                self._container,
                SYSTEM_SHELL,
                "-c",
                script,
            ]
        )
        _transport.sent("Docker", argv)
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *argv,
            stdin=(
                asyncio.subprocess.PIPE
                if stdin is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._sandbox._docker_client_env(),
        )
        stdout, stderr = await proc.communicate(stdin)
        message = (stderr or b"").decode(errors="replace")
        # A remote Docker host's ssh user can appear in client errors.
        redact = getattr(getattr(self._sandbox, "docker_host", None), "redact", None)
        if callable(redact):
            message = redact(message)
        code = proc.returncode if proc.returncode is not None else 1
        return AsUserResult(code, stdout or b"", message)
