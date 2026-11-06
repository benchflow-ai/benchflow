"""Docker-in-Docker compose strategy for the Daytona backend.

Extracted from ``benchflow.sandbox.daytona`` (via ``daytona_strategies``) as a
cohesion seam: the multi-container compose strategy ``_DaytonaDinD`` that runs a
task's ``docker-compose.yaml`` inside a Daytona DinD VM. The name is re-exported
from ``benchflow.sandbox.daytona`` so existing imports such as
``from benchflow.sandbox.daytona import _DaytonaDinD`` keep working unchanged.

Like ``_DaytonaDirect``, the ``start`` path reaches the lazily-materialized SDK
handles and the ``DaytonaClientManager`` singleton through a local import of
``benchflow.sandbox.daytona`` so the lazy-load / monkeypatch contract (#358) is
unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shlex
import tempfile
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchflow._utils.reachable_write import UnsafeDownloadEntryError
from benchflow.sandbox import _bind_targets, _inert_bootstrap
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    SYSTEM_SHELL,
    ExecResult,
    ExecRoute,
    _filter_compose_service_names,
    _root_exec_command,
    capture_pid1_image_env,
    restore_image_env,
    system_shell_env_flags,
    system_shell_route,
    with_system_path,
)
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_KEPT_FREEZER_PATH,
    COMPOSE_NET_ADMIN_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_NO_RESTART_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_UP_RETRY_DELAYS_SEC,
    DIND_ENVIRONMENT_DIR,
    DIND_LOGS_DIR,
    DIND_TASK_DIR,
    compose_container_ids,
    compose_cp_destination,
    compose_mkdir_p_command,
    compose_needs_kept_freezer,
    compose_needs_net_admin,
    compose_parent_mkdir_p_command,
    compose_stops_main_restart,
    decide_kept_service_freezer,
    is_compose_up_network_race_error,
    main_restart_refusal,
)
from benchflow.sandbox._compose_runtime import private_image_repository
from benchflow.sandbox._compose_teardown import (
    COMPOSE_KILL_COMMAND,
    compose_down_command,
)
from benchflow.sandbox._download import RemoteFileAbsentError
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._transport import STAGE_ROOT, fit, reader_cat
from benchflow.sandbox.daytona_pty import (
    _exec_failure_output,
    _wrap_daytona_command_with_env_file,
)
from benchflow.sandbox.daytona_reaper import _benchflow_owned_labels
from benchflow.sandbox.daytona_strategies import _DaytonaStrategy
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import SandboxPaths, TaskPaths

if TYPE_CHECKING:
    from benchflow.sandbox.daytona import DaytonaSandbox

# ``_SandboxParams`` mirrors the façade alias: the SDK types are loaded lazily
# (issue #358), so concrete sandbox params are typed as ``Any`` here — callers
# build them inside methods that have already called ``_load_daytona_sdk()``.
_SandboxParams = Any


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_SANDBOX_DISAPPEARED_MARKERS = (
    "sandbox container not found",
    "failed to inspect sandbox container",
    "no such container",
)
_LONG_EXEC_HEARTBEAT_MIN_TIMEOUT_SEC = 600
_LONG_EXEC_HEARTBEAT_INTERVAL_SEC = _positive_int_env(
    "BENCHFLOW_DAYTONA_DIND_EXEC_HEARTBEAT_SEC", 60
)


# The task's oracle and verifier folders (legacy and native names). A compose
# file that mounts the task folder hides them with empty volumes over
# ``<mount>/solution`` and ``<mount>/tests``; Docker needs those folders to be
# there as mount points, since the task folder is mounted read-only (#1181).
# BenchFlow's own commands on the DinD VM (docker, compose, its own files):
# root, outside every task container, which the agent never reaches; the VM
# image is BenchFlow's. The route _vm_exec gives DaytonaSandbox._sandbox_exec.
VM_ROUTE = ExecRoute(pin=None, task_root=False, explicit_root=True, own_vm=True)

_HIDDEN_TASK_DIRS = (
    TaskPaths.LEGACY_SOLUTION_DIRNAME,
    TaskPaths.LEGACY_TESTS_DIRNAME,
    TaskPaths.NATIVE_ORACLE_DIRNAME,
    TaskPaths.NATIVE_VERIFIER_DIRNAME,
)


def _task_folder(environment_dir: Path) -> Path | None:
    """The task folder around ``environment_dir``, or None if it is not one.

    Resolved like Docker's ``CONTEXT_DIR``, so ``${CONTEXT_DIR}/..`` names the
    same folder on both backends. Only a folder that holds ``task.toml`` or
    ``task.md`` beside ``environment/`` counts: an ``environment_dir`` from
    elsewhere must not upload whatever happens to be its parent.
    """
    # Never turn a selected task's environment link into authorization to
    # upload the linked task's parent.
    if environment_dir.is_symlink():
        raise ValueError("Daytona task environment must not be a symlink")
    task_root = environment_dir.parent.resolve()
    resolved = environment_dir.resolve()
    if resolved.parent != task_root:
        raise ValueError("Daytona task environment escapes its task root")
    if resolved.name != "environment":
        return None
    task_dir = resolved.parent
    if not any(
        (task_dir / name).is_file()
        for name in (TaskPaths.CONFIG_FILENAME, TaskPaths.DOCUMENT_FILENAME)
    ):
        return None
    return task_dir


def _is_sandbox_disappeared_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _SANDBOX_DISAPPEARED_MARKERS)


def _is_daytona_sdk_error(exc: BaseException) -> bool:
    return type(exc).__module__.startswith("daytona.")


def _raise_if_startup_provider_error(env: DaytonaSandbox, exc: BaseException) -> None:
    if not (_is_sandbox_disappeared_error(exc) or _is_daytona_sdk_error(exc)):
        return
    sandbox_id = getattr(env._sandbox, "id", None) if env._sandbox else None
    raise SandboxStartupError(
        f"Daytona DinD startup failed: {exc}",
        sandbox_id=sandbox_id,
        sandbox_state="not_found",
        attempts=1,
        build_timeout_sec=env.task_env_config.build_timeout_sec,
    ) from exc


def _with_long_exec_heartbeat(
    command: str, timeout_sec: int | None, tools: str = SYSTEM_PATH
) -> str:
    """*command* in the background with a heartbeat on stderr, for long execs.

    The env is inline in *command*, so it applies inside the worker, after
    the heartbeat's own variables are set: a task variable that shares a
    name with one of them keeps its value in the command (b162 PY8-MAIN-37).
    """
    if (
        timeout_sec is None
        or timeout_sec < _LONG_EXEC_HEARTBEAT_MIN_TIMEOUT_SEC
        or _LONG_EXEC_HEARTBEAT_INTERVAL_SEC <= 0
    ):
        return command
    interval = shlex.quote(str(_LONG_EXEC_HEARTBEAT_INTERVAL_SEC))
    # The wrapper's own tools come from the system folders: as root it
    # would otherwise run a mktemp, rm or sleep the agent left on the
    # image's PATH (C8). The command gets the PATH it was given.
    return "\n".join(
        [
            "bf_task_path=$PATH",
            f"PATH={tools}",
            "bf_out=$(mktemp)",
            "bf_err=$(mktemp)",
            'bf_stop="${bf_out}.stop"',
            'rm -f "$bf_stop"',
            'bf_cleanup() { rm -f "$bf_out" "$bf_err" "$bf_stop"; }',
            "trap bf_cleanup EXIT",
            f'(PATH="$bf_task_path"; {command}\n) >"$bf_out" 2>"$bf_err" &',
            "bf_pid=$!",
            "(",
            "  bf_heartbeat_elapsed=0",
            '  while [ ! -e "$bf_stop" ] && kill -0 "$bf_pid" 2>/dev/null; do',
            "    sleep 1",
            '    [ -e "$bf_stop" ] && break',
            "    bf_heartbeat_elapsed=$((bf_heartbeat_elapsed + 1))",
            f'    if [ "$bf_heartbeat_elapsed" -ge {interval} ] '
            '&& kill -0 "$bf_pid" 2>/dev/null; then',
            "      echo '[benchflow] Daytona DinD command still running' >&2",
            "      bf_heartbeat_elapsed=0",
            "    fi",
            "  done",
            ") &",
            "bf_heartbeat_pid=$!",
            'wait "$bf_pid"',
            "bf_rc=$?",
            ': > "$bf_stop"',
            'wait "$bf_heartbeat_pid" 2>/dev/null || true',
            'cat "$bf_out"',
            'cat "$bf_err" >&2',
            'exit "$bf_rc"',
        ]
    )


class _DaytonaDinD(_DaytonaStrategy):
    """Docker-in-Docker compose strategy for multi-container tasks.

    Topology:
        Local machine (benchflow CLI)
          +-- Daytona Sandbox (DinD VM, docker:28.3.3-dind)
                +-- dockerd (Docker daemon)
                +-- docker compose
                      +-- main        <- agent runs here
                      +-- mcp-server  <- sidecar services
                      +-- ...
    """

    _DOCKER_DAEMON_TIMEOUT_SEC = _positive_int_env(
        "BENCHFLOW_DAYTONA_DOCKER_DAEMON_TIMEOUT_SEC", 600
    )
    _COMPOSE_DIR = "/benchflow/compose"
    # Uploaded to ``_COMPOSE_DIR``; every overlay ``_compose_file_flags`` can
    # reference must be here, or ``docker compose -f`` fails on a missing file.
    _BENCHFLOW_COMPOSE_FILES = (
        COMPOSE_BASE_PATH,
        COMPOSE_BUILD_PATH,
        COMPOSE_PREBUILT_PATH,
        COMPOSE_NO_NETWORK_PATH,
        COMPOSE_NET_ADMIN_PATH,
        COMPOSE_NO_RESTART_PATH,
        COMPOSE_KEPT_FREEZER_PATH,
    )
    # The task folder sits at ``_TASK_DIR`` with its ``environment/`` inside,
    # as on the host, so a compose file that mounts ``${CONTEXT_DIR}/..`` or
    # ``${CONTEXT_DIR}/../setup_workspace`` finds the task's files (#1181).
    _TASK_DIR = DIND_TASK_DIR
    _ENVIRONMENT_DIR = DIND_ENVIRONMENT_DIR
    _LOGS_DIR = DIND_LOGS_DIR

    def __init__(self, env: DaytonaSandbox) -> None:
        super().__init__(env)
        self._use_prebuilt = False
        self._bind_baseline: _bind_targets.BindTargets | None = None
        self._runtime_compose_path: str | None = None
        self._build_compose_path: str | None = None
        self._build_image_name = private_image_repository(uuid4().hex)

        self._resolved_task_env: dict[str, str] = {}
        benchflow_keys = set(self._compose_env_vars().keys()) - set(
            self._env._persistent_env.keys()
        )
        if self._env.task_env_config.env:
            self._resolved_task_env = resolve_env_vars(self._env.task_env_config.env)

        resolved_task_keys = set(self._resolved_task_env.keys()) | set(
            self._env._persistent_env.keys()
        )
        if resolved_task_keys:
            collisions = benchflow_keys & resolved_task_keys
            if collisions:
                self._env.logger.warning(
                    "Environment vars override BenchFlow compose variable(s): %s",
                    ", ".join(sorted(collisions)),
                )

    async def _engine_info_json(self) -> str:
        """The inner daemon's ``docker info`` as JSON (empty when it cannot
        say)."""
        result = await self._vm_exec(
            "docker info --format '{{json .}}'", timeout_sec=60
        )
        return result.stdout or "" if result.return_code == 0 else ""

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        cleanup_session: bool = False,
        stdin_payload: bytes | None = None,
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM using sh (Alpine-compatible).
        *stdin_payload*: the command's stdin (see ``_sandbox_exec``)."""
        extra = {} if stdin_payload is None else {"stdin_payload": stdin_payload}
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell="/bin/sh -c",
            cleanup_session=cleanup_session,
            route=VM_ROUTE,
            **extra,
        )

    @staticmethod
    def _guard_daemon_command(
        ports: str, proc: str = "/proc", bridges: tuple[str, ...] = ()
    ) -> str:
        """The VM shell command for :meth:`_guard_daemon_from_containers`.

        ``proc`` is ``/proc`` in production; a test points it elsewhere to
        exercise the IPv6 path without the host's interfaces.
        dockerd-entrypoint.sh falls back to iptables-legacy (on its own
        PATH) where nf_tables does not work; pick the same way here.
        """
        return (
            "status=0; "
            'pick() { for c in "$1" "$1-legacy"; do '
            'command -v "$c" >/dev/null 2>&1 && "$c" -w -nL >/dev/null 2>&1 && '
            '"$c" -w -t mangle -S OUTPUT >/dev/null 2>&1 && { echo "$c"; return 0; }; '
            "done; return 1; }; "
            "for fam in iptables ip6tables; do "
            'if ! fw=$(pick "$fam"); then '
            '[ "$fam" = iptables ] && status=2; '
            # IPv6 is up but ip6tables does not work: do not pass silently.
            f'[ "$fam" = ip6tables ] && [ -e {proc}/net/if_inet6 ] && '
            '[ "$status" = 0 ] && status=3; '
            "continue; fi; "
            f"for i in {shlex.join(['docker0', 'br-+', *bridges])}; do "
            f"for p in {ports}; do "
            '"$fw" -w -C INPUT -i "$i" -p tcp --dport "$p" -j REJECT --reject-with tcp-reset 2>/dev/null || '
            '"$fw" -w -I INPUT 1 -i "$i" -p tcp --dport "$p" -j REJECT --reject-with tcp-reset || status=1; '
            "done; done; "
            f"for p in {ports}; do "
            '"$fw" -w -t mangle -C OUTPUT -p tcp --dport "$p" -m owner ! --uid-owner 0 -j DROP 2>/dev/null || '
            '"$fw" -w -t mangle -I OUTPUT 1 -p tcp --dport "$p" -m owner ! --uid-owner 0 -j DROP || status=1; '
            "done; done; "
            'echo "BF dind-daemon-guard status=$status"'
        )

    async def _guard_daemon_from_containers(
        self, bridges: tuple[str, ...] = ()
    ) -> None:
        """Refuse connections from the compose containers to the daemon.

        Daytona's toolbox daemon (root on this VM) listens on every address,
        so a compose container reaches it through its bridge gateway, and
        the daemon can ``docker exec`` into ``main`` as root. Traffic from
        the containers arrives on ``docker0`` or a ``br-*`` bridge; the
        runner's own reaches the VM elsewhere and is untouched. A container
        on the host network shares the VM's namespace, so it crosses no
        bridge: the VM also drops every non-root uid's own connections to
        those ports (the daemon and dockerd run as root, and Docker does not
        remap uids, so the agent keeps its uid there). Raises when the rules
        cannot be installed, so the run is refused.
        """
        from benchflow.sandbox.lockdown import (
            SANDBOX_DAEMON_PORTS,
            SandboxDaemonGuardError,
        )

        ports = " ".join(str(p) for p in SANDBOX_DAEMON_PORTS)
        command = self._guard_daemon_command(ports, bridges=bridges)
        result = await self._vm_exec(command, timeout_sec=60)
        out = result.stdout or ""
        if result.return_code == 0 and "BF dind-daemon-guard status=0" in out:
            return
        raise SandboxDaemonGuardError(
            "sandbox startup refused: the compose containers could not be kept "
            "away from Daytona's root toolbox daemon on the DinD VM "
            f"({out.strip() or 'no report'}; {(result.stderr or '').strip()[:300]}), so "
            "the agent could reach root through the bridge gateway."
        )

    async def _guard_compose_networks(self) -> None:
        """Guard actual bridges after Compose has created its networks.

        Only bridge, host and null drivers have a known isolation mechanism.
        This VM belongs to the trial, so inspect all networks, including
        external networks and those created by the trusted pre-compose hook.
        No agent starts until this second guard succeeds.
        """
        from benchflow.sandbox.lockdown import SandboxDaemonGuardError

        result = await self._vm_exec(
            "docker network ls -q | xargs -r docker network inspect", timeout_sec=60
        )
        bridges: set[str] = set()
        try:
            networks = json.loads(result.stdout or "")
            if (
                result.return_code != 0
                or not isinstance(networks, list)
                or not networks
            ):
                raise ValueError("network inventory unavailable")
            for network in networks:
                driver = network["Driver"]
                if driver in {"host", "null"}:
                    continue
                if driver != "bridge":
                    raise ValueError("network driver has no daemon guard")
                options = network.get("Options") or {}
                bridge = options.get("com.docker.network.bridge.name")
                if not bridge:
                    bridge = "br-" + network["Id"][:12]
                if (
                    not isinstance(bridge, str)
                    or not 1 <= len(bridge) <= 15
                    or "\x00" in bridge
                ):
                    raise ValueError("invalid bridge interface")
                bridges.add(bridge)
        except (ValueError, KeyError, TypeError) as error:
            raise SandboxDaemonGuardError(
                "sandbox startup refused: cannot establish Daytona Compose network isolation"
            ) from error
        await self._guard_daemon_from_containers(tuple(sorted(bridges)))

    def _compose_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {
            "CONTEXT_DIR": self._ENVIRONMENT_DIR,
            "MAIN_IMAGE_NAME": f"bf__{self._env.environment_name}",
            "HOST_VERIFIER_LOGS_PATH": f"{self._LOGS_DIR}/verifier",
            "HOST_AGENT_LOGS_PATH": f"{self._LOGS_DIR}/agent",
            "HOST_ARTIFACTS_PATH": f"{self._LOGS_DIR}/artifacts",
            "ENV_VERIFIER_LOGS_PATH": str(SandboxPaths.verifier_dir),
            "ENV_AGENT_LOGS_PATH": str(SandboxPaths.agent_dir),
            "ENV_ARTIFACTS_PATH": str(SandboxPaths.artifacts_dir),
            "CPUS": str(self._env.task_env_config.cpus),
            "MEMORY": f"{self._env.task_env_config.memory_mb}M",
        }
        if self._use_prebuilt and self._env.task_env_config.docker_image:
            env_vars["PREBUILT_IMAGE_NAME"] = self._env.task_env_config.docker_image
        if self._resolved_task_env:
            env_vars.update(self._resolved_task_env)
        if self._env._persistent_env:
            env_vars.update(self._env._persistent_env)
        return env_vars

    def _compose_file_flags(self) -> list[str]:
        build_or_prebuilt = (
            "docker-compose-prebuilt.yaml"
            if self._use_prebuilt
            else "docker-compose-build.yaml"
        )
        files = [
            f"{self._COMPOSE_DIR}/docker-compose-base.yaml",
            f"{self._COMPOSE_DIR}/{build_or_prebuilt}",
            f"{self._ENVIRONMENT_DIR}/{self._env._environment_docker_compose_path.name}",
        ]
        if not self._env.task_env_config.allow_internet:
            files.append(f"{self._COMPOSE_DIR}/{COMPOSE_NO_NETWORK_PATH.name}")
        if compose_needs_net_admin(self._env.task_env_config):
            files.append(f"{self._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}")
        if compose_stops_main_restart(self._env.task_env_config):
            files.append(f"{self._COMPOSE_DIR}/{COMPOSE_NO_RESTART_PATH.name}")
        # Before builds.yaml: its security_opt (!override, read from a project
        # that already holds this file's writable-cgroups) is the last word,
        # so no later file appends that option a second time, which Compose
        # 2.40 refuses ("security_opt items ... are equal").
        if compose_needs_kept_freezer(self._env):
            files.append(f"{self._COMPOSE_DIR}/{COMPOSE_KEPT_FREEZER_PATH.name}")
        if self._build_compose_path:
            files.append(self._build_compose_path)
        if self._runtime_compose_path:
            files.append(self._runtime_compose_path)

        bfx_overlay = getattr(self, "_bfx_compose_path", None)
        if bfx_overlay:
            files.append(bfx_overlay)
        bootstrap = getattr(self, "_bootstrap_override", None)
        if bootstrap is not None:
            files.append(bootstrap)
        flags: list[str] = []
        for f in files:
            flags.extend(["-f", f])
        return flags

    async def _upload_runtime_compose_file(self) -> None:
        from benchflow.sandbox._compose_runtime import prebuilt_main_override

        self._runtime_compose_path = None
        values = {**self._resolved_task_env, **self._env._persistent_env}
        if not self._use_prebuilt and not values:
            return
        document = "services:\n  main:\n"
        if self._use_prebuilt:
            image = self._env.task_env_config.docker_image
            assert image
            document = prebuilt_main_override(image)
        if values:
            # These are container values, not Compose interpolation input.
            # Match Docker's precedence and preserve every literal dollar.
            literal = {key: value.replace("$", "$$") for key, value in values.items()}
            document += f"    environment: {json.dumps(literal)}\n"
        destination = f"{self._COMPOSE_DIR}/runtime.yaml"
        with tempfile.TemporaryDirectory(prefix="benchflow-compose-") as directory:
            path = Path(directory) / "runtime.yaml"
            path.write_text(document)
            await self._env._sdk_upload_file(path, destination)
        self._runtime_compose_path = destination

    async def _prepare_compose_builds(self) -> None:
        """Isolate all outputs in this daemon before any service can build.

        A private daemon still has shared tags within a trial: a sidecar must
        not overwrite main's selected prebuilt image through image or tags.
        Resolve the final task/hook plan and refuse an unreadable plan.
        """
        from benchflow.sandbox._compose_runtime import (
            image_mount_daemon_version,
            implicit_build_aliases,
            private_build_override,
        )

        self._build_compose_path = None
        try:
            result = await self._compose_exec(
                ["config", "--format", "json"], timeout_sec=30
            )
            if result.return_code != 0:
                raise ValueError("Compose could not resolve the build plan")
            config = json.loads(result.stdout or "{}")
            if not isinstance(config.get("services"), dict):
                raise ValueError("Compose did not resolve services")
            document, _refs = private_build_override(
                config,
                self._build_image_name,
                prebuilt=self._use_prebuilt,
                agent_security=True,
                daemon_version=await image_mount_daemon_version(
                    config,
                    lambda: self._vm_exec(
                        "docker version --format '{{.Server.Version}}'",
                        env=self._compose_env_vars(),
                        timeout_sec=30,
                    ),
                ),
                implicit_aliases=await implicit_build_aliases(
                    config,
                    prebuilt=self._use_prebuilt,
                    compose=self._compose_exec,
                ),
            )
            if config["services"]:
                destination = f"{self._COMPOSE_DIR}/builds.yaml"
                with tempfile.TemporaryDirectory(
                    prefix="benchflow-compose-"
                ) as directory:
                    path = Path(directory) / "builds.yaml"
                    path.write_text(document)
                    await self._env._sdk_upload_file(path, destination)
                self._build_compose_path = destination
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"Could not isolate Compose build output references: {exc}"
            ) from exc

    @property
    def _project_name(self) -> str:
        return self._env.session_id.lower().replace(".", "-")

    def _compose_cmd(self, subcommand: list[str]) -> str:
        parts = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._ENVIRONMENT_DIR,
            *self._compose_file_flags(),
            *subcommand,
        ]
        return shlex.join(parts)

    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
        cleanup_session: bool = False,
        *,
        merge_stderr: bool = True,
        stdin_data: bytes | None = None,
    ) -> ExecResult:
        command = self._compose_cmd(subcommand)
        if getattr(self, "_bootstrap_override", None) is not None:
            command = "unset " + " ".join(_inert_bootstrap.EMPTY_ENV) + "; " + command
        # The VM SDK already returns separate stdout/stderr fields. Accept the
        # shared discovery flag without ever folding diagnostics into records.
        # *stdin_data*: the command's stdin (a container's staged command).
        extra = {} if stdin_data is None else {"stdin_payload": stdin_data}
        return await self._vm_exec(
            command,
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
            cleanup_session=cleanup_session,
            **extra,
        )

    async def _check_compose_project(self) -> None:
        """Refuse a project that reaches the VM outside the uploaded
        environment folder and BenchFlow's log folders
        (:func:`benchflow.task.compose_env.check_compose_in_vm`)."""
        from benchflow.task.compose_env import check_compose_in_vm
        from benchflow.task.env import TaskEnvNotAllowedError

        async def run(command: str) -> ExecResult:
            return await self._vm_exec(
                command, env=self._compose_env_vars(), timeout_sec=60
            )

        refused = await check_compose_in_vm(
            run,
            shlex.split(self._compose_cmd([])),
            self._ENVIRONMENT_DIR,
            (
                self._TASK_DIR,
                self._ENVIRONMENT_DIR,
                f"{self._LOGS_DIR}/verifier",
                f"{self._LOGS_DIR}/agent",
                f"{self._LOGS_DIR}/artifacts",
            ),
        )
        if refused:
            raise TaskEnvNotAllowedError({self._env.environment_name: refused})

    async def _prepare_bfx(self) -> None:
        """bfx at /.bfx in main: the inner daemon's harness volume, filled and
        checked by digest through the VM's control path, mounted read-only
        with volume-nocopy by a Compose overlay (TRUSTED-RUNTIME.md §6).
        Until it passes a live run it is opt-in; without it the kit path
        stays."""
        from benchflow._bfx import delivery as bfx_delivery
        from benchflow._bfx.locate import BfxUnavailable

        self._env.bfx_delivery = None
        self._bfx_compose_path = None
        if not bfx_delivery.provider_enabled("daytona-dind"):
            return
        try:
            arch = await self._vm_arch()
            volume = await bfx_delivery.ensure_volume(
                [],
                None,
                arch=arch,
                daemon_key=f"daytona-dind:{self._env.session_id}",
                transport=DinDBfxTransport(self),
            )
        except BfxUnavailable as exc:
            self._env.logger.debug("bfx: not delivered (%s)", exc)
            return
        destination = f"{self._COMPOSE_DIR}/bfx.json"
        with tempfile.TemporaryDirectory(prefix="benchflow-compose-") as directory:
            path = Path(directory) / "bfx.json"
            path.write_text(
                bfx_delivery.compose_overlay(bfx_delivery.VOLUME, source=volume)
            )
            await self._env._sdk_upload_file(path, destination)
        self._bfx_compose_path = destination
        self._bfx_source = volume
        self._env.bfx_delivery = bfx_delivery.VOLUME

    async def _vm_arch(self) -> str:
        from benchflow._bfx.locate import BfxUnavailable, normalize_arch

        result = await self._vm_exec(
            "docker info --format '{{.Architecture}}'", timeout_sec=30
        )
        if result.return_code != 0:
            raise BfxUnavailable("cannot read the inner daemon's architecture")
        return normalize_arch(result.stdout or "")

    async def _docker_cli(self, args: list[str], check: bool = True) -> ExecResult:
        result = await self._vm_exec(shlex.join(["docker", *args]), timeout_sec=60)
        if check and result.return_code:
            raise RuntimeError("Cannot inspect DinD bind identities; retry startup")
        return result

    async def main_restart_refusal(self) -> str:
        """Why the cloud metadata block cannot hold this sandbox's ``main``
        ("" when it cannot restart): its restart policy as the VM's daemon
        holds it (``_compose.main_restart_refusal``)."""
        try:
            ids = await compose_container_ids(
                self._compose_exec, service="main", all_containers=False
            )
            if len(ids) != 1:
                return main_restart_refusal(None)
            # _docker_cli raises on a failed inspect (check=True).
            result = await exec_raw(
                self._docker_cli,
                ["inspect", "--format", "{{.HostConfig.RestartPolicy.Name}}", ids[0]],
            )
        except Exception:
            return main_restart_refusal(None)
        return main_restart_refusal(str(result.stdout).strip())

    async def _inspect_container(self, container: str) -> dict:
        result = await exec_raw(self._docker_cli, ["inspect", container])
        records = json.loads(result.stdout)
        if (
            not isinstance(records, list)
            or len(records) != 1
            or not isinstance(records[0], dict)
        ):
            raise RuntimeError("Cannot inspect DinD container; retry startup")
        return records[0]

    def _check_bfx_mounts(self, inspected: dict) -> None:
        """main as created: only BenchFlow's read-only bfx volume at or
        beneath /.bfx (pin_project calls this on main's inspect)."""
        from benchflow._bfx import delivery as bfx_delivery

        mode = getattr(self._env, "bfx_delivery", None)
        if mode is None:
            return
        problems = bfx_delivery.mount_problems(
            inspected, mode=mode, source=getattr(self, "_bfx_source", None)
        )
        if problems:
            raise RuntimeError(
                "main has mounts at or beneath /.bfx, where only BenchFlow's "
                f"read-only bfx may be: {'; '.join(problems)}"
            )

    async def _pin_or_verify_bind_targets(self) -> None:
        ids = await compose_container_ids(self._compose_exec)
        self._bind_baseline = await _bind_targets.pin_project(
            self, "\n".join(ids), self._bind_baseline
        )

    async def _prepare_inert_bootstrap(self) -> None:
        import tempfile

        self._bootstrap_override = None
        document = await _inert_bootstrap.prepare(self, self._compose_exec)
        with tempfile.TemporaryDirectory(prefix="benchflow-bootstrap-") as folder:
            path = Path(folder) / "bootstrap.json"
            path.write_text(json.dumps(document).replace("$", "$$"))
            remote = f"{self._COMPOSE_DIR}/bootstrap.json"
            await self._env._sdk_upload_file(path, remote)
        self._bootstrap_override = remote

    async def _compose_teardown(self, timeout_sec: int | None = None) -> ExecResult:
        """Kill every container at once, then ``down -t 0``, in one VM exec.

        The same commands as the Docker backends (_compose_teardown). Images
        and volumes go with the sandbox, which :meth:`stop` deletes next.
        """
        kill = self._compose_cmd(list(COMPOSE_KILL_COMMAND))
        down = self._compose_cmd(compose_down_command())
        return await self._vm_exec(
            f"{kill}; {down}",
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    async def _run_pre_compose_hook(self) -> None:
        hook = f"{self._ENVIRONMENT_DIR}/benchflow-pre-compose.sh"
        timeout_sec = max(120, round(self._env.task_env_config.build_timeout_sec))
        result = await self._vm_exec(
            f"if [ -f {shlex.quote(hook)} ]; then "
            f"chmod +x {shlex.quote(hook)} && {shlex.quote(hook)}; "
            "fi",
            cwd=self._ENVIRONMENT_DIR,
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "pre-compose hook failed inside DinD sandbox: "
                f"{result.stdout} {result.stderr}"
            )

    async def _compose_up_with_retry(self, env: DaytonaSandbox) -> None:
        """Run ``compose up -d`` honouring the author's build budget and retrying
        a fresh-daemon network create/attach race (mirrors the host docker path).

        The ``up`` timeout is derived from ``build_timeout_sec`` (floored at the
        historical 120s) so a prebuilt-image pull gets the full author budget,
        rather than a hardcoded 120s.
        """
        timeout_sec = max(120, round(env.task_env_config.build_timeout_sec))
        max_attempts = len(COMPOSE_UP_RETRY_DELAYS_SEC) + 1
        for attempt in range(1, max_attempts + 1):
            command = _inert_bootstrap.up_args(self, "--no-build", detach="-d")
            if command is None:
                return
            result = await self._compose_exec(command, timeout_sec=timeout_sec)
            if result.return_code == 0:
                return
            message = f"{result.stdout} {result.stderr}"
            if attempt == max_attempts or not is_compose_up_network_race_error(message):
                raise RuntimeError(f"docker compose up failed: {message}")
            delay = COMPOSE_UP_RETRY_DELAYS_SEC[attempt - 1]
            env.logger.warning(
                "Retrying docker compose up inside DinD after network "
                "create/attach race (attempt %s/%s, retrying in %.1fs): %s",
                attempt,
                max_attempts,
                delay,
                message,
            )
            await asyncio.sleep(delay)

    async def _wait_for_docker_daemon(self) -> None:
        self._env.logger.debug("Waiting for Docker daemon inside DinD sandbox...")
        last_output = ""
        for _ in range(self._DOCKER_DAEMON_TIMEOUT_SEC // 2):
            result = await self._vm_exec("docker info", timeout_sec=10)
            if result.return_code == 0:
                self._env.logger.debug("Docker daemon is ready")
                return
            last_output = (result.stdout or "") + (result.stderr or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {self._DOCKER_DAEMON_TIMEOUT_SEC}s. "
            f"Last output: {last_output}"
        )

    async def _wait_for_main_container(self, timeout_sec: int = 60) -> None:
        self._env.logger.debug("Waiting for main container to be running...")
        for _ in range(timeout_sec // 2):
            # A builtin in the system shell by path, its loader variables
            # empty: nothing from the image's PATH (C8).
            result = await self._compose_exec(
                [
                    "exec",
                    "-T",
                    *system_shell_env_flags(),
                    "main",
                    SYSTEM_SHELL,
                    "-c",
                    ":",
                ],
                timeout_sec=10,
            )
            if result.return_code == 0:
                self._env.logger.debug("Main container is running")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

    async def _upload_task_folder(self) -> list[str]:
        """Upload the task folder the compose file sees; return the mount points.

        On Docker, ``${CONTEXT_DIR}/..`` is the task folder on the host, so a
        compose file can mount it (ML-Dev-Bench mounts it read-only at
        ``/opt/task-src`` and ``setup_workspace/`` beside it, #1181). Here the
        task folder goes to ``_TASK_DIR`` without its oracle and verifier
        folders; the returned paths are the empty folders to make in their
        place, one per folder the task has, so a volume over them mounts as
        on Docker. An ``environment_dir`` that is not inside a task folder
        uploads only itself.
        """
        env = self._env
        task_dir = _task_folder(env.environment_dir)
        if task_dir is None:
            await env._sdk_upload_dir(
                env.environment_dir, self._ENVIRONMENT_DIR, route=VM_ROUTE
            )
            return []
        # Private package copies and Git history can both retain held-out
        # answers even when their working-tree folders were withheld.
        await env._sdk_upload_dir(
            task_dir,
            self._TASK_DIR,
            exclude=(*_HIDDEN_TASK_DIRS, ".taskmd", ".git"),
            route=VM_ROUTE,
        )
        return [
            f"{self._TASK_DIR}/{name}"
            for name in _HIDDEN_TASK_DIRS
            if os.path.lexists(task_dir / name)
        ]

    async def start(self, force_build: bool) -> None:
        from benchflow.sandbox import daytona as _sdk

        env = self._env

        resources = _sdk.Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )

        env._client_manager = await _sdk.DaytonaClientManager.get_instance()

        dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
        dind_snapshot: str | None = env._kwargs.get("dind_snapshot")

        params: _SandboxParams
        if dind_snapshot:
            params = _sdk.CreateSandboxFromSnapshotParams(
                snapshot=dind_snapshot,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                network_block_all=False,
                labels=_benchflow_owned_labels(),
            )
        else:
            image = _sdk.Image.base(dind_image)
            params = _sdk.CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=False,
                labels=_benchflow_owned_labels(),
            )

        env._create_attempts = 0
        try:
            # The VM runs BenchFlow's own DinD image (or the operator's
            # snapshot of it), never the task's: the task image's env
            # reaches only the inner containers, which the bootstrap overlay
            # admits (_inert_bootstrap.overlay, startup_name).
            await env._create_sandbox(params=params)
        except Exception as e:
            raise env._startup_failure(e) from e

        try:
            env.logger.debug("Starting Docker daemon inside DinD sandbox...")
            # Root's staging folder on the VM, root's and mode 700 before
            # the first staged command (each staged command proves it again
            # before it opens its file; DX-1 review A L2).
            run = shlex.quote(STAGE_ROOT)
            await self._vm_exec(
                f"mkdir -p {run} && chown 0:0 {run} && chmod 700 {run}; "
                "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
                timeout_sec=10,
            )

            await self._wait_for_docker_daemon()
            await self._guard_daemon_from_containers()
            # The inner Engine's report decides the kept-service freezer
            # before Compose starts anything.
            await decide_kept_service_freezer(env, self._engine_info_json)

            # Upload BenchFlow compose files to the sandbox
            for path in self._BENCHFLOW_COMPOSE_FILES:
                await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")

            mount_points = await self._upload_task_folder()

            # Create log directories, and the empty stand-ins for the task
            # folders that were left out.
            log_dirs = [
                f"{self._LOGS_DIR}/{name}"
                for name in ("verifier", "agent", "artifacts")
            ]
            await self._vm_exec(
                shlex.join(["mkdir", "-p", *log_dirs, *mount_points])
                + " && "
                + shlex.join(["chmod", "777", *log_dirs])
            )

            # Build and start compose services
            self._use_prebuilt = not force_build and bool(
                env.task_env_config.docker_image
            )
            await self._upload_runtime_compose_file()

            # Compose resolves the task's files here, in the VM; what it
            # resolves is checked before the task's hook runs, and again
            # after it, since the hook can rewrite them.
            await self._check_compose_project()
            env.logger.debug(
                "Running pre-compose hook inside DinD sandbox if present..."
            )
            await self._run_pre_compose_hook()
            hook = env.environment_dir / "benchflow-pre-compose.sh"
            if hook.is_file() and not hook.is_symlink():
                await self._check_compose_project()
            await self._prepare_compose_builds()

            env.logger.debug("Building compose services inside DinD sandbox...")
            # The SDK takes integer seconds. Enforce the exact budget here,
            # including fractions, and settle/delete the transient build
            # session on timeout or cancellation. Up is never allowed to build.
            budget = env.task_env_config.build_timeout_sec
            try:
                result = await asyncio.wait_for(
                    self._compose_exec(
                        ["build"],
                        timeout_sec=max(1, math.ceil(budget)),
                        cleanup_session=True,
                    ),
                    timeout=budget,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Docker build timed out after {budget} seconds"
                ) from exc
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose build failed: {result.stdout} {result.stderr}"
                )

            await self._prepare_bfx()
            await self._prepare_inert_bootstrap()
            env.logger.debug("Starting compose services inside DinD sandbox...")
            await self._compose_up_with_retry(env)

            await self._wait_for_main_container()
            await self._guard_compose_networks()
            await self._pin_or_verify_bind_targets()
            await _inert_bootstrap.release(self, self._compose_exec)
        except Exception as e:
            _raise_if_startup_provider_error(env, e)
            raise

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        if env._sandbox:
            try:
                await self._compose_teardown(timeout_sec=30)
            except Exception as e:
                env.logger.warning(f"docker compose down failed: {e}")

        try:
            if not env._sandbox:
                env._log_no_sandbox_to_stop()
            else:
                try:
                    await env._stop_sandbox()
                except BaseException as e:
                    # Cancelled too: the delete may be in flight (the ledger
                    # confirms it).
                    env._log_sandbox_left_behind(e)
                    if not isinstance(e, Exception):
                        raise
                finally:
                    env._sandbox = None
        finally:
            env._client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult:
        return await self._exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            cleanup_session=False,
            route=route,
        )

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult:
        return await self._exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            cleanup_session=True,
            route=route,
        )

    async def _exec(
        self,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
        service: str,
        cleanup_session: bool,
        *,
        route: ExecRoute,
    ) -> ExecResult:
        """Run a command in a compose service inside the DinD VM.

        ``service`` defaults to ``"main"`` (the agent container); pass
        another name to target an additional container declared in the
        task's ``docker-compose.yaml`` (vulhub-style targets — see #248).
        """
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if user is not None:
            parts.extend(["-u", str(user)])
        # C8's routes (see benchflow.sandbox._base): plumbing pins
        # route.pin; the task's own command as root (route.task_root) starts
        # without the loader variables too and gets them back before it runs.
        pin, task_root = route.pin, route.task_root
        if pin or task_root:
            # The image's loader variables, set empty before /bin/sh loads
            # (empty values only, nothing secret).
            parts.extend(system_shell_env_flags())
        # Env vars are materialized inside the target container via a
        # mode-0600 file and sourced, rather than passed as ``-e KEY=VALUE``
        # flags. The flags would land in the DinD VM's process list (visible
        # to ``ps`` and any Daytona session audit log) and leak verifier
        # API keys / agent secrets on every multi-service task (#412
        # follow-up). Matches the wrapping that ``DaytonaSandbox._sandbox_exec``
        # already applies to the outer VM-side command. An env too large for
        # one command is refused before anything runs (b173 PY8-MAIN-39).
        if not pin and not getattr(
            getattr(self, "_env", None), "_services_are_kept", False
        ):
            restored = _inert_bootstrap.restored_env(self, service)
            if task_root:
                # The receipt holds image defaults, not an explicit task override.
                # Keep them inert until PT's provenance-aware restore admits them.
                restored = {
                    f"_BF_IMAGE_{name}": value for name, value in restored.items()
                }
            env = {**restored, **(env or {})}
        if not pin and task_root:
            command = (
                restore_image_env(
                    env or (), tools=route.tools if route.kit and route.root else None
                )
                + command
            )
        if env:
            command = _wrap_daytona_command_with_env_file(
                env, command, path=route.tools, private=route.kit and route.root
            )
        # The heartbeat starts the command as a background job, whose stdin
        # is /dev/null: it stays inside what a staged command's reader runs.
        command = _with_long_exec_heartbeat(command, timeout_sec, route.tools)

        def routed(inner: str) -> list[str]:
            if pin:
                inner = with_system_path(inner, pin)
            else:
                inner = _root_exec_command(
                    inner, route.root and user is not None, route.root_path
                )
                if task_root:
                    inner = capture_pid1_image_env(route.tools) + inner
            # Use POSIX ``sh`` rather than ``bash``: with multi-service support
            # (#248), ``service`` can target arbitrary task containers, and
            # Alpine/distroless/minimal images often ship no ``/bin/bash``. The
            # wrapped command uses only POSIX constructs (``trap``, ``base64 -d``,
            # ``set -a``/``. file``).
            # The route's shell: the protected kit's for root in ``main`` once
            # the sandbox has one (C8).
            return [*parts, service, route.shell, "-c", inner]

        # The transport contract (benchflow.sandbox._transport), at each
        # level where a command is built: the container's argv (one argument
        # of ``docker compose exec``) fits the bound, or the container's
        # shell runs the reader and the command is its stdin, which
        # ``docker compose exec`` passes on from root's staged file on the
        # VM. The VM's own session command is Daytona's to fit in turn.
        fitted = fit(routed, command, cat=reader_cat(route))
        if fitted.payload is not None:
            command = fitted.line
        extra: dict[str, Any] = {}
        if cleanup_session:
            extra["cleanup_session"] = True
        if fitted.payload is not None:
            extra["stdin_data"] = fitted.payload
        return await self._compose_exec(
            routed(command), timeout_sec=timeout_sec, **extra
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            target_parent_cmd = compose_parent_mkdir_p_command(target_path)
            if target_parent_cmd is not None:
                prep_result = await exec_raw(
                    self.exec,
                    target_parent_cmd,
                    timeout_sec=30,
                    user="root",
                    service="main",
                    route=await system_shell_route(self._env),
                )
                if prep_result.return_code != 0:
                    raise RuntimeError(
                        "docker compose upload_file destination prep failed: "
                        f"{_exec_failure_output(prep_result)}"
                    )
            result = await self._compose_exec(
                ["cp", temp, compose_cp_destination("main", target_path)],
                timeout_sec=60,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service inside the DinD VM.

        ``service`` defaults to ``"main"``; pass a target service to land the
        directory in an additional vulhub-style container (#248).
        """
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp, route=VM_ROUTE)
            prep_result = await exec_raw(
                self.exec,
                compose_mkdir_p_command(target_dir),
                timeout_sec=30,
                user="root",
                service=service,
                route=await system_shell_route(self._env, service),
            )
            if prep_result.return_code != 0:
                raise RuntimeError(
                    "docker compose upload_dir destination prep failed: "
                    f"{_exec_failure_output(prep_result)}"
                )
            result = await self._compose_exec(
                ["cp", f"{temp}/.", compose_cp_destination(service, target_dir)],
                timeout_sec=120,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        # Compose overrides can replace any base bind with another bind,
        # volume or tmpfs. Only the container namespace is authoritative.
        await self._compose_download_file(source_path, target_path)

    async def _compose_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await self._check_compose_download_file(source_path)
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            result = await self._compose_exec(
                ["cp", f"main:{source_path}", temp], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_file(temp, target_path)
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def _download_path_stat(self, container: str, source: str) -> dict | None:
        """Inspect through the VM's daemon, without running task-image tools.

        Archive HEAD returns 404 for both a missing path and a lost container;
        only a later successful ancestor lookup can establish file absence.
        """
        endpoint = (
            f"/containers/{urllib.parse.quote(container, safe='')}/archive?"
            + urllib.parse.urlencode({"path": source})
        )
        response = await exec_raw(
            self._vm_exec,
            "docker system dial-stdio",
            stdin_payload=(
                f"HEAD {endpoint} HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"
            ).encode("ascii"),
            timeout_sec=20,
            cleanup_session=True,
        )
        header, separator, _ = (response.stdout or "").partition("\r\n\r\n")
        lines = header.split("\r\n")
        status = lines[0].split()
        if (
            response.return_code != 0
            or not separator
            or len(status) < 2
            or status[0] != "HTTP/1.1"
        ):
            raise RuntimeError("DinD archive metadata response is not HTTP/1.1")
        if status[1] == "404":
            return None
        if status[1] != "200":
            raise RuntimeError(f"DinD archive metadata failed (HTTP {status[1]})")
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
            raise RuntimeError("DinD archive metadata is invalid") from exc
        return metadata

    async def _check_compose_download_file(self, source: str) -> None:
        result = await exec_raw(
            self._compose_exec, ["ps", "--all", "-q", "main"], timeout_sec=30
        )
        ids = (result.stdout or "").strip().splitlines()
        if result.return_code != 0 or len(ids) != 1:
            raise RuntimeError("DinD could not locate the capture container")
        metadata = await self._download_path_stat(ids[0], source)
        type_mask, directory_mode = 0x8F280000, 0x80000000
        if metadata is not None:
            if metadata["mode"] & type_mask or metadata.get("linkTarget"):
                raise UnsafeDownloadEntryError(f"not a regular remote file: {source}")
            return
        for parent in PurePosixPath(source).parents:
            metadata = await self._download_path_stat(ids[0], str(parent))
            if metadata is None:
                continue
            if metadata["mode"] & type_mask != directory_mode or metadata.get(
                "linkTarget"
            ):
                raise UnsafeDownloadEntryError(f"unsafe remote file parent: {source}")
            raise RemoteFileAbsentError(f"remote file is absent: {source}")
        raise RuntimeError("DinD could not confirm the capture container exists")

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service inside the DinD VM.

        ``service`` defaults to ``"main"``; pass a target service to fetch
        target-side verifier output from a vulhub-style container (#248).
        Always copy from the selected service's namespace: a task may replace
        any base log bind, including with an empty but valid named volume.
        """
        await self._compose_download_dir(source_dir, target_dir, service=service)

    async def _compose_download_dir(
        self, source_dir: str, target_dir: Path | str, service: str
    ) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"{service}:{source_dir}/.", temp], timeout_sec=120
            )
            if result.return_code != 0:
                self._env.logger.error(
                    f"download_dir: docker compose cp failed: {result.stdout} {result.stderr}"
                )
                raise RuntimeError(
                    f"download_dir: docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_dir(temp, target_dir)
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def is_dir(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool:
        result = await exec_raw(
            self.exec,
            f"test -d {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
            route=route,
        )
        return result.return_code == 0

    async def is_file(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool:
        result = await exec_raw(
            self.exec,
            f"test -f {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
            route=route,
        )
        return result.return_code == 0

    async def script_container_config(self) -> dict[str, Any]:
        """The inner Docker container has the same non-root PID-1 gap."""
        return await self.identity_container_config()

    async def _identity_container_id(self, service: str) -> str:
        result = await self._compose_exec(["ps", "-q", service], timeout_sec=30)
        ids = (result.stdout or "").strip().splitlines()
        if result.return_code != 0 or len(ids) != 1:
            raise RuntimeError("script: could not locate the Docker task container")
        return ids[0]

    async def identity_container_config(
        self, service: str = "main", *, container: str | None = None
    ) -> dict[str, Any]:
        if container is None:
            container = await self._identity_container_id(service)
        result = await self._vm_exec(
            "docker inspect --format '{{json .Config}}' " + shlex.quote(container),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "script: could not read the Docker container environment"
            )
        try:
            config = json.loads(result.stdout or "")
        except json.JSONDecodeError:
            raise RuntimeError(
                "script: invalid Docker container configuration"
            ) from None
        if not isinstance(config, dict):
            raise RuntimeError("script: invalid Docker container configuration")
        return config

    async def identity_exec(
        self, command: str, *, service: str = "main", **kwargs: Any
    ) -> ExecResult:
        from benchflow.sandbox._identities import (
            protected_exec_argv,
            removed_env_names,
        )

        container = await self._identity_container_id(service)
        config = await self.identity_container_config(service, container=container)
        argv = protected_exec_argv(
            config,
            command,
            service=container,
            route=await system_shell_route(self._env, service),
        )
        # Plain docker exec on the container, not compose exec: a bare -e NAME
        # (a removed loader name) is filled in only from the docker CLI's own
        # environment, unset here, never from the project's .env as Compose
        # would (SCRIPT-AGENT-DOCKER-F1). docker exec has no -T (Compose's,
        # which protected_exec_argv puts second): it never allocates a TTY
        # unless asked.
        argv.pop(1)
        # Only shell identifiers: `unset LD_ODD-NAME` ends the shell with
        # status 2 before docker runs (round 2, review B M1). Docker accepts
        # such a name, but no loader reads it, so protected_exec_argv sets
        # it empty rather than removing it, and it never needs unsetting.
        unset = sorted(
            name
            for name in {*removed_env_names(argv), *_inert_bootstrap.EMPTY_ENV}
            if name.isascii() and name.isidentifier()
        )
        return await self._vm_exec(
            f"unset {' '.join(shlex.quote(n) for n in unset)}; docker {shlex.join(argv)}",
            timeout_sec=kwargs.get("timeout_sec"),
        )

    async def services(self) -> list[str]:
        """List compose service names defined inside the DinD VM.

        Runs ``docker compose config --services`` against the task's compose
        stack — includes BenchFlow's ``main`` service plus any vulhub-style
        target/database containers the task declares (#248). The output is
        filtered to the compose service-name grammar so stray warning lines
        cannot be mistaken for services.
        """
        result = await self._compose_exec(["config", "--services"], timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose config --services failed: "
                f"{result.stdout} {result.stderr}"
            )
        return _filter_compose_service_names(result.stdout or "")

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()

        from benchflow.sandbox.docker import interactive_root_shell, route_exec_argv

        # Root's interactive shell comes from the route (C8): an absolute
        # shell (the protected kit's once the sandbox has one), no profile or
        # rc file, the loader variables empty where compose starts it.
        route = await env.exec_route("root", system_shell=True)
        compose_cmd = self._compose_cmd(
            route_exec_argv(
                route,
                "",
                container="main",
                user="0",
                stdin=True,
                tty=True,
                shell_argv=interactive_root_shell(route),
            )
        )
        compose_env = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in self._compose_env_vars().items()
        )
        remote_cmd = f"{compose_env} {compose_cmd}"

        os.execvp(
            "ssh",
            [
                "ssh",
                "-t",
                f"{ssh_access.token}@ssh.app.daytona.io",
                remote_cmd,
            ],
        )


class DinDBfxTransport:
    """bfx's volume steps (:func:`benchflow._bfx.delivery.ensure_volume`) on
    the inner daemon, through the DinD VM's control path: the ``docker`` CLI
    runs on the VM, the verified binary reaches the VM by the SDK's upload,
    and digests are taken on the VM, so no binary travels as command text."""

    def __init__(self, strategy: _DaytonaDinD) -> None:
        self._strategy = strategy

    async def run(
        self, args: list[str], *, timeout: float = 120.0
    ) -> tuple[int, bytes, bytes]:
        result = await self._strategy._vm_exec(
            shlex.join(["docker", *args]), timeout_sec=max(1, math.ceil(timeout))
        )
        return (
            result.return_code,
            (result.stdout or "").encode(),
            (result.stderr or "").encode(),
        )

    async def file_digest(self, container: str, path: str) -> str | None:
        source = shlex.quote(f"{container}:{path}")
        result = await self._strategy._vm_exec(
            "d=$(mktemp -d) && docker cp " + source + ' "$d/f" && '
            '[ -f "$d/f" ] && [ ! -L "$d/f" ] && sha256sum "$d/f"; '
            's=$?; rm -rf "$d"; exit $s',
            timeout_sec=120,
        )
        digest = (result.stdout or "").split(" ")[0].strip()
        if result.return_code != 0 or len(digest) != 64:
            return None
        return digest

    async def folder_names(self, container: str, path: str) -> list[str] | None:
        source = shlex.quote(f"{container}:{path}")
        result = await self._strategy._vm_exec(
            "d=$(mktemp -d) && docker cp " + source + ' "$d/v" && '
            '[ -d "$d/v" ] && [ ! -L "$d/v" ] && ls -A1 "$d/v"; '
            's=$?; rm -rf "$d"; exit $s',
            timeout_sec=120,
        )
        if result.return_code != 0:
            return None
        return sorted(n for n in (result.stdout or "").splitlines() if n)

    async def import_image(
        self, archive: Path, changes: list[str], *, timeout: float
    ) -> tuple[int, bytes, bytes]:
        """``docker import`` on the VM of *archive* (the helper image's tar,
        made on the host from the verified bytes), sent by the SDK's upload."""
        stage = f"/tmp/benchflow-bfx-{uuid4().hex[:12]}"
        made = await self._strategy._vm_exec(
            shlex.join(["mkdir", "-m", "0700", stage]), timeout_sec=30
        )
        if made.return_code != 0:
            return made.return_code, b"", b"cannot stage bfx on the DinD VM"
        try:
            await self._strategy._env._sdk_upload_file(archive, f"{stage}/bfx.tar")
            flags = [part for change in changes for part in ("--change", change)]
            return await self.run(
                ["import", *flags, f"{stage}/bfx.tar"], timeout=timeout
            )
        finally:
            await self._strategy._vm_exec(
                shlex.join(["rm", "-rf", stage]), timeout_sec=30
            )
