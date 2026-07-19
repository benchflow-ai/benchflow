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
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchflow.sandbox._base import ExecResult, _filter_compose_service_names
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    COMPOSE_UP_RETRY_DELAYS_SEC,
    compose_cp_destination,
    compose_mkdir_p_command,
    compose_parent_mkdir_p_command,
    is_compose_up_network_race_error,
)
from benchflow.sandbox.daytona_pty import (
    _exec_failure_output,
    _wrap_daytona_command_with_env_file,
)
from benchflow.sandbox.daytona_reaper import _benchflow_owned_labels
from benchflow.sandbox.daytona_strategies import _DaytonaStrategy
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import SandboxPaths

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


def _with_long_exec_heartbeat(command: str, timeout_sec: int | None) -> str:
    if (
        timeout_sec is None
        or timeout_sec < _LONG_EXEC_HEARTBEAT_MIN_TIMEOUT_SEC
        or _LONG_EXEC_HEARTBEAT_INTERVAL_SEC <= 0
    ):
        return command
    interval = shlex.quote(str(_LONG_EXEC_HEARTBEAT_INTERVAL_SEC))
    return "\n".join(
        [
            "bf_out=$(mktemp)",
            "bf_err=$(mktemp)",
            'bf_stop="${bf_out}.stop"',
            'rm -f "$bf_stop"',
            'bf_cleanup() { rm -f "$bf_out" "$bf_err" "$bf_stop"; }',
            "trap bf_cleanup EXIT",
            f'({command}\n) >"$bf_out" 2>"$bf_err" &',
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
    _ENVIRONMENT_DIR = "/benchflow/environment"
    _LOGS_DIR = "/benchflow/logs"

    def __init__(self, env: DaytonaSandbox) -> None:
        super().__init__(env)
        self._use_prebuilt = False

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

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        cleanup_session: bool = False,
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM using sh (Alpine-compatible)."""
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell="sh -c",
            cleanup_session=cleanup_session,
        )

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
            f"{self._ENVIRONMENT_DIR}/docker-compose.yaml",
        ]
        if not self._env.task_env_config.allow_internet:
            files.append(f"{self._COMPOSE_DIR}/docker-compose-no-network.yaml")

        flags: list[str] = []
        for f in files:
            flags.extend(["-f", f])
        return flags

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
    ) -> ExecResult:
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
            cleanup_session=cleanup_session,
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
            result = await self._compose_exec(["up", "-d"], timeout_sec=timeout_sec)
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
            result = await self._compose_exec(
                ["exec", "-T", "main", "true"], timeout_sec=10
            )
            if result.return_code == 0:
                self._env.logger.debug("Main container is running")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

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

        try:
            await env._create_sandbox(params=params)
        except (TimeoutError, RuntimeError, Exception) as e:
            sandbox_id = getattr(env._sandbox, "id", None) if env._sandbox else None
            raise SandboxStartupError(
                f"Sandbox creation failed after retries: {e}",
                sandbox_id=sandbox_id,
                sandbox_state="error",
                attempts=3,
                build_timeout_sec=env.task_env_config.build_timeout_sec,
            ) from e

        try:
            env.logger.debug("Starting Docker daemon inside DinD sandbox...")
            await self._vm_exec(
                "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
                timeout_sec=10,
            )

            await self._wait_for_docker_daemon()

            # Upload BenchFlow compose files to the sandbox
            for path in (
                COMPOSE_BASE_PATH,
                COMPOSE_BUILD_PATH,
                COMPOSE_PREBUILT_PATH,
                COMPOSE_NO_NETWORK_PATH,
            ):
                await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")

            # Upload task environment directory
            await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)

            # Create log directories
            await self._vm_exec(
                f"mkdir -p {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
                f"{self._LOGS_DIR}/artifacts && "
                f"chmod 777 {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
                f"{self._LOGS_DIR}/artifacts"
            )

            # Build and start compose services
            self._use_prebuilt = not force_build and bool(
                env.task_env_config.docker_image
            )

            env.logger.debug(
                "Running pre-compose hook inside DinD sandbox if present..."
            )
            await self._run_pre_compose_hook()

            env.logger.debug("Building compose services inside DinD sandbox...")
            result = await self._compose_exec(
                ["build"],
                timeout_sec=round(env.task_env_config.build_timeout_sec),
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose build failed: {result.stdout} {result.stderr}"
                )

            env.logger.debug("Starting compose services inside DinD sandbox...")
            await self._compose_up_with_retry(env)

            await self._wait_for_main_container()
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
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as e:
                env.logger.warning(f"docker compose down failed: {e}")

        try:
            if not env._sandbox:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
            else:
                try:
                    await env._stop_sandbox()
                except Exception as e:
                    env.logger.error(f"Error stopping sandbox {env._sandbox.id}: {e}")
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
    ) -> ExecResult:
        return await self._exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            cleanup_session=False,
        )

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        return await self._exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            cleanup_session=True,
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
        # Env vars are materialized inside the target container via a
        # mode-0600 file and sourced, rather than passed as ``-e KEY=VALUE``
        # flags. The flags would land in the DinD VM's process list (visible
        # to ``ps`` and any Daytona session audit log) and leak verifier
        # API keys / agent secrets on every multi-service task (#412
        # follow-up). Matches the wrapping that ``DaytonaSandbox._sandbox_exec``
        # already applies to the outer VM-side command.
        if env:
            command = _wrap_daytona_command_with_env_file(env, command)
        command = _with_long_exec_heartbeat(command, timeout_sec)
        # Use POSIX ``sh`` rather than ``bash``: with multi-service support
        # (#248), ``service`` can target arbitrary task containers, and
        # Alpine/distroless/minimal images often ship no ``/bin/bash``. The
        # wrapped command uses only POSIX constructs (``trap``, ``base64 -d``,
        # ``set -a``/``. file``).
        parts.extend([service, "sh", "-c", command])

        if cleanup_session:
            return await self._compose_exec(
                parts,
                timeout_sec=timeout_sec,
                cleanup_session=True,
            )
        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            target_parent_cmd = compose_parent_mkdir_p_command(target_path)
            if target_parent_cmd is not None:
                prep_result = await self.exec(
                    target_parent_cmd,
                    timeout_sec=30,
                    user="root",
                    service="main",
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
            await self._env._sdk_upload_dir(source_dir, temp)
            prep_result = await self.exec(
                compose_mkdir_p_command(target_dir),
                timeout_sec=30,
                user="root",
                service=service,
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

    def _sandbox_log_path(self, container_path: str) -> str | None:
        mappings = {
            str(SandboxPaths.verifier_dir): f"{self._LOGS_DIR}/verifier",
            str(SandboxPaths.agent_dir): f"{self._LOGS_DIR}/agent",
            str(SandboxPaths.artifacts_dir): f"{self._LOGS_DIR}/artifacts",
        }
        for env_prefix, sandbox_prefix in mappings.items():
            if container_path == env_prefix or container_path.startswith(
                env_prefix + "/"
            ):
                return container_path.replace(env_prefix, sandbox_prefix, 1)
        return None

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox_path = self._sandbox_log_path(source_path)
        if sandbox_path:
            try:
                await self._env._sdk_download_file(sandbox_path, target_path)
            except Exception as host_error:
                self._env.logger.warning(
                    "Daytona host log download_file failed for %s; falling back "
                    "to docker compose cp: %s",
                    source_path,
                    host_error,
                )
                try:
                    await self._compose_download_file(source_path, target_path)
                except Exception as compose_error:
                    raise RuntimeError(
                        "Daytona log download_file failed via host path and "
                        "compose fallback"
                    ) from compose_error
            return

        await self._compose_download_file(source_path, target_path)

    async def _compose_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
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

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service inside the DinD VM.

        ``service`` defaults to ``"main"``; pass a target service to fetch
        target-side verifier output from a vulhub-style container (#248).
        The host-mounted-log fast path only applies to ``main`` — target
        services have no host bind mount, so their contents are always
        copied out via ``docker compose cp``.
        """
        if service == "main":
            sandbox_path = self._sandbox_log_path(source_dir)
            if sandbox_path:
                try:
                    await self._env._sdk_download_dir(sandbox_path, target_dir)
                except Exception as host_error:
                    self._env.logger.warning(
                        "Daytona host log download_dir failed for %s; falling "
                        "back to docker compose cp: %s",
                        source_dir,
                        host_error,
                    )
                    try:
                        await self._compose_download_dir(
                            source_dir, target_dir, service=service
                        )
                    except Exception as compose_error:
                        raise RuntimeError(
                            "Daytona log download_dir failed via host path and "
                            "compose fallback"
                        ) from compose_error
                return

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

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

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

        compose_cmd = self._compose_cmd(["exec", "-it", "main", "bash"])
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
