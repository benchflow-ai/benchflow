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
    compose_cp_destination,
    compose_mkdir_p_command,
    compose_parent_mkdir_p_command,
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

    _DOCKER_DAEMON_TIMEOUT_SEC = 60
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
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM using sh (Alpine-compatible)."""
        return await self._env._sandbox_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell="sh -c"
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
    ) -> ExecResult:
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

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
        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)

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
        result = await self._compose_exec(["up", "-d"], timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {result.stdout} {result.stderr}"
            )

        await self._wait_for_main_container()

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
        # Use POSIX ``sh`` rather than ``bash``: with multi-service support
        # (#248), ``service`` can target arbitrary task containers, and
        # Alpine/distroless/minimal images often ship no ``/bin/bash``. The
        # wrapped command uses only POSIX constructs (``trap``, ``base64 -d``,
        # ``set -a``/``. file``).
        parts.extend([service, "sh", "-c", command])

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
