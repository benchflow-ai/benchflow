from __future__ import annotations

import asyncio
import logging
import os
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.contracts.task_config import EnvironmentConfig
from benchflow.contracts.paths import EnvironmentPaths, TrialPaths
from benchflow.sandbox._base import BaseSandbox, ExecResult, SandboxType
from benchflow.sandbox._client import DaytonaClientManager
from benchflow.sandbox._sdk_ops import (
    create_sandbox,
    import_daytona_sdk,
    sandbox_exec,
    sdk_download_dir,
    sdk_download_file,
    sdk_upload_dir,
    sdk_upload_file,
    stop_sandbox,
)
from benchflow.sandbox._compose import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    resolve_env_vars,
)

logger = logging.getLogger(__name__)
_SandboxParams = Any


def _daytona_preflight() -> None:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit(
            "Daytona requires DAYTONA_API_KEY to be set. "
            "Please set this environment variable and try again."
        )


class _DaytonaStrategy(ABC):
    """Base for Daytona implementation strategies."""

    def __init__(self, env: DaytonaSandbox) -> None:
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None: ...

    @abstractmethod
    async def is_dir(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def attach(self) -> None: ...


class _DaytonaDirect(_DaytonaStrategy):
    """Single-container Daytona sandbox strategy."""

    async def start(self, force_build: bool) -> None:
        env = self._env
        sdk = import_daytona_sdk()
        resources = sdk.Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )

        env.client_manager = await DaytonaClientManager.get_instance()
        daytona = await env.client_manager.get_client()

        snapshot_name: str | None = None
        snapshot_exists = False
        if env._snapshot_template_name:
            snapshot_name = env._snapshot_template_name.format(
                name=env.environment_name
            )
            try:
                snapshot = await daytona.snapshot.get(snapshot_name)
                if snapshot.state == sdk.SnapshotState.ACTIVE:
                    snapshot_exists = True
            except Exception:
                snapshot_exists = False

        if snapshot_exists and force_build:
            env.logger.warning(
                "Snapshot template specified but force_build is True. "
                "Snapshot will be used instead of building from scratch."
            )

        params: _SandboxParams
        if snapshot_exists and snapshot_name:
            env.logger.debug("Using snapshot: %s", snapshot_name)
            params = sdk.CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=snapshot_name,
                network_block_all=env._network_block_all,
            )
        elif force_build or not env.task_env_config.docker_image:
            env.logger.debug("Building environment from %s", env._dockerfile_path)
            image = sdk.Image.from_dockerfile(env._dockerfile_path)
            params = sdk.CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=env._network_block_all,
            )
        else:
            env.logger.debug(
                "Using prebuilt image: %s", env.task_env_config.docker_image
            )
            image = sdk.Image.base(env.task_env_config.docker_image)
            params = sdk.CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=env._network_block_all,
            )

        await env._create_sandbox(params=params)
        await env._sandbox_exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} && "
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )
        try:
            if not env.sandbox:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
            else:
                try:
                    await env._stop_sandbox()
                except Exception as exc:
                    env.logger.error(
                        "Error stopping sandbox %s: %s", env.sandbox.id, exc
                    )
                finally:
                    env.sandbox = None
        finally:
            env.client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        if not self._env.sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        file_info = await self._env.sandbox.fs.get_file_info(path)
        return file_info.is_dir

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        if not self._env.sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        file_info = await self._env.sandbox.fs.get_file_info(path)
        return not file_info.is_dir

    async def attach(self) -> None:
        env = self._env
        if not env.sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        ssh_access = await env.sandbox.create_ssh_access()
        os.execvp("ssh", ["ssh", f"{ssh_access.token}@ssh.app.daytona.io"])


class _DaytonaDinD(_DaytonaStrategy):
    """Compose-over-DinD Daytona strategy for multi-container tasks."""

    _DOCKER_DAEMON_TIMEOUT_SEC = 60
    _COMPOSE_DIR = "/benchflow/compose"
    _ENVIRONMENT_DIR = "/benchflow/environment"
    _LOGS_DIR = "/benchflow/logs"

    def __init__(self, env: DaytonaSandbox) -> None:
        super().__init__(env)
        self._use_prebuilt = False
        self._resolved_task_env: dict[str, str] = {}
        benchflow_keys = set(self.compose_env_vars().keys()) - set(
            env._persistent_env.keys()
        )
        if env.task_env_config.env:
            self._resolved_task_env = resolve_env_vars(env.task_env_config.env)
        resolved_task_keys = set(self._resolved_task_env.keys()) | set(
            env._persistent_env.keys()
        )
        if resolved_task_keys:
            collisions = benchflow_keys & resolved_task_keys
            if collisions:
                env.logger.warning(
                    "Environment vars override compose variable(s): %s",
                    ", ".join(sorted(collisions)),
                )

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._env._sandbox_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell="sh -c"
        )

    def compose_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {
            "CONTEXT_DIR": self._ENVIRONMENT_DIR,
            "MAIN_IMAGE_NAME": f"hb__{self._env.environment_name}",
            "HOST_VERIFIER_LOGS_PATH": f"{self._LOGS_DIR}/verifier",
            "HOST_AGENT_LOGS_PATH": f"{self._LOGS_DIR}/agent",
            "HOST_ARTIFACTS_PATH": f"{self._LOGS_DIR}/artifacts",
            "ENV_VERIFIER_LOGS_PATH": str(EnvironmentPaths.verifier_dir),
            "ENV_AGENT_LOGS_PATH": str(EnvironmentPaths.agent_dir),
            "ENV_ARTIFACTS_PATH": str(EnvironmentPaths.artifacts_dir),
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
        for file_path in files:
            flags.extend(["-f", file_path])
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
        self, subcommand: list[str], timeout_sec: int | None = None
    ) -> ExecResult:
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self.compose_env_vars(),
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
        env = self._env
        sdk = import_daytona_sdk()
        resources = sdk.Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )
        env.client_manager = await DaytonaClientManager.get_instance()

        dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
        dind_snapshot: str | None = env._kwargs.get("dind_snapshot")
        if dind_snapshot:
            params = sdk.CreateSandboxFromSnapshotParams(
                snapshot=dind_snapshot,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                network_block_all=False,
            )
        else:
            image = sdk.Image.base(dind_image)
            params = sdk.CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=False,
            )

        await env._create_sandbox(params=params)
        env.logger.debug("Starting Docker daemon inside DinD sandbox...")
        await self._vm_exec(
            "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
            timeout_sec=10,
        )
        await self._wait_for_docker_daemon()

        for path in (
            COMPOSE_BASE_PATH,
            COMPOSE_BUILD_PATH,
            COMPOSE_PREBUILT_PATH,
            COMPOSE_NO_NETWORK_PATH,
        ):
            await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")
        await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)
        await self._vm_exec(
            f"mkdir -p {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts && "
            f"chmod 777 {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts"
        )

        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)
        env.logger.debug("Building compose services inside DinD sandbox...")
        result = await self._compose_exec(
            ["build"], timeout_sec=round(env.task_env_config.build_timeout_sec)
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
        if env.sandbox:
            try:
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as exc:
                env.logger.warning("docker compose down failed: %s", exc)
        try:
            if not env.sandbox:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
            else:
                try:
                    await env._stop_sandbox()
                except Exception as exc:
                    env.logger.error(
                        "Error stopping sandbox %s: %s", env.sandbox.id, exc
                    )
                finally:
                    env.sandbox = None
        finally:
            env.client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                parts.extend(["-e", f"{key}={value}"])
        if user is not None:
            parts.extend(["-u", str(user)])
        parts.extend(["main", "bash", "-lc", command])
        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            result = await self._compose_exec(
                ["cp", temp, f"main:{target_path}"], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp)
            result = await self._compose_exec(
                ["cp", f"{temp}/.", f"main:{target_dir}"], timeout_sec=120
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    def _sandbox_log_path(self, container_path: str) -> str | None:
        mappings = {
            str(EnvironmentPaths.verifier_dir): f"{self._LOGS_DIR}/verifier",
            str(EnvironmentPaths.agent_dir): f"{self._LOGS_DIR}/agent",
            str(EnvironmentPaths.artifacts_dir): f"{self._LOGS_DIR}/artifacts",
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
            await self._env._sdk_download_file(sandbox_path, target_path)
            return
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

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        sandbox_path = self._sandbox_log_path(source_dir)
        if sandbox_path:
            await self._env._sdk_download_dir(sandbox_path, target_dir)
            return
        temp = f"/tmp/benchflow_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"main:{source_dir}/.", temp], timeout_sec=120
            )
            if result.return_code != 0:
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

    async def attach(self) -> None:
        env = self._env
        if not env.sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        ssh_access = await env.sandbox.create_ssh_access()
        compose_cmd = self._compose_cmd(["exec", "-it", "main", "bash"])
        compose_env = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in self.compose_env_vars().items()
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


class DaytonaSandbox(BaseSandbox):
    sandbox_type = SandboxType.DAYTONA

    @classmethod
    def preflight(cls) -> None:
        _daytona_preflight()

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths | None,
        task_env_config: EnvironmentConfig,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
        auto_stop_interval_mins: int = 0,
        auto_delete_interval_mins: int = 0,
        persistent_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._compose_mode = (environment_dir / "docker-compose.yaml").exists()
        self._kwargs = kwargs
        super().__init__()
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.default_user: str | int | None = None
        self._persistent_env: dict[str, str] = persistent_env or {}
        self.logger = logger
        self._auto_stop_interval = auto_stop_interval_mins
        self._auto_delete_interval = auto_delete_interval_mins
        self._snapshot_template_name = snapshot_template_name
        self._validate_definition()
        if network_block_all is not None:
            self._network_block_all = network_block_all
            expected = not task_env_config.allow_internet
            if network_block_all != expected:
                self.logger.warning(
                    "network_block_all=%s overrides task config allow_internet=%s",
                    network_block_all,
                    task_env_config.allow_internet,
                )
        else:
            self._network_block_all = not task_env_config.allow_internet
        self.sandbox: Any | None = None
        self.client_manager: DaytonaClientManager | None = None
        self.strategy: _DaytonaStrategy = (
            _DaytonaDinD(self) if self._compose_mode else _DaytonaDirect(self)
        )
        self.logger.debug("Selected strategy: %s", self.strategy.__class__.__name__)

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return True

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        return user if user is not None else self.default_user

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self._persistent_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        return merged or None

    def _validate_definition(self) -> None:
        path = (
            self._environment_docker_compose_path
            if self._compose_mode
            else self._dockerfile_path
        )
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    async def _create_sandbox(self, params: _SandboxParams) -> None:
        if not self.client_manager:
            raise RuntimeError(
                "Client manager not initialized. This should never happen."
            )
        self.sandbox = await create_sandbox(
            client_manager=self.client_manager,
            params=params,
            build_timeout_sec=self.task_env_config.build_timeout_sec,
        )

    async def _stop_sandbox(self) -> None:
        await stop_sandbox(self.sandbox)

    async def _sandbox_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash -c",
        user: str | int | None = None,
    ) -> ExecResult:
        return await sandbox_exec(
            self.sandbox,
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell=shell,
            user=user,
        )

    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        await sdk_upload_file(self.sandbox, source_path, target_path)

    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await sdk_upload_dir(self.sandbox, source_dir, target_dir)

    async def _sdk_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await sdk_download_file(self.sandbox, source_path, target_path)

    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await sdk_download_dir(self.sandbox, source_dir, target_dir)

    async def start(self, force_build: bool = False) -> None:
        await self.strategy.start(force_build)

    async def stop(self, delete: bool = True) -> None:
        await self.strategy.stop(delete)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)
        return await self.strategy.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.strategy.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.strategy.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.strategy.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self.strategy.download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, *, user: str | int | None = None) -> bool:
        return await self.strategy.is_dir(path, user=self._resolve_user(user))

    async def is_file(self, path: str, *, user: str | int | None = None) -> bool:
        return await self.strategy.is_file(path, user=self._resolve_user(user))

    async def attach(self) -> None:
        await self.strategy.attach()
