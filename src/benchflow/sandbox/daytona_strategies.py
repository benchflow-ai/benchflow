"""Daytona implementation strategies — base interface and direct single-container.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam: the strategy
base class ``_DaytonaStrategy`` and the single-container ``_DaytonaDirect``
strategy. The Docker-in-Docker compose strategy ``_DaytonaDinD`` lives in the
sibling ``daytona_dind`` module (it subclasses ``_DaytonaStrategy`` from here).
All three names are re-exported from ``benchflow.sandbox.daytona`` so existing
imports such as ``from benchflow.sandbox.daytona import _DaytonaDirect`` keep
working unchanged.

The optional Daytona SDK handles (``Resources``, ``Image``,
``CreateSandboxFromImageParams`` …) and the ``DaytonaClientManager`` singleton
live on ``benchflow.sandbox.daytona`` — they are materialized there lazily by
``_load_daytona_sdk`` (#358). The ``start`` and ``restore`` paths reach them
through a local import of that module so the lazy-load / monkeypatch contract
is unchanged.
"""

from __future__ import annotations

import os
import shlex
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.daytona_pty import (
    _exec_failure_output,
    _reject_non_main_service,
)
from benchflow.sandbox.daytona_reaper import _benchflow_owned_labels
from benchflow.sandbox.protocol import (
    SandboxImage,
    SandboxSnapshotNotSupported,
    SandboxStartupError,
)
from benchflow.task.paths import SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.daytona import DaytonaSandbox

# ``_SandboxParams`` mirrors the façade alias: the SDK types are loaded lazily
# (issue #358), so concrete sandbox params are typed as ``Any`` here — callers
# build them inside methods that have already called ``_load_daytona_sdk()``.
_SandboxParams = Any


class _DaytonaStrategy:
    """Base for Daytona implementation strategies."""

    # Strategies declare whether they can satisfy container-level snapshots;
    # the DaytonaSandbox wrapper forwards the question to its strategy so the
    # capability tracks the active mode (direct vs. DinD/compose) — see #384.
    supports_snapshot: bool = False

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
        service: str = "main",
    ) -> ExecResult: ...

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a child-free command, cleaning its provider session if supported."""
        return await self.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
        )

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def is_dir(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def services(self) -> list[str]: ...

    @abstractmethod
    async def attach(self) -> None: ...

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots."
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore from a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level restore."
        )


class _DaytonaDirect(_DaytonaStrategy):
    """Direct sandbox strategy — single-container behavior."""

    # Daytona ships a native sandbox-snapshot API on AsyncSandbox; the direct
    # strategy uses it for the container layer of Branch (#384).
    supports_snapshot: bool = True

    async def start(self, force_build: bool) -> None:
        from benchflow.sandbox import daytona as _sdk

        env = self._env
        resources = _sdk.Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=env.task_env_config.storage_mb // 1024,
        )

        env._client_manager = await _sdk.DaytonaClientManager.get_instance()
        daytona = await env._client_manager.get_client()

        snapshot_name: str | None = None
        snapshot_exists = False

        if env._snapshot_template_name:
            snapshot_name = env._snapshot_template_name.format(
                name=env.environment_name
            )
            try:
                snapshot = await daytona.snapshot.get(snapshot_name)
                if snapshot.state == _sdk.SnapshotState.ACTIVE:
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
            env.logger.debug(f"Using snapshot: {snapshot_name}")
            params = _sdk.CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=snapshot_name,
                network_block_all=env._network_block_all,
                labels=_benchflow_owned_labels(),
            )
        else:
            # Both non-snapshot paths build identical CreateSandboxFromImageParams
            # (including the benchflow.managed ownership label); only the image
            # source and log line differ between build-from-Dockerfile and
            # prebuilt-image.
            if force_build or not env.task_env_config.docker_image:
                env.logger.debug(f"Building environment from {env._dockerfile_path}")
                image = _sdk.Image.from_dockerfile(env._dockerfile_path)
            else:
                env.logger.debug(
                    f"Using prebuilt image: {env.task_env_config.docker_image}"
                )
                image = _sdk.Image.base(env.task_env_config.docker_image)
            params = _sdk.CreateSandboxFromImageParams(
                image=image,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                resources=resources,
                network_block_all=env._network_block_all,
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

        await env._sandbox_exec(
            f"mkdir -p {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir} && "
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

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
        _reject_non_main_service(service)
        return await self._env._sandbox_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
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
        _reject_non_main_service(service)
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            cleanup_session=True,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        prep_result = await self._env._sandbox_exec(
            f"mkdir -p {shlex.quote(target_dir)}",
            timeout_sec=30,
            user="root",
        )
        if prep_result.return_code != 0:
            raise RuntimeError(
                "Daytona direct upload_dir destination prep failed: "
                f"{_exec_failure_output(prep_result)}"
            )
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return file_info.is_dir

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return not file_info.is_dir

    async def services(self) -> list[str]:
        raise NotImplementedError(
            "Direct (non-compose) Daytona sandbox is single-container and has "
            "no compose topology. services() requires a multi-container "
            "docker-compose task (#248)."
        )

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()
        os.execvp(
            "ssh",
            ["ssh", f"{ssh_access.token}@ssh.app.daytona.io"],
        )

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Create a Daytona snapshot from the current sandbox state.

        Wraps ``AsyncSandbox._experimental_create_snapshot``; the snapshot
        name is the ``ref`` other Daytona sandboxes can be created from via
        :class:`CreateSandboxFromSnapshotParams`.
        """
        env = self._env
        if not env._sandbox:
            raise SandboxSnapshotNotSupported(
                "DaytonaSandbox.snapshot requires a started sandbox; call "
                "start() before snapshot()."
            )
        snap_name = name or f"bf-snap-{env.environment_name}-{uuid4().hex[:12]}"
        # Daytona names: lowercase, dash-separated, ascii — sanitize defensively.
        snap_name = snap_name.lower().replace("_", "-")
        await env._sandbox._experimental_create_snapshot(snap_name)
        env.logger.info(f"Snapshot created: {snap_name}")
        return SandboxImage(
            provider="daytona",
            ref=snap_name,
            meta={"sandbox_id": getattr(env._sandbox, "id", "") or ""},
        )

    async def restore(self, image: SandboxImage) -> None:
        """Replace the current Daytona sandbox with one from ``image``.

        Daytona snapshots are immutable — restore is implemented by deleting
        the current sandbox and creating a fresh one from the snapshot,
        matching the provider's native semantics.
        """
        from benchflow.sandbox import daytona as _sdk

        env = self._env
        if image.provider != "daytona":
            raise SandboxSnapshotNotSupported(
                f"DaytonaSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )
        if env._sandbox is not None:
            try:
                await env._sandbox.delete()
            except Exception as e:
                env.logger.warning(f"Failed to delete sandbox before restore: {e}")
            env._sandbox = None

        params = _sdk.CreateSandboxFromSnapshotParams(
            auto_delete_interval=env._auto_delete_interval,
            auto_stop_interval=env._auto_stop_interval,
            snapshot=image.ref,
            network_block_all=env._network_block_all,
            labels=_benchflow_owned_labels(),
        )
        await env._create_sandbox(params=params)
        if env._network_block_all:
            verify = getattr(env, "_verify_network_enforcement", None)
            if verify is not None:
                await verify()
        else:
            await env.relock_network(
                extra_allowed_hosts=tuple(getattr(env, "_extra_allowed_hosts", ()))
            )
        env.logger.info(f"Snapshot restored: {image.ref}")
