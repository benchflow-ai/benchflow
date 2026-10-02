"""Local Apptainer sandbox backend."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from benchflow.sandbox._base import BaseSandbox, ExecResult, wrap_command_with_env_file
from benchflow.sandbox.apptainer_image import (
    ApptainerImageBuilder,
    require_apptainer,
)
from benchflow.task.config import NetworkMode
from benchflow.task.paths import SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

_STARTUP_TIMEOUT = 60
_STOP_TIMEOUT = 30
_STAGING_DIR = PurePosixPath("/benchflow/staging")


def _safe_instance_name(environment_name: str, session_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"bf-{environment_name}-{session_id}")
    return f"{value[:96].strip('-')}-{uuid.uuid4().hex[:8]}"


async def _run_apptainer(*args: str, timeout_sec: float | None = None) -> ExecResult:
    process = await asyncio.create_subprocess_exec(
        "apptainer",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_sec
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return ExecResult(
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        return_code=process.returncode or 0,
    )


def _failure(result: ExecResult) -> str:
    return (result.stderr or result.stdout or "").strip()[-2000:]


class ApptainerSandbox(BaseSandbox):
    """A writable per-rollout Apptainer instance backed by a cached SIF."""

    def __init__(
        self, *args, image_builder: ApptainerImageBuilder | None = None, **kwargs
    ):
        self._instance_name: str | None = None
        self._runtime_dir: Path | None = None
        self._staging_dir: Path | None = None
        self._overlay_path: Path | None = None
        self._default_cwd: str | None = None
        self._image_builder = image_builder
        super().__init__(*args, **kwargs)

    def _validate_definition(self) -> None:
        image = self.task_env_config.docker_image
        dockerfile = self.environment_dir / "Dockerfile"
        if image:
            path = Path(image).expanduser()
            if path.suffix != ".sif":
                raise ValueError(
                    "Apptainer sandbox image must point to a local .sif file"
                )
        elif not dockerfile.is_file():
            raise FileNotFoundError(
                f"Apptainer requires {dockerfile} or sandbox.docker_image=<file.sif>"
            )

    @classmethod
    def preflight(cls) -> None:
        executable = require_apptainer()
        result = subprocess.run(
            [executable, "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Apptainer preflight failed: {result.stderr.strip()}")

    @property
    def sandbox_id(self) -> str | None:
        return self._instance_name

    @property
    def is_mounted(self) -> bool:
        return True

    def _require_started(self) -> tuple[str, Path]:
        if not self._instance_name or not self._staging_dir:
            raise RuntimeError("Apptainer sandbox not started")
        return self._instance_name, self._staging_dir

    def _mounted_host_path(self, sandbox_path: str) -> Path | None:
        if self.rollout_paths is None:
            return None
        path = PurePosixPath(sandbox_path)
        try:
            relative = path.relative_to(SandboxPaths().logs_dir)
        except ValueError:
            return None
        if ".." in relative.parts:
            return None
        return self.rollout_paths.rollout_dir.joinpath(*relative.parts)

    async def start(self, force_build: bool) -> None:
        if self._instance_name:
            return
        if self.rollout_paths is None:
            raise RuntimeError("Apptainer sandbox requires rollout paths")
        builder = self._image_builder or ApptainerImageBuilder(
            timeout_sec=self.task_env_config.build_timeout_sec
        )
        image = await builder.resolve(
            dockerfile=self.environment_dir / "Dockerfile",
            context_dir=self.environment_dir,
            prebuilt=self.task_env_config.docker_image,
            force_build=force_build,
        )
        runtime_dir = Path(tempfile.mkdtemp(prefix="benchflow-apptainer-"))
        staging_dir = runtime_dir / "staging"
        overlay_path = runtime_dir / "overlay.img"
        # Register partial state before fallible setup so cleanup can reclaim it.
        self._runtime_dir = runtime_dir
        self._staging_dir = staging_dir
        self._overlay_path = overlay_path
        try:
            staging_dir.mkdir()
            self.rollout_paths.mkdir()
            os.chmod(self.rollout_paths.rollout_dir, 0o777)
            for directory in (
                self.rollout_paths.agent_dir,
                self.rollout_paths.verifier_dir,
                self.rollout_paths.artifacts_dir,
            ):
                os.chmod(directory, 0o777)

            instance_name = _safe_instance_name(self.environment_name, self.session_id)
            self._default_cwd = self.task_env_config.workdir or image.workdir

            overlay = await _run_apptainer(
                "overlay",
                "create",
                "--fakeroot",
                "--size",
                str(self.task_env_config.storage_mb),
                "--sparse",
                str(overlay_path),
                timeout_sec=self.task_env_config.build_timeout_sec,
            )
            if overlay.return_code != 0:
                raise RuntimeError(
                    f"Could not create Apptainer overlay: {_failure(overlay)}"
                )

            args = [
                "instance",
                "start",
                "--fakeroot",
                "--overlay",
                str(overlay_path),
                "--containall",
                "--no-home",
                "--cleanenv",
                "--bind",
                f"{staging_dir}:{_STAGING_DIR}",
                "--bind",
                f"{self.rollout_paths.rollout_dir}:{SandboxPaths().logs_dir}",
            ]
            if self.task_env_config.network_mode is NetworkMode.NO_NETWORK:
                args.extend(["--net", "--network", "none"])
            args.extend([str(image.path), instance_name])

            # A timed-out start may still leave an instance under this name.
            self._instance_name = instance_name
            started = await _run_apptainer(*args, timeout_sec=_STARTUP_TIMEOUT)
            if started.return_code != 0:
                raise RuntimeError(
                    f"Could not start Apptainer instance: {_failure(started)}"
                )
            probe = await self.exec("true", timeout_sec=30, user="root")
            if probe.return_code != 0:
                raise RuntimeError(
                    f"Apptainer instance readiness check failed: {_failure(probe)}"
                )
        except BaseException:
            await self._force_cleanup()
            raise

    async def _forced_stop(self, name: str) -> str | None:
        """Force-stop *name*, returning a failure detail if it remains live."""
        try:
            forced = await _run_apptainer(
                "instance", "stop", "--force", name, timeout_sec=_STOP_TIMEOUT
            )
        except TimeoutError:
            detail = f"forced stop timed out after {_STOP_TIMEOUT}s"
        except Exception as error:
            detail = str(error)[:200]
        else:
            if forced.return_code == 0:
                return None
            detail = _failure(forced)

        try:
            listed = await _run_apptainer(
                "instance", "list", "--json", name, timeout_sec=_STOP_TIMEOUT
            )
            instances = json.loads(listed.stdout or "").get("instances")
            if listed.return_code == 0 and instances == []:
                return None
        except (AttributeError, TypeError, ValueError, OSError, TimeoutError):
            pass
        return detail

    async def _force_cleanup(self) -> None:
        """Tear down after a failed start or timed-out command."""
        name = self._instance_name
        if name:
            detail = await self._forced_stop(name)
            if detail is not None:
                self.logger.warning(
                    "Could not force-stop Apptainer instance %s (%s); keeping "
                    "%s for retry during teardown.",
                    name,
                    detail,
                    self._runtime_dir,
                )
                return
        if self._runtime_dir:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
        self._instance_name = None
        self._runtime_dir = None
        self._staging_dir = None
        self._overlay_path = None

    async def stop(self, delete: bool) -> None:
        name = self._instance_name
        if name:
            try:
                result = await _run_apptainer(
                    "instance", "stop", name, timeout_sec=_STOP_TIMEOUT
                )
                detail = None if result.return_code == 0 else _failure(result)
            except TimeoutError:
                detail = f"graceful stop timed out after {_STOP_TIMEOUT}s"
            if detail is not None:
                forced = await self._forced_stop(name)
                if forced is not None:
                    raise RuntimeError(
                        f"Could not stop Apptainer instance {name}: {forced} "
                        f"(graceful stop: {detail})"
                    )
        self._instance_name = None
        if delete and self._runtime_dir:
            shutil.rmtree(self._runtime_dir)
            self._runtime_dir = None
            self._staging_dir = None
            self._overlay_path = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        if service != "main":
            raise ValueError("apptainer is single-container; service must be 'main'")
        name, _ = self._require_started()
        merged = self._merge_env(env)
        wrapped = (
            wrap_command_with_env_file(
                merged,
                command,
                env_path_prefix="/tmp/.benchflow_exec_env_",
            )
            if merged
            else command
        )
        resolved_user = self._resolve_user(user)
        if resolved_user not in {None, "root", 0, "0"}:
            wrapped = (
                f"exec su -s /bin/sh {shlex.quote(str(resolved_user))} "
                f"-c {shlex.quote(wrapped)}"
            )
        args = ["exec", "--cleanenv"]
        effective_cwd = cwd or self._default_cwd
        if effective_cwd:
            args.extend(["--cwd", effective_cwd])
        args.extend([f"instance://{name}", "sh", "-c", wrapped])
        try:
            return await _run_apptainer(*args, timeout_sec=timeout_sec)
        except TimeoutError:
            await self._force_cleanup()
            raise RuntimeError(
                f"Command timed out after {timeout_sec}s: {command[:100]}"
            ) from None

    def _staging_path(self) -> tuple[Path, PurePosixPath]:
        _, staging = self._require_started()
        name = uuid.uuid4().hex
        return staging / name, _STAGING_DIR / name

    async def upload_file(
        self,
        source_path: Path | str,
        target_path: str,
        *,
        mode: str | None = None,
    ) -> None:
        source = Path(source_path)
        host_target = self._mounted_host_path(target_path)
        if host_target is not None:
            host_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, host_target)
            if mode is not None:
                host_target.chmod(int(mode, 8))
            return
        host_stage, sandbox_stage = self._staging_path()
        shutil.copy2(source, host_stage)
        try:
            target = PurePosixPath(target_path)
            result = await self.exec(
                f"mkdir -p {shlex.quote(str(target.parent))} && "
                f"cp -p {shlex.quote(str(sandbox_stage))} {shlex.quote(str(target))}",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(f"upload_file failed: {_failure(result)}")
            await self._apply_upload_mode(target_path, mode)
        finally:
            host_stage.unlink(missing_ok=True)

    async def upload_dir(
        self,
        source_dir: Path | str,
        target_dir: str,
        service: str = "main",
    ) -> None:
        if service != "main":
            raise ValueError("apptainer is single-container; service must be 'main'")
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory {source} does not exist")
        host_target = self._mounted_host_path(target_dir)
        if host_target is not None:
            if host_target.exists():
                shutil.rmtree(host_target)
            shutil.copytree(source, host_target)
            return
        host_stage, sandbox_stage = self._staging_path()
        shutil.copytree(source, host_stage)
        try:
            result = await self.exec(
                f"mkdir -p {shlex.quote(target_dir)} && "
                f"cp -a {shlex.quote(str(sandbox_stage))}/. {shlex.quote(target_dir)}/",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(f"upload_dir failed: {_failure(result)}")
        finally:
            shutil.rmtree(host_stage, ignore_errors=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        host_source = self._mounted_host_path(source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if host_source is not None:
            shutil.copy2(host_source, target)
            return
        host_stage, sandbox_stage = self._staging_path()
        try:
            result = await self.exec(
                f"cp -p {shlex.quote(source_path)} {shlex.quote(str(sandbox_stage))}",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(f"download_file failed: {_failure(result)}")
            shutil.copy2(host_stage, target)
        finally:
            host_stage.unlink(missing_ok=True)

    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        service: str = "main",
    ) -> None:
        if service != "main":
            raise ValueError("apptainer is single-container; service must be 'main'")
        target = Path(target_dir)
        host_source = self._mounted_host_path(source_dir)
        if target.exists():
            shutil.rmtree(target)
        if host_source is not None:
            shutil.copytree(host_source, target)
            return
        host_stage, sandbox_stage = self._staging_path()
        host_stage.mkdir()
        try:
            result = await self.exec(
                f"cp -a {shlex.quote(source_dir)}/. {shlex.quote(str(sandbox_stage))}/",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(f"download_dir failed: {_failure(result)}")
            shutil.copytree(host_stage, target)
        finally:
            shutil.rmtree(host_stage, ignore_errors=True)

    async def attach(self) -> None:
        name, _ = self._require_started()
        process = await asyncio.create_subprocess_exec(
            "apptainer", "shell", f"instance://{name}"
        )
        await process.wait()

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        from benchflow.sandbox.process import ApptainerProcess

        return ApptainerProcess.from_sandbox_env(self)
