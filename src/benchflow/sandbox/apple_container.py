"""Apple Container sandbox backend using macOS Virtualization.framework.

The backend targets Apple Container 1.1+ on Apple silicon. It delegates process
placement, detached lifecycle, and file transfer to the native CLI instead of
reimplementing those primitives with host subprocess state or shell pipelines.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from benchflow.sandbox._base import BaseSandbox, ExecResult, wrap_command_with_env_file

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

_MIN_CONTAINER_VERSION = (1, 1, 0)
_KALLOC_SAFE_LIMIT = 3_000_000
_KALLOC_MIN_HEADROOM = 200_000
_DISK_MIN_GB = 5.0
_STARTUP_TIMEOUT = 30
_STOP_TIMEOUT = 30

# Apple Container currently leaks data.kalloc.1024 allocations across VM
# lifecycles. One active sandbox per BenchFlow process keeps the headroom check
# and launch atomic and matches the provider's documented safety envelope.
_RUN_SLOT = asyncio.Semaphore(1)


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", value)
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )


def _container_cli_version() -> tuple[int, int, int] | None:
    """Read the Apple Container CLI version from its machine-readable output."""

    try:
        result = subprocess.run(
            ["container", "system", "version", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return None

    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if not isinstance(row, dict):
            continue
        app_name = str(row.get("appName", "")).lower()
        if app_name in {"container", "container cli"}:
            return _parse_version(str(row.get("version", "")))
    return None


def _kalloc_headroom() -> tuple[int, int]:
    """Return current in-use elements and safe headroom for data.kalloc.1024.

    macOS 26 ``zprint`` emits this row shape when headings are disabled:

    ``name elem_size cur_size max_size cur_elts max_elts cur_inuse ...``

    ``cur_inuse`` is the live allocation count relevant to the Apple Container
    VM leak. Keeping the real schema in this parser avoids the fictional
    ``elems/maxelts`` fixture that previously selected the wrong column.
    """

    try:
        result = subprocess.run(
            ["zprint", "-H", "-L", "data.kalloc.1024"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return -1, -1
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "data.kalloc.1024" and len(parts) >= 7:
                current_inuse = int(parts[6])
                return current_inuse, _KALLOC_SAFE_LIMIT - current_inuse
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return -1, -1


def _require_kalloc_headroom() -> None:
    current, headroom = _kalloc_headroom()
    if current < 0:
        raise RuntimeError(
            "Unable to read data.kalloc.1024 usage from zprint. "
            "Apple Container launch is blocked because VM headroom cannot be verified."
        )
    if headroom < _KALLOC_MIN_HEADROOM:
        raise RuntimeError(
            "data.kalloc.1024 is near the safe Apple Container limit "
            f"(in_use={current}, headroom={headroom}). Reboot your Mac before "
            "starting another sandbox."
        )


def _disk_free_gb() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


async def _run_cli(
    *args: str,
    timeout: float | None = None,
    stdin_data: bytes | None = None,
) -> ExecResult:
    """Run an Apple ``container`` CLI command asynchronously."""

    proc = await asyncio.create_subprocess_exec(
        "container",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_data), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return ExecResult(
        stdout=stdout.decode(errors="replace") if stdout else None,
        stderr=stderr.decode(errors="replace") if stderr else None,
        return_code=proc.returncode if proc.returncode is not None else 1,
    )


def _failure_output(result: ExecResult) -> str:
    return (result.stderr or result.stdout or "no command output").strip()


class AppleContainerSandbox(BaseSandbox):
    """Sandbox backend for Apple Container 1.1+ micro-VMs."""

    _container_name: str | None = None
    _holds_run_slot: bool = False

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def sandbox_id(self) -> str | None:
        return self._container_name

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        from benchflow.sandbox.process import AppleContainerProcess

        return AppleContainerProcess.from_sandbox_env(self)

    @classmethod
    def preflight(cls) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "apple-container sandbox requires macOS (Virtualization.framework)"
            )
        if platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("apple-container sandbox requires Apple silicon")
        if not shutil.which("container"):
            raise RuntimeError(
                "container CLI not found. Install Apple Container 1.1 or newer: "
                "https://github.com/apple/container/releases"
            )

        version = _container_cli_version()
        if version is None:
            raise RuntimeError(
                "Unable to determine Apple Container CLI version with "
                "`container system version --format json`."
            )
        if version < _MIN_CONTAINER_VERSION:
            actual = ".".join(str(part) for part in version)
            required = ".".join(str(part) for part in _MIN_CONTAINER_VERSION)
            raise RuntimeError(
                f"Apple Container {required}+ is required; found {actual}. "
                "Upgrade with `/usr/local/bin/update-container.sh`."
            )

        try:
            result = subprocess.run(
                ["container", "ls"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Unable to query the Apple Container service: {exc}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "container system not running. Start it with: container system start\n"
                f"Error: {result.stderr.strip()}"
            )

        _require_kalloc_headroom()
        free_gb = _disk_free_gb()
        if free_gb < _DISK_MIN_GB:
            raise RuntimeError(
                f"Insufficient disk space ({free_gb:.1f}GB free, "
                f"need >{_DISK_MIN_GB}GB for container images and VM storage."
            )

    def _validate_definition(self) -> None:
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists() and not self.task_env_config.docker_image:
            raise ValueError(
                f"No Dockerfile found in {self.environment_dir} and no "
                "docker_image specified in task config."
            )
        if not self.task_env_config.allow_internet:
            raise ValueError(
                "apple-container does not currently enforce no-network sandboxing. "
                "Use docker, daytona, or modal for tasks that require "
                "environment.network_mode='no-network'."
            )

    def _image_tag(self) -> str:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.environment_name)
        return f"bf__{safe_name}"

    async def _resolve_image(self, *, force_build: bool) -> str:
        dockerfile = self.environment_dir / "Dockerfile"
        if self.task_env_config.docker_image and not force_build:
            return self.task_env_config.docker_image
        if not dockerfile.exists():
            raise ValueError(
                f"No Dockerfile found in {self.environment_dir}; "
                "cannot force-build apple-container image."
            )

        args = ["build"]
        if force_build:
            args.append("--no-cache")
        args.extend(
            [
                "--platform",
                "linux/arm64",
                "-f",
                str(dockerfile),
                "-t",
                self._image_tag(),
                str(self.environment_dir),
            ]
        )
        result = await _run_cli(
            *args,
            timeout=self.task_env_config.build_timeout_sec,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"container build failed (exit {result.return_code}):\n"
                f"{_failure_output(result)}"
            )
        return self._image_tag()

    def _release_run_slot(self) -> None:
        if self._holds_run_slot:
            self._holds_run_slot = False
            _RUN_SLOT.release()

    async def start(self, force_build: bool) -> None:
        if self._holds_run_slot or self._container_name is not None:
            raise RuntimeError("Apple Container sandbox is already started.")

        await _RUN_SLOT.acquire()
        self._holds_run_slot = True
        try:
            await asyncio.to_thread(_require_kalloc_headroom)
            image = await self._resolve_image(force_build=force_build)
            safe_session = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.session_id)
            self._container_name = f"bf_{safe_session}"[:63]

            cmd_args = [
                "run",
                "--detach",
                "--name",
                self._container_name,
                "--platform",
                "linux/arm64",
                "-c",
                str(self.task_env_config.cpus or 2),
                "-m",
                f"{self.task_env_config.memory_mb or 2048}M",
            ]

            if self.rollout_paths:
                logs_dir = self.rollout_paths.rollout_dir
                for sub in ("verifier", "agent", "artifacts"):
                    os.makedirs(logs_dir / sub, exist_ok=True)
                    os.chmod(logs_dir / sub, 0o777)
                cmd_args.extend(
                    ["--mount", f"type=bind,source={logs_dir},target=/logs"]
                )

            cmd_args.extend(
                [
                    "--entrypoint",
                    "/bin/sh",
                    image,
                    "-c",
                    "sleep infinity || while :; do sleep 3600; done",
                ]
            )
            result = await _run_cli(
                *cmd_args,
                timeout=self.task_env_config.build_timeout_sec,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"container run failed (exit {result.return_code}):\n"
                    f"{_failure_output(result)}"
                )

            deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                try:
                    ready = await _run_cli(
                        "exec", self._container_name, "true", timeout=5
                    )
                    if ready.return_code == 0:
                        self.logger.info(
                            "Started container %s (image=%s)",
                            self._container_name,
                            image,
                        )
                        return
                except (TimeoutError, OSError):
                    pass
                await asyncio.sleep(0.5)

            logs = await _run_cli("logs", self._container_name, timeout=10)
            raise RuntimeError(
                f"Container {self._container_name} did not become ready within "
                f"{_STARTUP_TIMEOUT}s: {_failure_output(logs)}"
            )
        except BaseException:
            await self._force_cleanup()
            raise

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
            raise ValueError(
                "apple-container is a single-container backend; "
                f"service={service!r} is not supported."
            )
        if not self._container_name:
            raise RuntimeError("Container not started. Call start() first.")

        wrapped = command
        merged_env = self._merge_env(env)
        if merged_env:
            wrapped = wrap_command_with_env_file(
                merged_env, wrapped, env_path_prefix="/tmp/.bf_env_"
            )

        args = ["exec"]
        if cwd:
            args.extend(["--workdir", cwd])
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            args.extend(["--user", str(resolved_user)])
        args.extend([self._container_name, "sh", "-c", wrapped])

        try:
            return await _run_cli(*args, timeout=timeout_sec)
        except TimeoutError:
            self.logger.error(
                "exec timed out after %ss, removing container", timeout_sec
            )
            await self._force_cleanup()
            raise RuntimeError(
                f"Command timed out after {timeout_sec}s: {command[:100]}"
            ) from None

    async def _copy(
        self, source: str, destination: str, *, operation: str, timeout: float = 120
    ) -> None:
        result = await _run_cli("cp", source, destination, timeout=timeout)
        if result.return_code != 0:
            raise RuntimeError(f"{operation} failed: {_failure_output(result)}")

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        source = Path(source_path)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        host_path = self._mounted_host_path(target_path)
        if host_path is not None:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, host_path)
            if mode is not None:
                host_path.chmod(int(mode, 8))
            return

        target_parent = str(PurePosixPath(target_path).parent)
        prep = await self.exec(
            f"mkdir -p {shlex.quote(target_parent)}", timeout_sec=30, user="root"
        )
        if prep.return_code != 0:
            raise RuntimeError(
                f"upload_file destination prep failed: {_failure_output(prep)}"
            )
        await self._copy(
            str(source),
            f"{self._container_name}:{target_path}",
            operation="upload_file",
        )
        await self._apply_upload_mode(target_path, mode)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                "apple-container is single-container; service must be 'main'."
            )
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory {source} does not exist")
        if not self._container_name:
            raise RuntimeError("Container not started.")

        host_path = self._mounted_host_path(target_dir)
        if host_path is not None:
            if host_path.exists():
                shutil.rmtree(host_path)
            shutil.copytree(source, host_path)
            return

        prep = await self.exec(
            f"mkdir -p {shlex.quote(target_dir)}", timeout_sec=30, user="root"
        )
        if prep.return_code != 0:
            raise RuntimeError(
                f"upload_dir destination prep failed: {_failure_output(prep)}"
            )
        # Apple ``container cp source/ existing-target`` nests ``source`` under
        # the target. BenchFlow's upload_dir contract copies the directory's
        # contents, so copy each immediate child through the native primitive.
        # This includes dotfiles and preserves nested subtrees without a shell
        # archive/base64 transport.
        destination = f"{self._container_name}:{target_dir.rstrip('/')}/"
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            await self._copy(
                str(child),
                destination,
                operation=f"upload_dir ({child.name})",
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        host_path = self._mounted_host_path(source_path)
        if host_path is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(host_path, target)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        await self._copy(
            f"{self._container_name}:{source_path}",
            str(target),
            operation="download_file",
        )

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                "apple-container is single-container; service must be 'main'."
            )
        target = Path(target_dir)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        host_path = self._mounted_host_path(source_dir)
        if host_path is not None:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(host_path, target)
            return

        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._copy(
            f"{self._container_name}:{source_dir}",
            str(target),
            operation="download_dir",
        )

    async def stop(self, delete: bool) -> None:
        name = self._container_name
        try:
            if not name:
                return
            try:
                stopped = await _run_cli(
                    "stop", "--time", "5", name, timeout=_STOP_TIMEOUT
                )
                if stopped.return_code != 0:
                    self.logger.warning(
                        "Failed to stop Apple container %s cleanly: %s",
                        name,
                        _failure_output(stopped),
                    )
            except (TimeoutError, OSError) as exc:
                self.logger.warning(
                    "Apple container %s stop failed; continuing cleanup: %s",
                    name,
                    exc,
                )

            if delete:
                try:
                    removed = await _run_cli("rm", "--force", name, timeout=10)
                    if removed.return_code != 0:
                        self.logger.warning(
                            "Failed to remove Apple container %s: %s",
                            name,
                            _failure_output(removed),
                        )
                except (TimeoutError, OSError) as exc:
                    self.logger.warning(
                        "Apple container %s removal failed: %s", name, exc
                    )
                self._container_name = None
            self.logger.info("Stopped container %s", name)
        finally:
            self._release_run_slot()

    async def _force_cleanup(self) -> None:
        """Best-effort atomic cleanup after startup or execution failure."""

        name = self._container_name
        try:
            if name:
                try:
                    result = await _run_cli("rm", "--force", name, timeout=10)
                    if result.return_code != 0:
                        self.logger.warning(
                            "Forced Apple container cleanup failed for %s: %s",
                            name,
                            _failure_output(result),
                        )
                except (TimeoutError, OSError) as exc:
                    self.logger.warning(
                        "Forced Apple container cleanup failed for %s: %s", name, exc
                    )
        finally:
            self._container_name = None
            self._release_run_slot()

    def _mounted_host_path(self, container_path: str) -> Path | None:
        """Map a container path to a host path when it is under ``/logs``."""

        if not self.rollout_paths:
            return None
        path = PurePosixPath(container_path)
        prefix = PurePosixPath("/logs")
        if not path.is_absolute():
            return None
        if path == prefix:
            rel_parts: tuple[str, ...] = ()
        elif path.parts[: len(prefix.parts)] == prefix.parts:
            rel_parts = path.parts[len(prefix.parts) :]
        else:
            return None
        if any(part == ".." for part in rel_parts):
            raise ValueError(f"Unsafe mounted path escapes /logs: {container_path!r}")

        host_root = self.rollout_paths.rollout_dir.resolve()
        candidate = host_root.joinpath(*rel_parts).resolve(strict=False)
        try:
            candidate.relative_to(host_root)
        except ValueError as exc:
            raise ValueError(
                f"Unsafe mounted path escapes /logs: {container_path!r}"
            ) from exc
        return candidate
