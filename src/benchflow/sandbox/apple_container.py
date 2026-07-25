"""Apple Container sandbox backend using macOS Virtualization.framework.

Uses the system `container` CLI to run tasks in lightweight micro-VMs.
Requires macOS with Apple Silicon and the container system service running.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import TextIO

from benchflow.sandbox._base import BaseSandbox, ExecResult, wrap_command_with_env_file

_KALLOC_THRESHOLD = 8_000_000
_KALLOC_MIN_HEADROOM = 200_000
_DISK_MIN_GB = 5.0
_STARTUP_TIMEOUT = 30
_STOP_TIMEOUT = 30
_AMD64_PATTERN = re.compile(r"amd64|x86_64|--platform=linux/amd64", re.IGNORECASE)


def _kalloc_headroom() -> tuple[int, int]:
    """Return (current_elements, headroom) for data.kalloc.1024 zone."""
    try:
        out = subprocess.run(
            ["zprint"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.startswith("data.kalloc.1024"):
                parts = line.split()
                if len(parts) >= 7:
                    elts = int(parts[4])
                    return elts, _KALLOC_THRESHOLD - elts
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        pass
    return -1, -1


def _disk_free_gb() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


def _dockerfile_is_arm64_clean(path: Path) -> bool:
    """Reject Dockerfiles with amd64/x86_64 assumptions."""
    try:
        content = path.read_text()
        return not _AMD64_PATTERN.search(content)
    except OSError:
        return True


async def _run_cli(
    *args: str,
    timeout: float | None = None,
    stdin_data: bytes | None = None,
) -> ExecResult:
    """Run a `container` CLI command asynchronously."""
    cmd = ["container", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
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
        return_code=proc.returncode or 0,
    )


class AppleContainerSandbox(BaseSandbox):
    """Sandbox backend using Apple's `container` CLI (Virtualization.framework micro-VMs).

    Runs each task in an isolated arm64 Linux micro-VM. The VM stays alive
    via `sleep infinity` and commands are executed through `container exec`.
    """

    _container_name: str | None = None
    _bg_proc: asyncio.subprocess.Process | None = None
    _watcher_task: asyncio.Task | None = None
    _log_file: TextIO | None = None

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def sandbox_id(self) -> str | None:
        return self._container_name

    @classmethod
    def preflight(cls) -> None:
        if sys.platform != "darwin":
            raise RuntimeError(
                "apple-container sandbox requires macOS (Virtualization.framework)"
            )
        if not shutil.which("container"):
            raise RuntimeError(
                "container CLI not found. Install Apple Container: "
                "https://developer.apple.com/documentation/virtualization"
            )
        result = subprocess.run(
            ["container", "ls"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"container system not running. Start it with: container system start\n"
                f"Error: {result.stderr.strip()}"
            )
        _, headroom = _kalloc_headroom()
        if headroom >= 0 and headroom < _KALLOC_MIN_HEADROOM:
            raise RuntimeError(
                f"kalloc zone nearly exhausted (headroom={headroom}). "
                "Reboot your Mac to reclaim kernel memory before running containers."
            )
        if _disk_free_gb() < _DISK_MIN_GB:
            raise RuntimeError(
                f"Insufficient disk space ({_disk_free_gb():.1f}GB free, "
                f"need >{_DISK_MIN_GB}GB for container images and VM storage."
            )

    def _validate_definition(self) -> None:
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists() and not self.task_env_config.docker_image:
            raise ValueError(
                f"No Dockerfile found in {self.environment_dir} and no "
                "docker_image specified in task config."
            )
        if dockerfile.exists() and not _dockerfile_is_arm64_clean(dockerfile):
            raise ValueError(
                f"Dockerfile at {dockerfile} contains amd64/x86_64 references. "
                "Apple Container only supports arm64 images. Remove --platform "
                "flags and architecture-specific references."
            )
        if not self.task_env_config.allow_internet:
            raise ValueError(
                "apple-container does not currently enforce no-network sandboxing. "
                "Use docker, daytona, or modal for tasks that require "
                "environment.network_mode='no-network'."
            )

    def _image_tag(self) -> str:
        return f"bf__{self.environment_name}".replace("/", "_").replace(":", "_")

    async def _build_image(self) -> str:
        tag = self._image_tag()
        dockerfile = self.environment_dir / "Dockerfile"

        if self.task_env_config.docker_image and not dockerfile.exists():
            return self.task_env_config.docker_image

        build_timeout = self.task_env_config.build_timeout_sec
        result = await _run_cli(
            "build",
            "--no-cache",
            "-f",
            str(dockerfile),
            "-t",
            tag,
            str(self.environment_dir),
            timeout=build_timeout,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"container build failed (exit {result.return_code}):\n"
                f"{result.stderr or result.stdout}"
            )
        # BuildKit VM is spawned by build; stop it to free resources
        await _run_cli("stop", "buildkit", timeout=10)
        return tag

    async def start(self, force_build: bool) -> None:
        _, headroom = _kalloc_headroom()
        if headroom >= 0 and headroom < _KALLOC_MIN_HEADROOM:
            raise RuntimeError(
                f"kalloc headroom too low ({headroom}). Reboot required."
            )

        image = await self._build_image()
        self._container_name = f"bf_{self.session_id}".replace("/", "_")[:63]

        cpus = self.task_env_config.cpus or 2
        memory_mb = self.task_env_config.memory_mb or 2048

        cmd_args = [
            "run",
            "--name",
            self._container_name,
            "-c",
            str(cpus),
            "-m",
            f"{memory_mb}M",
        ]

        # Environment variables
        env = self._merge_env(None)
        if env:
            for k, v in env.items():
                cmd_args.extend(["-e", f"{k}={v}"])

        # Bind mount: rollout_dir as /logs so subdirectories (verifier/, agent/,
        # artifacts/) are regular dirs inside the mount — chmod works on them
        # unlike the mount point itself. Output lands where benchflow expects.
        if self.rollout_paths:
            logs_dir = self.rollout_paths.rollout_dir
            for sub in ("verifier", "agent", "artifacts"):
                os.makedirs(logs_dir / sub, exist_ok=True)
                os.chmod(logs_dir / sub, 0o777)
            cmd_args.extend(["--mount", f"type=bind,source={logs_dir},target=/logs"])

        # Skills directory mount
        skills_dir = self.task_env_config.skills_dir
        if skills_dir and Path(skills_dir).exists():
            cmd_args.extend(
                ["--mount", f"type=bind,source={skills_dir},target=/skills"]
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

        # Launch backgrounded VM — stdout/stderr to log file, NOT PIPE
        log_path = Path(f"/tmp/{self._container_name}.log")
        self._log_file = open(log_path, "w")  # noqa: SIM115 — closed in stop()
        self._bg_proc = await asyncio.create_subprocess_exec(
            "container",
            *cmd_args,
            stdout=self._log_file,
            stderr=self._log_file,
        )

        # Readiness poll
        deadline = asyncio.get_event_loop().time() + _STARTUP_TIMEOUT
        ready = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                result = await _run_cli("exec", self._container_name, "true", timeout=5)
                if result.return_code == 0:
                    ready = True
                    break
            except (TimeoutError, OSError):
                pass
            await asyncio.sleep(0.5)

        if not ready:
            await self._force_cleanup()
            raise RuntimeError(
                f"Container {self._container_name} did not become ready "
                f"within {_STARTUP_TIMEOUT}s. Check /tmp/{self._container_name}.log"
            )

        self._watcher_task = asyncio.create_task(self._watch_process())
        self.logger.info("Started container %s (image=%s)", self._container_name, image)

    async def _watch_process(self) -> None:
        """Detect early VM exit (e.g., kalloc crash = exit 128)."""
        if self._bg_proc is None:
            return
        returncode = await self._bg_proc.wait()
        if returncode != 0 and self._container_name:
            self.logger.error(
                "Container %s exited early with code %d (128 = kalloc crash). "
                "Reboot may be required.",
                self._container_name,
                returncode,
            )

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
        if cwd:
            wrapped = f"cd {shlex.quote(cwd)} && {wrapped}"

        merged_env = self._merge_env(env)
        if merged_env:
            wrapped = wrap_command_with_env_file(
                merged_env, wrapped, env_path_prefix="/tmp/.bf_env_"
            )

        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            wrapped = f"su {resolved_user} -s /bin/sh -c {shlex.quote(wrapped)}"

        args = ["exec", self._container_name, "sh", "-c", wrapped]
        try:
            result = await _run_cli(*args, timeout=timeout_sec)
        except TimeoutError:
            self.logger.error(
                "exec timed out after %ss, stopping container", timeout_sec
            )
            await self._force_cleanup()
            raise RuntimeError(
                f"Command timed out after {timeout_sec}s: {command[:100]}"
            ) from None
        return result

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source_path = Path(source_path)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        # Check if target is under a mounted path (zero-copy via virtiofs)
        host_path = self._mounted_host_path(target_path)
        if host_path is not None:
            os.makedirs(host_path.parent, exist_ok=True)
            shutil.copy2(source_path, host_path)
            return

        # Fallback: base64 through exec -i (stdin)
        data = source_path.read_bytes()
        encoded = base64.b64encode(data).decode()
        target_dir = os.path.dirname(target_path) or "/"
        cmd = f"mkdir -p {shlex.quote(target_dir)} && base64 -d > {shlex.quote(target_path)}"
        result = await _run_cli(
            "exec",
            "-i",
            self._container_name,
            "sh",
            "-c",
            cmd,
            stdin_data=encoded.encode(),
            timeout=60,
        )
        if result.return_code != 0:
            raise RuntimeError(f"upload_file failed: {result.stderr or result.stdout}")

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                "apple-container is single-container; service must be 'main'."
            )
        source_dir = Path(source_dir)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        # Check if target is under a mounted path
        host_path = self._mounted_host_path(target_dir)
        if host_path is not None:
            if host_path.exists():
                shutil.rmtree(host_path)
            shutil.copytree(source_dir, host_path)
            return

        # Walk and upload file-by-file
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", timeout_sec=10)
        sem = asyncio.Semaphore(4)

        async def _upload_one(src: Path, rel: str) -> None:
            async with sem:
                dst = f"{target_dir}/{rel}"
                await self.upload_file(src, dst)

        tasks = []
        for root, _dirs, files in os.walk(source_dir):
            for fname in files:
                src = Path(root) / fname
                if src.is_symlink():
                    continue
                rel = str(src.relative_to(source_dir))
                tasks.append(_upload_one(src, rel))
        await asyncio.gather(*tasks)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target_path = Path(target_path)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        # Check if source is under a mounted path
        host_path = self._mounted_host_path(source_path)
        if host_path is not None:
            os.makedirs(target_path.parent, exist_ok=True)
            shutil.copy2(host_path, target_path)
            return

        # Fallback: base64 through exec stdout
        result = await _run_cli(
            "exec",
            self._container_name,
            "base64",
            source_path,
            timeout=60,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"download_file failed: {result.stderr or result.stdout}"
            )
        os.makedirs(target_path.parent, exist_ok=True)
        target_path.write_bytes(base64.b64decode(result.stdout or ""))

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                "apple-container is single-container; service must be 'main'."
            )
        target_dir = Path(target_dir)
        if not self._container_name:
            raise RuntimeError("Container not started.")

        # Check if source is under a mounted path
        host_path = self._mounted_host_path(source_dir)
        if host_path is not None:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(host_path, target_dir)
            return

        # List files and download concurrently
        result = await self.exec(
            f"find {shlex.quote(source_dir)} -type f", timeout_sec=30
        )
        if result.return_code != 0 or not result.stdout:
            return

        files = [f for f in result.stdout.strip().splitlines() if f]
        sem = asyncio.Semaphore(4)

        async def _download_one(container_path: str) -> None:
            async with sem:
                rel = container_path.removeprefix(source_dir).lstrip("/")
                local = target_dir / rel
                await self.download_file(container_path, local)

        await asyncio.gather(*[_download_one(f) for f in files])

    async def stop(self, delete: bool) -> None:
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            self._watcher_task = None

        if self._container_name:
            await _run_cli("stop", self._container_name, timeout=_STOP_TIMEOUT)
            if delete:
                await _run_cli("rm", self._container_name, timeout=10)
            self.logger.info("Stopped container %s", self._container_name)

        if self._bg_proc and self._bg_proc.returncode is None:
            self._bg_proc.terminate()
            try:
                await asyncio.wait_for(self._bg_proc.wait(), timeout=5)
            except TimeoutError:
                self._bg_proc.kill()
                await self._bg_proc.wait()
        self._bg_proc = None

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        # Best-effort BuildKit cleanup
        await _run_cli("stop", "buildkit", timeout=5)

        if delete:
            self._container_name = None

    async def _force_cleanup(self) -> None:
        """Emergency cleanup on timeout or crash."""
        if self._container_name:
            await _run_cli("stop", self._container_name, timeout=10)
            await _run_cli("rm", self._container_name, timeout=10)
        if self._bg_proc and self._bg_proc.returncode is None:
            self._bg_proc.kill()
            await self._bg_proc.wait()

    def _mounted_host_path(self, container_path: str) -> Path | None:
        """Map a container path to host path if it's under a bind mount."""
        mount_map: dict[str, Path] = {}
        if self.rollout_paths:
            mount_map["/logs"] = self.rollout_paths.rollout_dir
        path = PurePosixPath(container_path)
        if not path.is_absolute():
            return None
        for prefix, host_base in mount_map.items():
            prefix_path = PurePosixPath(prefix)
            prefix_parts = prefix_path.parts
            if path == prefix_path:
                rel_parts: tuple[str, ...] = ()
            elif path.parts[: len(prefix_parts)] == prefix_parts:
                rel_parts = path.parts[len(prefix_parts) :]
            else:
                continue
            if any(part == ".." for part in rel_parts):
                raise ValueError(
                    f"Unsafe mounted path escapes {prefix}: {container_path!r}"
                )
            host_root = host_base.resolve()
            candidate = host_root.joinpath(*rel_parts).resolve(strict=False)
            try:
                candidate.relative_to(host_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe mounted path escapes {prefix}: {container_path!r}"
                ) from exc
            return candidate
        return None
