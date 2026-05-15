"""Native Docker sandbox provider.

This module owns Docker execution directly instead of routing through Harbor.
It intentionally has a small surface: build/start/stop, command execution, and
file transfer. Rollout orchestration should depend on the Sandbox protocol, not
on Docker-specific details.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from benchflow.sandboxes.specs import ExecResult, SandboxSpec


@dataclass(frozen=True)
class _CompletedProcess:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Docker-specific sandbox startup config."""

    task_path: Path
    session_id: str = field(default_factory=lambda: f"bf-{uuid4().hex[:12]}")
    image_tag: str | None = None
    workdir: str = "/app"
    command: tuple[str, ...] = ("sleep", "infinity")
    force_build: bool = False

    @property
    def environment_dir(self) -> Path:
        return self.task_path / "environment"

    @property
    def dockerfile(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def resolved_image_tag(self) -> str:
        if self.image_tag:
            return self.image_tag
        safe_task = self.task_path.name.lower().replace("_", "-")
        return f"benchflow-{safe_task}:{self.session_id}"


class DockerSandbox:
    """Isolated Docker container sandbox."""

    def __init__(
        self,
        config: DockerSandboxConfig,
        spec: SandboxSpec | None = None,
    ) -> None:
        self.config = config
        self.spec = spec or SandboxSpec(provider="docker")
        self.container_name = f"benchflow-{config.session_id}"
        self._started = False

    async def start(self) -> None:
        """Build the task image if needed and start a fresh container."""

        image = self.config.resolved_image_tag
        if self.config.force_build or not await self._image_exists(image):
            await self._check(
                [
                    "docker",
                    "build",
                    "-t",
                    image,
                    "-f",
                    str(self.config.dockerfile),
                    str(self.config.environment_dir),
                ]
            )

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--label",
            f"benchflow.session={self.config.session_id}",
            "--workdir",
            self.config.workdir,
        ]
        if not self.spec.allow_internet:
            run_cmd.extend(["--network", "none"])
        if self.spec.cpus is not None:
            run_cmd.extend(["--cpus", str(self.spec.cpus)])
        if self.spec.memory_mb is not None:
            run_cmd.extend(["--memory", f"{self.spec.memory_mb}m"])
        for key, value in self.spec.env.items():
            run_cmd.extend(["--env", f"{key}={value}"])
        run_cmd.append(image)
        run_cmd.extend(self.config.command)
        await self._check(run_cmd)
        self._started = True

    async def stop(self, *, delete: bool = True) -> None:
        """Stop and optionally delete the container."""

        if not self._started:
            return
        await self._run(["docker", "stop", self.container_name])
        if delete:
            await self._run(["docker", "rm", "-f", self.container_name])
        self._started = False

    async def exec(
        self,
        cmd: str | Sequence[str],
        *,
        user: str = "root",
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int = 30,
    ) -> ExecResult:
        """Execute a command in the running container."""

        docker_cmd = ["docker", "exec", "-i"]
        if user:
            docker_cmd.extend(["--user", user])
        if cwd:
            docker_cmd.extend(["--workdir", cwd])
        for key, value in (env or {}).items():
            docker_cmd.extend(["--env", f"{key}={value}"])
        docker_cmd.append(self.container_name)
        if isinstance(cmd, str):
            docker_cmd.extend(["bash", "-lc", cmd])
        else:
            docker_cmd.extend(cmd)
        result = await self._run(docker_cmd, timeout_sec=timeout_sec)
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.returncode,
        )

    async def read_file(self, path: str) -> bytes:
        """Read a file from the container."""

        result = await self._check(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                f"base64 -w0 {shlex.quote(path)}",
            ]
        )
        return base64.b64decode(result.stdout)

    async def write_file(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the container."""

        parent = shlex.quote(str(Path(path).parent))
        q_path = shlex.quote(path)
        encoded = base64.b64encode(content).decode()
        await self._check(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                f"mkdir -p {parent} && base64 -d > {q_path}",
            ],
            stdin=encoded.encode(),
        )

    async def upload_dir(self, src: Path, dst: str) -> None:
        """Copy a host directory into the container."""

        await self._check(["docker", "cp", str(src), f"{self.container_name}:{dst}"])

    async def download_dir(self, src: str, dst: Path) -> None:
        """Copy a container directory to the host."""

        dst.parent.mkdir(parents=True, exist_ok=True)
        await self._check(["docker", "cp", f"{self.container_name}:{src}", str(dst)])

    async def _image_exists(self, image: str) -> bool:
        result = await self._run(["docker", "image", "inspect", image])
        return result.returncode == 0

    async def _check(
        self,
        cmd: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_sec: int = 600,
    ) -> _CompletedProcess:
        result = await self._run(cmd, stdin=stdin, timeout_sec=timeout_sec)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            stdout = result.stdout.decode(errors="replace")
            rendered = " ".join(shlex.quote(part) for part in cmd)
            raise RuntimeError(
                f"Command failed rc={result.returncode}: {rendered}\n{stderr or stdout}"
            )
        return result

    async def _run(
        self,
        cmd: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_sec: int = 600,
    ) -> _CompletedProcess:
        env = os.environ.copy()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin),
            timeout=timeout_sec,
        )

        return _CompletedProcess(proc.returncode or 0, stdout, stderr)


class DockerSandboxProvider:
    """Factory for native Docker sandboxes."""

    name = "docker"

    def __init__(self, task_path: Path) -> None:
        self.task_path = task_path

    async def create(self, spec: SandboxSpec) -> DockerSandbox:
        return DockerSandbox(DockerSandboxConfig(task_path=self.task_path), spec=spec)
