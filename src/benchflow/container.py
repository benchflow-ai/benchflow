"""Live stdio connection to a process inside a Docker container.

Harbor's env.exec() waits for command completion with stdin closed.
This module provides a live bidirectional pipe — send lines in, read lines out —
needed for ACP agents running inside containers.
"""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ContainerProcess:
    """Live stdin/stdout connection to a process inside a Docker compose service."""

    def __init__(
        self,
        project_name: str,
        project_dir: str,
        compose_files: list[str],
        service: str = "main",
    ):
        self._project_name = project_name
        self._project_dir = project_dir
        self._compose_files = compose_files
        self._service = service
        self._process: asyncio.subprocess.Process | None = None

    @classmethod
    def from_harbor_env(cls, env: Any) -> "ContainerProcess":
        """Create from a Harbor DockerEnvironment instance.

        Extracts the compose project name, directory, and file paths
        that Harbor already configured.
        """
        # Harbor DockerEnvironment stores these as properties
        project_name = env.session_id.lower().replace(".", "-")
        project_dir = str(env.environment_dir.resolve().absolute())
        compose_files = [str(p.resolve().absolute()) for p in env._docker_compose_paths]
        return cls(
            project_name=project_name,
            project_dir=project_dir,
            compose_files=compose_files,
        )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Start a process inside the container with live stdin/stdout."""
        cmd = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._project_dir,
        ]
        for f in self._compose_files:
            cmd.extend(["-f", f])
        cmd.extend(["exec", "-i", "-T"])
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([self._service, "bash", "-c", command])

        proc_env = os.environ.copy()
        # Resolve DOCKER_HOST if needed
        if not proc_env.get("DOCKER_HOST"):
            import subprocess

            try:
                r = subprocess.run(
                    [
                        "docker",
                        "context",
                        "inspect",
                        "--format",
                        "{{.Endpoints.docker.Host}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip():
                    proc_env["DOCKER_HOST"] = r.stdout.strip()
            except Exception:
                pass

        logger.debug(f"ContainerProcess: {' '.join(cmd)}")
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            limit=10 * 1024 * 1024,  # 10MB line buffer (default 64KB too small for large tool results)
        )
        logger.info(
            f"Container process started (pid={self._process.pid}, "
            f"project={self._project_name})"
        )

    async def readline(self) -> bytes:
        """Read one line from the container process stdout."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("Process not started")
        line = await self._process.stdout.readline()
        if not line:
            # Process died — try to capture stderr for diagnostics
            stderr_text = ""
            if self._process.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        self._process.stderr.read(8192), timeout=2
                    )
                    stderr_text = stderr_bytes.decode(errors="replace").strip()
                except Exception:
                    pass
            rc = self._process.returncode
            msg = f"Container process closed stdout (rc={rc})"
            if stderr_text:
                msg += f"\nstderr: {stderr_text[:1000]}"
            raise ConnectionError(msg)
        return line

    async def writeline(self, data: str) -> None:
        """Write one line to the container process stdin."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Process not started")
        self._process.stdin.write((data + "\n").encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        """Terminate the container process."""
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info("Container process terminated")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None
