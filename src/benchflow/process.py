"""Live stdio connection to a process inside a sandbox.

Provides a bidirectional pipe (send lines in, read lines out) needed for
ACP agents running inside containers.  Two implementations:

- DockerProcess: uses `docker compose exec -i` (local Docker)
- DaytonaProcess: uses SSH to a Daytona sandbox
"""

import asyncio
import logging
import os
import shlex
import tempfile
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_BUFFER_LIMIT = 10 * 1024 * 1024  # 10MB readline buffer


class LiveProcess(ABC):
    """Abstract live stdin/stdout connection to a process inside a sandbox."""

    _process: asyncio.subprocess.Process | None = None

    @abstractmethod
    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Start the process with live stdin/stdout."""

    async def readline(self) -> bytes:
        """Read one line from stdout."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("Process not started")
        try:
            line = await self._process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as e:
            # Buffer overflow — line exceeds _BUFFER_LIMIT.
            # Drain the buffer and skip to next newline.
            reader = self._process.stdout
            skipped = len(reader._buffer)
            reader._buffer.clear()
            reader._maybe_resume_transport()
            # Try to consume remaining bytes up to next newline
            try:
                await asyncio.wait_for(reader.readuntil(b"\n"), timeout=5)
            except Exception:
                pass
            logger.warning(f"Skipped oversized line ({skipped} bytes): {e}")
            # Return empty line — caller will retry readline
            return b""
        if not line:
            stderr_text = ""
            if self._process and self._process.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        self._process.stderr.read(8192), timeout=2
                    )
                    stderr_text = stderr_bytes.decode(errors="replace").strip()
                except Exception:
                    pass
            rc = self._process.returncode if self._process else None
            msg = f"Process closed stdout (rc={rc})"
            if stderr_text:
                msg += f"\nstderr: {stderr_text[:2000]}"
            raise ConnectionError(msg)
        return line

    async def writeline(self, data: str) -> None:
        """Write one line to stdin."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Process not started")
        self._process.stdin.write((data + "\n").encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        """Terminate the process."""
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info("Process terminated")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None


class DockerProcess(LiveProcess):
    """Live stdin/stdout via `docker compose exec -i`."""

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

    @classmethod
    def from_harbor_env(cls, env: Any) -> "DockerProcess":
        """Create from a Harbor DockerEnvironment."""
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
        cmd = [
            "docker", "compose",
            "-p", self._project_name,
            "--project-directory", self._project_dir,
        ]
        for f in self._compose_files:
            cmd.extend(["-f", f])
        cmd.extend(["exec", "-i", "-T"])
        if cwd:
            cmd.extend(["-w", cwd])

        # Write env vars to a temp file instead of passing as -e K=V args
        # (which are visible in ps aux).
        env_file_path = None
        if env:
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".env", prefix="benchflow_", delete=False,
            )
            try:
                os.chmod(f.name, 0o600)
                for k, v in env.items():
                    f.write(f"{k}={v}\n")
                f.close()
                env_file_path = f.name
                cmd.extend(["--env-file", env_file_path])
            except BaseException:
                f.close()
                os.unlink(f.name)
                raise

        cmd.extend([self._service, "bash", "-c", command])

        proc_env = os.environ.copy()
        if not proc_env.get("DOCKER_HOST"):
            import subprocess
            try:
                r = subprocess.run(
                    ["docker", "context", "inspect", "--format",
                     "{{.Endpoints.docker.Host}}"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip():
                    proc_env["DOCKER_HOST"] = r.stdout.strip()
            except Exception:
                pass

        try:
            logger.debug(f"DockerProcess: {' '.join(cmd)}")
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=_BUFFER_LIMIT,
            )
            logger.info(
                f"Docker process started (pid={self._process.pid}, "
                f"project={self._project_name})"
            )
        finally:
            if env_file_path:
                os.unlink(env_file_path)


class DaytonaProcess(LiveProcess):
    """Live stdin/stdout via SSH to a Daytona sandbox.

    For DinD (compose) sandboxes, the SSH connects to the VM and then
    `docker compose exec -i main bash -c <command>` is run remotely.
    For direct sandboxes, the command runs directly via SSH.
    """

    def __init__(self, sandbox: Any, is_dind: bool = False, compose_cmd_prefix: str = ""):
        """
        Args:
            sandbox: Daytona AsyncSandbox instance
            is_dind: True if sandbox is DinD (compose) mode
            compose_cmd_prefix: For DinD, the docker compose prefix with env vars
        """
        self._sandbox = sandbox
        self._is_dind = is_dind
        self._compose_cmd_prefix = compose_cmd_prefix

    @classmethod
    async def from_harbor_env(cls, env: Any) -> "DaytonaProcess":
        """Create from a Harbor DaytonaEnvironment."""
        sandbox = env._sandbox
        if not sandbox:
            raise RuntimeError("Daytona sandbox not started")

        # Detect DinD mode by checking if the environment uses compose
        is_dind = hasattr(env, '_strategy') and hasattr(env._strategy, '_compose_cmd')

        compose_cmd_prefix = ""
        if is_dind:
            # Build compose env vars and command prefix for DinD
            strategy = env._strategy
            compose_env = " ".join(
                f"{k}={shlex.quote(v)}"
                for k, v in strategy._compose_env_vars().items()
            )
            compose_cmd_prefix = compose_env

        return cls(sandbox=sandbox, is_dind=is_dind, compose_cmd_prefix=compose_cmd_prefix)

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # Get SSH credentials
        ssh_access = await self._sandbox.create_ssh_access()
        ssh_target = f"{ssh_access.token}@ssh.app.daytona.io"

        if self._is_dind:
            # Build the docker compose exec command to run inside the DinD VM
            inner_parts = ["docker", "compose", "exec", "-i", "-T"]
            if cwd:
                inner_parts.extend(["-w", cwd])
            # Write env vars to a temp file on the remote VM instead of passing
            # as -e K=V args (which are visible in ps aux on the remote host).
            remote_env_path = None
            if env:
                remote_env_path = "/tmp/benchflow_env_$$.env"
                env_lines = "\n".join(f"{k}={v}" for k, v in env.items())
                inner_parts.extend(["--env-file", remote_env_path])
            inner_parts.extend(["main", "bash", "-c", shlex.quote(command)])
            inner_cmd = " ".join(inner_parts)

            if self._compose_cmd_prefix:
                remote_cmd = f"{self._compose_cmd_prefix} {inner_cmd}"
            else:
                remote_cmd = inner_cmd

            if remote_env_path:
                # Write env file, set trap for cleanup, then run the command
                write_cmd = (
                    f"umask 077 && cat > {remote_env_path} <<'__BENCHFLOW_ENV__'\n"
                    f"{env_lines}\n__BENCHFLOW_ENV__"
                )
                remote_cmd = f"{write_cmd}\ntrap 'rm -f {remote_env_path}' EXIT\n{remote_cmd}"
        else:
            # Direct sandbox — run command via SSH.
            # Write env vars to a file on the remote host and source it,
            # instead of passing as `env K=V` args visible in ps aux.
            env_prefix = ""
            remote_env_path = None
            if env:
                remote_env_path = "/tmp/benchflow_env_$$.env"
                env_lines = "\n".join(
                    f"export {k}={shlex.quote(v)}" for k, v in env.items()
                )
                env_prefix = f". {remote_env_path} && "
            if cwd:
                remote_cmd = f"cd {shlex.quote(cwd)} && {env_prefix}{command}"
            else:
                remote_cmd = f"{env_prefix}{command}"

            if remote_env_path:
                write_cmd = (
                    f"umask 077 && cat > {remote_env_path} <<'__BENCHFLOW_ENV__'\n"
                    f"{env_lines}\n__BENCHFLOW_ENV__"
                )
                remote_cmd = f"{write_cmd}\ntrap 'rm -f {remote_env_path}' EXIT\n{remote_cmd}"

        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            ssh_target,
            remote_cmd,
        ]

        logger.debug(f"DaytonaProcess: ssh {ssh_target} {remote_cmd[:100]}...")
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_BUFFER_LIMIT,
        )
        logger.info(f"Daytona process started (pid={self._process.pid})")
