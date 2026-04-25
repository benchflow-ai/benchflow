"""Live stdio connection to a process inside a sandbox.

Provides a bidirectional pipe (send lines in, read lines out) needed for
ACP agents running inside containers.  Two implementations:

- DockerProcess: uses `docker compose exec -i` (local Docker)
- DaytonaProcess: uses SSH to a Daytona sandbox
"""

import asyncio
import contextlib
import logging
import os
import shlex
import uuid
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_BUFFER_LIMIT = 10 * 1024 * 1024  # 10MB readline buffer
_DIAG_TRUNCATE = 2000  # max chars for diagnostic stderr in error messages


async def drain_oversized_line(reader: asyncio.StreamReader) -> int:
    """Drain an oversized line from *reader* after a buffer overflow.

    Clears the internal buffer and attempts to skip ahead to the next
    newline.  Returns the number of bytes discarded.
    """
    # Reach into asyncio.StreamReader internals to clear the buffer after
    # a LimitOverrunError. There's no public API for this; the private
    # attributes are stable across Python 3.10+.
    skipped = len(reader._buffer)  # ty: ignore[unresolved-attribute]
    reader._buffer.clear()  # ty: ignore[unresolved-attribute]
    reader._maybe_resume_transport()  # ty: ignore[unresolved-attribute]
    try:
        await asyncio.wait_for(reader.readuntil(b"\n"), timeout=5)
    except Exception:
        logger.debug("Could not find next newline after buffer overflow")
    return skipped


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
            skipped = await drain_oversized_line(self._process.stdout)
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
                    logger.debug("Could not read stderr from closed process")
            rc = self._process.returncode if self._process else None
            # Diagnose: rc=None with closed stdout usually means the *transport*
            # died (SSH/Daytona idle sleep, container killed) while the local
            # subprocess wrapper is still alive. rc set means the local process
            # actually exited. Surfacing the distinction makes the failure
            # actionable instead of cryptic.
            pid = self._process.pid if self._process else None
            if rc is None:
                hint = (
                    f"Local subprocess (pid={pid}) is still alive but its "
                    "stdout/transport closed. This usually means the remote "
                    "container or SSH session was killed (e.g. Daytona idle "
                    "sleep, agent hung with no output)."
                )
            else:
                hint = f"Local subprocess exited with rc={rc} before stdout closed."
            msg = f"Process closed stdout (rc={rc}): {hint}"
            if stderr_text:
                msg += f"\nstderr: {stderr_text[:_DIAG_TRUNCATE]}"
            raise ConnectionError(msg)
        return line

    async def writeline(self, data: str) -> None:
        """Write one line to stdin."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Process not started")
        self._process.stdin.write((data + "\n").encode())
        await self._process.stdin.drain()

    async def close(self) -> None:
        """Terminate the process (idempotent — safe to call after process death)."""
        if self._process:
            if self._process.stdin:
                with contextlib.suppress(OSError):  # already closed
                    self._process.stdin.close()
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except TimeoutError:
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

    def _compose_cmd(self) -> list[str]:
        """Base docker compose command with project/file flags."""
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
        return cmd

    def _host_env(self) -> dict[str, str]:
        """Host process env with DOCKER_HOST resolved if needed."""
        proc_env = os.environ.copy()
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
                logger.debug("Could not inspect docker context", exc_info=True)
        return proc_env

    _ENV_PATH = "/tmp/.benchflow_env"

    async def _write_env_to_container(
        self,
        env: dict[str, str],
        proc_env: dict[str, str],
    ) -> None:
        """Write env vars to a file inside the container (not visible in ps aux)."""
        lines = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        write_cmd = self._compose_cmd()
        write_cmd.extend(
            [
                "exec",
                "-T",
                self._service,
                "bash",
                "-c",
                f"cat > {self._ENV_PATH} && chmod 600 {self._ENV_PATH}",
            ]
        )
        proc = await asyncio.create_subprocess_exec(
            *write_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(lines.encode()), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to write env file in container (rc={proc.returncode}): "
                f"{stderr.decode()[:500]}"
            )
        logger.debug("Env file written inside container (%d vars)", len(env))

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        proc_env = self._host_env()

        # Write env vars to a file inside the container, then source it
        # in the main command. This keeps secrets off `ps aux` on the host
        # and avoids `--env-file` (not supported in all Compose versions).
        if env:
            await self._write_env_to_container(env, proc_env)
            command = f"source {self._ENV_PATH} && rm -f {self._ENV_PATH} && {command}"

        cmd = self._compose_cmd()
        cmd.extend(["exec", "-i", "-T"])
        if cwd:
            cmd.extend(["-w", cwd])
        cmd.extend([self._service, "bash", "-c", command])

        logger.debug(f"DockerProcess: {' '.join(cmd[:10])}...")
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


class DaytonaProcess(LiveProcess):
    """Live stdin/stdout via SSH to a Daytona sandbox.

    For DinD (compose) sandboxes, the SSH connects to the VM and then
    `docker compose exec -i main bash -c <command>` is run remotely.
    For direct sandboxes, the command runs directly via SSH.
    """

    def __init__(
        self,
        sandbox: Any,
        is_dind: bool = False,
        compose_cmd_prefix: str = "",
        compose_cmd_base: str = "",
    ):
        self._sandbox = sandbox
        self._is_dind = is_dind
        self._compose_cmd_prefix = compose_cmd_prefix
        self._compose_cmd_base = compose_cmd_base

    @classmethod
    async def from_harbor_env(cls, env: Any) -> "DaytonaProcess":
        """Create from a Harbor DaytonaEnvironment."""
        sandbox = env._sandbox
        if not sandbox:
            raise RuntimeError("Daytona sandbox not started")

        # Detect DinD mode by checking if the environment uses compose
        is_dind = hasattr(env, "_strategy") and hasattr(env._strategy, "_compose_cmd")

        compose_cmd_prefix = ""
        compose_cmd_base = ""
        if is_dind:
            # Build compose env vars and command prefix for DinD
            strategy = env._strategy
            compose_env = " ".join(
                f"{k}={shlex.quote(v)}" for k, v in strategy._compose_env_vars().items()
            )
            compose_cmd_prefix = compose_env
            # Extract the full compose base command with project/file flags
            # (e.g. "docker compose -p NAME --project-directory DIR -f F1 -f F2")
            # so that `docker compose exec` can find the running project.
            compose_cmd_base = strategy._compose_cmd([])

        return cls(
            sandbox=sandbox,
            is_dind=is_dind,
            compose_cmd_prefix=compose_cmd_prefix,
            compose_cmd_base=compose_cmd_base,
        )

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
            # Build the docker compose exec command to run inside the DinD VM.
            # Use the full compose base command (with -p, --project-directory,
            # and -f flags) so that exec can find the running project.
            if self._compose_cmd_base:
                inner_parts = [*shlex.split(self._compose_cmd_base), "exec", "-i", "-T"]
            else:
                inner_parts = ["docker", "compose", "exec", "-i", "-T"]
            if cwd:
                inner_parts.extend(["-w", cwd])
            # Write env vars to a temp file on the remote VM instead of passing
            # as -e K=V args (which are visible in ps aux on the remote host).
            # Use a Python-generated unique suffix instead of a shell `$$`
            # expansion: shlex.join() (below) single-quotes the --env-file arg,
            # so `$$` would survive as a literal in the docker compose call
            # while the cat > ... heredoc would expand it — the file would be
            # written to one path and read from another. uuid.uuid4 sidesteps
            # the entire shell-expansion-vs-quoting problem.
            remote_env_path = None
            if env:
                remote_env_path = f"/tmp/benchflow_env_{uuid.uuid4().hex[:16]}.env"
                env_lines = "\n".join(f"{k}={v}" for k, v in env.items())
                inner_parts.extend(["--env-file", remote_env_path])
            inner_parts.extend(["main", "bash", "-c", command])
            inner_cmd = shlex.join(inner_parts)

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
                remote_cmd = (
                    f"{write_cmd}\ntrap 'rm -f {remote_env_path}' EXIT\n{remote_cmd}"
                )
        else:
            # Direct sandbox — run command via SSH.
            # Write env vars to a file on the remote host and source it,
            # instead of passing as `env K=V` args visible in ps aux.
            env_prefix = ""
            remote_env_path = None
            if env:
                # Python-generated unique suffix; see DinD branch above for why
                # $$ shell expansion is fragile across quoting boundaries.
                remote_env_path = f"/tmp/benchflow_env_{uuid.uuid4().hex[:16]}.env"
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
                remote_cmd = (
                    f"{write_cmd}\ntrap 'rm -f {remote_env_path}' EXIT\n{remote_cmd}"
                )

        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
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


class DaytonaPtyProcess(LiveProcess):
    """Live stdin/stdout via Daytona PTY WebSocket API.

    Uses the Daytona SDK's PTY session (WebSocket) instead of SSH, which
    maintains long-lived interactive pipes through DinD compose layers.
    Falls back to this for DinD sandboxes where SSH pipes break.
    """

    _process = None  # Not used — override readline/writeline/close

    def __init__(self, sandbox: Any, compose_cmd_prefix: str, compose_cmd_base: str):
        self._sandbox = sandbox
        self._compose_cmd_prefix = compose_cmd_prefix
        self._compose_cmd_base = compose_cmd_base
        self._pty = None
        self._line_buffer = asyncio.Queue()
        self._partial = b""
        self._closed = False

    @classmethod
    async def from_harbor_env(cls, env: Any) -> "DaytonaPtyProcess":
        sandbox = env._sandbox
        if not sandbox:
            raise RuntimeError("Daytona sandbox not started")
        strategy = env._strategy
        compose_env = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in strategy._compose_env_vars().items()
        )
        compose_cmd_base = strategy._compose_cmd([])
        return cls(
            sandbox=sandbox,
            compose_cmd_prefix=compose_env,
            compose_cmd_base=compose_cmd_base,
        )

    async def _on_pty_data(self, data: bytes) -> None:
        self._partial += data
        while b"\n" in self._partial:
            line, self._partial = self._partial.split(b"\n", 1)
            line = line.replace(b"\r", b"")
            await self._line_buffer.put(line + b"\n")

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        session_id = f"acp-{uuid.uuid4().hex[:8]}"
        pty_env = {}
        if self._compose_cmd_prefix:
            for part in shlex.split(self._compose_cmd_prefix):
                if "=" in part:
                    k, v = part.split("=", 1)
                    pty_env[k] = v

        self._pty = await self._sandbox.process.create_pty_session(
            id=session_id,
            on_data=self._on_pty_data,
            envs=pty_env if pty_env else None,
        )
        await self._pty.wait_for_connection()
        logger.info(f"DaytonaPtyProcess: PTY connected (session={session_id})")

        compose_parts = (
            shlex.split(self._compose_cmd_base)
            if self._compose_cmd_base
            else ["docker", "compose"]
        )
        exec_parts = [*compose_parts, "exec", "-i", "-T"]
        if cwd:
            exec_parts.extend(["-w", cwd])
        # Write env vars to a file inside the container (not visible in ps aux),
        # matching the approach in DaytonaProcess.start().
        env_file_cmd = ""
        if env:
            env_file_path = f"/tmp/.benchflow_env_{uuid.uuid4().hex[:16]}"
            env_lines = "\n".join(
                f"export {k}={shlex.quote(v)}" for k, v in env.items()
            )
            env_file_cmd = (
                f"cat > {env_file_path} <<'__EOF__'\n{env_lines}\n__EOF__\n"
                f". {env_file_path} && rm -f {env_file_path} && "
            )
        exec_parts.extend(["main", "bash", "-lc", f"{env_file_cmd}{command}"])
        exec_cmd = shlex.join(exec_parts)

        # Use a marker + stty to cleanly hand over the PTY to the agent.
        # 1. Disable echo so typed commands don't appear in output
        # 2. Print marker so we know when to start reading ACP output
        # 3. exec into compose exec so the agent owns the PTY
        marker = f"__BENCHFLOW_ACP_{session_id}__"
        setup = f"stty -echo 2>/dev/null; echo '{marker}'; exec {exec_cmd}\n"
        await self._pty.send_input(setup)
        logger.info("DaytonaPtyProcess: sent setup, waiting for marker...")

        while True:
            try:
                line = await asyncio.wait_for(self._line_buffer.get(), timeout=120)
                decoded = line.decode(errors="replace").strip()
                logger.debug(f"DaytonaPtyProcess drain: {decoded[:120]}")
                if marker in decoded:
                    break
            except TimeoutError as e:
                raise ConnectionError(
                    "DaytonaPtyProcess: timeout waiting for agent start marker"
                ) from e

        logger.info("DaytonaPtyProcess: marker seen, agent starting")

    async def readline(self) -> bytes:
        if self._closed:
            raise ConnectionError("PTY closed")
        try:
            line = await asyncio.wait_for(self._line_buffer.get(), timeout=900)
            return line
        except TimeoutError as e:
            raise ConnectionError("PTY readline timeout (900s)") from e
        except Exception as e:
            raise ConnectionError(f"PTY readline error: {e}") from e

    async def writeline(self, data: str) -> None:
        if not self._pty or self._closed:
            raise RuntimeError("PTY not started")
        await self._pty.send_input(data + "\n")

    async def close(self) -> None:
        self._closed = True
        if self._pty:
            with contextlib.suppress(Exception):
                await self._pty.kill()
            with contextlib.suppress(Exception):
                await self._pty.disconnect()
            logger.info("DaytonaPtyProcess terminated")

    @property
    def is_running(self) -> bool:
        return self._pty is not None and not self._closed
