"""Live stdio to a Daytona sandbox — SSH subprocess and PTY WebSocket variants."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import shlex
import tempfile
import uuid
from typing import Any

from benchflow.sandbox.process._acp_lines import filter_command
from benchflow.sandbox.process._base import (
    _BOOTSTRAP_DONE,
    _BUFFER_LIMIT,
    _DIAG_TRUNCATE,
    _ENV_KEY_RE,
    LiveProcess,
    SubprocessLiveProcess,
    _readline_timeout_sec,
)

logger = logging.getLogger(__name__)

_DAYTONA_PTY_READLINE_TIMEOUT_ENV = "BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT"
_DAYTONA_PTY_READLINE_TIMEOUT_DEFAULT_SEC = 900.0
_DAYTONA_SSH_ACCESS_TTL_MINUTES = 48 * 60
_DAYTONA_SSH_SERVER_ALIVE_INTERVAL_SEC = 30
_DAYTONA_SSH_SERVER_ALIVE_COUNT_MAX = 12


def _daytona_pty_readline_timeout_sec(silence_floor: float | None = None) -> float:
    return _readline_timeout_sec(
        _DAYTONA_PTY_READLINE_TIMEOUT_ENV,
        _DAYTONA_PTY_READLINE_TIMEOUT_DEFAULT_SEC,
        silence_floor,
    )


def _watch_pty(pty: Any) -> asyncio.Task[Any] | None:
    """A task that finishes when the SDK's PTY handle loses its websocket.

    ``AsyncPtyHandle.wait()`` returns once the handle's websocket reader has
    ended: a close frame (agent exit, a server-side 1006/1008), a socket
    error, or a missed heartbeat pong. None when the handle has no such
    coroutine (fakes, other SDKs): reads then wait on the line queue alone.
    """
    wait = getattr(pty, "wait", None)
    if not callable(wait):
        return None
    try:
        awaitable = wait()
    except Exception:
        return None
    if not inspect.isawaitable(awaitable):
        return None
    task = asyncio.ensure_future(awaitable)
    # disconnect() cancels the reader, which ends wait() with CancelledError
    # or an SDK error; retrieve it so it is never reported as unhandled.
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return task


def _pty_connected(pty: Any) -> bool | None:
    """``AsyncPtyHandle.is_connected()``: False as soon as the SDK has seen
    the websocket close, a few loop turns before ``wait()`` returns. None
    when the handle cannot say."""
    is_connected = getattr(pty, "is_connected", None)
    if not callable(is_connected):
        return None
    try:
        state = is_connected()
    except Exception:
        return None
    if inspect.iscoroutine(state):  # a mock, not the SDK's plain method
        state.close()
        return None
    return state if isinstance(state, bool) else None


def _pty_closed_reason(pty: Any) -> tuple[str, int | None]:
    """Why the PTY's websocket is gone, with its close code; and the exit code."""
    exit_code = getattr(pty, "exit_code", None)
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        exit_code = None
    error = getattr(pty, "error", None)
    ws = getattr(pty, "_ws", None)  # the SDK keeps no public close code
    close_code = getattr(ws, "close_code", None)
    socket_error = None
    exception = getattr(ws, "exception", None)
    if callable(exception):
        with contextlib.suppress(Exception):
            socket_error = exception()
    details = []
    if isinstance(close_code, int) and not isinstance(close_code, bool):
        details.append(f"close_code={close_code}")
    if isinstance(socket_error, BaseException):
        details.append(f"socket error: {socket_error!r}")
    detail = f" ({', '.join(details)})" if details else ""
    if isinstance(error, str) and error:
        return f"{error}{detail}", exit_code
    if exit_code is not None:
        return f"agent process exited with code {exit_code}{detail}", exit_code
    return f"websocket closed{detail}", exit_code


async def _cleanup_daytona_remote_env_file(
    sandbox: Any,
    remote_env_path: str,
) -> None:
    # timeout=10 is the server-side exec limit; the extra wait_for bounds the
    # client-side await too, so a dead connection can't hang a teardown path
    # (this runs inside close()'s finally).
    with contextlib.suppress(Exception):
        await asyncio.wait_for(
            sandbox.process.exec(
                f"rm -f {shlex.quote(remote_env_path)}",
                timeout=10,
            ),
            timeout=30,
        )


async def _bootstrap_daytona_script_file(
    sandbox: Any,
    *,
    remote_script_path: str,
    script: str,
    error_label: str,
) -> None:
    delimiter = f"__BENCHFLOW_SCRIPT_{uuid.uuid4().hex}__"
    remote_script_path_q = shlex.quote(remote_script_path)
    command = (
        f"cat > {remote_script_path_q} <<'{delimiter}'\n"
        f"{script}"
        f"{delimiter}\n"
        f"chmod 700 {remote_script_path_q}\n"
        f"echo {_BOOTSTRAP_DONE}\n"
    )
    response = await sandbox.process.exec(
        command,
        timeout=30,
    )
    stdout_text = str(getattr(response, "result", "") or "")
    exit_code = getattr(response, "exit_code", 1)
    if exit_code != 0 or _BOOTSTRAP_DONE not in stdout_text.splitlines():
        raise RuntimeError(
            f"Failed to bootstrap {error_label} "
            f"(rc={exit_code}): {stdout_text[:_DIAG_TRUNCATE]}"
        )


async def _bootstrap_daytona_env_file(
    sandbox: Any,
    *,
    remote_env_path: str,
    env: dict[str, str],
    shell_exports: bool,
    error_label: str,
) -> None:
    env_keys = list(env)
    invalid = [key for key in env_keys if not _ENV_KEY_RE.match(key)]
    if invalid:
        raise ValueError(
            "Invalid environment variable name(s): " + ", ".join(sorted(invalid))
        )
    command = DaytonaProcess._bootstrap_env_command(
        remote_env_path=remote_env_path,
        env_keys=env_keys,
        shell_exports=shell_exports,
    )
    response = await sandbox.process.exec(command, env=env, timeout=30)
    stdout_text = str(getattr(response, "result", "") or "")
    exit_code = getattr(response, "exit_code", 1)
    if exit_code != 0 or _BOOTSTRAP_DONE not in stdout_text.splitlines():
        raise RuntimeError(
            f"Failed to bootstrap {error_label} "
            f"(rc={exit_code}): {stdout_text[:_DIAG_TRUNCATE]}"
        )


class DaytonaProcess(SubprocessLiveProcess):
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
        self._ssh_config_path: str | None = None
        self._ssh_config_cleanup_task: asyncio.Task[None] | None = None

    @staticmethod
    def _write_ssh_config(ssh_user: str) -> str:
        fd, path = tempfile.mkstemp(prefix="benchflow_daytona_ssh_", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write("Host benchflow-daytona\n")
                f.write("  HostName ssh.app.daytona.io\n")
                f.write(f"  User {ssh_user}\n")
                f.write("  StrictHostKeyChecking no\n")
                f.write("  UserKnownHostsFile /dev/null\n")
                f.write(
                    f"  ServerAliveInterval {_DAYTONA_SSH_SERVER_ALIVE_INTERVAL_SEC}\n"
                )
                f.write(
                    f"  ServerAliveCountMax {_DAYTONA_SSH_SERVER_ALIVE_COUNT_MAX}\n"
                )
                f.write("  TCPKeepAlive yes\n")
                f.write("  LogLevel ERROR\n")
            os.chmod(path, 0o600)
        except Exception:
            with contextlib.suppress(Exception):
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
            raise
        return path

    @staticmethod
    def _ssh_args(ssh_config_path: str, remote_cmd: str) -> list[str]:
        return [
            "ssh",
            "-F",
            ssh_config_path,
            "benchflow-daytona",
            remote_cmd,
        ]

    def _unlink_ssh_config(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        if self._ssh_config_path == path:
            self._ssh_config_path = None

    async def _cleanup_ssh_config_after_exit(
        self,
        process: asyncio.subprocess.Process,
        path: str,
    ) -> None:
        with contextlib.suppress(Exception):
            await process.wait()
        self._unlink_ssh_config(path)

    @classmethod
    async def from_sandbox_env(cls, env: Any) -> DaytonaProcess:
        """Create from a sandbox environment (DaytonaSandbox)."""
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

    @staticmethod
    def _bootstrap_env_command(
        *,
        remote_env_path: str,
        env_keys: list[str],
        shell_exports: bool,
    ) -> str:
        remote_env_path_q = shlex.quote(remote_env_path)
        keys = " ".join(shlex.quote(key) for key in env_keys)
        if shell_exports:
            write_value = (
                '  printf \'export %s=\' "$key" >> "$env_file"\n'
                '  quote_env_value "$(printenv "$key")" >> "$env_file"\n'
                "  printf '\\n' >> \"$env_file\"\n"
            )
        else:
            write_value = (
                '  printf \'%s=%s\\n\' "$key" "$(printenv "$key")" >> "$env_file"\n'
            )
        script = (
            f"env_file={remote_env_path_q}\n"
            "success=0\n"
            "umask 077\n"
            'trap \'[ "$success" = 1 ] || rm -f "$env_file"\' EXIT\n'
            ': > "$env_file"\n'
            "quote_env_value() {\n"
            '  printf "\'"\n'
            "  printf '%s' \"$1\" | sed \"s/'/'\\\\\\\\''/g\"\n"
            '  printf "\'"\n'
            "}\n"
            f"for key in {keys}; do\n"
            f"{write_value}"
            "done\n"
            "success=1\n"
            f"echo {_BOOTSTRAP_DONE}\n"
        )
        return f"sh -c {shlex.quote(script)}"

    async def _cleanup_remote_env_file(self, remote_env_path: str) -> None:
        await _cleanup_daytona_remote_env_file(self._sandbox, remote_env_path)

    async def _bootstrap_env_file(
        self,
        *,
        remote_env_path: str,
        env: dict[str, str],
        shell_exports: bool,
    ) -> None:
        await _bootstrap_daytona_env_file(
            self._sandbox,
            remote_env_path=remote_env_path,
            env=env,
            shell_exports=shell_exports,
            error_label="Daytona agent env",
        )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # Lines over the ACP limit are rewritten in the sandbox: Daytona
        # closes the PTY websocket (1008) on a multi-MB message (#1138).
        command = filter_command(command)
        remote_env_path = None

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
            # Source provider values into the remote shell, then pass only
            # env names through `docker compose exec --env KEY`. Compose exec
            # does not accept --env-file, and `--env KEY=value` would leak
            # provider values into the remote command line.
            if env:
                remote_env_path = f"/tmp/benchflow_env_{uuid.uuid4().hex[:16]}.env"
                await self._bootstrap_env_file(
                    remote_env_path=remote_env_path,
                    env=env,
                    shell_exports=True,
                )
                for key in env:
                    inner_parts.extend(["--env", key])
            inner_parts.extend(["main", "bash", "-c", command])
            inner_cmd = shlex.join(inner_parts)

            remote_cmd = (
                f"{self._compose_cmd_prefix} {inner_cmd}"
                if self._compose_cmd_prefix
                else inner_cmd
            )
            if remote_env_path:
                remote_env_path_q = shlex.quote(remote_env_path)
                # Source the env file to populate vars, then exec compose.
                # The EXIT trap handles cleanup; we intentionally keep the file
                # alive until exit so that compose_cmd_prefix wrappers (which
                # may re-exec the shell) still inherit the sourced vars.
                remote_cmd = (
                    f"trap 'rm -f {remote_env_path_q}' EXIT; "
                    f". {remote_env_path_q} && "
                    f"{remote_cmd}"
                )
        else:
            # Direct sandbox — run command via SSH.
            # Write env vars to a file on the remote host and source it,
            # instead of passing as `env K=V` args visible in ps aux.
            env_prefix = ""
            if env:
                # Python-generated unique suffix; see DinD branch above for why
                # $$ shell expansion is fragile across quoting boundaries.
                remote_env_path = f"/tmp/benchflow_env_{uuid.uuid4().hex[:16]}.env"
                await self._bootstrap_env_file(
                    remote_env_path=remote_env_path,
                    env=env,
                    shell_exports=True,
                )
                remote_env_path_q = shlex.quote(remote_env_path)
                env_prefix = f". {remote_env_path_q} && rm -f {remote_env_path_q} && "
            if cwd:
                remote_cmd = f"cd {shlex.quote(cwd)} && {env_prefix}{command}"
            else:
                remote_cmd = f"{env_prefix}{command}"
            if remote_env_path:
                remote_cmd = (
                    f"trap 'rm -f {shlex.quote(remote_env_path)}' EXIT; {remote_cmd}"
                )

        try:
            ssh_access = await self._sandbox.create_ssh_access(
                expires_in_minutes=_DAYTONA_SSH_ACCESS_TTL_MINUTES
            )
            ssh_config_path = self._write_ssh_config(ssh_access.token)
            self._ssh_config_path = ssh_config_path
            cmd = self._ssh_args(ssh_config_path, remote_cmd)

            logger.debug(
                "DaytonaProcess: ssh benchflow-daytona %s...",
                remote_cmd[:100],
            )
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_BUFFER_LIMIT,
            )
            self._set_process(process)
        except Exception:
            if remote_env_path:
                await self._cleanup_remote_env_file(remote_env_path)
            await self.close()
            raise
        self._ssh_config_cleanup_task = asyncio.create_task(
            self._cleanup_ssh_config_after_exit(process, ssh_config_path)
        )
        logger.info(f"Daytona process started (pid={process.pid})")

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            if self._ssh_config_cleanup_task:
                self._ssh_config_cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._ssh_config_cleanup_task
                self._ssh_config_cleanup_task = None
            if self._ssh_config_path:
                self._unlink_ssh_config(self._ssh_config_path)


class DaytonaPtyProcess(LiveProcess):
    """Live stdin/stdout via Daytona PTY WebSocket API.

    Uses the Daytona SDK's PTY session (WebSocket) instead of SSH, which
    maintains long-lived interactive pipes through Daytona sandboxes.
    Compose sandboxes enter the ``main`` service through ``docker compose exec``;
    direct sandboxes run the agent command directly in the PTY shell.
    """

    _START_MARKER_TIMEOUT_SEC = 120

    _silence_floor_sec: float | None = None

    def __init__(self, sandbox: Any, compose_cmd_prefix: str, compose_cmd_base: str):
        self._sandbox = sandbox
        self._compose_cmd_prefix = compose_cmd_prefix
        self._compose_cmd_base = compose_cmd_base
        self._pty = None
        # Done once the PTY's websocket is gone (_watch_pty).
        self._pty_gone: asyncio.Task[Any] | None = None
        self._line_buffer = asyncio.Queue()
        self._partial = b""
        self._closed = False
        self._remote_env_path: str | None = None
        self._remote_script_path: str | None = None

    @classmethod
    async def from_sandbox_env(cls, env: Any) -> DaytonaPtyProcess:
        sandbox = env._sandbox
        if not sandbox:
            raise RuntimeError("Daytona sandbox not started")
        strategy = getattr(env, "_strategy", None)
        compose_env = ""
        compose_cmd_base = ""
        if (
            strategy is not None
            and hasattr(strategy, "_compose_env_vars")
            and hasattr(strategy, "_compose_cmd")
        ):
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

    async def _bootstrap_env_file(
        self,
        *,
        remote_env_path: str,
        env: dict[str, str],
    ) -> None:
        await _bootstrap_daytona_env_file(
            self._sandbox,
            remote_env_path=remote_env_path,
            env=env,
            shell_exports=True,
            error_label="Daytona PTY agent env",
        )

    async def _cleanup_started_env_file(self) -> None:
        remote_env_path = self._remote_env_path
        self._remote_env_path = None
        if remote_env_path:
            await _cleanup_daytona_remote_env_file(self._sandbox, remote_env_path)
        remote_script_path = self._remote_script_path
        self._remote_script_path = None
        if remote_script_path:
            await _cleanup_daytona_remote_env_file(self._sandbox, remote_script_path)

    def _clear_startup_output(self) -> None:
        self._partial = b""
        while not self._line_buffer.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._line_buffer.get_nowait()

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # Lines over the ACP limit are rewritten in the sandbox: Daytona
        # closes the PTY websocket (1008) on a multi-MB message (#1138).
        command = filter_command(command)
        remote_env_path = None
        session_id = f"acp-{uuid.uuid4().hex[:8]}"
        pty_env = {}
        if self._compose_cmd_prefix:
            for part in shlex.split(self._compose_cmd_prefix):
                if "=" in part:
                    k, v = part.split("=", 1)
                    pty_env[k] = v

        try:
            self._pty = await self._sandbox.process.create_pty_session(
                id=session_id,
                on_data=self._on_pty_data,
                envs=pty_env if pty_env else None,
            )
            await self._pty.wait_for_connection()
            self._pty_gone = _watch_pty(self._pty)
            logger.info(f"DaytonaPtyProcess: PTY connected (session={session_id})")

            if env:
                remote_env_path = f"/tmp/benchflow_env_{uuid.uuid4().hex[:16]}.env"
                self._remote_env_path = remote_env_path
                await self._bootstrap_env_file(
                    remote_env_path=remote_env_path,
                    env=env,
                )

            if self._compose_cmd_base:
                compose_parts = shlex.split(self._compose_cmd_base)
                exec_parts = [*compose_parts, "exec", "-i", "-T"]
                if cwd:
                    exec_parts.extend(["-w", cwd])
                if env:
                    for key in env:
                        exec_parts.extend(["--env", key])
                exec_parts.extend(["main", "bash", "-lc", command])
                exec_cmd = shlex.join(exec_parts)
                if remote_env_path:
                    remote_env_path_q = shlex.quote(remote_env_path)
                    setup_exec_cmd = (
                        f". {remote_env_path_q} && rm -f {remote_env_path_q} && "
                        f"exec {exec_cmd}"
                    )
                else:
                    setup_exec_cmd = f"exec {exec_cmd}"
            else:
                direct_parts: list[str] = []
                if cwd:
                    direct_parts.append(f"cd {shlex.quote(cwd)}")
                if remote_env_path:
                    remote_env_path_q = shlex.quote(remote_env_path)
                    direct_parts.append(f". {remote_env_path_q}")
                    direct_parts.append(f"rm -f {remote_env_path_q}")
                direct_parts.append(f"exec bash -lc {shlex.quote(command)}")
                setup_exec_cmd = " && ".join(direct_parts)

            remote_script_path = f"/tmp/benchflow_pty_exec_{uuid.uuid4().hex[:16]}.sh"
            self._remote_script_path = remote_script_path
            await _bootstrap_daytona_script_file(
                self._sandbox,
                remote_script_path=remote_script_path,
                script=(
                    "#!/usr/bin/env bash\n"
                    "set -e\n"
                    f"rm -f {shlex.quote(remote_script_path)}\n"
                    f"{setup_exec_cmd}\n"
                ),
                error_label="Daytona PTY agent command",
            )
            setup_exec_cmd = f"exec sh {shlex.quote(remote_script_path)}"

            # Use a marker + stty to cleanly hand over the PTY to the agent.
            # 1. Disable echo and canonical line buffering before ACP traffic.
            #    ACP JSON-RPC messages can be much longer than the usual
            #    4096-byte terminal line discipline limit, and canonical mode
            #    can corrupt long prompts before the agent JSON parser sees them.
            # 2. Print marker so we know when to start reading ACP output
            # 3. After the marker, exec a short uploaded script so the agent owns
            #    the PTY. The long compose/agent command must not be typed into
            #    the interactive PTY input path.
            #
            # Keep the marker command separate from the agent command. The
            # OpenHands launch command contains nested shell quoting; putting
            # it on the same interactive-shell line means the shell must parse
            # that whole line before running the marker echo.
            marker = f"__BENCHFLOW_ACP_{session_id}__"
            await self._pty.send_input(
                "stty raw -echo 2>/dev/null || "
                "stty -echo -icanon min 1 time 0 2>/dev/null || true; "
                f"echo '{marker}'\n"
            )
            logger.info("DaytonaPtyProcess: sent setup, waiting for marker...")

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._next_line(),
                        timeout=self._START_MARKER_TIMEOUT_SEC,
                    )
                    decoded = line.decode(errors="replace").strip()
                    logger.debug(f"DaytonaPtyProcess drain: {decoded[:120]}")
                    if marker in decoded:
                        break
                except TimeoutError as e:
                    from benchflow.diagnostics import (
                        TransportClosedDiagnostic,
                        TransportClosedError,
                    )

                    msg = (
                        "DaytonaPtyProcess: timeout waiting for agent start "
                        f"marker (session={session_id})"
                    )
                    raise TransportClosedError(
                        msg,
                        TransportClosedDiagnostic(
                            raw_message=msg,
                            transport_diagnosis="pty_startup_timeout",
                        ),
                    ) from e
            self._clear_startup_output()
            await self._pty.send_input(setup_exec_cmd + "\n")
        except Exception:
            await self._cleanup_started_env_file()
            await self.close()
            raise

        logger.info("DaytonaPtyProcess: marker seen, agent starting")

    async def readline(self) -> bytes:
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        if self._closed:
            msg = "PTY closed"
            raise TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg, transport_diagnosis="pty_error"
                ),
            )
        timeout = _daytona_pty_readline_timeout_sec(self._silence_floor_sec)
        try:
            line = await asyncio.wait_for(self._next_line(), timeout=timeout)
            return line
        except TimeoutError as e:
            msg = f"PTY readline timeout ({timeout:g}s)"
            raise TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg, transport_diagnosis="pty_error"
                ),
            ) from e
        except TransportClosedError:
            raise
        except Exception as e:
            msg = f"PTY readline error: {e}"
            raise TransportClosedError(
                msg,
                TransportClosedDiagnostic(
                    raw_message=msg[:500], transport_diagnosis="pty_error"
                ),
            ) from e

    async def _next_line(self) -> bytes:
        """The next line from the agent.

        Lines the websocket delivered before it went away are still read;
        after them, a gone websocket raises at once instead of leaving the
        read to wait out its timeout (#1143, #1144): a closed channel used to
        surface only as ``PTY readline timeout``, up to the prompt's whole
        silence budget later, and the reviewer's transport retry skips that.
        """
        gone = self._pty_gone
        if gone is None:
            return await self._line_buffer.get()
        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                return self._line_buffer.get_nowait()
            if self._peer_gone():
                raise self._peer_closed_error()
            get = asyncio.ensure_future(self._line_buffer.get())
            try:
                await asyncio.wait({get, gone}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not get.done():
                    get.cancel()
            if get.done() and not get.cancelled():
                return get.result()

    def _peer_closed_error(self) -> Exception:
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        if self._closed:  # our own close() ended the reader
            return TransportClosedError(
                "PTY closed",
                TransportClosedDiagnostic(
                    raw_message="PTY closed", transport_diagnosis="pty_error"
                ),
            )
        reason, exit_code = _pty_closed_reason(self._pty)
        msg = f"PTY closed by the peer: {reason}"
        return TransportClosedError(
            msg,
            TransportClosedDiagnostic(
                raw_message=msg[:500],
                process_exit_code=exit_code,
                transport_diagnosis="pty_closed",
            ),
        )

    def _peer_gone(self) -> bool:
        if self._pty_gone is None:
            return False
        return self._pty_gone.done() or _pty_connected(self._pty) is False

    def expect_silence(self, seconds: float) -> None:
        self._silence_floor_sec = seconds

    async def writeline(self, data: str) -> None:
        if not self._pty or self._closed:
            raise RuntimeError("PTY not started")
        if self._peer_gone():
            raise self._peer_closed_error()
        try:
            await self._pty.send_input(data + "\n")
        except Exception as exc:
            if self._peer_gone():
                raise self._peer_closed_error() from exc
            raise

    async def close(self) -> None:
        self._closed = True
        gone, self._pty_gone = self._pty_gone, None
        if gone is not None and not gone.done():
            gone.cancel()
        try:
            if self._pty:
                # kill/disconnect go over the PTY's websocket; on a dead or
                # wedged connection either await can block indefinitely, and a
                # hung close() freezes the whole rollout teardown (this exact
                # shape wedged a 25-task job for 11+ hours). Bound each call —
                # the sandbox is deleted by env.stop() regardless, so an
                # abandoned server-side PTY session costs nothing.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._pty.kill(), timeout=15)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._pty.disconnect(), timeout=15)
                logger.info("DaytonaPtyProcess terminated")
        finally:
            await self._cleanup_started_env_file()

    @property
    def is_running(self) -> bool:
        return self._pty is not None and not self._closed and not self._peer_gone()
