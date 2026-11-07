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

from benchflow._root_only import root_only_path_check
from benchflow._runtime_tree import KIT_BIN, RUN_DIR
from benchflow.sandbox import _transport
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    SYSTEM_SHELL,
    SYSTEM_SHELL_EMPTY_VARS,
    system_shell_prefix,
)
from benchflow.sandbox._root_launch import (
    inert_task_env,
    launch_bash,
    launch_path,
    launch_route,
    protect_root_terminal_home,
    require_protected_root_terminal,
    root_launch_argv,
    root_launch_script,
    terminal_env,
    typed_root_prelude,
)
from benchflow.sandbox.process._base import (
    _ANSI_CSI_RE_BYTES,
    _ANSI_OSC_RE_BYTES,
    _BOOTSTRAP_DONE,
    _BUFFER_LIMIT,
    _DIAG_TRUNCATE,
    _ENV_KEY_RE,
    LiveProcess,
    SubprocessLiveProcess,
    _readline_timeout_sec,
)

logger = logging.getLogger(__name__)

# The terminal's stty, by path: on a direct sandbox the shell typing it is
# root's (C8).
_STTY = "/bin/stty"


def _tool(route: Any, name: str) -> str:
    """A tool root's launch names by path in the sandbox: the protected
    kit's on the kit (C8), else the system one. *route* is None for the
    DinD VM's own files, outside the task container."""
    return f"{KIT_BIN}/{name}" if getattr(route, "kit", False) else f"/bin/{name}"


def _shell(route: Any) -> str:
    shell = getattr(route, "shell", None)
    return shell if isinstance(shell, str) and shell else SYSTEM_SHELL


def _stage_path(route: Any, plain: str) -> str:
    """Where a launch stages a file in the sandbox: a fresh name in
    BenchFlow's private staging folder on the protected kit (C8,
    PT-C8-R3-8), else *plain* (a random name in /tmp: no sandbox user, or
    the DinD VM's own files)."""
    if getattr(route, "kit", False):
        return f"{RUN_DIR}/{plain.rsplit('/', 1)[-1]}"
    return plain


def _staging_prefix(path: str) -> str:
    """sh run before writing *path*: BenchFlow's private staging folder
    made root's (mode 700) and checked along its route when *path* is in it
    (PT-C8-R3-8); nothing for the DinD VM's own files."""
    if not path.startswith(RUN_DIR + "/"):
        return ""
    run_dir = shlex.quote(RUN_DIR)
    return (
        f"mkdir -p {run_dir} && chown 0:0 {run_dir} && chmod 700 {run_dir} && "
        + root_only_path_check(RUN_DIR, "BenchFlow's private staging folder")
        + " || exit 1\n"
    )


def _routed(route: Any, script: str) -> str:
    """*script* for the provider to run: as *route*'s shell by absolute path
    with PATH pinned to the route's before its first program (C8,
    PY8-PT-R8-2, PT-C8-R8-1); unchanged for the DinD VM's own files (no
    route). The values the provider hands over as the command's environment
    stay readable (no ``env -i``): they are data, the staged env's.
    """
    if route is None:
        return script
    path = getattr(route, "pin", None) or getattr(route, "tools", None) or SYSTEM_PATH
    return f"{_shell(route)} -c {shlex.quote(system_shell_prefix(path) + chr(10) + script)}"


def _empty_loader_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """``env`` with each loader variable empty (C8).

    The provider starts the command's shell with these, so neither the
    image's ``LD_PRELOAD`` nor one the launch env names reaches the shell or
    the system tools the command names by path. The launch env's own values
    travel under inert names (``inert_task_env``) and come back after the
    drop (PT-C8-3).
    """
    return {**(env or {}), **{name: "" for name in SYSTEM_SHELL_EMPTY_VARS}}


def _compose_empty_loader_flags(env: dict[str, str] | None) -> list[str]:
    """``-e NAME=`` for every loader variable: root's shell in the task
    container starts with none, the launch env's included (PT-C8-3)."""
    return [arg for name in SYSTEM_SHELL_EMPTY_VARS for arg in ("-e", f"{name}=")]


_DAYTONA_PTY_READLINE_TIMEOUT_ENV = "BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT"
_DAYTONA_PTY_READLINE_TIMEOUT_DEFAULT_SEC = 900.0
# Daytona refuses SSH access for more than 1440 minutes (400 "expiresInMinutes
# must be between 1 and 1440"), so a longer ask failed every ssh-transport
# launch before the agent started (#1208). 24 h still outlives the longest
# agent budgets in use (50400 s).
_DAYTONA_SSH_ACCESS_TTL_MINUTES = 24 * 60
# What Daytona's PTY session gives the agent process and its ssh command does
# not (read in the reviewer's image: PATH, HOME and the user are the same). A
# launch moved from the PTY to ssh carries them, under the image's and the
# agent env's own values.
PTY_SESSION_ENV = {"TERM": "xterm-256color", "SHELL": "/bin/bash"}
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
    *,
    route: Any,
) -> None:
    # timeout=10 is the server-side exec limit; the extra wait_for bounds the
    # client-side await too, so a dead connection can't hang a teardown path
    # (this runs inside close()'s finally).
    command = _routed(route, f"{_tool(route, 'rm')} -f {shlex.quote(remote_env_path)}")
    _transport.sent("Daytona", command)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(
            sandbox.process.exec(
                command,
                env=_empty_loader_env(),
                timeout=10,
            ),
            timeout=30,
        )


async def _bootstrap_daytona_script_file(
    sandbox: Any,
    *,
    route: Any,
    remote_script_path: str,
    script: str,
    error_label: str,
) -> None:
    delimiter = f"__BENCHFLOW_SCRIPT_{uuid.uuid4().hex}__"
    remote_script_path_q = shlex.quote(remote_script_path)
    # A fresh name, created exclusively (set -C), with the route's tools.
    command = (
        _staging_prefix(remote_script_path) + "umask 077; set -C\n"
        f"{_tool(route, 'cat')} > {remote_script_path_q} <<'{delimiter}'\n"
        f"{script}"
        f"{delimiter}\n"
        f"{_tool(route, 'chmod')} 700 {remote_script_path_q}\n"
        f"echo {_BOOTSTRAP_DONE}\n"
    )
    command = _routed(route, command)
    # The live launcher's script is constant text around the agent's launch
    # line; the transport bound holds here too (sandbox/_transport.py).
    _transport.sent("Daytona", command)
    response = await sandbox.process.exec(
        command,
        env=_empty_loader_env(),
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
    route: Any,
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
        route=route,
    )
    _transport.sent("Daytona", command)
    response = await sandbox.process.exec(
        command, env=_empty_loader_env(env), timeout=30
    )
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
        route: Any = None,
        bf_env: Any = None,
        terminal_env: dict[str, str] | None = None,
    ):
        self._bootstrap_env: dict[str, str] = {}
        # PTY_SESSION_ENV for a launch moved from the PTY (a reviewer's), on a
        # direct sandbox: compose's exec gives the agent no terminal either way.
        self._terminal_env = {} if is_dind else dict(terminal_env or {})
        self._sandbox = sandbox
        # BenchFlow's sandbox (its exec route), for the terminal check.
        self._bf_env = bf_env
        self._is_dind = is_dind
        # Root's route in the task container (C8): the launch bootstrap's
        # PATH and bash. The env and script files are the task sandbox's own
        # on a direct sandbox (the same route's tools), the DinD VM's
        # otherwise (its system tools).
        self._route = route
        self._file_route = None if is_dind else route
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
    async def from_sandbox_env(
        cls, env: Any, *, terminal_env: dict[str, str] | None = None
    ) -> DaytonaProcess:
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

        process = cls(
            sandbox=sandbox,
            is_dind=is_dind,
            compose_cmd_prefix=compose_cmd_prefix,
            compose_cmd_base=compose_cmd_base,
            route=await launch_route(env),
            bf_env=env,
            terminal_env=terminal_env,
        )
        from benchflow.sandbox._inert_bootstrap import restored_env

        process._bootstrap_env = restored_env(getattr(env, "_strategy", None))
        return process

    @staticmethod
    def _bootstrap_env_command(
        *,
        remote_env_path: str,
        env_keys: list[str],
        shell_exports: bool,
        route: Any = None,
    ) -> str:
        remote_env_path_q = shlex.quote(remote_env_path)
        keys = " ".join(shlex.quote(key) for key in env_keys)
        # Only builtins and tools by path: this runs as root, and the
        # values come in as the command's own environment (C8). The key
        # names were checked against _ENV_KEY_RE.
        read_value = '  eval "value=\\${$key}"\n'
        if shell_exports:
            write_value = (
                read_value + '  printf \'export %s=\' "$key" >> "$env_file"\n'
                '  quote_env_value "$value" >> "$env_file"\n'
                "  printf '\\n' >> \"$env_file\"\n"
            )
        else:
            write_value = (
                read_value + '  printf \'%s=%s\\n\' "$key" "$value" >> "$env_file"\n'
            )
        script = (
            f"env_file={remote_env_path_q}\n"
            "success=0\n"
            "umask 077\n"
            f'trap \'[ "$success" = 1 ] || {_tool(route, "rm")} -f "$env_file"\' EXIT\n'
            # A fresh name, created exclusively.
            'set -C; : > "$env_file" || exit 1; set +C\n'
            "quote_env_value() {\n"
            '  printf "\'"\n'
            f"  printf '%s' \"$1\" | {_tool(route, 'sed')} \"s/'/'\\\\\\\\''/g\"\n"
            '  printf "\'"\n'
            "}\n"
            f"for key in {keys}; do\n"
            f"{write_value}"
            "done\n"
            "success=1\n"
            f"echo {_BOOTSTRAP_DONE}\n"
        )
        staged = _staging_prefix(remote_env_path) + script
        if route is None:
            return f"{SYSTEM_SHELL} -c {shlex.quote(staged)}"
        return _routed(route, staged)

    async def _cleanup_remote_env_file(self, remote_env_path: str) -> None:
        await _cleanup_daytona_remote_env_file(
            self._sandbox, remote_env_path, route=self._file_route
        )

    async def _bootstrap_env_file(
        self,
        *,
        remote_env_path: str,
        env: dict[str, str],
        shell_exports: bool,
    ) -> None:
        await _bootstrap_daytona_env_file(
            self._sandbox,
            route=self._file_route,
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
        # The launch env's loader and start-up values travel under inert
        # names and become real only after the drop (C8, PT-C8-3).
        env = inert_task_env(
            {
                **getattr(self, "_terminal_env", {}),
                **getattr(self, "_bootstrap_env", {}),
                **(env or {}),
            }
        )
        # Lines over the ACP limit are rewritten in the sandbox: Daytona
        # closes the PTY websocket (1008) on a multi-MB message (#1138). The
        # filter runs after the privilege drop (root_launch_script).
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
                remote_env_path = f"/tmp/runtime_env_{uuid.uuid4().hex[:16]}.env"
                await self._bootstrap_env_file(
                    remote_env_path=remote_env_path,
                    env=env,
                    shell_exports=True,
                )
                for key in env:
                    inner_parts.extend(["--env", key])
            # Root's shell in the task container: /bin/bash with the image's
            # loader variables empty and the system PATH until the drop (C8).
            inner_parts.extend(_compose_empty_loader_flags(env))
            script = root_launch_script(
                command,
                filter_lines=True,
                loader_cleared=True,
                path=launch_path(self._route),
            )
            inner_parts.extend(
                ["main", *root_launch_argv(script, launch_bash(self._route))]
            )
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
            # Direct sandbox — run command via SSH. Daytona creates that root
            # session itself: refuse it unless what it reads at start is
            # root's alone (PY8-PT-R8-8).
            await protect_root_terminal_home(self._bf_env, "Daytona's SSH session")
            await require_protected_root_terminal(self._bf_env, "Daytona's SSH session")
            # Write env vars to a file on the remote host and source it,
            # instead of passing as `env K=V` args visible in ps aux.
            if env:
                # Python-generated unique suffix; see DinD branch above for why
                # $$ shell expansion is fragile across quoting boundaries.
                remote_env_path = _stage_path(
                    self._file_route, f"/tmp/runtime_env_{uuid.uuid4().hex[:16]}.env"
                )
                await self._bootstrap_env_file(
                    remote_env_path=remote_env_path,
                    env=env,
                    shell_exports=True,
                )
            # The SSH session's shell is root's: the shared bootstrap keeps
            # the task's environment and pins the system PATH with builtins
            # before anything else, and removes the env file itself (C8).
            remote_cmd = root_launch_script(
                command,
                env_path=remote_env_path,
                cwd=cwd,
                filter_lines=True,
                path=launch_path(self._route),
            )

        try:
            ssh_access = await self._sandbox.create_ssh_access(
                expires_in_minutes=_DAYTONA_SSH_ACCESS_TTL_MINUTES
            )
            ssh_config_path = self._write_ssh_config(ssh_access.token)
            self._ssh_config_path = ssh_config_path
            cmd = self._ssh_args(ssh_config_path, remote_cmd)
            _transport.sent("Daytona", cmd)

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

    def __init__(
        self,
        sandbox: Any,
        compose_cmd_prefix: str,
        compose_cmd_base: str,
        route: Any = None,
    ):
        self._bootstrap_env: dict[str, str] = {}
        self._sandbox = sandbox
        # As DaytonaProcess: root's route in the task container, and the
        # route of the files this launch stages (None on the DinD VM).
        self._route = route
        self._file_route = None if compose_cmd_base else route
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
        process = cls(
            sandbox=sandbox,
            compose_cmd_prefix=compose_env,
            compose_cmd_base=compose_cmd_base,
            route=await launch_route(env),
        )
        from benchflow.sandbox._inert_bootstrap import restored_env

        process._bootstrap_env = restored_env(getattr(env, "_strategy", None))
        return process

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
            route=self._file_route,
            remote_env_path=remote_env_path,
            env=env,
            shell_exports=True,
            error_label="Daytona PTY agent env",
        )

    async def _cleanup_started_env_file(self) -> None:
        remote_env_path = self._remote_env_path
        self._remote_env_path = None
        if remote_env_path:
            await _cleanup_daytona_remote_env_file(
                self._sandbox, remote_env_path, route=self._file_route
            )
        remote_script_path = self._remote_script_path
        self._remote_script_path = None
        if remote_script_path:
            await _cleanup_daytona_remote_env_file(
                self._sandbox, remote_script_path, route=self._file_route
            )

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
        # The launch env's loader and start-up values travel under inert
        # names and become real only after the drop (C8, PT-C8-3).
        env = inert_task_env({**getattr(self, "_bootstrap_env", {}), **(env or {})})
        # Lines over the ACP limit are rewritten in the sandbox: Daytona
        # closes the PTY websocket (1008) on a multi-MB message (#1138). The
        # filter runs after the privilege drop (root_launch_script).
        remote_env_path = None
        session_id = f"acp-{uuid.uuid4().hex[:8]}"
        pty_env = {}
        if self._compose_cmd_prefix:
            for part in shlex.split(self._compose_cmd_prefix):
                if "=" in part:
                    k, v = part.split("=", 1)
                    pty_env[k] = v
        if not self._compose_cmd_base:
            # Root's terminal in the task sandbox: created with the protected
            # HOME and no startup hook, before any startup file is read
            # (PY8-PT-R8-8).
            pty_env.update(terminal_env(self._route))

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
                remote_env_path = _stage_path(
                    self._file_route, f"/tmp/runtime_env_{uuid.uuid4().hex[:16]}.env"
                )
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
                # The DinD VM's own shell runs compose; root's shell in the
                # task container gets the shared bootstrap (C8).
                exec_parts.extend(_compose_empty_loader_flags(env))
                script = root_launch_script(
                    command,
                    login=True,
                    filter_lines=True,
                    loader_cleared=True,
                    path=launch_path(self._route),
                )
                exec_parts.extend(
                    ["main", *root_launch_argv(script, launch_bash(self._route))]
                )
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
                # Root's own sandbox: the shared bootstrap sources and removes
                # the env file and drops privileges with system tools (C8).
                setup_exec_cmd = root_launch_script(
                    command,
                    env_path=remote_env_path,
                    cwd=cwd,
                    login=True,
                    filter_lines=True,
                    path=launch_path(self._route),
                )

            remote_script_path = _stage_path(
                self._file_route,
                f"/tmp/runtime_pty_exec_{uuid.uuid4().hex[:16]}.sh",
            )
            self._remote_script_path = remote_script_path
            await _bootstrap_daytona_script_file(
                self._sandbox,
                route=self._file_route,
                remote_script_path=remote_script_path,
                script=(
                    "#!/bin/sh\n"
                    "set -e\n"
                    f"{_tool(self._file_route, 'rm')} -f {shlex.quote(remote_script_path)}\n"
                    f"{setup_exec_cmd}\n"
                ),
                error_label="Daytona PTY agent command",
            )
            # Typed into root's terminal: on a direct sandbox, keep the task's
            # environment and pin the system PATH before /bin/sh starts.
            setup_exec_cmd = (
                f"exec {_shell(self._file_route)} {shlex.quote(remote_script_path)}"
            )
            if not self._compose_cmd_base:
                setup_exec_cmd = (
                    f"{typed_root_prelude(launch_path(self._route))}; {setup_exec_cmd}"
                )

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
            # On a direct sandbox this terminal is root's: the keep-and-pin
            # line comes before stty, the first program it starts (PT-C8-4).
            # The terminal's stty by path: the protected kit's on a direct
            # sandbox with one (C8).
            stty = _tool(self._file_route, "stty")
            stty_prefix = (
                ""
                if self._compose_cmd_base
                else f"{typed_root_prelude(launch_path(self._route))}; "
            )
            setup_line = (
                f"{stty_prefix}{stty} raw -echo 2>/dev/null || "
                f"{stty} -echo -icanon min 1 time 0 2>/dev/null || true; "
                f"echo '{marker}'\n"
            )
            _transport.sent("Daytona", setup_line)
            await self._pty.send_input(setup_line)
            logger.info("DaytonaPtyProcess: sent setup, waiting for marker...")

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._next_line(),
                        timeout=self._START_MARKER_TIMEOUT_SEC,
                    )
                    cleaned = _ANSI_CSI_RE_BYTES.sub(
                        b"", _ANSI_OSC_RE_BYTES.sub(b"", line)
                    )
                    decoded = cleaned.decode(errors="replace").strip()
                    logger.debug(f"DaytonaPtyProcess drain: {decoded[:120]}")
                    if marker == decoded:
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
            _transport.sent("Daytona", setup_exec_cmd)
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
            # The local task, not self._pty_gone: close() clears that while a
            # read may still be waiting here.
            if self._closed or gone.done() or _pty_connected(self._pty) is False:
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
