"""Live stdio to a container managed by local Docker Compose."""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import os
import shlex
from collections.abc import AsyncIterator, Callable
from typing import Any

from benchflow.sandbox import _transport
from benchflow.sandbox._base import system_shell_env_flags
from benchflow.sandbox._root_launch import (
    inert_task_env,
    launch_bash,
    launch_path,
    launch_route,
    root_launch_argv,
    root_launch_script,
)
from benchflow.sandbox.process._base import (
    _BUFFER_LIMIT,
    _ENV_KEY_RE,
    _STDERR_TAIL_LIMIT,
    SubprocessLiveProcess,
    read_whole_line,
)

logger = logging.getLogger(__name__)
_STDERR_LINE_TOO_LONG = "<Docker stderr line exceeded diagnostic limit>\n"


async def redacted_stderr_lines(
    stderr: asyncio.StreamReader, redact: Callable[[str], str]
) -> AsyncIterator[bytes]:
    """Redact complete diagnostics before exposing or retaining their tail.

    read_whole_line retains context across read limits and chunk boundaries,
    including EOF without a newline. An oversized line is consumed and
    replaced as a unit, so no suffix can lose its redaction context. The
    10 MiB cap bounds memory even when task stderr never sends a newline.
    """
    while True:
        line = await read_whole_line(stderr, cap=_BUFFER_LIMIT)
        if line == b"":
            return
        if line is None:
            # A stateful remote redactor must learn that the context was
            # lost, so later lines cannot escape an oversized envelope.
            yield redact(_STDERR_LINE_TOO_LONG).encode()
        else:
            yield redact(line.decode(errors="replace")).encode()


class DockerProcess(SubprocessLiveProcess):
    """Live stdin/stdout via `docker compose exec -i`."""

    _sandbox: Any = None

    def __init__(
        self,
        project_name: str,
        project_dir: str,
        compose_files: list[str],
        service: str = "main",
        client_env: dict[str, str] | None = None,
        redact: Callable[..., str] | None = None,
        global_args: list[str] | None = None,
        redact_factory: Callable[[], Callable[..., str]] | None = None,
        sandbox: Any = None,
    ):
        # The sandbox whose exec route gives root's launch its shell and
        # PATH (C8); None for a bare process (tests), which keeps the
        # system's.
        self._sandbox = sandbox
        self._bootstrap_env: dict[str, str] = {}
        self._project_name = project_name
        self._project_dir = project_dir
        self._compose_files = compose_files
        self._service = service
        # Set for a remote daemon: the exact environment that reaches it.
        self._client_env = client_env
        # Set for a remote daemon: the docker CLI flags that address it
        # (``docker <flags> compose``), as every other Compose call gives them.
        self._global_args = list(global_args or [])
        # Set for a remote daemon: hides its ssh user (often a token) from the
        # stderr this process keeps and the errors it raises.
        self._redact = redact
        self._redact_factory = redact_factory

    def _redacted(self, text: str) -> str:
        if self._redact is None:
            return text
        return self._redact(text, command_output=True)

    async def _drain_stderr(self, stderr: asyncio.StreamReader | None) -> None:
        if self._redact is None:
            await super()._drain_stderr(stderr)
        elif stderr is not None:
            # Each subprocess stream owns its taint, including when this
            # DockerProcess object is reused after a failed setup/start.
            redact = (
                functools.partial(self._redact_factory(), command_output=True)
                if self._redact_factory is not None
                else self._redacted
            )
            async for line in redacted_stderr_lines(stderr, redact):
                self._stderr_tail.extend(line)
                del self._stderr_tail[:-_STDERR_TAIL_LIMIT]

    @property
    def stderr_tail(self) -> str:
        """The base class's stderr tail; for a remote daemon without its ssh
        user (docker's ssh errors share this stream with the agent's)."""
        return self._redacted(super().stderr_tail)

    @classmethod
    def from_sandbox_env(cls, env: Any, service: str = "main") -> DockerProcess:
        """Create from a sandbox environment (DockerSandbox).

        ``service`` selects which compose service the agent process runs
        in. Defaults to ``"main"``. Multi-container (vulhub-style) tasks
        may run the agent in a dedicated attacker container — e.g. a Kali
        service — while keeping target containers separate (#248).
        """
        # The sandbox's own project (DockerSandbox.compose_project_name);
        # an environment without one names its project after the session.
        project_name = getattr(env, "compose_project_name", None)
        if not isinstance(project_name, str) or not project_name:
            project_name = env.session_id.lower().replace(".", "-")
        # The frozen launch's folder once the sandbox froze its project.
        project_dir = str(
            env.compose_project_dir
            if isinstance(getattr(type(env), "compose_project_dir", None), property)
            else env.environment_dir.resolve().absolute()
        )
        compose_files = [str(p.resolve().absolute()) for p in env._docker_compose_paths]
        host = getattr(env, "docker_host", None)
        redact = getattr(host, "redact", None)
        stream_redactor = getattr(host, "stream_redactor", None)
        # A remote daemon's sandbox gives the flags and environment of its
        # own Compose calls (RemoteDockerSandbox._compose_client), never the
        # raw docker calls' shim environment: one transport for every call.
        # Looked up on the class: only a sandbox that defines it has one.
        if callable(getattr(type(env), "_compose_client", None)):
            global_args, client_env = env._compose_client()
        else:
            raw = getattr(env, "_docker_client_env", None)
            global_args, client_env = [], raw() if callable(raw) else None
        process = cls(
            project_name=project_name,
            project_dir=project_dir,
            compose_files=compose_files,
            service=service,
            client_env=client_env,
            redact=redact if callable(redact) else None,
            redact_factory=stream_redactor if callable(stream_redactor) else None,
            global_args=global_args,
            sandbox=env,
        )
        from benchflow.sandbox._inert_bootstrap import restored_env

        process._bootstrap_env = restored_env(env, service)
        return process

    def _compose_cmd(self) -> list[str]:
        """Base docker compose command with project/file flags."""
        cmd = [
            "docker",
            *self._global_args,
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
        if self._client_env is not None:
            return dict(self._client_env)
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

    async def _launch_route(self) -> Any:
        """Root's route for this launch (C8): its shell, PATH and tools, the
        protected kit's once the sandbox has one; None without a sandbox."""
        return await launch_route(self._sandbox, self._service)

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # The launch env's loader and start-up values travel under inert
        # names and become real only after the drop (C8, PT-C8-3).
        env = inert_task_env({**getattr(self, "_bootstrap_env", {}), **(env or {})})
        proc_env = await asyncio.to_thread(self._host_env)
        route = await self._launch_route()

        # The first stdin line belongs to the transport, before ACP starts.
        # No credential ever touches a container pathname or a process argv:
        # static aliases and failed writes cannot leave an unledgered copy.
        # Bash read consumes exactly one line, leaving later ACP bytes intact.
        payload = None
        if env:
            if any(not _ENV_KEY_RE.fullmatch(key) for key in env):
                raise ValueError("Invalid Docker environment variable name")
            exports = "".join(
                f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
            )
            payload = base64.b64encode(exports.encode()) + b"\n"
        # Root's shell has the image's loader variables empty and the route's
        # PATH until the privilege drop (C8); the env line is applied there
        # with PATH unset (root_launch_script's env_stdin). Lines over the ACP
        # limit are rewritten in the container (#1138), as the agent user.
        script = root_launch_script(
            command,
            env_stdin=payload is not None,
            filter_lines=True,
            loader_cleared=True,
            path=launch_path(route),
        )

        cmd = self._compose_cmd()
        cmd.extend(["exec", "-i", "-T", *system_shell_env_flags()])
        if cwd:
            cmd.extend(["-w", cwd])
        # Root's shell is the route's bash (the protected kit's once the
        # sandbox has one), by absolute path, with -p as integ's bootstrap
        # had it: neither image PATH lookup nor BASH_ENV/imported functions
        # may select the credential reader (C8).
        argv = root_launch_argv(script, launch_bash(route))
        cmd.extend([self._service, *argv[:-2], "-p", *argv[-2:]])

        # Global flags hold the remote login, endpoint and TLS file paths.
        # The command can also hold task secrets: log neither argv slice.
        logger.debug("DockerProcess: starting Compose exec")
        _transport.sent("Docker", cmd)
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                limit=_BUFFER_LIMIT,
            )
            self._set_process(process)
            if payload is not None:
                assert process.stdin is not None
                process.stdin.write(payload)
                await asyncio.wait_for(process.stdin.drain(), timeout=30)
        except BaseException:
            # Includes cancellation/partial handoff. There is no file to
            # recover; close the owned transport before returning failure.
            await asyncio.shield(self.close())
            raise
        logger.info(
            f"Docker process started (pid={process.pid}, project={self._project_name})"
        )
