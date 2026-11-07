"""Live stdio through Apple Container's native exec transport."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import uuid
from typing import Any

from benchflow._root_only import root_only_path_check
from benchflow._runtime_tree import RUN_DIR
from benchflow.sandbox import _transport
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    SYSTEM_SHELL,
    system_shell_env_flags,
    with_system_path,
)
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
    SubprocessLiveProcess,
)

logger = logging.getLogger(__name__)


class AppleContainerProcess(SubprocessLiveProcess):
    """Live stdin/stdout through Apple Container's native exec transport."""

    _route: Any = None
    _sandbox: Any = None

    def __init__(self, container_name: str, sandbox: Any = None):
        self._container_name = container_name
        # A fresh name, created exclusively; in BenchFlow's private staging
        # folder once the sandbox has its protected kit (C8, PT-C8-R3-8).
        self._env_path = f"/tmp/.runtime_agent_env_{uuid.uuid4().hex[:16]}"
        # The sandbox whose exec route gives root's launch its shell and PATH.
        self._sandbox = sandbox
        self._route: Any = None

    @classmethod
    def from_sandbox_env(cls, env: Any) -> AppleContainerProcess:
        """Create from a started AppleContainerSandbox."""

        container_name = getattr(env, "_container_name", None)
        if not isinstance(container_name, str) or not container_name:
            raise RuntimeError("Apple Container sandbox not started")
        return cls(container_name, sandbox=env)

    def _route_shell(self) -> str:
        return getattr(self._route, "shell", SYSTEM_SHELL)

    def _route_path(self) -> str:
        return getattr(self._route, "pin", None) or SYSTEM_PATH

    async def _write_env_to_container(self, env: dict[str, str]) -> None:
        invalid = [key for key in env if not _ENV_KEY_RE.match(key)]
        if invalid:
            raise ValueError(
                "Invalid environment variable name(s): " + ", ".join(sorted(invalid))
            )

        lines = "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
        )
        folder = ""
        if getattr(self._route, "kit", False):
            self._env_path = f"{RUN_DIR}/env-{uuid.uuid4().hex}"
            run_dir = shlex.quote(RUN_DIR)
            folder = (
                f"mkdir -p {run_dir} && chown 0:0 {run_dir} && chmod 700 {run_dir} && "
                + root_only_path_check(RUN_DIR, "BenchFlow's private staging folder")
                + " && "
            )
        env_path = shlex.quote(self._env_path)
        argv = [
            "container",
            "exec",
            "--interactive",
            "--user",
            "root",
            *system_shell_env_flags(),
            self._container_name,
            self._route_shell(),
            "-c",
            with_system_path(
                f"umask 077 && {folder}(set -C; cat > {env_path}) && chmod 600 {env_path}",
                self._route_path(),
            ),
        ]
        _transport.sent("apple-container", argv)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(lines.encode()), timeout=30
            )
        except TimeoutError:
            # Killing a child that finished inside the timeout window raises an
            # empty ProcessLookupError over the TimeoutError below (#1065).
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to write agent env in Apple container "
                f"(rc={proc.returncode}): {stderr.decode(errors='replace')[:500]}"
            )

    async def _remove_env_from_container(self) -> None:
        """Remove a staged env file when the live process cannot start."""
        env_path = shlex.quote(self._env_path)
        argv = [
            "container",
            "exec",
            "--user",
            "root",
            *system_shell_env_flags(),
            self._container_name,
            self._route_shell(),
            "-c",
            with_system_path(f"rm -f {env_path}", self._route_path()),
        ]
        _transport.sent("apple-container", argv)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to remove staged Apple container env file "
                f"(rc={proc.returncode}): {stderr.decode(errors='replace')[:500]}"
            )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        # The launch env's loader and start-up values travel under inert
        # names and become real only after the drop (C8, PT-C8-3).
        env = inert_task_env(env)
        self._route = await launch_route(self._sandbox)
        if env:
            await self._write_env_to_container(env)
        # Root's shell is the route's bash (the protected kit's once the
        # sandbox has one) with the image's loader variables empty and the
        # route's PATH until the privilege drop (C8).
        script = root_launch_script(
            command,
            env_path=self._env_path if env else None,
            loader_cleared=True,
            path=launch_path(self._route),
        )

        args = ["container", "exec", "--interactive", *system_shell_env_flags()]
        if cwd:
            args.extend(["--workdir", cwd])
        args.extend(
            [self._container_name, *root_launch_argv(script, launch_bash(self._route))]
        )
        _transport.sent("apple-container", args)
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_BUFFER_LIMIT,
            )
            self._set_process(process)
        except BaseException:
            if env:
                try:
                    await asyncio.shield(self._remove_env_from_container())
                except Exception:
                    logger.warning(
                        "Could not remove staged Apple container env after launch failure",
                        exc_info=True,
                    )
            raise
        logger.info(
            "Apple Container process started (pid=%s, container=%s)",
            process.pid,
            self._container_name,
        )
