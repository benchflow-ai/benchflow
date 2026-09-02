"""Live stdio through Apple Container's native exec transport."""

from __future__ import annotations

import asyncio
import logging
import shlex
import uuid
from typing import Any

from benchflow.sandbox.process._base import (
    _BUFFER_LIMIT,
    _ENV_KEY_RE,
    _RemoteProcessGroupLiveProcess,
)

logger = logging.getLogger(__name__)


class AppleContainerProcess(_RemoteProcessGroupLiveProcess):
    """Live stdin/stdout through Apple Container's native exec transport."""

    def __init__(self, container_name: str):
        self._container_name = container_name
        self._env_path = f"/tmp/.benchflow_agent_env_{uuid.uuid4().hex[:16]}"

    @classmethod
    def from_sandbox_env(cls, env: Any) -> AppleContainerProcess:
        """Create from a started AppleContainerSandbox."""

        container_name = getattr(env, "_container_name", None)
        if not isinstance(container_name, str) or not container_name:
            raise RuntimeError("Apple Container sandbox not started")
        return cls(container_name)

    async def _write_env_to_container(self, env: dict[str, str]) -> None:
        invalid = [key for key in env if not _ENV_KEY_RE.match(key)]
        if invalid:
            raise ValueError(
                "Invalid environment variable name(s): " + ", ".join(sorted(invalid))
            )

        lines = "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
        )
        env_path = shlex.quote(self._env_path)
        proc = await asyncio.create_subprocess_exec(
            "container",
            "exec",
            "--interactive",
            "--user",
            "root",
            self._container_name,
            "sh",
            "-c",
            f"cat > {env_path} && chmod 600 {env_path}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(lines.encode()), timeout=30
            )
        except TimeoutError:
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
        proc = await asyncio.create_subprocess_exec(
            "container",
            "exec",
            "--user",
            "root",
            self._container_name,
            "sh",
            "-c",
            f"rm -f {env_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to remove staged Apple container env file "
                f"(rc={proc.returncode}): {stderr.decode(errors='replace')[:500]}"
            )

    async def _exec_remote_process_group_command(self, command: str) -> int:
        process = await asyncio.create_subprocess_exec(
            "container",
            "exec",
            self._container_name,
            "bash",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        return_code = process.returncode
        if return_code is None:
            return 2
        if return_code not in (0, 1, 2):
            logger.warning(
                "Apple process-group control failed (rc=%s): %s",
                return_code,
                stderr.decode(errors="replace")[:500],
            )
        return return_code

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if env:
            await self._write_env_to_container(env)
            env_path = shlex.quote(self._env_path)
            command = f". {env_path} && rm -f {env_path} && {command}"
        command = self._wrap_remote_process_group(command)

        args = ["container", "exec", "--interactive"]
        if cwd:
            args.extend(["--workdir", cwd])
        args.extend([self._container_name, "bash", "-c", command])
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
            self._remote_process_group_path = None
            raise
        logger.info(
            "Apple Container process started (pid=%s, container=%s)",
            process.pid,
            self._container_name,
        )
