"""Live stdio through an Apptainer instance."""

from __future__ import annotations

import asyncio
import logging
import shlex
import uuid
from typing import Any

from benchflow.sandbox.process._base import (
    _BUFFER_LIMIT,
    _ENV_KEY_RE,
    SubprocessLiveProcess,
)

logger = logging.getLogger(__name__)


class ApptainerProcess(SubprocessLiveProcess):
    """Bidirectional process transport backed by ``apptainer exec``."""

    def __init__(self, instance_name: str, default_cwd: str | None = None):
        self._instance_name = instance_name
        self._default_cwd = default_cwd
        self._env_path = f"/tmp/.benchflow_agent_env_{uuid.uuid4().hex[:16]}"

    @classmethod
    def from_sandbox_env(cls, env: Any) -> ApptainerProcess:
        name = getattr(env, "_instance_name", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError("Apptainer sandbox not started")
        return cls(name, getattr(env, "_default_cwd", None))

    async def _write_env(self, env: dict[str, str]) -> None:
        invalid = [key for key in env if not _ENV_KEY_RE.match(key)]
        if invalid:
            raise ValueError(
                "Invalid environment variable name(s): " + ", ".join(sorted(invalid))
            )
        body = "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
        )
        process = await asyncio.create_subprocess_exec(
            "apptainer",
            "exec",
            "--cleanenv",
            f"instance://{self._instance_name}",
            "sh",
            "-c",
            f"cat > {shlex.quote(self._env_path)} && "
            f"chmod 600 {shlex.quote(self._env_path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(body.encode()), timeout=30
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to stage Apptainer process environment: "
                + stderr.decode(errors="replace")[:500]
            )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if env:
            await self._write_env(env)
            env_path = shlex.quote(self._env_path)
            command = f". {env_path} && rm -f {env_path} && {command}"
        args = ["apptainer", "exec", "--cleanenv"]
        effective_cwd = cwd or self._default_cwd
        if effective_cwd:
            args.extend(["--cwd", effective_cwd])
        args.extend([f"instance://{self._instance_name}", "sh", "-c", command])
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_BUFFER_LIMIT,
        )
        self._set_process(process)
        logger.info(
            "Apptainer process started (pid=%s, instance=%s)",
            process.pid,
            self._instance_name,
        )
