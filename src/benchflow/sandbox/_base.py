"""BaseSandbox — abstract base for sandbox backends.

Internalized from Harbor's BaseEnvironment with RL-first terminology:
- environment -> sandbox
- rollout_paths -> rollout_paths
- EnvironmentConfig -> SandboxConfig
"""

from __future__ import annotations

import logging
import re
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths

logger = logging.getLogger("benchflow")

# Docker Compose service names are restricted to this grammar. Used to filter
# `docker compose config --services` output, whose stdout may be polluted with
# warning lines because the compose command merges stderr into stdout.
_COMPOSE_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _filter_compose_service_names(output: str) -> list[str]:
    """Extract valid compose service names from `config --services` output.

    Drops blank lines and anything that does not match the Docker Compose
    service-name grammar, so a warning line merged into stdout cannot be
    mistaken for a service (#248).
    """
    return [
        line
        for raw in output.splitlines()
        if (line := raw.strip()) and _COMPOSE_SERVICE_NAME_RE.match(line)
    ]


class ExecResult(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    return_code: int


class BaseSandbox(ABC):
    """Abstract base for sandbox environments (Docker, Daytona, Modal).

    Provides the containerized execution environment for agent rollouts.
    """

    environment_dir: Path
    environment_name: str
    session_id: str
    rollout_paths: RolloutPaths | None
    task_env_config: SandboxConfig
    logger: logging.Logger
    default_user: str | int | None

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        _logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        suppress_override_warnings: bool = False,
        persistent_env: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.rollout_paths = rollout_paths
        self.default_user = None
        self.task_env_config = task_env_config

        self._override_cpus = override_cpus
        self._override_memory_mb = override_memory_mb
        self._override_storage_mb = override_storage_mb
        self._override_gpus = override_gpus
        self._suppress_override_warnings = suppress_override_warnings
        self._persistent_env: dict[str, str] = persistent_env or {}

        self.logger = (_logger or logger).getChild(type(self).__name__)

        self._maybe_override_task_env_config()
        self._maybe_resolve_task_env()
        self._validate_definition()

    @property
    def _uses_compose(self) -> bool:
        return False

    def _maybe_resolve_task_env(self) -> None:
        if self.task_env_config.env and not self._uses_compose:
            resolved = resolve_env_vars(self.task_env_config.env)
            self._persistent_env = {**resolved, **self._persistent_env}

    def _maybe_override_task_env_config(self) -> None:
        if self._override_cpus is not None:
            self.task_env_config.cpus = self._override_cpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding CPU count to %d alters the task from its "
                    "intended configuration.",
                    self._override_cpus,
                )
        if self._override_memory_mb is not None:
            self.task_env_config.memory_mb = self._override_memory_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding memory to %d MB alters the task from its "
                    "intended configuration.",
                    self._override_memory_mb,
                )
        if self._override_storage_mb is not None:
            self.task_env_config.storage_mb = self._override_storage_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding storage to %d MB alters the task from its "
                    "intended configuration.",
                    self._override_storage_mb,
                )
        if self._override_gpus is not None:
            self.task_env_config.gpus = self._override_gpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    "Overriding GPU count to %d alters the task from its "
                    "intended configuration.",
                    self._override_gpus,
                )

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        return user if user is not None else self.default_user

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self._persistent_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        return merged or None

    @abstractmethod
    def _validate_definition(self) -> None: ...

    @classmethod
    @abstractmethod
    def preflight(cls) -> None:
        """Check that required credentials/config are available."""
        ...

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        """Run a command inside the sandbox.

        ``service`` selects which compose service (container) the command
        runs in. The default ``"main"`` is the agent container. Multi-
        container (vulhub-style) tasks define additional services in the
        task's ``docker-compose.yaml`` and target them via this argument
        — for flag injection into a vulnerable target before the agent
        runs, or target-side verification afterwards (#248). Sandbox
        backends without compose support reject non-``main`` values.
        """
        ...

    async def exec_in_service(
        self,
        service: str,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run a command in a named compose service — ergonomic wrapper around ``exec``.

        Sugar for ``exec(command, ..., service=service)``. Useful for
        multi-container tasks where verifier code needs to inspect a
        target container's state, e.g.::

            await sandbox.exec_in_service("target", "test -f /tmp/pwned")

        See #248.
        """
        return await self.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
        )

    async def services(self) -> list[str]:
        """List the compose service (container) names available in this sandbox.

        Multi-container (vulhub-style) tasks declare extra services in their
        ``docker-compose.yaml`` alongside the agent's ``main`` container; this
        enumerates them so verifier code can target each via
        ``exec(..., service=...)`` (#248).

        Single-container backends (Modal, direct Daytona) have no compose
        topology and raise a clear error — overriding backends must implement
        this themselves.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a single-container backend; "
            "services() requires a multi-container docker-compose task. "
            "Use the Docker sandbox or the Daytona DinD (compose) sandbox "
            "for vulhub-style tasks (#248)."
        )

    async def is_dir(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user, service=service
        )
        return result.return_code == 0

    async def is_file(
        self, path: str, user: str | int | None = None, service: str = "main"
    ) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user, service=service
        )
        return result.return_code == 0

    async def attach(self) -> None:
        raise NotImplementedError("This environment does not support attaching.")
