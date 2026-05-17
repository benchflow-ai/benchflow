"""BaseSandboxEnvironment — abstract base for sandbox backends.

Internalized from Harbor's BaseEnvironment with RL-first terminology:
- environment -> sandbox
- rollout_paths -> rollout_paths
- EnvironmentConfig -> SandboxConfig
"""

from __future__ import annotations

import logging
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchflow.task.config import SandboxConfig
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths

logger = logging.getLogger("benchflow")


class ExecResult(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    return_code: int


class BaseSandboxEnvironment(ABC):
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
    async def download_dir(
        self, source_dir: str, target_dir: Path | str
    ) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def attach(self) -> None:
        raise NotImplementedError("This environment does not support attaching.")
