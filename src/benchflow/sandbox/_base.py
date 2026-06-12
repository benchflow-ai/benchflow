"""BaseSandbox — abstract base for sandbox backends.

Internalized from Harbor's BaseEnvironment with RL-first terminology:
- environment -> sandbox
- rollout_paths -> rollout_paths
- EnvironmentConfig -> SandboxConfig
"""

from __future__ import annotations

import base64
import logging
import re
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from benchflow.sandbox.protocol import SandboxImage, SandboxSnapshotNotSupported
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


# A POSIX shell identifier: a name the shell can `export`. Keys outside this
# grammar (e.g. containing `.` or `-`) are valid process env keys but cannot be
# assigned via `export NAME=...`; sourcing such a line aborts the whole
# `. {env_path}` step, so the user command would never run (PR #323).
_SHELL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def wrap_command_with_env_file(
    env: dict[str, str], command: str, *, env_path_prefix: str
) -> str:
    """Return *command* prefixed to materialize *env* from a sourced file.

    Single canonical home for the secret-redaction wrapper shared by every
    sandbox backend (Docker compose ``exec``, Daytona direct ``exec``, Daytona
    DinD compose ``exec``). Keeping one implementation guarantees identical
    redaction across backends — two drifting copies of this logic was a
    release-quality security risk (issue #412).

    The env vars are base64-encoded into the command string (not visible as
    individual ``KEY=VALUE`` args in ``ps aux`` / provider command audit logs),
    decoded to a mode-0600 file inside the sandbox, and sourced before the real
    command runs. A ``trap ... EXIT`` deletes the temp file unconditionally —
    even if the decode/source step fails — so a failed ``&&`` chain can never
    leave the env file behind.

    Only keys that are valid POSIX shell identifiers are emitted as ``export``
    lines. Keys that are not (e.g. containing ``.`` or ``-``) cannot be assigned
    by the shell — emitting them would make ``. {env_path}`` fail and the user
    command would never run — so they are skipped with a warning rather than
    silently breaking exec (regression guarded — PR #323).

    The ``umask 077`` that protects the env file is scoped to a subshell so it
    does not leak into the user's command — otherwise files the command creates
    would get mode-0600 unexpectedly (PR #323).

    *env_path_prefix* is the temp-file path prefix the caller wants for the
    decoded env file (a unique 16-hex suffix is appended so concurrent
    ``exec()`` calls in one sandbox can't clobber each other's env file). It is
    the only thing that legitimately varies between backends; the redaction and
    command shape are otherwise byte-for-byte identical.
    """
    exportable: dict[str, str] = {}
    skipped: list[str] = []
    for k, v in env.items():
        if _SHELL_IDENTIFIER_RE.match(k):
            exportable[k] = v
        else:
            skipped.append(k)
    if skipped:
        logger.warning(
            "Skipping env var(s) with non-identifier names (cannot be "
            "exported by the shell): %s",
            ", ".join(sorted(skipped)),
        )

    env_body = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in exportable.items())
    encoded = base64.b64encode(env_body.encode()).decode()
    env_path = f"{env_path_prefix}{uuid4().hex[:16]}"
    return (
        f"trap 'rm -f {env_path}' EXIT && "
        f"(umask 077 && printf %s {shlex.quote(encoded)} | base64 -d > "
        f"{env_path}) && set -a && . {env_path} && set +a && "
        f"{command}"
    )


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

    @property
    def is_mounted(self) -> bool:
        """Whether the rollout dir is host-bind-mounted into the sandbox.

        When ``True`` the agent container's verifier output is already visible
        on the host, so the verifier can skip a ``download_dir`` round-trip.
        Backends that run remotely (Daytona, Modal) have no bind mount and
        override this to ``False``. The default is ``False`` so non-mounted
        backends are the safe assumption.
        """
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

    @property
    def sandbox_id(self) -> str | None:
        """Provider-side identifier for this sandbox instance.

        Backends that allocate a remote resource override this to return the
        provider's ID once the sandbox has been created. Today only Daytona
        does (Modal does not expose one); Docker and process backends have no
        provider-side id and keep this ``None``. Used for post-mortem cleanup
        and audit.
        """
        return None

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
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        """Upload a directory into a compose service container.

        ``service`` selects which container receives the files. The default
        ``"main"`` is the agent container. Multi-container (vulhub-style)
        tasks pass a target service so the test-script verifier's ``/tests``
        dir lands in the container being inspected (#248). Single-container
        backends reject non-``main`` values.
        """
        ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        """Download a directory from a compose service container.

        ``service`` selects which container the files come from. The default
        ``"main"`` is the agent container. Multi-container (vulhub-style)
        tasks pass a target service so target-side verifier output — e.g. a
        ``reward.txt`` written by a ``test.sh`` running in the target — can be
        retrieved (#248). Single-container backends reject non-``main`` values.
        """
        ...

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

    # Container-level snapshot/restore (Branch substrate)
    #
    # Part of the Sandbox contract (``docs/architecture.md``): the Branch
    # lifecycle composes container ⊃ environment-state ⊃ agent-session in
    # that order. Backends that cannot snapshot the container (Modal, the
    # Daytona DinD/compose strategy today) leave the defaults in place and
    # raise :class:`SandboxSnapshotNotSupported`. ``Rollout.branch()`` gates
    # on :attr:`supports_snapshot` and fails closed with a clear diagnostic
    # rather than producing a half-consistent checkpoint (#384).

    @property
    def supports_snapshot(self) -> bool:
        """Whether this backend implements container-level snapshot/restore.

        Default ``False`` so new backends are safe-by-default — they must
        opt-in by overriding both this property and ``snapshot``/``restore``.
        """
        return False

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture the current container state as a re-usable image.

        Default implementation raises :class:`SandboxSnapshotNotSupported`;
        Docker and Daytona-direct override this with provider-native commit
        and snapshot APIs respectively.
        """
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots. "
            "Branch requires a provider whose Sandbox can checkpoint the "
            "container layer; see docs/architecture.md, 'The hard part'."
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore the container to a previously captured snapshot.

        Default implementation raises :class:`SandboxSnapshotNotSupported`.
        """
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level restore. "
            "Branch requires a provider whose Sandbox can checkpoint the "
            "container layer; see docs/architecture.md, 'The hard part'."
        )
