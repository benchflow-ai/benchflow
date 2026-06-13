from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ExecResult(BaseModel):
    """Result of running a command inside a :class:`Sandbox`.

    The single ``ExecResult`` contract — the public type and the type every
    backend (Docker, Daytona, Modal, Cua) actually returns are one and the
    same. ``stdout``/``stderr`` are optional because backends decode empty
    process output to ``None`` rather than ``""``.
    """

    stdout: str | None = None
    stderr: str | None = None
    return_code: int


@dataclass(frozen=True)
class SandboxImage:
    """A provider-opaque handle to a container-level checkpoint.

    The Branch lifecycle (``docs/architecture.md``) composes three snapshot
    layers — container, environment-state, agent-session — and this is the
    container layer's unit of roll-back. Concrete providers carry whatever
    they need to round-trip a snapshot inside ``provider``-scoped fields:

    * Docker:  ``provider="docker"``, ``ref`` is the committed image tag.
    * Daytona: ``provider="daytona"``, ``ref`` is the daytona snapshot name.

    ``ref`` is opaque to the kernel — only the originating provider knows
    how to ``restore`` from it. ``meta`` carries provider-specific extras
    (image digest, parent container id, etc.) for diagnostics.
    """

    provider: str
    ref: str
    meta: dict[str, str] = field(default_factory=dict)


class SandboxSnapshotNotSupported(NotImplementedError):
    """Raised when a Sandbox backend cannot satisfy ``snapshot``/``restore``.

    The Sandbox contract declares snapshot/restore, but not every backend can
    implement them (Daytona DinD/compose, Modal). Callers — notably
    ``Rollout.branch()`` — catch this to fail closed with a clear diagnostic
    when the run requires container-level checkpointing.
    """


class SandboxStartupError(RuntimeError):
    """Raised when sandbox creation fails or times out.

    Lives in the core ``benchflow.sandbox.protocol`` module — and not in any
    provider-specific backend — so a base install of ``benchflow`` (no
    ``sandbox-daytona`` / ``sandbox-modal`` / ``sandbox-cua`` extras) can still import
    ``benchflow.rollout`` and reference this exception type without pulling
    in optional provider SDKs (issue #358). Provider-specific backends
    re-raise this same type with a structured
    :class:`~benchflow.diagnostics.SandboxStartupDiagnostic` for
    ``result.json``.
    """

    def __init__(
        self,
        message: str,
        *,
        sandbox_id: str | None = None,
        sandbox_state: str | None = None,
        attempts: int = 0,
        build_timeout_sec: float | None = None,
    ) -> None:
        super().__init__(message)
        # Local import keeps ``protocol`` cycle-free for any future
        # ``diagnostics`` -> sandbox importers.
        from benchflow.diagnostics import SandboxStartupDiagnostic

        self.diagnostic: SandboxStartupDiagnostic = SandboxStartupDiagnostic(
            sandbox_id=sandbox_id,
            sandbox_state=sandbox_state,
            attempts=attempts,
            build_timeout_sec=build_timeout_sec,
            raw_message=str(message)[:500],
        )


SandboxStartupFailure = SandboxStartupError


@runtime_checkable
class Sandbox(Protocol):
    """Run-only: isolated execution environment.

    All roles in a scene share one Sandbox instance, so inter-agent
    communication over localhost is available by default.

    BenchFlow provides the sandbox infrastructure; it does **not**
    orchestrate agent-internal loops or tool protocols (ENG-50).

    This Protocol is the literal projection of the surface every backend
    shares via :class:`~benchflow.sandbox._base.BaseSandbox` — the method
    signatures (params and defaults) match ``BaseSandbox`` so a backend
    instance passes ``isinstance(sb, Sandbox)``. Capabilities that only some
    backends provide are deliberately **not** part of this contract.

    Roll-back (``snapshot``/``restore``) is part of the contract — Branch
    composes container, environment-state, and agent-session checkpoints in
    that order (``docs/architecture.md``). Backends that cannot snapshot the
    container raise :class:`SandboxSnapshotNotSupported`; callers gate on
    :attr:`supports_snapshot` to fail closed before running.
    """

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None: ...
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None: ...

    async def start(self, force_build: bool) -> None: ...
    async def stop(self, delete: bool) -> None: ...

    # container-level roll-back (the substrate Branch runs on)
    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture the current container state as a re-usable image.

        Raises :class:`SandboxSnapshotNotSupported` on backends without a
        provider-level snapshot primitive. Branchable runs should gate on
        :attr:`supports_snapshot` before calling.
        """
        ...

    async def restore(self, image: SandboxImage) -> None:
        """Restore the container to a previously captured snapshot.

        Raises :class:`SandboxSnapshotNotSupported` on backends without a
        provider-level snapshot primitive.
        """
        ...

    @property
    def supports_snapshot(self) -> bool:
        """Whether this backend implements container-level snapshot/restore.

        Capability gate for ``Rollout.branch()`` — see the Branch lifecycle
        in ``docs/architecture.md``.
        """
        ...


@dataclass(frozen=True)
class ImageRef:
    tag: str
    digest: str | None = None


@dataclass
class ImageConfig:
    dockerfile: Path
    context_dir: Path
    build_args: dict[str, str] | None = None
    cache_key: str | None = None


class ImageBuilder(Protocol):
    """Build-only: produces image refs from Dockerfiles/configs."""

    async def build(self, config: ImageConfig) -> ImageRef: ...
    async def cached(self, config: ImageConfig) -> ImageRef | None: ...
