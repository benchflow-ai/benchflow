"""Sandbox and image value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecResult:
    """Result of a command executed inside a sandbox."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    @property
    def success(self) -> bool:
        return self.return_code == 0


@dataclass(frozen=True)
class ImageRef:
    """Provider-specific immutable image/template/snapshot reference."""

    provider: str
    ref: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageConfig:
    """Build inputs for an image builder."""

    dockerfile: Path | None = None
    docker_image: str | None = None
    build_args: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxSpec:
    """Requested isolated compute for a rollout."""

    provider: str = "docker"
    image: ImageRef | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    allow_internet: bool = True
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
