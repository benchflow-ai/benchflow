"""Manifest helpers for trainer launches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

TrainComponentRole = Literal["primary", "service"]
TrainComponentStatus = Literal["pending", "running", "succeeded", "failed"]
TrainRunStatus = Literal["pending", "running", "succeeded", "failed"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class CommandRecord:
    id: str
    argv: list[str]
    cwd: str
    env_keys: list[str] = field(default_factory=list)


@dataclass
class TrainComponent:
    name: str
    role: TrainComponentRole
    command_id: str
    status: TrainComponentStatus = "pending"
    logs: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainRunManifest:
    schema_version: int
    run_type: str
    backend: str
    config: str
    work_dir: str
    output_dir: str
    dry_run: bool
    created_at: str
    updated_at: str
    overall_status: TrainRunStatus
    commands: list[CommandRecord]
    components: list[TrainComponent]
    artifacts: dict[str, list[str]] = field(
        default_factory=lambda: {"checkpoints": [], "exported_models": []}
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(path: Path, manifest: TrainRunManifest) -> None:
    manifest.updated_at = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
