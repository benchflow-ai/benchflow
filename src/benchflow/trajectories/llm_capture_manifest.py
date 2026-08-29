"""Typed provenance contract for ``llm_trajectory.jsonl``.

The JSONL filename is intentionally stable across every authentication path.
This sidecar records what the file can actually prove so downstream code never
confuses a native-agent reconstruction with provider-wire traffic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

LLM_TRAJECTORY_FILENAME = "llm_trajectory.jsonl"
LLM_TRAJECTORY_MANIFEST_FILENAME = "llm_trajectory.manifest.json"
LLM_TRAJECTORY_SCHEMA_VERSION = 2


class CaptureStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_MODEL_CALL = "no_model_call"
    CAPTURE_FAILED = "capture_failed"


class CaptureFidelity(StrEnum):
    PROVIDER_WIRE = "provider_wire"
    AGENT_SESSION = "agent_session"
    ACP_PROJECTION = "acp_projection"
    NONE = "none"


class CaptureSource(StrEnum):
    LITELLM_PROXY = "litellm_proxy"
    CLAUDE_OTEL_RAW_BODY = "claude_otel_raw_body"
    CLAUDE_NATIVE_SESSION = "claude_native_session"
    CODEX_NATIVE_SESSION = "codex_native_session"
    ACP_PROJECTION = "acp_projection"
    NONE = "none"


class AuthMode(StrEnum):
    API_KEY = "api_key"
    OAUTH_SUBSCRIPTION = "oauth_subscription"
    UNKNOWN = "unknown"


class LLMTrajectoryManifest(BaseModel):
    """Machine-readable fidelity and lifecycle state for the JSONL artifact."""

    schema_version: int = LLM_TRAJECTORY_SCHEMA_VERSION
    status: CaptureStatus = CaptureStatus.PENDING
    capture_source: CaptureSource = CaptureSource.NONE
    capture_fidelity: CaptureFidelity = CaptureFidelity.NONE
    auth_mode: AuthMode = AuthMode.UNKNOWN
    agent: str
    model: str | None = None
    session_id: str = ""
    exchange_count: int = 0
    request_complete: bool = False
    response_complete: bool = False
    payload_redacted: bool = True
    started_at: datetime
    finished_at: datetime | None = None
    missing_fields: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def initialize_llm_trajectory_artifacts(
    rollout_dir: Path,
    *,
    agent: str,
    model: str | None,
    session_id: str,
    started_at: datetime,
) -> LLMTrajectoryManifest:
    """Create the always-present empty JSONL and its initial sidecar."""

    trajectory_dir = rollout_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / LLM_TRAJECTORY_FILENAME
    if not trajectory_path.exists():
        _atomic_write_text(trajectory_path, "")
    manifest = LLMTrajectoryManifest(
        agent=agent,
        model=model,
        session_id=session_id,
        started_at=started_at,
    )
    write_llm_trajectory_manifest(rollout_dir, manifest)
    return manifest


def write_llm_trajectory_manifest(
    rollout_dir: Path, manifest: LLMTrajectoryManifest
) -> None:
    path = rollout_dir / "trajectory" / LLM_TRAJECTORY_MANIFEST_FILENAME
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    _atomic_write_text(path, payload + "\n")


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
