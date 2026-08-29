"""Typed provenance contract for ``llm_trajectory.jsonl``.

The JSONL filename is intentionally stable across every authentication path.
This sidecar records what the file can actually prove so downstream code never
confuses a native-agent reconstruction with provider-wire traffic.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

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
    MIXED = "mixed"
    NONE = "none"


class CaptureSource(StrEnum):
    LITELLM_PROXY = "litellm_proxy"
    REPLAY_PROXY = "replay_proxy"
    CLAUDE_OTEL_RAW_BODY = "claude_otel_raw_body"
    CLAUDE_NATIVE_SESSION = "claude_native_session"
    CODEX_NATIVE_SESSION = "codex_native_session"
    ACP_PROJECTION = "acp_projection"
    MIXED = "mixed"
    NONE = "none"


class AuthMode(StrEnum):
    API_KEY = "api_key"
    OAUTH_SUBSCRIPTION = "oauth_subscription"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LLMRoleCapture(BaseModel):
    """Per prepared role provenance for mixed-auth/mixed-agent rollouts."""

    role: str = "agent"
    leg: Literal["recorded", "live"] | None = None
    agent: str
    model: str | None = None
    auth_mode: AuthMode
    capture_source: CaptureSource
    capture_fidelity: CaptureFidelity
    exchange_count: int
    request_complete: bool
    response_complete: bool


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
    role_captures: list[LLMRoleCapture] = Field(default_factory=list)


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


def read_llm_trajectory_manifest(rollout_dir: Path) -> dict[str, Any] | None:
    """Read the sidecar, distinguishing absent legacy data from corruption."""

    path = rollout_dir / "trajectory" / LLM_TRAJECTORY_MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "capture_failed", "capture_fidelity": "none"}
    if not isinstance(manifest, dict):
        return {"status": "capture_failed", "capture_fidelity": "none"}
    return manifest


def capture_manifest_allows_training(
    manifest: dict[str, Any], *, exchange_count: int
) -> bool:
    """Return whether manifest fidelity and JSONL cardinality allow training."""

    expected_count = manifest.get("exchange_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        return False

    return bool(
        manifest.get("status") == "complete"
        and manifest.get("capture_fidelity") == "provider_wire"
        and manifest.get("request_complete") is True
        and manifest.get("response_complete") is True
        and exchange_count > 0
        and expected_count == exchange_count
    )


def capture_artifact_allows_training(
    manifest: dict[str, Any] | None,
    *,
    exchanges: Sequence[dict[str, Any]],
) -> bool:
    """Apply the sidecar contract while retaining genuine legacy JSONL support."""

    if manifest is not None:
        return bool(
            capture_manifest_allows_training(
                manifest,
                exchange_count=len(exchanges),
            )
            and successful_exchanges_have_positive_usage(exchanges)
        )
    return not any(_exchange_requires_manifest(exchange) for exchange in exchanges)


def successful_exchanges_have_positive_usage(
    exchanges: Sequence[dict[str, Any]],
) -> bool:
    """Require positive token evidence for every successful provider response."""

    successful = [
        exchange
        for exchange in exchanges
        if isinstance((response := exchange.get("response")), dict)
        and isinstance(response.get("status_code"), int)
        and 200 <= response["status_code"] < 300
    ]
    return bool(successful) and all(
        _exchange_has_positive_usage(exchange) for exchange in successful
    )


def _exchange_has_positive_usage(exchange: dict[str, Any]) -> bool:
    response = exchange.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    if not isinstance(body, dict):
        return False
    for container_name in ("usage", "usageMetadata"):
        container = body.get(container_name)
        if isinstance(container, dict) and _usage_payload_has_positive_tokens(
            container
        ):
            return True
    return False


def _usage_payload_has_positive_tokens(payload: dict[str, Any]) -> bool:
    token_keys = {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "cachedContentTokenCount",
        "toolUsePromptTokenCount",
        "thoughtsTokenCount",
        "cached_tokens",
    }
    for key, value in payload.items():
        if key in token_keys:
            if isinstance(value, bool):
                continue
            try:
                if int(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        if isinstance(value, dict) and _usage_payload_has_positive_tokens(value):
            return True
    return False


def _exchange_requires_manifest(exchange: dict[str, Any]) -> bool:
    metadata = exchange.get("metadata")
    if not isinstance(metadata, dict):
        return False
    schema_version = metadata.get("schema_version")
    return bool(
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version >= LLM_TRAJECTORY_SCHEMA_VERSION
    )


def capture_manifest_has_oauth_role_capture(manifest: dict[str, Any]) -> bool:
    """Return whether a mixed manifest contains captured OAuth role evidence."""

    role_captures = manifest.get("role_captures")
    if not isinstance(role_captures, list):
        return False
    for value in role_captures:
        try:
            role_capture = LLMRoleCapture.model_validate(value)
        except ValidationError:
            continue
        if (
            role_capture.auth_mode is AuthMode.OAUTH_SUBSCRIPTION
            and role_capture.capture_source is not CaptureSource.NONE
            and role_capture.capture_fidelity is not CaptureFidelity.NONE
            and role_capture.exchange_count > 0
        ):
            return True
    return False


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
