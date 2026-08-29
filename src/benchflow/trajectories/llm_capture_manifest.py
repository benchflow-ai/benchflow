"""Typed provenance contract for ``llm_trajectory.jsonl``.

The JSONL filename is intentionally stable across every authentication path.
This sidecar records what the file can actually prove so downstream code never
confuses a native-agent reconstruction with provider-wire traffic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from benchflow.trajectories.io import atomic_write_text

LLM_TRAJECTORY_FILENAME = "llm_trajectory.jsonl"
LLM_TRAJECTORY_MANIFEST_FILENAME = "llm_trajectory.manifest.json"
LLM_TRAJECTORY_SCHEMA_VERSION = 2
REPLAY_PROXY_INGRESS_AUDIT_ERROR = (
    "live continuation request was captured at replay-proxy ingress, "
    "before provider transformation"
)
CONTINUATION_SOURCE_AUDIT_ERROR = (
    "source LLM trajectory is not complete provider-wire capture"
)
_OAUTH_AUDIT_MISSING_FIELDS = frozenset(
    {
        "headers",
        "instructions",
        "provider_request",
        "provider_response",
        "provider_response_envelope",
        "system_prompt",
        "tool_definitions",
    }
)
_REPLAY_AUDIT_MISSING_FIELD = "live_provider_request"


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


_OAUTH_AUDIT_CAPTURE_SOURCES = frozenset(
    {
        CaptureSource.CLAUDE_OTEL_RAW_BODY,
        CaptureSource.CLAUDE_NATIVE_SESSION,
        CaptureSource.CODEX_NATIVE_SESSION,
        CaptureSource.ACP_PROJECTION,
    }
)


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
    payload_redacted: bool = False
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
    atomic_write_text(trajectory_path, "")
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
    atomic_write_text(path, payload + "\n")


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
        and manifest.get("payload_redacted") is True
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


def rollout_capture_is_training_grade(
    rollout_dir: Path,
    *,
    exchanges: Sequence[dict[str, Any]],
) -> bool:
    """Apply the canonical training-admission contract for one rollout."""

    return capture_artifact_allows_training(
        read_llm_trajectory_manifest(rollout_dir),
        exchanges=exchanges,
    )


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


def capture_manifest_preserves_audit_completion(manifest: dict[str, Any]) -> bool:
    """Accept only expected, internally complete audit-only capture states."""

    if manifest.get("status") not in {
        CaptureStatus.NO_MODEL_CALL.value,
        CaptureStatus.PARTIAL.value,
    }:
        return False
    raw_errors = manifest.get("errors", [])
    if not isinstance(raw_errors, list) or not all(
        isinstance(error, str) for error in raw_errors
    ):
        return False
    errors = set(raw_errors)
    raw_missing_fields = manifest.get("missing_fields", [])
    if not isinstance(raw_missing_fields, list) or not all(
        isinstance(field, str) for field in raw_missing_fields
    ):
        return False
    missing_fields = set(raw_missing_fields)
    role_captures = _validated_role_captures(manifest)
    if role_captures is None or not _role_captures_match_manifest(
        manifest, role_captures
    ):
        return False
    has_oauth_capture = any(
        _is_oauth_audit_role_capture(capture) for capture in role_captures
    )
    if manifest.get("status") == CaptureStatus.NO_MODEL_CALL.value:
        return bool(
            manifest.get("auth_mode") == AuthMode.OAUTH_SUBSCRIPTION.value
            and not errors
            and not missing_fields
            and all(
                capture.auth_mode is AuthMode.OAUTH_SUBSCRIPTION
                and capture.capture_source is CaptureSource.NONE
                and capture.capture_fidelity is CaptureFidelity.NONE
                and capture.exchange_count == 0
                and capture.request_complete is False
                and capture.response_complete is False
                for capture in role_captures
            )
        )
    if not role_captures or not all(
        _role_capture_preserves_audit_completion(capture) for capture in role_captures
    ):
        return False
    has_replay_capture = any(
        capture.capture_source is CaptureSource.REPLAY_PROXY
        for capture in role_captures
    )
    if not has_replay_capture:
        return bool(
            has_oauth_capture
            and not errors
            and missing_fields.issubset(_OAUTH_AUDIT_MISSING_FIELDS)
        )

    allowed_errors = {REPLAY_PROXY_INGRESS_AUDIT_ERROR}
    allowed_missing_fields = {_REPLAY_AUDIT_MISSING_FIELD}
    has_recorded_replay_capture = any(
        capture.capture_source is CaptureSource.REPLAY_PROXY
        and capture.leg == "recorded"
        for capture in role_captures
    )
    if has_oauth_capture or has_recorded_replay_capture:
        allowed_errors.add(CONTINUATION_SOURCE_AUDIT_ERROR)
    if has_oauth_capture:
        allowed_missing_fields.update(_OAUTH_AUDIT_MISSING_FIELDS)
    return bool(
        REPLAY_PROXY_INGRESS_AUDIT_ERROR in errors
        and errors.issubset(allowed_errors)
        and _REPLAY_AUDIT_MISSING_FIELD in missing_fields
        and missing_fields.issubset(allowed_missing_fields)
        and manifest.get("response_complete") is True
    )


def _validated_role_captures(
    manifest: dict[str, Any],
) -> list[LLMRoleCapture] | None:
    raw_role_captures = manifest.get("role_captures", [])
    if not isinstance(raw_role_captures, list) or any(
        not isinstance(value, dict)
        or not isinstance(value.get("exchange_count"), int)
        or isinstance(value.get("exchange_count"), bool)
        or value["exchange_count"] < 0
        or not isinstance(value.get("request_complete"), bool)
        or not isinstance(value.get("response_complete"), bool)
        for value in raw_role_captures
    ):
        return None
    try:
        return [LLMRoleCapture.model_validate(value) for value in raw_role_captures]
    except ValidationError:
        return None


def _is_oauth_audit_role_capture(capture: LLMRoleCapture) -> bool:
    return bool(
        capture.auth_mode is AuthMode.OAUTH_SUBSCRIPTION
        and capture.capture_source in _OAUTH_AUDIT_CAPTURE_SOURCES
        and capture.capture_fidelity
        in {CaptureFidelity.AGENT_SESSION, CaptureFidelity.ACP_PROJECTION}
        and capture.exchange_count > 0
    )


def _role_captures_match_manifest(
    manifest: dict[str, Any], role_captures: list[LLMRoleCapture]
) -> bool:
    exchange_count = manifest.get("exchange_count")
    if (
        not isinstance(exchange_count, int)
        or isinstance(exchange_count, bool)
        or exchange_count < 0
        or sum(capture.exchange_count for capture in role_captures) != exchange_count
    ):
        return False
    if exchange_count == 0:
        return True
    active_auth_modes = {
        capture.auth_mode for capture in role_captures if capture.exchange_count > 0
    }
    expected_auth_mode = (
        next(iter(active_auth_modes)) if len(active_auth_modes) == 1 else AuthMode.MIXED
    )
    return manifest.get("auth_mode") == expected_auth_mode.value


def _role_capture_preserves_audit_completion(capture: LLMRoleCapture) -> bool:
    if capture.capture_source is CaptureSource.REPLAY_PROXY:
        return bool(
            capture.capture_fidelity is CaptureFidelity.AGENT_SESSION
            and capture.exchange_count > 0
            and capture.request_complete is False
            and capture.response_complete is True
        )
    if _is_oauth_audit_role_capture(capture):
        return True
    return bool(
        capture.capture_source is not CaptureSource.NONE
        and capture.capture_fidelity is not CaptureFidelity.NONE
        and capture.exchange_count > 0
        and capture.request_complete is True
        and capture.response_complete is True
    )
