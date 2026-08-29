"""Pure trajectory stitching and provenance reconciliation for continuation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow.trajectories.llm_capture_manifest import (
    LLM_TRAJECTORY_SCHEMA_VERSION,
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMRoleCapture,
    LLMTrajectoryManifest,
    capture_artifact_allows_training,
    read_llm_trajectory_manifest,
    successful_exchanges_have_positive_usage,
    write_llm_trajectory_manifest,
)
from benchflow.trajectories.types import LLMExchange, redact_trajectory_obj


def stitched_trajectory_lines(
    original_llm_trajectory: Path, live_exchanges: list[LLMExchange]
) -> list[str]:
    """Build the continuous llm_trajectory: recorded prefix + live suffix.

    The recorded request/response payloads are preserved, while their metadata
    is promoted to the current schema so a newly stitched artifact can never be
    mistaken for sidecar-optional legacy data. The live suffix is redacted on
    the way out.
    """
    lines: list[str] = []
    if original_llm_trajectory.is_file():
        for raw in original_llm_trajectory.read_text().splitlines():
            if raw.strip():
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    lines.append(raw)
                    continue
                if not isinstance(payload, dict):
                    lines.append(raw)
                    continue
                metadata = payload.setdefault("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    payload["metadata"] = metadata
                metadata["schema_version"] = LLM_TRAJECTORY_SCHEMA_VERSION
                lines.append(json.dumps(redact_trajectory_obj(payload), default=str))
    for exchange in live_exchanges:
        payload = redact_trajectory_obj(exchange.model_dump(mode="json"))
        lines.append(json.dumps(payload, default=str))
    return lines


def write_stitched_trajectory(
    rollout_dir: Path,
    original_llm_trajectory: Path,
    live_exchanges: list[LLMExchange],
    *,
    live_model: str | None = None,
    live_capture_trusted: bool = True,
) -> Path:
    """Write the stitched continuous trajectory into the new rollout folder."""
    out = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = stitched_trajectory_lines(original_llm_trajectory, [])
    for exchange in live_exchanges:
        payload = exchange.model_dump(mode="json")
        metadata = payload.setdefault("metadata", {})
        metadata.update(
            {
                "agent": "openhands",
                "role": "agent",
                "model": live_model,
                "auth_mode": AuthMode.API_KEY.value,
                "capture_source": CaptureSource.LITELLM_PROXY.value,
                "capture_fidelity": (
                    CaptureFidelity.PROVIDER_WIRE.value
                    if live_capture_trusted
                    else CaptureFidelity.AGENT_SESSION.value
                ),
                "schema_version": LLM_TRAJECTORY_SCHEMA_VERSION,
                "request_complete": True,
                "response_complete": True,
                "role_attribution_complete": True,
                "payload_redacted": True,
            }
        )
        if not live_capture_trusted:
            metadata["capture_custody"] = "agent_writable_sandbox"
        lines.append(json.dumps(redact_trajectory_obj(payload), default=str))
    rendered = "\n".join(lines) + ("\n" if lines else "")
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(rendered)
    os.replace(temporary, out)
    return out


def refresh_stitched_trajectory_manifest(
    rollout_dir: Path,
    source_rollout_dir: Path,
    *,
    original_model: str | None,
    live_model: str | None,
    n_recorded: int,
    n_live: int,
    live_attempt_count: int,
    live_errors: list[str],
    live_capture_trusted: bool = True,
) -> LLMTrajectoryManifest:
    """Replace rollout-finalization provenance with the final stitched contract."""

    current_raw = read_llm_trajectory_manifest(rollout_dir)
    source_raw = read_llm_trajectory_manifest(source_rollout_dir)
    try:
        current = (
            LLMTrajectoryManifest.model_validate(current_raw)
            if current_raw is not None
            else None
        )
    except ValueError:
        current = None
    try:
        source = (
            LLMTrajectoryManifest.model_validate(source_raw)
            if source_raw is not None
            else None
        )
    except ValueError:
        source = None

    trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    trajectory_lines = [
        line for line in trajectory_path.read_text().splitlines() if line.strip()
    ]
    exchange_count = len(trajectory_lines)
    malformed_count = 0
    for line in trajectory_lines:
        try:
            LLMExchange.model_validate_json(line)
        except ValueError:
            malformed_count += 1
    rows_valid = malformed_count == 0
    parsed_rows: list[dict[str, Any]] = []
    if rows_valid:
        parsed_rows = [json.loads(line) for line in trajectory_lines]
    expected_count = n_recorded + live_attempt_count
    count_matches = exchange_count == expected_count
    source_rows: list[dict[str, Any]] = []
    try:
        source_rows = [
            json.loads(line)
            for line in (source_rollout_dir / "trajectory" / "llm_trajectory.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        source_rows = []
    source_allows_training = bool(
        source_raw is not None
        and len(source_rows) == n_recorded
        and capture_artifact_allows_training(source_raw, exchanges=source_rows)
    )
    live_capture_complete = live_attempt_count == n_live and not live_errors
    usage_complete = bool(
        parsed_rows and successful_exchanges_have_positive_usage(parsed_rows)
    )
    complete = (
        source_allows_training
        and count_matches
        and live_capture_complete
        and (live_capture_trusted or n_live == 0)
        and rows_valid
        and usage_complete
    )

    source_capture_source = source.capture_source if source else CaptureSource.NONE
    source_fidelity = source.capture_fidelity if source else CaptureFidelity.NONE
    source_auth = source.auth_mode if source else AuthMode.UNKNOWN
    if n_live:
        capture_source = (
            CaptureSource.LITELLM_PROXY
            if source_capture_source is CaptureSource.LITELLM_PROXY
            else CaptureSource.MIXED
        )
        live_fidelity = (
            CaptureFidelity.PROVIDER_WIRE
            if live_capture_trusted
            else CaptureFidelity.AGENT_SESSION
        )
        capture_fidelity = (
            live_fidelity if source_fidelity is live_fidelity else CaptureFidelity.MIXED
        )
        auth_mode = (
            AuthMode.API_KEY if source_auth is AuthMode.API_KEY else AuthMode.MIXED
        )
    else:
        capture_source = source_capture_source
        capture_fidelity = source_fidelity
        auth_mode = source_auth

    request_complete = bool(
        source
        and source.request_complete
        and count_matches
        and live_capture_complete
        and rows_valid
    )
    response_complete = bool(
        source
        and source.response_complete
        and count_matches
        and live_capture_complete
        and rows_valid
    )
    errors = list(source.errors) if source and not complete else []
    missing_fields = list(source.missing_fields) if source and not complete else []
    errors.extend(live_errors)
    if n_live and not live_capture_trusted:
        errors.append("sandbox replay capture shared root custody with the agent")
    if source is None:
        errors.append("source LLM trajectory manifest is missing or malformed")
        missing_fields.append("source_capture_provenance")
    elif not source_allows_training:
        errors.append("source LLM trajectory is not complete provider-wire capture")
    if not count_matches:
        errors.append(
            "stitched LLM trajectory count mismatch: "
            f"expected {expected_count}, found {exchange_count}"
        )
        missing_fields.append("exchange_count")
    if not rows_valid:
        errors.append(
            f"stitched LLM trajectory contains {malformed_count} malformed row(s)"
        )
        missing_fields.append("valid_provider_exchange")
    if rows_valid and not usage_complete:
        errors.append("stitched LLM trajectory lacks positive provider token usage")
        missing_fields.append("token_usage")
    if not live_capture_complete:
        missing_fields.append("live_provider_exchange")

    models = {
        value
        for value, present in (
            (original_model, n_recorded > 0),
            (live_model, live_attempt_count > 0),
        )
        if present and value
    }
    stitched_model = next(iter(models)) if len(models) == 1 else None
    manifest = LLMTrajectoryManifest(
        status=CaptureStatus.COMPLETE if complete else CaptureStatus.PARTIAL,
        capture_source=capture_source,
        capture_fidelity=capture_fidelity,
        auth_mode=auth_mode,
        agent="openhands",
        model=stitched_model,
        session_id=current.session_id if current else rollout_dir.name,
        exchange_count=exchange_count,
        request_complete=request_complete,
        response_complete=response_complete,
        payload_redacted=source.payload_redacted if source else True,
        started_at=current.started_at if current else datetime.now(UTC),
        finished_at=datetime.now(UTC),
        missing_fields=sorted(set(missing_fields)),
        errors=errors,
        role_captures=[
            LLMRoleCapture(
                role="agent",
                agent="openhands",
                model=stitched_model,
                auth_mode=auth_mode,
                capture_source=capture_source,
                capture_fidelity=capture_fidelity,
                exchange_count=exchange_count,
                request_complete=request_complete,
                response_complete=response_complete,
            )
        ],
    )
    write_llm_trajectory_manifest(rollout_dir, manifest)
    return manifest
