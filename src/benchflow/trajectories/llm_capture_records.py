"""Pure record assembly for uniform LLM trajectory capture."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.providers.litellm_config import safe_model_alias
from benchflow.trajectories.llm_capture_manifest import (
    LLM_TRAJECTORY_SCHEMA_VERSION,
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMRoleCapture,
    successful_exchanges_have_positive_usage,
)
from benchflow.trajectories.native_capture_parsers import NativeParseResult
from benchflow.trajectories.types import redact_trajectory_obj


@dataclass(frozen=True)
class CaptureTarget:
    """One prepared agent role whose model calls should be attributable."""

    agent: str
    model: str | None
    credential_home: str
    auth_mode: AuthMode
    native: bool
    role: str = "agent"
    native_session_ids: tuple[str, ...] = ()
    provider_capture_trusted: bool = True


@dataclass(frozen=True)
class NativeCaptureBundle:
    """A native parser result plus the roles it may describe."""

    targets: tuple[CaptureTarget, ...]
    result: NativeParseResult


@dataclass(frozen=True)
class CaptureAssembly:
    """Terminal artifact state derived from every available capture source."""

    records: list[dict[str, Any]]
    status: CaptureStatus
    source: CaptureSource
    fidelity: CaptureFidelity
    auth_mode: AuthMode
    request_complete: bool
    response_complete: bool
    missing_fields: list[str]
    errors: list[str]
    role_captures: list[LLMRoleCapture]


def load_provider_wire_records(
    path: Path,
    *,
    targets: list[CaptureTarget],
    fallback_agent: str,
    fallback_model: str | None,
    fallback_auth: AuthMode,
) -> list[dict[str, Any]]:
    """Load and attribute the shared LiteLLM JSONL without guessing roles."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid LLM trajectory JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"invalid LLM trajectory JSONL at line {line_number}: expected object"
            )
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            record["metadata"] = metadata
        target = _target_for_provider_record(targets, record)
        attribution_complete = target is not None or not targets
        capture_trusted = (
            target.provider_capture_trusted if target is not None else True
        )
        request_complete = metadata.get("request_complete") is True
        provider_request_observed = (
            metadata.get("request_capture_source")
            == "litellm_pre_api_call_complete_input_dict"
        )
        metadata.update(
            {
                "schema_version": LLM_TRAJECTORY_SCHEMA_VERSION,
                "capture_source": CaptureSource.LITELLM_PROXY.value,
                "capture_fidelity": (
                    CaptureFidelity.PROVIDER_WIRE.value
                    if capture_trusted and provider_request_observed
                    else CaptureFidelity.AGENT_SESSION.value
                ),
                "auth_mode": (
                    target.auth_mode.value
                    if target is not None
                    else fallback_auth.value
                ),
                "agent": (
                    target.agent
                    if target is not None
                    else (fallback_agent if not targets else "mixed")
                ),
                "role": (
                    target.role
                    if target is not None
                    else ("primary" if not targets else "mixed")
                ),
                "model": (
                    target.model
                    if target is not None
                    else (_record_model(record) or fallback_model)
                ),
                "role_attribution_complete": attribution_complete,
                "request_complete": request_complete,
                "response_complete": True,
                "payload_redacted": True,
            }
        )
        if not capture_trusted:
            metadata["capture_custody"] = "agent_writable_sandbox"
        if not attribution_complete:
            metadata["role_candidates"] = _role_candidates(targets)
        records.append(redact_trajectory_obj(record))
    return records


def assemble_capture(
    *,
    provider_records: list[dict[str, Any]],
    native_bundles: list[NativeCaptureBundle],
    targets: list[CaptureTarget],
    collection_errors: list[str],
    model_call_seen: bool,
    fallback_auth: AuthMode,
) -> CaptureAssembly:
    """Merge all sources and derive one fail-closed terminal contract."""

    native_records = [
        record for bundle in native_bundles for record in _native_bundle_records(bundle)
    ]
    records = _sort_exchange_records([*provider_records, *native_records])
    errors = list(collection_errors)
    if any(
        _record_metadata(record).get("capture_custody") == "agent_writable_sandbox"
        for record in provider_records
    ):
        errors.append(
            "sandbox-local LiteLLM capture shared root custody with the agent"
        )
    attribution_incomplete = any(
        _record_metadata(record).get("role_attribution_complete") is False
        for record in records
    )
    if attribution_incomplete:
        errors.append(
            "one or more exchanges could not be attributed to a unique prepared role"
        )

    captured_targets = {
        *_captured_targets(provider_records, targets),
        *_captured_targets(native_records, targets),
    }
    missing_targets = [target for target in targets if target not in captured_targets]
    errors.extend(
        f"no capture was attributable to {target.agent} "
        f"({target.model or 'unknown model'}, {target.auth_mode.value})"
        for target in missing_targets
    )
    role_captures = _role_captures(records, targets=targets)
    auth_mode = _aggregate_auth_mode(
        records, targets=targets, fallback_auth=fallback_auth
    )

    if not records:
        if model_call_seen and not errors:
            errors.append("model call observed but no LLM capture source was readable")
        return CaptureAssembly(
            records=[],
            status=(
                CaptureStatus.CAPTURE_FAILED
                if model_call_seen
                else CaptureStatus.NO_MODEL_CALL
            ),
            source=CaptureSource.NONE,
            fidelity=CaptureFidelity.NONE,
            auth_mode=auth_mode,
            request_complete=False,
            response_complete=False,
            missing_fields=(
                ["provider_request", "provider_response"] if model_call_seen else []
            ),
            errors=errors,
            role_captures=role_captures,
        )

    missing_fields = {
        field for bundle in native_bundles for field in bundle.result.missing_fields
    }
    if any(
        not _record_metadata_bool(record, "request_complete")
        for record in provider_records
    ):
        missing_fields.add("provider_request")
    successful_records = [
        record
        for record in records
        if isinstance((response := record.get("response")), dict)
        and isinstance(response.get("status_code"), int)
        and 200 <= response["status_code"] < 300
    ]
    usage_complete = not successful_records or successful_exchanges_have_positive_usage(
        successful_records
    )
    if not usage_complete:
        missing_fields.add("token_usage")
    if attribution_incomplete:
        missing_fields.add("role_attribution")
    source = _aggregate_source(records)
    fidelity = _aggregate_fidelity(records)
    if missing_targets:
        source = CaptureSource.MIXED
        fidelity = CaptureFidelity.MIXED
    request_complete = (
        not missing_targets
        and all(_record_metadata_bool(record, "request_complete") for record in records)
        and all(bundle.result.request_complete for bundle in native_bundles)
    )
    response_complete = (
        not missing_targets
        and all(
            _record_metadata_bool(record, "response_complete") for record in records
        )
        and all(bundle.result.response_complete for bundle in native_bundles)
    )
    errors.extend(error for bundle in native_bundles for error in bundle.result.errors)
    status = (
        CaptureStatus.COMPLETE
        if fidelity is CaptureFidelity.PROVIDER_WIRE
        and request_complete
        and response_complete
        and usage_complete
        and not errors
        else CaptureStatus.PARTIAL
    )
    return CaptureAssembly(
        records=records,
        status=status,
        source=source,
        fidelity=fidelity,
        auth_mode=auth_mode,
        request_complete=request_complete,
        response_complete=response_complete,
        missing_fields=sorted(missing_fields),
        errors=errors,
        role_captures=role_captures,
    )


def write_exchange_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace the JSONL with redacted assembled records."""

    payload = "".join(
        json.dumps(redact_trajectory_obj(record), default=str) + "\n"
        for record in records
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def role_captures_for_targets(targets: list[CaptureTarget]) -> list[LLMRoleCapture]:
    """Return explicit zero-exchange provenance for a failed finalization."""

    return _role_captures([], targets=targets)


def _native_bundle_records(bundle: NativeCaptureBundle) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in bundle.result.trajectory.to_jsonl(redact_keys=True).splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            record["metadata"] = metadata
        target = _target_for_record(list(bundle.targets), record)
        if target is not None:
            metadata.update(
                {
                    "agent": target.agent,
                    "role": target.role,
                    "model": target.model or _record_model(record),
                    "auth_mode": target.auth_mode.value,
                    "role_attribution_complete": True,
                }
            )
        elif bundle.targets:
            metadata.update(
                {
                    "agent": "mixed",
                    "role": "mixed",
                    "model": _record_model(record),
                    "auth_mode": AuthMode.MIXED.value,
                    "role_attribution_complete": False,
                    "role_candidates": _role_candidates(list(bundle.targets)),
                }
            )
        records.append(redact_trajectory_obj(record))
    return records


def _role_candidates(targets: list[CaptureTarget]) -> list[dict[str, Any]]:
    return [
        {
            "agent": target.agent,
            "role": target.role,
            "model": target.model,
            "auth_mode": target.auth_mode.value,
        }
        for target in targets
    ]


def _captured_targets(
    records: list[dict[str, Any]], targets: list[CaptureTarget]
) -> set[CaptureTarget]:
    captured: set[CaptureTarget] = set()
    for record in records:
        metadata = _record_metadata(record)
        if metadata.get("role_attribution_complete") is not True:
            continue
        for target in targets:
            if (
                metadata.get("agent") == target.agent
                and metadata.get("role") == target.role
                and metadata.get("auth_mode") == target.auth_mode.value
                and _model_matches_target(metadata.get("model"), target.model)
            ):
                captured.add(target)
    return captured


def _model_matches_target(value: Any, configured_model: str | None) -> bool:
    return _model_match_score(value, configured_model) > 0


def _model_match_score(value: Any, configured_model: str | None) -> int:
    if configured_model is None:
        return 1
    if not isinstance(value, str):
        return 0
    normalized_value = value.casefold()
    normalized_target = configured_model.casefold()
    proxy_alias = safe_model_alias(configured_model).casefold()
    if normalized_value == normalized_target or normalized_value in {
        proxy_alias,
        f"openai/{proxy_alias}",
    }:
        return 2
    if normalized_value.endswith(f"/{normalized_target}") or normalized_target.endswith(
        f"/{normalized_value}"
    ):
        return 1
    return 0


def _record_model(record: dict[str, Any]) -> str | None:
    request = record.get("request")
    body = request.get("body") if isinstance(request, dict) else None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) and model else None


def _target_for_model(
    targets: list[CaptureTarget], model: str | None
) -> CaptureTarget | None:
    if len(targets) == 1:
        return targets[0]
    if model is None:
        return None
    matches = [
        target
        for target in targets
        if target.model and _model_matches_target(model, target.model)
    ]
    return matches[0] if len(matches) == 1 else None


def _target_for_provider_record(
    targets: list[CaptureTarget], record: dict[str, Any]
) -> CaptureTarget | None:
    if len(targets) == 1:
        return targets[0]
    metadata = _record_metadata(record)
    stamped_role = metadata.get("benchflow_role")
    stamped_agent = metadata.get("benchflow_agent")
    if isinstance(stamped_role, str) and stamped_role:
        targets = [
            target
            for target in targets
            if target.role == stamped_role
            and (
                not isinstance(stamped_agent, str)
                or not stamped_agent
                or target.agent == stamped_agent
            )
        ]
        if len(targets) == 1:
            return targets[0]
        if not targets:
            return None
    candidates = {_record_model(record)}
    candidates.update(
        value
        for key in (
            "benchflow_requested_model",
            "benchflow_model_alias",
            "model_group",
            "request_model",
            "provider_model",
        )
        if isinstance((value := metadata.get(key)), str) and value
    )
    scored = [
        (
            max(
                _model_match_score(candidate, target.model) for candidate in candidates
            ),
            target,
        )
        for target in targets
        if target.model
    ]
    best_score = max((score for score, _target in scored), default=0)
    matches = [target for score, target in scored if score == best_score > 0]
    return matches[0] if len(matches) == 1 else None


def _target_for_record(
    targets: list[CaptureTarget], record: dict[str, Any]
) -> CaptureTarget | None:
    native_session_id = _record_metadata(record).get("native_session_id")
    if isinstance(native_session_id, str):
        session_matches = [
            target
            for target in targets
            if native_session_id in target.native_session_ids
        ]
        if len(session_matches) == 1:
            return session_matches[0]
    return _target_for_model(targets, _record_model(record))


def _sort_exchange_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def timestamp(record: dict[str, Any]) -> str:
        request = record.get("request")
        value = request.get("timestamp") if isinstance(request, dict) else None
        return value if isinstance(value, str) else ""

    return sorted(records, key=timestamp)


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _record_metadata_bool(record: dict[str, Any], key: str) -> bool:
    return _record_metadata(record).get(key) is True


def _aggregate_source(records: list[dict[str, Any]]) -> CaptureSource:
    values = {
        CaptureSource(str(_record_metadata(record).get("capture_source")))
        for record in records
    }
    return next(iter(values)) if len(values) == 1 else CaptureSource.MIXED


def _aggregate_fidelity(records: list[dict[str, Any]]) -> CaptureFidelity:
    values = {
        CaptureFidelity(str(_record_metadata(record).get("capture_fidelity")))
        for record in records
    }
    return next(iter(values)) if len(values) == 1 else CaptureFidelity.MIXED


def _aggregate_auth_mode(
    records: list[dict[str, Any]],
    *,
    targets: list[CaptureTarget],
    fallback_auth: AuthMode,
) -> AuthMode:
    values = {
        AuthMode(str(_record_metadata(record).get("auth_mode"))) for record in records
    }
    values.update(target.auth_mode for target in targets)
    if not values:
        return fallback_auth
    return next(iter(values)) if len(values) == 1 else AuthMode.MIXED


def _role_captures(
    records: list[dict[str, Any]], *, targets: list[CaptureTarget]
) -> list[LLMRoleCapture]:
    grouped: dict[
        tuple[str, str, str | None, AuthMode, CaptureSource, CaptureFidelity],
        list[dict[str, Any]],
    ] = {}
    for record in records:
        metadata = _record_metadata(record)
        role = str(metadata.get("role") or "unknown")
        agent = str(metadata.get("agent") or "unknown")
        model_value = metadata.get("model") or _record_model(record)
        model = str(model_value) if model_value is not None else None
        key = (
            role,
            agent,
            model,
            AuthMode(str(metadata.get("auth_mode"))),
            CaptureSource(str(metadata.get("capture_source"))),
            CaptureFidelity(str(metadata.get("capture_fidelity"))),
        )
        grouped.setdefault(key, []).append(record)
    captures = [
        LLMRoleCapture(
            role=role,
            agent=agent,
            model=model,
            auth_mode=auth_mode,
            capture_source=source,
            capture_fidelity=fidelity,
            exchange_count=len(group),
            request_complete=all(
                _record_metadata_bool(record, "request_complete") for record in group
            ),
            response_complete=all(
                _record_metadata_bool(record, "response_complete") for record in group
            ),
        )
        for (role, agent, model, auth_mode, source, fidelity), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        )
    ]
    captured_roles = {
        (capture.role, capture.agent, capture.model, capture.auth_mode)
        for capture in captures
    }
    captures.extend(
        LLMRoleCapture(
            role=target.role,
            agent=target.agent,
            model=target.model,
            auth_mode=target.auth_mode,
            capture_source=CaptureSource.NONE,
            capture_fidelity=CaptureFidelity.NONE,
            exchange_count=0,
            request_complete=False,
            response_complete=False,
        )
        for target in targets
        if (target.role, target.agent, target.model, target.auth_mode)
        not in captured_roles
    )
    return sorted(
        captures,
        key=lambda capture: (capture.role, capture.agent, capture.model or ""),
    )
