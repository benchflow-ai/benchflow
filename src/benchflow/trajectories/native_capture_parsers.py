"""Normalize Claude Code and Codex native capture surfaces into LLM exchanges."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow.trajectories.llm_capture_manifest import (
    LLM_TRAJECTORY_SCHEMA_VERSION,
    CaptureFidelity,
    CaptureSource,
)
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


@dataclass(frozen=True)
class NativeParseResult:
    trajectory: Trajectory
    source: CaptureSource
    fidelity: CaptureFidelity
    request_complete: bool
    response_complete: bool
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _OtelEvent:
    name: str
    timestamp: datetime
    attributes: dict[str, Any]


@dataclass(frozen=True)
class _RequestBodyEvent:
    path: Path
    timestamp: datetime
    session_key: str | None
    otel_referenced: bool


def parse_claude_raw_capture(
    capture_dir: Path,
    *,
    agent: str,
    session_id: str,
    started_at: datetime,
) -> NativeParseResult | None:
    """Pair Claude raw bodies using OTLP log order within each session."""

    raw_dir = capture_dir / "raw"
    if not raw_dir.is_dir():
        return None
    bodies = _load_body_files(raw_dir)
    events = _load_otel_events(capture_dir / "otel")
    if not bodies or not events:
        return None

    pending = _request_body_events(raw_dir, bodies, events)
    exchanges: list[LLMExchange] = []
    errors: list[str] = []
    pairing_ambiguous = False
    observed_responses: set[Path] = set()
    for event in sorted(events, key=lambda item: item.timestamp):
        event_session = _event_session_key(event.attributes)
        refs = _body_references(event.attributes)
        if "api_response" not in event.name:
            continue
        response_ref = next((ref for ref in refs if ".response.json" in ref), None)
        if response_ref is None:
            continue
        response_path = _resolve_body_reference(raw_dir, response_ref, bodies)
        if response_path is not None:
            if response_path in observed_responses:
                continue
            observed_responses.add(response_path)
        candidates = [
            request
            for request in pending
            if request.timestamp <= event.timestamp
            and request.session_key in {None, event_session}
        ]
        if response_path is None or not candidates:
            errors.append("unpaired Claude raw API response")
            continue
        if len(candidates) > 1:
            pairing_ambiguous = True
        request_event = candidates[0]
        pending.remove(request_event)
        request_path = request_event.path
        request_timestamp = request_event.timestamp
        request_body = bodies[request_path]
        response_body = bodies[response_path]
        response_timestamp = event.timestamp
        request_id = _request_id(event.attributes) or response_path.name.removesuffix(
            ".response.json"
        )
        exchanges.append(
            _exchange(
                request_body=request_body,
                response_body=response_body,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
                path="/v1/messages",
                source=CaptureSource.CLAUDE_OTEL_RAW_BODY,
                fidelity=CaptureFidelity.PROVIDER_WIRE,
                auth_mode="oauth_subscription",
                request_complete=True,
                response_complete=True,
                extra_metadata={
                    "provider_request_id": request_id,
                    "native_session_id": event_session,
                    "request_body_file": request_path.name,
                    "response_body_file": response_path.name,
                    "pairing": (
                        "otel_session_fifo"
                        if request_event.otel_referenced
                        else "raw_file_mtime_fifo"
                    ),
                    "correlation_complete": len(candidates) == 1,
                },
            )
        )

    dangling = len(pending)
    if dangling:
        errors.append(f"{dangling} unpaired Claude raw API request(s)")
    unseen_responses = sum(
        path.name.endswith(".response.json") and path not in observed_responses
        for path in bodies
    )
    if unseen_responses:
        errors.append(
            f"{unseen_responses} Claude raw API response(s) lacked an OTLP body event"
        )
    if not exchanges:
        return None
    if pairing_ambiguous:
        errors.append(
            "concurrent Claude requests lacked a provider correlation id; "
            "request/response pairing is ambiguous"
        )
        for exchange in exchanges:
            exchange.metadata["request_complete"] = False
            exchange.metadata["correlation_complete"] = False
    finished_at = max(exchange.response.timestamp for exchange in exchanges)
    return NativeParseResult(
        trajectory=Trajectory(
            session_id=session_id,
            agent_name=agent,
            started_at=started_at,
            finished_at=finished_at,
            exchanges=exchanges,
        ),
        source=CaptureSource.CLAUDE_OTEL_RAW_BODY,
        fidelity=CaptureFidelity.PROVIDER_WIRE,
        request_complete=not pairing_ambiguous,
        response_complete=True,
        errors=errors,
    )


def parse_claude_sessions(
    sessions_dir: Path,
    *,
    agent: str,
    session_id: str,
    started_at: datetime,
) -> NativeParseResult | None:
    """Reconstruct model turns from Claude Code's native session transcript."""

    record_groups = _read_jsonl_files(sessions_dir, started_at=started_at)
    exchanges = [
        exchange
        for records in record_groups
        for exchange in _parse_claude_session_records(records, started_at=started_at)
    ]
    if not exchanges:
        return None
    exchanges.sort(key=lambda exchange: exchange.request.timestamp)
    return NativeParseResult(
        trajectory=Trajectory(
            session_id=session_id,
            agent_name=agent,
            started_at=started_at,
            finished_at=max(exchange.response.timestamp for exchange in exchanges),
            exchanges=exchanges,
        ),
        source=CaptureSource.CLAUDE_NATIVE_SESSION,
        fidelity=CaptureFidelity.AGENT_SESSION,
        request_complete=False,
        response_complete=False,
        missing_fields=[
            "system_prompt",
            "tool_definitions",
            "provider_response_envelope",
            "headers",
        ],
    )


def _parse_claude_session_records(
    records: list[dict[str, Any]], *, started_at: datetime
) -> list[LLMExchange]:
    """Parse one Claude session file without leaking history across sessions."""

    messages: list[dict[str, Any]] = []
    exchanges: list[LLMExchange] = []
    assistant_group: list[dict[str, Any]] = []
    group_key: str | None = None

    def flush_group() -> None:
        nonlocal assistant_group, group_key
        if not assistant_group:
            return
        message = _merge_claude_assistant_group(assistant_group)
        request_timestamp = _record_timestamp(assistant_group[0], started_at)
        response_timestamp = _record_timestamp(assistant_group[-1], request_timestamp)
        model = message.get("model")
        request_body: dict[str, Any] = {"messages": list(messages)}
        if isinstance(model, str) and model:
            request_body["model"] = model
        exchanges.append(
            _exchange(
                request_body=request_body,
                response_body=message,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
                path="/v1/messages",
                source=CaptureSource.CLAUDE_NATIVE_SESSION,
                fidelity=CaptureFidelity.AGENT_SESSION,
                auth_mode="oauth_subscription",
                request_complete=False,
                response_complete=False,
                extra_metadata={
                    "provider_request_id": group_key,
                    "missing_fields": [
                        "system_prompt",
                        "tool_definitions",
                        "provider_response_envelope",
                        "headers",
                    ],
                },
            )
        )
        messages.append(_message_for_history(message))
        assistant_group = []
        group_key = None

    for record in records:
        record_type = record.get("type")
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        if record_type == "user":
            flush_group()
            messages.append(_message_for_history(message))
            continue
        if record_type != "assistant":
            continue
        key = str(
            record.get("requestId")
            or message.get("id")
            or record.get("uuid")
            or f"assistant-{len(exchanges)}"
        )
        if assistant_group and key != group_key:
            flush_group()
        group_key = key
        assistant_group.append(record)
    flush_group()
    return exchanges


def parse_codex_sessions(
    sessions_dir: Path,
    *,
    agent: str,
    session_id: str,
    started_at: datetime,
    configured_model: str | None,
    auth_mode: str = "oauth_subscription",
) -> NativeParseResult | None:
    """Reconstruct Responses-style calls from Codex native session records."""

    record_groups = _read_jsonl_files(sessions_dir, started_at=started_at)
    exchanges = [
        exchange
        for records in record_groups
        for exchange in _parse_codex_session_records(
            records,
            started_at=started_at,
            configured_model=configured_model,
            auth_mode=auth_mode,
        )
    ]
    if not exchanges:
        return None
    exchanges.sort(key=lambda exchange: exchange.request.timestamp)
    return NativeParseResult(
        trajectory=Trajectory(
            session_id=session_id,
            agent_name=agent,
            started_at=started_at,
            finished_at=max(exchange.response.timestamp for exchange in exchanges),
            exchanges=exchanges,
        ),
        source=CaptureSource.CODEX_NATIVE_SESSION,
        fidelity=CaptureFidelity.AGENT_SESSION,
        request_complete=False,
        response_complete=False,
        missing_fields=[
            "instructions",
            "tool_definitions",
            "provider_response_envelope",
            "headers",
        ],
    )


def _parse_codex_session_records(
    records: list[dict[str, Any]],
    *,
    started_at: datetime,
    configured_model: str | None,
    auth_mode: str,
) -> list[LLMExchange]:
    """Parse one Codex session file without leaking history across sessions."""

    history: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    exchanges: list[LLMExchange] = []
    output_started_at: datetime | None = None
    last_timestamp = started_at
    pending_usage: dict[str, Any] | None = None
    model = configured_model or _codex_session_model(records)

    def flush_output(response_timestamp: datetime) -> None:
        nonlocal output, output_started_at, pending_usage
        if not output:
            return
        response_body: dict[str, Any] = {"status": "completed", "output": output}
        if pending_usage:
            response_body["usage"] = _normalize_codex_usage(pending_usage)
        exchanges.append(
            _exchange(
                request_body={"model": model, "input": list(history)},
                response_body=response_body,
                request_timestamp=output_started_at or response_timestamp,
                response_timestamp=response_timestamp,
                path="/v1/responses",
                source=CaptureSource.CODEX_NATIVE_SESSION,
                fidelity=CaptureFidelity.AGENT_SESSION,
                auth_mode=auth_mode,
                request_complete=False,
                response_complete=False,
                extra_metadata={
                    "missing_fields": [
                        "instructions",
                        "tool_definitions",
                        "provider_response_envelope",
                        "headers",
                    ]
                },
            )
        )
        history.extend(output)
        output = []
        output_started_at = None
        pending_usage = None

    for record in records:
        timestamp = _record_timestamp(record, last_timestamp)
        last_timestamp = timestamp
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "event_msg" and payload.get("type") == "token_count":
            usage = payload.get("info")
            if isinstance(usage, dict):
                last_usage = usage.get("last_token_usage")
                pending_usage = last_usage if isinstance(last_usage, dict) else usage
            flush_output(timestamp)
            continue
        if record.get("type") != "response_item":
            continue
        payload_type = str(payload.get("type") or "")
        role = str(payload.get("role") or "")
        is_input = (payload_type == "message" and role in {"user", "developer"}) or (
            payload_type.endswith("_call_output")
        )
        if is_input:
            flush_output(timestamp)
            history.append(payload)
            continue
        if payload_type in {
            "message",
            "reasoning",
            "function_call",
            "custom_tool_call",
            "web_search_call",
        } or payload_type.endswith("_call"):
            if output_started_at is None:
                output_started_at = timestamp
            output.append(payload)
    flush_output(last_timestamp)
    return exchanges


def project_acp_trajectory(
    events: list[dict[str, Any]],
    *,
    agent: str,
    session_id: str,
    started_at: datetime,
    auth_mode: str,
) -> NativeParseResult | None:
    """Last-resort projection that remains explicit about its low fidelity."""

    prompts: list[dict[str, Any]] = []
    exchanges: list[LLMExchange] = []
    for index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "user_message":
            prompts.append({"role": "user", "content": event.get("text", "")})
        elif event_type in {"agent_message", "agent_thought"}:
            timestamp = _record_timestamp(event, started_at)
            response_body = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": event.get("text", "")}
                        ],
                    }
                ],
            }
            exchanges.append(
                _exchange(
                    request_body={"input": list(prompts)},
                    response_body=response_body,
                    request_timestamp=timestamp,
                    response_timestamp=timestamp,
                    path="acp://session",
                    source=CaptureSource.ACP_PROJECTION,
                    fidelity=CaptureFidelity.ACP_PROJECTION,
                    auth_mode=auth_mode,
                    request_complete=False,
                    response_complete=False,
                    extra_metadata={
                        "acp_event_index": index,
                        "missing_fields": ["provider_request", "provider_response"],
                    },
                )
            )
            prompts.append(
                {"role": "assistant", "content": str(event.get("text") or "")}
            )
    if not exchanges:
        return None
    return NativeParseResult(
        trajectory=Trajectory(
            session_id=session_id,
            agent_name=agent,
            started_at=started_at,
            finished_at=max(exchange.response.timestamp for exchange in exchanges),
            exchanges=exchanges,
        ),
        source=CaptureSource.ACP_PROJECTION,
        fidelity=CaptureFidelity.ACP_PROJECTION,
        request_complete=False,
        response_complete=False,
        missing_fields=["provider_request", "provider_response"],
    )


def _exchange(
    *,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    request_timestamp: datetime,
    response_timestamp: datetime,
    path: str,
    source: CaptureSource,
    fidelity: CaptureFidelity,
    auth_mode: str,
    request_complete: bool,
    response_complete: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> LLMExchange:
    metadata = {
        "schema_version": LLM_TRAJECTORY_SCHEMA_VERSION,
        "capture_source": source.value,
        "capture_fidelity": fidelity.value,
        "auth_mode": auth_mode,
        "request_complete": request_complete,
        "response_complete": response_complete,
        "payload_redacted": True,
        **(extra_metadata or {}),
    }
    return LLMExchange(
        request=LLMRequest(timestamp=request_timestamp, path=path, body=request_body),
        response=LLMResponse(
            timestamp=response_timestamp,
            status_code=200,
            body=response_body,
        ),
        duration_ms=max(
            0.0, (response_timestamp - request_timestamp).total_seconds() * 1000
        ),
        metadata=metadata,
    )


def _read_jsonl_files(
    root: Path, *, started_at: datetime | None = None
) -> list[list[dict[str, Any]]]:
    if not root.is_dir():
        return []
    boundary = started_at.timestamp() - 1.0 if started_at is not None else None
    record_groups: list[list[dict[str, Any]]] = []
    for path in sorted(root.rglob("*.jsonl"), key=lambda item: item.stat().st_mtime):
        if boundary is not None and path.stat().st_mtime < boundary:
            continue
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                timestamp = _optional_record_timestamp(record)
                if (
                    boundary is not None
                    and timestamp is not None
                    and timestamp.timestamp() < boundary
                ):
                    continue
                records.append(record)
        if records:
            record_groups.append(records)
    return record_groups


def _load_body_files(root: Path) -> dict[Path, dict[str, Any]]:
    bodies: dict[Path, dict[str, Any]] = {}
    for path in root.rglob("*.json"):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(body, dict):
            bodies[path] = body
    return bodies


def _request_body_events(
    root: Path,
    bodies: dict[Path, dict[str, Any]],
    otel_events: list[_OtelEvent],
) -> list[_RequestBodyEvent]:
    """Resolve request bodies even when Claude omits the first body log event.

    Claude writes every raw request before sending it, but its periodic OTLP exporter
    can start after the first request and omit that request's ``api_request_body``
    record.  File modification time is therefore a safe ordering fallback.  Pairing
    still fails closed if more than one request could match a response.
    """

    referenced: dict[Path, _RequestBodyEvent] = {}
    for event in otel_events:
        request_ref = next(
            (
                ref
                for ref in _body_references(event.attributes)
                if ".request.json" in ref
            ),
            None,
        )
        request_path = _resolve_body_reference(root, request_ref, bodies)
        if request_path is None or request_path in referenced:
            continue
        referenced[request_path] = _RequestBodyEvent(
            path=request_path,
            timestamp=event.timestamp,
            session_key=_event_session_key(event.attributes),
            otel_referenced=True,
        )

    requests = list(referenced.values())
    for path in bodies:
        if not path.name.endswith(".request.json") or path in referenced:
            continue
        requests.append(
            _RequestBodyEvent(
                path=path,
                timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                session_key=None,
                otel_referenced=False,
            )
        )
    return sorted(requests, key=lambda item: (item.timestamp, item.path.name))


def _load_otel_events(root: Path) -> list[_OtelEvent]:
    events: list[_OtelEvent] = []
    if not root.is_dir():
        return events
    for path in root.rglob("*.json"):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in _walk_log_records(envelope):
            attributes = {
                str(item.get("key")): _otel_value(item.get("value"))
                for item in record.get("attributes") or []
                if isinstance(item, dict) and item.get("key")
            }
            body = _otel_value(record.get("body"))
            name = str(
                attributes.get("event.name")
                or attributes.get("event_name")
                or body
                or ""
            )
            timestamp = _unix_nanos_timestamp(
                record.get("timeUnixNano") or record.get("observedTimeUnixNano")
            )
            events.append(
                _OtelEvent(name=name, timestamp=timestamp, attributes=attributes)
            )
    return events


def _walk_log_records(value: Any):
    if isinstance(value, dict):
        records = value.get("logRecords")
        if isinstance(records, list):
            yield from (record for record in records if isinstance(record, dict))
        for nested in value.values():
            yield from _walk_log_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_log_records(nested)


def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
        if key in value:
            return value[key]
    array = value.get("arrayValue")
    if isinstance(array, dict):
        return [_otel_value(item) for item in array.get("values") or []]
    pairs = value.get("kvlistValue")
    if isinstance(pairs, dict):
        return {
            str(item.get("key")): _otel_value(item.get("value"))
            for item in pairs.get("values") or []
            if isinstance(item, dict)
        }
    return value


def _body_references(attributes: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for value in attributes.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item)
            if ".request.json" in text or ".response.json" in text:
                refs.append(text)
    return refs


def _resolve_body_reference(
    root: Path, reference: str | None, bodies: dict[Path, dict[str, Any]]
) -> Path | None:
    if reference is None:
        return None
    candidate = Path(reference)
    for path in bodies:
        if path == candidate or path.name == candidate.name:
            return path
    relative = root / reference
    return relative if relative in bodies else None


def _event_session_key(attributes: dict[str, Any]) -> str:
    for key in ("session.id", "session_id", "sessionId", "conversation.id"):
        value = attributes.get(key)
        if value:
            return str(value)
    return "default"


def _request_id(attributes: dict[str, Any]) -> str | None:
    for key in ("request.id", "request_id", "requestId", "message.id"):
        value = attributes.get(key)
        if value:
            return str(value)
    return None


def _merge_claude_assistant_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"role": "assistant", "content": []}
    seen_content: set[str] = set()
    for record in records:
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        for key in ("id", "type", "role", "model", "stop_reason", "stop_sequence"):
            if key in message:
                merged[key] = message[key]
        if isinstance(message.get("usage"), dict):
            merged["usage"] = message["usage"]
        content = message.get("content")
        blocks = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        for block in blocks:
            marker = json.dumps(block, sort_keys=True, default=str)
            if marker not in seen_content:
                seen_content.add(marker)
                merged["content"].append(block)
    return merged


def _message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key in {"role", "content"}}


def _codex_session_model(records: list[dict[str, Any]]) -> str:
    for record in records:
        payload = record.get("payload")
        if isinstance(payload, dict):
            model = payload.get("model")
            if isinstance(model, str) and model:
                return model
    return "unknown"


def _normalize_codex_usage(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "input_tokens_details": {"cached_tokens": usage.get("cached_input_tokens", 0)},
        "output_tokens": usage.get("output_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": usage.get("reasoning_output_tokens", 0)
        },
        "total_tokens": usage.get("total_tokens", 0),
    }


def _record_timestamp(record: dict[str, Any], default: datetime) -> datetime:
    return _optional_record_timestamp(record) or default


def _optional_record_timestamp(record: dict[str, Any]) -> datetime | None:
    value = record.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _unix_nanos_timestamp(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1_000_000_000, tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(tz=UTC)
