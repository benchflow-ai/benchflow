"""Synchronized multimodal trace index for an embodied trial.

A physical trial writes several independently clocked streams next to its
normal BenchFlow trajectory: camera frames, per-command observation images,
bridge action events, harness telemetry and the agent's own trajectory. The
index lists every stream once with its native clock, maps its timestamps onto
one reference clock (UTC epoch seconds), and states how complete it is.

Bridge events are split into the four stages that must never be conflated:
the agent's *request*, the bridge's *guard decision* (a rejection never
reached the arm), the *dispatch* to the harness, and the harness *receipt*.
A dispatch without a receipt, or a receipt marked uncertain, is an outcome
nobody observed; it is listed rather than guessed.

Indexing only reads. Files left short by a timeout, cancellation or crash are
reported as ``partial`` (truncated tail, capture errors, incomplete capture
summary) and never rewritten, trimmed or removed. Large files are hashed in
chunks, so indexing a multi-gigabyte recording needs constant memory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from .bridge import MOTION_COMMANDS

_CHUNK = 1 << 20
_UNCERTAIN = "Command outcome uncertain"
# Harness clocks below this are relative seconds, not UTC epoch (2001-09-09).
_MIN_EPOCH = 1e9


def utc_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iso_epoch(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


class _Clock:
    """Estimates the host monotonic -> UTC offset from paired samples."""

    def __init__(self, anchor: Mapping[str, Any] | None):
        self.anchor = None
        if isinstance(anchor, Mapping):
            mono, utc = (
                _number(anchor.get("monotonic")),
                _number(anchor.get("utc_epoch")),
            )
            if mono is not None and utc is not None:
                self.anchor = utc - mono
        self.count = 0
        self.total = 0.0
        self.low = math.inf
        self.high = -math.inf

    def observe(self, record: Mapping[str, Any]) -> None:
        mono, utc = _number(record.get("monotonic")), _number(record.get("utc_epoch"))
        if mono is None or utc is None:
            return
        offset = utc - mono
        self.count += 1
        self.total += offset
        self.low = min(self.low, offset)
        self.high = max(self.high, offset)

    @property
    def offset(self) -> float | None:
        if self.anchor is not None:
            return self.anchor
        return self.total / self.count if self.count else None

    def to_utc(self, record: Mapping[str, Any]) -> float | None:
        utc = _number(record.get("utc_epoch"))
        if utc is not None:
            return utc
        mono, offset = _number(record.get("monotonic")), self.offset
        return mono + offset if mono is not None and offset is not None else None

    def describe(self) -> dict[str, Any]:
        return {
            "reference": "utc_epoch_s",
            "host_monotonic_offset_s": self.offset,
            "offset_source": "manifest_anchor"
            if self.anchor is not None
            else "paired_samples"
            if self.count
            else None,
            "paired_samples": self.count,
            "offset_spread_s": self.high - self.low if self.count else None,
        }


def _scan_jsonl(path: Path, visit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Stream one JSONL file: digest, record count and tail integrity."""
    digest = hashlib.sha256()
    size = records = unparsable = 0
    truncated = False
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            size += len(raw)
            last = not raw.endswith(b"\n")
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                if last:
                    truncated = True
                else:
                    unparsable += 1
                continue
            if last:
                # The writers always end a record with a newline.
                truncated = True
            if isinstance(record, dict):
                records += 1
                visit(record)
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "records": records,
        "unparsable_lines": unparsable,
        "truncated_tail": truncated,
    }


class _Span:
    def __init__(self) -> None:
        self.first: float | None = None
        self.last: float | None = None
        self.untimed = 0

    def add(self, epoch: float | None) -> None:
        if epoch is None:
            self.untimed += 1
            return
        self.first = epoch if self.first is None else min(self.first, epoch)
        self.last = epoch if self.last is None else max(self.last, epoch)

    def fields(self) -> dict[str, Any]:
        return {
            "first_utc": self.first,
            "last_utc": self.last,
            "first_iso": utc_iso(self.first),
            "last_iso": utc_iso(self.last),
            "untimed_records": self.untimed,
        }


def _status(scan: Mapping[str, Any], span: _Span, *, incomplete: bool = False) -> str:
    if scan["truncated_tail"] or scan["unparsable_lines"] or incomplete:
        return "partial"
    if not scan["records"]:
        return "empty"
    if span.first is None:
        return "untimed"
    return "complete"


def _missing(name: str, kind: str, path: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "path": path, "status": "missing"}


def _camera_streams(
    trial: Path, cameras: Iterable[str], clock: _Clock
) -> list[dict[str, Any]]:
    summary_path = trial / "cameras" / "capture-summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        summary = None
    per_camera = summary.get("cameras", {}) if isinstance(summary, dict) else {}
    streams = []
    for name in cameras:
        index = trial / "cameras" / f"{name}.jsonl"
        relative = index.relative_to(trial).as_posix()
        if not index.is_file():
            streams.append(_missing(f"camera:{name}", "video", relative))
            continue
        span, errors = _Span(), 0

        def visit(record: dict[str, Any], span: _Span = span) -> None:
            nonlocal errors
            clock.observe(record)
            if "error_type" in record:
                errors += 1
                return
            span.add(clock.to_utc(record))

        scan = _scan_jsonl(index, visit)
        media = [
            {"path": path.relative_to(trial).as_posix(), **file_digest(path)}
            for path in (index.with_suffix(".mjpeg"), index.with_suffix(".mp4"))
            if path.is_file()
        ]
        captured = per_camera.get(name) if isinstance(per_camera, dict) else None
        gap = (
            _number(captured.get("maximum_gap_s"))
            if isinstance(captured, dict)
            else None
        )
        streams.append(
            {
                "name": f"camera:{name}",
                "kind": "video",
                "path": relative,
                "clock": "utc_epoch",
                "time_field": "utc_epoch",
                **scan,
                "records": scan["records"] - errors,
                "capture_errors": errors,
                "capture_summary": captured is not None,
                "maximum_gap_s": gap,
                "media": media,
                # Same bar as RecordingSidecar.stop(): no errors, no gap >= 3 s,
                # and a finalized capture summary (absent after a crash).
                "status": _status(
                    scan,
                    span,
                    incomplete=bool(errors)
                    or captured is None
                    or gap is None
                    or gap >= 3
                    or not media,
                ),
                **span.fields(),
            }
        )
    return streams


class _Actions:
    """Joins bridge events by request id across the four action stages."""

    STAGES: ClassVar[dict[str, str]] = {
        "command_requested": "action_request",
        "command_rejected": "guard_decision",
        "command_started": "dispatch",
        "command_finished": "receipt",
    }

    def __init__(self, clock: _Clock):
        self.clock = clock
        self.spans = {stage: _Span() for stage in self.STAGES.values()}
        self.counts = dict.fromkeys(self.STAGES.values(), 0)
        self.requests_logged = False
        self.dispatched: dict[str, str] = {}
        self.receipted: set[str] = set()
        self.uncertain: list[str] = []
        self.rejections: dict[str, int] = {}
        self.motion_dispatched = 0
        self.observations_ok = 0
        self.image_times: dict[str, float] = {}
        self.transport_failures = 0

    def visit(self, record: dict[str, Any]) -> None:
        self.clock.observe(record)
        event = record.get("event")
        if event == "transport_failure":
            self.transport_failures += 1
            return
        stage = self.STAGES.get(event) if isinstance(event, str) else None
        if stage is None:
            return
        epoch = self.clock.to_utc(record)
        self.counts[stage] += 1
        self.spans[stage].add(epoch)
        if stage == "action_request":
            self.requests_logged = True
        elif stage == "guard_decision":
            reason = str(record.get("reason") or "unspecified")
            self.rejections[reason] = self.rejections.get(reason, 0) + 1
        elif stage == "dispatch":
            command = str(record.get("command"))
            self.dispatched[str(record.get("request_id"))] = command
            if command in MOTION_COMMANDS:
                self.motion_dispatched += 1
        else:
            receipt = record.get("receipt")
            receipt = receipt if isinstance(receipt, dict) else {}
            request_id = str(receipt.get("request_id"))
            self.receipted.add(request_id)
            if _UNCERTAIN in str(receipt.get("error") or ""):
                self.uncertain.append(request_id)
            elif receipt.get("ok") and self.dispatched.get(request_id) in {
                "observe",
                "finish",
            }:
                self.observations_ok += 1
            for image in receipt.get("images") or []:
                if isinstance(image, dict) and epoch is not None:
                    self.image_times[str(image.get("name"))] = epoch

    def streams(self, relative: str, scan: Mapping[str, Any]) -> list[dict[str, Any]]:
        streams = []
        for stage, span in self.spans.items():
            derived = stage == "action_request" and not self.requests_logged
            if derived:
                # Logs written before request events existed: every dispatched
                # or rejected command was requested.
                count = self.counts["dispatch"] + self.counts["guard_decision"]
                for other in ("dispatch", "guard_decision"):
                    for value in (self.spans[other].first, self.spans[other].last):
                        if value is not None:
                            span.add(value)
            else:
                count = self.counts[stage]
            streams.append(
                {
                    "name": f"bridge:{stage}",
                    "kind": stage,
                    "path": relative,
                    "clock": "utc_epoch",
                    "time_field": "utc_epoch",
                    "records": count,
                    "derived": derived,
                    "status": "partial"
                    if scan["truncated_tail"] or scan["unparsable_lines"]
                    else "complete",
                    **span.fields(),
                }
            )
        return streams

    def summary(self) -> dict[str, Any]:
        return {
            "requests": self.counts["action_request"] if self.requests_logged else None,
            "rejected": self.counts["guard_decision"],
            "rejections_by_reason": dict(sorted(self.rejections.items())),
            "dispatched": self.counts["dispatch"],
            "receipts": self.counts["receipt"],
            "motion_dispatched": self.motion_dispatched,
            "observations_ok": self.observations_ok,
            "transport_failures": self.transport_failures,
            "dispatched_without_receipt": sorted(set(self.dispatched) - self.receipted),
            "uncertain_outcomes": self.uncertain,
        }


def _telemetry_stream(trial: Path, name: str, path: Path) -> dict[str, Any]:
    relative = path.relative_to(trial).as_posix()
    if not path.is_file():
        return _missing(name, "telemetry", relative)
    span = _Span()
    relative_clock = False

    def visit(record: dict[str, Any]) -> None:
        nonlocal relative_clock
        epoch = _number(record.get("utc_epoch"))
        if epoch is None:
            epoch = _number(record.get("t"))
        if epoch is not None and epoch < _MIN_EPOCH:
            relative_clock = True
            epoch = None
        span.add(epoch)

    scan = _scan_jsonl(path, visit)
    return {
        "name": name,
        "kind": "telemetry",
        "path": relative,
        "clock": "relative_s" if relative_clock else "utc_epoch",
        "time_field": "utc_epoch|t",
        **scan,
        "status": _status(scan, span),
        **span.fields(),
    }


def _agent_streams(trial: Path) -> list[dict[str, Any]]:
    streams = []
    candidates = [
        ("agent:acp", "ts", path)
        for path in sorted(trial.glob("benchflow/*/*/trajectory/acp_trajectory.jsonl"))
    ]
    if (trial / "agent-events.jsonl").is_file():
        candidates.append(
            ("agent:native_cli", "timestamp", trial / "agent-events.jsonl")
        )
    for name, field, path in candidates:
        span = _Span()

        def visit(
            record: dict[str, Any], field: str = field, span: _Span = span
        ) -> None:
            value = record.get(field) or record.get("started_at")
            span.add(
                _number(value) if _number(value) is not None else _iso_epoch(value)
            )

        scan = _scan_jsonl(path, visit)
        streams.append(
            {
                "name": name,
                "kind": "agent_trajectory",
                "path": path.relative_to(trial).as_posix(),
                "clock": "iso8601",
                "time_field": field,
                **scan,
                "status": _status(scan, span),
                **span.fields(),
            }
        )
    return streams


def summarize_actions(
    trial: Path, *, clock_anchor: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """The ``actions`` summary of :func:`build_trace_index`, from the command log only.

    For callers that need the counts behind the execution state (such as
    ``report`` over many trials) without indexing every camera and recording.
    None when the trial has no ``commands.jsonl``.
    """
    commands = Path(trial) / "commands.jsonl"
    if not commands.is_file():
        return None
    actions = _Actions(_Clock(clock_anchor))
    _scan_jsonl(commands, actions.visit)
    return actions.summary()


def build_trace_index(
    trial: Path,
    *,
    cameras: Iterable[str] | None = None,
    arms: Mapping[str, Any] | None = None,
    primary_arm: str | None = None,
    clock_anchor: Mapping[str, Any] | None = None,
    expect_agent: bool = True,
) -> dict[str, Any]:
    """Index every recorded stream of ``trial`` without modifying any file.

    ``cameras`` and ``arms`` name the streams the trial was expected to
    record; an expected stream that is absent is listed as ``missing``. Without
    them, the camera indexes present on disk are used. ``intact`` is false when
    any expected stream is missing or partial; ``synchronized`` is false when a
    present stream has no usable timestamps, and null when no stream is
    present at all. Rerun recordings are read from the trial root and from
    ``recordings/``. An empty stream is intact but is
    still reported (``status: "empty"``): its cause is not inferable from size.
    """
    trial = Path(trial)
    clock = _Clock(clock_anchor)
    streams: list[dict[str, Any]] = []

    commands = trial / "commands.jsonl"
    actions = _Actions(clock)
    if commands.is_file():
        scan = _scan_jsonl(commands, actions.visit)
        streams.extend(actions.streams("commands.jsonl", scan))
    else:
        streams.append(_missing("bridge:commands", "dispatch", "commands.jsonl"))

    if cameras is None:
        cameras = sorted(p.stem for p in (trial / "cameras").glob("*.jsonl"))
    streams.extend(_camera_streams(trial, cameras, clock))

    images = sorted((trial / "observations").glob("*.jpg"))
    span = _Span()
    for image in images:
        span.add(actions.image_times.get(image.name))
    if not (trial / "observations").is_dir():
        streams.append(_missing("observations", "observation", "observations"))
    else:
        streams.append(
            {
                "name": "observations",
                "kind": "observation",
                "path": "observations",
                "clock": "utc_epoch",
                "time_field": "commands.jsonl receipt",
                "records": len(images),
                # Initial/final host observations are taken outside any command,
                # so they have no receipt time; ``untimed_records`` counts them.
                "status": "complete" if images else "empty",
                **span.fields(),
            }
        )

    for name in arms or {}:
        filename = (
            "encoders.jsonl"
            if name == primary_arm or len(arms or {}) == 1
            else f"encoders-{name}.jsonl"
        )
        streams.append(_telemetry_stream(trial, f"telemetry:{name}", trial / filename))
    if not arms and (trial / "encoders.jsonl").is_file():
        streams.append(
            _telemetry_stream(trial, "telemetry:arm", trial / "encoders.jsonl")
        )

    agent = _agent_streams(trial)
    if not agent and expect_agent:
        agent = [_missing("agent", "agent_trajectory", "benchflow")]
    streams.extend(agent)

    # Rerun files sit in the trial root or, in the demo and viewer layout,
    # under recordings/; the name keeps that directory so it stays unique.
    for recording in [
        *sorted(trial.glob("*.rrd")),
        *sorted(trial.glob("recordings/*.rrd")),
    ]:
        relative = recording.relative_to(trial).as_posix()
        streams.append(
            {
                "name": f"rerun:{relative.removesuffix('.rrd')}",
                "kind": "rerun_recording",
                "path": relative,
                "clock": "utc_epoch",
                "time_field": "rerun timeline 'unix'",
                **file_digest(recording),
                "status": "complete",
            }
        )

    present = [stream for stream in streams if stream["status"] != "missing"]
    return {
        "clock": clock.describe(),
        "streams": streams,
        "actions": actions.summary() if commands.is_file() else None,
        # Every expected stream exists and none was cut short or damaged.
        "intact": all(
            stream["status"] in {"complete", "empty", "untimed"} for stream in streams
        ),
        # Every present stream can be placed on the reference clock; null
        # when no stream is present, as there is nothing to synchronize.
        "synchronized": all(
            stream["status"] != "untimed" and stream.get("clock") != "relative_s"
            for stream in present
        )
        if present
        else None,
    }
