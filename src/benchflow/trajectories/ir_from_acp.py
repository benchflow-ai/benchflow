"""ACP-session capture events → canonical Trace IR (Slice C).

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, which is
> itself an unapproved proposal (`docs/trace-interop.md` §8). Nothing imports
> this module, nothing writes its output to disk, and no capture path, exporter
> or artifact changes because it exists.

This is the first real edge of the hub, and its job is as much to *stress the
contract* as to convert: every `None` the IR carries has to be justified by a
`LossRecord`, and this converter is where that stops being a design statement.

## What it converts

The event list as written to `trajectory/acp_trajectory.jsonl` — the ACP-session
capture vocabulary pinned by Slice A's schema, plus the two other record
families §2.4 documents as sharing that file. It reads the events and nothing
else: `result.json`, `timing.json` and the proxy capture are separate artifacts
and are not consulted, so a value that only exists there is a declared loss
rather than a silent enrichment.

## What it deliberately does not do

**It does not prepend `prompts` as leading user events.** `acp_events_to_atif_steps`
and `acp_events_to_adp_content` both do, and §5.2 records the consequence: an
ATIF document opens with two identical `user` steps — one from the `prompts`
argument, one from the captured `user_message` event — so a consumer counting
user turns over-counts by one. Those steps are not ACP events. If a target
format wants them, its own converter adds them and records
:attr:`LossClass.SYNTHESIZED`; inventing them here would put the defect in the
hub, where every format would inherit it.

It also does not synthesize tool-call ids, timestamps, an agent version, or
arguments. Those are target-side obligations (§8.2).

## Loss addressing

Two path shapes appear in the report, and the difference is load-bearing:

- ``events[i].…`` — a field of the IR event at index *i*. This is what
  :func:`~benchflow.trajectories.ir.validate_trace` matches against, so the
  per-call `arguments` records must use it.
- ``source[i]`` — an entry of the *input* list that produced no IR event at all.
  It cannot be addressed as ``events[i]``, because that index belongs to a
  different event once an entry is skipped.

Systemic losses — the ones that hold for every event of a kind rather than for
one event — are declared **once** with an unindexed ``events[].…`` path. Writing
them per event would multiply the report by the trace length while adding no
information; see §8.6 and the volume test in the suite.

## What the report does and does not claim

Losses are declared against **the documented emitter contract** (§2.3), not
against arbitrary input. `_events_to_trajectory` writes all six tool-call fields
as literals, so a record missing one did not come from that emitter; the IR
carries ``None`` for it and declares nothing, because "the ACP-session emitter
does not produce this value" would be a false statement about a record it did
not produce. The one exception is ``arguments``, which the emitter *never*
produces for any record and which :func:`validate_trace` therefore requires to
be declared every time.

A value this converter reshapes — a coerced non-string, an unmappable status —
is always declared, whatever the record's provenance. Reshaping is this
module's own act, and the point of the contract is that it owns up to it.
"""

from __future__ import annotations

from typing import Any

from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossRecord,
    LossReport,
    ModelInfo,
    OutcomeStatus,
    Provenance,
    Role,
    ToolCall,
    ToolStatus,
    TraceEvent,
    TraceOutcome,
)

LOSS_DIRECTION = "acp->ir"

# Source labels for per-event provenance. A single ``acp_trajectory.jsonl`` can
# hold records from more than one producer (§2.4), which is why the IR carries
# provenance per event and not only per trace.
ACP_CAPTURE_SOURCE = "acp-capture-v1"
"""The ACP-session capture vocabulary — exactly what Slice A's schema pins."""

ORACLE_SOURCE = "oracle"
"""`_run_oracle`'s alternative trajectory. Not an ACP-session record."""

UNKNOWN_SOURCE = "acp-trajectory-unknown"
"""In the file, outside every vocabulary this repository documents."""

ACP_TRAJECTORY_SOURCE = "acp-trajectory"
"""Trace level: names the artifact family, since its events may disagree."""

_ACP_CAPTURE_PRODUCER = "_events_to_trajectory"

# Two of the three text-bearing capture types. ``agent_thought`` has its own
# branch below: it becomes reasoning rather than text, and carries the segment
# list that keeps its boundaries.
_TEXT_EVENTS: dict[str, tuple[EventKind, Role]] = {
    "user_message": (EventKind.USER_MESSAGE, Role.USER),
    "agent_message": (EventKind.AGENT_MESSAGE, Role.AGENT),
}

_STATUS_BY_VALUE = {status.value: status for status in ToolStatus}

# Keys the emitter writes on each record. Anything else on a recognized record
# is carried into ``extensions`` rather than dropped.
_TOOL_CALL_KEYS = frozenset(
    {"type", "tool_call_id", "kind", "title", "status", "content"}
)
_TEXT_KEYS = frozenset({"type", "text"})
_TIMEOUT_KEYS = frozenset({"type", "reason"})
_ORACLE_KEYS = frozenset({"type"})


def _str_field(
    raw: dict[str, Any], key: str, field: str, losses: LossReport
) -> str | None:
    """Read a string-typed source field, coercing and declaring if it is not one.

    The emitter only ever writes strings here, so the coercion path is
    unreachable from `_events_to_trajectory`. It exists because this converter
    accepts the *file*, and §7 lists fixtures in this repository that use the
    ACP filename with other shapes. Coercing silently would be exactly the kind
    of undeclared normalization the IR exists to stop.

    ``None`` when the key is absent or explicitly null — absent and empty stay
    distinguishable, which is the tri-state rule (§8.2).
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None or isinstance(value, str):
        return value
    losses.add(
        field,
        LossClass.NORMALIZED,
        f"{key} is {type(value).__name__}, not a string; coerced with str()",
    )
    return str(value)


def _extras(raw: dict[str, Any], known: frozenset[str]) -> dict[str, Any]:
    """Every field of *raw* the mapping below does not place explicitly.

    Carrying them verbatim is what keeps a record that is *almost* a known
    shape — an extra key from a future capture-layer change — from being
    silently truncated to the fields this converter happens to know.
    """
    return {key: value for key, value in raw.items() if key not in known}


def _content_block_to_ir(block: Any) -> ContentBlock | None:
    """Classify one ACP content block, mirroring ``content_blocks_to_text``.

    That helper renders the two text shapes it recognizes — the nested ACP
    ``{"type": "content", "content": {"type": "text", "text": …}}`` and the flat
    ``{"text": …}`` — and skips everything else, which is §5 loss #5. Here the
    same two shapes become :attr:`ContentBlockKind.TEXT` and *everything else
    becomes* :attr:`ContentBlockKind.OPAQUE` with the block kept verbatim, so
    the skip stops being a loss.

    ``None`` for a block that cannot be represented at all (a non-object entry);
    the caller declares that as a loss.
    """
    if not isinstance(block, dict):
        return None
    inner = block.get("content")
    if isinstance(inner, dict):
        inner = inner.get("text")
    text = block.get("text") or inner
    if text:
        return ContentBlock(kind=ContentBlockKind.TEXT, text=str(text), raw=block)
    return ContentBlock(kind=ContentBlockKind.OPAQUE, raw=block)


def _tool_call_to_ir(
    raw: dict[str, Any], index: int, losses: LossReport
) -> tuple[ToolCall, dict[str, Any]]:
    """Build the IR tool call, and return the extras to hang on its event."""
    extras = _extras(raw, _TOOL_CALL_KEYS)

    raw_status = raw.get("status")
    status = _STATUS_BY_VALUE.get(str(raw_status)) if raw_status is not None else None
    if raw_status is not None and status is None:
        # Unreachable from the emitter — the serialized value is always
        # ``ToolCallStatus(...).value`` — but reachable from a hand-written or
        # future record. Keep the original addressable instead of only in prose.
        status = ToolStatus.UNKNOWN
        extras["source_status"] = raw_status
        losses.add(
            f"events[{index}].tool_call.status",
            LossClass.NORMALIZED,
            f"status {raw_status!r} is outside the ACP ToolCallStatus vocabulary; "
            "mapped to unknown, original kept in extensions.source_status",
            "§2.3",
        )

    content: list[ContentBlock] = []
    raw_content = raw.get("content")
    if isinstance(raw_content, str):
        # ``content_blocks_to_text`` accepts a bare string; mirror it rather
        # than treating the whole field as unrepresentable.
        content.append(ContentBlock(kind=ContentBlockKind.TEXT, text=raw_content))
    elif isinstance(raw_content, list):
        for position, block in enumerate(raw_content):
            converted = _content_block_to_ir(block)
            if converted is None:
                losses.add(
                    f"events[{index}].tool_call.content",
                    LossClass.DROPPED,
                    f"source content block {position} is {type(block).__name__}, "
                    "not a JSON object, and has no IR representation",
                )
                continue
            content.append(converted)
    elif raw_content is not None:
        losses.add(
            f"events[{index}].tool_call.content",
            LossClass.DROPPED,
            f"content is {type(raw_content).__name__}; the emitter writes a list "
            "and content_blocks_to_text also accepts a string, so neither this "
            "converter nor any existing consumer can read it",
        )

    tool_call = ToolCall(
        call_id=_str_field(
            raw, "tool_call_id", f"events[{index}].tool_call.call_id", losses
        ),
        name=_str_field(raw, "kind", f"events[{index}].tool_call.name", losses),
        # An ACP ``kind`` is a category (``execute``, ``read``), not a function
        # name. Recording that is what keeps the IR from laundering the
        # normalization ATIF performs when it lands the value in
        # ``function_name`` (§5.1).
        name_semantics="acp_kind" if "kind" in raw else None,
        title=_str_field(raw, "title", f"events[{index}].tool_call.title", losses),
        status=status,
        # Never ``{}``: the capture path does not read ``rawInput``, so this is
        # "not carried", not "carried and empty" (§8.2).
        arguments=None,
        content=content,
    )
    losses.add(
        f"events[{index}].tool_call.arguments",
        LossClass.UNSUPPORTED,
        "ACPSession.handle_update reads toolCallId/title/kind/status/content and "
        "drops rawInput, so no ACP-derived tool call carries arguments",
        "§5 loss #1",
    )
    return tool_call, extras


def _declare_systemic_losses(losses: LossReport, *, had_tool_call: bool) -> None:
    """Losses that hold for the whole conversion, declared once each.

    Every one of these is a row of the §5 table that no per-event record could
    make more precise: the capture events carry the value nowhere, so an
    indexed path would repeat one fact once per event.
    """
    if had_tool_call:
        losses.add(
            "events[].tool_call.started_at",
            LossClass.UNSUPPORTED,
            "ToolCallRecord tracks started_at/finished_at in memory and "
            "_events_to_trajectory serializes neither",
            "§5 loss #3",
        )
        losses.add(
            "events[].tool_call.finished_at",
            LossClass.UNSUPPORTED,
            "ToolCallRecord tracks started_at/finished_at in memory and "
            "_events_to_trajectory serializes neither",
            "§5 loss #3",
        )
    losses.add(
        "events[].usage",
        LossClass.UNSUPPORTED,
        "ACP capture events carry no usage; ACPSession.usage_snapshots is routed "
        "to result.json and never into the trajectory",
        "§5 losses #6, #8",
    )
    losses.add(
        "agent.agent_version",
        LossClass.UNSUPPORTED,
        'BenchFlow does not track agent binary versions; ATIF\'s "unknown" is a '
        "target-side obligation, not an observation",
        "§5 loss #7",
    )
    losses.add(
        "outcome.stop_reason",
        LossClass.UNSUPPORTED,
        "ACPSession.stop_reason is captured on the session and exported nowhere",
        "§5 loss #9",
    )


def acp_events_to_ir(
    events: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
) -> CanonicalTrace:
    """Convert captured ACP trajectory events into one :class:`CanonicalTrace`.

    *session_id*, *agent_name* and *model* are metadata the caller already has
    from elsewhere (``result.json``, the rollout config); they are not read out
    of the events, because the events do not carry them. Everything else comes
    from *events* alone.

    The returned trace carries its own :class:`LossReport` on
    :attr:`CanonicalTrace.losses`, and satisfies
    :func:`~benchflow.trajectories.ir.validate_trace` for any input — including
    input the Slice A schema would reject.

    Ordering is preserved exactly. An entry that produces no IR event (a
    non-object entry in the list) is declared under a ``source[i]`` path and
    leaves no hole: IR indices stay dense, which is invariant 2.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    ir_events: list[TraceEvent] = []
    had_tool_call = False
    timed_out = False

    for source_position, raw in enumerate(events):
        if not isinstance(raw, dict):
            losses.add(
                f"source[{source_position}]",
                LossClass.DROPPED,
                f"entry is {type(raw).__name__}, not a JSON object; the IR has no "
                "representation for it",
            )
            continue

        index = len(ir_events)
        etype = raw.get("type")
        etype_str = str(etype) if isinstance(etype, str) else None

        if etype_str in _TEXT_EVENTS:
            kind, role = _TEXT_EVENTS[etype_str]
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=kind,
                    source_type=etype_str,
                    role=role,
                    # ``""`` is preserved: it is an observed value, and both
                    # exporters drop text-empty events today (§5.1).
                    text=_str_field(raw, "text", f"events[{index}].text", losses),
                    provenance=Provenance(
                        source_format=ACP_CAPTURE_SOURCE,
                        producer=_ACP_CAPTURE_PRODUCER,
                    ),
                    extensions=_extras(raw, _TEXT_KEYS),
                )
            )
        elif etype_str == "agent_thought":
            text = _str_field(raw, "text", f"events[{index}].reasoning", losses)
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=EventKind.AGENT_REASONING,
                    source_type=etype_str,
                    role=Role.AGENT,
                    reasoning=text,
                    # One capture record is one thought. Keeping the segment
                    # list means the boundary survives; ``ThoughtBuffer`` joins
                    # thoughts with a blank line and makes the count
                    # unrecoverable (§5 loss #10). Nothing is joined here, so
                    # the loss is avoided rather than reproduced — and a thought
                    # whose own text contains a blank line stays one segment,
                    # because splitting it would invent a boundary.
                    reasoning_segments=[text] if text is not None else None,
                    provenance=Provenance(
                        source_format=ACP_CAPTURE_SOURCE,
                        producer=_ACP_CAPTURE_PRODUCER,
                    ),
                    extensions=_extras(raw, _TEXT_KEYS),
                )
            )
        elif etype_str == "tool_call":
            had_tool_call = True
            tool_call, extras = _tool_call_to_ir(raw, index, losses)
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=EventKind.TOOL_CALL,
                    source_type=etype_str,
                    role=Role.AGENT,
                    tool_call=tool_call,
                    provenance=Provenance(
                        source_format=ACP_CAPTURE_SOURCE,
                        producer=_ACP_CAPTURE_PRODUCER,
                    ),
                    extensions=extras,
                )
            )
        elif etype_str == "agent_timeout":
            timed_out = True
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=EventKind.TIMEOUT,
                    source_type=etype_str,
                    # No role: this is BenchFlow's own marker, not an agent
                    # action and not an ACP notification. Attributing it to the
                    # agent would be inventing semantics.
                    outcome=_str_field(
                        raw, "reason", f"events[{index}].outcome", losses
                    ),
                    provenance=Provenance(
                        source_format=ACP_CAPTURE_SOURCE,
                        producer="record_agent_timeout",
                    ),
                    # timeout_sec / pending_tool_call_ids /
                    # terminal_trajectory_complete ride here rather than as three
                    # IR fields no other source would ever populate.
                    extensions=_extras(raw, _TIMEOUT_KEYS),
                )
            )
        elif etype_str == "oracle":
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=EventKind.ORACLE,
                    source_type=etype_str,
                    role=Role.ORACLE,
                    provenance=Provenance(
                        source_format=ORACLE_SOURCE, producer="_run_oracle"
                    ),
                    # command / return_code / stdout, verbatim. The ATIF
                    # exporter renders these as an agent step prefixed
                    # "[oracle: …]", which a consumer can only undo by string
                    # matching (§5.1); here the record stays itself.
                    extensions=_extras(raw, _ORACLE_KEYS),
                )
            )
        else:
            ir_events.append(
                TraceEvent(
                    index=index,
                    kind=EventKind.UNKNOWN,
                    # Verbatim, including ``None`` for a record with no ``type``.
                    source_type=etype_str,
                    provenance=Provenance(source_format=UNKNOWN_SOURCE),
                    # Nothing is lost: the whole record is carried. Every
                    # exporter in the tree skips these silently today (§5.1).
                    extensions=dict(raw),
                )
            )

    _declare_systemic_losses(losses, had_tool_call=had_tool_call)

    return CanonicalTrace(
        session_id=session_id,
        agent=ModelInfo(agent_name=agent_name, model=model),
        events=ir_events,
        # A timeout marker in the stream is an observation about how the run
        # ended. Every other outcome — pass, fail, reward — lives in
        # ``result.json``, which this converter does not read, so the status
        # stays ``None`` rather than being guessed. The section itself is always
        # present: ``outcome.stop_reason`` is a loss this converter always
        # declares, and a record cannot address a path through a null parent.
        outcome=TraceOutcome(
            status=OutcomeStatus.TIMEOUT if timed_out else None,
        ),
        provenance=Provenance(source_format=ACP_TRAJECTORY_SOURCE),
        losses=losses,
    )


def loss_summary(report: LossReport) -> dict[str, int]:
    """Count a report's records by class — for logs, reviews and eyeballing.

    A conversion's cost should be readable without printing every record; a
    50-tool-call trace declares 50 `arguments` losses that say the same thing.
    """
    summary: dict[str, int] = {}
    for record in report.records:
        summary[record.loss_class.value] = summary.get(record.loss_class.value, 0) + 1
    return summary


def _is_per_event(field: str) -> bool:
    """True for a record addressed to one event or one source entry.

    ``events[]…`` is the unindexed, systemic form and is deliberately not
    per-event, so the prefix test has to run before the general one.
    """
    if field.startswith("events[]"):
        return False
    return field.startswith(("events[", "source["))


def systemic_losses(report: LossReport) -> list[LossRecord]:
    """The records declared once for the whole trace rather than per event.

    The complement — :func:`per_event_losses` — is the part that grows with the
    trace, and is what makes the report's size worth watching.
    """
    return [record for record in report.records if not _is_per_event(record.field)]


def per_event_losses(report: LossReport) -> list[LossRecord]:
    """The records addressed to a single event or a single source entry."""
    return [record for record in report.records if _is_per_event(record.field)]
