"""Canonical Trace IR → ACP-session capture events (Slice G).

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, itself an
> unapproved proposal (`docs/trace-interop.md` §8). Nothing imports this module,
> nothing writes its output to disk, and no capture path, exporter or artifact
> changes because it exists.

## What this edge targets, and what it does not

**The target is the ACP-session capture event format** — the record vocabulary
pinned by Slice A's JSON Schema and produced by
`benchflow.trajectories._capture._events_to_trajectory`. Three record shapes,
each with `additionalProperties: false`.

**The target is *not* the `acp_trajectory.jsonl` artifact.** That file holds
records from several producers — the ACP-session emitter, `_run_oracle`,
`hosted_env._row_to_acp_events`, and whatever a session-factory `Session` puts
in `steps` — and §2.1 records that **no artifact-level contract exists in this
repository**. An edge cannot target a format nobody has defined, so this one
targets the part that *is* defined. Output is a list of event dictionaries in
memory; writing them anywhere is not this module's business.

## Representability is a property of the data, never of the provenance

A trace is exportable when its events carry what the ACP contract requires. It
is **not** exportable because it came from ACP, and not unexportable because it
came from OTel or ATIF. :attr:`~benchflow.trajectories.ir.Provenance` is never
read here, and a test asserts that: a hand-built trace carrying every required
value exports whatever its ``source_format`` says, and an ACP-derived trace
missing one does not.

The practical consequence today is that ACP-derived traces usually export and
OTel-derived ones usually do not — but that is an observation about what those
edges currently carry, not a rule this module applies.

## Fail closed

**A conversion either represents every event or represents none of it.**
:func:`ir_to_acp_capture_events` raises :class:`AcpCaptureNotRepresentable`
rather than returning the events it managed to build, because a partial event
list is indistinguishable from a complete one once it leaves this function — the
target has no envelope, no count and no marker for "some events are missing".
Returning one would be handing a caller a trajectory that silently lost a tool
call and looks like a successful conversion.

The exception is a ``ValueError`` subclass, following
`PrimeSftTrajectoryJsonlError` and the ``ValueError`` `ir_to_atif` already
raises, and it carries the blockers as
:class:`~benchflow.trajectories.ir.LossRecord` values — the loss model this
family already uses to address a field by path. A caller that wants to *ask*
rather than to *handle* calls :func:`acp_capture_blockers` and gets the same
records with no exception involved.

## What is never invented

- **`status` is never synthesized.** The ACP vocabulary is
  ``pending`` / ``in_progress`` / ``completed`` / ``failed`` / ``cancelled``,
  and every member asserts something about a tool call's lifecycle. There is no
  neutral value, the IR's :attr:`~benchflow.trajectories.ir.ToolStatus.UNKNOWN`
  has no counterpart, and a fabricated status reaching the viewer would be
  displayed to a person as an observation. A tool call whose status the IR does
  not know is **not representable**, full stop.
- **`oracle` and `unknown` events are outside the codomain.** The schema models
  three record shapes and neither of these is one of them. They are not dropped
  to make a conversion succeed, the schema is not widened to admit them, and no
  schema-invalid record is emitted: a trace containing one is not representable.
- **No trace-level value is smuggled in.** The capture format is a flat event
  stream with no envelope, so `trace_id`, `session_id`, the agent block, usage
  and outcome have nowhere to go. They are declared dropped, not attached to an
  event that never carried them.

## The `kind` slot takes a category, not a tool name

ACP's `kind` is a **category tag** — `benchflow.acp.types.ToolKind` says so in
its own docstring, `_canonical_tool_kind` defaults an absent one to `"other"`,
and the values seen in production (`execute`, `edit`, `fetch`, `think`) are
categories. ATIF's `function_name` and OTel's `gen_ai.tool.name` are names of
*particular tools*.

So a tool call is writable into that slot **only when
:attr:`~benchflow.trajectories.ir.ToolCall.name_semantics` is `"acp_kind"`**.
Any other label is a refusal, not a normalization: a normalization is a value a
reader can undo with convention knowledge, and there is nothing to undo here —
the record would simply assert a category nobody observed.

The string itself is never inspected. A `function_name` of `"read"` coincides
with a real `ToolKind` member and is still refused, because matching the
vocabulary by accident is not being drawn from it, and a rule that looked at the
value would admit exactly the cases most likely to be wrong.

## The one place an empty string is written for an absence

`title` — and only because the contract says so. Slice A's schema documents the
field as *"Human-readable label. Empty string when the agent supplied none."*,
so ``""`` is the contract's own representation of an absent title rather than
this converter's invention. It is still declared
:attr:`~benchflow.trajectories.ir.LossClass.SYNTHESIZED`, because a reader of
the output cannot tell that empty string from one the agent really supplied.

`tool_call_id` carries the *same* documentation — *"May be the empty string when
the agent omitted it"* — and is deliberately **not** treated the same way here.
Extending the argument to a second field is a decision worth taking explicitly
rather than by analogy; until then a tool call with no id is not representable.

## What is recovered from ``extensions``

Only the three `agent_timeout` fields, and only because
:func:`~benchflow.trajectories.ir_from_acp.acp_events_to_ir` puts them there by
name: ``timeout_sec``, ``pending_tool_call_ids`` and
``terminal_trajectory_complete`` are keys that edge preserved explicitly, so
reading them back is deterministic rather than a search. Each is type-checked
against what the schema requires; a missing or wrongly-typed one makes the event
not representable rather than defaulted.
"""

from __future__ import annotations

from typing import Any

from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    EventKind,
    LossClass,
    LossRecord,
    LossReport,
    PathSpace,
    ToolCall,
    ToolStatus,
    TraceEvent,
)

LOSS_DIRECTION = "ir->acp"

ACP_TEXT_TYPE: dict[EventKind, str] = {
    EventKind.USER_MESSAGE: "user_message",
    EventKind.AGENT_MESSAGE: "agent_message",
    EventKind.AGENT_REASONING: "agent_thought",
}
"""The three IR kinds Slice A's ``text_event.type`` enum admits."""

ACP_TOOL_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed", "failed", "cancelled"}
)
"""``tool_call_event.status`` — a CLOSED enum in the schema.

:attr:`~benchflow.trajectories.ir.ToolStatus.UNKNOWN` is deliberately absent: the
IR has a value for "the source carried a status this converter could not map",
and ACP has none.
"""

ACP_KIND_SEMANTICS = "acp_kind"
"""The only :attr:`~benchflow.trajectories.ir.ToolCall.name_semantics` value
that may be written into ACP's ``kind`` slot.

``ir_from_acp`` writes this label when it reads a capture record's ``kind``.
Any other label — ``function_name``, ``gen_ai.tool.name`` — says the name is
not a category, and this edge refuses rather than reinterpreting it.
"""

ACP_TIMEOUT_REASON = "wall_clock_timeout"
"""``agent_timeout_event.reason`` — a single-member enum. ``record_agent_timeout``
is its only producer and hardcodes this value."""

TIMEOUT_EXTENSION_KEYS: tuple[str, ...] = (
    "timeout_sec",
    "pending_tool_call_ids",
    "terminal_trajectory_complete",
)
"""The keys `ir_from_acp` preserves verbatim on a timeout event's extensions."""

# Key order matches ``_events_to_trajectory`` exactly. It costs nothing and it
# is what lets the round-trip anchor be checked on the serialized bytes as well
# as on the structures; see the module's test suite.
_TEXT_KEYS = ("type", "text")
_TOOL_CALL_KEYS = ("type", "tool_call_id", "kind", "title", "status", "content")
_TIMEOUT_KEYS = ("type", "reason", *TIMEOUT_EXTENSION_KEYS)


class AcpCaptureNotRepresentable(ValueError):
    """A trace cannot become ACP capture events without inventing a value.

    Carries the blockers as :class:`~benchflow.trajectories.ir.LossRecord`
    values so the failure names the field that caused it by path, in the same
    vocabulary every other edge of this family uses, instead of a bare message a
    caller would have to parse.
    """

    def __init__(self, blockers: list[LossRecord]) -> None:
        self.blockers: tuple[LossRecord, ...] = tuple(blockers)
        first = blockers[0] if blockers else None
        summary = (
            f"{len(blockers)} value(s) cannot be represented as ACP capture "
            f"events; first: {first.field} — {first.detail}"
            if first is not None
            else "trace is not representable as ACP capture events"
        )
        super().__init__(summary)


def acp_capture_blockers(trace: CanonicalTrace) -> list[LossRecord]:
    """What stops *trace* from being exportable, without raising.

    An empty list means :func:`ir_to_acp_capture_events` will succeed. Each
    record addresses the IR field responsible, in
    :attr:`~benchflow.trajectories.ir.PathSpace.HUB`, so a caller can report the
    obstruction without catching anything.
    """
    return _convert(trace)[1]


def ir_to_acp_capture_events(
    trace: CanonicalTrace,
) -> tuple[list[dict[str, Any]], LossReport]:
    """Build the ACP capture events for *trace*, or refuse.

    Returns the events **and** the report of what the target could not carry.
    The report is returned rather than attached: one trace may be converted to
    several targets and none of those conversions describes how the trace came
    to exist, so :attr:`~benchflow.trajectories.ir.CanonicalTrace.losses` is left
    alone and *trace* is not modified.

    Raises :class:`AcpCaptureNotRepresentable` when any event needs a value the
    IR does not have. Nothing partial is returned — see the module docstring.
    """
    events, blockers, losses = _convert(trace)
    if blockers:
        raise AcpCaptureNotRepresentable(blockers)
    return events, losses


def _convert(
    trace: CanonicalTrace,
) -> tuple[list[dict[str, Any]], list[LossRecord], LossReport]:
    """One pass: the events, what blocked, and what the target could not carry.

    Both public entry points go through here, so asking whether a trace is
    representable and converting it cannot disagree.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    blockers: list[LossRecord] = []
    events: list[dict[str, Any]] = []

    for event in trace.events:
        record, event_blockers = _event_to_acp(event, losses)
        blockers.extend(event_blockers)
        if record is not None:
            events.append(record)

    _declare_trace_level(trace, losses)
    return events, blockers, losses


def _blocker(field: str, detail: str) -> LossRecord:
    """A record that makes a conversion impossible rather than lossy.

    ``UNSUPPORTED`` is the honest class: the value is not in the trace, so no
    change to this converter could produce it. It is not ``DROPPED`` — nothing
    was discarded here — and it is emphatically not ``SYNTHESIZED``, which is
    the class this edge exists to avoid needing.
    """
    return LossRecord(
        field=field,
        space=PathSpace.HUB,
        loss_class=LossClass.UNSUPPORTED,
        detail=detail,
    )


def _event_to_acp(
    event: TraceEvent, losses: LossReport
) -> tuple[dict[str, Any] | None, list[LossRecord]]:
    """One IR event as one ACP capture record, or the reasons it cannot be."""
    where = f"events[{event.index}]"

    if event.kind in ACP_TEXT_TYPE:
        return _text_event(event, where, losses)
    if event.kind is EventKind.TOOL_CALL:
        return _tool_call_event(event, where, losses)
    if event.kind is EventKind.TIMEOUT:
        return _timeout_event(event, where, losses)

    # ORACLE and UNKNOWN. The schema models three record shapes and neither of
    # these is one of them; §2.4 documents the oracle record as something the
    # capture emitter never produces. Widening the schema, emitting an invalid
    # record, or dropping the event would each be a different way of pretending
    # the codomain is bigger than it is.
    return None, [
        _blocker(
            where,
            f"an event of kind {event.kind.value!r} has no ACP capture record "
            "shape; the Slice A contract defines text, tool_call and "
            "agent_timeout events only",
        )
    ]


def _text_event(
    event: TraceEvent, where: str, losses: LossReport
) -> tuple[dict[str, Any] | None, list[LossRecord]]:
    """A user / agent / thought record.

    The schema requires ``text`` and documents ``""`` as a value the capture
    path really records — an unconditionally captured empty prompt — rather than
    as the representation of an absent one. So an empty string passes through as
    the observation it is, and a ``None`` is a blocker rather than an empty
    string: writing one would turn "the source carried no text" into "the source
    carried empty text", which is exactly the tri-state collapse §8.2 forbids.
    """
    is_reasoning = event.kind is EventKind.AGENT_REASONING
    source_field = "reasoning" if is_reasoning else "text"
    value = event.reasoning if is_reasoning else event.text

    if value is None:
        return None, [
            _blocker(
                f"{where}.{source_field}",
                f"a {ACP_TEXT_TYPE[event.kind]} record requires text and the "
                f"event carries none; the contract documents the empty string as "
                "an observed value, not as a way to write an absence",
            )
        ]

    _declare_event_level(event, where, losses)
    if is_reasoning and event.text is not None:
        losses.add(
            f"{where}.text",
            LossClass.DROPPED,
            "an agent_thought record carries only its reasoning text; the ACP "
            "capture format has no second text slot on the same event",
        )
    if (
        is_reasoning
        and event.reasoning_segments is not None
        and len(event.reasoning_segments) > 1
    ):
        losses.add(
            f"{where}.reasoning_segments",
            LossClass.DROPPED,
            f"{len(event.reasoning_segments)} thought segments become one "
            "agent_thought record; the boundary between them is not "
            "representable, which is §5 loss #10 in the writing direction",
        )
    return dict(zip(_TEXT_KEYS, (ACP_TEXT_TYPE[event.kind], value), strict=True)), []


def _tool_call_event(
    event: TraceEvent, where: str, losses: LossReport
) -> tuple[dict[str, Any] | None, list[LossRecord]]:
    """A tool-call record, or every reason it cannot be built.

    All blockers are collected rather than short-circuiting on the first: a
    caller fixing a producer wants the whole list, and reporting one field at a
    time would make an unrepresentable trace look like a sequence of small
    problems.
    """
    call = event.tool_call
    if call is None:  # pragma: no cover - invariant 3 forbids it
        return None, [
            _blocker(f"{where}.tool_call", "tool_call event with no tool_call payload")
        ]

    blockers: list[LossRecord] = []

    status = _acp_status(call, where, blockers)
    call_id = _acp_call_id(call, where, blockers)
    kind = _acp_kind(call, where, blockers)
    content = _acp_content(call, where, blockers)

    if blockers:
        return None, blockers

    _declare_event_level(event, where, losses)
    title = _acp_title(call, where, losses)
    _declare_tool_call_losses(event, call, where, losses)

    return (
        dict(
            zip(
                _TOOL_CALL_KEYS,
                ("tool_call", call_id, kind, title, status, content),
                strict=True,
            )
        ),
        [],
    )


def _acp_status(call: ToolCall, where: str, blockers: list[LossRecord]) -> str | None:
    """The one field this edge will never invent."""
    if call.status is not None and call.status.value in ACP_TOOL_STATUSES:
        return call.status.value
    observed = (
        "no status"
        if call.status is None
        else f"the status {call.status.value!r}, which ACP has no member for"
    )
    blockers.append(
        _blocker(
            f"{where}.tool_call.status",
            f"the tool call carries {observed}; the ACP vocabulary is "
            f"{sorted(ACP_TOOL_STATUSES)} and every member asserts something "
            "about the call's lifecycle, so none of them can stand in for an "
            "unknown one",
        )
    )
    if call.status is ToolStatus.UNKNOWN:
        # Worth its own sentence: this is not a gap in the trace, it is the IR
        # saying something ACP cannot say.
        blockers[-1] = _blocker(
            f"{where}.tool_call.status",
            "the IR records this status as UNKNOWN — the source carried one this "
            "family could not map — and ACP's closed enum has no member with "
            "that meaning; writing any of the five would assert a lifecycle "
            "state nobody observed",
        )
    return None


def _acp_call_id(call: ToolCall, where: str, blockers: list[LossRecord]) -> str | None:
    """``tool_call_id``, which is required and is not defaulted here.

    The schema documents ``""`` as what the capture path records when the agent
    omitted an id, which is the same argument that lets ``title`` be written
    empty. It is not applied here on purpose — see the module docstring.
    """
    if call.call_id is not None:
        return call.call_id
    blockers.append(
        _blocker(
            f"{where}.tool_call.call_id",
            "the tool call carries no id; the contract documents the empty "
            "string for an id the agent omitted, but this edge does not write "
            "one for an id the trace never had",
        )
    )
    return None


def _acp_kind(call: ToolCall, where: str, blockers: list[LossRecord]) -> str | None:
    """``kind`` — required, and only writable from a name that *is* an ACP kind.

    An ACP ``kind`` is a **category**, not a tool name.
    ``benchflow.acp.types.ToolKind`` calls itself "Category tag for tool calls,
    used for metrics and trajectory display", `_canonical_tool_kind` defaults an
    absent one to ``"other"``, and the values seen in production — ``execute``,
    ``edit``, ``fetch``, ``think`` — are categories too.

    ATIF's ``function_name`` and OTel's ``gen_ai.tool.name`` are names of
    *particular tools*. Writing one into this slot would not be a normalization,
    which is a value a reader can undo with convention knowledge; it would be a
    **reinterpretation** — the record would assert a category that was never
    observed, and nothing downstream could tell. The IR carries
    :attr:`~benchflow.trajectories.ir.ToolCall.name_semantics` precisely so this
    edge does not have to guess, and refusing is what makes that field
    load-bearing rather than decorative.

    So the representability rule reads the semantics, not the string:

    - no ``name`` — nothing to write, and the contract documents no meaning for
      an empty ``kind`` the way it does for an empty ``title``;
    - no ``name_semantics`` — the trace does not say what kind of name it holds,
      and an unlabelled name is not evidence of a category;
    - ``name_semantics`` other than ``"acp_kind"`` — the trace says explicitly
      that this is *not* an ACP kind.

    **The string is never inspected.** A ``function_name`` of ``"read"``
    coincides with a real ``ToolKind`` member and is still refused: matching the
    vocabulary by accident is not the same as being drawn from it, and a rule
    that looked at the value would silently admit exactly the cases most likely
    to be wrong. Provenance is not consulted either — ``name_semantics`` is
    trace data, and a hand-built trace that labels its names ``acp_kind``
    exports whatever its ``source_format`` says.
    """
    # Only this function's own findings decide its return value; the list is
    # shared with the other field readers and may already hold theirs.
    before = len(blockers)
    if call.name is None:
        blockers.append(
            _blocker(
                f"{where}.tool_call.name",
                "the tool call carries no name and ACP requires a kind; unlike "
                "title, the contract documents no meaning for an empty one",
            )
        )
    if call.name_semantics is None:
        blockers.append(
            _blocker(
                f"{where}.tool_call.name_semantics",
                "the trace does not say what kind of name this is, and ACP's "
                "kind is a category rather than a tool name; an unlabelled name "
                "is not evidence that it was drawn from that vocabulary",
            )
        )
    elif call.name_semantics != ACP_KIND_SEMANTICS:
        blockers.append(
            _blocker(
                f"{where}.tool_call.name_semantics",
                f"the name is a {call.name_semantics!r}, and ACP's kind is a "
                "category tag; writing it into that slot would assert a category "
                "nobody observed, which is a reinterpretation rather than a "
                "normalization a reader could undo",
            )
        )
    return None if len(blockers) > before else call.name


def _acp_title(call: ToolCall, where: str, losses: LossReport) -> str:
    """``title``, defaulting to ``""`` **because the contract says so**.

    Slice A documents the field as "Empty string when the agent supplied none",
    so the empty string is the contract's representation of an absent title
    rather than this converter's invention. It is declared ``SYNTHESIZED``
    anyway: a reader of the output cannot distinguish it from an empty title the
    agent really supplied, and that indistinguishability is the loss.
    """
    if call.title is not None:
        return call.title
    losses.add(
        f"{where}.tool_call.title",
        LossClass.SYNTHESIZED,
        "the trace carries no title and ACP requires one; the empty string is "
        "the contract's own representation of an absent title, and is written "
        "here as such — a reader cannot tell it from an observed empty title",
    )
    return ""


def _acp_content(
    call: ToolCall, where: str, blockers: list[LossRecord]
) -> list[dict[str, Any]] | None:
    """The captured output blocks, verbatim.

    ACP stores ``session/update.content`` as the wire delivered it and the
    schema keeps ``content_block`` permissive for exactly that reason, so the
    only faithful thing to write is the block the IR preserved in
    :attr:`~benchflow.trajectories.ir.ContentBlock.raw`. A block with no ``raw``
    carries text but no wire shape, and choosing one — the nested ACP form or
    the flat form, both of which `content_blocks_to_text` accepts — would be
    this converter inventing structure the source never had.

    An empty list is not a blocker: the schema documents it as what a call with
    no output records.
    """
    blocks: list[dict[str, Any]] = []
    for position, block in enumerate(call.content):
        raw = _raw_block(block)
        if raw is None:
            blockers.append(
                _blocker(
                    f"{where}.tool_call.content[{position}].raw",
                    "the content block carries rendered text but not the source "
                    "block it came from; ACP stores wire blocks verbatim and "
                    "there is no single shape to wrap the text in",
                )
            )
            continue
        blocks.append(raw)
    return None if blockers else blocks


def _raw_block(block: ContentBlock) -> dict[str, Any] | None:
    return block.raw


def _declare_event_level(event: TraceEvent, where: str, losses: LossReport) -> None:
    """IR event fields no ACP record shape has a slot for.

    Declared only when the event actually carries the value — the outbound rule
    Slice D adopted: an edge declares what it loses given the trace in hand,
    not what it would lose given a fuller one.
    """
    for field in ("started_at", "finished_at", "usage", "role", "source_type"):
        if getattr(event, field) is not None:
            losses.add(
                f"{where}.{field}",
                LossClass.DROPPED,
                f"ACP capture records have no {field} slot",
            )
    if event.extensions and event.kind is not EventKind.TIMEOUT:
        losses.add(
            f"{where}.extensions",
            LossClass.DROPPED,
            "the record shapes forbid additional properties, so carried source "
            "fields have nowhere to go",
        )


def _declare_tool_call_losses(
    event: TraceEvent, call: ToolCall, where: str, losses: LossReport
) -> None:
    """What a tool-call record cannot carry, given this call."""
    if call.arguments is not None:
        losses.add(
            f"{where}.tool_call.arguments",
            LossClass.DROPPED,
            "the ACP capture record has no arguments field; ACPSession."
            "handle_update never read rawInput, so the format never grew one",
            "§5 loss #1",
        )
    if call.name_semantics is not None:
        losses.add(
            f"{where}.tool_call.name_semantics",
            LossClass.NORMALIZED,
            f"{call.name_semantics!r} is not written anywhere in the record; it "
            "is recoverable only from the convention that an ACP kind is an ACP "
            "kind, which is what reading the file as ACP already assumes",
        )
    for field in ("started_at", "finished_at"):
        if getattr(call, field) is not None:
            losses.add(
                f"{where}.tool_call.{field}",
                LossClass.DROPPED,
                "ToolCallRecord tracks both in memory and the capture format "
                "serializes neither",
                "§5 loss #3",
            )
    for position, block in enumerate(call.content):
        if block.raw is not None and block.text is not None:
            # The block is written verbatim, so nothing is lost — but the IR's
            # classification of it is, and a consumer re-reading the record has
            # to re-derive text from the wire shape.
            losses.add(
                f"{where}.tool_call.content[{position}].kind",
                LossClass.NORMALIZED,
                "the block is written verbatim; the IR's text/opaque "
                "classification is not part of the wire shape and is re-derived "
                "by whoever reads it back",
            )
    if event.text is not None:
        losses.add(
            f"{where}.text",
            LossClass.DROPPED,
            "a tool_call record has no text slot; a step's message rides on the "
            "event in some source formats and cannot here",
        )


def _timeout_event(
    event: TraceEvent, where: str, losses: LossReport
) -> tuple[dict[str, Any] | None, list[LossRecord]]:
    """An ``agent_timeout`` record, rebuilt from the fields the inbound edge kept.

    Four required values, and every one of them is checked rather than
    defaulted. ``reason`` is a single-member enum, so an outcome that is not
    exactly ``wall_clock_timeout`` cannot be written as one; the other three are
    read from ``extensions`` by the names
    :func:`~benchflow.trajectories.ir_from_acp.acp_events_to_ir` preserved them
    under, which makes the recovery deterministic rather than a search.
    """
    blockers: list[LossRecord] = []

    if event.outcome != ACP_TIMEOUT_REASON:
        blockers.append(
            _blocker(
                f"{where}.outcome",
                f"agent_timeout.reason is a single-member enum "
                f"({ACP_TIMEOUT_REASON!r}) and the event's outcome is "
                f"{event.outcome!r}; there is no other reason the record can state",
            )
        )

    values: dict[str, Any] = {}
    checks = {
        "timeout_sec": (
            lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
            "a number",
        ),
        "pending_tool_call_ids": (lambda v: isinstance(v, list), "an array"),
        "terminal_trajectory_complete": (lambda v: isinstance(v, bool), "a boolean"),
    }
    for key, (ok, expected) in checks.items():
        if key not in event.extensions:
            blockers.append(
                _blocker(
                    f"{where}.extensions.{key}",
                    f"agent_timeout requires {key} and the event carries none; "
                    "it is preserved by name when the trace came through the ACP "
                    "edge, and inventing a budget that was never observed is not "
                    "a substitute",
                )
            )
            continue
        value = event.extensions[key]
        if not ok(value):
            blockers.append(
                _blocker(
                    f"{where}.extensions.{key}",
                    f"{key} is {type(value).__name__} and the contract requires "
                    f"{expected}",
                )
            )
            continue
        values[key] = value

    if blockers:
        return None, blockers

    _declare_event_level(event, where, losses)
    unused = set(event.extensions) - set(TIMEOUT_EXTENSION_KEYS)
    if unused:
        losses.add(
            f"{where}.extensions",
            LossClass.DROPPED,
            "extension keys with no agent_timeout field: " + ", ".join(sorted(unused)),
        )
    return (
        dict(
            zip(
                _TIMEOUT_KEYS,
                (
                    "agent_timeout",
                    ACP_TIMEOUT_REASON,
                    *(values[key] for key in TIMEOUT_EXTENSION_KEYS),
                ),
                strict=True,
            )
        ),
        [],
    )


def _declare_trace_level(trace: CanonicalTrace, losses: LossReport) -> None:
    """Everything the capture format has no envelope for.

    The target is a flat stream of event records. There is no header, no
    trailer and no per-file object, so this is not a list of fields ACP happens
    to lack — it is the whole of the trace level, and no future ACP record shape
    could take it without the artifact growing a concept it does not have.

    Declared only for values this trace actually carries, per the outbound rule.
    """
    for field in ("trace_id", "session_id", "started_at", "finished_at"):
        if getattr(trace, field) is not None:
            losses.add(
                field,
                LossClass.DROPPED,
                "the ACP capture format is a flat event stream with no envelope "
                "to carry a trace-level value",
            )
    for field, value in trace.agent.model_dump().items():
        if value is not None:
            losses.add(
                f"agent.{field}",
                LossClass.DROPPED,
                "no capture record identifies the agent; the artifact's readers "
                "take that from result.json",
            )
    if trace.usage is not None:
        losses.add(
            "usage",
            LossClass.DROPPED,
            "token accounting is routed to result.json and has never been part "
            "of the capture event stream",
            "§5 losses #6, #8",
        )
    for field, value in trace.outcome.model_dump().items():
        if value is not None:
            losses.add(
                f"outcome.{field}",
                LossClass.DROPPED,
                "the capture stream has no run-outcome record; a timeout is "
                "expressible only as its own event",
            )
    if trace.extensions:
        losses.add(
            "extensions",
            LossClass.DROPPED,
            "trace-level extensions have no envelope to ride in",
        )
