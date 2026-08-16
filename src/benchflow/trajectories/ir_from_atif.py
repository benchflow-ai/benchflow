"""ATIF → canonical Trace IR — the inbound ATIF edge.

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, itself an
> unapproved proposal (`docs/trace-interop.md` §8). Nothing imports this module,
> nothing writes its output to disk, and `export_atif.py` is untouched and still
> the only writer of `trainer/atif.json`.

This closes the ATIF pair: :mod:`benchflow.trajectories.ir_to_atif` writes the
document, this reads one back. Together they make the round trip
`ACP → IR → ATIF → IR′` measurable, which is the question
:mod:`benchflow.trajectories.ir_round_trip` exists to answer — *how much of a
trace survives a trip through the interchange format?*

## The rule this module is built on

**Read what the document says, never what it probably meant.**

An ATIF document is the output of a lossy conversion, and several of its values
were fabricated by the converter that wrote it: `agent.version` is the literal
`"unknown"` whenever BenchFlow had no version, `arguments` is `{}` for every
ACP-derived call, `message` is `""` on any step carrying only a tool call.
Reading those back as "absent" would make the round trip look better than it is
— the converter would be *guessing* which values were real, and a guess that
happens to be right is still a guess.

So this edge takes every value verbatim. `"unknown"` becomes the agent version
`"unknown"`; `{}` becomes an observed empty argument map; `""` becomes observed
empty text. The consequence is the most interesting thing the round trip
measures, and it is not a subtraction:

**A fabricated value returns indistinguishable from an observed one.** On the
way out, `ir_to_atif` declares `arguments` ``SYNTHESIZED`` — "the target
required this and the source never had it". On the way back, the document says
`{}` and nothing marks it as invented, so the reconstructed trace states, with
the full authority of the IR's tri-state contract, that the tool was observed to
be called with no arguments. The information did not merely disappear; it was
*replaced by a false statement of the same shape*. Only the pair of loss reports
carries the truth, and only for the edges that had one.

That is a fact about ATIF, not a defect of this converter, and it is the reason
the round trip is worth measuring rather than asserting.

## What has no ATIF antecedent

ATIF carries no trace id, no timestamps at any level, no run outcome, no
provider, and no per-event provenance. Each is declared ``UNSUPPORTED`` — the
source never carried it — exactly as the ACP edge declares what the capture
events never carried. ``DROPPED`` would be a lie: there is nothing in the
document to drop.

## Fusion, and why no boundary is invented

`ir_to_atif` folds a run of reasoning events into the *next* agent step's
`reasoning_content`, joined by a blank line. Reading back, one step is one
event: a step with both `message` and `reasoning_content` becomes one
``AGENT_MESSAGE`` carrying both, not a reasoning event followed by a message
event. Splitting it would invent a boundary the document does not contain — the
join is not injective, which is `§5 loss #10` — so the fusion is reported as
what it is: fewer events out than in.

The one place a step legitimately becomes more than one event is a step with
several `tool_calls`. The IR models one tool call per event (invariant 3), so an
*n*-call step becomes *n* events. `trajectory_to_atif_record` never writes such
a step; a document from another producer can.

## Step metrics are carried, not interpreted

ATIF allows `metrics` on an agent step. This repository never writes one, and no
ATIF schema is vendored here to read its vocabulary against, so a step's
`metrics` object is carried verbatim into ``extensions`` rather than mapped onto
:class:`~benchflow.trajectories.ir.TraceUsage`. Mapping it would be asserting a
field correspondence nobody has checked. `final_metrics` *is* mapped: this
repository writes it, and `ir_to_atif` pins the four keys.
"""

from __future__ import annotations

from typing import Any

from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossReport,
    ModelInfo,
    PathSpace,
    Provenance,
    Role,
    ToolCall,
    ToolStatus,
    TraceEvent,
    TraceOutcome,
    TraceUsage,
)

LOSS_DIRECTION = "atif->ir"

ATIF_SOURCE = "atif"
"""Trace- and event-level provenance for anything read out of an ATIF document."""

_STATUS_BY_VALUE = {status.value: status for status in ToolStatus}

# ``source`` values ``ir_to_atif`` and ``export_atif`` emit. A step whose source
# is outside this map becomes an ``UNKNOWN`` event with the string kept, rather
# than being dropped the way every exporter drops what it does not recognize.
_ROLE_BY_SOURCE: dict[str, Role] = {
    "user": Role.USER,
    "agent": Role.AGENT,
    "oracle": Role.ORACLE,
}

# Keys the two writers in this repository produce. Anything else on a recognized
# object rides into ``extensions`` verbatim instead of being discarded.
_DOCUMENT_KEYS = frozenset(
    {"schema_version", "session_id", "agent", "steps", "final_metrics"}
)
_AGENT_KEYS = frozenset({"name", "version", "model_name"})
_STEP_KEYS = frozenset(
    {
        "step_id",
        "source",
        "message",
        "reasoning_content",
        "tool_calls",
        "observation",
        "metrics",
    }
)
_TOOL_CALL_KEYS = frozenset({"tool_call_id", "function_name", "arguments", "extra"})

# ``final_metrics`` keys with an IR home, in the direction ``ir_to_atif`` writes
# them. Inverting the same tuple in both modules is what keeps the pair honest.
_USAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_prompt_tokens", "input_tokens"),
    ("total_completion_tokens", "output_tokens"),
    ("total_cached_tokens", "cache_read_tokens"),
    ("total_cost_usd", "cost_usd"),
)

# IR usage fields no ATIF document carries.
_USAGE_UNSUPPORTED: tuple[str, ...] = (
    "cache_creation_tokens",
    "reasoning_tokens",
    "total_tokens",
    "source",
    "price_source",
)


def _extras(raw: dict[str, Any], known: frozenset[str]) -> dict[str, Any]:
    """Keys of *raw* outside *known*, carried verbatim."""
    return {key: value for key, value in raw.items() if key not in known}


def _coerce_str(
    raw: dict[str, Any],
    key: str,
    field: str,
    losses: LossReport,
) -> str | None:
    """Read a string field, declaring any coercion this module performs.

    Absence returns ``None`` and declares nothing: the field simply is not in
    the document, and the systemic records cover what ATIF never carries. A
    non-string *present* value is this module's own reshaping act, so it is
    declared — the same rule :mod:`ir_from_acp` follows.
    """
    if key not in raw:
        return None
    value = raw[key]
    if isinstance(value, str):
        return value
    losses.add(
        field,
        LossClass.NORMALIZED,
        f"document carries {type(value).__name__} where ATIF specifies a string; "
        "coerced with str()",
    )
    return str(value)


def _int_or_none(value: Any) -> int | None:
    """An int, or ``None`` for anything that is not one.

    ``bool`` is excluded on purpose: it is an ``int`` subclass in Python, and a
    token count of ``True`` is a malformed document, not the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _content_blocks(
    results: list[Any],
    call_id: str | None,
    where: str,
    losses: LossReport,
) -> list[ContentBlock]:
    """Observation results belonging to *call_id*, as IR content blocks.

    Every block is ``TEXT`` with ``raw=None``: ATIF stores rendered content, so
    the source block `ir_to_atif` declared dropped is genuinely not there to
    reconstruct.
    """
    blocks: list[ContentBlock] = []
    for position, result in enumerate(results):
        if not isinstance(result, dict) or result.get("source_call_id") != call_id:
            continue
        content = result.get("content")
        if not isinstance(content, str):
            losses.add(
                f"{where}.tool_call.content[{len(blocks)}]",
                LossClass.NORMALIZED,
                f"observation.results[{position}].content is "
                f"{type(content).__name__}, not a string; coerced with str()",
            )
            content = str(content)
        blocks.append(ContentBlock(kind=ContentBlockKind.TEXT, text=content))
    return blocks


def _tool_call_to_ir(
    raw: dict[str, Any],
    where: str,
    losses: LossReport,
) -> tuple[ToolCall, dict[str, Any]]:
    """One ATIF ``tool_calls`` entry as an IR tool call, and its unmapped keys.

    :class:`~benchflow.trajectories.ir.ToolCall` has no ``extensions`` of its
    own, so the leftovers are returned for the caller to put on the event, under
    a ``tool_call`` key — merging them into the event's own extensions would let
    a step key and a tool-call key of the same name collide silently.

    ``arguments`` is taken verbatim, including the ``{}`` that `ir_to_atif`
    writes for every ACP-derived call. This is the module's central rule and its
    most consequential application: reading ``{}`` back as ``None`` would
    reconstruct the ACP-side truth by guessing, and would make the round trip
    appear to preserve a distinction the document does not carry.
    """
    extra = raw.get("extra")
    extra_map = extra if isinstance(extra, dict) else {}

    status: ToolStatus | None = None
    extras = _extras(raw, _TOOL_CALL_KEYS)
    raw_status = extra_map.get("status")
    if raw_status is not None:
        status = _STATUS_BY_VALUE.get(str(raw_status))
        if status is None:
            status = ToolStatus.UNKNOWN
            extras["source_status"] = raw_status
            losses.add(
                f"{where}.tool_call.status",
                LossClass.NORMALIZED,
                f"status {raw_status!r} is outside the IR vocabulary; recorded as "
                "unknown with the original kept in extensions",
            )

    arguments = raw.get("arguments")
    if "arguments" not in raw:
        # ATIF requires the field; a document without it is non-conformant, and
        # invariant 7 requires the resulting ``None`` to be declared.
        losses.add(
            f"{where}.tool_call.arguments",
            LossClass.UNSUPPORTED,
            "ATIF requires tool_calls[].arguments and this document omits it, so "
            "the call carries no arguments to read",
        )
        arguments = None
    elif not isinstance(arguments, dict):
        losses.add(
            f"{where}.tool_call.arguments",
            LossClass.UNSUPPORTED,
            f"arguments is {type(arguments).__name__}, not an object; the IR "
            "models arguments as a mapping and cannot carry this value",
        )
        extras["source_arguments"] = arguments
        arguments = None

    unmapped_extra = {
        key: value for key, value in extra_map.items() if key not in ("title", "status")
    }
    if unmapped_extra:
        extras["extra"] = unmapped_extra

    title = extra_map.get("title")
    tool_call = ToolCall(
        call_id=_coerce_str(raw, "tool_call_id", f"{where}.tool_call.call_id", losses),
        name=_coerce_str(raw, "function_name", f"{where}.tool_call.name", losses),
        # What the document says it is. The ACP edge writes ``"acp_kind"`` for
        # the same value, and the two disagreeing is exactly the normalization
        # `ir_to_atif` declared on the way out: ATIF has one slot and no way to
        # say the string in it is a category rather than a function name.
        name_semantics="function_name" if "function_name" in raw else None,
        title=str(title) if title is not None else None,
        status=status,
        arguments=arguments,
    )
    return tool_call, extras


def atif_to_ir(document: dict[str, Any]) -> CanonicalTrace:
    """Convert one ATIF document into a :class:`CanonicalTrace`.

    Everything comes from *document*: no rollout artifact is consulted, so a
    value that lives only in `result.json` or `timing.json` is a declared loss
    rather than a silent enrichment — the same discipline
    :func:`~benchflow.trajectories.ir_from_acp.acp_events_to_ir` follows.

    The returned trace carries its own :class:`LossReport` on
    :attr:`~benchflow.trajectories.ir.CanonicalTrace.losses` and satisfies
    :func:`~benchflow.trajectories.ir.validate_trace` for any input, including
    documents ATIF's own validator would reject.

    Raises ``TypeError`` for a non-mapping document. There is no useful partial
    reading of something that is not an ATIF document at all, and returning an
    empty trace would claim a conversion happened.
    """
    if not isinstance(document, dict):
        raise TypeError(
            f"ATIF document must be a mapping, got {type(document).__name__}"
        )

    losses = LossReport(direction=LOSS_DIRECTION)
    ir_events: list[TraceEvent] = []

    raw_steps = document.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    if raw_steps is not None and not isinstance(raw_steps, list):
        losses.add(
            "steps",
            LossClass.DROPPED,
            f"steps is {type(raw_steps).__name__}, not a list; no event could be "
            "read from it",
            space=PathSpace.SOURCE,
        )

    had_tool_call = False
    for position, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            losses.add(
                f"steps[{position}]",
                LossClass.DROPPED,
                f"step is {type(raw_step).__name__}, not a JSON object; the IR has "
                "no representation for it",
                space=PathSpace.SOURCE,
            )
            continue
        events = _step_to_events(raw_step, len(ir_events), losses)
        had_tool_call = had_tool_call or any(
            event.kind is EventKind.TOOL_CALL for event in events
        )
        ir_events.extend(events)

    usage = _read_usage(document, losses)
    agent = _read_agent(document, losses)
    _declare_systemic_losses(losses, had_tool_call=had_tool_call, has_usage=usage)

    extensions = _extras(document, _DOCUMENT_KEYS)
    schema_version = document.get("schema_version")
    if schema_version is not None:
        # Kept because it identifies the dialect this trace was read from, and
        # the IR has no field for a source format's own version.
        extensions["schema_version"] = schema_version

    return CanonicalTrace(
        session_id=_coerce_str(document, "session_id", "session_id", losses),
        agent=agent,
        events=ir_events,
        usage=usage,
        # Every field stays ``None``: ATIF has no outcome section, so there is
        # nothing to read. The section itself is present because
        # ``outcome.stop_reason`` is declared below and a record cannot address
        # a path through a null parent.
        outcome=TraceOutcome(),
        provenance=Provenance(source_format=ATIF_SOURCE),
        extensions=extensions,
        losses=losses,
    )


def _step_to_events(
    raw_step: dict[str, Any],
    first_index: int,
    losses: LossReport,
) -> list[TraceEvent]:
    """One ATIF step as one IR event — or *n* for an *n*-tool-call step."""
    where = f"events[{first_index}]"
    source = raw_step.get("source")
    source_str = source if isinstance(source, str) else None
    role = _ROLE_BY_SOURCE.get(source_str or "")

    message = _coerce_str(raw_step, "message", f"{where}.text", losses)
    reasoning = _coerce_str(raw_step, "reasoning_content", f"{where}.reasoning", losses)

    extensions = _extras(raw_step, _STEP_KEYS)
    if "step_id" in raw_step:
        # Carried, not mapped onto ``index``: invariant 2 makes the IR index a
        # dense position, and a document whose step_ids are sparse or restarted
        # would otherwise be silently renumbered with no trace of the original.
        extensions["step_id"] = raw_step["step_id"]
    if "metrics" in raw_step:
        extensions["metrics"] = raw_step["metrics"]
        losses.add(
            f"{where}.usage",
            LossClass.NORMALIZED,
            "step metrics are carried verbatim into extensions; no ATIF schema is "
            "vendored here to map their vocabulary onto TraceUsage",
        )

    raw_calls = raw_step.get("tool_calls")
    calls = (
        [call for call in raw_calls if isinstance(call, dict)]
        if isinstance(raw_calls, list)
        else []
    )
    if isinstance(raw_calls, list) and len(calls) != len(raw_calls):
        losses.add(
            f"steps[{first_index}].tool_calls",
            LossClass.DROPPED,
            "the step carries tool_calls entries that are not JSON objects",
            space=PathSpace.SOURCE,
        )

    observation = raw_step.get("observation")
    results = (
        observation.get("results")
        if isinstance(observation, dict)
        and isinstance(observation.get("results"), list)
        else []
    )

    if source_str is not None and role is None:
        # Outside the vocabulary this repository writes. Kept whole rather than
        # forced into a role the document does not support.
        return [
            TraceEvent(
                index=first_index,
                kind=EventKind.UNKNOWN,
                source_type=source_str,
                text=message,
                reasoning=reasoning,
                provenance=Provenance(source_format=ATIF_SOURCE),
                extensions={**extensions, **_unread(raw_step)},
            )
        ]

    if calls:
        return _tool_call_events(
            calls,
            results,
            first_index=first_index,
            source_str=source_str,
            role=role,
            message=message,
            reasoning=reasoning,
            extensions=extensions,
            losses=losses,
        )

    if results:
        # An observation with nothing to attach it to: the IR models tool output
        # under a tool call, so it rides verbatim rather than being dropped.
        extensions["observation"] = observation
        losses.add(
            f"{where}.tool_call",
            LossClass.NORMALIZED,
            "the step carries an observation but no tool call; the IR models "
            "content blocks under a tool call, so it is kept in extensions",
        )

    kind = _kind_for_step(role, message, reasoning)
    return [
        TraceEvent(
            index=first_index,
            kind=kind,
            source_type=source_str,
            role=role,
            text=message,
            reasoning=reasoning,
            # One step is one segment. `ThoughtBuffer` joined an unknown number
            # of thoughts with a blank line and the join is not injective, so
            # splitting on blank lines would invent boundaries — §5 loss #10 is
            # exactly this, and it is not recoverable here.
            reasoning_segments=[reasoning] if reasoning is not None else None,
            provenance=Provenance(source_format=ATIF_SOURCE),
            extensions=extensions,
        )
    ]


def _kind_for_step(
    role: Role | None, message: str | None, reasoning: str | None
) -> EventKind:
    """The event kind a non-tool step maps to.

    A step with reasoning and no message is what `ir_to_atif` writes when it
    flushes buffered thoughts, so it reads back as reasoning. A step with both
    is one event carrying both — the fusion is reported, not undone.
    """
    if role is Role.USER:
        return EventKind.USER_MESSAGE
    if role is Role.ORACLE:
        return EventKind.ORACLE
    if reasoning is not None and not message:
        return EventKind.AGENT_REASONING
    return EventKind.AGENT_MESSAGE


def _tool_call_events(
    calls: list[dict[str, Any]],
    results: list[Any],
    *,
    first_index: int,
    source_str: str | None,
    role: Role | None,
    message: str | None,
    reasoning: str | None,
    extensions: dict[str, Any],
    losses: LossReport,
) -> list[TraceEvent]:
    """The events for a step carrying tool calls, one per call.

    Text, reasoning and the step's extra keys ride on the **first** event only.
    Copying them onto each event of a multi-call step would multiply one
    observation into several; the alternative — dropping them — would lose the
    step's message. A multi-call step is declared normalized, since one source
    object became several IR events.
    """
    if len(calls) > 1:
        losses.add(
            f"steps[{first_index}]",
            LossClass.NORMALIZED,
            f"the step carries {len(calls)} tool calls and the IR models one per "
            "event, so it became that many events; the grouping is not preserved",
            space=PathSpace.SOURCE,
        )

    events: list[TraceEvent] = []
    for offset, raw_call in enumerate(calls):
        index = first_index + offset
        where = f"events[{index}]"
        tool_call, call_extras = _tool_call_to_ir(raw_call, where, losses)
        tool_call.content = _content_blocks(results, tool_call.call_id, where, losses)

        event_extensions = dict(extensions) if offset == 0 else {}
        if call_extras:
            event_extensions["tool_call"] = call_extras
        events.append(
            TraceEvent(
                index=index,
                kind=EventKind.TOOL_CALL,
                source_type=source_str,
                role=role,
                text=message if offset == 0 else None,
                reasoning=reasoning if offset == 0 else None,
                reasoning_segments=(
                    [reasoning] if offset == 0 and reasoning is not None else None
                ),
                tool_call=tool_call,
                provenance=Provenance(source_format=ATIF_SOURCE),
                extensions=event_extensions,
            )
        )

    # Results addressing no call in this step. ATIF resolves ``source_call_id``
    # within one step, so these are malformed rather than cross-step references;
    # they ride on the first event so they stay in the trace.
    known_ids = {
        event.tool_call.call_id for event in events if event.tool_call is not None
    }
    unmatched = [
        result
        for result in results
        if not (isinstance(result, dict) and result.get("source_call_id") in known_ids)
    ]
    if unmatched and events:
        events[0].extensions["unmatched_observation_results"] = unmatched
        losses.add(
            f"steps[{first_index}].observation.results",
            LossClass.NORMALIZED,
            "results whose source_call_id matches no tool call in the step; kept "
            "verbatim in extensions rather than attached to a call they do not "
            "belong to",
            space=PathSpace.SOURCE,
        )
    return events


def _unread(raw_step: dict[str, Any]) -> dict[str, Any]:
    """The recognized-but-unmapped keys of an unknown-source step.

    An unknown step keeps its whole body, since this converter cannot say which
    parts it understood.
    """
    return {
        key: raw_step[key] for key in ("tool_calls", "observation") if key in raw_step
    }


def _read_agent(document: dict[str, Any], losses: LossReport) -> ModelInfo:
    """The agent block, verbatim.

    ``"unknown"`` is **not** translated back to ``None``. `ir_to_atif` writes
    that literal whenever the trace had no name or version, but a document can
    equally carry it as an observed value, and this converter has no way to tell
    the two apart. Guessing would be the one thing this module refuses to do —
    see the module docstring.
    """
    raw_agent = document.get("agent")
    if not isinstance(raw_agent, dict):
        if raw_agent is not None:
            losses.add(
                "agent",
                LossClass.DROPPED,
                f"agent is {type(raw_agent).__name__}, not an object",
                space=PathSpace.SOURCE,
            )
        return ModelInfo()
    return ModelInfo(
        agent_name=_coerce_str(raw_agent, "name", "agent.agent_name", losses),
        agent_version=_coerce_str(raw_agent, "version", "agent.agent_version", losses),
        model=_coerce_str(raw_agent, "model_name", "agent.model", losses),
    )


def _read_usage(document: dict[str, Any], losses: LossReport) -> TraceUsage | None:
    """Trace usage from ``final_metrics``, or ``None`` when it carries none.

    ``total_steps`` is a property of the document, not of the run: `ir_to_atif`
    declares it ``SYNTHESIZED`` in the target space on the way out, and it is
    declared dropped in the source space here. The two records are the same fact
    from the two sides of the edge.
    """
    raw_metrics = document.get("final_metrics")
    if not isinstance(raw_metrics, dict):
        if raw_metrics is not None:
            losses.add(
                "final_metrics",
                LossClass.DROPPED,
                f"final_metrics is {type(raw_metrics).__name__}, not an object",
                space=PathSpace.SOURCE,
            )
        return None

    if "total_steps" in raw_metrics:
        losses.add(
            "final_metrics.total_steps",
            LossClass.DROPPED,
            "a count of the document's own steps; the IR carries no step count "
            "and deriving one would restate the event list",
            space=PathSpace.SOURCE,
        )

    values: dict[str, Any] = {}
    for atif_field, ir_field in _USAGE_FIELDS:
        if atif_field not in raw_metrics:
            continue
        raw_value = raw_metrics[atif_field]
        value = (
            _float_or_none(raw_value)
            if ir_field == "cost_usd"
            else _int_or_none(raw_value)
        )
        if value is None:
            losses.add(
                f"usage.{ir_field}",
                LossClass.DROPPED,
                f"final_metrics.{atif_field} is {type(raw_value).__name__}, which "
                "is not a number the IR can carry for this field",
                space=PathSpace.SOURCE,
            )
            continue
        values[ir_field] = value

    unmapped = (
        set(raw_metrics) - {field for field, _ in _USAGE_FIELDS} - {"total_steps"}
    )
    if unmapped:
        losses.add(
            "final_metrics",
            LossClass.DROPPED,
            "final_metrics keys with no IR field: " + ", ".join(sorted(unmapped)),
            space=PathSpace.SOURCE,
        )

    if not values:
        return None
    return TraceUsage(**values)


def _declare_systemic_losses(
    losses: LossReport,
    *,
    had_tool_call: bool,
    has_usage: TraceUsage | None,
) -> None:
    """What no ATIF document carries, declared once each.

    All ``UNSUPPORTED``: the values are absent from the source, not discarded by
    this conversion. That is the same distinction `ir_from_acp` draws, and it is
    what makes the two inbound reports comparable — a reader can ask which of
    two source formats carries more, and the answer is in the class rather than
    in the prose.

    Declared unconditionally, because they hold for every ATIF document rather
    than for the one in hand. The outbound rule is the opposite (declare only
    what *this* trace actually loses) and the asymmetry is deliberate: an
    inbound report describes a format's ceiling, an outbound one describes a
    conversion's cost.
    """
    losses.add(
        "trace_id",
        LossClass.UNSUPPORTED,
        "ATIF documents carry no trace id",
    )
    for field in ("started_at", "finished_at"):
        losses.add(
            field,
            LossClass.UNSUPPORTED,
            "ATIF has no run-level timestamps; wall clock lives in timing.json, "
            "which this converter does not read",
        )
    losses.add(
        "agent.provider",
        LossClass.UNSUPPORTED,
        "ATIF's agent block is name/version/model_name only",
    )
    losses.add(
        "outcome",
        LossClass.UNSUPPORTED,
        "ATIF has no run-outcome section; status, stop_reason, reward and error "
        "category live in result.json, which this converter does not read",
        "§5 loss #9",
    )
    losses.add(
        "events[].started_at",
        LossClass.UNSUPPORTED,
        "ATIF steps carry no timestamps",
    )
    losses.add(
        "events[].finished_at",
        LossClass.UNSUPPORTED,
        "ATIF steps carry no timestamps",
    )
    losses.add(
        "events[].usage",
        LossClass.UNSUPPORTED,
        "ATIF per-step metrics are not written by any producer in this "
        "repository, and an unrecognized metrics object is carried verbatim "
        "rather than interpreted",
    )
    losses.add(
        "events[].outcome",
        LossClass.UNSUPPORTED,
        "ATIF steps carry no per-step outcome; a timeout event does not survive "
        "the outbound edge at all",
        "§5 loss #4",
    )
    if had_tool_call:
        for field in ("started_at", "finished_at"):
            losses.add(
                f"events[].tool_call.{field}",
                LossClass.UNSUPPORTED,
                "ATIF tool calls carry no timestamps",
                "§5 loss #3",
            )
        losses.add(
            "events[].tool_call.content[].raw",
            LossClass.UNSUPPORTED,
            "ATIF stores rendered observation text; the source content block is "
            "not in the document to reconstruct",
            "§5 loss #5",
        )
    if has_usage is None:
        losses.add(
            "usage",
            LossClass.UNSUPPORTED,
            "the document carries no final_metrics this converter can read",
        )
    else:
        for field in _USAGE_UNSUPPORTED:
            if getattr(has_usage, field) is None:
                losses.add(
                    f"usage.{field}",
                    LossClass.UNSUPPORTED,
                    "ATIF final_metrics has no slot for it",
                )
