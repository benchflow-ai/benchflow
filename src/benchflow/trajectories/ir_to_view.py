"""Canonical Trace IR → BenchFlow viewer trace steps — outbound, provisional.

> **PROVISIONAL.** Part of the unwired IR family (`docs/trace-interop.md` §8.7).
> Nothing imports it from a run path, no artifact changes, and no page is
> rendered from it.

This edge produces **only the step list** — the part of a viewer page that is
actually a function of the trace. It deliberately does not build a
``ViewerPayload``: that document's other four fields (``rollout_name``,
``meta``, ``verifier``, and the schema version) come from ``result.json``,
``timing.json``, the ``verifier/`` sidecars and the rollout directory's name.
The IR carries none of them and has no slot for ``task_name``, ``skill_mode``,
``reward``, ``partial_trajectory`` or ``trajectory_source`` at all, so
assembling a payload here would mean **synthesizing run metadata to fill a
shape** rather than converting a trace. Assembly belongs to the wiring slice,
which has the directory.

## What the shape is, and how provisional that is

The wire shape is taken from the viewer package proposed in
`benchflow-ai/benchflow#1034`, read at :data:`VIEW_SCHEMA_ORIGIN`. **No code is
imported from it and none is vendored** — that branch is unmerged, and the
family rule is the one `ir_to_atif` already follows for ATIF: read the target
format as data, pin what matters by test, never reach into the module that
handles it. The constants below are our copy of that vocabulary, and
`tests/trajectories/test_ir_to_view.py` freezes them. That protects **our**
contract from drifting; it cannot notice #1034 changing.

## The two additions, and why they are here

Both are keys #1034's shape does not define, and both are additive: a renderer
that does not know them ignores them and loses nothing it had before.

``steps[].reasoning`` carries reasoning observed on an event that is **not**
itself a reasoning event — the shape ATIF produces, since it folds a thought
into the agent step it precedes. Without the key that value reached no slot and
no record, which is the one thing this family is not allowed to do. A second
`thought` step was refused (it would invent an event boundary the source never
declared) and so was `steps[].text` (it would keep the string and lose that it
is reasoning). See :func:`_carry_reasoning`.

``tool.name_semantics`` is **ours** — #1034's ``ToolCall`` has six fields and
this is a seventh. It carries :attr:`ToolCall.name_semantics` through
unchanged, because without it the viewer boundary loses the only thing that
distinguishes an ACP *category* (``execute``, ``read``) from a *function name*
(``read_file``) from a *span name*. #1034 resolves that by
``tool_hue(kind, title)``, which infers a category from substrings of the two
strings — so an OTel ``gen_ai.tool.name`` of ``read_file`` acquires the ``read``
category because the word "read" appears in it. This edge does not do that, and
does not need to: the hue it emits is neutral unless a real category was
observed. Emitting one extra key is additive — the renderer reads named fields
— but it is a divergence from #1034's contract, not an agreed extension to it.

## Loss regime: declare, don't refuse

Unlike `ir_to_acp`, this edge never raises. A viewer is a display: an event it
cannot type must still reach the page. Every event produces exactly one step,
`UNKNOWN` and `ORACLE` included, and what the shape cannot hold is written to
the :class:`LossReport` instead of being dropped. Sentinels are declared
**per slot**, and only where the target has no null to write: an absent title
and an observed empty title both render as ``""`` and must not read as the same
observation, so only the absent one produces a record.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from benchflow.trajectories.ir import (
    CanonicalTrace,
    EventKind,
    LossClass,
    LossReport,
    PathSpace,
    ToolCall,
    TraceEvent,
)

VIEW_SCHEMA_ORIGIN = "benchflow-ai/benchflow#1034@79695125"
"""The viewer design this shape was read from, at the exact commit audited.

Recorded so the provenance of every constant below is checkable by a person.
It is not a dependency: nothing here fetches, imports or vendors that branch.
"""

LOSS_DIRECTION = "ir->view"

VIEW_STEP_KINDS: tuple[str, ...] = (
    "prompt",
    "message",
    "thought",
    "tool",
    "timeout",
    "unknown",
)
"""The renderer's step vocabulary. Note there is no ``oracle`` member."""

VIEW_TOOL_HUES: tuple[str, ...] = (
    "read",
    "edit",
    "execute",
    "fetch",
    "search",
    "think",
    "skill",
    "other",
)
"""Display hues. The renderer whitelists exactly these and maps each to a CSS
custom-property set; ``other`` resolves to the neutral secondary/border tokens,
which is what makes it usable as "no category was observed" rather than as a
claim about the tool."""

NEUTRAL_HUE = "other"
"""The member of :data:`VIEW_TOOL_HUES` that asserts nothing."""

ACP_KIND_SEMANTICS = "acp_kind"
"""The one :attr:`ToolCall.name_semantics` value that *is* a category.

Defined here rather than imported from `ir_to_acp` so this edge does not depend
on the ACP edge to know what an ACP kind is; a test pins the two equal."""

STEP_KIND: dict[EventKind, str] = {
    EventKind.USER_MESSAGE: "prompt",
    EventKind.AGENT_MESSAGE: "message",
    EventKind.AGENT_REASONING: "thought",
    EventKind.TOOL_CALL: "tool",
    EventKind.TIMEOUT: "timeout",
    EventKind.ORACLE: "unknown",
    EventKind.UNKNOWN: "unknown",
}
"""Every :class:`EventKind`, mapped. Total by test, so a new kind cannot reach
the viewer by vanishing from it."""

TRACE_LEVEL_PATHS: tuple[str, ...] = (
    "trace_id",
    "session_id",
    "agent",
    "usage",
    "outcome",
    "started_at",
    "finished_at",
)
"""Trace-level fields this edge deliberately does not read.

They are not losses of the viewer shape — `meta` holds most of them — and they
are not losses of this edge either, because a step list is not where they go.
Declaring them ``UNSUPPORTED`` says exactly that, at paths a reader can resolve
in the trace, instead of leaving the omission unexplained."""

DIAGNOSTIC_KINDS = frozenset({EventKind.ORACLE, EventKind.UNKNOWN})
"""Kinds with no typed slot, rendered as a serialization of the canonical
event. `ORACLE` is here because #1034's ``StepKind`` has no member for it."""

_TIMEOUT_SEC_KEY = "timeout_sec"
_TIMEOUT_PENDING_KEY = "pending_tool_call_ids"
_TIMEOUT_COMPLETE_KEY = "terminal_trajectory_complete"


def _epoch(value: datetime) -> float:
    return value.timestamp()


def _diagnostic_text(event: TraceEvent) -> str:
    """The canonical event, serialized — **not** a source record.

    The distinction is the point. #1034's own unknown branch serializes the raw
    ACP dict it read off disk; this edge has no such record, only the IR event
    the inbound converter built from one. Presenting that as a raw payload
    would assert a source document the IR does not possess, so the loss record
    names it for what it is and the wiring slice is expected to label it in the
    page the same way.
    """
    return json.dumps(
        event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )


def _hue(call: ToolCall, where: str, losses: LossReport) -> str:
    """The display hue, by direct membership or not at all.

    Two ways to reach a real hue, and both require the source to have said so:
    the semantics must be :data:`ACP_KIND_SEMANTICS` — an ACP ``kind`` *is* a
    category — and the value must already **be** a member of the display
    vocabulary. There is no third path. No substring is examined, no title is
    consulted, and :attr:`Provenance.source_format` is never read: where a
    trace came from is not evidence about what a field means.
    """
    if call.name_semantics == ACP_KIND_SEMANTICS and call.name in VIEW_TOOL_HUES:
        return call.name

    if call.name_semantics is None:
        reason = "the IR carries no semantics for this tool name"
    elif call.name_semantics != ACP_KIND_SEMANTICS:
        reason = (
            f"{call.name_semantics!r} is a name, not a category — inferring one "
            "from the string would be the reinterpretation this family refuses"
        )
    else:
        reason = (
            f"the observed category {call.name!r} is outside the viewer's "
            "display vocabulary"
        )
    losses.add(
        f"{where}.tool_call.name_semantics",
        LossClass.SYNTHESIZED,
        f"the viewer step requires a hue and {reason}; the neutral "
        f"{NEUTRAL_HUE!r} was written, which asserts no category",
        space=PathSpace.HUB,
    )
    return NEUTRAL_HUE


def _content_texts(call: ToolCall, where: str, losses: LossReport) -> list[str]:
    """The text actually observed, and nothing standing in for what was not.

    A block the IR holds as ``OPAQUE`` — or a ``TEXT`` block whose text is
    ``None`` — has no string to contribute. Rendering its ``raw`` as JSON would
    manufacture a tool observation that the capture never made, so the block is
    declared and omitted. An observed empty string is kept: it is a value.
    """
    texts: list[str] = []
    for position, block in enumerate(call.content):
        if block.text is None:
            losses.add(
                f"{where}.tool_call.content[{position}]",
                LossClass.DROPPED,
                f"a {block.kind.value!r} block with no text; the viewer holds "
                "tool output as strings and this edge does not invent one from "
                "the block's raw form",
                space=PathSpace.HUB,
            )
            continue
        texts.append(block.text)
    return texts


def _tool_payload(call: ToolCall, where: str, losses: LossReport) -> dict[str, Any]:
    """The tool object, with every non-nullable slot accounted for."""
    if call.call_id is None:
        losses.add(
            f"{where}.tool_call.call_id",
            LossClass.SYNTHESIZED,
            "the viewer requires an id string and the source carried no id "
            'field at all, so "" was written; it is not an observed empty id',
            space=PathSpace.HUB,
        )
    if call.name is None:
        losses.add(
            f"{where}.tool_call.name",
            LossClass.SYNTHESIZED,
            'the viewer requires a name string and the IR carries none, so "" '
            "was written; the renderer's own fallback label is its business, "
            "not an observation this edge should make",
            space=PathSpace.HUB,
        )
    if call.title is None:
        losses.add(
            f"{where}.tool_call.title",
            LossClass.SYNTHESIZED,
            "the viewer requires a title string and the source carried none, "
            'so "" was written; an observed empty title declares nothing here',
            space=PathSpace.HUB,
        )
    if call.status is None:
        losses.add(
            f"{where}.tool_call.status",
            LossClass.SYNTHESIZED,
            "the viewer requires a status string and the IR carries none, so "
            '"" was written; the renderer shows "?" for it, and that is a '
            "display choice rather than a status the run had",
            space=PathSpace.HUB,
        )
    if call.arguments is not None:
        losses.add(
            f"{where}.tool_call.arguments",
            LossClass.DROPPED,
            "the viewer's tool object has no slot for arguments; the values "
            "are observed and reach no field",
            space=PathSpace.HUB,
        )

    return {
        "id": call.call_id or "",
        "kind": call.name or "",
        "title": call.title or "",
        "status": call.status.value if call.status is not None else "",
        "content": _content_texts(call, where, losses),
        "hue": _hue(call, where, losses),
        "name_semantics": call.name_semantics,
    }


def _timeout_payload(
    event: TraceEvent, where: str, losses: LossReport
) -> dict[str, Any]:
    """The typed timeout object — this kind does not fall through to unknown.

    Two of the four slots accept ``None`` and so represent their own absence;
    the other two do not, and each sentinel is declared on its own.
    """
    extensions = event.extensions
    if event.outcome is None:
        losses.add(
            f"{where}.outcome",
            LossClass.SYNTHESIZED,
            "the viewer requires a timeout reason string and the event carries "
            'no terminal signal, so "" was written',
            space=PathSpace.HUB,
        )
    pending_raw = extensions.get(_TIMEOUT_PENDING_KEY)
    if pending_raw is None:
        losses.add(
            f"{where}.extensions",
            LossClass.SYNTHESIZED,
            f"the viewer requires a list of pending tool-call ids and the event "
            f"carries no {_TIMEOUT_PENDING_KEY!r}, so [] was written; it is not "
            "an observation that none were pending",
            space=PathSpace.HUB,
        )
        pending: list[str] = []
    else:
        pending = [str(item) for item in pending_raw]

    return {
        "reason": event.outcome or "",
        "timeout_sec": extensions.get(_TIMEOUT_SEC_KEY),
        "pending": pending,
        "complete": extensions.get(_TIMEOUT_COMPLETE_KEY),
    }


def _timestamps(event: TraceEvent, step: dict[str, Any], is_tool: bool) -> bool:
    """Attach ``t``/``dur`` when observed. Returns whether anything was written.

    A tool step prefers the call's own window when it has one — that is the
    narrower observation — and falls back to the event's. ``dur`` needs both
    ends and a non-negative interval; a finish before a start is not a duration
    and is left out rather than clamped.
    """
    started, finished = event.started_at, event.finished_at
    if is_tool and event.tool_call is not None and event.tool_call.started_at:
        started = event.tool_call.started_at
        finished = event.tool_call.finished_at or finished
    if started is None:
        return False
    step["t"] = _epoch(started)
    if finished is not None and finished >= started:
        step["dur"] = _epoch(finished) - _epoch(started)
    return True


def _declare_codomain(losses: LossReport) -> None:
    """What this edge is not for.

    These are not step-level losses and must not be counted as any: the trace
    carries them, and the document that would hold them is assembled elsewhere
    from artifacts this edge never sees. Calling them ``DROPPED`` would say the
    viewer cannot represent them, which is false — it says nothing about them
    here because here is the wrong place.
    """
    for path in TRACE_LEVEL_PATHS:
        losses.add(
            path,
            LossClass.UNSUPPORTED,
            "run metadata, not a step. The viewer holds this in `meta`, which "
            "the wiring slice builds from result.json and timing.json — it is "
            "outside this edge's codomain rather than something the edge lost",
            space=PathSpace.HUB,
        )
    losses.add(
        "steps[].label",
        LossClass.UNSUPPORTED,
        "prompt labels number the prompts a run was given, which live in "
        "prompts.json; a trace does not know its own prompt ordinals, so this "
        "edge never writes the key",
        space=PathSpace.TARGET,
    )


def _declare_systemic(losses: LossReport, trace: CanonicalTrace) -> None:
    """Properties of the mapping itself, declared once rather than per event."""
    losses.add(
        "steps[].i",
        LossClass.SYNTHESIZED,
        f"viewer step positions, dense from 1 over the {len(trace.events)} "
        "steps emitted. The IR's event index is a position in a different "
        "sequence and this is not that index renumbered",
        space=PathSpace.TARGET,
    )
    losses.add(
        "events[].kind",
        LossClass.NORMALIZED,
        "the IR's event vocabulary is projected onto the viewer's six step "
        "kinds; oracle has no member and shares 'unknown' with unrecognized "
        "records, which is why the source type is carried alongside",
        space=PathSpace.HUB,
    )

    if any(event.role is not None for event in trace.events):
        losses.add(
            "events[].role",
            LossClass.DROPPED,
            "the viewer step has no slot for who an event is attributable to; "
            "the renderer labels a step from its kind alone",
            space=PathSpace.HUB,
        )
    if any(
        event.tool_call is not None and event.tool_call.content
        for event in trace.events
    ):
        losses.add(
            "events[].tool_call.content[].kind",
            LossClass.DROPPED,
            "tool output is a list of strings in the viewer; whether a block "
            "was text or opaque does not survive",
            space=PathSpace.HUB,
        )
        losses.add(
            "events[].tool_call.content[].raw",
            LossClass.DROPPED,
            "the source block is kept verbatim in the IR and has no viewer slot",
            space=PathSpace.HUB,
        )
        losses.add(
            "steps[].tool.content",
            LossClass.NORMALIZED,
            "content blocks become plain strings, in order, keeping only text "
            "that was observed",
            space=PathSpace.TARGET,
        )
    if any(event.usage is not None for event in trace.events):
        losses.add(
            "events[].usage",
            LossClass.DROPPED,
            "the viewer aggregates usage at the run level in `meta`; a step has "
            "no usage slot, so a per-event observation reaches no field",
            space=PathSpace.HUB,
        )
    if any(
        event.reasoning is not None
        and event.kind is not EventKind.AGENT_REASONING
        and event.kind not in DIAGNOSTIC_KINDS
        for event in trace.events
    ):
        losses.add(
            "steps[].reasoning",
            LossClass.NORMALIZED,
            "reasoning observed alongside a non-reasoning event keeps its own "
            "key on that step rather than becoming a separate thought step: "
            "the source declared no such event, and the value stays labelled "
            "as reasoning instead of being merged into the step's text",
            space=PathSpace.TARGET,
        )
    if any(event.reasoning_segments for event in trace.events):
        losses.add(
            "events[].reasoning_segments",
            LossClass.DROPPED,
            "thought boundaries are not expanded into separate steps by this "
            "slice: there is no contract yet for doing so without also emitting "
            "the joined `reasoning`, which would show the same text twice",
            space=PathSpace.HUB,
        )
    if any(
        event.tool_call is not None and event.tool_call.status for event in trace.events
    ):
        losses.add(
            "steps[].tool.status",
            LossClass.NORMALIZED,
            "the IR's status enum becomes the viewer's status string",
            space=PathSpace.TARGET,
        )
    if any(
        event.tool_call is not None
        and event.tool_call.name_semantics == ACP_KIND_SEMANTICS
        and event.tool_call.name in VIEW_TOOL_HUES
        for event in trace.events
    ):
        losses.add(
            "steps[].tool.hue",
            LossClass.NORMALIZED,
            "an observed ACP category that is already a member of the display "
            "vocabulary is carried across directly; membership is tested, never "
            "inferred from the string",
            space=PathSpace.TARGET,
        )
    if any(event.kind in DIAGNOSTIC_KINDS for event in trace.events):
        losses.add(
            "steps[].text",
            LossClass.SYNTHESIZED,
            "events with no typed viewer slot are rendered as a serialization "
            "of the **canonical IR event**. It is not a raw source record — the "
            "IR does not hold one — and a page showing it must say so",
            space=PathSpace.TARGET,
        )


def _carry_reasoning(
    event: TraceEvent, step: dict[str, Any], where: str, losses: LossReport
) -> None:
    """Reasoning observed on an event that is not itself a reasoning event.

    ATIF folds a thought into the agent step it precedes
    (``reasoning_content`` beside ``tool_calls``), so a faithful reading of one
    produces a `TOOL_CALL` event that carries reasoning. Three ways to place
    that value were available and two are refused:

    - a second `thought` step would invent an event boundary and an ordering
      the source never declared;
    - `steps[].text` would keep the string and lose the one thing that makes
      it different from a message — that it is reasoning. Concatenating it
      into a neighbouring slot does the same, worse.

    So it gets its own key. Like `tool.name_semantics` this is **additive to**
    :data:`VIEW_SCHEMA_ORIGIN`'s shape, not part of it: a renderer that does
    not know the key ignores it and loses nothing it had before.

    Diagnostic kinds are excluded: their whole event is already serialized into
    the step, reasoning included, and a second copy would show it twice.
    """
    if event.reasoning is None:
        return
    if event.kind is EventKind.AGENT_REASONING or event.kind in DIAGNOSTIC_KINDS:
        return
    step["reasoning"] = event.reasoning


def _declare_unconsumed_text(
    event: TraceEvent, step: dict[str, Any], where: str, losses: LossReport
) -> None:
    """Per-event records for observed text this shape has no slot for.

    Two cases, both real rather than defensive: a reasoning event that also
    carries user-visible text (the step's one text slot is already holding the
    reasoning), and a terminal signal on an event that is not the timeout —
    only the timeout step has somewhere to put one.
    """
    if (
        event.kind is EventKind.AGENT_REASONING
        and event.text is not None
        and event.text != step.get("text")
    ):
        losses.add(
            f"{where}.text",
            LossClass.DROPPED,
            "a reasoning event carrying user-visible text as well; the step "
            "has one text slot and it is holding the reasoning, and joining "
            "the two strings would present them as one utterance",
            space=PathSpace.HUB,
        )

    if (
        event.outcome is not None
        and event.kind is not EventKind.TIMEOUT
        and event.kind not in DIAGNOSTIC_KINDS
    ):
        losses.add(
            f"{where}.outcome",
            LossClass.DROPPED,
            "a terminal signal on an event that is not the timeout; only the "
            "timeout step has a slot for one, so the value reaches no field",
            space=PathSpace.HUB,
        )


def ir_to_view_steps(trace: CanonicalTrace) -> tuple[list[dict[str, Any]], LossReport]:
    """Project a canonical trace onto the viewer's step list.

    Returns the steps and a report of everything the shape could not carry.
    **Every event becomes exactly one step** — there is no branch that skips a
    record, and a test pins the counts equal — so an event this edge cannot
    type is visible on the page as a diagnostic rather than absent from it.

    The input is not modified and its own inbound report is not touched: a
    trace may be converted to many targets and none of them describes it.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    _declare_codomain(losses)
    _declare_systemic(losses, trace)

    steps: list[dict[str, Any]] = []
    for event in trace.events:
        where = f"events[{event.index}]"
        kind = STEP_KIND[event.kind]
        step: dict[str, Any] = {"i": len(steps) + 1, "kind": kind}

        if event.source_type is not None:
            step["type"] = event.source_type
        elif event.kind is EventKind.ORACLE:
            # The IR observed oracle-ness in `kind`; without a source string the
            # step would land in the same undifferentiated 'unknown' as a record
            # nobody recognized. Writing the kind here reshapes an observation,
            # it does not invent one.
            step["type"] = EventKind.ORACLE.value
            losses.add(
                "steps[].type",
                LossClass.NORMALIZED,
                "the event kind was written into the type slot for an oracle "
                "record that carried no source type string of its own",
                space=PathSpace.TARGET,
            )

        if event.kind is EventKind.AGENT_REASONING:
            if event.reasoning is not None:
                step["text"] = event.reasoning
        elif event.kind in DIAGNOSTIC_KINDS:
            step["text"] = _diagnostic_text(event)
        elif event.text is not None:
            step["text"] = event.text

        _carry_reasoning(event, step, where, losses)
        _declare_unconsumed_text(event, step, where, losses)

        if event.kind is EventKind.TOOL_CALL:
            if event.tool_call is None:
                losses.add(
                    f"{where}.tool_call",
                    LossClass.DROPPED,
                    "a tool-call event with no tool call; the step keeps its "
                    "kind and carries no tool object",
                    space=PathSpace.HUB,
                )
            else:
                step["tool"] = _tool_payload(event.tool_call, where, losses)
        elif event.kind is EventKind.TIMEOUT:
            step["timeout"] = _timeout_payload(event, where, losses)

        _timestamps(event, step, is_tool=event.kind is EventKind.TOOL_CALL)
        steps.append(step)

    if any("t" in step for step in steps):
        losses.add(
            "steps[].t",
            LossClass.NORMALIZED,
            "observed timestamps become epoch seconds; a tool step prefers the "
            "call's own window over the event's when it has one",
            space=PathSpace.TARGET,
        )

    return steps, losses
