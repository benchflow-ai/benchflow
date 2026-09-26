"""Canonical Trace IR → ATIF (Slice D) — the first outbound edge.

> **PROVISIONAL.** Companion to :mod:`benchflow.trajectories.ir`, itself an
> unapproved proposal (`docs/trace-interop.md` §8). Nothing imports this module,
> nothing writes its output to disk, and `export_atif.py` is untouched and still
> the only writer of `trainer/atif.json`.

Where :mod:`benchflow.trajectories.ir_from_acp` stress-tested the
declared-absence rule, this edge stress-tests the other half of the taxonomy:
**ATIF requires values the IR does not have**, so this is the first converter
that must fabricate, and every fabrication is recorded as
:attr:`~benchflow.trajectories.ir.LossClass.SYNTHESIZED`.

## The claim this module is built to support

`ir_to_atif(acp_events_to_ir(events), prompts=P)` produces **the same document**
as `trajectory_to_atif_record(events=events, prompts=P)` — the existing direct
exporter — for any conformant capture input, with one deliberate exception
(oracle, below). If the hub lost something the direct path preserved, that
equality would fail, so the parity test is the strongest available evidence that
the IR is sufficient for this format.

Parity is against the *document*, not the report: the direct exporter produces
no report, which is the difference this whole proposal is about.

## Deliberate deviation from the direct exporter

**`oracle` steps.** `acp_events_to_atif_steps` renders an oracle record as a
`source: "agent"` step whose message is prefixed `[oracle: …]`, so a consumer
can only recover the distinction by string matching — §5.1 records this as a
live divergence, since the in-repo validator already accepts
`source: "oracle"` while no emitter produces it. The IR carries
:attr:`~benchflow.trajectories.ir.Role.ORACLE`, so this converter emits
`source: "oracle"` and puts the command in `message` without the prefix. That is
the first thing the hub buys that the direct path could not, and it is
enumerated by a test rather than left as a silent difference.

## What ATIF forces

Seven values ATIF requires and the IR does not carry, each recorded where it is
invented:

- `agent.version`, and `agent.name` when the trace has none — hub space, at the
  IR field whose absence forced it;
- a tool-call id when the IR's is empty, and a `function_name` when its name is;
- `arguments`, always — see below;
- `steps[].message` on steps that carry only a tool call or a flushed thought,
  and `final_metrics.total_steps` — target space, since no IR field corresponds;
- the leading `user` steps built from *prompts*, which are not trace data at all
  but an argument of this conversion — target space.

## `arguments`

The document still receives `{}`. It is what ATIF requires and what the direct
exporter writes, and departing from it would trade a real compatibility property
for a cosmetic one. What changes is that it is no longer **silent**: every
fabricated `{}` is declared `SYNTHESIZED` at the hub path
``events[i].tool_call.arguments`` — the same path the ACP edge declared
``UNSUPPORTED``. Read together, the two reports say the source never had
arguments and the target demanded them anyway.

A tool call whose IR `arguments` is a real mapping — including a genuinely
captured `{}` — passes through and is declared nothing. The two cases are
indistinguishable in the ATIF document, which is a limit of ATIF; they are
distinguishable in the report, which is the point.
"""

from __future__ import annotations

from typing import Any

from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossReport,
    PathSpace,
    Role,
    TraceEvent,
)

LOSS_DIRECTION = "ir->atif"

# The ATIF `source` is derived from the event kind, not from the IR role, and
# for anything an ACP capture produces the two agree. They can disagree in a
# trace from another source, and then the role is information this edge cannot
# carry — declared per event, because it is a fact about that one event.
_IMPLIED_ROLE = {
    EventKind.USER_MESSAGE: Role.USER,
    EventKind.AGENT_MESSAGE: Role.AGENT,
    EventKind.AGENT_REASONING: Role.AGENT,
    EventKind.TOOL_CALL: Role.AGENT,
    EventKind.ORACLE: Role.ORACLE,
}

ATIF_SCHEMA_VERSION = "ATIF-v1.7"
"""Deliberately not imported from ``export_atif``.

The hub must not depend on the exporters it sits between; a test pins this
constant equal to that module's, so drift fails instead of going unnoticed.
"""

# Steps carrying only a tool call or a flushed thought get an empty message,
# because ATIF steps always have one. Declared once, not per step.
_EMPTY_MESSAGE_FIELD = "steps[].message"


def _reasoning_texts(event: TraceEvent) -> list[str]:
    """The thought texts this event contributes, boundaries first.

    ``reasoning_segments`` is the richer encoding and is preferred; ``reasoning``
    is the fallback for a trace built without segments. Empty strings are
    dropped, matching the direct exporter's ``if text:`` guard.
    """
    if event.reasoning_segments is not None:
        return [text for text in event.reasoning_segments if text]
    return [event.reasoning] if event.reasoning else []


def _observation_text(event: TraceEvent) -> str:
    """Text output of a tool call, joined as ``content_blocks_to_text`` joins it.

    Opaque blocks contribute nothing — ATIF has no slot for them — which is why
    the caller declares them dropped.
    """
    if event.tool_call is None:
        return ""
    return "\n".join(
        block.text or ""
        for block in event.tool_call.content
        if block.kind is ContentBlockKind.TEXT
    )


def ir_to_atif(
    trace: CanonicalTrace,
    *,
    prompts: list[str] | None = None,
) -> tuple[dict[str, Any], LossReport]:
    """Build one ATIF document from a canonical trace.

    *prompts* are the user-facing prompts handed to the agent before any event
    was captured. They are **not** trace data — `acp_events_to_ir` deliberately
    does not invent events for them (§5.2: doing so is what makes an ATIF
    document open with two identical `user` steps) — so they enter here, at the
    edge whose format wants them, and every step they produce is declared
    ``SYNTHESIZED`` in the target space.

    Returns the document **and** its report. The report is returned rather than
    attached because a trace may be converted to many targets and none of those
    conversions describes how the trace was built; ``trace`` is not modified.

    Raises ``ValueError`` for a trace with no representable event, mirroring the
    direct exporter: ATIF requires at least one step and fabricating an empty one
    would be inventing.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    steps: list[dict[str, Any]] = []
    pending_thoughts: list[str] = []
    empty_message_declared = False

    def append_step(source: str, body: dict[str, Any]) -> None:
        steps.append({"step_id": len(steps) + 1, "source": source, **body})

    def take_reasoning() -> str | None:
        """Join buffered thoughts with a blank line, then clear.

        The join is ``ThoughtBuffer``'s, reimplemented rather than imported so
        the hub does not depend on export plumbing; a test pins the two equal.
        """
        if not pending_thoughts:
            return None
        joined = "\n\n".join(pending_thoughts)
        pending_thoughts.clear()
        return joined

    def declare_empty_message() -> None:
        nonlocal empty_message_declared
        if empty_message_declared:
            return
        empty_message_declared = True
        losses.add(
            _EMPTY_MESSAGE_FIELD,
            LossClass.SYNTHESIZED,
            "ATIF steps always carry a message; a step holding only a tool call "
            "or a flushed thought has no IR text to put there",
            space=PathSpace.TARGET,
        )

    def flush_thoughts() -> None:
        reasoning = take_reasoning()
        if reasoning:
            declare_empty_message()
            append_step("agent", {"message": "", "reasoning_content": reasoning})

    for prompt in prompts or []:
        if not prompt:
            continue
        append_step("user", {"message": str(prompt)})
        losses.add(
            f"steps[{len(steps) - 1}]",
            LossClass.SYNTHESIZED,
            "step built from the prompts argument, not from any IR event; the "
            "trace carries no antecedent for it",
            space=PathSpace.TARGET,
        )

    for event in trace.events:
        where = f"events[{event.index}]"

        implied_role = _IMPLIED_ROLE.get(event.kind)
        if (
            event.role is not None
            and implied_role is not None
            and event.role is not implied_role
        ):
            losses.add(
                f"{where}.role",
                LossClass.DROPPED,
                f"the IR attributes this event to {event.role.value!r}, but ATIF "
                f"derives step source from the event kind and will write "
                f"{implied_role.value!r}; the disagreement has no ATIF slot",
            )

        if event.kind is EventKind.USER_MESSAGE:
            if event.text:
                flush_thoughts()
                append_step("user", {"message": event.text})
            else:
                losses.add(
                    where,
                    LossClass.DROPPED,
                    "text-empty event; ATIF has no step for it and the direct "
                    "exporter drops it too",
                )

        elif event.kind is EventKind.AGENT_REASONING:
            texts = _reasoning_texts(event)
            if texts:
                pending_thoughts.extend(texts)
            else:
                losses.add(where, LossClass.DROPPED, "reasoning event with no text")

        elif event.kind is EventKind.AGENT_MESSAGE:
            if event.text:
                body: dict[str, Any] = {"message": event.text}
                reasoning = take_reasoning()
                if reasoning:
                    body["reasoning_content"] = reasoning
                append_step("agent", body)
            else:
                losses.add(
                    where,
                    LossClass.DROPPED,
                    "text-empty event; ATIF has no step for it and the direct "
                    "exporter drops it too",
                )

        elif event.kind is EventKind.TOOL_CALL and event.tool_call is not None:
            call = event.tool_call
            # Computed before the step is appended, so the fallback id matches
            # the step_id the step is about to receive — the direct exporter's
            # convention, which ATIF resolves within one step anyway.
            call_id = call.call_id or f"call_{len(steps) + 1}"
            if not call.call_id:
                losses.add(
                    f"{where}.tool_call.call_id",
                    LossClass.SYNTHESIZED,
                    f"ATIF needs an id to bind the observation to; the IR carries "
                    f"{call.call_id!r}, so {call_id!r} was generated",
                )
            function_name = call.name or "tool"
            if not call.name:
                losses.add(
                    f"{where}.tool_call.name",
                    LossClass.SYNTHESIZED,
                    "ATIF requires function_name; the IR carries no tool name, "
                    'so "tool" was generated',
                )

            tool_call: dict[str, Any] = {
                "tool_call_id": call_id,
                "function_name": function_name,
                "arguments": {},
            }
            if call.arguments is None:
                losses.add(
                    f"{where}.tool_call.arguments",
                    LossClass.SYNTHESIZED,
                    "the IR carries no arguments for this call and ATIF requires "
                    "the field, so an empty mapping was written; it is not an "
                    "observation that the tool was called with none",
                    "§5 loss 1",
                )
            else:
                tool_call["arguments"] = call.arguments

            extra: dict[str, str] = {}
            if call.title:
                extra["title"] = str(call.title)
            if call.status:
                extra["status"] = call.status.value
            if extra:
                tool_call["extra"] = extra

            declare_empty_message()
            tool_body: dict[str, Any] = {"message": "", "tool_calls": [tool_call]}
            reasoning = take_reasoning()
            if reasoning:
                tool_body["reasoning_content"] = reasoning
            result_text = _observation_text(event)
            if result_text:
                tool_body["observation"] = {
                    "results": [{"source_call_id": call_id, "content": result_text}]
                }
            append_step("agent", tool_body)

            if any(block.kind is ContentBlockKind.OPAQUE for block in call.content):
                losses.add(
                    f"{where}.tool_call.content",
                    LossClass.DROPPED,
                    "non-text content blocks have no ATIF representation; the IR "
                    "carries them verbatim and this edge cannot",
                    "§5 loss 5",
                )

        elif event.kind is EventKind.ORACLE:
            command = str(event.extensions.get("command") or "oracle")
            append_step("oracle", {"message": command})
            losses.add(
                f"{where}.extensions",
                LossClass.NORMALIZED,
                "the oracle command becomes the step message; return_code and "
                "stdout have no ATIF slot",
            )

        else:
            losses.add(
                where,
                LossClass.DROPPED,
                f"{event.kind.value} events have no ATIF representation"
                + (
                    "; the timeout marker is absent from every exported document"
                    if event.kind is EventKind.TIMEOUT
                    else ""
                ),
                "§5 loss 4" if event.kind is EventKind.TIMEOUT else None,
            )

    flush_thoughts()

    if not steps:
        raise ValueError("ATIF requires at least one step; trace is empty")

    agent: dict[str, Any] = {
        "name": trace.agent.agent_name or "unknown",
        "version": trace.agent.agent_version or "unknown",
    }
    if not trace.agent.agent_name:
        losses.add(
            "agent.agent_name",
            LossClass.SYNTHESIZED,
            'ATIF requires agent.name; the trace carries none, so "unknown" was '
            "written",
        )
    if not trace.agent.agent_version:
        losses.add(
            "agent.agent_version",
            LossClass.SYNTHESIZED,
            "ATIF requires agent.version; BenchFlow does not track agent binary "
            'versions, so "unknown" was written',
            "§5 loss 7",
        )
    if trace.agent.model:
        agent["model_name"] = trace.agent.model

    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    losses.add(
        "final_metrics.total_steps",
        LossClass.SYNTHESIZED,
        "computed from the produced document; no IR field corresponds to it",
        space=PathSpace.TARGET,
    )
    if trace.usage is not None:
        for source_field, atif_field in (
            ("input_tokens", "total_prompt_tokens"),
            ("output_tokens", "total_completion_tokens"),
            ("cache_read_tokens", "total_cached_tokens"),
            ("cost_usd", "total_cost_usd"),
        ):
            value = getattr(trace.usage, source_field)
            if value is not None:
                final_metrics[atif_field] = value
        for unmapped in (
            "cache_creation_tokens",
            "reasoning_tokens",
            "total_tokens",
            "source",
            "price_source",
        ):
            if getattr(trace.usage, unmapped) is not None:
                losses.add(
                    f"usage.{unmapped}",
                    LossClass.DROPPED,
                    "ATIF final_metrics has no slot for it",
                )

    _declare_systemic_losses(trace, losses)
    _declare_structural_metadata(steps, losses)

    record: dict[str, Any] = {"schema_version": ATIF_SCHEMA_VERSION}
    if trace.session_id:
        record["session_id"] = trace.session_id
    record["agent"] = agent
    record["steps"] = steps
    record["final_metrics"] = final_metrics
    return record, losses


def _declare_structural_metadata(
    steps: list[dict[str, Any]], losses: LossReport
) -> None:
    """Declare the two values this edge writes that describe the *document*.

    Neither is a statement about the run, and both are deterministic — one is a
    constant, the other a position. That is precisely why they were easy to miss:
    a value that is obviously not an observation still *becomes* one once a
    reader takes it back into the hub. `ATIF → IR` puts `schema_version` into
    the trace's ``extensions`` and each `step_id` into its event's, faithfully,
    because they really are in the document it is reading. So the round trip
    ends with two values the input never had, and only a declaration here
    distinguishes them from an observation.

    ``SYNTHESIZED`` rather than ``NORMALIZED``: there is no source value being
    reshaped. Nothing in the IR corresponds to either of them.

    The space is ``TARGET`` and the paths are the ATIF document's, because that
    is where this edge writes them and neither has an IR antecedent. A hub path
    would be a lie twice over — it would address a node the trace being
    converted does not contain, and the resolvability guard would be right to
    reject it.
    """
    losses.add(
        "schema_version",
        LossClass.SYNTHESIZED,
        f"structural metadata of the ATIF document: the literal "
        f"{ATIF_SCHEMA_VERSION!r} identifying the dialect, written on every "
        "record. It is not an observation about the run, and the IR carries no "
        "field for a target format's own version",
        space=PathSpace.TARGET,
    )
    if steps:
        losses.add(
            "steps[].step_id",
            LossClass.SYNTHESIZED,
            f"structural metadata the exporter introduces: {len(steps)} step "
            "positions, dense from 1 over the steps actually emitted. No id was "
            "observed in the source — the IR's event index is a position in a "
            "different sequence, and this is not that index renumbered",
            space=PathSpace.TARGET,
        )


def _declare_systemic_losses(trace: CanonicalTrace, losses: LossReport) -> None:
    """Losses that hold for the conversion rather than for one event.

    Declared **only when the trace actually carries the value**, which is the
    rule this edge is built on: *the outbound report describes what is lost from
    the trace it received, not what an ACP-derived trace typically lacks*. An
    ACP trace has no per-event timestamps or usage, and the inbound report
    already declared those absences ``UNSUPPORTED``; repeating them here as
    ``DROPPED`` would double-count one fact and misdescribe an edge that loses
    nothing it was given. A trace from a richer source carries them, and then
    every one is declared.

    Two IR fields are deliberately never declared, because they describe the
    representation rather than the run: ``ir_version`` and ``losses``. Every
    other field of every IR model is accounted for here or in the event walk,
    and a test derives that list from the models themselves so a new field
    cannot be added without a disposition.
    """
    events = trace.events

    if trace.trace_id:
        losses.add("trace_id", LossClass.DROPPED, "ATIF documents carry no trace id")
    for field in ("started_at", "finished_at"):
        if getattr(trace, field) is not None:
            losses.add(
                field,
                LossClass.DROPPED,
                "ATIF has no run-level timestamps; wall clock lives in "
                "timing.json for the direct path too",
            )
    losses.add(
        "provenance",
        LossClass.DROPPED,
        "ATIF records no provenance for the document it describes",
    )
    if trace.extensions:
        losses.add("extensions", LossClass.DROPPED, "carried by the IR, no ATIF slot")
    if trace.agent.provider:
        losses.add(
            "agent.provider",
            LossClass.DROPPED,
            "ATIF's agent block is name/version/model_name only",
        )
    if trace.outcome and any(
        value is not None
        for value in (
            trace.outcome.status,
            trace.outcome.stop_reason,
            trace.outcome.reward,
            trace.outcome.error_category,
        )
    ):
        losses.add(
            "outcome",
            LossClass.DROPPED,
            "ATIF has no run-outcome section; reward and error category live in "
            "result.json for the direct path too",
        )

    if not events:
        return

    losses.add(
        "events[].index",
        LossClass.NORMALIZED,
        "ATIF step_id is dense from 1 over the steps actually emitted, so an "
        "event that produces no step shifts every later number; event identity "
        "does not survive",
    )
    losses.add(
        "events[].provenance",
        LossClass.DROPPED,
        "ATIF records no per-step provenance",
    )
    if any(event.source_type for event in events):
        losses.add(
            "events[].source_type",
            LossClass.DROPPED,
            "the source's own type string has no ATIF slot; only the normalized "
            "kind survives, through the step shape",
        )
    if any(event.extensions and event.kind is not EventKind.ORACLE for event in events):
        losses.add(
            "events[].extensions",
            LossClass.DROPPED,
            "carried verbatim by the IR, no ATIF slot",
        )
    if any(event.tool_call and event.tool_call.name_semantics for event in events):
        losses.add(
            "events[].tool_call.name_semantics",
            LossClass.DROPPED,
            "ATIF has one function_name slot and no way to say that the value in "
            "it is an ACP kind rather than a function name",
        )
    if any(event.reasoning_segments for event in events):
        losses.add(
            "events[].reasoning_segments",
            LossClass.NORMALIZED,
            "thoughts are joined with a blank line into reasoning_content, so "
            "the boundaries the IR preserved are not recoverable",
            "§5 loss 10",
        )
    if any(event.usage for event in events):
        losses.add(
            "events[].usage",
            LossClass.DROPPED,
            "ATIF per-step metrics are not emitted by this converter",
        )
    if any(event.outcome and event.kind is not EventKind.TIMEOUT for event in events):
        losses.add(
            "events[].outcome",
            LossClass.DROPPED,
            "ATIF steps carry no per-step outcome; a timeout event is dropped "
            "whole and declared at its own index instead",
        )
    for field in ("started_at", "finished_at"):
        if any(getattr(event, field) is not None for event in events):
            losses.add(
                f"events[].{field}",
                LossClass.DROPPED,
                "ATIF steps carry no timestamps",
            )
        # Addressed under the tool call, never under the event: they are
        # different values, and blaming the event would misdescribe which one
        # was dropped.
        if any(
            event.tool_call and getattr(event.tool_call, field) is not None
            for event in events
        ):
            losses.add(
                f"events[].tool_call.{field}",
                LossClass.DROPPED,
                "ATIF tool calls carry no timestamps",
            )
    if any(
        event.tool_call
        and any(
            block.kind is ContentBlockKind.TEXT and block.raw is not None
            for block in event.tool_call.content
        )
        for event in events
    ):
        losses.add(
            "events[].tool_call.content[].raw",
            LossClass.DROPPED,
            "only the rendered text of a block reaches observation; the source "
            "block the IR kept verbatim does not",
        )
