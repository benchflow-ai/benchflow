"""Conversion suite for ``ATIF → canonical Trace IR`` — the inbound ATIF edge.

The two edges before this one had an easy source of truth: `ACP → IR` could be
checked against events a real `ACPSession` emitted, and `IR → ATIF` against the
document the direct exporter writes. This edge reads a document that is *itself*
the output of a lossy conversion, and its whole discipline is one rule:

    read what the document says, never what it probably meant.

Most of what follows tests that rule at the places where breaking it would be
tempting and would make the round trip look better than it is — `arguments: {}`,
`agent.version: "unknown"`, `message: ""`, and the thought boundaries the
outbound join destroyed.

Nothing here writes to disk. `export_atif.py` is imported read-only, as the
producer whose documents this edge has to be able to read.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from benchflow.trajectories import ir_from_atif as ir_from_atif_module
from benchflow.trajectories.export_atif import trajectory_to_atif_record
from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    ModelInfo,
    PathSpace,
    Role,
    ToolCall,
    ToolStatus,
    TraceEvent,
    TraceOutcome,
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_from_atif import (
    ATIF_SOURCE,
    LOSS_DIRECTION,
    atif_to_ir,
)
from benchflow.trajectories.ir_to_atif import ir_to_atif
from tests.trajectories.test_atif_preservation import _rich_events
from tests.trajectories.test_trace_ir import resolve_ir_path

PROMPTS = ["Solve the task.", "Then stop."]


def _rich_document(**kwargs: Any) -> dict[str, Any]:
    """An ATIF document from the direct exporter, over real captured events."""
    return trajectory_to_atif_record(
        session_id="sess-e",
        agent_name="claude-code",
        events=_rich_events(),
        prompts=PROMPTS,
        model="claude-sonnet-5",
        **kwargs,
    )


def _hub_document(**kwargs: Any) -> dict[str, Any]:
    """The same document, produced through the hub instead."""
    trace = acp_events_to_ir(
        _rich_events(),
        session_id="sess-e",
        agent_name="claude-code",
        model="claude-sonnet-5",
    )
    document, _ = ir_to_atif(trace, prompts=PROMPTS, **kwargs)
    return document


def _minimal(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "a", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
        "final_metrics": {"total_steps": 1},
    }
    document.update(overrides)
    return document


def _fields(trace: CanonicalTrace, loss_class: LossClass) -> set[str]:
    return {
        record.field
        for record in trace.losses.records
        if record.loss_class is loss_class
    }


# ---------------------------------------------------------------------------
# The documents this repository actually produces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("document_factory", [_rich_document, _hub_document])
def test_a_real_document_reads_back_into_a_valid_trace(document_factory):
    """Both writers' output is readable, and satisfies every IR invariant."""
    trace = atif_to_ir(document_factory())
    assert validate_trace(trace) == []
    assert trace.provenance.source_format == ATIF_SOURCE
    assert trace.losses.direction == LOSS_DIRECTION


def test_the_two_writers_produce_the_same_trace():
    """Parity, read from the other side.

    Slice D asserts the hub and the direct exporter write the same document.
    Reading both back has to give the same trace — a weaker claim implied by
    the first, and worth pinning here because this suite would otherwise never
    notice if that parity broke underneath it.
    """
    direct = atif_to_ir(_rich_document()).model_dump(mode="json")
    hub = atif_to_ir(_hub_document()).model_dump(mode="json")
    assert direct == hub


def test_step_order_is_preserved_and_indices_are_dense():
    trace = atif_to_ir(_rich_document())
    assert [event.index for event in trace.events] == list(range(len(trace.events)))


def test_every_step_becomes_an_event():
    """No step is silently skipped — the failure mode every exporter has today."""
    document = _rich_document()
    trace = atif_to_ir(document)
    assert len(trace.events) == len(document["steps"])


# ---------------------------------------------------------------------------
# The rule: verbatim, never inferred
# ---------------------------------------------------------------------------


def test_empty_arguments_are_read_as_observed_not_as_absent():
    """The central rule, at the field that matters most.

    `ir_to_atif` writes ``{}`` for a call whose IR ``arguments`` is ``None`` and
    declares it SYNTHESIZED. Nothing in the document marks it as invented, so
    reading it back as ``None`` would be this converter guessing which values
    its counterpart fabricated. It reads ``{}``.

    That is the fabrication the round trip measures: ``None`` ("never observed")
    becomes ``{}`` ("observed empty"), and the tri-state contract cannot tell.
    """
    trace = atif_to_ir(_rich_document())
    calls = [event.tool_call for event in trace.events if event.tool_call]
    assert calls, "fixture must contain a tool call"
    assert all(call.arguments == {} for call in calls)
    assert all(call.arguments is not None for call in calls)
    # And nothing is declared, because nothing was lost *here*.
    assert not [
        record
        for record in trace.losses.records
        if record.field.endswith("tool_call.arguments")
    ]


def test_unknown_agent_version_is_read_verbatim():
    """``"unknown"`` is the exporter's filler and also a legal observed value.

    Translating it back to ``None`` would recover the ACP-side truth by
    guessing, and would silently corrupt a document that really did carry an
    agent named ``unknown``.
    """
    trace = atif_to_ir(_rich_document())
    assert trace.agent.agent_version == "unknown"


def test_empty_message_is_read_as_observed_empty_text():
    """ATIF requires a message on every step; tool steps get ``""``.

    The IR distinguishes ``None`` from ``""``, and this edge cannot tell which
    empty string was observed and which was filler — so it reads what is there.
    """
    trace = atif_to_ir(_rich_document())
    tool_events = [event for event in trace.events if event.kind is EventKind.TOOL_CALL]
    assert tool_events
    assert all(event.text == "" for event in tool_events)


def test_a_fused_step_stays_one_event():
    """A step with message *and* reasoning_content is one event, not two.

    The outbound edge folded a run of reasoning events into the next agent
    step, joined by a blank line. The join is not injective (§5 loss #10), so
    splitting it here would invent boundaries. The fusion is reported by the
    event count, not undone.
    """
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "done",
                "reasoning_content": "first\n\nsecond",
            }
        ]
    )
    trace = atif_to_ir(document)
    assert len(trace.events) == 1
    event = trace.events[0]
    assert event.kind is EventKind.AGENT_MESSAGE
    assert event.text == "done"
    assert event.reasoning == "first\n\nsecond"
    assert event.reasoning_segments == ["first\n\nsecond"]
    assert validate_trace(trace) == []


def test_a_flushed_thought_step_reads_as_reasoning():
    """message ``""`` plus reasoning_content is what a flushed thought looks
    like, and it comes back as a reasoning event."""
    document = _minimal(
        steps=[
            {"step_id": 1, "source": "agent", "message": "", "reasoning_content": "t"}
        ]
    )
    event = atif_to_ir(document).events[0]
    assert event.kind is EventKind.AGENT_REASONING
    assert event.reasoning == "t"


def test_no_prompt_step_is_recognized_as_synthetic():
    """The leading `user` steps built from *prompts* read as user messages.

    They are not trace data — `ir_to_atif` declares each one SYNTHESIZED in the
    target space — but the document does not say so, and this edge does not
    guess. The document's first two user steps carry the same text and both
    become events, which is the over-count §5.2 measured, now visible from the
    other side.
    """
    trace = atif_to_ir(_rich_document())
    user_texts = [
        event.text for event in trace.events if event.kind is EventKind.USER_MESSAGE
    ]
    assert user_texts[: len(PROMPTS)] == PROMPTS


# ---------------------------------------------------------------------------
# Step shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind", "role"),
    [
        ("user", EventKind.USER_MESSAGE, Role.USER),
        ("agent", EventKind.AGENT_MESSAGE, Role.AGENT),
        ("oracle", EventKind.ORACLE, Role.ORACLE),
    ],
)
def test_step_source_maps_to_kind_and_role(source, kind, role):
    document = _minimal(steps=[{"step_id": 1, "source": source, "message": "m"}])
    event = atif_to_ir(document).events[0]
    assert event.kind is kind
    assert event.role is role
    assert event.source_type == source


def test_an_unknown_step_source_survives_whole():
    """Every exporter in the tree drops what it does not recognize. This does
    not: the kind becomes UNKNOWN and the source string is kept."""
    document = _minimal(steps=[{"step_id": 1, "source": "environment", "message": "m"}])
    event = atif_to_ir(document).events[0]
    assert event.kind is EventKind.UNKNOWN
    assert event.source_type == "environment"
    assert event.text == "m"
    assert event.role is None


def test_tool_call_fields_are_read_into_the_ir():
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "execute",
                        "arguments": {"cmd": "ls"},
                        "extra": {"title": "list files", "status": "completed"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "a\nb"}]
                },
            }
        ]
    )
    event = atif_to_ir(document).events[0]
    assert event.kind is EventKind.TOOL_CALL
    call = event.tool_call
    assert call.call_id == "c1"
    assert call.name == "execute"
    assert call.title == "list files"
    assert call.status is ToolStatus.COMPLETED
    assert call.arguments == {"cmd": "ls"}
    assert call.content == [ContentBlock(kind=ContentBlockKind.TEXT, text="a\nb")]


def test_function_name_semantics_are_recorded_as_the_document_states_them():
    """ATIF calls the slot ``function_name`` and the IR records that, even
    though an ACP-derived document holds a *kind* in it.

    This is the normalization `ir_to_atif` declared on the way out, seen from
    the other side: the semantics the ACP edge recorded (``acp_kind``) is not
    in the document, so it cannot come back.
    """
    trace = atif_to_ir(_rich_document())
    calls = [event.tool_call for event in trace.events if event.tool_call]
    assert all(call.name_semantics == "function_name" for call in calls)


def test_a_multi_call_step_becomes_one_event_per_call():
    """The IR models one tool call per event, so an n-call step is n events.

    No writer in this repository emits such a step; another producer can, and
    the grouping it had is declared lost rather than silently flattened.
    """
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "a", "arguments": {}},
                    {"tool_call_id": "c2", "function_name": "b", "arguments": {}},
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "c2", "content": "second"},
                        {"source_call_id": "c1", "content": "first"},
                    ]
                },
            }
        ]
    )
    trace = atif_to_ir(document)
    assert [event.index for event in trace.events] == [0, 1]
    assert [event.tool_call.call_id for event in trace.events] == ["c1", "c2"]
    # Results are matched by id, not by position.
    assert trace.events[0].tool_call.content[0].text == "first"
    assert trace.events[1].tool_call.content[0].text == "second"
    assert validate_trace(trace) == []
    normalized = [
        record
        for record in trace.losses.records
        if record.loss_class is LossClass.NORMALIZED
        and record.space is PathSpace.SOURCE
        and record.field == "steps[0]"
    ]
    assert len(normalized) == 1


def test_a_result_matching_no_call_is_kept_in_extensions():
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "a", "arguments": {}}
                ],
                "observation": {
                    "results": [{"source_call_id": "other", "content": "orphan"}]
                },
            }
        ]
    )
    trace = atif_to_ir(document)
    event = trace.events[0]
    assert event.tool_call.content == []
    assert event.extensions["unmatched_observation_results"] == [
        {"source_call_id": "other", "content": "orphan"}
    ]


def test_an_observation_without_a_tool_call_is_kept_verbatim():
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "m",
                "observation": {"results": [{"source_call_id": "x", "content": "c"}]},
            }
        ]
    )
    trace = atif_to_ir(document)
    event = trace.events[0]
    assert event.kind is EventKind.AGENT_MESSAGE
    assert event.extensions["observation"]["results"][0]["content"] == "c"
    assert "events[0].tool_call" in _fields(trace, LossClass.NORMALIZED)


def test_step_id_is_carried_rather_than_mapped_onto_index():
    """Invariant 2 makes ``index`` a dense position, so a document with sparse
    or restarted step_ids would be renumbered with no record of the original."""
    document = _minimal(
        steps=[
            {"step_id": 7, "source": "user", "message": "a"},
            {"step_id": 9, "source": "user", "message": "b"},
        ]
    )
    trace = atif_to_ir(document)
    assert [event.index for event in trace.events] == [0, 1]
    assert [event.extensions["step_id"] for event in trace.events] == [7, 9]


def test_step_metrics_are_carried_without_being_interpreted():
    """No ATIF schema is vendored here, so mapping the vocabulary onto
    TraceUsage would assert a correspondence nobody checked."""
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "m",
                "metrics": {"tokens": 12},
            }
        ]
    )
    trace = atif_to_ir(document)
    assert trace.events[0].usage is None
    assert trace.events[0].extensions["metrics"] == {"tokens": 12}
    assert "events[0].usage" in _fields(trace, LossClass.NORMALIZED)


# ---------------------------------------------------------------------------
# Vocabulary and coercion — this module's own reshaping, always declared
# ---------------------------------------------------------------------------


def test_a_status_outside_the_vocabulary_becomes_unknown_and_is_kept():
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "c",
                        "function_name": "f",
                        "arguments": {},
                        "extra": {"status": "reticulating"},
                    }
                ],
            }
        ]
    )
    trace = atif_to_ir(document)
    event = trace.events[0]
    assert event.tool_call.status is ToolStatus.UNKNOWN
    # The original is kept: normalizing to ``unknown`` must not destroy what the
    # document said, or a future vocabulary could not be reconstructed.
    assert event.extensions["tool_call"]["source_status"] == "reticulating"
    assert "events[0].tool_call.status" in _fields(trace, LossClass.NORMALIZED)


def test_unmapped_tool_call_keys_are_kept_under_their_own_extension_key():
    """``ToolCall`` has no extensions of its own, and merging its leftovers into
    the event's would let a step key and a tool-call key collide."""
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "latency_ms": "step",
                "tool_calls": [
                    {
                        "tool_call_id": "c",
                        "function_name": "f",
                        "arguments": {},
                        "latency_ms": "call",
                        "extra": {"title": "t", "vendor_field": 1},
                    }
                ],
            }
        ]
    )
    extensions = atif_to_ir(document).events[0].extensions
    assert extensions["latency_ms"] == "step"
    assert extensions["tool_call"]["latency_ms"] == "call"
    assert extensions["tool_call"]["extra"] == {"vendor_field": 1}


def test_a_non_string_message_is_coerced_and_declared():
    document = _minimal(steps=[{"step_id": 1, "source": "user", "message": 42}])
    trace = atif_to_ir(document)
    assert trace.events[0].text == "42"
    assert "events[0].text" in _fields(trace, LossClass.NORMALIZED)


def test_missing_arguments_are_declared_so_the_invariant_holds():
    """A non-conformant document with no ``arguments`` produces ``None``, and
    invariant 7 requires that absence to be declared."""
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": "c", "function_name": "f"}],
            }
        ]
    )
    trace = atif_to_ir(document)
    assert trace.events[0].tool_call.arguments is None
    assert "events[0].tool_call.arguments" in _fields(trace, LossClass.UNSUPPORTED)
    assert validate_trace(trace) == []


def test_non_object_arguments_are_kept_and_declared():
    document = _minimal(
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c", "function_name": "f", "arguments": "ls -la"}
                ],
            }
        ]
    )
    trace = atif_to_ir(document)
    event = trace.events[0]
    assert event.tool_call.arguments is None
    # Not modelled, but not thrown away either.
    assert event.extensions["tool_call"]["source_arguments"] == "ls -la"
    assert "events[0].tool_call.arguments" in _fields(trace, LossClass.UNSUPPORTED)
    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# Malformed input — a document read off disk is not a document we wrote
# ---------------------------------------------------------------------------


def test_a_non_mapping_document_is_rejected():
    with pytest.raises(TypeError):
        atif_to_ir(["not", "a", "document"])


def test_a_non_object_step_is_declared_in_the_source_space():
    document = _minimal(
        steps=[{"step_id": 1, "source": "user", "message": "a"}, "junk"]
    )
    trace = atif_to_ir(document)
    assert len(trace.events) == 1
    dropped = [
        record
        for record in trace.losses.records
        if record.space is PathSpace.SOURCE and record.field == "steps[1]"
    ]
    assert len(dropped) == 1
    assert validate_trace(trace) == []


def test_steps_of_the_wrong_type_leave_an_empty_but_valid_trace():
    trace = atif_to_ir(_minimal(steps={"not": "a list"}))
    assert trace.events == []
    assert validate_trace(trace) == []
    assert "steps" in {
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.SOURCE
    }


def test_a_document_with_no_steps_is_a_valid_empty_trace():
    trace = atif_to_ir({"schema_version": "ATIF-v1.7"})
    assert trace.events == []
    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# final_metrics
# ---------------------------------------------------------------------------


def test_final_metrics_map_onto_trace_usage():
    document = _minimal(
        final_metrics={
            "total_steps": 1,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "total_cached_tokens": 3,
            "total_cost_usd": 0.5,
        }
    )
    usage = atif_to_ir(document).usage
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    assert usage.cache_read_tokens == 3
    assert usage.cost_usd == 0.5


def test_total_steps_is_declared_dropped_in_the_source_space():
    """The mirror of `ir_to_atif`'s target-space SYNTHESIZED record: a count of
    the document's own steps is not a property of the run."""
    trace = atif_to_ir(_minimal())
    assert "final_metrics.total_steps" in {
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.SOURCE and record.loss_class is LossClass.DROPPED
    }


def test_usage_fields_atif_cannot_carry_are_declared_unsupported():
    document = _minimal(final_metrics={"total_steps": 1, "total_prompt_tokens": 1})
    trace = atif_to_ir(document)
    unsupported = _fields(trace, LossClass.UNSUPPORTED)
    for field in (
        "usage.cache_creation_tokens",
        "usage.reasoning_tokens",
        "usage.total_tokens",
        "usage.source",
        "usage.price_source",
    ):
        assert field in unsupported


def test_a_document_with_no_readable_metrics_declares_the_outermost_node():
    """The addressing rule: name the section, not a field inside it."""
    trace = atif_to_ir(_minimal(final_metrics={"total_steps": 1}))
    assert trace.usage is None
    assert "usage" in _fields(trace, LossClass.UNSUPPORTED)


def test_unknown_final_metrics_keys_are_declared():
    trace = atif_to_ir(_minimal(final_metrics={"total_steps": 1, "total_reward": 1.0}))
    records = [
        record
        for record in trace.losses.records
        if record.field == "final_metrics" and record.space is PathSpace.SOURCE
    ]
    assert len(records) == 1
    assert "total_reward" in records[0].detail


def test_a_non_numeric_token_count_is_declared_rather_than_coerced():
    trace = atif_to_ir(
        _minimal(final_metrics={"total_steps": 1, "total_prompt_tokens": "many"})
    )
    assert trace.usage is None
    assert "usage.input_tokens" in {
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.SOURCE
    }


# ---------------------------------------------------------------------------
# Unmapped document content is carried, not dropped
# ---------------------------------------------------------------------------


def test_schema_version_and_unknown_document_keys_ride_in_extensions():
    trace = atif_to_ir(_minimal(dialect="harbor-1.4"))
    assert trace.extensions["schema_version"] == "ATIF-v1.7"
    assert trace.extensions["dialect"] == "harbor-1.4"


def test_unknown_step_keys_ride_in_event_extensions():
    document = _minimal(
        steps=[{"step_id": 1, "source": "user", "message": "m", "latency_ms": 12}]
    )
    assert atif_to_ir(document).events[0].extensions["latency_ms"] == 12


# ---------------------------------------------------------------------------
# What ATIF never carries — UNSUPPORTED, never DROPPED
# ---------------------------------------------------------------------------


ALWAYS_UNSUPPORTED = (
    "trace_id",
    "started_at",
    "finished_at",
    "agent.provider",
    "outcome",
    "events[].started_at",
    "events[].finished_at",
    "events[].usage",
    "events[].outcome",
)


@pytest.mark.parametrize("field", ALWAYS_UNSUPPORTED)
def test_what_atif_never_carries_is_declared_unsupported(field):
    """``UNSUPPORTED`` and ``DROPPED`` decide where a fix would have to land.

    Nothing in an ATIF document holds these, so calling them dropped would
    blame this converter for an absence it inherited — and would make the two
    inbound reports incomparable, since `ACP → IR` draws the same line.
    """
    trace = atif_to_ir(_rich_document())
    assert field in _fields(trace, LossClass.UNSUPPORTED)


def test_the_inbound_report_declares_nothing_dropped_about_the_hub():
    """Every DROPPED record addresses the source document, not the IR.

    This edge cannot drop an IR field: it is the thing building the IR. What it
    drops are parts of the document with no IR home, and those are source-space
    records by construction.
    """
    trace = atif_to_ir(_rich_document())
    dropped = [
        record
        for record in trace.losses.records
        if record.loss_class is LossClass.DROPPED
    ]
    assert dropped
    assert all(record.space is PathSpace.SOURCE for record in dropped)


def test_tool_call_timestamps_are_only_declared_when_a_call_exists():
    """Declaring a per-call absence for a trace with no calls would describe a
    conversion that never happened."""
    with_calls = atif_to_ir(_rich_document())
    without = atif_to_ir(_minimal())
    assert "events[].tool_call.started_at" in _fields(with_calls, LossClass.UNSUPPORTED)
    assert "events[].tool_call.started_at" not in _fields(
        without, LossClass.UNSUPPORTED
    )


def test_every_hub_loss_path_resolves_in_the_trace_it_describes():
    """A record must address something a reader of the document can find.

    The same guard Slice B applies to the canonical encoding, applied to the
    report this edge attaches: an unresolvable path is a declaration nobody can
    check. Unindexed ``events[].…`` paths are systemic and exempt, since they
    name a field of every event rather than one node.
    """
    trace = atif_to_ir(_rich_document())
    document = trace.model_dump(mode="json")
    for record in trace.losses.records:
        if record.space is not PathSpace.HUB or record.field.startswith("events[]"):
            continue
        resolved, _ = resolve_ir_path(document, record.field)
        assert resolved, record.field


# ---------------------------------------------------------------------------
# Field coverage — read off the models, so a new IR field cannot slip through
# ---------------------------------------------------------------------------


FIELD_DISPOSITION: dict[str, dict[str, str]] = {
    "CanonicalTrace": {
        "ir_version": "representation",
        "trace_id": "declared",
        "session_id": "read",
        "agent": "container",
        "started_at": "declared",
        "finished_at": "declared",
        "events": "container",
        "usage": "read",
        "outcome": "container",
        "provenance": "representation",
        "extensions": "read",
        "losses": "representation",
    },
    "TraceEvent": {
        "index": "read",
        "kind": "read",
        "source_type": "read",
        "role": "read",
        "text": "read",
        "reasoning": "read",
        "reasoning_segments": "read",
        "tool_call": "container",
        "started_at": "declared",
        "finished_at": "declared",
        "outcome": "declared",
        "usage": "declared",
        "provenance": "representation",
        "extensions": "read",
    },
    "ToolCall": {
        "call_id": "read",
        "name": "read",
        "name_semantics": "read",
        "title": "read",
        "status": "read",
        "arguments": "read",
        "content": "container",
        "started_at": "declared",
        "finished_at": "declared",
    },
    "ModelInfo": {
        "agent_name": "read",
        "agent_version": "read",
        "model": "read",
        "provider": "declared",
    },
    "TraceUsage": {
        "input_tokens": "read",
        "output_tokens": "read",
        "cache_read_tokens": "read",
        "cost_usd": "read",
        "cache_creation_tokens": "declared",
        "reasoning_tokens": "declared",
        "total_tokens": "declared",
        "source": "declared",
        "price_source": "declared",
    },
    "TraceOutcome": {
        "status": "declared-as-section",
        "stop_reason": "declared-as-section",
        "reward": "declared-as-section",
        "error_category": "declared-as-section",
    },
    "ContentBlock": {
        "kind": "read",
        "text": "read",
        "raw": "declared",
    },
}


def test_every_ir_field_has_a_disposition_at_this_edge():
    """``read`` is populated from the document, ``declared`` is absent from ATIF
    and recorded, ``container`` holds fields covered by their own entry,
    ``representation`` describes the IR rather than the run, and
    ``declared-as-section`` is covered by the single ``outcome`` record — the
    addressing rule says to name the outermost absent node.
    """
    models = {
        "CanonicalTrace": CanonicalTrace,
        "TraceEvent": TraceEvent,
        "ToolCall": ToolCall,
        "ModelInfo": ModelInfo,
        "TraceUsage": TraceUsage,
        "TraceOutcome": TraceOutcome,
        "ContentBlock": ContentBlock,
    }
    for name, model in models.items():
        assert set(model.model_fields) == set(FIELD_DISPOSITION[name]), {
            "model": name,
            "undecided": sorted(set(model.model_fields) - set(FIELD_DISPOSITION[name])),
            "stale": sorted(set(FIELD_DISPOSITION[name]) - set(model.model_fields)),
        }


def test_every_declared_field_really_produces_a_record():
    """The table is a claim about behaviour, not a comment.

    Every ``declared`` field must appear in the report of a document rich
    enough to reach it — with the per-call ones only when the trace has a call,
    which is the conditional the previous test pins.
    """
    trace = atif_to_ir(_rich_document(total_prompt_tokens=5))
    declared = {
        record.field for record in trace.losses.records if record.space is PathSpace.HUB
    }
    expected = {
        "trace_id",
        "started_at",
        "finished_at",
        "agent.provider",
        "outcome",
        "events[].started_at",
        "events[].finished_at",
        "events[].outcome",
        "events[].usage",
        "events[].tool_call.started_at",
        "events[].tool_call.finished_at",
        "events[].tool_call.content[].raw",
        "usage.cache_creation_tokens",
        "usage.reasoning_tokens",
        "usage.total_tokens",
        "usage.source",
        "usage.price_source",
    }
    assert expected <= declared, sorted(expected - declared)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_this_module_imports_only_the_ir_from_benchflow():
    """The converter depends on the hub and on nothing else in the package.

    An edge that imported the ACP layer or an exporter would make the hub's
    neutrality a matter of convention; here it is a property of the import
    graph.
    """
    tree = ast.parse(Path(ir_from_atif_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    benchflow_imports = {name for name in imported if name.startswith("benchflow")}
    assert benchflow_imports == {"benchflow.trajectories.ir"}
