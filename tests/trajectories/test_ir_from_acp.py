"""Conversion suite for ``ACP capture events → canonical Trace IR`` (Slice C).

Slice B asserted a contract: every ``None`` in the IR is covered by a
``LossRecord``, and a conversion's cost is a value rather than a comment. This
suite is where that contract meets a real converter, so it is written to be
able to *fail* the design, not only the code:

* **Preservation** is checked field by field against events produced by the
  production capture path (``_rich_events`` from the Slice A2 suite drives a
  real :class:`ACPSession` through ``handle_update``), not against hand-written
  dicts that happen to match the converter.
* **The loss report is asserted as a complete set**, not with membership
  checks. An undeclared loss and a spurious one both fail.
* **The volume test** measures how the report scales, because "declare every
  absence" is only a workable contract if the report stays readable on a real
  trace. It pins the property that matters: the report grows with tool calls,
  not with trace length.
* **Robustness** is exercised with input the Slice A schema would reject —
  §7 lists fixtures in this repository that use the ACP filename with other
  shapes, so a converter that only accepts conformant input would be a
  converter for a file that does not always exist.

Nothing here writes to disk and no runtime module is imported for its side
effects.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from benchflow.trajectories import ir_from_acp
from benchflow.trajectories._export_common import content_blocks_to_text
from benchflow.trajectories.ir import (
    ContentBlockKind,
    EventKind,
    LossClass,
    OutcomeStatus,
    Role,
    ToolStatus,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import (
    ACP_CAPTURE_SOURCE,
    ACP_TRAJECTORY_SOURCE,
    LOSS_DIRECTION,
    ORACLE_SOURCE,
    UNKNOWN_SOURCE,
    acp_events_to_ir,
    loss_summary,
    per_event_losses,
    systemic_losses,
)
from tests.trajectories.test_acp_capture_event_schema import _emitted_event_types
from tests.trajectories.test_atif_preservation import (
    PENDING_TOOL_CALL_ID,
    TIMEOUT_SEC,
    _rich_events,
)

# The systemic records every conversion of a tool-bearing trace declares. Held
# as a literal so adding one is a deliberate edit to this suite.
SYSTEMIC_FIELDS = {
    "events[].tool_call.started_at",
    "events[].tool_call.finished_at",
    "events[].usage",
    "agent.agent_version",
    "outcome.stop_reason",
}


def _fields(trace) -> set[str]:
    return {record.field for record in trace.losses.records}


def test_the_converter_depends_on_the_ir_and_nothing_else_in_benchflow():
    """Slice C must not drag the hub into the rest of the tree.

    ``tests/trajectories/test_trace_ir.py`` checks the other direction — that
    nothing outside the family imports the IR. This one checks that the family
    itself stays a leaf: the converter reads the capture format as *data*, and
    reaching into ``_capture`` or an exporter would couple the hub to the very
    modules it is meant to sit between.
    """
    tree = ast.parse(Path(ir_from_acp.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    benchflow_imports = {name for name in imported if name.startswith("benchflow")}
    assert benchflow_imports == {"benchflow.trajectories.ir"}, sorted(benchflow_imports)


# ---------------------------------------------------------------------------
# 1. Field-by-field preservation, from real captured events
# ---------------------------------------------------------------------------


def test_every_emitted_event_type_becomes_a_typed_ir_event():
    """No event the capture path emits may land in ``UNKNOWN``.

    The producer's vocabulary is read out of ``_events_to_trajectory`` by AST
    (the Slice A mechanism), so adding a branch there fails this test until the
    converter handles it — rather than silently degrading it to ``unknown``.
    """
    events = [{"type": etype} for etype in sorted(_emitted_event_types())]
    trace = acp_events_to_ir(events)
    unknown = [e.source_type for e in trace.events if e.kind is EventKind.UNKNOWN]
    assert unknown == [], unknown


def test_text_events_preserve_role_text_and_source_type():
    events = [
        {"type": "user_message", "text": "List the files."},
        {"type": "agent_message", "text": "Done."},
    ]
    trace = acp_events_to_ir(events)

    user, agent = trace.events
    assert (user.kind, user.role, user.text) == (
        EventKind.USER_MESSAGE,
        Role.USER,
        "List the files.",
    )
    assert (agent.kind, agent.role, agent.text) == (
        EventKind.AGENT_MESSAGE,
        Role.AGENT,
        "Done.",
    )
    assert [e.source_type for e in trace.events] == ["user_message", "agent_message"]
    assert all(
        e.provenance.source_format == ACP_CAPTURE_SOURCE
        and e.provenance.producer == "_events_to_trajectory"
        for e in trace.events
    )
    assert trace.provenance.source_format == ACP_TRAJECTORY_SOURCE


def test_empty_text_is_preserved_rather_than_dropped():
    """Both exporters drop text-empty events (§5.1); ``""`` is an observation."""
    trace = acp_events_to_ir([{"type": "agent_message", "text": ""}])
    assert len(trace.events) == 1
    assert trace.events[0].text == ""
    assert trace.events[0].text is not None


def test_a_real_captured_trace_converts_field_by_field():
    events = _rich_events()
    trace = acp_events_to_ir(
        events, session_id="sess-a2", agent_name="claude-code", model="claude-sonnet-5"
    )

    assert [e.source_type for e in trace.events] == [e["type"] for e in events]
    assert trace.session_id == "sess-a2"
    assert trace.agent.agent_name == "claude-code"
    assert trace.agent.model == "claude-sonnet-5"
    # Never fabricated, even though ATIF requires the field.
    assert trace.agent.agent_version is None

    source_tool = next(e for e in events if e["type"] == "tool_call")
    ir_tool = next(e for e in trace.events if e.kind is EventKind.TOOL_CALL)
    assert ir_tool.tool_call.call_id == source_tool["tool_call_id"]
    assert ir_tool.tool_call.name == source_tool["kind"]
    assert ir_tool.tool_call.title == source_tool["title"]
    assert ir_tool.tool_call.status.value == source_tool["status"]
    assert ir_tool.role is Role.AGENT
    # The ACP kind is a category, not a function name — and the IR says so.
    assert ir_tool.tool_call.name_semantics == "acp_kind"

    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# 2. Ordering
# ---------------------------------------------------------------------------


def test_ordering_and_dense_indices_are_preserved():
    events = [{"type": "agent_message", "text": f"m{n}"} for n in range(6)]
    events.insert(3, {"type": "tool_call", "tool_call_id": "x", "kind": "read"})
    trace = acp_events_to_ir(events)

    assert [e.index for e in trace.events] == list(range(len(events)))
    assert [e.source_type for e in trace.events] == [e["type"] for e in events]


def test_an_unrepresentable_entry_leaves_no_hole():
    """A skipped source entry is declared under ``source[i]``, not ``events[i]``.

    The distinction matters: index 1 of the IR belongs to the event that
    followed the skipped entry, so addressing the loss as ``events[1]`` would
    blame a different event.
    """
    events: list[Any] = [
        {"type": "agent_message", "text": "a"},
        "not an object",
        {"type": "agent_message", "text": "b"},
    ]
    trace = acp_events_to_ir(events)

    assert [e.index for e in trace.events] == [0, 1]
    assert [e.text for e in trace.events] == ["a", "b"]
    dropped = trace.losses.for_field("source[1]")
    assert len(dropped) == 1
    assert dropped[0].loss_class is LossClass.DROPPED
    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# 3. Tri-state arguments
# ---------------------------------------------------------------------------


def test_arguments_are_none_never_empty_and_always_declared():
    """`{}` would claim the agent called the tool with no arguments (§8.2)."""
    trace = acp_events_to_ir(
        [
            {"type": "tool_call", "tool_call_id": "a", "kind": "execute"},
            {"type": "tool_call", "tool_call_id": "b", "kind": "read"},
        ]
    )
    for position, event in enumerate(trace.events):
        assert event.tool_call.arguments is None
        assert event.tool_call.arguments != {}
        declared = trace.losses.for_field(f"events[{position}].tool_call.arguments")
        assert len(declared) == 1
        assert declared[0].loss_class is LossClass.UNSUPPORTED
        assert declared[0].doc_ref == "§5 loss #1"
    assert validate_trace(trace) == []


def test_the_serialized_trace_never_carries_an_empty_argument_map():
    """A whole-document check, in the A2 style: no `"arguments": {}` anywhere."""
    document = acp_events_to_ir(_rich_events()).model_dump(mode="json")
    assert '"arguments": {}' not in json.dumps(document, indent=1)


# ---------------------------------------------------------------------------
# 4. Timeout
# ---------------------------------------------------------------------------


def test_the_timeout_marker_survives_with_its_fields():
    """§5 loss #4: every exporter drops this event. The IR keeps it whole."""
    events = _rich_events()
    trace = acp_events_to_ir(events)

    timeout = next(e for e in trace.events if e.kind is EventKind.TIMEOUT)
    assert timeout.source_type == "agent_timeout"
    assert timeout.outcome == "wall_clock_timeout"
    assert timeout.extensions["timeout_sec"] == TIMEOUT_SEC
    assert timeout.extensions["pending_tool_call_ids"] == [PENDING_TOOL_CALL_ID]
    assert timeout.extensions["terminal_trajectory_complete"] is False
    # BenchFlow's own marker: not an agent action, so no role is invented.
    assert timeout.role is None
    assert trace.outcome.status is OutcomeStatus.TIMEOUT


def test_a_trace_without_a_timeout_claims_no_outcome():
    """The capture events say nothing about pass/fail; that lives in result.json."""
    trace = acp_events_to_ir([{"type": "agent_message", "text": "done"}])
    assert trace.outcome is None


# ---------------------------------------------------------------------------
# 5. Reasoning boundaries
# ---------------------------------------------------------------------------


def test_consecutive_thoughts_keep_their_boundaries():
    """The loss the ATIF/ADP path cannot avoid (#10) is avoided by not joining."""
    events = [
        {"type": "agent_thought", "text": "first"},
        {"type": "agent_thought", "text": "second"},
    ]
    trace = acp_events_to_ir(events)

    assert [e.kind for e in trace.events] == [EventKind.AGENT_REASONING] * 2
    assert [e.reasoning_segments for e in trace.events] == [["first"], ["second"]]
    assert validate_trace(trace) == []


def test_a_thought_containing_a_blank_line_stays_one_segment():
    """Splitting it would invent a boundary the source does not have.

    This is the exact ambiguity §5 loss #10 describes: after ``ThoughtBuffer``
    joins, one thought with a blank line and two thoughts are indistinguishable.
    The converter refuses to guess in either direction.
    """
    trace = acp_events_to_ir([{"type": "agent_thought", "text": "one\n\ntwo"}])
    event = trace.events[0]
    assert event.reasoning_segments == ["one\n\ntwo"]
    assert event.reasoning == "one\n\ntwo"
    assert validate_trace(trace) == []


def test_reasoning_is_not_merged_into_text():
    trace = acp_events_to_ir([{"type": "agent_thought", "text": "thinking"}])
    assert trace.events[0].text is None
    assert trace.events[0].reasoning == "thinking"


# ---------------------------------------------------------------------------
# 6. Unknown, oracle and non-conformant records
# ---------------------------------------------------------------------------


def test_unknown_event_types_are_carried_not_dropped():
    """Every exporter skips these silently today (§5.1, last rows)."""
    raw = {"type": "some_future_event", "payload": {"a": 1}, "n": 3}
    trace = acp_events_to_ir([raw])

    event = trace.events[0]
    assert event.kind is EventKind.UNKNOWN
    assert event.source_type == "some_future_event"
    assert event.extensions == raw
    assert event.provenance.source_format == UNKNOWN_SOURCE
    assert validate_trace(trace) == []


def test_a_record_without_a_type_is_still_carried():
    trace = acp_events_to_ir([{"role": "assistant"}])
    assert trace.events[0].kind is EventKind.UNKNOWN
    assert trace.events[0].source_type is None
    assert trace.events[0].extensions == {"role": "assistant"}


def test_the_oracle_record_keeps_its_identity():
    """ATIF renders it as an agent step prefixed ``[oracle: …]`` (§5.1)."""
    raw = {"type": "oracle", "command": "solve.sh", "return_code": 0, "stdout": "ok"}
    trace = acp_events_to_ir([raw])

    event = trace.events[0]
    assert event.kind is EventKind.ORACLE
    assert event.role is Role.ORACLE
    assert event.provenance.source_format == ORACLE_SOURCE
    assert event.extensions == {
        "command": "solve.sh",
        "return_code": 0,
        "stdout": "ok",
    }
    assert "[oracle:" not in json.dumps(trace.model_dump(mode="json"))


def test_extra_fields_on_a_known_record_are_carried_into_extensions():
    """A capture-layer change that adds a field must not be truncated away."""
    trace = acp_events_to_ir(
        [{"type": "agent_message", "text": "hi", "future_field": 42}]
    )
    assert trace.events[0].extensions == {"future_field": 42}
    assert trace.events[0].text == "hi"


def test_non_string_values_are_coerced_and_the_coercion_is_declared():
    """Unreachable from the emitter, reachable from the file (§7)."""
    trace = acp_events_to_ir([{"type": "agent_message", "text": 42}])
    assert trace.events[0].text == "42"
    declared = trace.losses.for_field("events[0].text")
    assert len(declared) == 1
    assert declared[0].loss_class is LossClass.NORMALIZED


def test_an_out_of_vocabulary_status_is_mapped_and_kept_recoverable():
    trace = acp_events_to_ir(
        [{"type": "tool_call", "tool_call_id": "a", "status": "exploded"}]
    )
    event = trace.events[0]
    assert event.tool_call.status is ToolStatus.UNKNOWN
    assert event.extensions["source_status"] == "exploded"
    assert (
        trace.losses.for_field("events[0].tool_call.status")[0].loss_class
        is LossClass.NORMALIZED
    )


# ---------------------------------------------------------------------------
# 7. Tool status, title, id and content
# ---------------------------------------------------------------------------


def test_empty_tool_id_and_title_are_preserved_as_empty_not_synthesized():
    """ATIF and ADP both synthesize ids here; the IR records what happened."""
    trace = acp_events_to_ir(
        [
            {
                "type": "tool_call",
                "tool_call_id": "",
                "kind": "tool",
                "title": "",
                "status": "in_progress",
                "content": [],
            }
        ]
    )
    call = trace.events[0].tool_call
    assert call.call_id == ""
    assert call.title == ""
    assert call.status is ToolStatus.IN_PROGRESS
    assert call.content == []
    assert not any("call_" in record.field for record in trace.losses.records)


def test_absent_tool_fields_stay_none_and_are_not_confused_with_empty():
    trace = acp_events_to_ir([{"type": "tool_call"}])
    call = trace.events[0].tool_call
    assert call.call_id is None
    assert call.title is None
    assert call.name is None
    assert call.name_semantics is None
    assert call.status is None


def test_non_text_content_blocks_are_carried_as_opaque():
    """§5 loss #5: ``content_blocks_to_text`` skips these; the IR keeps them."""
    diff_block = {"type": "diff", "path": "/w/a.py", "oldText": "a", "newText": "b"}
    trace = acp_events_to_ir(
        [
            {
                "type": "tool_call",
                "tool_call_id": "t1",
                "kind": "edit",
                "status": "completed",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "ok"}},
                    diff_block,
                ],
            }
        ]
    )
    blocks = trace.events[0].tool_call.content
    assert [b.kind for b in blocks] == [ContentBlockKind.TEXT, ContentBlockKind.OPAQUE]
    assert blocks[0].text == "ok"
    assert blocks[1].raw == diff_block
    # The verbatim block reached the serialized trace, not just the object.
    assert "newText" in json.dumps(trace.model_dump(mode="json"))
    assert validate_trace(trace) == []


def test_text_blocks_agree_with_the_shared_renderer():
    """Pins the classifier against ``content_blocks_to_text`` rather than a copy.

    The two must recognize the same blocks as text, or the IR and every
    existing consumer would disagree about what a tool produced.
    """
    content = [
        {"type": "content", "content": {"type": "text", "text": "first"}},
        {"text": "flat form"},
        {"type": "diff", "oldText": "a", "newText": "b"},
        {"type": "content", "content": {"type": "image", "data": "…"}},
    ]
    trace = acp_events_to_ir(
        [{"type": "tool_call", "tool_call_id": "t", "content": content}]
    )
    blocks = trace.events[0].tool_call.content
    rendered = "\n".join(b.text for b in blocks if b.kind is ContentBlockKind.TEXT)
    assert rendered == content_blocks_to_text(content)
    assert sum(1 for b in blocks if b.kind is ContentBlockKind.OPAQUE) == 2


def test_a_non_object_content_block_is_declared_dropped():
    trace = acp_events_to_ir(
        [{"type": "tool_call", "tool_call_id": "t", "content": ["bare string"]}]
    )
    assert trace.events[0].tool_call.content == []
    declared = trace.losses.for_field("events[0].tool_call.content")
    assert len(declared) == 1
    assert declared[0].loss_class is LossClass.DROPPED


def test_a_string_content_field_is_read_like_the_shared_renderer_reads_it():
    trace = acp_events_to_ir(
        [{"type": "tool_call", "tool_call_id": "t", "content": "plain output"}]
    )
    blocks = trace.events[0].tool_call.content
    assert [b.kind for b in blocks] == [ContentBlockKind.TEXT]
    assert blocks[0].text == content_blocks_to_text("plain output")


# ---------------------------------------------------------------------------
# 8. Loss report completeness
# ---------------------------------------------------------------------------


def test_the_report_for_a_real_trace_is_exactly_this_set():
    """Asserted as a set: an undeclared loss and a spurious one both fail."""
    trace = acp_events_to_ir(_rich_events())
    tool_index = next(e.index for e in trace.events if e.kind is EventKind.TOOL_CALL)
    assert _fields(trace) == SYSTEMIC_FIELDS | {
        f"events[{tool_index}].tool_call.arguments"
    }
    assert trace.losses.direction == LOSS_DIRECTION
    assert loss_summary(trace.losses) == {"unsupported": 6}


def test_every_record_names_a_symbol_or_a_document_section():
    """A loss whose detail does not say *why* is a loss nobody can act on."""
    trace = acp_events_to_ir(_rich_events())
    for record in trace.losses.records:
        assert record.detail.strip()
        assert record.doc_ref or any(
            token in record.detail
            for token in ("handle_update", "ToolCallRecord", "usage_snapshots", "ATIF")
        ), record


def test_a_trace_with_no_tool_calls_declares_no_tool_losses():
    trace = acp_events_to_ir([{"type": "agent_message", "text": "hi"}])
    assert _fields(trace) == {
        "events[].usage",
        "agent.agent_version",
        "outcome.stop_reason",
    }


def test_an_empty_event_list_still_produces_a_report():
    """An empty report would be a claim that nothing was lost."""
    trace = acp_events_to_ir([])
    assert trace.events == []
    assert not trace.losses.lossless
    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# 9 & 10. The contract itself
# ---------------------------------------------------------------------------


def test_validate_trace_is_green_on_every_shape_this_suite_exercises():
    shapes: list[list[Any]] = [
        [],
        _rich_events(),
        [{"type": "oracle", "command": "x"}],
        [{"type": "mystery"}, {"no": "type"}, "not an object", 7],
        [{"type": "tool_call"}],
        [{"type": "agent_thought", "text": "a\n\nb"}],
        [{"type": "tool_call", "tool_call_id": "t", "content": [1, 2, 3]}],
    ]
    for events in shapes:
        trace = acp_events_to_ir(events)
        assert validate_trace(trace) == [], (events, validate_trace(trace))


def test_removing_one_declared_loss_makes_the_trace_invalid():
    """The Slice B contract, demonstrated end to end on a real conversion.

    This is what stops a future converter from quietly failing to carry
    arguments: the absence is only legal while it is declared.
    """
    trace = acp_events_to_ir(_rich_events())
    assert validate_trace(trace) == []

    field = next(f for f in _fields(trace) if f.endswith(".tool_call.arguments"))
    trace.losses.records = [r for r in trace.losses.records if r.field != field]

    issues = validate_trace(trace)
    assert any("absence must be declared" in issue for issue in issues), issues


def test_dropping_the_whole_report_invalidates_a_tool_bearing_trace():
    trace = acp_events_to_ir(_rich_events())
    trace.losses = None
    assert validate_trace(trace) != []


# ---------------------------------------------------------------------------
# 11. Volume and ergonomics
# ---------------------------------------------------------------------------


def _synthetic_trace_events(tool_calls: int, chatter: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{"type": "user_message", "text": "go"}]
    for n in range(tool_calls):
        events.append({"type": "agent_thought", "text": f"thinking {n}"})
        events.append(
            {
                "type": "tool_call",
                "tool_call_id": f"call_{n}",
                "kind": "execute",
                "title": f"cmd {n}",
                "status": "completed",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "out"}}
                ],
            }
        )
    for n in range(chatter):
        events.append({"type": "agent_message", "text": f"message {n}"})
    return events


def test_the_report_grows_with_tool_calls_not_with_trace_length():
    """The ergonomics property that decides whether the contract is workable.

    "Declare every absence" is only affordable if the report is bounded by the
    thing the absences are about. Doubling the chatter must not change it.
    """
    lean = acp_events_to_ir(_synthetic_trace_events(tool_calls=40, chatter=5))
    chatty = acp_events_to_ir(_synthetic_trace_events(tool_calls=40, chatter=200))

    assert len(lean.losses.records) == len(chatty.losses.records)
    assert len(systemic_losses(lean.losses)) == len(SYSTEMIC_FIELDS)
    assert len(per_event_losses(lean.losses)) == 40
    assert len(lean.losses.records) == 40 + len(SYSTEMIC_FIELDS)


def test_a_realistic_trace_keeps_the_report_readable():
    """One record per tool call, and a summary that fits on one line."""
    events = _synthetic_trace_events(tool_calls=50, chatter=20)
    trace = acp_events_to_ir(events)

    assert len(trace.events) == len(events)
    assert loss_summary(trace.losses) == {"unsupported": 55}
    # Every per-event record says the same thing about a different call, which
    # is why `loss_summary` exists — the detail is worth reading once.
    details = {r.detail for r in per_event_losses(trace.losses)}
    assert len(details) == 1

    report_bytes = len(trace.losses.model_dump_json())
    trace_bytes = len(trace.model_dump_json())
    assert report_bytes < trace_bytes
