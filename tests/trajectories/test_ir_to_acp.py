"""Conversion suite for ``canonical Trace IR → ACP capture events`` (Slice G).

The target is the **ACP-session capture event format** pinned by Slice A's JSON
Schema, not the `acp_trajectory.jsonl` artifact — §2.1 records that no
artifact-level contract exists, and several producers write records into that
file that the schema deliberately does not model.

Two properties carry most of the weight here:

- **the round-trip anchor** — a conformant ACP event list converted to the IR
  and back is the same event list, which is what makes "this edge writes the
  format" a measurement rather than an assertion;
- **fail closed** — a trace with one unrepresentable event yields no events at
  all, because a partial list is indistinguishable from a complete one once it
  leaves the function.

Everything else is the negative space around those: the values this edge refuses
to invent, and the proof that refusing is what actually happens.

Nothing here writes to disk and nothing imports a run path. `_capture.py` is
imported read-only, as the producer whose records this edge has to reproduce.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from benchflow.trajectories import ir_to_acp as ir_to_acp_module
from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    ModelInfo,
    PathSpace,
    Provenance,
    Role,
    ToolCall,
    ToolStatus,
    TraceEvent,
    TraceOutcome,
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_to_acp import (
    ACP_TIMEOUT_REASON,
    ACP_TOOL_STATUSES,
    LOSS_DIRECTION,
    AcpCaptureNotRepresentable,
    acp_capture_blockers,
    ir_to_acp_capture_events,
)
from benchflow.trajectories.types import redact_acp_trajectory_jsonl
from tests.trajectories.test_trace_ir import resolve_ir_path

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/benchflow/trajectories/schemas/acp-capture-event-v1.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def assert_schema_valid(events: list[dict[str, Any]]) -> None:
    """Every emitted record must validate against the Slice A contract.

    This is the check that makes "the target is the capture format" mean
    something: the edge is held to the published schema rather than to whatever
    it happens to produce.
    """
    for position, record in enumerate(events):
        errors = sorted(VALIDATOR.iter_errors(record), key=lambda e: e.json_path)
        assert errors == [], (position, record, [e.message for e in errors])


CONFORMANT: list[dict[str, Any]] = [
    {"type": "user_message", "text": "solve the task"},
    {"type": "agent_thought", "text": "reading the repo"},
    {
        "type": "tool_call",
        "tool_call_id": "tc-1",
        "kind": "execute",
        "title": "ls -la",
        "status": "completed",
        "content": [{"type": "content", "content": {"type": "text", "text": "out"}}],
    },
    {"type": "agent_message", "text": ""},
    {
        "type": "agent_timeout",
        "reason": "wall_clock_timeout",
        "timeout_sec": 300,
        "pending_tool_call_ids": ["tc-9"],
        "terminal_trajectory_complete": True,
    },
]
"""One record of every shape the emitter produces, including the two empty
values the contract documents: an empty `agent_message` and a text-empty title
is exercised separately."""


def tool_event(index: int = 0, **overrides: Any) -> TraceEvent:
    """A representable tool-call event, before overrides."""
    call_fields: dict[str, Any] = {
        "call_id": "tc-1",
        "name": "execute",
        # An ACP kind is a category, and only a name labelled as one may be
        # written into that slot — so a representable tool call must say so.
        "name_semantics": "acp_kind",
        "title": "ls",
        "status": ToolStatus.COMPLETED,
        "arguments": {},
        "content": [
            ContentBlock(kind=ContentBlockKind.TEXT, text="o", raw={"text": "o"})
        ],
    }
    for key in list(overrides):
        if key in call_fields:
            call_fields[key] = overrides.pop(key)
    event: dict[str, Any] = {
        "index": index,
        "kind": EventKind.TOOL_CALL,
        "tool_call": ToolCall(**call_fields),
        "provenance": Provenance(source_format="hand-built"),
    }
    event.update(overrides)
    return TraceEvent(**event)


def timeout_event(index: int = 0, **overrides: Any) -> TraceEvent:
    extensions = overrides.pop(
        "extensions",
        {
            "timeout_sec": 300,
            "pending_tool_call_ids": [],
            "terminal_trajectory_complete": True,
        },
    )
    event: dict[str, Any] = {
        "index": index,
        "kind": EventKind.TIMEOUT,
        "outcome": ACP_TIMEOUT_REASON,
        "extensions": extensions,
        "provenance": Provenance(source_format="hand-built"),
    }
    event.update(overrides)
    return TraceEvent(**event)


def trace_of(*events: TraceEvent, **overrides: Any) -> CanonicalTrace:
    payload: dict[str, Any] = {
        "events": list(events),
        "provenance": Provenance(source_format="hand-built"),
    }
    payload.update(overrides)
    return CanonicalTrace(**payload)


def blocker_fields(trace: CanonicalTrace) -> list[str]:
    return [record.field for record in acp_capture_blockers(trace)]


# ---------------------------------------------------------------------------
# The anchor: ACP → IR → ACP
# ---------------------------------------------------------------------------


def test_a_conformant_event_list_survives_the_round_trip():
    """The property the whole slice rests on.

    If this fails, the edge is not writing the capture format — it is writing
    something that resembles it.

    **Scope of the evidence.** This is measured on `CONFORMANT` — five records,
    one of each shape the emitter produces — and, in the drop-one test below, on
    its five subsets. It is **not** a demonstration that every conformant
    capture document round-trips: there is no property test and no corpus taken
    from captured rollouts. What holds for unmeasured inputs is the weaker but
    structural guarantee that the edge either reproduces its input or refuses,
    so an unmeasured shape cannot quietly produce a degraded record.
    """
    trace = acp_events_to_ir(CONFORMANT)
    assert validate_trace(trace) == []
    events, report = ir_to_acp_capture_events(trace)

    assert events == CONFORMANT
    assert report.direction == LOSS_DIRECTION
    assert_schema_valid(events)


def test_the_round_trip_preserves_key_order_too():
    """Structural equality does not pin key order; this does.

    It matters because the artifact is JSONL: two records with the same keys in
    a different order are the same object and different bytes, and the capture
    file is read by tools that diff it.
    """
    trace = acp_events_to_ir(CONFORMANT)
    events, _ = ir_to_acp_capture_events(trace)
    for original, produced in zip(CONFORMANT, events, strict=True):
        assert list(original.keys()) == list(produced.keys())


def test_the_round_trip_is_byte_identical_through_the_project_serializer():
    """True, and worth stating precisely rather than loudly.

    `redact_acp_trajectory_jsonl` is what every write path in the repository
    uses, and the two serializations are identical. But this is **not an
    independent property of the conversion**: the serializer applies redaction
    to both sides, so redaction cancels, and what byte equality adds over
    structural equality is exactly the key order the previous test pins. The
    honest claim for this edge is structural equality plus key order; the byte
    result is their consequence.
    """
    trace = acp_events_to_ir(CONFORMANT)
    events, _ = ir_to_acp_capture_events(trace)
    assert redact_acp_trajectory_jsonl(events) == redact_acp_trajectory_jsonl(
        CONFORMANT
    )


def test_the_serializer_redacts_which_is_why_the_byte_claim_is_qualified():
    """The reason the test above is not the headline.

    A secret-shaped value is rewritten on the way out, by the serializer and not
    by this edge. A byte claim stated over that function would be partly a claim
    about redaction.
    """
    events = [{"type": "agent_message", "text": "AKIAIOSFODNN7EXAMPLE"}]
    assert "AKIAIOSFODNN7EXAMPLE" not in redact_acp_trajectory_jsonl(events)


@pytest.mark.parametrize("dropped", range(len(CONFORMANT)))
def test_every_record_shape_round_trips_on_its_own(dropped):
    """The anchor holds per shape, not only for the full list.

    Guards against a mapping that is right in aggregate because two errors
    cancel — dropping one record and mis-emitting another would still produce a
    list of the same length.
    """
    subset = [e for i, e in enumerate(CONFORMANT) if i != dropped]
    events, _ = ir_to_acp_capture_events(acp_events_to_ir(subset))
    assert events == subset
    assert_schema_valid(events)


# ---------------------------------------------------------------------------
# Representability is about the data, not the provenance
# ---------------------------------------------------------------------------


def test_provenance_is_never_read_to_decide_representability():
    """Two traces, identical data, different declared origin.

    The rule is that the ACP contract is satisfied by values, not by lineage. A
    converter that consulted `provenance` would be deciding a data question with
    a metadata answer, and would make an OTel trace that grew the missing fields
    permanently unexportable.
    """
    outputs = []
    for source in ("acp-capture-v1", "otel", "atif", "invented"):
        event = tool_event(provenance=Provenance(source_format=source))
        trace = trace_of(event, provenance=Provenance(source_format=source))
        events, _ = ir_to_acp_capture_events(trace)
        outputs.append(events)
    assert all(out == outputs[0] for out in outputs)
    assert_schema_valid(outputs[0])


def test_the_module_never_references_provenance():
    """Static half of the rule above, so it cannot be reintroduced quietly."""
    source = Path(ir_to_acp_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"provenance", "source_format"}
    ]
    assert reads == [], [ast.unparse(node) for node in reads]


def test_an_acp_derived_trace_can_still_be_unrepresentable():
    """The other direction of the same rule.

    Origin does not buy a pass: an ACP-derived trace whose status this family
    could not map is refused exactly like any other.
    """
    trace = acp_events_to_ir(
        [
            {
                "type": "tool_call",
                "tool_call_id": "tc-1",
                "kind": "execute",
                "title": "ls",
                "status": "something_new",
                "content": [],
            }
        ]
    )
    assert trace.events[0].tool_call.status is ToolStatus.UNKNOWN
    assert blocker_fields(trace) == ["events[0].tool_call.status"]


# ---------------------------------------------------------------------------
# The value this edge will never invent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [None, ToolStatus.UNKNOWN])
def test_a_tool_call_with_no_acp_status_is_refused(status):
    """No member of the ACP enum can stand in for an unknown lifecycle state."""
    trace = trace_of(tool_event(status=status))
    blockers = acp_capture_blockers(trace)
    assert [b.field for b in blockers] == ["events[0].tool_call.status"]
    assert blockers[0].loss_class is LossClass.UNSUPPORTED
    assert blockers[0].space is PathSpace.HUB

    with pytest.raises(AcpCaptureNotRepresentable) as excinfo:
        ir_to_acp_capture_events(trace)
    assert excinfo.value.blockers == tuple(blockers)


def test_no_acp_status_string_appears_anywhere_when_the_status_is_unknown():
    """The strongest form: not "it raised", but "it never wrote one".

    A converter could raise *and* have built a record with a fabricated status
    on the way; this asserts the five vocabulary members are absent from
    everything the call produces, including the exception.
    """
    trace = trace_of(tool_event(status=None))
    with pytest.raises(AcpCaptureNotRepresentable) as excinfo:
        ir_to_acp_capture_events(trace)
    rendered = str(excinfo.value) + repr(excinfo.value.blockers)
    for member in ACP_TOOL_STATUSES:
        assert f"'{member}'" not in rendered.replace(
            str(sorted(ACP_TOOL_STATUSES)), ""
        ), member


@pytest.mark.parametrize(
    ("field", "path"),
    [("call_id", "events[0].tool_call.call_id"), ("name", "events[0].tool_call.name")],
)
def test_a_tool_call_missing_a_required_identifier_is_refused(field, path):
    """`kind` and `tool_call_id` are required and are not defaulted.

    The contract documents an empty `tool_call_id` for an id the *agent*
    omitted; this edge does not extend that to an id the *trace* never had. That
    restraint is deliberate and is a decision worth reviewing rather than an
    oversight — see the module docstring.
    """
    trace = trace_of(tool_event(**{field: None}))
    assert blocker_fields(trace) == [path]


# ---------------------------------------------------------------------------
# The kind slot: a category, not a tool name
# ---------------------------------------------------------------------------


def test_an_acp_kind_passes_through_unchanged():
    """The only semantics ACP's `kind` slot accepts, and it is preserved."""
    trace = acp_events_to_ir([CONFORMANT[2]])
    assert trace.events[0].tool_call.name_semantics == "acp_kind"
    events, _ = ir_to_acp_capture_events(trace)
    assert events[0]["kind"] == "execute"
    assert_schema_valid(events)


@pytest.mark.parametrize(
    "semantics",
    ["function_name", "gen_ai.tool.name", "span_name", "something_new"],
)
def test_a_name_that_is_not_an_acp_kind_is_refused(semantics):
    """ACP's `kind` is a category tag (`ToolKind`: "Category tag for tool calls").

    ATIF's `function_name` and OTel's `gen_ai.tool.name` name *particular tools*.
    Writing one into this slot is not a normalization a reader could undo — it
    asserts a category nobody observed. `name_semantics` exists so this edge does
    not have to guess, and refusing is what makes the field load-bearing.
    """
    trace = trace_of(tool_event(name="read_file", name_semantics=semantics))
    blockers = acp_capture_blockers(trace)
    assert [b.field for b in blockers] == ["events[0].tool_call.name_semantics"]
    assert semantics in blockers[0].detail


def test_an_unlabelled_name_is_refused_and_says_why():
    """An unlabelled name is not evidence that it came from the vocabulary.

    The detail is asserted, not just the path: "the trace does not say what kind
    of name this is" and "the name is a `function_name`" are different findings,
    and a reader fixing a producer needs to know which one they have. Found by
    mutation — collapsing the two branches left the path identical and only the
    message wrong, which the earlier version of this test could not see.
    """
    trace = trace_of(tool_event(name="execute", name_semantics=None))
    blockers = acp_capture_blockers(trace)
    assert [b.field for b in blockers] == ["events[0].tool_call.name_semantics"]
    assert "does not say what kind of name" in blockers[0].detail


def test_a_foreign_name_that_looks_like_an_acp_kind_is_still_refused():
    """The insidious case, and the reason the string is never inspected.

    `read` is a real `ToolKind` member. A `function_name` that happens to spell
    it is still a function name: matching the vocabulary by accident is not the
    same as being drawn from it, and a rule that looked at the value would
    silently admit exactly the cases most likely to be wrong.
    """
    for name in ("read", "write", "other", "bash", "search", "browser", "skill"):
        trace = trace_of(tool_event(name=name, name_semantics="function_name"))
        assert blocker_fields(trace) == ["events[0].tool_call.name_semantics"], name


def test_an_atif_derived_tool_call_cannot_launder_a_function_name():
    """The regression this rule was added for, end to end from the real edge.

    Before it, an ATIF document whose `extra.status` made the call otherwise
    representable exported `kind="read_file"` — a tool name in a category slot,
    with only a `DROPPED` record on `name_semantics` to show for it.
    """
    from benchflow.trajectories.ir_from_atif import atif_to_ir

    trace = atif_to_ir(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "read_file",
                            "arguments": {},
                            "extra": {"status": "completed", "title": "read"},
                        }
                    ],
                }
            ]
        }
    )
    call = trace.events[0].tool_call
    assert (call.name, call.name_semantics) == ("read_file", "function_name")
    assert "events[0].tool_call.name_semantics" in blocker_fields(trace)
    with pytest.raises(AcpCaptureNotRepresentable):
        ir_to_acp_capture_events(trace)


def test_an_otel_derived_tool_call_is_refused_on_semantics_too():
    """Blocked on status today; this pins that it is *also* blocked on semantics,
    so making the status representable would not open the laundering path."""
    trace = trace_of(
        tool_event(
            name="read_file",
            name_semantics="gen_ai.tool.name",
            status=ToolStatus.COMPLETED,
        )
    )
    assert blocker_fields(trace) == ["events[0].tool_call.name_semantics"]


def test_the_semantics_rule_does_not_consult_provenance():
    """`name_semantics` is trace data; `source_format` is not consulted."""
    for source in ("acp-capture-v1", "otel", "atif"):
        allowed = trace_of(
            tool_event(
                name_semantics="acp_kind",
                provenance=Provenance(source_format=source),
            ),
            provenance=Provenance(source_format=source),
        )
        refused = trace_of(
            tool_event(
                name_semantics="function_name",
                provenance=Provenance(source_format=source),
            ),
            provenance=Provenance(source_format=source),
        )
        assert acp_capture_blockers(allowed) == [], source
        assert blocker_fields(refused) == ["events[0].tool_call.name_semantics"], source


def test_a_missing_name_and_a_missing_semantics_are_reported_separately():
    """Two different absences, two different paths."""
    trace = trace_of(tool_event(name=None, name_semantics=None))
    assert sorted(blocker_fields(trace)) == [
        "events[0].tool_call.name",
        "events[0].tool_call.name_semantics",
    ]


def test_a_content_block_with_no_source_block_is_refused():
    """ACP stores wire blocks verbatim; there is no single shape to invent."""
    trace = trace_of(
        tool_event(content=[ContentBlock(kind=ContentBlockKind.TEXT, text="o")])
    )
    assert blocker_fields(trace) == ["events[0].tool_call.content[0].raw"]


def test_every_blocker_is_reported_not_just_the_first():
    """A producer being fixed wants the whole list in one pass."""
    trace = trace_of(tool_event(status=None, call_id=None, name=None))
    assert sorted(blocker_fields(trace)) == [
        "events[0].tool_call.call_id",
        "events[0].tool_call.name",
        "events[0].tool_call.status",
    ]


# ---------------------------------------------------------------------------
# The one documented empty string
# ---------------------------------------------------------------------------


def test_a_missing_title_becomes_the_empty_string_and_says_so():
    """Allowed because the contract defines it, declared because a reader cannot
    tell it from an observed empty title."""
    trace = trace_of(tool_event(title=None))
    events, report = ir_to_acp_capture_events(trace)
    assert events[0]["title"] == ""
    assert_schema_valid(events)

    records = report.for_field("events[0].tool_call.title")
    assert len(records) == 1
    assert records[0].loss_class is LossClass.SYNTHESIZED
    assert "contract" in records[0].detail


def test_an_observed_empty_title_declares_nothing():
    """The complement — otherwise the record above would be unconditional and
    would say nothing about which titles were real."""
    trace = trace_of(tool_event(title=""))
    events, report = ir_to_acp_capture_events(trace)
    assert events[0]["title"] == ""
    assert report.for_field("events[0].tool_call.title") == []


def test_a_missing_text_is_refused_rather_than_emptied():
    """`text` is not `title`.

    The schema documents the empty string as a value the capture path really
    records — an unconditionally captured empty prompt — not as a way to write
    an absence. Emitting one for `None` would collapse the tri-state §8.2 is
    built on.
    """
    event = TraceEvent(
        index=0,
        kind=EventKind.USER_MESSAGE,
        text=None,
        provenance=Provenance(source_format="hand-built"),
    )
    assert blocker_fields(trace_of(event)) == ["events[0].text"]


def test_carried_source_fields_never_reach_the_record():
    """`additionalProperties: false` on every shape, so extensions stay out.

    An IR event can carry arbitrary source keys the inbound edge preserved.
    Merging them into the record would be the easiest way to lose nothing — and
    would emit a schema-invalid record, which is a worse outcome than declaring
    the loss. Found by mutation: nothing asserted this until a mutation that
    merged `extensions` into the record left the suite green.
    """
    event = TraceEvent(
        index=0,
        kind=EventKind.USER_MESSAGE,
        text="go",
        extensions={"ts": "2026-08-18T00:00:00", "example_index": 0},
        provenance=Provenance(source_format="hand-built"),
    )
    events, report = ir_to_acp_capture_events(trace_of(event))

    assert events == [{"type": "user_message", "text": "go"}]
    assert_schema_valid(events)
    record = report.for_field("events[0].extensions")
    assert len(record) == 1
    assert record[0].loss_class is LossClass.DROPPED


def test_a_tool_call_records_extensions_are_dropped_not_merged():
    """The same property on the shape with the most required fields."""
    trace = trace_of(tool_event(extensions={"raw_input": {"cmd": "ls"}}))
    events, report = ir_to_acp_capture_events(trace)
    assert set(events[0]) == {
        "type",
        "tool_call_id",
        "kind",
        "title",
        "status",
        "content",
    }
    assert_schema_valid(events)
    assert report.for_field("events[0].extensions")


def test_a_timeout_keeps_its_three_keys_and_drops_the_rest():
    """The one shape that reads `extensions`, so the boundary is worth pinning:
    the three contract keys are consumed, anything else is declared."""
    trace = trace_of(
        timeout_event(
            extensions={
                "timeout_sec": 300,
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": True,
                "stray": "value",
            }
        )
    )
    events, report = ir_to_acp_capture_events(trace)
    assert "stray" not in events[0]
    assert_schema_valid(events)
    records = report.for_field("events[0].extensions")
    assert len(records) == 1
    assert "stray" in records[0].detail


def test_an_observed_empty_text_is_written_as_observed():
    trace = trace_of(
        TraceEvent(
            index=0,
            kind=EventKind.AGENT_MESSAGE,
            text="",
            provenance=Provenance(source_format="hand-built"),
        )
    )
    events, _ = ir_to_acp_capture_events(trace)
    assert events == [{"type": "agent_message", "text": ""}]
    assert_schema_valid(events)


# ---------------------------------------------------------------------------
# Outside the codomain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [EventKind.ORACLE, EventKind.UNKNOWN])
def test_an_event_outside_the_codomain_is_refused(kind):
    """Not dropped, not emitted invalid, and not a reason to widen the schema.

    §2.4 documents the oracle record as something the capture emitter never
    produces; `unknown` is unmodelled by definition. The schema has three record
    shapes and this edge writes only those three.
    """
    event = TraceEvent(
        index=0,
        kind=kind,
        source_type="oracle" if kind is EventKind.ORACLE else "mystery",
        extensions={"command": "solve.sh", "return_code": 0},
        provenance=Provenance(source_format="hand-built"),
    )
    blockers = acp_capture_blockers(trace_of(event))
    assert [b.field for b in blockers] == ["events[0]"]
    assert kind.value in blockers[0].detail


def test_an_oracle_trace_produces_no_records_at_all():
    """The failure is total, so nothing can leak into a caller's file."""
    trace = acp_events_to_ir(
        [{"type": "oracle", "command": "solve.sh", "return_code": 0, "stdout": "ok"}]
    )
    assert trace.events[0].kind is EventKind.ORACLE
    with pytest.raises(AcpCaptureNotRepresentable):
        ir_to_acp_capture_events(trace)


# ---------------------------------------------------------------------------
# Timeout: four required values, none defaulted
# ---------------------------------------------------------------------------


def test_a_timeout_rebuilds_from_the_keys_the_inbound_edge_preserved():
    trace = acp_events_to_ir([CONFORMANT[4]])
    events, _ = ir_to_acp_capture_events(trace)
    assert events == [CONFORMANT[4]]
    assert_schema_valid(events)


@pytest.mark.parametrize(
    ("extensions", "expected"),
    [
        (
            {},
            [
                "events[0].extensions.timeout_sec",
                "events[0].extensions.pending_tool_call_ids",
                "events[0].extensions.terminal_trajectory_complete",
            ],
        ),
        (
            {"timeout_sec": 300, "pending_tool_call_ids": []},
            ["events[0].extensions.terminal_trajectory_complete"],
        ),
        (
            {
                "timeout_sec": "300",
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": True,
            },
            ["events[0].extensions.timeout_sec"],
        ),
        (
            {
                "timeout_sec": True,
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": True,
            },
            ["events[0].extensions.timeout_sec"],
        ),
        (
            {
                "timeout_sec": 300,
                "pending_tool_call_ids": "none",
                "terminal_trajectory_complete": True,
            },
            ["events[0].extensions.pending_tool_call_ids"],
        ),
        (
            {
                "timeout_sec": 300,
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": 1,
            },
            ["events[0].extensions.terminal_trajectory_complete"],
        ),
    ],
)
def test_a_timeout_missing_or_mistyped_fields_is_refused(extensions, expected):
    """No budget is invented, and no type is coerced.

    `timeout_sec: True` is its own case: `bool` is an `int` subclass in Python
    and a timeout budget of `True` is a malformed record, not the number 1.
    """
    trace = trace_of(timeout_event(extensions=extensions))
    assert blocker_fields(trace) == expected


def test_a_timeout_with_another_reason_is_refused():
    """`reason` is a single-member enum; there is no other reason to state."""
    trace = trace_of(timeout_event(outcome="idle_timeout"))
    assert blocker_fields(trace) == ["events[0].outcome"]


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_one_unrepresentable_event_refuses_the_whole_trace():
    """No partial success.

    The target has no envelope, no count and no marker for "some events are
    missing", so a partial list is indistinguishable from a complete one the
    moment it leaves this function.
    """
    trace = trace_of(
        TraceEvent(
            index=0,
            kind=EventKind.USER_MESSAGE,
            text="go",
            provenance=Provenance(source_format="hand-built"),
        ),
        tool_event(index=1, status=None),
        TraceEvent(
            index=2,
            kind=EventKind.AGENT_MESSAGE,
            text="done",
            provenance=Provenance(source_format="hand-built"),
        ),
    )
    assert validate_trace(trace) == []
    with pytest.raises(AcpCaptureNotRepresentable) as excinfo:
        ir_to_acp_capture_events(trace)
    assert [b.field for b in excinfo.value.blockers] == ["events[1].tool_call.status"]


def test_the_inspection_helper_and_the_converter_never_disagree():
    """Both entry points share one pass, so "is it representable" and "convert
    it" cannot answer differently."""
    for trace in (
        acp_events_to_ir(CONFORMANT),
        trace_of(tool_event(status=None)),
        trace_of(timeout_event(extensions={})),
    ):
        blockers = acp_capture_blockers(trace)
        if blockers:
            with pytest.raises(AcpCaptureNotRepresentable):
                ir_to_acp_capture_events(trace)
        else:
            ir_to_acp_capture_events(trace)


def test_the_error_is_the_repositorys_existing_idiom():
    """A `ValueError` subclass defined in the exporter module, like
    `PrimeSftTrajectoryJsonlError` and the `ValueError` `ir_to_atif` raises."""
    assert issubclass(AcpCaptureNotRepresentable, ValueError)


# ---------------------------------------------------------------------------
# The loss report
# ---------------------------------------------------------------------------


def test_nothing_trace_level_is_smuggled_into_an_event():
    """The capture format is a flat stream with no envelope."""
    trace = trace_of(
        tool_event(),
        trace_id="t-1",
        session_id="s-1",
        agent=ModelInfo(agent_name="a", model="m"),
        usage=TraceUsage(input_tokens=5),
        outcome=TraceOutcome(status=None, stop_reason="end_turn"),
        extensions={"x": 1},
    )
    events, report = ir_to_acp_capture_events(trace)
    serialized = json.dumps(events)
    for leaked in ("t-1", "s-1", "end_turn"):
        assert leaked not in serialized
    declared = {r.field for r in report.records}
    assert {
        "trace_id",
        "session_id",
        "agent.agent_name",
        "agent.model",
        "usage",
        "outcome.stop_reason",
        "extensions",
    } <= declared


def test_the_outbound_edge_declares_only_what_this_trace_loses():
    """A trace carrying nothing extra declares nothing extra — the rule Slice D
    adopted, so a report stays a statement about the conversion in hand."""
    _, report = ir_to_acp_capture_events(trace_of(tool_event(arguments=None)))
    assert report.for_field("trace_id") == []
    assert report.for_field("usage") == []
    assert report.for_field("events[0].tool_call.arguments") == []


def test_every_declared_loss_path_resolves_in_the_canonical_encoding():
    """A declaration a reader of the trace cannot find is not a declaration."""
    trace = acp_events_to_ir(CONFORMANT)
    _, report = ir_to_acp_capture_events(trace)
    document = trace.model_dump(mode="json")
    unresolved = [
        record.field
        for record in report.records
        if record.space is PathSpace.HUB
        and "[]" not in record.field
        and not resolve_ir_path(document, record.field)[0]
    ]
    assert unresolved == [], unresolved


def test_no_required_acp_field_is_filled_by_an_undeclared_default():
    """Every required field is either carried or declared.

    `title` is the only one written for an absence, and it has a record. If any
    other required field ever starts being defaulted, this fails.
    """
    trace = trace_of(tool_event(title=None))
    events, report = ir_to_acp_capture_events(trace)
    declared = {r.field for r in report.records}
    record = events[0]
    for field, ir_path in (
        ("tool_call_id", "events[0].tool_call.call_id"),
        ("kind", "events[0].tool_call.name"),
        ("title", "events[0].tool_call.title"),
        ("status", "events[0].tool_call.status"),
    ):
        carried = getattr(
            trace.events[0].tool_call,
            {
                "tool_call_id": "call_id",
                "kind": "name",
                "title": "title",
                "status": "status",
            }[field],
        )
        assert record[field] is not None
        assert carried is not None or ir_path in declared, field


def test_the_report_grows_only_in_its_per_event_half():
    """Declared absence has to stay affordable, as on every other edge.

    The trace-level half is declared once per conversion and must not scale;
    the per-event half scales exactly linearly. Asserting the total would be
    asserting their sum, which hides which half moved.
    """

    def split(report):
        per_event = [r for r in report.records if r.field.startswith("events[")]
        return len(report.records) - len(per_event), len(per_event)

    one_trace, one_event = split(
        ir_to_acp_capture_events(acp_events_to_ir(CONFORMANT))[1]
    )
    ten_trace, ten_event = split(
        ir_to_acp_capture_events(acp_events_to_ir(CONFORMANT * 10))[1]
    )
    assert one_trace == ten_trace, "the trace-level half must not scale"
    assert ten_event == 10 * one_event, "the per-event half must scale linearly"


def test_no_record_is_ever_written_for_an_event_that_blocked():
    """Fail-closed at the record level, not only at the function level."""
    trace = trace_of(tool_event(index=0), tool_event(index=1, status=None))
    with pytest.raises(AcpCaptureNotRepresentable):
        ir_to_acp_capture_events(trace)
    assert len(acp_capture_blockers(trace)) == 1


# ---------------------------------------------------------------------------
# Schema conformance of everything this edge emits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trace_factory",
    [
        pytest.param(lambda: acp_events_to_ir(CONFORMANT), id="round-trip"),
        pytest.param(lambda: trace_of(tool_event(title=None)), id="synthesized-title"),
        pytest.param(
            lambda: trace_of(tool_event(title="", call_id="")), id="observed-empties"
        ),
        pytest.param(lambda: trace_of(tool_event(content=[])), id="no-content"),
        pytest.param(lambda: trace_of(timeout_event()), id="timeout"),
        pytest.param(
            lambda: trace_of(
                TraceEvent(
                    index=0,
                    kind=EventKind.AGENT_REASONING,
                    reasoning="t",
                    reasoning_segments=["t"],
                    role=Role.AGENT,
                    provenance=Provenance(source_format="hand-built"),
                )
            ),
            id="reasoning",
        ),
    ],
)
def test_every_successful_output_validates_against_the_slice_a_schema(trace_factory):
    events, _ = ir_to_acp_capture_events(trace_factory())
    assert events
    assert_schema_valid(events)


def test_the_schema_used_here_is_the_published_one():
    """Guards against the suite validating a copy that drifted."""
    assert SCHEMA["$id"].endswith("acp-capture-event-v1.schema.json")
    assert [ref["$ref"].split("/")[-1] for ref in SCHEMA["oneOf"]] == [
        "text_event",
        "tool_call_event",
        "agent_timeout_event",
    ]


def test_the_status_vocabulary_matches_the_schema():
    """The constant this edge refuses against is the schema's own enum."""
    assert (
        set(SCHEMA["$defs"]["tool_call_event"]["properties"]["status"]["enum"])
        == ACP_TOOL_STATUSES
    )
    assert (
        ACP_TIMEOUT_REASON
        in SCHEMA["$defs"]["agent_timeout_event"]["properties"]["reason"]["enum"]
    )


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_this_edge_imports_nothing_but_the_hub():
    source = Path(ir_to_acp_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert {name for name in imported if name.startswith("benchflow")} == {
        "benchflow.trajectories.ir"
    }


def test_the_edge_writes_nothing_to_disk():
    """It returns records; persisting them is a caller's decision and a wiring
    question this slice does not open."""
    source = Path(ir_to_acp_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("open(", "write_text", "Path(", "TrajectoryWriter"):
        assert forbidden not in source, forbidden
