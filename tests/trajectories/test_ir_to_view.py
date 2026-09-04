"""`IR → viewer trace steps`: the shape, the vocabulary, and the refusals.

The interesting half of this file is the second one. A converter that maps
fields is easy to test by mapping them back; what this edge actually promises
is *negative* — that it will not infer a tool category from a string, will not
manufacture tool output, will not let an event vanish, and will not confuse an
absent value with an observed empty one. Each of those has a test that fails
when the promise is removed.
"""

from __future__ import annotations

import json
import pathlib
from datetime import UTC, datetime
from typing import Any

import pytest

from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossReport,
    PathSpace,
    Provenance,
    Role,
    ToolCall,
    ToolStatus,
    TraceEvent,
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_from_atif import atif_to_ir
from benchflow.trajectories.ir_from_otel import otlp_json_to_ir
from benchflow.trajectories.ir_to_acp import ACP_KIND_SEMANTICS as ACP_EDGE_SEMANTICS
from benchflow.trajectories.ir_to_atif import ir_to_atif
from benchflow.trajectories.ir_to_view import (
    ACP_KIND_SEMANTICS,
    DIAGNOSTIC_KINDS,
    LOSS_DIRECTION,
    NEUTRAL_HUE,
    STEP_KIND,
    TRACE_LEVEL_PATHS,
    VIEW_SCHEMA_ORIGIN,
    VIEW_STEP_KINDS,
    VIEW_TOOL_HUES,
    ir_to_view_steps,
)
from tests.trajectories.test_atif_preservation import _rich_events
from tests.trajectories.test_ir_from_otel import PRODUCER_PAYLOAD_JSON
from tests.trajectories.test_trace_ir import resolve_ir_path

EVIDENCE = pathlib.Path(__file__).resolve().parents[2].parent / "e2e-a2" / "evidence"

_PROV = Provenance(source_format="hand-built")


def _trace(*events: TraceEvent) -> CanonicalTrace:
    return CanonicalTrace(provenance=_PROV, events=list(events))


def _event(index: int = 0, **kwargs: Any) -> TraceEvent:
    kwargs.setdefault("kind", EventKind.AGENT_MESSAGE)
    kwargs.setdefault("provenance", _PROV)
    return TraceEvent(index=index, **kwargs)


def _tool_event(index: int = 0, **call: Any) -> TraceEvent:
    call.setdefault("name", "execute")
    call.setdefault("name_semantics", ACP_KIND_SEMANTICS)
    return _event(index, kind=EventKind.TOOL_CALL, tool_call=ToolCall(**call))


def _records_at(report, field: str) -> list:
    return [r for r in report.records if r.field == field]


def _rollout(name: str) -> list[dict[str, Any]]:
    path = EVIDENCE / name / "acp_trajectory.jsonl"
    if not path.is_file():
        pytest.skip(f"captured rollout {name!r} is not in this tree")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# The frozen contract
# ---------------------------------------------------------------------------


def test_the_vocabularies_are_frozen():
    """Changing either is changing the wire contract, not an implementation
    detail — and the origin comment is what makes the values checkable."""
    assert VIEW_STEP_KINDS == (
        "prompt",
        "message",
        "thought",
        "tool",
        "timeout",
        "unknown",
    )
    assert VIEW_TOOL_HUES == (
        "read",
        "edit",
        "execute",
        "fetch",
        "search",
        "think",
        "skill",
        "other",
    )
    assert NEUTRAL_HUE in VIEW_TOOL_HUES
    assert VIEW_SCHEMA_ORIGIN == "benchflow-ai/benchflow#1034@79695125"


def test_every_event_kind_maps_to_a_step_kind():
    """Totality is the anti-drop guarantee at the type level.

    A new `EventKind` with no entry raises here rather than reaching the viewer
    by being quietly skipped — which is exactly how both existing normalizers
    lose unrecognized records today.
    """
    assert set(STEP_KIND) == set(EventKind)
    assert set(STEP_KIND.values()) <= set(VIEW_STEP_KINDS)


def test_the_acp_category_constant_matches_the_acp_edge():
    """Two edges, one meaning of "this name is a category".

    Pinned rather than imported so the viewer edge does not depend on the ACP
    edge, and so a rename on either side is a failing test instead of a silent
    divergence in what counts as a real category.
    """
    assert ACP_KIND_SEMANTICS == ACP_EDGE_SEMANTICS


def test_the_emitted_keys_are_frozen_per_kind():
    """The wire shape, pinned key by key."""
    trace = _trace(
        _event(0, kind=EventKind.USER_MESSAGE, source_type="user_message", text="hi"),
        _event(1, kind=EventKind.AGENT_MESSAGE, source_type="agent_message", text="yo"),
        _event(
            2,
            kind=EventKind.AGENT_REASONING,
            source_type="agent_thought",
            reasoning="hm",
        ),
        _event(
            3,
            kind=EventKind.TOOL_CALL,
            source_type="tool_call",
            tool_call=ToolCall(
                call_id="c1",
                name="execute",
                name_semantics=ACP_KIND_SEMANTICS,
                title="ls",
                status=ToolStatus.COMPLETED,
            ),
        ),
        _event(
            4,
            kind=EventKind.TIMEOUT,
            source_type="agent_timeout",
            outcome="wall_clock_timeout",
            extensions={
                "timeout_sec": 30.0,
                "pending_tool_call_ids": ["c1"],
                "terminal_trajectory_complete": False,
            },
        ),
        _event(5, kind=EventKind.UNKNOWN, source_type="future_record"),
    )
    steps, _ = ir_to_view_steps(trace)

    assert [set(step) for step in steps] == [
        {"i", "kind", "type", "text"},
        {"i", "kind", "type", "text"},
        {"i", "kind", "type", "text"},
        {"i", "kind", "type", "tool"},
        {"i", "kind", "type", "timeout"},
        {"i", "kind", "type", "text"},
    ]
    assert set(steps[3]["tool"]) == {
        "id",
        "kind",
        "title",
        "status",
        "content",
        "hue",
        "name_semantics",
    }
    assert set(steps[4]["timeout"]) == {
        "reason",
        "timeout_sec",
        "pending",
        "complete",
    }


def test_optional_keys_are_omitted_when_absent_not_nulled():
    """#1034's renderer reads key presence, so a null is not an omission.

    ``label`` is never written at all: prompt ordinals come from prompts.json,
    which is the wiring slice's input and not a property of a trace.
    """
    steps, report = ir_to_view_steps(_trace(_event(0, kind=EventKind.AGENT_MESSAGE)))
    assert set(steps[0]) == {"i", "kind"}
    assert "label" not in steps[0]
    assert _records_at(report, "steps[].label")[0].loss_class is LossClass.UNSUPPORTED


def test_step_numbering_is_dense_from_one_and_declared():
    trace = _trace(*(_event(i) for i in range(4)))
    steps, report = ir_to_view_steps(trace)
    assert [step["i"] for step in steps] == [1, 2, 3, 4]
    record = _records_at(report, "steps[].i")[0]
    assert record.loss_class is LossClass.SYNTHESIZED
    assert record.space is PathSpace.TARGET


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (EventKind.USER_MESSAGE, "prompt"),
        (EventKind.AGENT_MESSAGE, "message"),
        (EventKind.AGENT_REASONING, "thought"),
        (EventKind.TOOL_CALL, "tool"),
        (EventKind.TIMEOUT, "timeout"),
        (EventKind.ORACLE, "unknown"),
        (EventKind.UNKNOWN, "unknown"),
    ],
)
def test_each_event_kind_reaches_its_step_kind(kind, expected):
    event = (
        _tool_event(0)
        if kind is EventKind.TOOL_CALL
        else _event(0, kind=kind, text="t" if kind is EventKind.USER_MESSAGE else None)
    )
    steps, _ = ir_to_view_steps(_trace(event))
    assert steps[0]["kind"] == expected


def test_reasoning_is_read_from_reasoning_not_text():
    """The IR keeps thoughts in their own field; the viewer has one text slot."""
    trace = _trace(
        _event(0, kind=EventKind.AGENT_REASONING, reasoning="deliberating", text=None)
    )
    steps, _ = ir_to_view_steps(trace)
    assert steps[0] == {"i": 1, "kind": "thought", "text": "deliberating"}


def test_source_type_survives_on_known_kinds_too():
    """Not only on `unknown` — #1034 keeps it for unknown steps alone, and a
    normalized kind with its source string discarded is a lossy rename."""
    trace = _trace(
        _event(
            0,
            kind=EventKind.AGENT_REASONING,
            source_type="agent_thought",
            reasoning="x",
        ),
        _event(1, kind=EventKind.USER_MESSAGE, source_type="user", text="y"),
    )
    steps, _ = ir_to_view_steps(trace)
    assert [step["type"] for step in steps] == ["agent_thought", "user"]


def test_no_event_is_ever_dropped():
    """One step per event, whatever the event is.

    The corpora include kinds with no typed slot; if any branch could skip a
    record this count would drift, and the page would be missing history with
    nothing saying so.
    """
    for events in (_rollout("h1"), _rollout("h2")):
        trace = acp_events_to_ir(events)
        steps, _ = ir_to_view_steps(trace)
        assert len(steps) == len(trace.events)

    weird = _trace(
        _event(0, kind=EventKind.UNKNOWN, source_type=None),
        _event(1, kind=EventKind.ORACLE),
        _event(2, kind=EventKind.TOOL_CALL, tool_call=None),
    )
    steps, _ = ir_to_view_steps(weird)
    assert len(steps) == 3


# ---------------------------------------------------------------------------
# ORACLE and UNKNOWN
# ---------------------------------------------------------------------------


def test_an_oracle_event_keeps_its_identity_without_a_typed_slot():
    """`StepKind` has no oracle member, so the type slot carries the fact."""
    trace = acp_events_to_ir(
        [
            {
                "type": "oracle",
                "command": "python solve.py",
                "return_code": 0,
                "stdout": "ok\n",
            }
        ]
    )
    steps, _ = ir_to_view_steps(trace)
    assert steps[0]["kind"] == "unknown"
    assert steps[0]["type"] == "oracle"

    body = json.loads(steps[0]["text"])
    assert body["extensions"] == {
        "command": "python solve.py",
        "return_code": 0,
        "stdout": "ok\n",
    }
    assert body["kind"] == "oracle"


def test_an_oracle_with_no_source_string_still_says_it_is_an_oracle():
    """Reshaping the observed kind, not inventing a source record."""
    steps, report = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.ORACLE, source_type=None))
    )
    assert steps[0]["type"] == "oracle"
    assert _records_at(report, "steps[].type")[0].loss_class is LossClass.NORMALIZED


def test_the_diagnostic_text_is_the_canonical_event_not_a_source_record():
    """The distinction the loss record is required to make.

    #1034's unknown branch serializes the raw ACP dict it read off disk. This
    edge has no such document — only the IR event built from one — so what it
    renders carries IR field names, and the report says so rather than letting
    a page present it as the producer's own payload.
    """
    trace = acp_events_to_ir([{"type": "mystery", "a": 1}])
    steps, report = ir_to_view_steps(trace)

    body = json.loads(steps[0]["text"])
    assert body["provenance"]["source_format"] == "acp-trajectory-unknown"
    assert body["extensions"] == {"type": "mystery", "a": 1}
    assert set(body) >= {"index", "kind", "source_type", "provenance", "extensions"}

    record = _records_at(report, "steps[].text")[0]
    assert record.loss_class is LossClass.SYNTHESIZED
    assert "canonical IR event" in record.detail
    assert "not a raw source record" in record.detail


def test_an_unknown_without_a_source_type_omits_the_key():
    steps, _ = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.UNKNOWN, source_type=None))
    )
    assert "type" not in steps[0]
    assert steps[0]["kind"] == "unknown"


# ---------------------------------------------------------------------------
# Timeout stays typed
# ---------------------------------------------------------------------------


def test_a_timeout_is_typed_and_not_a_diagnostic_blob():
    trace = acp_events_to_ir(
        [
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 30.0,
                "pending_tool_call_ids": ["c1"],
                "terminal_trajectory_complete": False,
            }
        ]
    )
    steps, report = ir_to_view_steps(trace)
    assert steps[0]["kind"] == "timeout"
    assert "text" not in steps[0]
    assert steps[0]["timeout"] == {
        "reason": "wall_clock_timeout",
        "timeout_sec": 30.0,
        "pending": ["c1"],
        "complete": False,
    }
    assert not [r for r in report.records if r.field.startswith("steps[].text")]


def test_a_timeout_missing_everything_declares_each_sentinel():
    """Two of the four slots take null and represent their own absence; the
    other two do not, and each substitute is declared on its own."""
    trace = _trace(_event(0, kind=EventKind.TIMEOUT, outcome=None, extensions={}))
    steps, report = ir_to_view_steps(trace)

    assert steps[0]["timeout"] == {
        "reason": "",
        "timeout_sec": None,
        "pending": [],
        "complete": None,
    }
    assert _records_at(report, "events[0].outcome")[0].loss_class is (
        LossClass.SYNTHESIZED
    )
    pending = _records_at(report, "events[0].extensions")[0]
    assert pending.loss_class is LossClass.SYNTHESIZED
    assert "not an observation that none were pending" in pending.detail


# ---------------------------------------------------------------------------
# Hue: membership, never inference
# ---------------------------------------------------------------------------


def test_a_real_acp_category_becomes_its_hue():
    steps, report = ir_to_view_steps(
        _trace(_tool_event(0, name="read", name_semantics=ACP_KIND_SEMANTICS))
    )
    assert steps[0]["tool"]["hue"] == "read"
    assert _records_at(report, "steps[].tool.hue")[0].loss_class is (
        LossClass.NORMALIZED
    )


@pytest.mark.parametrize(
    ("name", "semantics"),
    [
        ("read", "function_name"),
        ("read_file", "gen_ai.tool.name"),
        ("read", "span_name"),
        ("read", None),
    ],
)
def test_a_name_never_becomes_a_category(name, semantics):
    """The laundering this edge exists to refuse.

    ``function_name="read"`` spells a real category exactly, and
    ``gen_ai.tool.name="read_file"`` contains one; #1034's ``tool_hue`` returns
    ``read`` for both. Neither is an observation that the tool *is* a read.
    """
    steps, report = ir_to_view_steps(
        _trace(_tool_event(0, name=name, name_semantics=semantics))
    )
    assert steps[0]["tool"]["hue"] == NEUTRAL_HUE
    assert steps[0]["tool"]["kind"] == name
    assert steps[0]["tool"]["name_semantics"] == semantics

    record = _records_at(report, "events[0].tool_call.name_semantics")[0]
    assert record.loss_class is LossClass.SYNTHESIZED


def test_the_title_is_never_consulted_for_a_hue():
    """`tool_hue` reads ``kind + " " + title``; this edge reads neither string
    for meaning. A title full of category words changes nothing."""
    steps, _ = ir_to_view_steps(
        _trace(
            _tool_event(
                0,
                name="mystery_tool",
                name_semantics="function_name",
                title="bash grep read write search fetch",
            )
        )
    )
    assert steps[0]["tool"]["hue"] == NEUTRAL_HUE


def test_a_category_outside_the_display_vocabulary_is_neutral():
    """`delete` and `move` are real ACP kinds with no hue; membership is tested
    directly rather than approximated to the nearest colour."""
    steps, report = ir_to_view_steps(
        _trace(_tool_event(0, name="delete", name_semantics=ACP_KIND_SEMANTICS))
    )
    assert steps[0]["tool"]["hue"] == NEUTRAL_HUE
    assert (
        "outside the viewer's display vocabulary"
        in _records_at(report, "events[0].tool_call.name_semantics")[0].detail
    )


def test_provenance_is_not_evidence_about_semantics():
    """Where a trace came from does not decide what its fields mean.

    An ACP-provenanced trace whose call is labelled a function name is still a
    function name; the edge reads ``name_semantics`` and nothing else.
    """
    event = _tool_event(0, name="read", name_semantics="function_name")
    event = event.model_copy(
        update={"provenance": Provenance(source_format="acp-capture-v1")}
    )
    trace = CanonicalTrace(
        provenance=Provenance(source_format="acp-capture-v1"), events=[event]
    )
    steps, _ = ir_to_view_steps(trace)
    assert steps[0]["tool"]["hue"] == NEUTRAL_HUE


def test_name_semantics_survives_the_boundary():
    """Our additive seventh key. Without it the viewer cannot tell the three
    corpora apart, which is what makes substring inference look reasonable."""
    for semantics in (ACP_KIND_SEMANTICS, "function_name", "gen_ai.tool.name", None):
        steps, _ = ir_to_view_steps(
            _trace(_tool_event(0, name="x", name_semantics=semantics))
        )
        assert steps[0]["tool"]["name_semantics"] == semantics


# ---------------------------------------------------------------------------
# Absent is not observed-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "wire", "path"),
    [
        ("call_id", "id", "events[0].tool_call.call_id"),
        ("name", "kind", "events[0].tool_call.name"),
        ("title", "title", "events[0].tool_call.title"),
    ],
)
def test_an_absent_string_is_declared_and_an_observed_empty_one_is_not(
    field, wire, path
):
    """Both render as ``""``; only one of them is a substitution.

    The report is the only place the difference survives, so a shared branch
    that treated ``None`` and ``""`` alike would erase a real distinction the
    IR went to some trouble to keep.
    """
    absent, _ = ir_to_view_steps(_trace(_tool_event(0, **{field: None})))
    observed, observed_report = ir_to_view_steps(_trace(_tool_event(0, **{field: ""})))

    assert absent[0]["tool"][wire] == ""
    assert observed[0]["tool"][wire] == ""
    assert _records_at(
        ir_to_view_steps(_trace(_tool_event(0, **{field: None})))[1], path
    )
    assert not _records_at(observed_report, path)


def test_an_absent_status_is_declared():
    steps, report = ir_to_view_steps(_trace(_tool_event(0, status=None)))
    assert steps[0]["tool"]["status"] == ""
    record = _records_at(report, "events[0].tool_call.status")[0]
    assert record.loss_class is LossClass.SYNTHESIZED


def test_a_status_enum_becomes_its_string():
    steps, report = ir_to_view_steps(
        _trace(_tool_event(0, status=ToolStatus.IN_PROGRESS))
    )
    assert steps[0]["tool"]["status"] == "in_progress"
    assert _records_at(report, "steps[].tool.status")[0].loss_class is (
        LossClass.NORMALIZED
    )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def test_a_block_with_no_text_is_declared_and_not_invented():
    """The viewer holds tool output as strings. A block that has none
    contributes none — serializing its ``raw`` would put a string on the page
    that no tool ever emitted."""
    call = ToolCall(
        name="execute",
        name_semantics=ACP_KIND_SEMANTICS,
        content=[
            ContentBlock(kind=ContentBlockKind.TEXT, text="stdout"),
            ContentBlock(
                kind=ContentBlockKind.OPAQUE, raw={"type": "image", "data": "…"}
            ),
        ],
    )
    steps, report = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.TOOL_CALL, tool_call=call))
    )
    assert steps[0]["tool"]["content"] == ["stdout"]
    record = _records_at(report, "events[0].tool_call.content[1]")[0]
    assert record.loss_class is LossClass.DROPPED
    assert "does not invent one" in record.detail


def test_an_observed_empty_block_is_kept():
    """#1034 filters falsy strings out; an empty observation is still one."""
    call = ToolCall(
        name="execute",
        name_semantics=ACP_KIND_SEMANTICS,
        content=[ContentBlock(kind=ContentBlockKind.TEXT, text="")],
    )
    steps, _ = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.TOOL_CALL, tool_call=call))
    )
    assert steps[0]["tool"]["content"] == [""]


def test_arguments_are_declared_dropped_when_observed():
    """The OTel corpus is the only one that has real arguments, and the shape
    has nowhere to put them."""
    steps, report = ir_to_view_steps(
        _trace(_tool_event(0, arguments={"path": "/repo/README.md"}))
    )
    assert "arguments" not in steps[0]["tool"]
    assert _records_at(report, "events[0].tool_call.arguments")[0].loss_class is (
        LossClass.DROPPED
    )


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_timestamps_become_epoch_seconds_when_observed():
    start = datetime(2025, 8, 18, 6, 53, 20, tzinfo=UTC)
    end = datetime(2025, 8, 18, 6, 53, 24, tzinfo=UTC)
    trace = _trace(_event(0, started_at=start, finished_at=end))
    steps, report = ir_to_view_steps(trace)
    assert steps[0]["t"] == start.timestamp()
    assert steps[0]["dur"] == 4.0
    assert _records_at(report, "steps[].t")[0].loss_class is LossClass.NORMALIZED


def test_a_finish_before_a_start_is_not_a_duration():
    start = datetime(2025, 8, 18, 6, 53, 24, tzinfo=UTC)
    end = datetime(2025, 8, 18, 6, 53, 20, tzinfo=UTC)
    steps, _ = ir_to_view_steps(_trace(_event(0, started_at=start, finished_at=end)))
    assert "dur" not in steps[0]


def test_a_tool_step_prefers_the_calls_own_window():
    outer = datetime(2025, 8, 18, 6, 0, 0, tzinfo=UTC)
    inner = datetime(2025, 8, 18, 6, 0, 5, tzinfo=UTC)
    event = _event(
        0,
        kind=EventKind.TOOL_CALL,
        started_at=outer,
        tool_call=ToolCall(
            name="execute",
            name_semantics=ACP_KIND_SEMANTICS,
            started_at=inner,
            finished_at=inner,
        ),
    )
    steps, _ = ir_to_view_steps(_trace(event))
    assert steps[0]["t"] == inner.timestamp()


def test_no_timestamps_means_no_keys_and_no_declaration():
    steps, report = ir_to_view_steps(_trace(_event(0)))
    assert "t" not in steps[0] and "dur" not in steps[0]
    assert not _records_at(report, "steps[].t")


# ---------------------------------------------------------------------------
# Run metadata is not step metadata
# ---------------------------------------------------------------------------


def test_trace_level_fields_are_unsupported_here_not_dropped():
    """They are not losses of this edge. Saying `DROPPED` would claim the
    viewer cannot show a model name, which is false — `meta` shows it, built
    from artifacts a trace does not contain."""
    trace = CanonicalTrace(
        provenance=_PROV,
        session_id="s1",
        usage=TraceUsage(input_tokens=10),
        events=[_event(0, text="x")],
    )
    steps, report = ir_to_view_steps(trace)

    flat = json.dumps(steps)
    assert "s1" not in flat
    for path in TRACE_LEVEL_PATHS:
        record = _records_at(report, path)[0]
        assert record.loss_class is LossClass.UNSUPPORTED
        assert record.space is PathSpace.HUB


def test_per_event_usage_is_a_real_loss_and_says_so():
    trace = _trace(_event(0, usage=TraceUsage(input_tokens=1204), text="x"))
    _, report = ir_to_view_steps(trace)
    assert _records_at(report, "events[].usage")[0].loss_class is LossClass.DROPPED


def test_role_and_reasoning_segments_are_declared_when_present():
    trace = _trace(
        _event(0, role=Role.AGENT, text="x"),
        _event(
            1,
            kind=EventKind.AGENT_REASONING,
            reasoning="a\n\nb",
            reasoning_segments=["a", "b"],
        ),
    )
    steps, report = ir_to_view_steps(trace)
    assert steps[1]["text"] == "a\n\nb"
    assert len(steps) == 2, "segments must not silently become extra steps"
    assert _records_at(report, "events[].role")[0].loss_class is LossClass.DROPPED
    segments = _records_at(report, "events[].reasoning_segments")[0]
    assert segments.loss_class is LossClass.DROPPED
    assert "same text twice" in segments.detail


# ---------------------------------------------------------------------------
# Report hygiene
# ---------------------------------------------------------------------------


def test_every_hub_record_resolves_in_the_trace():
    for events in (_rollout("h1"), _rollout("h2")):
        trace = acp_events_to_ir(events)
        _, report = ir_to_view_steps(trace)
        canonical = trace.model_dump(mode="json")
        for record in report.records:
            if record.space is not PathSpace.HUB:
                continue
            if record.field.startswith("events[]"):
                continue
            assert resolve_ir_path(canonical, record.field)[0], record.field


def test_target_records_are_not_readable_as_ir_paths():
    trace = acp_events_to_ir(_rich_events())
    _, report = ir_to_view_steps(trace)
    canonical = trace.model_dump(mode="json")
    for record in report.records:
        if record.space is PathSpace.TARGET:
            assert not resolve_ir_path(canonical, record.field)[0], record.field


def test_the_conversion_leaves_the_input_alone():
    trace = acp_events_to_ir(_rich_events())
    before = trace.model_dump_json()
    inbound = len(trace.losses.records) if trace.losses else 0

    _, report = ir_to_view_steps(trace)

    assert trace.model_dump_json() == before
    assert (len(trace.losses.records) if trace.losses else 0) == inbound
    assert report is not trace.losses
    assert report.direction == LOSS_DIRECTION
    assert validate_trace(trace) == []


# ---------------------------------------------------------------------------
# The three corpora
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h1", "h2"])
def test_a_captured_acp_rollout_converts(name):
    trace = acp_events_to_ir(_rollout(name))
    steps, report = ir_to_view_steps(trace)

    assert len(steps) == len(trace.events)
    assert {step["kind"] for step in steps} <= set(VIEW_STEP_KINDS)
    for step in steps:
        if step["kind"] != "tool":
            continue
        # Every captured ACP call is a category, and every category these
        # rollouts use is in the display vocabulary.
        assert step["tool"]["name_semantics"] == ACP_KIND_SEMANTICS
    assert report.records


def test_the_atif_fixture_converts_without_gaining_a_category():
    """ATIF puts an ACP kind in a `function_name` slot (§8.3). The value here
    is literally ``execute`` — a hue member — and it still must not become one.
    """
    document, _ = ir_to_atif(acp_events_to_ir(_rich_events()))
    trace = atif_to_ir(document)
    steps, _ = ir_to_view_steps(trace)

    tools = [step["tool"] for step in steps if step["kind"] == "tool"]
    assert tools, "the fixture must contain a tool call"
    for tool in tools:
        assert tool["name_semantics"] == "function_name"
        assert tool["hue"] == NEUTRAL_HUE
    assert len(steps) == len(trace.events)


def test_the_otel_fixture_converts_and_keeps_what_it_has():
    traces, _ = otlp_json_to_ir(json.loads(PRODUCER_PAYLOAD_JSON))
    assert traces
    for trace in traces:
        steps, _ = ir_to_view_steps(trace)
        assert len(steps) == len(trace.events)

        for step, event in zip(steps, trace.events, strict=True):
            if event.source_type is not None:
                assert step["type"] == event.source_type

        for step in steps:
            if step["kind"] == "tool":
                assert step["tool"]["name_semantics"] == "gen_ai.tool.name"
                assert step["tool"]["hue"] == NEUTRAL_HUE

        # OTel is the only corpus with timestamps, and the only one whose
        # unknown events carry the whole span — parentage included.
        assert any("t" in step for step in steps)
        unknown = [
            step
            for step, event in zip(steps, trace.events, strict=True)
            if event.kind in DIAGNOSTIC_KINDS
        ]
        for step in unknown:
            body = json.loads(step["text"])
            assert body["extensions"]["otel"]["span"]["name"] == step["type"]


# ---------------------------------------------------------------------------
# Consume or declare — every observed string on an event
# ---------------------------------------------------------------------------


def _event_string_fields() -> list[str]:
    """The `TraceEvent` fields that hold observed text, read off the model.

    Derived rather than listed so a new string field on the IR fails the guard
    below until somebody gives it a disposition — a slot on the step, or a
    record saying why it has none.
    """
    out = []
    for name, field in TraceEvent.model_fields.items():
        annotation = str(field.annotation)
        if annotation == "str | None" or "list[str]" in annotation:
            out.append(name)
    return out


def _declared_for(report, index: int, field: str) -> bool:
    """Whether a record addresses *field* on event *index*, per-event or systemic."""
    return any(
        record.field in (f"events[{index}].{field}", f"events[].{field}")
        for record in report.records
    )


def _unconsumed(steps, report, index: int, markers: dict[str, str]) -> list[str]:
    """The fields whose observed value reached neither the step nor a record.

    The guard's whole content, as a function, so it can be shown to bite: a
    predicate only asserted in the positive direction is a predicate nobody has
    tested.
    """
    emitted = json.dumps(steps[index], ensure_ascii=False)
    return sorted(
        name
        for name, marker in markers.items()
        if marker not in emitted and not _declared_for(report, index, name)
    )


def test_the_guard_predicate_notices_a_field_that_reaches_neither():
    """Negative control for :func:`_unconsumed`.

    A step and a report that mention nothing must make every field violate; a
    step that carries the value, and a report that names it, must not.
    """
    markers = {"text": "MARKER-TEXT", "reasoning": "MARKER-REASONING"}
    empty = LossReport(direction="probe")

    assert _unconsumed([{"i": 1, "kind": "tool"}], empty, 0, markers) == [
        "reasoning",
        "text",
    ]

    on_the_wire = [{"i": 1, "kind": "tool", "text": "MARKER-TEXT"}]
    assert _unconsumed(on_the_wire, empty, 0, markers) == ["reasoning"]

    declared = LossReport(direction="probe")
    declared.add("events[0].reasoning", LossClass.DROPPED, "x")
    assert _unconsumed(on_the_wire, declared, 0, markers) == []

    systemic = LossReport(direction="probe")
    systemic.add("events[].reasoning", LossClass.DROPPED, "x")
    assert _unconsumed(on_the_wire, systemic, 0, markers) == []


def test_the_model_gives_the_expected_string_fields():
    """A stale derivation would make the guard below quietly weaker."""
    assert set(_event_string_fields()) == {
        "source_type",
        "text",
        "reasoning",
        "reasoning_segments",
        "outcome",
    }


@pytest.mark.parametrize("kind", list(EventKind))
def test_every_observed_string_is_consumed_or_declared(kind):
    """The contract this edge broke once, as an executable property.

    An ATIF document folds a thought into the agent step it precedes, so a
    faithful reading of one produces a `TOOL_CALL` event carrying `reasoning`.
    That value reached no step key and no loss record: information observed in
    the hub left the conversion with nothing said about it. The guard is
    written over *every* kind and *every* string field, not over that one case,
    because the hole was in the shape of the code and not in the field.
    """
    markers = {name: f"MARKER-{name.upper()}" for name in _event_string_fields()}
    event = _event(
        0,
        kind=kind,
        source_type=markers["source_type"],
        text=markers["text"],
        reasoning=markers["reasoning"],
        reasoning_segments=[markers["reasoning_segments"]],
        outcome=markers["outcome"],
        extensions={"timeout_sec": 1.0, "pending_tool_call_ids": []},
        tool_call=ToolCall(
            call_id="c1",
            name="execute",
            name_semantics=ACP_KIND_SEMANTICS,
            title="ls",
            status=ToolStatus.COMPLETED,
        )
        if kind is EventKind.TOOL_CALL
        else None,
    )
    steps, report = ir_to_view_steps(_trace(event))

    assert _unconsumed(steps, report, 0, markers) == [], (
        f"{kind.value}: observed values that reached no step key and no record"
    )


def test_reasoning_beside_an_action_keeps_its_own_key():
    """The ATIF shape: a tool call whose step also carries the thought."""
    event = _tool_event(0, name="execute", name_semantics=ACP_KIND_SEMANTICS)
    event = event.model_copy(update={"reasoning": "why I am doing this"})
    steps, report = ir_to_view_steps(_trace(event))

    assert steps[0]["reasoning"] == "why I am doing this"
    assert steps[0]["kind"] == "tool", "no second step was invented"
    assert len(steps) == 1
    assert "why I am doing this" not in str(steps[0].get("text"))

    record = _records_at(report, "steps[].reasoning")
    assert len(record) == 1
    assert record[0].loss_class is LossClass.NORMALIZED
    assert record[0].space is PathSpace.TARGET


@pytest.mark.parametrize(
    "kind", [EventKind.USER_MESSAGE, EventKind.AGENT_MESSAGE, EventKind.TIMEOUT]
)
def test_reasoning_survives_on_every_kind_that_is_not_a_thought(kind):
    steps, _ = ir_to_view_steps(
        _trace(_event(0, kind=kind, text="visible", reasoning="internal"))
    )
    assert steps[0]["reasoning"] == "internal"
    if kind is not EventKind.TIMEOUT:
        assert steps[0]["text"] == "visible"


def test_text_and_reasoning_land_in_different_slots_and_are_never_joined():
    steps, _ = ir_to_view_steps(
        _trace(
            _event(
                0,
                kind=EventKind.TOOL_CALL,
                text="observed text",
                reasoning="internal",
                tool_call=ToolCall(name="execute", name_semantics=ACP_KIND_SEMANTICS),
            )
        )
    )
    assert steps[0]["text"] == "observed text"
    assert steps[0]["reasoning"] == "internal"
    assert steps[0]["text"] != steps[0]["reasoning"]


def test_a_reasoning_event_keeps_the_shape_it_already_had():
    """The verified contract for a real thought event does not move."""
    steps, report = ir_to_view_steps(
        _trace(
            _event(
                0,
                kind=EventKind.AGENT_REASONING,
                source_type="agent_thought",
                reasoning="hm",
            )
        )
    )
    assert steps[0] == {
        "i": 1,
        "kind": "thought",
        "type": "agent_thought",
        "text": "hm",
    }
    assert "reasoning" not in steps[0]
    assert _records_at(report, "steps[].reasoning") == []


def test_a_thought_that_also_carries_text_declares_the_text():
    """One text slot, already holding the reasoning — so say what happened."""
    steps, report = ir_to_view_steps(
        _trace(
            _event(
                0, kind=EventKind.AGENT_REASONING, reasoning="internal", text="visible"
            )
        )
    )
    assert steps[0]["text"] == "internal"
    record = _records_at(report, "events[0].text")
    assert len(record) == 1
    assert record[0].loss_class is LossClass.DROPPED
    assert "would present them as one utterance" in record[0].detail


def test_a_thought_whose_text_repeats_the_reasoning_declares_nothing():
    """The value is on the page; a record would claim a loss that did not happen."""
    steps, report = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.AGENT_REASONING, reasoning="same", text="same"))
    )
    assert steps[0]["text"] == "same"
    assert _records_at(report, "events[0].text") == []


def test_a_diagnostic_step_does_not_repeat_its_reasoning_in_a_key():
    """The whole canonical event is already the body of that card."""
    steps, _ = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.UNKNOWN, source_type="x", reasoning="internal"))
    )
    assert "reasoning" not in steps[0]
    assert "internal" in steps[0]["text"]


def test_reasoning_segments_keep_the_policy_they_already_had():
    steps, report = ir_to_view_steps(
        _trace(
            _event(
                0,
                kind=EventKind.AGENT_REASONING,
                reasoning="a b",
                reasoning_segments=["a", "b"],
            )
        )
    )
    assert "reasoning_segments" not in steps[0]
    record = _records_at(report, "events[].reasoning_segments")
    assert len(record) == 1
    assert record[0].loss_class is LossClass.DROPPED


def test_a_terminal_signal_outside_a_timeout_is_declared():
    steps, report = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.AGENT_MESSAGE, text="hi", outcome="stopped"))
    )
    assert "stopped" not in json.dumps(steps[0])
    record = _records_at(report, "events[0].outcome")
    assert len(record) == 1
    assert record[0].loss_class is LossClass.DROPPED


def test_a_timeout_consumes_its_own_outcome_without_a_record():
    steps, report = ir_to_view_steps(
        _trace(_event(0, kind=EventKind.TIMEOUT, outcome="wall_clock_timeout"))
    )
    assert steps[0]["timeout"]["reason"] == "wall_clock_timeout"
    assert _records_at(report, "events[0].outcome") == []


def test_the_atif_shape_of_a_captured_rollout_keeps_its_thought():
    """H1, exported to ATIF and read back: the regression this guard exists for.

    `export_atif` writes an `agent_thought` as `reasoning_content` on the agent
    step it precedes, so the thought arrives on a TOOL_CALL event rather than
    on one of its own. It must still reach the page.
    """
    events = _rollout("h1")
    thought = next(e["text"] for e in events if e["type"] == "agent_thought")
    document, _ = ir_to_atif(acp_events_to_ir(events))
    trace = atif_to_ir(document)

    carrier = next(e for e in trace.events if e.reasoning is not None)
    assert carrier.kind is EventKind.TOOL_CALL
    assert carrier.reasoning == thought

    steps, report = ir_to_view_steps(trace)
    assert len(steps) == len(trace.events)
    assert [s for s in steps if s.get("reasoning") == thought], (
        "the thought reached no step"
    )
    assert (
        _records_at(report, "steps[].reasoning")[0].loss_class is LossClass.NORMALIZED
    )
