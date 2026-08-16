"""Round-trip measurement suite — `ACP → IR → ATIF → IR′`.

Slice D proved the hub reproduces the direct exporter's document. This suite is
about the question that raises: **how much of the trace is still there
afterwards?**

Two properties are worth stating up front, because they are what keeps the
measurement from being a self-fulfilling one:

- The comparison reads **the two traces**, never the loss reports. A converter
  that lost something without declaring it shows up here anyway.
- The measurement is by canonical *path*, not by event position. After a
  conversion that fuses and drops events there is no recoverable correspondence
  between event *i* and event *j*, and inventing an alignment would be the same
  guessing this whole slice refuses to do.

The suite also pins the two findings the loop produced on real rollouts, so that
a change in either converter has to confront them: `arguments` comes back
fabricated, and a timeout does not come back at all.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from benchflow.trajectories.ir import (
    CanonicalTrace,
    EventKind,
    Provenance,
    ToolCall,
    TraceEvent,
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_round_trip import (
    ATIF_REPRESENTABILITY,
    DEFAULT_EMPTY_PATHS,
    EXCLUDED_PATHS,
    OPAQUE_PATHS,
    Representability,
    RoundTripOutcome,
    compare_traces,
    declared_entry_for,
    representability_of,
    round_trip_through_atif,
)
from tests.trajectories.test_atif_preservation import _rich_events

PROMPTS = ["Solve the task.", "Then stop."]


def _trip(**kwargs: Any):
    return round_trip_through_atif(
        _rich_events(),
        session_id="sess-f",
        agent_name="claude-code",
        model="claude-sonnet-5",
        **kwargs,
    )


def _outcome(report, path: str) -> RoundTripOutcome:
    for comparison in report.comparisons:
        if comparison.path == path:
            return comparison.outcome
    raise AssertionError(
        f"{path} not in the report: {[c.path for c in report.comparisons]}"
    )


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


def test_the_loop_produces_a_valid_trace_at_both_ends():
    trip = _trip(prompts=PROMPTS)
    assert validate_trace(trip.before) == []
    assert validate_trace(trip.after) == []
    assert trip.outbound.direction == "ir->atif"
    assert trip.inbound.direction == "atif->ir"


def test_comparing_a_trace_with_itself_finds_nothing():
    """The measurement's zero point.

    Without this, a comparator that reported everything as preserved — or as
    lost — would still pass every other test in this file.
    """
    trace = acp_events_to_ir(_rich_events(), session_id="s", agent_name="a")
    report = compare_traces(trace, trace)
    assert report.summary() == {
        "preserved": len(report.comparisons),
        "transformed": 0,
        "lost": 0,
        "non_representable": 0,
        "fabricated": 0,
    }
    values = report.value_summary()
    assert values["values_preserved"] == values["values_before"]


def test_a_removed_field_is_reported_lost_and_a_new_one_fabricated():
    """The comparator's two directions, on a difference built by hand."""
    before = CanonicalTrace(
        trace_id="t-1",
        provenance=Provenance(source_format="test"),
    )
    after = CanonicalTrace(
        session_id="s-1",
        provenance=Provenance(source_format="test"),
    )
    report = compare_traces(before, after)
    assert _outcome(report, "trace_id") is RoundTripOutcome.LOST
    assert _outcome(report, "session_id") is RoundTripOutcome.FABRICATED


def test_a_changed_value_is_transformed_not_lost():
    before = CanonicalTrace(session_id="a", provenance=Provenance(source_format="test"))
    after = CanonicalTrace(session_id="b", provenance=Provenance(source_format="test"))
    report = compare_traces(before, after)
    comparison = next(c for c in report.comparisons if c.path == "session_id")
    assert comparison.outcome is RoundTripOutcome.TRANSFORMED
    assert comparison.matched_count == 0
    assert (comparison.before_count, comparison.after_count) == (1, 1)


def test_values_are_matched_as_a_multiset_not_by_position():
    """Two events swapping places is not a loss.

    Positional comparison would report every field of both as transformed,
    which would drown the real findings in noise the moment an event is
    dropped ahead of another.
    """

    def _trace(texts: list[str]) -> CanonicalTrace:
        return CanonicalTrace(
            provenance=Provenance(source_format="test"),
            events=[
                TraceEvent(
                    index=index,
                    kind=EventKind.USER_MESSAGE,
                    text=text,
                    provenance=Provenance(source_format="test"),
                )
                for index, text in enumerate(texts)
            ],
        )

    report = compare_traces(_trace(["a", "b"]), _trace(["b", "a"]))
    assert _outcome(report, "events[].text") is RoundTripOutcome.PRESERVED


# ---------------------------------------------------------------------------
# The two findings, on events from the production capture path
# ---------------------------------------------------------------------------


def test_tool_arguments_come_back_fabricated():
    """The finding this measurement exists to surface.

    `None` — "the source never carried arguments" — leaves as `{}` because ATIF
    requires the field, and returns as `{}` — "observed with an empty argument
    map". Two different facts, one document, and no way to tell them apart
    without the reports.
    """
    trip = _trip(prompts=PROMPTS)
    comparison = next(
        c for c in trip.report.comparisons if c.path == "events[].tool_call.arguments"
    )
    assert comparison.outcome is RoundTripOutcome.FABRICATED
    assert comparison.before_count == 0
    assert comparison.after_count > 0
    assert comparison.sample_after == {}

    # The outbound edge did say so — and the inbound one cannot, which is the
    # asymmetry that makes the report worth carrying.
    assert [
        record
        for record in trip.outbound.records
        if record.field.endswith("tool_call.arguments")
    ]
    assert not [
        record
        for record in trip.inbound.records
        if record.field.endswith("tool_call.arguments")
    ]


def test_a_timeout_does_not_survive_the_loop():
    """§5 loss #4, measured end to end rather than asserted.

    The IR made the timeout representable; ATIF has no slot for it, so the
    event, its reason and the run-level status all go, and the trace that comes
    back describes a run that simply ended.
    """
    trip = _trip(prompts=PROMPTS)
    assert trip.report.kinds_before["timeout"] >= 1
    assert "timeout" not in trip.report.kinds_after
    assert _outcome(trip.report, "events[].outcome") is RoundTripOutcome.LOST
    assert _outcome(trip.report, "outcome.status") is RoundTripOutcome.LOST
    for path in ("events[].outcome", "outcome.status"):
        assert representability_of(path) is Representability.NOT_IN_ATIF


def test_reasoning_survives_but_its_boundaries_do_not_have_to():
    """The IR keeps thought boundaries; ATIF joins them with a blank line.

    A single thought round-trips intact, which is why this is stated as a
    boundary property rather than a loss: the join only becomes irreversible
    once there are two thoughts to join.
    """
    events = [
        {"type": "agent_thought", "text": "first"},
        {"type": "agent_thought", "text": "second"},
        {"type": "agent_message", "text": "done"},
    ]
    trip = round_trip_through_atif(events, session_id="s", agent_name="a")
    segments_before = [
        event.reasoning_segments
        for event in trip.before.events
        if event.reasoning_segments
    ]
    segments_after = [
        event.reasoning_segments
        for event in trip.after.events
        if event.reasoning_segments
    ]
    assert segments_before == [["first"], ["second"]]
    assert segments_after == [["first\n\nsecond"]]


def test_the_prompts_argument_returns_as_captured_user_messages():
    """The same laundering as `arguments`, one level up.

    *prompts* are not trace data. They become `user` steps declared SYNTHESIZED
    in the target space, and they read back as ordinary user messages — so the
    loop adds events that no capture ever produced.
    """
    with_prompts = _trip(prompts=PROMPTS)
    without = _trip()
    assert with_prompts.report.events_after - without.report.events_after == len(
        PROMPTS
    )
    assert with_prompts.report.kinds_after["user_message"] == (
        without.report.kinds_after["user_message"] + len(PROMPTS)
    )


def test_nothing_representable_is_lost_on_a_captured_rollout():
    """The result that makes the table worth having.

    Everything the loop drops from these events is dropped because ATIF has no
    slot for it. Nothing representable goes missing — so there is no gap in our
    own edges to close, and every remaining loss is a property of the format.

    If this ever fails, the failure names a converter bug rather than a fact
    about ATIF.
    """
    for trip in (_trip(prompts=PROMPTS), _trip()):
        fixable = [
            comparison
            for comparison in trip.report.comparisons
            if comparison.outcome is RoundTripOutcome.LOST
            and comparison.representability is Representability.REPRESENTABLE
        ]
        assert fixable == [], [c.path for c in fixable]
        assert trip.report.summary()["lost"] == 0


def test_usage_only_enters_the_measurement_when_it_is_supplied():
    """The capture events carry no usage, so without it the four representable
    usage fields are never exercised at all — the loop would be reporting on a
    trace poorer than a real rollout's."""
    without = _trip(prompts=PROMPTS)
    with_usage = _trip(
        prompts=PROMPTS, usage=TraceUsage(input_tokens=10, output_tokens=2)
    )
    assert not [c for c in without.report.comparisons if c.path.startswith("usage.")]
    assert (
        _outcome(with_usage.report, "usage.input_tokens") is RoundTripOutcome.PRESERVED
    )


# ---------------------------------------------------------------------------
# What the comparison deliberately does not look at
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(EXCLUDED_PATHS))
def test_representation_fields_are_not_compared(path):
    """A trace read from ATIF is correctly labelled as coming from ATIF.

    Counting that as damage would report a true statement as a loss, and
    `provenance` would be the single largest "loss" in every measurement.
    """
    trip = _trip(prompts=PROMPTS)
    assert not [
        comparison
        for comparison in trip.report.comparisons
        if comparison.path == path or comparison.path.startswith(f"{path}.")
    ]


def test_an_opaque_block_is_compared_whole():
    """A source block has no schema here, so its keys are not IR fields.

    Walking into it would report three findings for one loss whenever the block
    happened to have three keys — weighting a single fact by the shape of the
    data it described.
    """
    trip = _trip(prompts=PROMPTS)
    paths = {comparison.path for comparison in trip.report.comparisons}
    assert "events[].tool_call.content[].raw" in paths
    assert not [
        path for path in paths if path.startswith("events[].tool_call.content[].raw.")
    ]


def test_a_default_empty_container_is_not_counted_as_a_value():
    """``extensions: {}`` is the absence of extensions, not an observed empty
    mapping — unlike ``arguments: {}``, where the distinction is the finding."""
    before = CanonicalTrace(
        provenance=Provenance(source_format="test"),
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.USER_MESSAGE,
                text="hi",
                provenance=Provenance(source_format="test"),
            )
        ],
    )
    after = before.model_copy(deep=True)
    report = compare_traces(before, after)
    assert "events[].extensions" not in {c.path for c in report.comparisons}
    assert "extensions" not in {c.path for c in report.comparisons}


def test_arguments_are_not_treated_as_a_default_empty_container():
    """The exception that makes the rule above safe."""
    assert "events[].tool_call.arguments" not in DEFAULT_EMPTY_PATHS
    trace = CanonicalTrace(
        provenance=Provenance(source_format="test"),
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.TOOL_CALL,
                tool_call=ToolCall(call_id="c", arguments={}),
                provenance=Provenance(source_format="test"),
            )
        ],
    )
    report = compare_traces(trace, trace)
    comparison = next(
        c for c in report.comparisons if c.path == "events[].tool_call.arguments"
    )
    assert comparison.before_count == 1


# ---------------------------------------------------------------------------
# The declared half — read off the models, so no field escapes silently
# ---------------------------------------------------------------------------


def _nested_model(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """The model an annotation ultimately names, and whether it is a list of it.

    ``ToolCall | None`` and ``list[TraceEvent]`` both wrap a model, and only the
    second contributes ``[]`` to the path — conflating them would produce
    ``events[].tool_call[].arguments``, a path no dumped trace can contain.
    """
    origin = get_origin(annotation)
    if origin is list:
        nested, _ = _nested_model(get_args(annotation)[0])
        return nested, nested is not None
    if origin in (Union, UnionType):
        for argument in get_args(annotation):
            if argument is type(None):
                continue
            nested, is_list = _nested_model(argument)
            if nested is not None:
                return nested, is_list
        return None, False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False
    return None, False


def _model_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    """Every canonical field path reachable from *model*.

    Nested models are walked; a list of models contributes ``[]`` to the path,
    exactly as the harness produces it from a dumped trace.
    """
    paths: set[str] = set()
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        nested, is_list = _nested_model(field.annotation)
        if nested is not None and nested is not model:
            paths |= _model_paths(nested, f"{path}[]" if is_list else path)
        else:
            paths.add(path)
    return paths


def test_every_ir_field_has_a_declared_representability():
    """A new IR field cannot slip through the round trip undecided.

    The measurement is only as honest as its declared half: a field with no
    entry defaults to unrepresentable, which would quietly credit the format
    with a limit it may not have.
    """
    undecided = sorted(
        path
        for path in _model_paths(CanonicalTrace)
        if not any(
            path == excluded or path.startswith(f"{excluded}.")
            for excluded in EXCLUDED_PATHS
        )
        and declared_entry_for(path) is None
    )
    assert undecided == []


def test_no_stale_entry_in_the_table():
    """The complement: an entry naming a field the IR no longer has would make
    the table look more complete than it is.

    An entry may name a section rather than a leaf — ``events[].usage`` governs
    all nine of its fields, because ATIF's decision is the same for every one of
    them — so a valid entry is a leaf path or a prefix of one.
    """
    known = _model_paths(CanonicalTrace)
    prefixes = {
        ".".join(path.split(".")[:cut])
        for path in known
        for cut in range(1, len(path.split(".")) + 1)
    }
    stale = sorted(path for path in ATIF_REPRESENTABILITY if path not in prefixes)
    assert stale == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("events[].extensions.step_id", Representability.NOT_IN_ATIF),
        ("extensions.schema_version", Representability.NOT_IN_ATIF),
        ("events[].reasoning_segments[]", Representability.NOT_IN_ATIF),
        ("events[].tool_call.arguments", Representability.REPRESENTABLE),
        ("agent.agent_name", Representability.REPRESENTABLE),
        ("nothing.like.this", Representability.NOT_IN_ATIF),
    ],
)
def test_representability_lookup_walks_up_to_the_nearest_entry(path, expected):
    assert representability_of(path) is expected


def test_a_representable_loss_would_be_reported_separately_from_a_format_one():
    """The distinction the summary rests on, forced rather than observed.

    `events[].usage` is representable — ATIF has a per-step metrics slot that
    `ir_to_atif` does not write — so a trace carrying per-event usage loses
    something *we* could carry. It must not be counted with the timestamps
    beside it, which no ATIF document could hold.
    """
    before = CanonicalTrace(
        provenance=Provenance(source_format="test"),
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.AGENT_MESSAGE,
                text="m",
                usage=TraceUsage(input_tokens=1),
                started_at=None,
                provenance=Provenance(source_format="test"),
            )
        ],
    )
    after = CanonicalTrace(
        provenance=Provenance(source_format="test"),
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.AGENT_MESSAGE,
                text="m",
                provenance=Provenance(source_format="test"),
            )
        ],
    )
    summary = compare_traces(before, after).summary()
    assert summary["lost"] == 1
    assert summary["non_representable"] == 0


def test_the_summary_accounts_for_every_comparison():
    trip = _trip(prompts=PROMPTS)
    assert sum(trip.report.summary().values()) == len(trip.report.comparisons)


def test_the_value_summary_never_claims_more_survived_than_arrived():
    trip = _trip(prompts=PROMPTS)
    values = trip.report.value_summary()
    assert values["values_preserved"] <= values["values_before"]
    assert (
        values["values_preserved"] + values["values_not_preserved"]
        == values["values_before"]
    )


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_the_harness_reads_traces_and_not_reports():
    """`compare_traces` must not consult the loss reports it sits beside.

    A comparison derived from the declarations would confirm the converters
    against themselves: a loss nobody declared would be invisible, which is the
    one failure this measurement exists to catch.
    """
    trace = acp_events_to_ir(_rich_events(), session_id="s", agent_name="a")
    stripped = trace.model_copy(update={"losses": None})
    assert compare_traces(trace, trace).summary() == (
        compare_traces(stripped, stripped).summary()
    )


def test_a_text_block_and_its_source_block_are_declared_apart():
    """The rendered text is representable and the block that produced it is not.

    They sit at neighbouring paths, so a single entry covering ``content[]``
    would credit ATIF with carrying the source block — which is §5 loss #5,
    the one the IR was built to stop being silent.
    """
    assert (
        ATIF_REPRESENTABILITY["events[].tool_call.content[].text"]
        is Representability.REPRESENTABLE
    )
    assert (
        ATIF_REPRESENTABILITY["events[].tool_call.content[].raw"]
        is Representability.NOT_IN_ATIF
    )
    assert "events[].tool_call.content[].raw" in OPAQUE_PATHS
