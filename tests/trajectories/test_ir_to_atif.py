"""Conversion suite for ``canonical Trace IR → ATIF`` (Slice D).

The inbound edge tested whether every absence could be declared. This one tests
the opposite half of the taxonomy: ATIF *requires* values the IR does not carry,
so this is the first converter that fabricates, and `SYNTHESIZED` stops being a
decorative enum member.

The suite is organized around one claim:

    ir_to_atif(acp_events_to_ir(events), prompts=P)
        ==
    trajectory_to_atif_record(events=events, prompts=P)

Parity with the existing direct exporter, on the same inputs, for the document —
not the report, since the direct exporter produces none. If the hub lost
anything the direct path preserved, that equality fails. Every deviation is
enumerated in one test rather than left to be discovered in a diff.

Nothing here writes to disk and `export_atif.py` is imported read-only, as the
oracle to compare against.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from benchflow.trajectories import export_atif
from benchflow.trajectories import ir_to_atif as ir_to_atif_module
from benchflow.trajectories._export_common import ThoughtBuffer
from benchflow.trajectories.export_atif import trajectory_to_atif_record
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
    TraceUsage,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_to_atif import (
    ATIF_SCHEMA_VERSION,
    LOSS_DIRECTION,
    ir_to_atif,
)
from tests.trajectories.test_atif_preservation import _rich_events
from tests.trajectories.test_trace_ir import resolve_ir_path

PROMPTS = ["Solve the task.", "Then stop."]


def _both_paths(
    events: list[dict[str, Any]],
    *,
    prompts: list[str] | None = None,
    session_id: str = "sess-d",
    agent_name: str = "claude-code",
    model: str | None = "claude-sonnet-5",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the direct exporter and the hub over the same input."""
    direct = trajectory_to_atif_record(
        session_id=session_id,
        agent_name=agent_name,
        events=events,
        prompts=prompts,
        model=model,
    )
    trace = acp_events_to_ir(
        events, session_id=session_id, agent_name=agent_name, model=model
    )
    through_hub, _ = ir_to_atif(trace, prompts=prompts)
    return direct, through_hub


def _fields(report, space: PathSpace = PathSpace.HUB) -> set[str]:
    return {r.field for r in report.records if r.space is space}


# ---------------------------------------------------------------------------
# Parity with the direct exporter
# ---------------------------------------------------------------------------


def test_the_hub_reproduces_the_direct_exporter_on_a_real_captured_trace():
    """The load-bearing test of this slice.

    Events produced by driving a real `ACPSession` through the production
    capture path, then converted both ways. Byte-for-byte equality of the two
    documents is what says the IR did not lose anything ATIF was getting.
    """
    direct, through_hub = _both_paths(_rich_events(), prompts=PROMPTS)
    assert through_hub == direct


@pytest.mark.parametrize(
    "events",
    [
        pytest.param(
            [{"type": "agent_message", "text": "only a message"}], id="message"
        ),
        pytest.param(
            [
                {"type": "user_message", "text": "u"},
                {"type": "agent_thought", "text": "t"},
                {"type": "agent_message", "text": "a"},
            ],
            id="thought-then-message",
        ),
        pytest.param(
            [
                {"type": "agent_thought", "text": "one"},
                {"type": "agent_thought", "text": "two"},
                {"type": "agent_message", "text": "a"},
            ],
            id="consecutive-thoughts-joined",
        ),
        pytest.param(
            [{"type": "agent_thought", "text": "trailing"}], id="trailing-thought-flush"
        ),
        pytest.param(
            [
                {
                    "type": "tool_call",
                    "tool_call_id": "",
                    "kind": "",
                    "title": "",
                    "status": "completed",
                    "content": [],
                }
            ],
            id="empty-tool-fields-synthesized",
        ),
        pytest.param(
            [
                {"type": "agent_thought", "text": "why"},
                {
                    "type": "tool_call",
                    "tool_call_id": "t1",
                    "kind": "execute",
                    "title": "ls",
                    "status": "failed",
                    "content": [
                        {"type": "content", "content": {"type": "text", "text": "out"}}
                    ],
                },
            ],
            id="tool-with-reasoning-and-observation",
        ),
        pytest.param(
            [
                {"type": "user_message", "text": ""},
                {"type": "agent_message", "text": ""},
                {"type": "agent_message", "text": "kept"},
            ],
            id="text-empty-events-dropped-by-both",
        ),
        pytest.param(
            [
                {"type": "agent_message", "text": "a"},
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 1.0,
                    "pending_tool_call_ids": [],
                    "terminal_trajectory_complete": True,
                },
            ],
            id="timeout-dropped-by-both",
        ),
        pytest.param(
            [{"type": "mystery", "payload": 1}, {"type": "agent_message", "text": "a"}],
            id="unknown-dropped-by-both",
        ),
    ],
)
def test_parity_holds_shape_by_shape(events):
    direct, through_hub = _both_paths(events, prompts=PROMPTS)
    assert through_hub == direct


def test_parity_holds_without_prompts():
    direct, through_hub = _both_paths(_rich_events())
    assert through_hub == direct


def test_parity_holds_when_the_trace_has_no_session_id_or_model():
    direct, through_hub = _both_paths(
        _rich_events(), prompts=PROMPTS, session_id="", agent_name="", model=None
    )
    assert through_hub == direct
    assert through_hub["agent"] == {"name": "unknown", "version": "unknown"}
    assert "session_id" not in through_hub


def test_both_paths_refuse_a_trajectory_with_no_representable_step():
    """ATIF requires one step; fabricating an empty one would be inventing."""
    empty_events: list[dict[str, Any]] = []
    with pytest.raises(ValueError):
        trajectory_to_atif_record(
            session_id="s", agent_name="a", events=empty_events, prompts=None
        )
    with pytest.raises(ValueError):
        ir_to_atif(acp_events_to_ir(empty_events))


def test_the_schema_version_matches_the_direct_exporter():
    """Redefined rather than imported, so the equality is asserted not assumed."""
    assert ATIF_SCHEMA_VERSION == export_atif.ATIF_SCHEMA_VERSION


def test_the_thought_join_matches_the_shared_buffer():
    """The hub reimplements ``ThoughtBuffer``'s join; the two must agree."""
    buffer = ThoughtBuffer()
    for text in ("one", "two", "three"):
        buffer.push(text)
    expected = buffer.take()

    events = [{"type": "agent_thought", "text": t} for t in ("one", "two", "three")]
    events.append({"type": "agent_message", "text": "done"})
    document, _ = ir_to_atif(acp_events_to_ir(events))
    assert document["steps"][0]["reasoning_content"] == expected


# ---------------------------------------------------------------------------
# The one deliberate deviation
# ---------------------------------------------------------------------------


def test_oracle_becomes_its_own_source_instead_of_a_prefixed_agent_step():
    """The single enumerated divergence from the direct exporter.

    `acp_events_to_atif_steps` renders oracle activity as an `agent` step whose
    message is prefixed ``[oracle: …]``, recoverable only by string matching
    (§5.1). The in-repo validator already accepts ``source: "oracle"``, and the
    IR carries the role, so the hub emits it.
    """
    events = [{"type": "oracle", "command": "solve.sh", "return_code": 0}]
    direct, through_hub = _both_paths(events, prompts=None)

    assert direct["steps"][0] == {
        "step_id": 1,
        "source": "agent",
        "message": "[oracle: solve.sh]",
    }
    assert through_hub["steps"][0] == {
        "step_id": 1,
        "source": "oracle",
        "message": "solve.sh",
    }
    assert through_hub != direct
    # Everything except that step is still identical.
    assert through_hub["agent"] == direct["agent"]
    assert through_hub["final_metrics"] == direct["final_metrics"]


def test_the_oracle_deviation_is_the_only_one_on_conformant_input():
    """Enumerated: any *other* divergence has to fail a test, not be discovered.

    Runs both paths over a trajectory containing every capture event type plus
    an oracle record, and asserts the documents differ in exactly the oracle
    step and nothing else.
    """
    events = [*_rich_events(), {"type": "oracle", "command": "check.sh"}]
    direct, through_hub = _both_paths(events, prompts=PROMPTS)

    assert len(direct["steps"]) == len(through_hub["steps"])
    differing = [
        (a, b)
        for a, b in zip(direct["steps"], through_hub["steps"], strict=True)
        if a != b
    ]
    assert len(differing) == 1, differing
    assert differing[0][0]["source"] == "agent"
    assert differing[0][1]["source"] == "oracle"


def test_the_produced_document_passes_the_in_repo_atif_validator(tmp_path):
    """Including the oracle source, which that validator already accepts."""
    from tests.integration.scenarios import atif_issues

    events = [*_rich_events(), {"type": "oracle", "command": "check.sh"}]
    document, _ = ir_to_atif(
        acp_events_to_ir(events, session_id="s", agent_name="a"), prompts=PROMPTS
    )
    rollout = tmp_path / "rollout"
    (rollout / "trainer").mkdir(parents=True)
    (rollout / "trainer" / "atif.json").write_text(json.dumps(document))
    assert atif_issues(rollout) == []


# ---------------------------------------------------------------------------
# SYNTHESIZED — the class this slice exists to stress
# ---------------------------------------------------------------------------


def test_every_fabricated_value_is_declared_synthesized():
    """The four hub-space fabrications, on one trace that forces all of them."""
    events = [
        {
            "type": "tool_call",
            "tool_call_id": "",
            "kind": "",
            "title": "",
            "status": "completed",
            "content": [],
        }
    ]
    trace = acp_events_to_ir(events)  # no agent_name, no version
    document, report = ir_to_atif(trace)

    synthesized = {
        r.field
        for r in report.by_class(LossClass.SYNTHESIZED)
        if r.space is PathSpace.HUB
    }
    assert synthesized == {
        "events[0].tool_call.call_id",
        "events[0].tool_call.name",
        "events[0].tool_call.arguments",
        "agent.agent_name",
        "agent.agent_version",
    }

    call = document["steps"][0]["tool_calls"][0]
    assert call["tool_call_id"] == "call_1"
    assert call["function_name"] == "tool"
    assert call["arguments"] == {}
    assert document["agent"] == {"name": "unknown", "version": "unknown"}


def test_arguments_are_declared_synthesized_only_when_the_ir_carried_none():
    """`{}` in the document either way; the report is what tells them apart."""
    absent = acp_events_to_ir([{"type": "tool_call", "tool_call_id": "t"}])
    document, report = ir_to_atif(absent)
    assert document["steps"][0]["tool_calls"][0]["arguments"] == {}
    assert report.for_field("events[0].tool_call.arguments")

    captured = acp_events_to_ir([{"type": "tool_call", "tool_call_id": "t"}])
    captured.events[0].tool_call.arguments = {}
    document, report = ir_to_atif(captured)
    assert document["steps"][0]["tool_calls"][0]["arguments"] == {}
    assert report.for_field("events[0].tool_call.arguments") == []

    real = acp_events_to_ir([{"type": "tool_call", "tool_call_id": "t"}])
    real.events[0].tool_call.arguments = {"command": "ls"}
    document, report = ir_to_atif(real)
    assert document["steps"][0]["tool_calls"][0]["arguments"] == {"command": "ls"}
    assert report.for_field("events[0].tool_call.arguments") == []


def test_the_arguments_story_composes_across_the_two_edges():
    """The property the hub exists for, on one field and one path.

    The ACP edge says the source never carried arguments; the ATIF edge says the
    target demanded them anyway. Same hub path, two reports, one history.
    """
    trace = acp_events_to_ir(_rich_events())
    _, outbound = ir_to_atif(trace)

    field = next(
        r.field
        for r in trace.losses.records
        if r.field.endswith(".tool_call.arguments")
    )
    inbound_record = trace.losses.for_field(field)[0]
    outbound_record = outbound.for_field(field)[0]

    assert inbound_record.loss_class is LossClass.UNSUPPORTED
    assert outbound_record.loss_class is LossClass.SYNTHESIZED
    assert inbound_record.space is outbound_record.space is PathSpace.HUB
    assert trace.losses.direction == "acp->ir"
    assert outbound.direction == LOSS_DIRECTION


# ---------------------------------------------------------------------------
# TARGET space
# ---------------------------------------------------------------------------


def test_target_only_values_are_declared_in_the_target_space():
    trace = acp_events_to_ir(_rich_events(), agent_name="a")
    document, report = ir_to_atif(trace, prompts=PROMPTS)

    assert _fields(report, PathSpace.TARGET) == {
        "steps[0]",
        "steps[1]",
        "steps[].message",
        "final_metrics.total_steps",
    }
    assert all(
        r.loss_class is LossClass.SYNTHESIZED for r in report.by_space(PathSpace.TARGET)
    )
    # The prompt steps they name are really there.
    assert document["steps"][0]["message"] == PROMPTS[0]
    assert document["steps"][1]["message"] == PROMPTS[1]


def test_no_prompt_steps_means_no_prompt_records():
    _, report = ir_to_atif(acp_events_to_ir(_rich_events()))
    assert not [
        r for r in report.by_space(PathSpace.TARGET) if r.field.startswith("steps[0")
    ]


def test_target_records_are_not_read_as_ir_paths():
    """They address the ATIF document, which the IR does not contain."""
    trace = acp_events_to_ir(_rich_events())
    _, report = ir_to_atif(trace, prompts=PROMPTS)
    canonical = trace.model_dump(mode="json")
    for record in report.by_space(PathSpace.TARGET):
        assert not resolve_ir_path(canonical, record.field)[0]


def test_every_hub_record_of_the_outbound_report_resolves_in_the_trace():
    """The same guard as the inbound edge, applied to the outbound report."""
    for events in (
        _rich_events(),
        [*_rich_events(), {"type": "oracle", "command": "x"}],
        [{"type": "agent_message", "text": ""}, {"type": "agent_message", "text": "a"}],
    ):
        trace = acp_events_to_ir(events)
        _, report = ir_to_atif(trace, prompts=PROMPTS)
        canonical = trace.model_dump(mode="json")
        for record in report.records:
            if record.space is not PathSpace.HUB:
                continue
            if record.field.startswith("events[]"):
                continue
            assert resolve_ir_path(canonical, record.field)[0], record.field


# ---------------------------------------------------------------------------
# Report ownership
# ---------------------------------------------------------------------------


def test_an_outbound_conversion_leaves_the_input_trace_untouched():
    """A trace may be converted to many targets; none of them describes it."""
    trace = acp_events_to_ir(_rich_events(), agent_name="a")
    before = trace.model_dump_json()
    inbound_records = len(trace.losses.records)

    document, report = ir_to_atif(trace, prompts=PROMPTS)

    assert trace.model_dump_json() == before
    assert len(trace.losses.records) == inbound_records
    assert trace.losses.direction == "acp->ir"
    assert report is not trace.losses
    assert validate_trace(trace) == []
    assert document["schema_version"] == ATIF_SCHEMA_VERSION


def test_two_outbound_conversions_of_one_trace_are_independent():
    trace = acp_events_to_ir(_rich_events())
    _, first = ir_to_atif(trace, prompts=PROMPTS)
    _, second = ir_to_atif(trace)
    assert first is not second
    assert len(first.records) > len(second.records)  # the prompt steps
    assert trace.losses.direction == "acp->ir"


# ---------------------------------------------------------------------------
# Losses this edge really has
# ---------------------------------------------------------------------------


def test_opaque_content_blocks_are_declared_dropped():
    """§5 loss 5 reappears here: the IR carries them, ATIF has no slot."""
    trace = acp_events_to_ir(
        [
            {
                "type": "tool_call",
                "tool_call_id": "t",
                "kind": "edit",
                "status": "completed",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "ok"}},
                    {"type": "diff", "oldText": "a", "newText": "b"},
                ],
            }
        ]
    )
    document, report = ir_to_atif(trace)

    assert "newText" not in json.dumps(document)
    dropped = report.for_field("events[0].tool_call.content")
    assert len(dropped) == 1
    assert dropped[0].loss_class is LossClass.DROPPED


def test_the_timeout_is_dropped_here_and_says_so():
    """The hub preserved it; this edge cannot, and that is the honest result."""
    trace = acp_events_to_ir(
        [
            {"type": "agent_message", "text": "a"},
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 1.0,
                "pending_tool_call_ids": [],
                "terminal_trajectory_complete": True,
            },
        ]
    )
    document, report = ir_to_atif(trace)

    assert "wall_clock_timeout" not in json.dumps(document)
    dropped = report.for_field("events[1]")
    assert dropped and dropped[0].loss_class is LossClass.DROPPED
    assert dropped[0].doc_ref == "§5 loss 4"


def test_reasoning_boundaries_are_declared_normalized():
    trace = acp_events_to_ir(
        [
            {"type": "agent_thought", "text": "one"},
            {"type": "agent_thought", "text": "two"},
            {"type": "agent_message", "text": "a"},
        ]
    )
    document, report = ir_to_atif(trace)
    assert document["steps"][0]["reasoning_content"] == "one\n\ntwo"
    normalized = report.for_field("events[].reasoning_segments")
    assert normalized and normalized[0].loss_class is LossClass.NORMALIZED


def test_usage_maps_three_fields_and_declares_the_rest_dropped():
    trace = acp_events_to_ir(_rich_events(), agent_name="a")
    trace.usage = TraceUsage(
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=5,
        cache_creation_tokens=3,
        total_tokens=125,
        source="llm_proxy_normalized",
    )
    document, report = ir_to_atif(trace)

    assert document["final_metrics"]["total_prompt_tokens"] == 100
    assert document["final_metrics"]["total_completion_tokens"] == 20
    assert document["final_metrics"]["total_cached_tokens"] == 5
    assert _fields(report) >= {
        "usage.cache_creation_tokens",
        "usage.total_tokens",
        "usage.source",
    }
    assert "reasoning_tokens" not in json.dumps(document)


def test_a_trace_without_usage_declares_no_usage_losses():
    """This edge loses nothing it was never given; the inbound report said that."""
    _, report = ir_to_atif(acp_events_to_ir(_rich_events()))
    assert not [r for r in report.records if r.field.startswith("usage")]


def test_systemic_losses_are_declared_once_and_only_when_they_apply():
    trace = acp_events_to_ir(_rich_events(), agent_name="a")
    _, report = ir_to_atif(trace)
    hub = _fields(report)

    assert "events[].index" in hub
    assert "events[].provenance" in hub
    assert "events[].source_type" in hub
    assert "events[].tool_call.name_semantics" in hub
    assert "events[].reasoning_segments" in hub
    # No per-event usage or timestamps in an ACP-derived trace, so no claim.
    assert "events[].usage" not in hub
    assert "events[].started_at" not in hub


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_the_converter_imports_only_the_ir():
    """The hub must not depend on the exporters it sits between.

    In particular it does not import ``export_atif``: the schema version is
    redefined and pinned by a test instead, so the family stays a leaf.
    """
    tree = ast.parse(Path(ir_to_atif_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    benchflow_imports = {name for name in imported if name.startswith("benchflow")}
    assert benchflow_imports == {"benchflow.trajectories.ir"}, sorted(benchflow_imports)


def test_a_hand_built_trace_converts_without_going_through_acp():
    """The hub is not a wrapper around the ACP path."""
    provenance = Provenance(source_format="hand-built")
    trace = CanonicalTrace(
        provenance=provenance,
        agent=ModelInfo(agent_name="agent-x", agent_version="1.2.3", model="m"),
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.USER_MESSAGE,
                role=Role.USER,
                text="hello",
                provenance=provenance,
            ),
            TraceEvent(
                index=1,
                kind=EventKind.TOOL_CALL,
                role=Role.AGENT,
                tool_call=ToolCall(
                    call_id="c1",
                    name="search",
                    arguments={"q": "x"},
                    status=ToolStatus.COMPLETED,
                    content=[ContentBlock(kind=ContentBlockKind.TEXT, text="found")],
                ),
                provenance=provenance,
            ),
        ],
    )
    document, report = ir_to_atif(trace)

    assert document["agent"] == {
        "name": "agent-x",
        "version": "1.2.3",
        "model_name": "m",
    }
    assert document["steps"][1]["tool_calls"][0]["arguments"] == {"q": "x"}
    # A real version and real arguments mean nothing was fabricated for them.
    assert "agent.agent_version" not in _fields(report)
    assert "events[1].tool_call.arguments" not in _fields(report)
