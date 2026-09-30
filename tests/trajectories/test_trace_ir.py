"""Validation suite for the provisional Canonical Trace IR (Slice B).

The IR (:mod:`benchflow.trajectories.ir`) is a *proposal implemented as code*.
Nothing imports it and nothing writes it to disk, so the usual safety net — a
downstream consumer breaking — does not exist for it. This suite is that net:
it pins the invariants the module documents, the vocabularies it must stay
aligned with, and the isolation claim that makes it reversible.

Four groups, in the order they matter:

* **Invariants** — one test per rule in :func:`validate_trace`, each with a
  violating trace *and* a clean one, so a rule that stopped firing fails here
  rather than passing silently.
* **Vocabulary alignment** — the IR's tool-status and event-kind vocabularies
  are checked against the real producer (``_events_to_trajectory``) and the real
  ACP enum, both read from source. Adding a branch to the capture path fails
  this suite until the IR accounts for it, exactly as the Slice A conformance
  suite does for the schema.
* **Isolation** — nothing outside the IR module family imports the IR. This is
  the executable form of "zero runtime behaviour change"; if a future PR wires
  the IR into a run path, this test is the one that must be deliberately
  updated.
* **The worked example and the canonical encoding** — the JSON in
  ``docs/trace-interop.md`` §8 is built here in code and compared to the block
  in the document, so the documented example cannot drift from what the models
  actually produce. The encoding it is written in is itself pinned: nulls are
  retained, because a dropped key leaves the loss record that declares it
  pointing at nothing.

Nothing here writes outside ``tmp_path`` and no runtime module is imported for
its side effects.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from benchflow.acp.types import ToolCallStatus
from benchflow.trajectories import ir as ir_module
from benchflow.trajectories.ir import (
    TRACE_IR_VERSION,
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    LossRecord,
    LossReport,
    ModelInfo,
    OutcomeStatus,
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
from tests.trajectories.test_acp_capture_event_schema import _emitted_event_types

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "benchflow"
DOCS_PATH = REPO_ROOT / "docs" / "trace-interop.md"

ACP_PROVENANCE = Provenance(
    source_format="acp-capture-v1", producer="_events_to_trajectory"
)


def _event(index: int, kind: EventKind, **kwargs) -> TraceEvent:
    """A minimally-populated event, so each test states only what it is about."""
    return TraceEvent(index=index, kind=kind, provenance=ACP_PROVENANCE, **kwargs)


def _trace(*events: TraceEvent, losses: LossReport | None = None) -> CanonicalTrace:
    return CanonicalTrace(provenance=ACP_PROVENANCE, events=list(events), losses=losses)


def _tool_call(**kwargs) -> ToolCall:
    """A tool call whose ``arguments`` absence is already declared elsewhere."""
    kwargs.setdefault("arguments", {})
    return ToolCall(**kwargs)


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_version_constant_is_explicitly_v0():
    """v0 is a claim about stability, so the string is pinned, not derived."""
    assert TRACE_IR_VERSION == "bf-trace-ir-v0"
    assert CanonicalTrace(provenance=ACP_PROVENANCE).ir_version == TRACE_IR_VERSION
    assert LossReport(direction="acp->ir").ir_version == TRACE_IR_VERSION


def test_mismatched_ir_version_is_an_invariant_violation():
    trace = _trace()
    trace.ir_version = "bf-trace-ir-v1"
    issues = validate_trace(trace)
    assert any("ir_version" in issue for issue in issues), issues


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_dense_ordering_is_required():
    clean = _trace(
        _event(0, EventKind.AGENT_MESSAGE, text="a"),
        _event(1, EventKind.AGENT_MESSAGE, text="b"),
    )
    assert validate_trace(clean) == []

    holed = _trace(
        _event(0, EventKind.AGENT_MESSAGE, text="a"),
        _event(2, EventKind.AGENT_MESSAGE, text="b"),
    )
    assert any("index is 2, expected 1" in issue for issue in validate_trace(holed))


def test_tool_call_kind_and_payload_must_agree_both_ways():
    missing_payload = _trace(_event(0, EventKind.TOOL_CALL))
    assert any(
        "no tool_call payload" in issue for issue in validate_trace(missing_payload)
    )

    stray_payload = _trace(
        _event(0, EventKind.AGENT_MESSAGE, text="hi", tool_call=_tool_call())
    )
    assert any(
        "carries a tool_call payload" in issue
        for issue in validate_trace(stray_payload)
    )

    agreed = _trace(_event(0, EventKind.TOOL_CALL, tool_call=_tool_call()))
    assert validate_trace(agreed) == []


def test_content_blocks_must_carry_their_payload():
    """An opaque block with no ``raw`` is a dropped block claiming preservation."""
    empty_text = _trace(
        _event(
            0,
            EventKind.TOOL_CALL,
            tool_call=_tool_call(content=[ContentBlock(kind=ContentBlockKind.TEXT)]),
        )
    )
    assert any("is text but carries no text" in i for i in validate_trace(empty_text))

    empty_opaque = _trace(
        _event(
            0,
            EventKind.TOOL_CALL,
            tool_call=_tool_call(content=[ContentBlock(kind=ContentBlockKind.OPAQUE)]),
        )
    )
    assert any(
        "is opaque but carries no raw block" in i for i in validate_trace(empty_opaque)
    )

    carried = _trace(
        _event(
            0,
            EventKind.TOOL_CALL,
            tool_call=_tool_call(
                content=[
                    ContentBlock(kind=ContentBlockKind.TEXT, text=""),
                    ContentBlock(
                        kind=ContentBlockKind.OPAQUE,
                        raw={"type": "diff", "oldText": "a", "newText": "b"},
                    ),
                ]
            ),
        )
    )
    assert validate_trace(carried) == []


def test_a_text_block_may_be_empty_but_not_absent():
    """``""`` is an observed value; ``None`` means the block was not carried."""
    present = ContentBlock(kind=ContentBlockKind.TEXT, text="")
    assert present.text == ""
    assert (
        validate_trace(
            _trace(
                _event(0, EventKind.TOOL_CALL, tool_call=_tool_call(content=[present]))
            )
        )
        == []
    )


def test_reasoning_segments_must_join_to_reasoning():
    """The join is ``ThoughtBuffer``'s; the segments are the boundary it destroys."""
    consistent = _trace(
        _event(
            0,
            EventKind.AGENT_REASONING,
            reasoning="first\n\nsecond",
            reasoning_segments=["first", "second"],
        )
    )
    assert validate_trace(consistent) == []

    # The §5 loss #10 shape: one thought that already contains a blank line.
    # Same joined string, different segmentation — and the IR keeps them apart.
    single = _trace(
        _event(
            0,
            EventKind.AGENT_REASONING,
            reasoning="first\n\nsecond",
            reasoning_segments=["first\n\nsecond"],
        )
    )
    assert validate_trace(single) == []
    assert consistent.events[0].reasoning == single.events[0].reasoning
    assert (
        consistent.events[0].reasoning_segments != single.events[0].reasoning_segments
    )

    divergent = _trace(
        _event(
            0,
            EventKind.AGENT_REASONING,
            reasoning="first\n\nsecond",
            reasoning_segments=["totally", "different"],
        )
    )
    assert any("do not join" in issue for issue in validate_trace(divergent))


def test_a_reasoning_event_must_carry_reasoning():
    assert any(
        "carries no reasoning" in issue
        for issue in validate_trace(_trace(_event(0, EventKind.AGENT_REASONING)))
    )
    assert (
        validate_trace(
            _trace(_event(0, EventKind.AGENT_REASONING, reasoning_segments=["only"]))
        )
        == []
    )


def test_absent_arguments_require_a_declared_loss():
    """The invariant that makes the loss report a contract rather than a comment."""
    undeclared = _trace(
        _event(0, EventKind.TOOL_CALL, tool_call=ToolCall(arguments=None))
    )
    issues = validate_trace(undeclared)
    assert any("absence must be declared" in issue for issue in issues), issues

    losses = LossReport(direction="acp->ir")
    losses.add(
        "events[0].tool_call.arguments",
        LossClass.UNSUPPORTED,
        "handle_update never reads rawInput",
        "§5 loss #1",
    )
    declared = _trace(
        _event(0, EventKind.TOOL_CALL, tool_call=ToolCall(arguments=None)),
        losses=losses,
    )
    assert validate_trace(declared) == []


def test_empty_arguments_are_not_absent_arguments():
    """``{}`` needs no loss record: it is an observation, not a gap."""
    captured_empty = _trace(
        _event(0, EventKind.TOOL_CALL, tool_call=ToolCall(arguments={}))
    )
    assert validate_trace(captured_empty) == []
    assert captured_empty.events[0].tool_call.arguments == {}


def test_a_user_message_is_not_attributed_to_the_agent():
    assert any(
        "attributed to agent" in issue
        for issue in validate_trace(
            _trace(_event(0, EventKind.USER_MESSAGE, text="hi", role=Role.AGENT))
        )
    )
    assert (
        validate_trace(
            _trace(_event(0, EventKind.USER_MESSAGE, text="hi", role=Role.USER))
        )
        == []
    )
    assert validate_trace(_trace(_event(0, EventKind.USER_MESSAGE, text="hi"))) == []


def test_validate_trace_reports_every_violation_not_just_the_first():
    trace = _trace(
        _event(5, EventKind.TOOL_CALL),
        _event(9, EventKind.AGENT_REASONING),
    )
    issues = validate_trace(trace)
    assert len(issues) >= 4, issues


# ---------------------------------------------------------------------------
# Strictness and round-tripping
# ---------------------------------------------------------------------------


def test_unknown_fields_are_rejected_and_extensions_are_the_escape_hatch():
    """``extra="forbid"`` makes a typo a failure instead of a silent no-op."""
    with pytest.raises(ValidationError):
        TraceEvent(
            index=0,
            kind=EventKind.AGENT_MESSAGE,
            provenance=ACP_PROVENANCE,
            reasonning="typo",
        )

    carried = _event(
        0,
        EventKind.TIMEOUT,
        outcome="wall_clock_timeout",
        extensions={"timeout_sec": 90.0, "pending_tool_call_ids": []},
    )
    assert carried.extensions["timeout_sec"] == 90.0


def test_a_trace_round_trips_through_json():
    original = _example_trace()
    restored = CanonicalTrace.model_validate(json.loads(original.model_dump_json()))
    assert restored == original
    assert validate_trace(restored) == []


def test_unknown_source_types_survive_as_unknown_rather_than_vanishing():
    """Today every exporter skips what it does not recognize (§5.1, last rows)."""
    event = _event(0, EventKind.UNKNOWN, source_type="some_future_acp_event")
    assert validate_trace(_trace(event)) == []
    assert event.source_type == "some_future_acp_event"


# ---------------------------------------------------------------------------
# The loss model
# ---------------------------------------------------------------------------


def test_loss_report_is_empty_only_as_a_claim():
    report = LossReport(direction="acp->ir")
    assert report.lossless

    report.add("events[0].tool_call.arguments", LossClass.UNSUPPORTED, "no rawInput")
    report.add("agent.agent_version", LossClass.SYNTHESIZED, "ATIF requires a version")

    assert not report.lossless
    assert len(report.by_class(LossClass.UNSUPPORTED)) == 1
    assert len(report.by_class(LossClass.DROPPED)) == 0
    assert report.for_field("agent.agent_version")[0].loss_class is (
        LossClass.SYNTHESIZED
    )


# ---------------------------------------------------------------------------
# Path spaces
# ---------------------------------------------------------------------------


def test_the_three_path_spaces_are_the_documented_ones():
    """Three spaces cover every direction; a new format must add none."""
    assert {space.value for space in PathSpace} == {"hub", "source", "target"}


def test_hub_is_the_default_so_older_records_keep_their_meaning():
    """The field is additive: a record written without a space is a hub record."""
    record = LossRecord(
        field="events[0].tool_call.arguments",
        loss_class=LossClass.UNSUPPORTED,
        detail="no rawInput",
    )
    assert record.space is PathSpace.HUB
    report = LossReport(direction="acp->ir")
    report.add("agent.agent_version", LossClass.SYNTHESIZED, "ATIF requires it")
    assert report.records[0].space is PathSpace.HUB


def test_the_same_path_in_two_spaces_names_two_different_objects():
    """`for_field` therefore takes the space, and does not default it away."""
    report = LossReport(direction="ir->atif")
    report.add("events[0]", LossClass.DROPPED, "hub event dropped")
    report.add(
        "events[0]", LossClass.SYNTHESIZED, "target step", space=PathSpace.TARGET
    )

    assert len(report.records) == 2
    assert len(report.for_field("events[0]")) == 1
    assert report.for_field("events[0]")[0].loss_class is LossClass.DROPPED
    assert (
        report.for_field("events[0]", PathSpace.TARGET)[0].loss_class
        is LossClass.SYNTHESIZED
    )
    assert len(report.by_space(PathSpace.HUB)) == 1
    assert len(report.by_space(PathSpace.TARGET)) == 1


def test_only_hub_records_satisfy_the_declared_absence_invariant():
    """The space is checked, never inferred from the string.

    A `TARGET` record holding the identical path addresses another document, so
    it declares nothing about this trace — and a `SOURCE` one likewise. If the
    invariant sniffed the string, both would wrongly satisfy it.
    """
    for space in (PathSpace.SOURCE, PathSpace.TARGET):
        losses = LossReport(direction="ir->atif")
        losses.add(
            "events[0].tool_call.arguments",
            LossClass.SYNTHESIZED,
            "same path, different document",
            space=space,
        )
        trace = _trace(
            _event(0, EventKind.TOOL_CALL, tool_call=ToolCall(arguments=None)),
            losses=losses,
        )
        issues = validate_trace(trace)
        assert any("absence must be declared" in issue for issue in issues), (
            space,
            issues,
        )

    hub = LossReport(direction="acp->ir")
    hub.add("events[0].tool_call.arguments", LossClass.UNSUPPORTED, "no rawInput")
    assert (
        validate_trace(
            _trace(
                _event(0, EventKind.TOOL_CALL, tool_call=ToolCall(arguments=None)),
                losses=hub,
            )
        )
        == []
    )


def test_non_hub_records_are_not_read_as_ir_paths():
    """A source or target path may be unresolvable in the IR without penalty.

    Both of these address documents the IR does not contain, so the canonical
    resolvability guard must not look at them at all.
    """
    losses = LossReport(direction="ir->atif")
    losses.add("events[7]", LossClass.DROPPED, "input entry", space=PathSpace.SOURCE)
    losses.add(
        "final_metrics.total_steps",
        LossClass.SYNTHESIZED,
        "target-only value",
        space=PathSpace.TARGET,
    )
    trace = _trace(_event(0, EventKind.AGENT_MESSAGE, text="hi"), losses=losses)

    assert validate_trace(trace) == []

    canonical = trace.model_dump(mode="json")
    for record in trace.losses.records:
        assert record.space is not PathSpace.HUB
        # Unresolvable as an IR path, and that is precisely why it is not one.
        assert not resolve_ir_path(canonical, record.field)[0]


def test_the_space_survives_the_canonical_encoding():
    report = LossReport(direction="ir->atif")
    report.add("steps[0]", LossClass.SYNTHESIZED, "prompt step", space=PathSpace.TARGET)
    document = report.model_dump(mode="json")
    assert document["records"][0]["space"] == "target"
    assert LossReport.model_validate(document).records[0].space is PathSpace.TARGET


def test_every_loss_class_of_the_documented_taxonomy_exists():
    """The four classes are the ones §5.1 already uses; renaming one is a change."""
    assert {c.value for c in LossClass} == {
        "unsupported",
        "dropped",
        "normalized",
        "synthesized",
    }


# ---------------------------------------------------------------------------
# Vocabulary alignment with the real producer
# ---------------------------------------------------------------------------


def test_ir_tool_status_is_a_superset_of_the_acp_status_vocabulary():
    """Read off the ACP enum, so widening it fails here instead of silently."""
    acp_values = {status.value for status in ToolCallStatus}
    ir_values = {status.value for status in ToolStatus}
    assert acp_values <= ir_values, sorted(acp_values - ir_values)
    assert "unknown" in ir_values, "sources that carry no status need a value"


def test_every_emitted_capture_event_type_maps_to_an_ir_kind():
    """The producer's vocabulary, read from ``_events_to_trajectory`` by AST.

    Mirrors the Slice A conformance check: adding a branch to the capture path
    fails this test until the IR says what that event becomes.
    """
    mapping = {
        "user_message": EventKind.USER_MESSAGE,
        "agent_message": EventKind.AGENT_MESSAGE,
        "agent_thought": EventKind.AGENT_REASONING,
        "tool_call": EventKind.TOOL_CALL,
        "agent_timeout": EventKind.TIMEOUT,
    }
    emitted = _emitted_event_types()
    assert emitted, "AST extraction found no event types; the walker is broken"

    unmapped = emitted - set(mapping)
    assert not unmapped, (
        f"_events_to_trajectory emits {sorted(unmapped)}, which the IR does not "
        "map to an EventKind. Add the mapping here and to docs/trace-interop.md "
        "§8 rather than deleting this check."
    )
    assert set(mapping.values()) <= set(EventKind)


def test_the_oracle_record_has_its_own_kind_and_role():
    """§2.4's alternative trajectory is representable without being called agent.

    ``acp_events_to_atif_steps`` renders it as an ``agent`` step prefixed
    ``[oracle: …]``, which a consumer can only undo by string matching. The IR
    keeps the distinction the ATIF validator already accepts.
    """
    assert EventKind.ORACLE.value == "oracle"
    assert Role.ORACLE.value == "oracle"


# ---------------------------------------------------------------------------
# Isolation — the executable form of "zero runtime behaviour change"
# ---------------------------------------------------------------------------


IR_FAMILY = (
    "ir",
    "ir_from_acp",
    "ir_to_atif",
    "ir_from_atif",
    "ir_from_otel",
    "ir_to_acp",
    "_otlp_anyvalue",
    "ir_round_trip",
    "ir_conformance",
    "ir_to_view",
    "ir_to_view_html",
)
"""The unwired modules of the proposal: the IR and its converters.

They may import each other — a converter that could not import the IR would be
useless — but nothing else may import them. Growing this tuple is how a new
converter joins the family; it is not a way to let a run path in.
"""

IR_FAMILY_PATHS = {f"src/benchflow/trajectories/{name}.py" for name in IR_FAMILY}

WIRING_SITES = frozenset({"src/benchflow/trajectories/viewer/legacy.py"})
"""The modules outside the family that are allowed to reach into it.

Slice I wires the IR into one run path — the trajectory viewer — through a
single opt-in branch (`viewer.TRACE_IR_ENV`). Naming that site here is the
deliberate, narrowed end of the "nothing imports this" claim: a second importer
appearing anywhere still fails :func:`test_only_the_ir_family_imports_the_ir`,
the import at the listed site is lazy and inside the branch, and removing the
branch is still enough to unwire the family completely.
"""


def _imported_ir_modules(path: Path) -> set[str]:
    """Which members of :data:`IR_FAMILY` *path* imports, read by AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for name in IR_FAMILY:
                    if alias.name == f"benchflow.trajectories.{name}":
                        found.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in IR_FAMILY:
                if module == f"benchflow.trajectories.{name}" or (
                    node.level and module == name
                ):
                    found.add(name)
            if module in ("benchflow.trajectories", ""):
                found.update(
                    alias.name for alias in node.names if alias.name in IR_FAMILY
                )
    return found


def _imported_ir_modules_from_node(node: ast.stmt) -> set[str]:
    """The same question asked of a single import statement."""
    found: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            for name in IR_FAMILY:
                if alias.name == f"benchflow.trajectories.{name}":
                    found.add(name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        for name in IR_FAMILY:
            if module == f"benchflow.trajectories.{name}":
                found.add(name)
        if module == "benchflow.trajectories":
            found.update(alias.name for alias in node.names if alias.name in IR_FAMILY)
    return found


def test_only_the_ir_family_imports_the_ir():
    """The proposal is reversible: deleting it cannot break a run path.

    Slice B stated this as "nothing under ``src/benchflow`` imports the IR".
    Slice C adds `ir_from_acp`, which necessarily imports it, so the property
    is restated at the boundary that actually matters — the family is closed,
    and no module outside it may reach in.

    When the IR is deliberately wired into production, this test is the one to
    update, and updating it is the moment the reversibility claim ends.
    """
    importers = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if _imported_ir_modules(path)
    }
    outside = importers - IR_FAMILY_PATHS - WIRING_SITES
    assert outside == set(), sorted(outside)


def test_the_wiring_sites_are_exactly_the_declared_ones():
    """The allowlist is a statement about the tree, not a standing licence.

    A file that stops importing the family has to leave this set, or the next
    module to take that path would inherit an exemption nobody granted it.
    """
    for site in sorted(WIRING_SITES):
        path = REPO_ROOT / site
        assert path.exists(), site
        assert _imported_ir_modules(path), site


def test_the_wiring_is_opt_in_and_lazy():
    """The one property that keeps the wiring reversible.

    The import sits inside the function that the switch guards, so importing
    the viewer never imports the IR, and an unset environment leaves the ACP
    path untouched.
    """
    viewer_path = REPO_ROOT / "src/benchflow/trajectories/viewer/legacy.py"
    tree = ast.parse(viewer_path.read_text(encoding="utf-8"))
    module_level = {
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _imported_ir_modules_from_node(node)
    }
    assert module_level == set(), "the viewer must not import the IR at module level"


def test_every_ir_family_module_exists():
    """A stale name in :data:`IR_FAMILY` would silently weaken the test above."""
    missing = [path for path in IR_FAMILY_PATHS if not (REPO_ROOT / path).exists()]
    assert missing == [], missing


def test_the_ir_module_imports_no_benchflow_runtime_module():
    """The hub must not depend on the formats it is supposed to be neutral about."""
    tree = ast.parse(Path(ir_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not [name for name in imported if name.startswith("benchflow")], sorted(
        imported
    )


# ---------------------------------------------------------------------------
# The worked example, shared with the documentation
# ---------------------------------------------------------------------------


def _example_trace() -> CanonicalTrace:
    """The three-event trace documented in ``docs/trace-interop.md`` §8.4.

    Chosen to exercise the parts of the design that are actually load-bearing:
    a tool call whose arguments are unavailable (declared, not silent), a
    thought whose boundaries are kept alongside the joined form, a non-text
    content block carried as opaque, and a timeout whose source-specific fields
    ride in ``extensions`` instead of growing the IR a field each.
    """
    losses = LossReport(direction="acp->ir")
    losses.add(
        "events[1].tool_call.arguments",
        LossClass.UNSUPPORTED,
        "ACPSession.handle_update reads five fields and rawInput is not one of "
        "them, so no ACP-derived tool call carries arguments.",
        "§5 loss #1",
    )
    losses.add(
        "events[1].tool_call.started_at",
        LossClass.UNSUPPORTED,
        "ToolCallRecord tracks started_at/finished_at in memory and "
        "_events_to_trajectory serializes neither.",
        "§5 loss #3",
    )

    return CanonicalTrace(
        session_id="rollout-7f3a",
        agent=ModelInfo(agent_name="gemini", model="gemini-2.5-flash"),
        provenance=ACP_PROVENANCE,
        events=[
            TraceEvent(
                index=0,
                kind=EventKind.USER_MESSAGE,
                source_type="user_message",
                role=Role.USER,
                text="Count the rows in data.csv",
                provenance=ACP_PROVENANCE,
            ),
            TraceEvent(
                index=1,
                kind=EventKind.TOOL_CALL,
                source_type="tool_call",
                role=Role.AGENT,
                reasoning="Check the file first.\n\nThen count.",
                reasoning_segments=["Check the file first.", "Then count."],
                tool_call=ToolCall(
                    call_id="tc_1",
                    name="execute",
                    name_semantics="acp_kind",
                    title="wc -l data.csv",
                    status=ToolStatus.COMPLETED,
                    arguments=None,
                    content=[
                        ContentBlock(
                            kind=ContentBlockKind.TEXT,
                            text="42 data.csv",
                            raw={
                                "type": "content",
                                "content": {"type": "text", "text": "42 data.csv"},
                            },
                        ),
                        ContentBlock(
                            kind=ContentBlockKind.OPAQUE,
                            raw={
                                "type": "diff",
                                "path": "/w/data.csv",
                                "oldText": "a",
                                "newText": "b",
                            },
                        ),
                    ],
                ),
                provenance=ACP_PROVENANCE,
            ),
            TraceEvent(
                index=2,
                kind=EventKind.TIMEOUT,
                source_type="agent_timeout",
                outcome="wall_clock_timeout",
                extensions={
                    "timeout_sec": 90.0,
                    "pending_tool_call_ids": [],
                    "terminal_trajectory_complete": True,
                },
                provenance=ACP_PROVENANCE,
            ),
        ],
        usage=TraceUsage(
            input_tokens=1180,
            output_tokens=96,
            total_tokens=1276,
            source="llm_proxy_normalized",
        ),
        outcome=TraceOutcome(status=OutcomeStatus.TIMEOUT),
        losses=losses,
    )


def test_the_example_trace_satisfies_every_invariant():
    assert validate_trace(_example_trace()) == []


def test_the_example_demonstrates_what_the_ir_is_for():
    """Read as a spec: these four properties are the reason the module exists."""
    trace = _example_trace()
    tool_event = trace.events[1]

    # 1. An unavailable value is absent *and* declared.
    assert tool_event.tool_call.arguments is None
    assert trace.losses.for_field("events[1].tool_call.arguments")

    # 2. A non-text block is carried instead of skipped (§5 loss #5).
    opaque = [
        b for b in tool_event.tool_call.content if b.kind is ContentBlockKind.OPAQUE
    ]
    assert opaque and opaque[0].raw["newText"] == "b"

    # 3. Thought boundaries survive the join (§5 loss #10).
    assert tool_event.reasoning_segments == ["Check the file first.", "Then count."]
    assert "\n\n".join(tool_event.reasoning_segments) == tool_event.reasoning

    # 4. The timeout is representable at all (§5 loss #4).
    assert trace.events[2].kind is EventKind.TIMEOUT
    assert trace.events[2].extensions["timeout_sec"] == 90.0


def resolve_ir_path(document: dict, path: str) -> tuple[bool, Any]:
    """Walk a `LossRecord.field` path through a serialized trace.

    Returns ``(resolved, value)``. ``resolved`` is False as soon as a segment
    names a key the document does not contain — which is the whole point: a
    loss record that addresses a missing key is a declaration nobody can check.

    Shared with the Slice C suite, so both directions of the hub check the same
    property with the same walker.
    """
    node: Any = document
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            index = int(part[1:-1])
            if not isinstance(node, list) or index >= len(node):
                return False, None
            node = node[index]
        else:
            if not isinstance(node, dict) or part not in node:
                return False, None
            node = node[part]
    return True, node


def test_every_concrete_loss_path_resolves_in_the_canonical_encoding():
    """The canonical encoding keeps every declared absence addressable.

    Guards a real defect: the §8.4 example was published with
    ``exclude_none=True``, which drops ``arguments`` entirely while the loss
    report kept pointing at it — the declaration that makes the absence legal
    addressed a key no reader of that document could find.

    The second half is what makes this a guard rather than a tautology: it
    asserts the discarded encoding *fails*, so the test cannot pass for both.
    """
    trace = _example_trace()
    canonical = trace.model_dump(mode="json")

    concrete = [
        record.field
        for record in trace.losses.records
        if record.space is PathSpace.HUB
        and record.field.startswith("events[")
        and not record.field.startswith("events[]")
    ]
    assert concrete, "the example must declare at least one concrete-path loss"

    unresolved = [
        field for field in concrete if not resolve_ir_path(canonical, field)[0]
    ]
    assert unresolved == [], unresolved

    # Every one of them addresses a field that is present and explicitly null —
    # the positive statement "the source did not carry this".
    for field in concrete:
        resolved, value = resolve_ir_path(canonical, field)
        assert resolved and value is None, (field, value)

    # And the encoding this replaced does not resolve them.
    lean = trace.model_dump(mode="json", exclude_none=True)
    assert all(not resolve_ir_path(lean, field)[0] for field in concrete), (
        "exclude_none must not resolve these paths; if it does, this guard has "
        "stopped discriminating between the two encodings"
    )


def test_the_canonical_encoding_round_trips_like_the_model():
    """Keeping the nulls costs nothing in fidelity — it only adds legibility."""
    trace = _example_trace()
    canonical = trace.model_dump(mode="json")
    restored = CanonicalTrace.model_validate(json.loads(json.dumps(canonical)))
    assert restored == trace
    assert validate_trace(restored) == []


def _documented_example() -> dict:
    """The JSON block tagged ``<!-- ir-example -->`` in the interop document."""
    text = DOCS_PATH.read_text(encoding="utf-8")
    match = re.search(r"<!-- ir-example -->\s*```json\n(.*?)\n```", text, re.S)
    assert match, (
        "docs/trace-interop.md no longer contains an <!-- ir-example --> JSON "
        "block. The documented example is generated from the models; regenerate "
        "it rather than deleting this check."
    )
    return json.loads(match.group(1))


def test_the_documented_example_matches_the_models():
    """A doc example that cannot drift, because the drift fails a test.

    Serialized in the canonical encoding — nulls retained. The example is the
    one place a reviewer sees what a trace looks like, so it must not be the
    one place that shows an encoding the IR does not accept.
    """
    produced = _example_trace().model_dump(mode="json")
    assert produced == _documented_example()
