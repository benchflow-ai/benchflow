"""`viewer trace steps → page`: what reaches the page, and what refuses to.

The positive half is easy to state — six step kinds, six cards, nothing
skipped. The half worth having is negative: a tool name never acquires a
category from its spelling, a diagnostic body never passes for a source
record, a cut is never silent, and the legacy renderer keeps emitting exactly
what it emitted before (pinned next door in
``test_viewer_primitives.py``).
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from benchflow.trajectories import viewer
from benchflow.trajectories.ir import (
    CanonicalTrace,
    ContentBlock,
    ContentBlockKind,
    EventKind,
    LossClass,
    PathSpace,
    Provenance,
    ToolCall,
    ToolStatus,
    TraceEvent,
    validate_trace,
)
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_from_atif import atif_to_ir
from benchflow.trajectories.ir_from_otel import otlp_json_to_ir
from benchflow.trajectories.ir_to_atif import ir_to_atif
from benchflow.trajectories.ir_to_view import (
    ACP_KIND_SEMANTICS,
    VIEW_TOOL_HUES,
    ir_to_view_steps,
)
from benchflow.trajectories.ir_to_view_html import (
    DIAGNOSTIC_LABEL,
    HUE_ACCENT,
    LOSS_DIRECTION,
    NEUTRAL_ACCENT,
    TOOL_OUTPUT_PREVIEW,
    render_trace,
    view_steps_to_html,
)
from tests.trajectories.test_atif_preservation import _rich_events
from tests.trajectories.test_ir_from_otel import PRODUCER_PAYLOAD_JSON

EVIDENCE = pathlib.Path(__file__).resolve().parents[2].parent / "e2e-a2" / "evidence"

_PROV = Provenance(source_format="hand-built")

H1_EVENTS = [
    {"type": "user_message", "text": "count the lines in /etc/hostname"},
    {"type": "agent_thought", "text": "[cwd /app] reading the file"},
    {
        "type": "tool_call",
        "tool_call_id": "call_1317590",
        "kind": "execute",
        "title": "wc -l < /etc/hostname",
        "status": "completed",
        "content": [
            {"type": "content", "content": {"type": "text", "text": "TOOL-OUTPUT-1"}}
        ],
    },
    {
        "type": "tool_call",
        "tool_call_id": "call_832088",
        "kind": "read",
        "title": "answer.txt",
        "status": "completed",
        "content": [{"type": "content", "content": {"type": "text", "text": "1\n"}}],
    },
    {"type": "agent_message", "text": "answer.txt holds 1"},
]

H2_EVENTS = [
    {"type": "user_message", "text": "run both commands"},
    {
        "type": "tool_call",
        "tool_call_id": "call_1131058",
        "kind": "think",
        "title": "Update topic",
        "status": "completed",
        "content": [],
    },
    {"type": "agent_thought", "text": "[cwd /app] sleeping"},
    {
        "type": "agent_timeout",
        "reason": "wall_clock_timeout",
        "timeout_sec": 90.0,
        "pending_tool_call_ids": ["call_1131058"],
        "terminal_trajectory_complete": True,
    },
]


def _trace(*events: TraceEvent) -> CanonicalTrace:
    return CanonicalTrace(provenance=_PROV, events=list(events))


def _tool_event(index: int = 0, **call) -> TraceEvent:
    call.setdefault("title", "")
    call.setdefault("status", ToolStatus.COMPLETED)
    return TraceEvent(
        index=index,
        kind=EventKind.TOOL_CALL,
        provenance=_PROV,
        tool_call=ToolCall(**call),
    )


def _page(trace: CanonicalTrace, **kwargs) -> str:
    return render_trace("t", trace, **kwargs).html


def _accents(page: str) -> list[str]:
    import re

    return re.findall(r'<div class="step agent tool-step (acc-[a-z]+)"', page)


def _semantics(page: str) -> list[str]:
    import re

    return re.findall(r'data-name-semantics="([^"]*)"', page)


def _captured(name: str) -> list[dict]:
    path = EVIDENCE / name / "acp_trajectory.jsonl"
    if not path.is_file():
        pytest.skip(f"captured rollout {name!r} is not in this tree")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# The classification rule
# ---------------------------------------------------------------------------


def test_the_accent_table_is_total_over_the_display_vocabulary():
    assert set(HUE_ACCENT) == set(VIEW_TOOL_HUES)


def test_every_accent_it_can_emit_is_defined_in_the_stylesheet():
    for accent in set(HUE_ACCENT.values()) | {NEUTRAL_ACCENT}:
        assert f".{accent}" in viewer._VIEWER_CSS, accent


def test_the_module_never_reaches_for_the_substring_classifier():
    """The one import that would undo the slice, asserted absent by AST."""
    from benchflow.trajectories import ir_to_view_html

    source = pathlib.Path(ir_to_view_html.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "_tool_accent_class" not in names
    assert "_TOOL_ACCENTS" not in names


def test_an_acp_category_gets_its_accent_and_a_function_name_does_not():
    """The same string, twice, from the same run — the slice in one test."""
    as_category = _trace(_tool_event(name="execute", name_semantics=ACP_KIND_SEMANTICS))
    as_function = _trace(_tool_event(name="execute", name_semantics="function_name"))

    assert _accents(_page(as_category)) == ["acc-bash"]
    assert _accents(_page(as_function)) == [NEUTRAL_ACCENT]
    assert "execute" in _page(as_function), "the name is still shown, only uncoloured"


def test_a_span_name_never_acquires_a_category_from_its_spelling():
    trace = _trace(_tool_event(name="read_file", name_semantics="gen_ai.tool.name"))
    assert _accents(_page(trace)) == [NEUTRAL_ACCENT]
    # The classifier the legacy card uses would have said otherwise.
    assert viewer._tool_accent_class("read_file") == "acc-read"


def test_the_title_is_never_consulted_either():
    trace = _trace(
        _tool_event(
            name="mystery", name_semantics="function_name", title="bash -lc 'ls -la'"
        )
    )
    assert _accents(_page(trace)) == [NEUTRAL_ACCENT]
    assert viewer._tool_accent_class("bash -lc 'ls -la'") == "acc-bash"


def test_an_observed_category_with_no_accent_is_declared_not_hidden():
    """`think` is a real ACP kind and the stylesheet has no strip for it."""
    trace = _trace(_tool_event(name="think", name_semantics=ACP_KIND_SEMANTICS))
    rendered = render_trace("t", trace)
    assert _accents(rendered.html) == [NEUTRAL_ACCENT]
    assert "think" in rendered.html
    declared = [
        r
        for r in rendered.page_losses.records
        if r.field.endswith(".tool.hue") and r.loss_class is LossClass.DROPPED
    ]
    assert declared, rendered.page_losses.records


def test_name_semantics_is_observable_on_every_tool_card():
    trace = acp_events_to_ir(H1_EVENTS)
    page = _page(trace)
    assert _semantics(page) == [ACP_KIND_SEMANTICS, ACP_KIND_SEMANTICS]
    assert page.count("name_semantics: acp_kind") == 2


# ---------------------------------------------------------------------------
# The four corpora
# ---------------------------------------------------------------------------


def test_acp_h1_renders_one_card_per_event():
    trace = acp_events_to_ir(H1_EVENTS)
    page = _page(trace)
    assert page.count('<div class="step ') == len(H1_EVENTS)
    assert _accents(page) == ["acc-bash", "acc-read"]
    assert "answer.txt holds 1" in page


def test_acp_h2_puts_the_timeout_on_the_page():
    """The legacy renderer has no branch for it; three blocks become four."""
    trace = acp_events_to_ir(H2_EVENTS)
    page = _page(trace)

    legacy = viewer._render_acp_events("t", H2_EVENTS, None, None)
    assert "timeout" not in legacy.lower()

    assert "agent timeout" in page
    assert "wall_clock_timeout" in page
    assert "timeout_sec: 90.0" in page
    assert "call_1131058" in page
    assert "terminal trajectory complete: True" in page
    assert page.count('<div class="step ') == len(H2_EVENTS)


@pytest.mark.parametrize("name", ["h1", "h2"])
def test_a_captured_rollout_renders(name):
    events = _captured(name)
    trace = acp_events_to_ir(events)
    page = _page(trace)
    assert page.count('<div class="step ') == len(trace.events)
    for semantics in _semantics(page):
        assert semantics == ACP_KIND_SEMANTICS


def test_the_atif_document_of_the_same_run_stays_neutral():
    """ATIF puts an ACP kind in a `function_name` slot; the accent must not."""
    document, _ = ir_to_atif(acp_events_to_ir(H1_EVENTS))
    trace = atif_to_ir(document)
    page = _page(trace)

    assert set(_semantics(page)) == {"function_name"}
    assert set(_accents(page)) == {NEUTRAL_ACCENT}
    assert "execute" in page and "read" in page


def test_otlp_spans_render_and_keep_their_names_uncoloured():
    traces, _ = otlp_json_to_ir(json.loads(PRODUCER_PAYLOAD_JSON))
    assert traces
    for trace in traces:
        rendered = render_trace("t", trace)
        page = rendered.html
        assert page.count('<div class="step ') == len(trace.events)
        for semantics in _semantics(page):
            assert semantics == "gen_ai.tool.name"
        assert set(_accents(page)) <= {NEUTRAL_ACCENT}
        assert "read_file" in page


# ---------------------------------------------------------------------------
# Nothing vanishes
# ---------------------------------------------------------------------------


def test_no_step_is_skipped_whatever_the_kind():
    trace = _trace(
        TraceEvent(
            index=0,
            kind=EventKind.UNKNOWN,
            provenance=_PROV,
            source_type="session_meta",
        ),
        TraceEvent(index=1, kind=EventKind.ORACLE, provenance=_PROV),
        TraceEvent(
            index=2, kind=EventKind.AGENT_MESSAGE, provenance=_PROV, text="done"
        ),
        _tool_event(3, name="execute", name_semantics=ACP_KIND_SEMANTICS),
    )
    steps, _ = ir_to_view_steps(trace)
    page, _ = view_steps_to_html("t", steps)
    assert page.count('<div class="step ') == len(steps) == len(trace.events)


def test_unknown_and_oracle_are_visible_and_named():
    trace = _trace(
        TraceEvent(
            index=0,
            kind=EventKind.UNKNOWN,
            provenance=_PROV,
            source_type="session_meta",
        ),
        TraceEvent(index=1, kind=EventKind.ORACLE, provenance=_PROV),
    )
    page = _page(trace)
    assert page.count(f">{DIAGNOSTIC_LABEL}</span>") == 2
    assert "session_meta" in page
    assert "oracle" in page
    assert page.count('data-diagnostic="canonical-ir"') == 2


def test_an_unrecognized_acp_record_reaches_the_page_the_legacy_card_drops():
    events = [*H2_EVENTS, {"type": "reward", "value": 1.0}]
    page = _page(acp_events_to_ir(events))
    legacy = viewer._render_acp_events("t", events, None, None)

    assert "reward" not in legacy
    assert "reward" in page
    assert DIAGNOSTIC_LABEL in page


def test_the_diagnostic_body_is_the_canonical_event_and_says_so():
    event = TraceEvent(
        index=0, kind=EventKind.UNKNOWN, provenance=_PROV, source_type="session_meta"
    )
    page = _page(_trace(event))

    assert DIAGNOSTIC_LABEL in page
    # The body is the canonical event, not a source record: its keys are the
    # IR's, and a reader is told which document they are looking at.
    assert "&quot;kind&quot;: &quot;unknown&quot;" in page
    assert "&quot;provenance&quot;" in page


def test_tool_output_reaches_the_page_when_it_was_observed():
    trace = acp_events_to_ir(H1_EVENTS)
    page = _page(trace)
    assert "TOOL-OUTPUT-1" in page
    assert "TOOL-OUTPUT-1" not in viewer._render_acp_events("t", H1_EVENTS, None, None)


def test_a_block_with_no_text_contributes_nothing_rather_than_a_placeholder():
    trace = _trace(
        _tool_event(
            name="execute",
            name_semantics=ACP_KIND_SEMANTICS,
            content=[ContentBlock(kind=ContentBlockKind.OPAQUE, raw={"blob": "x"})],
        )
    )
    page = _page(trace)
    assert "blob" not in page
    assert '<div class="tool-args">' not in page


# ---------------------------------------------------------------------------
# Cuts, escaping, prompts
# ---------------------------------------------------------------------------


def test_a_cut_is_announced_on_the_page_and_in_the_report():
    long_output = "x" * (TOOL_OUTPUT_PREVIEW + 25)
    trace = _trace(
        _tool_event(
            name="execute",
            name_semantics=ACP_KIND_SEMANTICS,
            content=[ContentBlock(kind=ContentBlockKind.TEXT, text=long_output)],
        )
    )
    rendered = render_trace("t", trace)
    assert "[truncated, 25 more characters]" in rendered.html
    cuts = [
        r for r in rendered.page_losses.records if r.loss_class is LossClass.NORMALIZED
    ]
    assert any("truncated" in r.detail or "does not show" in r.detail for r in cuts)


def test_trajectory_text_is_escaped_everywhere_it_lands():
    payload = "<script>window.__PWNED__=1</script>"
    trace = _trace(
        TraceEvent(
            index=0, kind=EventKind.AGENT_MESSAGE, provenance=_PROV, text=payload
        ),
        _tool_event(
            1,
            name="execute",
            name_semantics=ACP_KIND_SEMANTICS,
            title=payload,
            content=[ContentBlock(kind=ContentBlockKind.TEXT, text=payload)],
        ),
        TraceEvent(
            index=2, kind=EventKind.UNKNOWN, provenance=_PROV, source_type=payload
        ),
    )
    page = _page(trace)
    assert "<script>" not in page
    assert page.count("&lt;script&gt;") >= 4


def test_prompts_json_is_shown_only_when_the_steps_carry_no_prompt():
    """The legacy renderer's own rule, applied to a step list."""
    with_prompt = acp_events_to_ir(H1_EVENTS)
    page = _page(with_prompt, prompts=["from prompts.json"])
    assert "from prompts.json" not in page
    assert page.count("PROMPT 1") == 1

    without = _trace(
        TraceEvent(index=0, kind=EventKind.AGENT_MESSAGE, provenance=_PROV, text="ok")
    )
    page = _page(without, prompts=["first", "second"])
    assert "PROMPT 1" in page and "PROMPT 2" in page
    assert "first" in page and "second" in page


def test_the_run_summary_comes_from_result_json_unchanged():
    result = {
        "agent_name": "gemini-cli",
        "rewards": {"reward": 1.0},
        "n_tool_calls": 2,
        "n_prompts": 1,
    }
    page = _page(acp_events_to_ir(H1_EVENTS), result_data=result)
    assert viewer._result_block(result) in page


# ---------------------------------------------------------------------------
# Report hygiene
# ---------------------------------------------------------------------------


def test_the_two_reports_are_separate_and_neither_touches_the_trace():
    trace = acp_events_to_ir(_rich_events())
    before = trace.model_dump_json()
    inbound = len(trace.losses.records) if trace.losses else 0

    rendered = render_trace("t", trace)

    assert trace.model_dump_json() == before
    assert (len(trace.losses.records) if trace.losses else 0) == inbound
    assert rendered.steps_losses is not rendered.page_losses
    assert rendered.page_losses.direction == LOSS_DIRECTION
    assert rendered.steps_losses.direction == "ir->view"
    assert validate_trace(trace) == []


def test_this_edge_never_addresses_the_hub():
    """It has the IR on neither side, so a hub path would be unresolvable."""
    rendered = render_trace("t", acp_events_to_ir(H1_EVENTS))
    assert all(r.space is not PathSpace.HUB for r in rendered.page_losses.records)


def test_an_empty_step_list_still_produces_a_page_and_a_claim():
    page, losses = view_steps_to_html("t", [])
    assert page.startswith("<!DOCTYPE html>")
    assert losses.records == [], "nothing was lost because nothing was given"


def test_the_page_names_the_renderer_that_produced_it():
    rendered = render_trace("t", acp_events_to_ir(H1_EVENTS))
    assert "Rendered from the canonical Trace IR" in rendered.html
    assert any(
        r.field == "provenance" and r.loss_class is LossClass.SYNTHESIZED
        for r in rendered.page_losses.records
    )
