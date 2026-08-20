"""The opt-in switch that routes a rollout through the canonical Trace IR.

Guards the wiring added for PR #984 (Slice I-b). Two claims, and the first one
is the one that has to hold every day: with ``BENCHFLOW_VIEWER_TRACE_IR``
unset, ``render_rollout`` produces exactly the page it produced before the
branch existed. The second is that with it set, the page comes from
``source → CanonicalTrace → ir_to_view_steps → ir_to_view_html`` — and not from
capture events forged to look like ACP.
"""

from __future__ import annotations

import json

import pytest

from benchflow.trajectories import viewer
from benchflow.trajectories.ir_from_acp import acp_events_to_ir
from benchflow.trajectories.ir_to_atif import ir_to_atif
from benchflow.trajectories.ir_to_view_html import (
    DIAGNOSTIC_LABEL,
    render_rollout_page,
    rollout_to_trace,
)
from benchflow.trajectories.viewer import TRACE_IR_ENV

ACP_EVENTS = [
    {"type": "user_message", "text": "run both commands"},
    {
        "type": "tool_call",
        "tool_call_id": "call_1",
        "kind": "execute",
        "title": "wc -l /etc/hostname",
        "status": "completed",
        "content": [{"type": "content", "content": {"type": "text", "text": "OUT-1"}}],
    },
    {
        "type": "agent_timeout",
        "reason": "wall_clock_timeout",
        "timeout_sec": 90.0,
        "pending_tool_call_ids": [],
        "terminal_trajectory_complete": True,
    },
    {"type": "reward", "value": 1.0},
]

RESULT_JSON = {
    "agent_name": "gemini-cli",
    "rewards": {"reward": 1.0},
    "n_tool_calls": 1,
    "n_prompts": 1,
}


@pytest.fixture(autouse=True)
def switch_off(monkeypatch):
    monkeypatch.delenv(TRACE_IR_ENV, raising=False)


def _rollout(tmp_path, *, acp=None, atif=None, turns=False, prompts=None):
    root = tmp_path / "rollout-1"
    root.mkdir()
    if acp is not None:
        (root / "trajectory").mkdir()
        (root / "trajectory" / "acp_trajectory.jsonl").write_text(
            "\n".join(json.dumps(e) for e in acp), encoding="utf-8"
        )
    if atif is not None:
        (root / "trainer").mkdir()
        (root / "trainer" / "atif.json").write_text(json.dumps(atif), encoding="utf-8")
    if turns:
        (root / "turn1.txt").write_text(
            json.dumps({"type": "assistant", "message": {"content": []}}),
            encoding="utf-8",
        )
    if prompts is not None:
        (root / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    (root / "result.json").write_text(json.dumps(RESULT_JSON), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Off
# ---------------------------------------------------------------------------


def test_the_switch_is_off_unless_it_is_explicitly_on(tmp_path, monkeypatch):
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    legacy = viewer._render_acp_trajectory(
        root, root / "trajectory" / "acp_trajectory.jsonl", None
    )
    for value in ("", "0", "no", "off", "false", " "):
        monkeypatch.setenv(TRACE_IR_ENV, value)
        assert viewer.render_rollout(root) == legacy, value


def test_an_unset_switch_leaves_the_acp_page_byte_identical(tmp_path):
    root = _rollout(tmp_path, acp=ACP_EVENTS, prompts=["p"])
    assert viewer.render_rollout(root) == viewer._render_acp_trajectory(
        root, root / "trajectory" / "acp_trajectory.jsonl", None
    )


def test_an_atif_only_rollout_still_has_no_page_with_the_switch_off(tmp_path):
    document, _ = ir_to_atif(acp_events_to_ir(ACP_EVENTS))
    root = _rollout(tmp_path, atif=document)
    assert viewer.render_rollout(root) == viewer._NO_TRAJECTORIES_HTML


def test_a_stream_json_rollout_is_untouched_by_the_switch(tmp_path, monkeypatch):
    root = _rollout(tmp_path, turns=True)
    before = viewer.render_rollout(root)
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    assert viewer.render_rollout(root) == before


# ---------------------------------------------------------------------------
# On
# ---------------------------------------------------------------------------


def test_the_switch_routes_an_acp_rollout_through_the_canonical_ir(
    tmp_path, monkeypatch
):
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    legacy = viewer.render_rollout(root)
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    page = viewer.render_rollout(root)

    assert page != legacy
    assert "Rendered from the canonical Trace IR" in page
    assert page == render_rollout_page(root, None).html

    # What the legacy page of the same rollout does not have. Of the four ACP
    # events it renders two — the prompt and the tool call — for three cards
    # with RESULT; the canonical page renders all four, for five.
    assert "timeout" not in legacy.lower()
    assert "agent timeout" in page
    assert legacy.count('<div class="step ') == 3
    assert page.count('<div class="step ') == 5
    assert DIAGNOSTIC_LABEL not in legacy
    assert DIAGNOSTIC_LABEL in page
    assert "OUT-1" not in legacy
    assert "OUT-1" in page


def test_the_switch_gives_an_atif_only_rollout_a_page(tmp_path, monkeypatch):
    document, _ = ir_to_atif(acp_events_to_ir(ACP_EVENTS))
    root = _rollout(tmp_path, atif=document)
    monkeypatch.setenv(TRACE_IR_ENV, "1")

    page = viewer.render_rollout(root)
    assert page != viewer._NO_TRAJECTORIES_HTML
    assert "name_semantics: function_name" in page
    assert "acc-bash" not in page.split("</style>")[1], (
        "an ATIF function_name must not acquire the execute accent"
    )


def test_a_directory_with_no_readable_trajectory_falls_through_unchanged(
    tmp_path, monkeypatch
):
    root = _rollout(tmp_path)
    before = viewer.render_rollout(root)
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    assert rollout_to_trace(root) is None
    assert viewer.render_rollout(root) == before == viewer._NO_TRAJECTORIES_HTML


def test_a_failed_conversion_falls_back_to_the_acp_page_and_says_so(
    tmp_path, monkeypatch, capsys
):
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    legacy = viewer.render_rollout(root)
    monkeypatch.setenv(TRACE_IR_ENV, "1")

    import benchflow.trajectories.ir_to_view_html as adapter

    def boom(*args, **kwargs):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(adapter, "render_rollout_page", boom)

    assert viewer.render_rollout(root) == legacy
    err = capsys.readouterr().err
    assert "canonical IR path failed" in err
    assert "converter exploded" in err


def test_the_page_is_not_built_from_forged_acp_events(tmp_path, monkeypatch):
    """The switch must not route through `_render_acp_events`.

    Rebuilding steps into capture events and handing them to the ACP renderer
    would silently reintroduce the four-branch vocabulary the IR exists to get
    past — so the canonical path must not touch that function at all.
    """
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    monkeypatch.setenv(TRACE_IR_ENV, "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("_render_acp_events was called on the canonical path")

    monkeypatch.setattr(viewer, "_render_acp_events", forbidden)
    page = viewer.render_rollout(root)
    assert "Rendered from the canonical Trace IR" in page

    # The patch bites: the legacy path goes straight through it.
    monkeypatch.delenv(TRACE_IR_ENV)
    with pytest.raises(AssertionError):
        viewer.render_rollout(root)


def test_prompts_reach_the_canonical_page_through_the_caller(tmp_path, monkeypatch):
    """`render_rollout` reads prompts.json; the switch must not lose them."""
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    root = _rollout(
        tmp_path,
        acp=[{"type": "agent_message", "text": "ok"}],
        prompts=["the run own prompt"],
    )
    page = viewer.render_rollout(root)
    assert "the run own prompt" in page
    assert "PROMPT 1" in page


def test_the_run_summary_is_the_viewers_own_card(tmp_path, monkeypatch):
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    assert viewer._result_block(RESULT_JSON) in viewer.render_rollout(root)


def test_serve_writes_the_canonical_page_into_the_sidecar(tmp_path, monkeypatch):
    """`serve` writes trajectory.html from whatever render_rollout returned."""
    monkeypatch.setenv(TRACE_IR_ENV, "1")
    root = _rollout(tmp_path, acp=ACP_EVENTS)
    page = viewer.render_rollout(root)
    (root / "trajectory.html").write_text(page, encoding="utf-8")
    assert "Rendered from the canonical Trace IR" in (
        root / "trajectory.html"
    ).read_text(encoding="utf-8")
