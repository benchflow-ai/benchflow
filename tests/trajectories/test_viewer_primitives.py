"""The viewer's page primitives, frozen at the markup they were extracted with.

Guards the pure refactor in PR #984 (Slice I): ``_page``, ``_prompt_block``,
``_message_block``, ``_thought_block`` and ``_result_block`` were lifted out of
``_render_acp_events`` so a second producer of blocks can emit the same page.
The refactor claimed byte-identical output; these tests are what keeps that
claim true, by pinning each card's markup and the assembled body of a whole
ACP page.

They also pin one thing the refactor deliberately did **not** repair: the two
prompt sites truncate and escape in opposite orders. Fixing that is a change to
the legacy renderer's output and belongs to its own commit, not to an
extraction.
"""

from __future__ import annotations

import json

from benchflow.trajectories.viewer import (
    _message_block,
    _page,
    _prompt_block,
    _render_acp_events,
    _result_block,
    _thought_block,
)

ACP_EVENTS = [
    {"type": "user_message", "text": "count the lines"},
    {"type": "agent_thought", "text": "reading the file"},
    {
        "type": "tool_call",
        "tool_call_id": "call_1",
        "kind": "execute",
        "title": "wc -l /etc/hostname",
        "status": "completed",
        "content": [{"type": "content", "content": {"type": "text", "text": "1"}}],
    },
    {"type": "agent_message", "text": "the answer is 1"},
]

RESULT_DATA = {
    "agent_name": "gemini-cli",
    "rewards": {"reward": 1.0},
    "n_tool_calls": 1,
    "n_prompts": 1,
}

# The exact body the current renderer emits for ACP_EVENTS + RESULT_DATA —
# everything between the header and </body>.
FROZEN_BODY = (
    '<div class="step prompt">'
    '<div class="step-header"><span class="label prompt">PROMPT 1</span></div>'
    '<div class="msg">count the lines</div>'
    "</div>"
    '<div class="step agent"><div class="thinking">reading the file</div></div>'
    '<div class="step agent tool-step acc-bash">'
    '<div class="tool"><span class="tool-name">execute</span> wc -l /etc/hostname</div>'
    '<div class="metrics">completed</div>'
    "</div>"
    '<div class="step agent"><div class="msg">the answer is 1</div></div>'
    '<div class="step result">'
    '<div class="step-header"><span class="label result">RESULT</span></div>'
    "<div class=\"msg\">Agent: gemini-cli | Rewards: {'reward': 1.0} | "
    "Tool calls: 1 | Prompts: 1</div>"
    "</div>"
)


def _body(page: str) -> str:
    """The block region of a rendered page."""
    return page.split("</div>\n", 1)[1].rsplit("\n</body>", 1)[0]


def test_the_acp_page_body_is_frozen():
    page = _render_acp_events("t", ACP_EVENTS, RESULT_DATA, None)
    assert _body(page) == FROZEN_BODY


def test_each_card_is_the_markup_it_was_extracted_with():
    assert _prompt_block("PROMPT 1", "hi") == (
        '<div class="step prompt">'
        '<div class="step-header"><span class="label prompt">PROMPT 1</span></div>'
        '<div class="msg">hi</div>'
        "</div>"
    )
    assert (
        _message_block("hi")
        == '<div class="step agent"><div class="msg">hi</div></div>'
    )
    assert _thought_block("hi") == (
        '<div class="step agent"><div class="thinking">hi</div></div>'
    )
    assert _result_block(RESULT_DATA) == (
        '<div class="step result">'
        '<div class="step-header"><span class="label result">RESULT</span></div>'
        "<div class=\"msg\">Agent: gemini-cli | Rewards: {'reward': 1.0} | "
        "Tool calls: 1 | Prompts: 1</div>"
        "</div>"
    )


def test_the_page_shell_carries_the_stylesheet_and_the_escaped_title():
    page = _page("a & b", ['<div class="step"></div>'])
    assert page.startswith("<!DOCTYPE html>")
    assert "<title>benchflow — a &amp; b</title>" in page
    assert "<h1>a &amp; b</h1>" in page
    assert ".acc-bash" in page, "the one stylesheet ships with every page"
    assert page.rstrip().endswith("</body></html>")
    assert '<div class="step"></div>' in page


def test_the_block_builders_emit_what_they_are_given():
    """Escaping is the caller's job, at the site that knows what the value is.

    Pinned because it is what lets the two prompt sites keep their different
    truncation orders, and what lets a second producer truncate before
    escaping without changing anything here.
    """
    raw = "<script>x</script>"
    assert raw in _message_block(raw)
    assert raw in _prompt_block("L", raw)


def test_the_two_prompt_sites_still_truncate_in_opposite_orders():
    """Documented, not repaired, by the extraction.

    ``prompts.json`` entries are sliced then escaped; inline ``user_message``
    text is escaped then sliced, so a long prompt can lose its last entity to
    the 500-char cut. Changing either is a behaviour change and needs its own
    commit.
    """
    long_amp = "a" * 498 + "&&&"

    from_prompts_json = _render_acp_events("t", [], None, [long_amp])
    assert "&amp;" in from_prompts_json

    inline = _render_acp_events(
        "t", [{"type": "user_message", "text": long_amp}], None, None
    )
    body = _body(inline)
    assert "&am</div>" in body or "&a</div>" in body or "&</div>" in body, body[-80:]


def test_unknown_event_types_and_timeouts_are_absent_from_the_legacy_page():
    """The current renderer's four branches, stated as a fact rather than a bug.

    Slice I's adapter exists because of this line: an ``agent_timeout`` and an
    unrecognized record reach no card here. Pinned so the day someone adds a
    branch, the adapter's reason for existing is revisited too.
    """
    events = [
        {"type": "agent_timeout", "reason": "wall_clock_timeout", "timeout_sec": 90.0},
        {"type": "reward", "value": 1.0},
    ]
    body = _body(_render_acp_events("t", events, None, None))
    assert body == ""
    assert "timeout" not in _render_acp_events("t", events, None, None).lower()


def test_tool_output_text_is_absent_from_the_legacy_tool_card():
    """Also a fact, not a bug: the tool card carries kind, title and status."""
    marker = "MARKER-TOOL-OUTPUT"
    events = [
        {
            "type": "tool_call",
            "tool_call_id": "c1",
            "kind": "execute",
            "title": "echo",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"type": "text", "text": marker}}
            ],
        }
    ]
    assert marker not in _render_acp_events("t", events, None, None)
    assert json.dumps(events[0]["content"]) not in _render_acp_events(
        "t", events, None, None
    )
