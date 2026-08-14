"""The browser half of the trace viewer: ``viewer_assets/render.js``.

``render.js`` is a pure payload → HTML transform precisely so it can be driven
from here without a browser. The assertions name the vendored PostTrainBench
class hooks on purpose: those are the contract that makes the vendored
stylesheet apply, so a rename here is a silent visual regression. The published static site (slice 5) serves this
same file against a fetched payload, so a break here breaks both delivery
paths.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchflow.trajectories.viewer import PAYLOAD_SCHEMA_VERSION

_RENDER_JS = (
    Path(__file__).resolve().parents[1]
    / "src/benchflow/trajectories/viewer_assets/render.js"
)

_HARNESS = """
const fs = require("fs");
const BFViewer = require(process.argv[2]);
process.stdout.write(BFViewer.renderRunHtml(JSON.parse(fs.readFileSync(process.argv[3], "utf8"))));
"""

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to drive render.js"
)


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("viewer-js") / "harness.js"
    path.write_text(_HARNESS)
    return path


def render(harness: Path, tmp_path: Path, payload: dict[str, Any]) -> str:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    completed = subprocess.run(
        ["node", str(harness), str(_RENDER_JS), str(payload_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "name": "task__abc123",
        "source": "acp",
        "status": {"slug": "passed", "label": "Passed"},
        "reward": 1.0,
        "meta": {
            "task_name": "task",
            "agent_name": "Codex",
            "agent": "codex-acp",
            "model": "openai/gpt-5.6",
            "skill_mode": "with-skill",
            "n_tool_calls": 1,
            "n_skill_invocations": 2,
        },
        "usage": {
            "input": 1234,
            "output": 321,
            "cache_creation": 67,
            "cache_read": 890,
            "total": 2512,
            "cost_usd": 0.012345,
            "source": "provider_response",
            "price_source": "litellm",
        },
        "timing": {"environment_setup": 2.0, "agent_execution": 12.0, "total": 21.0},
        "notices": [],
        "turns": [],
        "artifacts": {"trajectory": "trajectory/acp_trajectory.jsonl"},
    }
    payload.update(overrides)
    return payload


def _turn(*events: dict[str, Any], number: int | None = 1) -> dict[str, Any]:
    return {"number": number, "events": list(events)}


def test_tool_observation_reaches_the_page_intact(
    harness: Path, tmp_path: Path
) -> None:
    """Slice 1's first contract, asserted on rendered output rather than data."""
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {
                        "type": "tool_call",
                        "kind": "bash",
                        "title": "pytest -q",
                        "status": "failed",
                        "tool_call_id": "tc-1",
                        "blocks": [
                            {"kind": "text", "text": "assert 1 == 2\nFAILED-NEEDLE"},
                            {
                                "kind": "diff",
                                "path": "/app/main.py",
                                "old": "value = 1",
                                "new": "value = 2",
                            },
                            {"kind": "binary"},
                        ],
                    }
                )
            ]
        ),
    )

    assert "pytest -q" in page
    assert "FAILED-NEEDLE" in page
    assert 'class="tool-call tool-bash"' in page
    assert "bf-tool-status-failed" in page
    # The observation must sit in upstream's .tool-result-body: that wrapper
    # carries the height cap the "expand outputs" control lifts.
    assert '<div class="tool-result-body error">' in page
    assert '<pre class="diff-remove">- value = 1</pre>' in page
    assert '<pre class="diff-add">+ value = 2</pre>' in page
    assert "/app/main.py" in page
    assert "[binary output omitted]" in page


def test_clipped_output_shows_both_ends_and_the_artifact_path(
    harness: Path, tmp_path: Path
) -> None:
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {
                        "type": "tool_call",
                        "kind": "bash",
                        "title": "pytest -q",
                        "status": "failed",
                        "tool_call_id": "tc-1",
                        "blocks": [
                            {
                                "kind": "text",
                                "text": "HEAD-NEEDLETAIL-NEEDLE",
                                "clip": {
                                    "dropped": 123456,
                                    "at": len("HEAD-NEEDLE"),
                                    "artifact": "trajectory/acp_trajectory.jsonl",
                                },
                            }
                        ],
                    }
                )
            ]
        ),
    )

    assert "HEAD-NEEDLE" in page
    assert "TAIL-NEEDLE" in page
    assert "123,456 characters truncated" in page
    assert "trajectory/acp_trajectory.jsonl" in page


def test_agent_messages_render_as_markdown(harness: Path, tmp_path: Path) -> None:
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {
                        "type": "agent_message",
                        "text": (
                            "## Summary\n\nFixed `main.py`.\n\n"
                            "- ran the suite\n- patched the bug\n\n"
                            "```python\nvalue = 2\n```\n"
                            "See [docs](https://example.com/x)."
                        ),
                    }
                )
            ]
        ),
    )

    assert "<h4>Summary</h4>" in page
    assert "<code>main.py</code>" in page
    assert "<li>ran the suite</li>" in page
    assert "<pre><code>value = 2</code></pre>" in page
    assert '<a href="https://example.com/x"' in page
    assert "## Summary" not in page
    # Prose lands in upstream's agent-text card, which styles the markdown.
    assert 'class="block-card agent-text"' in page


@pytest.mark.parametrize(
    "text",
    [
        "<img src=x onerror=alert(1)>",
        "[click](javascript:alert(1))",
        "`</code><script>alert(1)</script>`",
    ],
)
def test_markdown_never_emits_trace_controlled_markup(
    harness: Path, tmp_path: Path, text: str
) -> None:
    """A trace is untrusted input: an agent can write anything into a message."""
    page = render(
        harness,
        tmp_path,
        _payload(turns=[_turn({"type": "agent_message", "text": text})]),
    )

    # Scoped to the message card. The page shell has its own legitimate links,
    # so a page-wide "no anchors" assertion would pass for the wrong reason and
    # miss a link the trace injected.
    card = page.split('class="block-card agent-text">')[1].split("</article>")[0]

    assert "<img" not in card
    assert "<script>" not in card
    # A rejected link stays visible as inert text — the trace is evidence, so
    # nothing is deleted, only defanged.
    assert "<a " not in card
    assert "alert(1)" in card


def test_a_thought_with_no_text_is_marked_not_shown_as_an_empty_card(
    harness: Path, tmp_path: Path
) -> None:
    """@agentclientprotocol/claude-agent-acp sends agent_thought_chunk with
    content.text == "" — verified on the wire against v0.40.0. The event is
    evidence that reasoning happened at that point, so it is marked; an empty
    expandable "Thought" card would just read as a broken renderer."""
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {"type": "agent_thought", "text": ""},
                    {"type": "agent_thought", "text": "real reasoning here"},
                )
            ]
        ),
    )

    assert "bf-thought-empty" in page
    assert "the harness sent no text" in page
    # The populated thought still renders as a normal expandable card.
    assert "real reasoning here" in page
    assert page.count('class="block-card agent-thinking"') == 1


def test_terminal_output_keeps_color_and_drops_control_bytes(
    harness: Path, tmp_path: Path
) -> None:
    escape = "\x1b"
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {
                        "type": "tool_call",
                        "kind": "bash",
                        "title": "pytest",
                        "status": "completed",
                        "tool_call_id": "tc-1",
                        "blocks": [
                            {
                                "kind": "text",
                                "text": (
                                    f"{escape}[31mFAILED{escape}[0m ok"
                                    f"{escape}[2K{escape}[1Gspinner"
                                ),
                            }
                        ],
                    }
                )
            ]
        ),
    )

    assert '<span class="ansi-red">FAILED</span>' in page
    assert "spinner" in page
    assert escape not in page
    assert "[31m" not in page


def test_rail_reports_reward_tokens_timing_and_usage_provenance(
    harness: Path, tmp_path: Path
) -> None:
    page = render(harness, tmp_path, _payload())

    assert 'class="bf-status bf-status-passed"' in page
    assert (
        '<div class="score-big">1<span class="bf-score-unit">reward</span></div>'
        in page
    )
    assert '<div class="score-bar-fill" style="width:100%"></div>' in page
    assert "1,234" in page
    assert "$0.012345" in page
    assert "Skill invocations" in page
    assert "usage: provider_response · price: litellm" in page
    assert "Environment" in page
    assert "21.0s" in page
    assert "stream-json fallback" not in page
    # Upstream shell hooks — without these the vendored stylesheet has
    # nothing to attach to.
    assert '<aside class="rail rail-left">' in page
    assert 'class="card summary-card"' in page
    assert '<dl class="summary-stats">' in page
    assert 'class="layout bf-no-right-rail"' in page


@pytest.mark.parametrize(
    ("reward", "status", "needle"),
    [
        (
            0.0,
            {"slug": "failed", "label": "Failed"},
            '<div class="score-big">0<span class="bf-score-unit">reward</span></div>',
        ),
        (
            None,
            {"slug": "not-scored", "label": "Not scored"},
            '<div class="score-big bf-unscored">Not scored</div>',
        ),
    ],
)
def test_zero_reward_is_not_the_same_as_unscored(
    harness: Path,
    tmp_path: Path,
    reward: float | None,
    status: dict[str, str],
    needle: str,
) -> None:
    page = render(harness, tmp_path, _payload(reward=reward, status=status))

    assert needle in page
    assert f"bf-status-{status['slug']}" in page


def test_setup_and_turns_get_their_own_anchors(harness: Path, tmp_path: Path) -> None:
    page = render(
        harness,
        tmp_path,
        _payload(
            turns=[
                _turn(
                    {
                        "type": "oracle",
                        "title": "bash oracle/solve.sh",
                        "status": "completed",
                        "return_code": 0,
                        "blocks": [{"kind": "text", "text": "oracle complete"}],
                    },
                    number=None,
                ),
                _turn({"type": "user_message", "text": "first"}, number=1),
                _turn({"type": "user_message", "text": "second"}, number=2),
            ]
        ),
    )

    assert 'id="setup"' in page
    assert 'id="turn-1"' in page
    assert 'id="turn-2"' in page
    assert "oracle complete" in page
    assert page.count('<div class="turn-num">1</div>') == 1
    # The marker is the permalink handle boot.js wires up.
    assert '<aside class="event-marker" data-anchor="turn-1">' in page
    assert "2 turns · 3 events" in page
    assert "bf-setup-divider" in page


def test_timeout_event_cannot_look_like_a_clean_finish(
    harness: Path, tmp_path: Path
) -> None:
    page = render(
        harness,
        tmp_path,
        _payload(
            status={"slug": "timeout", "label": "Timeout"},
            notices=[
                {
                    "level": "warning",
                    "title": "Partial trajectory",
                    "body": "This trace may end before the agent stopped.",
                }
            ],
            turns=[
                _turn(
                    {
                        "type": "agent_timeout",
                        "text": "wall_clock_timeout",
                        "timeout_sec": 900.0,
                        "pending_tool_call_ids": ["tc-pending"],
                        "terminal_trajectory_complete": False,
                    }
                )
            ],
        ),
    )

    assert "bf-timeout-card" in page
    assert "bf-status-timeout" in page
    assert "15.0m" in page
    assert "tc-pending" in page
    assert "Partial trajectory" in page
    assert "bf-notice-warning" in page


def test_a_payload_from_a_newer_schema_is_refused_not_mangled(
    harness: Path, tmp_path: Path
) -> None:
    """Slice 5 ships pages and data separately; a mismatch must be loud."""
    page = render(
        harness, tmp_path, _payload(schema_version=PAYLOAD_SCHEMA_VERSION + 1)
    )

    assert "Unsupported trace" in page
    assert 'id="trace"' not in page
