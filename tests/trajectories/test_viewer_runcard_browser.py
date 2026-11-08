"""The run card in real Chromium: evidence one hover or focus away.

Guards EN-327 (bench view run card): a summary on the card opens its proof
on hover and on keyboard focus, Escape closes it, and the one action jumps
to the evidence. Self-skips without playwright or its chromium binary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

from benchflow.trajectories.viewer.render import _render_acp_trajectory  # noqa: E402

pytestmark = pytest.mark.browser

TRACE = (
    ">       assert 10000 < val\n"
    "E       AssertionError: Ystar_base should be in thousands, got 78.74\n"
    "\n"
    "/tests/test_outputs.py:261: AssertionError"
)


def _page(base: Path, name: str, events: list[dict], result: dict, ctrf=None) -> Path:
    folder = base / name
    (folder / "trajectory").mkdir(parents=True)
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    (folder / "result.json").write_text(json.dumps(result))
    (folder / "timing.json").write_text(
        json.dumps(
            {
                "environment_setup": 3.5,
                "agent_setup": 27.2,
                "agent_execution": 875.6,
                "verifier": 7.9,
                "total": 928.4,
            }
        )
    )
    if ctrf is not None:
        (folder / "verifier").mkdir()
        (folder / "verifier" / "ctrf.json").write_text(json.dumps(ctrf))
    out = base / f"{name}.html"
    out.write_text(
        _render_acp_trajectory(
            folder, folder / "trajectory" / "acp_trajectory.jsonl", None
        )
    )
    return out


def _tool(call_id: str, title: str, status: str) -> dict:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": "execute",
        "title": title,
        "status": status,
        "content": [
            {"type": "content", "content": {"type": "text", "text": "building"}}
        ],
    }


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except pw.Error:  # pragma: no cover — binary not installed
            pytest.skip("chromium not installed — run `playwright install chromium`")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture(scope="module")
def failed_page(tmp_path_factory) -> Path:
    return _page(
        tmp_path_factory.mktemp("runcard"),
        "shock__ad908e49",
        [{"type": "agent_message", "text": "done"}],
        {"task_name": "shock-analysis", "rewards": {"reward": 0.0}, "model": "m-1"},
        ctrf={
            "results": {
                "tool": {"name": "pytest", "version": "8.4.1"},
                "summary": {"start": 100.0, "stop": 101.08},
                "tests": [
                    {"name": "test_outputs.py::test_ok", "status": "passed"},
                    {
                        "name": "test_outputs.py::test_value_magnitudes",
                        "status": "failed",
                        "trace": TRACE,
                    },
                ],
            }
        },
    )


def test_side_panel_shows_verdict_checks_and_why(page, failed_page):
    page.goto(failed_page.as_uri())
    page.locator("#side .score").wait_for()
    assert page.locator("#hdr .badge.fail").inner_text().lower() == "failed"
    assert page.locator("#side .score-v").inner_text() == "0.0"
    assert page.locator("#side .check-row.bad .check-name").inner_text() == (
        "test_value_magnitudes"
    )
    assert (
        page.locator("#side .runner").inner_text().endswith("pytest 8.4.1 \u00b7 1.1s")
    )
    phases = page.locator("#side .kv-row.num .kv-k").all_inner_texts()
    assert phases[:4] == ["Environment setup", "Agent setup", "Agent", "Verifier"]
    assert page.locator("#why").inner_text() == (
        "Scored 0.0 because test_value_magnitudes failed: Ystar_base should be in "
        "thousands, got 78.74 (tests/test_outputs.py:261)."
    )
    # one primary action, at most
    assert page.locator(".rc-action").count() == 1
    assert page.locator(".timeline .tl-mark").count() == 1


def test_failed_check_opens_its_assertion_on_hover_and_focus(page, failed_page):
    page.goto(failed_page.as_uri())
    trigger = page.locator("#side .check-row.bad")
    trigger.hover()
    pop = page.locator(".pop.open")
    pop.wait_for()
    assert "AssertionError: Ystar_base should be in thousands" in pop.inner_text()
    assert "tests/test_outputs.py:261" in pop.inner_text()

    page.mouse.move(0, 0)
    pop.wait_for(state="hidden")

    trigger.focus()
    page.locator(".pop.open").wait_for()
    assert trigger.get_attribute("aria-describedby") == page.locator(
        ".pop.open"
    ).get_attribute("id")
    page.keyboard.press("Escape")
    page.locator(".pop.open").wait_for(state="detached")


def test_action_opens_the_failing_test_in_the_verifier_tab(page, failed_page):
    page.goto(failed_page.as_uri())
    page.locator(".rc-action").click()
    assert page.locator("#view-verifier").is_visible()
    failure = page.locator("#view-verifier tr.failure")
    assert "tests/test_outputs.py:261" in failure.inner_text()


def test_timeout_points_at_the_interrupted_call(page, tmp_path):
    html = _page(
        tmp_path,
        "build__b2d1a2fb",
        [
            _tool("c1", "ls", "completed"),
            _tool("c2", "make build", "in_progress"),
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 900,
                "pending_tool_call_ids": ["c2"],
            },
        ],
        {
            "task_name": "build",
            "error": "Agent timed out after 900s",
            "error_category": "timeout",
        },
    )
    page.goto(html.as_uri())
    assert page.locator("#hdr .badge").inner_text().lower() == "timed out"
    call = page.locator("#side .call-row").first
    assert "make build" in call.inner_text()
    call.hover()
    assert "building" in page.locator(".pop.open").inner_text()

    # the timeline names each mark on hover; the named call is marked
    page.mouse.move(0, 0)
    page.locator('.tl-mark[data-i="2"]').hover()
    tip = page.locator(".tl-tip.open")
    assert "make build" in tip.inner_text()
    assert "named" in page.locator('.tl-mark[data-i="2"]').get_attribute("class")

    page.locator(".rc-action", has_text="Show call #2").click()
    assert page.locator("#e2").is_visible()
    assert "flash" in page.locator("#e2").get_attribute("class")


def test_focus_folds_completed_tool_output_and_the_head_opens_it(page, tmp_path):
    html = _page(
        tmp_path,
        "fold__00000001",
        [_tool("c1", "make", "completed"), _tool("c2", "make test", "failed")],
        {"task_name": "fold", "rewards": {"reward": 0.0}},
    )
    page.goto(html.as_uri())
    done, failed = page.locator("#e1 .tout"), page.locator("#e2 .tout")
    assert done.is_hidden() and failed.is_visible()
    page.locator("#e1 .chead").click()
    assert done.is_visible()
