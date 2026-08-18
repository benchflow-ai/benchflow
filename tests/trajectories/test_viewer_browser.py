"""Browser-level tests for the interactive viewer (review point 5).

Everything here drives the real template JavaScript in headless Chromium via
Playwright — the behaviors no Python-side test can reach: catalog grouping /
filtering / sorting with URL-state restoration, group collapsing, run
selection with back navigation that preserves the catalog view, the detail
tabs, Focus/Full switching, in-trace search and event anchors, error states,
untrusted trajectory content staying text, and representative
desktop/narrow-viewport screenshots.

The suite self-skips when playwright or its chromium binary is unavailable,
so a plain ``pytest tests/`` stays runnable without ``playwright install``.
"""

import json
import socket
import threading
import urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

from benchflow.trajectories import viewer  # noqa: E402

pytestmark = pytest.mark.browser


# ── corpus ────────────────────────────────────────────────────────────────


def _rollout(base: Path, rel: str, events: list[dict], result: dict) -> Path:
    d = base / rel
    (d / "trajectory").mkdir(parents=True)
    (d / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    (d / "result.json").write_text(json.dumps(result))
    return d


def _result(task, agent, model, reward, **extra):
    out = {
        "task_name": task,
        "agent_name": agent,
        "model": model,
        "rewards": {"reward": reward} if reward is not None else None,
        "skill_mode": "no-skill",
        "timing": {"agent_execution": 10.0, "total": 12.0},
        "agent_result": {"total_tokens": 1234, "cost_usd": 0.05},
        "n_tool_calls": 1,
    }
    out.update(extra)
    return out


HOSTILE_TEXT = '<img src=x onerror="alert(1)"> and </script><script>alert(2)</script>'


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("browser-corpus")
    say = lambda text: {"type": "agent_message", "text": text}  # noqa: E731
    think = lambda text: {"type": "agent_thought", "text": text}  # noqa: E731
    tool = {
        "type": "tool_call",
        "tool_call_id": "c1",
        "kind": "execute",
        "title": "pytest -q",
        "status": "completed",
        "content": [
            {"type": "content", "content": {"type": "text", "text": "3 passed"}}
        ],
        "started_at": "2026-08-15T09:41:05+00:00",
        "finished_at": "2026-08-15T09:41:07+00:00",
    }
    _rollout(
        base,
        "job-1/alpha__aaaa0001",
        [
            {
                "type": "user_message",
                "text": "solve alpha",
                "ts": "2026-08-15T09:41:03+00:00",
            },
            think("thinking about zebras first"),
            tool,
            say("alpha solved"),
        ],
        _result("alpha-task", "claude-code", "sonnet-x", 1.0),
    )
    _rollout(
        base,
        "job-1/alpha__aaaa0002",
        [say("gave up")],
        _result("alpha-task", "claude-code", "sonnet-x", 0.0),
    )
    _rollout(
        base,
        "job-1/alpha__aaaa0003",
        [say("hmm")],
        _result(
            "alpha-task",
            "claude-code",
            "sonnet-x",
            None,
            error="agent timed out",
            error_category="agent_timeout",
        ),
    )
    _rollout(
        base,
        "job-2/beta__bbbb0001",
        [say("beta fine")],
        _result("beta-task", "gemini", "flash-y", 1.0),
    )
    hostile = _rollout(
        base,
        "job-2/beta__bbbb0002",
        [say(HOSTILE_TEXT)],
        _result("beta-task", "gemini", "flash-y", 1.0),
    )
    vdir = hostile / "verifier"
    vdir.mkdir()
    (vdir / "reward.txt").write_text("1.0")
    (vdir / "test-stdout.txt").write_text("all good\n")
    (vdir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tests": [{"name": "t_one", "status": "passed", "duration": 12}]
                }
            }
        )
    )
    return base


@pytest.fixture(scope="session")
def server(corpus) -> str:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    thread = threading.Thread(
        target=viewer.serve, args=(str(corpus), port), daemon=True
    )
    thread.start()
    deadline = 50
    for _ in range(deadline):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
            break
        except OSError:
            import time

            time.sleep(0.2)
    else:  # pragma: no cover
        pytest.fail("viewer server did not come up")
    return f"http://localhost:{port}"


@pytest.fixture(scope="module")
def browser():
    # Module scope on purpose: playwright's sync API keeps an asyncio loop
    # "running" on this thread for its whole lifetime, and a session-scoped
    # instance leaks that loop into other modules' pytest-asyncio tests
    # ("Runner.run() cannot be called from a running event loop"). Closing
    # at module teardown keeps the one-launch speed without the leak.
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


# ── catalog ───────────────────────────────────────────────────────────────


def test_grouping_filtering_sorting_and_url_state(page, server):
    page.goto(server)
    names = page.locator(".group-head .gname")
    assert sorted(names.all_inner_texts()) == ["alpha-task", "beta-task"]

    # group by model + harness
    page.locator("#ixcontrols select").first.select_option("agent")
    assert sorted(page.locator(".group-head .gname").all_inner_texts()) == [
        "claude-code · sonnet-x",
        "gemini · flash-y",
    ]
    assert "group=agent" in page.url

    # filter down to beta, sort by reward — both land in the URL
    page.fill("#ixsearch", "beta")
    page.locator("#ixcontrols select").nth(1).select_option("reward")
    assert page.locator(".runrow").count() == 2
    assert "q=beta" in page.url and "sort=reward" in page.url

    # a full reload restores the exact view from the URL
    page.reload()
    assert page.locator(".runrow").count() == 2
    assert page.input_value("#ixsearch") == "beta"
    assert page.locator("#ixcontrols select").first.input_value() == "agent"
    assert page.locator("#ixcontrols select").nth(1).input_value() == "reward"


def test_groups_collapse_even_when_open_by_default(page, server):
    # Regression: with ≤2 groups the default-open state used to short-circuit
    # the toggle, making headers unclickable-in-effect.
    page.goto(server)
    alpha_rows = page.locator(".group", has_text="alpha-task").locator(".runrow")
    assert alpha_rows.count() == 3
    page.locator(".group-head", has_text="alpha-task").click()
    assert alpha_rows.count() == 0  # collapsed
    assert "toggled=" in page.url

    page.reload()  # collapsed state survives reload via the URL
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 0

    page.locator(".group-head", has_text="alpha-task").click()
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 3


def test_run_selection_and_back_preserve_catalog_state(page, server):
    page.goto(server)
    page.fill("#ixsearch", "alpha")
    page.locator(".runrow").first.click()
    page.wait_for_selector("#hdr h1")
    assert "run=" in page.url
    assert page.locator("#backbar").is_visible()
    assert page.locator("#view-index").is_hidden()

    page.click("#backbtn")
    assert page.locator("#view-index").is_visible()
    assert page.input_value("#ixsearch") == "alpha"
    assert "run=" not in page.url

    page.go_back()  # browser back re-enters the detail view
    page.wait_for_selector("#hdr h1")
    assert page.locator("#backbar").is_visible()


def test_error_state_for_unknown_run(page, server):
    page.goto(server + "/?run=job-1/nope__00000000")
    banner = page.locator(".errbox")
    assert banner.is_visible()
    assert "is not among" in banner.inner_text()


# ── detail ────────────────────────────────────────────────────────────────


def test_detail_tabs(page, server):
    page.goto(server + "/?run=job-2/beta__bbbb0002")
    page.wait_for_selector("#tabs button")
    labels = page.locator("#tabs button").all_inner_texts()
    assert labels == ["Trace", "Verifier", "Metrics"]

    page.locator("#tabs button", has_text="Verifier").click()
    assert page.locator(".bigreward").is_visible()
    assert page.locator("#view-trace").is_hidden()
    assert "t_one" in page.locator("#view-verifier").inner_text()

    page.locator("#tabs button", has_text="Metrics").click()
    # inner_text() reflects CSS text-transform (headings render uppercase)
    assert "phase timing" in page.locator("#view-metrics").inner_text().lower()


def test_focus_hides_thoughts_and_full_restores(page, server):
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    page.wait_for_selector(".card.k-thought")
    assert page.locator(".card.k-thought").first.is_visible()
    page.click("#btn-focus")
    assert page.locator(".card.k-thought").first.is_hidden()
    page.locator("#toolbar button", has_text="Full").click()
    assert page.locator(".card.k-thought").first.is_visible()


def test_search_matches_and_event_anchors_and_timeline(page, server):
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    page.wait_for_selector("#search")
    page.fill("#search", "zebras")
    # a hit immediately jumps to it, so the counter shows position / total
    assert page.locator("#matchinfo").inner_text() == "1 / 1"

    anchors = page.locator(".card .seq")
    assert anchors.first.get_attribute("href") == "#e1"
    # timestamps in the corpus render offset chips (timeline forward-compat)
    assert page.locator(".tstamp").count() > 0


def test_untrusted_content_stays_text(page, server):
    fired = []
    page.on("dialog", lambda d: (fired.append(d.message), d.dismiss()))
    page.goto(server + "/?run=job-2/beta__bbbb0002")
    page.wait_for_selector(".card.k-message")
    # the hostile markup is visible as literal text …
    assert HOSTILE_TEXT in page.locator("#view-trace").inner_text()
    # … and never became elements or executed
    assert page.locator("#view-trace img").count() == 0
    assert fired == []


# ── viewports ─────────────────────────────────────────────────────────────


def test_representative_screenshots(browser, server, tmp_path):
    desktop = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
    desktop.goto(server)
    desktop.wait_for_selector(".group-head")
    shot_d = tmp_path / "catalog-desktop.png"
    desktop.screenshot(path=shot_d, full_page=True)

    narrow = browser.new_context(viewport={"width": 390, "height": 844}).new_page()
    narrow.goto(server + "/?run=job-1/alpha__aaaa0001")
    narrow.wait_for_selector("#hdr h1")
    shot_n = tmp_path / "detail-narrow.png"
    narrow.screenshot(path=shot_n, full_page=True)

    assert shot_d.stat().st_size > 10_000
    assert shot_n.stat().st_size > 10_000
    desktop.context.close()
    narrow.context.close()
