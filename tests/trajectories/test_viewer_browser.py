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
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

pw = pytest.importorskip("playwright.sync_api")

from benchflow.trajectories import viewer  # noqa: E402
from benchflow.trajectories.viewer.catalog import _rollout_summary  # noqa: E402
from benchflow.trajectories.viewer.payload import (  # noqa: E402
    _build_acp_payload,
    _safe_json,
)
from benchflow.trajectories.viewer.render import _render_shell  # noqa: E402

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
RACE_SLOW_ID = "job-1/alpha__aaaa0001"
RACE_FAST_ID = "job-1/alpha__aaaa0002"
LONG_RUN_ID = "edge/long__dddd0001"


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
    # Third task group (>2 groups → task grouping defaults collapsed) whose
    # name carries a comma — group keys are user data and must survive the
    # URL round trip. 119.6s pins fmtDuration's round-then-split ("2m 0s",
    # never "1m 60s"); the agent matches alpha's so agent grouping stays at
    # 2 groups and keeps exercising the default-open branch.
    _rollout(
        base,
        "job-3/gamma__cccc0001",
        [say("gamma attempt")],
        _result(
            "gamma, task",
            "claude-code",
            "sonnet-x",
            0.5,
            timing={"agent_execution": 100.0, "total": 119.6},
        ),
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


def _start_server(base: Path) -> str:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    thread = threading.Thread(target=viewer.serve, args=(str(base), port), daemon=True)
    thread.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:  # pragma: no cover
        pytest.fail("viewer server did not come up")
    return f"http://localhost:{port}"


@pytest.fixture(scope="session")
def server(corpus) -> str:
    return _start_server(corpus)


@pytest.fixture(scope="module")
def big_server(tmp_path_factory) -> str:
    """105 same-task runs: enough to cross PAGE_SIZE=100 and page."""
    base = tmp_path_factory.mktemp("browser-big-corpus")
    for i in range(105):
        _rollout(
            base,
            f"job/big__{i:08d}",
            [{"type": "agent_message", "text": "ok"}],
            _result("big-task", "claude-code", "sonnet-x", 1.0),
        )
    return _start_server(base)


@pytest.fixture(scope="module")
def edge_server(tmp_path_factory) -> str:
    """Catalog with hostile object keys and identifiers wider than a phone."""
    base = tmp_path_factory.mktemp("browser-edge-corpus")
    for index, task in enumerate(("constructor", "toString", "__proto__")):
        _rollout(
            base,
            f"edge/reserved__{index:08d}",
            [{"type": "agent_message", "text": "reserved key works"}],
            _result(task, "edge-agent", "edge-model", 1.0),
        )
    long_name = "task-" + "x" * 320
    long_model = "provider/model-" + "y" * 320
    _rollout(
        base,
        LONG_RUN_ID,
        [{"type": "agent_message", "text": "long output " + "z" * 900}],
        _result(long_name, "harness-" + "h" * 160, long_model, 1.0),
    )
    return _start_server(base)


@pytest.fixture(scope="module")
def race_server(corpus) -> str:
    """Threaded browse server whose first rollout is deliberately slow."""
    run_ids = [RACE_SLOW_ID, RACE_FAST_ID]
    payloads = {
        run_id: _build_acp_payload(corpus / run_id, None).to_payload()
        for run_id in run_ids
    }
    shell = _render_shell(
        "race corpus",
        {
            "mode": "browse",
            "rollouts": [_rollout_summary(corpus, run_id) for run_id in run_ids],
        },
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                body = shell
                content_type = "text/html; charset=utf-8"
            elif parsed.path == "/api/rollout":
                run_id = (parse_qs(parsed.query).get("id") or [""])[0]
                if run_id not in payloads:
                    self.send_error(404)
                    return
                if run_id == RACE_SLOW_ID:
                    time.sleep(0.5)
                body = _safe_json(payloads[run_id]).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(OSError):
                self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    httpd = ThreadingHTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://localhost:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


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
    """Guards PR #1034's catalog controls and URL-state restoration."""
    page.goto(server)
    names = page.locator(".group-head .gname")
    assert sorted(names.all_inner_texts()) == ["alpha-task", "beta-task", "gamma, task"]

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
    """Guards PR #1034 against inert disclosure buttons for default-open groups."""
    # Regression: with ≤2 groups the default-open state used to short-circuit
    # the toggle, making headers unclickable-in-effect. Agent grouping keeps
    # exactly 2 groups, so it pins the default-open branch of the XOR.
    page.goto(server + "/?group=agent")
    claude_rows = page.locator(".group", has_text="claude-code · sonnet-x").locator(
        ".runrow"
    )
    assert claude_rows.count() == 4  # 3 alpha + 1 gamma
    page.locator(".group-head", has_text="claude-code · sonnet-x").click()
    assert claude_rows.count() == 0  # collapsed
    assert "toggled=" in page.url

    page.reload()  # collapsed state survives reload via the URL
    assert (
        page.locator(".group", has_text="claude-code · sonnet-x")
        .locator(".runrow")
        .count()
        == 0
    )

    page.locator(".group-head", has_text="claude-code · sonnet-x").click()
    assert (
        page.locator(".group", has_text="claude-code · sonnet-x")
        .locator(".runrow")
        .count()
        == 4
    )


def test_groups_expand_when_collapsed_by_default(page, server):
    """Guards PR #1034's explicit expansion of default-collapsed groups."""
    # The >2-groups branch: task grouping (3 groups) starts collapsed and a
    # toggle opens exactly one group; the URL keeps it open across reload.
    page.goto(server)
    alpha_rows = page.locator(".group", has_text="alpha-task").locator(".runrow")
    assert alpha_rows.count() == 0
    page.locator(".group-head", has_text="alpha-task").click()
    assert alpha_rows.count() == 3
    assert "toggled=" in page.url

    page.reload()
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 3
    assert page.locator(".group", has_text="beta-task").locator(".runrow").count() == 0

    page.locator(".group-head", has_text="alpha-task").click()
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 0


def test_comma_in_group_key_url_roundtrip(page, server):
    """Guards PR #1034's encoding of punctuation in group-state URLs."""
    # Group keys are user data — a task named "gamma, task" must not split on
    # the toggled-list delimiter and corrupt other groups' state on reload.
    page.goto(server)
    page.locator(".group-head", has_text="gamma, task").click()
    assert (
        page.locator(".group", has_text="gamma, task").locator(".runrow").count() == 1
    )

    page.reload()
    assert (
        page.locator(".group", has_text="gamma, task").locator(".runrow").count() == 1
    )
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 0
    assert page.locator(".group", has_text="beta-task").locator(".runrow").count() == 0


def test_filter_does_not_invert_explicit_toggles(page, server):
    """Guards PR #1034 against catalog filtering corrupting disclosure state."""
    # Regression: defaultOpen was recomputed from the FILTERED group count, so
    # narrowing to ≤2 groups flipped every explicitly toggled group.
    page.goto(server)
    page.locator(".group-head", has_text="alpha-task").click()  # open alpha
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 3

    page.fill("#ixsearch", "alpha")  # narrows to 1 group — matches stay visible
    assert page.locator(".runrow").count() == 3

    page.fill("#ixsearch", "")  # clearing the filter restores the toggle state
    assert page.locator(".group", has_text="alpha-task").locator(".runrow").count() == 3
    assert page.locator(".group", has_text="beta-task").locator(".runrow").count() == 0


def test_typing_in_filter_preserves_caret(page, server):
    """Guards PR #1034 against catalog rerenders moving the filter caret."""
    # Regression: each keystroke rebuilt the input and forced the caret to the
    # end, so inserting into the middle of a query scrambled it.
    page.goto(server)
    box = page.locator("#ixsearch")
    box.click()
    box.type("beta")
    box.press("ArrowLeft")
    box.press("ArrowLeft")
    box.type("x")  # caret sits between "be" and "ta"
    assert box.input_value() == "bexta"


def test_sort_orders_rows_and_formats_durations(page, server):
    """Guards PR #1034's catalog ordering and normalized duration display."""
    # group=none exercises the flat list; reward sort must order actual rows
    # (not just land in the URL).
    page.goto(server + "/?group=none&sort=reward")
    rewards = page.locator(".runrow .rreward").all_inner_texts()
    assert rewards == ["pass", "pass", "pass", "fail 0.5", "fail 0", "—"]
    # 119.6s renders rounded-then-split ("2m 0s"), never "1m 60s"
    assert "2m 0s" in page.locator(".runrow", has_text="gamma, task").inner_text()


def test_unknown_group_and_sort_params_fall_back(page, server):
    """Guards PR #1034 against hostile or unsupported catalog URL options."""
    # ?group=constructor must not walk the prototype chain into Object
    page.goto(server + "/?group=constructor&sort=hasOwnProperty")
    assert page.locator(".group-head").count() == 3  # rendered, grouped by task
    assert page.locator("#ixcontrols select").first.input_value() == "task"
    assert page.locator("#ixcontrols select").nth(1).input_value() == "name"


def test_pagination_caps_and_shows_more(page, big_server):
    """Guards PR #1034's incremental pagination for large run groups."""
    page.goto(big_server + "/?group=none")
    assert page.locator(".runrow").count() == 100
    more = page.locator(".showmore")
    assert "5 hidden" in more.inner_text()
    more.click()
    assert page.locator(".runrow").count() == 105
    assert page.locator(".showmore").count() == 0


def test_head_requests_never_touch_the_filesystem(server):
    """Guards PR #1034 against inherited static-file HEAD handling."""
    # SimpleHTTPRequestHandler's inherited do_HEAD serves files from the
    # process cwd — the browse handler must answer from its own routes only.
    req = urllib.request.Request(server + "/", method="HEAD")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
    req = urllib.request.Request(server + "/pyproject.toml", method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=5)
    assert excinfo.value.code == 405


def test_run_selection_and_back_preserve_catalog_state(page, server):
    """Guards PR #1034 and restores focus to the selected run after Back."""
    page.goto(server)
    page.fill("#ixsearch", "alpha")
    selected = page.locator(".runrow").first
    selected_id = selected.get_attribute("data-run-id")
    selected.click()
    page.wait_for_selector("#hdr h1")
    assert "run=" in page.url
    assert page.locator("#backbar").is_visible()
    assert page.locator("#view-index").is_hidden()

    page.click("#backbtn")
    assert page.locator("#view-index").is_visible()
    assert page.input_value("#ixsearch") == "alpha"
    assert "run=" not in page.url
    page.wait_for_function(
        "runId => document.activeElement && document.activeElement.dataset.runId === runId",
        arg=selected_id,
    )

    page.go_back()  # browser back re-enters the detail view
    page.wait_for_selector("#hdr h1")
    assert page.locator("#backbar").is_visible()


def test_error_state_for_unknown_run(page, server):
    """Guards PR #1034 against silently substituting an unknown deep link."""
    page.goto(server + "/?run=job-1/nope__00000000")
    banner = page.locator(".errbox")
    assert banner.is_visible()
    assert "is not among" in banner.inner_text()


def test_reserved_group_keys_render_and_page_normally(page, edge_server):
    """Guards PR #1034 against prototype-key task names corrupting page state."""
    page.goto(edge_server)
    for key in ("constructor", "toString", "__proto__"):
        heading = page.locator(".group-head", has_text=key)
        assert heading.count() == 1
        heading.click()
        group = page.locator(".group", has=page.locator(".gname", has_text=key))
        assert group.locator(".runrow").count() == 1


def test_slow_run_cannot_overwrite_a_newer_fast_run(page, race_server):
    """Guards PR #1034 against a slow A response overwriting fast run B."""
    page.goto(race_server)
    page.locator(f'.runrow[data-run-id="{RACE_SLOW_ID}"]').click()
    page.evaluate(
        "runId => [...document.querySelectorAll('.runrow')].find(b => b.dataset.runId === runId).click()",
        RACE_FAST_ID,
    )
    page.wait_for_selector(".card", state="visible")
    assert "gave up" in page.locator("#view-trace").inner_text()
    page.wait_for_timeout(650)
    assert "gave up" in page.locator("#view-trace").inner_text()
    assert "alpha solved" not in page.locator("#view-trace").inner_text()


def test_back_aborts_a_slow_detail_transition(page, race_server):
    """Guards PR #1034 against a late run response reopening detail after Back."""
    page.goto(race_server)
    page.locator(f'.runrow[data-run-id="{RACE_SLOW_ID}"]').click()
    page.locator("#backbtn").click()
    assert page.locator("#view-index").is_visible()
    page.wait_for_timeout(650)
    assert page.locator("#view-index").is_visible()
    assert page.locator("#content").is_hidden()
    assert "run=" not in page.url


# ── detail ────────────────────────────────────────────────────────────────


def test_detail_tabs(page, server):
    """Guards PR #1034's Trace, Verifier, and Metrics tab behavior."""
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
    """Guards PR #1034: Focus is the default and Full reveals thoughts."""
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    page.wait_for_selector(".card.k-thought", state="attached")
    assert page.locator("#btn-focus").get_attribute("aria-pressed") == "true"
    assert page.locator(".card.k-thought").first.is_hidden()
    page.locator("#toolbar button", has_text="Full").click()
    assert page.locator(".card.k-thought").first.is_visible()


def test_search_reveal_is_reduced_with_active_filters(page, server):
    """Guards PR #1034 against search permanently unhiding a filtered card."""
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    thought = page.locator(".card.k-thought").first
    page.wait_for_selector(".card.k-thought", state="attached")
    page.locator("#toolbar button", has_text="Full").click()
    page.locator("#toolbar button", has_text="Prompts").click()
    assert thought.is_hidden()
    page.fill("#search", "zebras")
    assert thought.is_visible()
    page.fill("#search", "")
    assert thought.is_hidden()


def test_malformed_api_payload_has_a_visible_fallback(page, server):
    """Guards PR #1034 against malformed rollout JSON leaving a blank viewer."""
    page.route(
        "**/api/rollout?*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"steps":"not-an-array"}',
        ),
    )
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    banner = page.locator(".errbox")
    banner.wait_for(state="visible")
    assert "payload.steps must be an array" in banner.inner_text()
    assert page.locator("#view-index").is_hidden()


def test_malformed_embedded_boot_has_a_visible_fallback(page, server):
    """Guards PR #1034 against invalid embedded boot JSON blanking the page."""
    html = urllib.request.urlopen(server, timeout=5).read().decode("utf-8")
    malformed = re.sub(
        r'(<script id="bf-payload" type="application/json">).*?(</script>)',
        r"\1{not-json\2",
        html,
        count=1,
        flags=re.DOTALL,
    )
    page.set_content(malformed, wait_until="load")
    banner = page.locator(".errbox")
    assert banner.is_visible()
    assert "not valid JSON" in banner.inner_text()


def test_script_data_double_escape_cannot_swallow_the_viewer(page):
    """Guards PR #1034 against ``<!--<script>`` swallowing the viewer script."""
    hostile = "<!--<script>"
    html = _render_shell(
        "script tokenizer regression",
        {
            "mode": "single",
            "payload": {
                "rollout_name": "hostile-script-data",
                "meta": {"task_name": "script tokenizer task"},
                "steps": [{"i": 1, "kind": "message", "text": hostile}],
                "verifier": {},
            },
        },
    )
    assert hostile not in html
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_content(html, wait_until="load")
    page.wait_for_selector(".card.k-message", state="visible")
    assert hostile in page.locator("#view-trace").inner_text()
    assert errors == []


def test_tool_status_class_uses_a_closed_whitelist(page, server):
    """Guards PR #1034 against untrusted tool statuses becoming CSS classes."""
    payload = {
        "rollout_name": "hostile-status",
        "meta": {"task_name": "status task"},
        "steps": [
            {
                "i": 1,
                "kind": "tool",
                "dur": 119.6,
                "tool": {
                    "kind": "execute",
                    "hue": "execute",
                    "title": "safe title",
                    "status": "completed injected-class",
                    "content": [],
                },
            }
        ],
    }
    page.route(
        "**/api/rollout?*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(payload)
        ),
    )
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    status = page.locator(".status")
    status.wait_for(state="visible")
    assert status.get_attribute("class") == "status unknown"
    assert status.inner_text() == "completed injected-class"
    assert "2m 0s" in page.locator(".card .tstamp").inner_text()


def test_tabs_and_disclosures_are_keyboard_complete(page, server, edge_server):
    """Guards PR #1034 keyboard semantics, labels, and disclosure focus."""
    page.goto(server + "/?run=job-2/beta__bbbb0002")
    trace = page.locator("#tab-trace")
    trace.focus()
    trace.press("ArrowRight")
    assert page.locator("#tab-verifier").get_attribute("aria-selected") == "true"
    assert page.locator("#view-verifier").is_visible()
    page.locator("#tab-verifier").press("End")
    assert page.locator("#tab-metrics").get_attribute("aria-selected") == "true"
    page.locator("#tab-metrics").press("Home")
    assert page.locator("#tab-trace").get_attribute("aria-selected") == "true"

    page.goto(server)
    assert (
        page.evaluate("document.querySelector('label[for=ixgroup]').control.id")
        == "ixgroup"
    )
    assert (
        page.evaluate("document.querySelector('label[for=ixsort]').control.id")
        == "ixsort"
    )
    heading = page.locator(".group-head", has_text="alpha-task")
    heading.focus()
    heading.press("Enter")
    heading = page.locator(".group-head", has_text="alpha-task")
    assert heading.get_attribute("aria-expanded") == "true"
    assert page.evaluate("document.activeElement.dataset.groupKey") == "alpha-task"
    assert page.locator("#" + heading.get_attribute("aria-controls")).count() == 1

    page.goto(edge_server + "/?run=" + LONG_RUN_ID)
    expander = page.locator(".expander").first
    expander.wait_for(state="visible")
    controlled = expander.get_attribute("aria-controls")
    assert expander.get_attribute("aria-expanded") == "false"
    expander.focus()
    expander.press("Enter")
    assert expander.get_attribute("aria-expanded") == "true"
    assert page.locator("#" + controlled).count() == 1
    assert page.evaluate("document.activeElement.classList.contains('expander')")


def test_search_matches_and_event_anchors_and_timeline(page, server):
    """Guards PR #1034's search, event-anchor, and timeline interactions."""
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    page.wait_for_selector("#search")
    page.fill("#search", "zebras")
    # a hit immediately jumps to it, so the counter shows position / total
    assert page.locator("#matchinfo").inner_text() == "1 / 1"

    anchors = page.locator(".card .seq")
    assert anchors.first.get_attribute("href") == "#e1"
    # timestamps in the corpus render offset chips (timeline forward-compat)
    assert page.locator(".tstamp").count() > 0


def test_malformed_event_fragment_does_not_break_trace_render(page, server):
    """Guards PR #1034 against malformed percent escapes crashing the trace pump."""
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(server + "/?run=job-1/alpha__aaaa0001#e%")
    page.wait_for_selector(".card", state="visible")
    page.wait_for_timeout(100)
    assert errors == []


def test_failed_detail_load_shows_visible_error(page, server):
    """Guards PR #1034 against stale detail UI after a rollout request fails."""
    # Regression: a failed /api/rollout used to leave the previous run's
    # header/tabs rendered and write the error into a #view-trace pane that a
    # prior tab switch had hidden — the user saw run A's data as if run B
    # loaded. The URL also fell back to the catalog, so reload lost the run.
    page.goto(server + "/?run=job-2/beta__bbbb0002")
    page.wait_for_selector("#tabs button")
    page.locator("#tabs button", has_text="Metrics").click()  # hides #view-trace
    page.click("#backbtn")
    page.route("**/api/rollout?*", lambda route: route.abort())
    page.fill("#ixsearch", "alpha")
    page.locator(".runrow").first.click()

    # the aborted fetch rejects asynchronously — wait for the error to render
    banner = page.locator(".errbox")
    banner.wait_for(state="visible")
    assert "Failed to load run" in banner.inner_text()
    assert page.locator("#hdr h1").count() == 0  # stale run header cleared
    assert "run=" in page.url  # reload retries the run instead of the catalog


def test_anchor_back_stays_on_detail_without_refetch(page, server):
    """Guards PR #1034 against refetching a run on hash-only Back navigation."""
    # Regression: popstate treated the hash-only traversal from an in-trace
    # #eN anchor as catalog navigation and refetched + re-rendered the run.
    calls = []
    page.on(
        "request",
        lambda r: calls.append(r.url) if "/api/rollout?" in r.url else None,
    )
    page.goto(server + "/?run=job-1/alpha__aaaa0001")
    page.wait_for_selector(".card .seq")
    assert len(calls) == 1
    page.locator(".card .seq").first.click()  # hash-navigate to #e1
    page.go_back()
    page.wait_for_timeout(200)
    assert page.locator("#backbar").is_visible()
    assert page.locator("#view-index").is_hidden()
    assert len(calls) == 1  # no refetch on the hash-only back


def test_untrusted_content_stays_text(page, server):
    """Guards PR #1034 against DOM injection from trajectory content."""
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


def test_long_real_identifiers_do_not_overflow_narrow_viewports(page, edge_server):
    """Guards PR #1034 against long task/model names causing page-wide scroll."""
    page.set_viewport_size({"width": 320, "height": 760})
    page.goto(edge_server)
    page.wait_for_selector(".group-head")
    assert page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) <= document.documentElement.clientWidth"
    )

    page.goto(edge_server + "/?run=" + LONG_RUN_ID)
    page.wait_for_selector("#hdr h1")
    assert page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) <= document.documentElement.clientWidth"
    )


def test_representative_screenshots(browser, server, tmp_path):
    """Guards PR #1034's representative desktop and narrow layouts."""
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
