"""The job views in real Chromium: the exported page and the browse server.

Self-skips when playwright or its chromium binary is unavailable.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

from benchflow.trajectories.viewer.jobviews import export_html  # noqa: E402

pytestmark = pytest.mark.browser


def _trial(folder: Path, task: str, reward: float | None, model: str, **extra) -> Path:
    folder.mkdir(parents=True)
    result = {
        "task_name": task,
        "rollout_name": folder.name,
        "agent": "claude-agent-acp",
        "model": model,
        "rewards": {"reward": reward} if reward is not None else None,
        "n_tool_calls": 2,
        "started_at": "2026-09-30 10:00:00",
        "finished_at": "2026-09-30 10:01:00",
        "agent_result": {"total_tokens": 500 if model == "m1" else 2000},
        **extra,
    }
    (folder / "result.json").write_text(json.dumps(result))
    (folder / "trajectory").mkdir()
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": f"working on {task}"}) + "\n"
    )
    return folder


@pytest.fixture
def job(tmp_path: Path) -> Path:
    root = tmp_path / "job"
    _trial(root / "alpha__1", "alpha", 1.0, "m1")
    _trial(root / "alpha__2", "alpha", 0.5, "m2")
    _trial(
        root / "beta__1",
        "beta",
        None,
        "m1",
        error="sandbox never started",
        error_category="sandbox_setup",
    )
    hacked = _trial(root / "beta__2", "beta", 1.0, "m2")
    (hacked / "integrity").mkdir()
    (hacked / "integrity" / "claim_verdict.json").write_text(
        json.dumps({"core_verdict": "AgentViolation", "reason": "wrote reward.txt"})
    )
    return root


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:
            pytest.skip("chromium not installed; run `playwright install chromium`")
        yield b
        b.close()


def _page(browser, errors: list[str]):
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)))
    return page


def test_exported_page_draws_every_view(browser, job: Path, tmp_path: Path) -> None:
    out, _, _ = export_html([job], tmp_path / "o.html")
    errors: list[str] = []
    page = _page(browser, errors)
    page.goto(out.as_uri())
    page.wait_for_selector(".jgrid")
    # No Runs tab in an export; Training is hidden without steps.
    assert page.locator(".jtab").all_inner_texts() == ["Outcomes", "Pareto"]
    assert page.locator(".jgrid .sq").count() == 4
    assert page.locator(".jgrid .sq.u").count() == 1  # hatched, not a reward color
    assert page.locator(".jgrid .sq.x").count() == 1  # the exploit mark
    assert page.locator(".jgrid .sq.l").count() == 0  # no trial links in an export
    page.locator(".jgrid .sq.u").hover()
    tip = page.locator(".jtip")
    assert "the sandbox did not start" in tip.inner_text()
    assert "infrastructure problem" in tip.inner_text()
    page.locator(".jgrid .sq.x").hover()
    assert "AgentViolation" in tip.inner_text()
    # Columns default to the dimension that varies (model); transpose swaps.
    assert page.locator(".jgrid thead .chl").all_inner_texts() == ["m1", "m2"]
    page.locator("#jtranspose").check()
    assert page.locator(".jgrid thead .chl").all_inner_texts() == ["alpha", "beta"]
    assert page.locator(".jgrid tbody .rl").all_inner_texts() == ["m1", "m2"]
    page.get_by_role("button", name="Pareto").click()
    page.wait_for_selector(".jchart")
    assert page.locator(".jchart .pt").count() == 2
    assert "view=pareto" in page.url
    assert errors == []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def test_a_square_opens_its_trial_and_back_returns(browser, job: Path) -> None:
    from benchflow.trajectories.viewer.server import serve

    port = _free_port()
    threading.Thread(target=serve, args=(str(job), port), daemon=True).start()
    base = f"http://localhost:{port}/"
    deadline = time.monotonic() + 20
    while True:
        try:
            urllib.request.urlopen(base, timeout=1).read()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    errors: list[str] = []
    page = _page(browser, errors)
    page.goto(base)
    page.get_by_role("button", name="Outcomes").click()
    page.wait_for_selector(".jgrid .sq.l")
    page.locator(".jgrid .sq.l").first.click()
    page.wait_for_selector("#content:not(.hidden)")
    assert "run=" in page.url and "view=outcomes" in page.url
    assert page.locator("#view-job").is_hidden()
    page.locator("#backbtn").click()
    page.wait_for_selector(".jgrid")
    assert page.locator("#view-job").is_visible()
    assert errors == []


def test_a_late_build_does_not_draw_over_an_open_trial(
    browser, job: Path, monkeypatch
) -> None:
    """Open Outcomes on a job still building, go to Runs and open a trial:
    when the build lands, the trial stays alone on screen."""
    from benchflow.trajectories.viewer import jobviews
    from benchflow.trajectories.viewer.server import serve

    real = jobviews.build_for_roots

    def slow(*args, **kwargs):
        time.sleep(2.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(jobviews, "build_for_roots", slow)
    port = _free_port()
    threading.Thread(target=serve, args=(str(job), port), daemon=True).start()
    base = f"http://localhost:{port}/"
    deadline = time.monotonic() + 20
    while True:
        try:
            urllib.request.urlopen(base, timeout=1).read()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    errors: list[str] = []
    page = _page(browser, errors)
    page.goto(base + "?view=outcomes")
    page.wait_for_selector("#view-job .jnote")  # still building
    page.get_by_role("button", name="Runs").click()
    page.locator(".runrow").first.click()
    page.wait_for_selector("#content:not(.hidden)")
    page.wait_for_timeout(3500)  # the build lands meanwhile
    assert page.locator("#view-job").is_hidden()
    assert page.locator("#view-index").is_hidden()
    assert page.locator("#content").is_visible()
    assert errors == []
