"""Guards the fix at 363d815b in payloads and Chromium."""

import json

import pytest

from benchflow.trajectories.viewer.catalog import _rollout_summary
from benchflow.trajectories.viewer.payload import _build_acp_payload
from benchflow.trajectories.viewer.render import _render_shell
from tests.trajectories.test_viewer_assessment import COMPLETE_SCORING


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"rewards": {"reward": 0.75}, "scoring": COMPLETE_SCORING}, "pass"),
        (
            {
                "rewards": {"reward": 0},
                "scoring": {
                    **COMPLETE_SCORING,
                    "passed": False,
                    "tests_pass": False,
                    "verifier_reward": 0,
                },
            },
            "fail",
        ),
        (
            {
                "rewards": {"reward": 1},
                "scoring": {"status": "error", "error": "dummy"},
            },
            "unscored",
        ),
        ({"rewards": {"reward": 1}, "assessment": {"status": "pending"}}, "unscored"),
        ({"rewards": {"reward": 2}}, "fail"),
        ({"rewards": {"reward": 1}}, "pass"),
    ],
)
def test_catalog_and_detail_use_canonical_score(tmp_path, result, expected):
    """Guards FF-sw-trajectories badge, pass counts and catalog filtering."""
    from playwright.sync_api import sync_playwright

    folder = tmp_path / "trial"
    (folder / "trajectory").mkdir(parents=True)
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        '{"type":"agent_message","text":"done"}\n'
    )
    (folder / "result.json").write_text(
        json.dumps({"task_name": "dummy-task", **result})
    )
    payload = _build_acp_payload(folder, None).to_payload()
    summary = _rollout_summary(tmp_path, "trial")
    browse = _render_shell("dummy", {"mode": "browse", "rollouts": [summary]})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.route(
            "**/*",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json"
                if "/api/" in route.request.url
                else "text/html",
                body=json.dumps(payload) if "/api/" in route.request.url else browse,
            ),
        )
        page.goto("http://localhost:8900/")
        assert f"{expected} 1" in page.locator("#ixstats").inner_text()
        assert page.locator(f".runrow .dot.{expected}").count() == 1
        page.locator(".runrow").click()
        page.wait_for_selector("#hdr .badge")
        assert page.locator(f"#hdr .badge.{expected}").count() == 1
        browser.close()
