"""Rubric tab of the interactive viewer (issue #1101)."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from benchflow.trajectories.viewer.payload import _build_acp_payload, _load_rubric


def _rollout(jobs: Path, name: str) -> Path:
    rollout = jobs / "2026-01-01__00-00-00" / name
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_call",
                "tool_call_id": "t1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [],
            }
        )
    )
    (rollout / "result.json").write_text(json.dumps({"task_name": "t", "model": "m"}))
    return rollout


def _report(out_dir: Path, name: str, *, valid: bool, gated: float = 0.5) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial: dict = {
        "trial_name": name,
        "review_valid": valid,
        "summary": "Derivation complete, one unsupported claim.",
        "criterion_metadata": [
            {"name": "no_fabrication", "blocker": 1, "weight": 1},
            {"name": "derivation", "blocker": 0, "weight": 5},
        ],
        "checks": {
            "no_fabrication": {"outcome": "pass", "explanation": "cites the data"},
            "derivation": {"score": 1, "explanation": "skips one step"},
        },
        "scoring": {
            "deterministic_pass": True,
            "all_blockers_pass": True,
            "failed_blockers": [],
            "raw_quality": gated,
            "gated_quality": gated,
            "decision": "presentable_with_revisions",
        }
        if valid
        else None,
    }
    path = out_dir / "review_report.json"
    path.write_text(
        json.dumps({"reviewer": {"model": "gateway/reviewer-x"}, "trials": [trial]})
    )
    return path


def test_rubric_is_found_in_the_sibling_review_dir(tmp_path: Path) -> None:
    """The bench review default layout (jobs/review-<stamp>/ next to the run
    directory) resolves without any configuration."""
    jobs = tmp_path / "jobs"
    rollout = _rollout(jobs, "task__abcd1234")
    _report(
        jobs / "review-2026-01-02__00-00-00", "task__abcd1234", valid=True, gated=0.7
    )
    rubric = _load_rubric(rollout)
    assert rubric is not None
    assert rubric.reviewer_model == "gateway/reviewer-x"
    assert rubric.scoring["gated_quality"] == 0.7
    assert [c.name for c in rubric.criteria] == ["no_fabrication", "derivation"]
    assert rubric.criteria[0].blocker and rubric.criteria[0].outcome == "pass"
    assert not rubric.criteria[1].blocker and rubric.criteria[1].score == 1


def test_valid_review_wins_over_an_invalid_one_and_last_sorted_breaks_ties(
    tmp_path: Path,
) -> None:
    """An invalid (unscored) entry never shadows a valid one; among valid
    entries the last report in sorted order wins."""
    jobs = tmp_path / "jobs"
    rollout = _rollout(jobs, "task__abcd1234")
    _report(jobs / "review-a", "task__abcd1234", valid=True, gated=0.2)
    _report(jobs / "review-b", "task__abcd1234", valid=True, gated=0.9)
    _report(jobs / "review-c", "task__abcd1234", valid=False)
    rubric = _load_rubric(rollout)
    assert rubric is not None
    assert rubric.scoring["gated_quality"] == 0.9


def test_other_rollouts_reports_and_missing_reports_yield_none(tmp_path: Path) -> None:
    """A report for a different trial_name is not this rollout's review."""
    jobs = tmp_path / "jobs"
    rollout = _rollout(jobs, "task__abcd1234")
    assert _load_rubric(rollout) is None
    _report(jobs / "review-x", "other__00000000", valid=True)
    assert _load_rubric(rollout) is None


def test_payload_carries_the_rubric_section(tmp_path: Path) -> None:
    """The rubric rides in the payload as its own section, null when absent."""
    jobs = tmp_path / "jobs"
    rollout = _rollout(jobs, "task__abcd1234")
    assert _build_acp_payload(rollout, None).to_payload()["rubric"] is None
    _report(jobs / "review-x", "task__abcd1234", valid=True, gated=0.4)
    section = _build_acp_payload(rollout, None).to_payload()["rubric"]
    assert section["scoring"]["decision"] == "presentable_with_revisions"
    assert section["criteria"][1] == {
        "name": "derivation",
        "blocker": False,
        "weight": 5,
        "outcome": None,
        "score": 1,
        "explanation": "skips one step",
    }


def test_rubric_tab_renders_in_the_browser(tmp_path: Path) -> None:
    """The Rubric tab appears for a reviewed rollout and lists the criteria."""
    pw = pytest.importorskip("playwright.sync_api")
    from benchflow.trajectories.viewer.server import _serve_browse

    jobs = tmp_path / "jobs"
    _rollout(jobs, "task__abcd1234")
    _report(jobs / "review-x", "task__abcd1234", valid=True, gated=0.4)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(
        target=_serve_browse,
        args=(jobs, port, 1),
        kwargs={"capped": False},
        daemon=True,
    ).start()

    with pw.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(f"http://localhost:{port}/")
        page.locator(".runrow", has_text="task__abcd1234").first.click()
        page.get_by_role("tab", name="Rubric").click()
        pane = page.locator("#view-rubric")
        assert "no_fabrication" in pane.inner_text()
        assert "presentable_with_revisions" in pane.inner_text()
        browser.close()
