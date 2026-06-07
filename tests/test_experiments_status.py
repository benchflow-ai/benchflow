"""Unit tests for the dashboard Experiments panel's snapshot() domain logic.

Mirrors ``test_daytona_status``: exercises ledger summarization (state counts,
the four-bucket progressive rollup, target passthrough/fallback) and the
error-returning contract, without a live server or network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import experiments_status  # noqa: E402


def _write(tmp_path, rows, target=10):
    p = tmp_path / "experiments_ledger.json"
    p.write_text(
        json.dumps(
            {"as_of": "2026-06-03T00:00:00+00:00", "target": target, "rows": rows}
        )
    )
    return p


def _healthy_review(**overrides):
    row = {
        "status": "review_pass",
        "health": "healthy",
        "review_checklist": {
            "complete_traj": True,
            "meta": True,
            "no_reward_hacking": True,
        },
        "skill_mode": "with",
        "task_skills_loading": 1,
        "sandbox": "daytona",
        "reward": 1.0,
        "tokens": {"total": 123},
        "timing_total_s": 45,
        "hf_path": "openhands/model/task/trial/trajectory.jsonl",
    }
    row.update(overrides)
    return row


def test_snapshot_counts_states_and_buckets(tmp_path):
    """Guards PR #622 against counting unreviewed rows as reviewed-healthy."""
    rows = [
        {"cell_id": "a", "status": "queued"},
        {"cell_id": "b", "status": "queued"},
        {"cell_id": "c", "status": "running"},
        {"cell_id": "d", "status": "completed"},
        {"cell_id": "e", "status": "review_fail"},
        {"cell_id": "f", "status": "quarantined"},
        _healthy_review(cell_id="g", status="review_pass"),
        _healthy_review(cell_id="h", status="published"),
    ]
    r = experiments_status.snapshot(_write(tmp_path, rows, target=42))

    assert "error" not in r
    assert r["target"] == 42
    assert r["summary"]["total"] == 8
    bs = r["summary"]["by_state"]
    assert bs["queued"] == 2 and bs["running"] == 1 and bs["completed"] == 1
    assert bs["review_fail"] == 1 and bs["quarantined"] == 1
    assert bs["review_pass"] == 1 and bs["published"] == 1
    bb = r["summary"]["by_bucket"]
    # progressive, non-overlapping rollup into the four panels
    assert bb["queue"] == 2
    assert bb["running"] == 1
    assert bb["completed"] == 3  # completed + review_fail + quarantined
    assert bb["reviewed"] == 2  # review_pass + published
    assert len(r["rows"]) == 8


def test_snapshot_defaults_missing_status_to_queued(tmp_path):
    """Guards PR #622 against dropping rows that lack a status."""
    r = experiments_status.snapshot(_write(tmp_path, [{"cell_id": "x"}], target=1))
    assert r["summary"]["by_state"]["queued"] == 1
    assert r["summary"]["by_bucket"]["queue"] == 1


def test_snapshot_target_falls_back_to_skillsbench_matrix_size(tmp_path, monkeypatch):
    """Guards PR #622 against treating a partial ledger as a complete target."""
    monkeypatch.delenv("EXPERIMENTS_TARGET", raising=False)
    p = tmp_path / "experiments_ledger.json"
    p.write_text(json.dumps({"rows": [{"status": "queued"}, {"status": "running"}]}))
    r = experiments_status.snapshot(p)
    assert r["target"] == 1584
    assert r["summary"]["missing"] == 1582
    assert r["as_of"]  # filled with now() when absent from the ledger


def test_snapshot_allows_target_env_override(tmp_path, monkeypatch):
    """Guards PR #622 against hard-coding the target when operators override it."""
    monkeypatch.setenv("EXPERIMENTS_TARGET", "12")
    p = tmp_path / "experiments_ledger.json"
    p.write_text(json.dumps({"rows": [{"status": "queued"}]}))
    r = experiments_status.snapshot(p)
    assert r["target"] == 12
    assert r["summary"]["missing"] == 11


def test_review_pass_requires_complete_health_gate(tmp_path):
    """Guards PR #622 against accepting incomplete review artifacts as healthy."""
    p = _write(
        tmp_path,
        [
            _healthy_review(cell_id="ok"),
            _healthy_review(cell_id="missing_tokens", tokens={}),
            _healthy_review(cell_id="missing_skill", task_skills_loading=0),
        ],
        target=3,
    )
    r = experiments_status.snapshot(p)

    assert r["summary"]["by_state"]["review_pass"] == 3
    assert r["summary"]["by_bucket"]["reviewed"] == 1
    assert r["summary"]["by_bucket"]["completed"] == 2
    bad = {row["cell_id"]: row for row in r["rows"] if not row["review_health_ok"]}
    assert bad["missing_tokens"]["dashboard_bucket"] == "completed"
    assert "token usage missing" in bad["missing_tokens"]["review_health_notes"]
    assert (
        "skill/no-skill detection missing or failed"
        in bad["missing_skill"]["review_health_notes"]
    )


def test_published_hf_seeded_rows_count_as_reviewed(tmp_path):
    """Guards PR #622 against hiding healthy HF-seeded PR rows from progress."""
    p = _write(
        tmp_path,
        [
            _healthy_review(
                cell_id="hf_seeded",
                status="published",
                review_checklist=None,
                review_verdict=None,
                tokens=319293,
            ),
            _healthy_review(
                cell_id="redacted",
                status="published",
                review_checklist=None,
                review_verdict=None,
                tokens="[REDACTED]",
            ),
        ],
        target=2,
    )
    r = experiments_status.snapshot(p)

    assert r["summary"]["by_bucket"]["reviewed"] == 1
    redacted = next(row for row in r["rows"] if row["cell_id"] == "redacted")
    assert redacted["dashboard_bucket"] == "completed"
    assert "token usage missing" in redacted["review_health_notes"]


def test_partial_rows_require_explicit_timeout_overlay(tmp_path):
    """Guards PR #638 follow-up against dashboard crediting raw partial timeouts."""
    p = _write(
        tmp_path,
        [
            _healthy_review(
                cell_id="raw_partial",
                status="published",
                partial_trajectory=True,
                error="Agent timed out after 900s",
                timeout_complete_artifacts=True,
            ),
            _healthy_review(
                cell_id="summary_partial",
                status="published",
                trajectory_summary={"partial_trajectory": True},
                error="Agent timed out after 900s",
                timeout_complete_artifacts=True,
            ),
            _healthy_review(
                cell_id="accepted_timeout",
                status="published",
                partial_trajectory=True,
                error="Agent timed out after 900s",
                timeout_complete_artifacts=True,
                accepted_normal_timeout=True,
            ),
        ],
        target=3,
    )
    r = experiments_status.snapshot(p)

    assert r["summary"]["by_bucket"]["reviewed"] == 1
    raw = next(row for row in r["rows"] if row["cell_id"] == "raw_partial")
    assert raw["dashboard_bucket"] == "completed"
    assert (
        "partial trajectory not accepted by strict timeout overlay"
        in raw["review_health_notes"]
    )


def test_without_skills_rows_must_not_load_or_access_skills(tmp_path):
    """Guards PR #622 against no-skill leakage being marked reviewed-healthy."""
    p = _write(
        tmp_path,
        [
            _healthy_review(
                cell_id="clean",
                skill_mode="without",
                task_skills_loading=0,
                no_skill_leakage=True,
            ),
            _healthy_review(
                cell_id="leaked",
                skill_mode="without",
                task_skills_loading=0,
                skill_files_accessed=[".claude/skills/task/SKILL.md"],
            ),
        ],
        target=2,
    )
    r = experiments_status.snapshot(p)

    assert r["summary"]["by_bucket"]["reviewed"] == 1
    leaked = next(row for row in r["rows"] if row["cell_id"] == "leaked")
    assert leaked["dashboard_bucket"] == "completed"
    assert "skill/no-skill detection missing or failed" in leaked["review_health_notes"]


def test_snapshot_missing_file_returns_error(tmp_path):
    """Guards PR #622 against live dashboard 500s when the ledger is missing."""
    r = experiments_status.snapshot(tmp_path / "nope.json")
    assert "error" in r
    assert r["rows"] == []
    assert r["summary"]["total"] == 0
    assert r["summary"]["by_bucket"] == {
        "queue": 0,
        "running": 0,
        "completed": 0,
        "reviewed": 0,
    }


def test_snapshot_corrupt_file_returns_error(tmp_path):
    """Guards PR #622 against live dashboard 500s on corrupt ledger JSON."""
    p = tmp_path / "experiments_ledger.json"
    p.write_text("{not valid json")
    r = experiments_status.snapshot(p)
    assert "error" in r and "unreadable" in r["error"]
    assert r["rows"] == []
