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
        json.dumps({"as_of": "2026-06-03T00:00:00+00:00", "target": target, "rows": rows})
    )
    return p


def test_snapshot_counts_states_and_buckets(tmp_path):
    rows = [
        {"cell_id": "a", "status": "queued"},
        {"cell_id": "b", "status": "queued"},
        {"cell_id": "c", "status": "running"},
        {"cell_id": "d", "status": "completed"},
        {"cell_id": "e", "status": "review_fail"},
        {"cell_id": "f", "status": "quarantined"},
        {"cell_id": "g", "status": "review_pass"},
        {"cell_id": "h", "status": "published"},
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
    r = experiments_status.snapshot(_write(tmp_path, [{"cell_id": "x"}], target=1))
    assert r["summary"]["by_state"]["queued"] == 1
    assert r["summary"]["by_bucket"]["queue"] == 1


def test_snapshot_target_falls_back_to_row_count(tmp_path):
    p = tmp_path / "experiments_ledger.json"
    p.write_text(json.dumps({"rows": [{"status": "queued"}, {"status": "running"}]}))
    r = experiments_status.snapshot(p)
    assert r["target"] == 2
    assert r["as_of"]  # filled with now() when absent from the ledger


def test_snapshot_missing_file_returns_error(tmp_path):
    r = experiments_status.snapshot(tmp_path / "nope.json")
    assert "error" in r
    assert r["rows"] == []
    assert r["summary"]["total"] == 0
    assert r["summary"]["by_bucket"] == {"queue": 0, "running": 0, "completed": 0, "reviewed": 0}


def test_snapshot_corrupt_file_returns_error(tmp_path):
    p = tmp_path / "experiments_ledger.json"
    p.write_text("{not valid json")
    r = experiments_status.snapshot(p)
    assert "error" in r and "unreadable" in r["error"]
    assert r["rows"] == []
