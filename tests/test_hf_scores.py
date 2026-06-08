"""Unit tests for the Experiments page's HF PR score aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import hf_scores  # noqa: E402


def test_build_scoreboard_combines_hf_pr_modes(monkeypatch):
    """Guards the HF-score dashboard PR against mixing pass-rate and fill progress."""

    def fake_analysis(buckets):
        hf_scores._add_bucket(
            buckets,
            pr=2,
            harness="openhands",
            model="gpt-5.5",
            mode="with-skills",
            passed=6,
            failed=4,
        )
        hf_scores._add_bucket(
            buckets,
            pr=2,
            harness="openhands",
            model="gpt-5.5",
            mode="without-skills",
            passed=2,
            failed=8,
        )
        return []

    def fake_direct(buckets):
        hf_scores._add_bucket(
            buckets,
            pr=5,
            harness="openhands",
            model="gpt-5.5",
            mode="with-skills",
            passed=1,
            failed=0,
        )
        return []

    monkeypatch.setattr(hf_scores, "_read_analysis_prs", fake_analysis)
    monkeypatch.setattr(hf_scores, "_read_direct_prs", fake_direct)

    result = hf_scores.build_scoreboard()

    with_row = result["by_mode"]["with-skills"][0]
    assert with_row["label"] == "OpenHands GPT-5.5"
    assert with_row["passed"] == 7
    assert with_row["total"] == 11
    assert with_row["prs"] == [2, 5]
    gain_row = result["by_mode"]["normalized-gain"][0]
    assert gain_row["without_pass_rate"] == 0.2
    assert round(gain_row["gain"], 3) == 0.545


def test_snapshot_serves_cached_hf_scores_without_inline_refresh(tmp_path, monkeypatch):
    """Guards the HF-score dashboard PR against blocking live requests on HF."""
    cache = {
        "as_of": "2026-06-08T00:00:00+00:00",
        "source": "HuggingFace PR2/PR3/PR4/PR5",
        "repo": hf_scores.REPO,
        "refs": ["refs/pr/2", "refs/pr/3", "refs/pr/4", "refs/pr/5"],
        "scored_trials": 1,
        "groups": 1,
        "by_mode": {"with-skills": [], "without-skills": [], "normalized-gain": []},
        "warnings": [],
        "warning_count": 0,
    }
    path = tmp_path / "hf_scoreboard_cache.json"
    path.write_text(json.dumps(cache))

    def fail_refresh():
        raise AssertionError("snapshot should not refresh inline by default")

    monkeypatch.setattr(hf_scores, "build_scoreboard", fail_refresh)

    result = hf_scores.snapshot(path)

    assert result["cached"] is True
    assert result["scored_trials"] == 1
