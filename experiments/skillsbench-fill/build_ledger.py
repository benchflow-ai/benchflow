#!/usr/bin/env python3
"""Merge the base grid + per-cell runner/review/publish state -> experiments_ledger.json.

Reads (under --root):
  grid.json            base 1638-cell grid from reconcile (immutable)
  state/<cell>.json    runner state (run_cell.sh): running|completed|run_failed (+reward/timing/tokens)
  review/<cell>.json   review verdict: {verdict: pass|fail|quarantine, health, task_skills_loading, note}
  published/<cell>.json publish record: {hf_path}

Writes experiments_ledger.json (atomic) for the dashboard. Status precedence:
published > review_* > completed/run_failed/running > queued (later overlays win).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

RECONCILE_FILES = {
    "experiments_ledger.json",
    "queue.jsonl",
    "repair_queue.jsonl",
    "reconcile_report.json",
    "grid.json",
    "raw_pool_availability.json",
}
RUNNER_KEYS = ("status", "sandbox", "run_root", "rollout_dir", "reward", "error", "partial",
               "timing_total_s", "tokens", "usage_source", "started_at", "updated_at", "enospc")


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)

    base = _load(root / "grid.json") or _load(root / "state" / "experiments_ledger.json")
    if not base:
        raise SystemExit("no base grid (grid.json) — run reconcile.py first")
    rows = {r["cell_id"]: dict(r) for r in base["rows"]}

    # 1) runner state
    for f in glob.glob(str(root / "state" / "*.json")):
        if os.path.basename(f) in RECONCILE_FILES:
            continue
        st = _load(f)
        if not st or st.get("cell_id") not in rows:
            continue
        r = rows[st["cell_id"]]
        for k in RUNNER_KEYS:
            if st.get(k) is not None:
                r[k] = st[k]

    # 2) review verdicts
    for f in glob.glob(str(root / "review" / "*.json")):
        rv = _load(f)
        if not rv or rv.get("cell_id") not in rows:
            continue
        r = rows[rv["cell_id"]]
        r["review_verdict"] = rv.get("verdict")
        if rv.get("health") is not None:
            r["health"] = rv["health"]
        if rv.get("task_skills_loading") is not None:
            r["task_skills_loading"] = rv["task_skills_loading"]
        if rv.get("note"):
            r["note"] = rv["note"]
        # carry the measured metrics the review recorded (reward/tokens/timing/effort/loaded skills)
        # so the dashboard's Reviewed panel shows them — a 'healthy' verdict implies these exist.
        for k in (
            "reward",
            "tokens",
            "timing_total_s",
            "effort_ok",
            "loaded_task_skills",
            "usage_source",
            "rollout_dir",
            "trial_id",
            "partial_trajectory",
            "trajectory_summary_partial",
            "trajectory_source",
            "timeout_complete_artifacts",
            "accepted_normal_timeout",
            "opus_adaptive_thinking",
            "opus_output_config_effort_max",
        ):
            if rv.get(k) is not None:
                r[k] = rv[k]
        if rv.get("checklist") is not None:
            r["review_checklist"] = rv["checklist"]
        if rv.get("partial_trajectory") is not None:
            r["partial"] = rv["partial_trajectory"]
        r["status"] = {"pass": "review_pass", "fail": "review_fail",
                       "quarantine": "quarantined"}.get(rv.get("verdict"), r.get("status"))
        if rv.get("updated_at"):
            r["updated_at"] = rv["updated_at"]

    # 3) publish records (terminal). A publish record may point at an
    # already-existing HF trial id/path when publish.py dedups a rerun. Do not
    # credit that duplicate path into another trial slot.
    used_hf_paths = {
        r["hf_path"]
        for r in rows.values()
        if r.get("status") == "published" and r.get("hf_path")
    }
    for f in glob.glob(str(root / "published" / "*.json")):
        pb = _load(f)
        if not pb or pb.get("cell_id") not in rows:
            continue
        r = rows[pb["cell_id"]]
        hf_path = pb.get("hf_path")
        if hf_path and hf_path in used_hf_paths and r.get("hf_path") != hf_path:
            r["duplicate_hf_path"] = hf_path
            r["note"] = "duplicate HF trial id/path; not credited as a new published cell"
            if pb.get("updated_at"):
                r["updated_at"] = pb["updated_at"]
            continue
        reviewed_tid = r.get("trial_id")
        publish_tid = pb.get("source_tid") or pb.get("tid")
        hf_leaf = (hf_path or "").rstrip("/").split("/")[-1]
        hf_tid = hf_leaf.rsplit("__", 1)[1] if "__" in hf_leaf else None
        raw_partial = bool(r.get("partial") or r.get("partial_trajectory"))
        accepted_timeout = bool(
            r.get("accepted_normal_timeout") and r.get("timeout_complete_artifacts")
        )
        publish_creditable = (
            r.get("review_verdict") == "pass"
            and r.get("health") == "healthy"
            and (not raw_partial or accepted_timeout)
            and (not reviewed_tid or not publish_tid or str(reviewed_tid) == str(publish_tid))
            and (not reviewed_tid or not hf_tid or str(reviewed_tid) == str(hf_tid))
        )
        if not publish_creditable:
            r["uncredited_hf_path"] = hf_path
            r["note"] = "publish record is not strict-creditable for this reviewed cell"
            if pb.get("updated_at"):
                r["updated_at"] = pb["updated_at"]
            continue
        r["status"] = "published"
        r["hf_path"] = hf_path
        if pb.get("accepted_normal_timeout") is not None:
            r["accepted_normal_timeout"] = pb["accepted_normal_timeout"]
        if hf_path:
            used_hf_paths.add(hf_path)
        if pb.get("updated_at"):
            r["updated_at"] = pb["updated_at"]

    out = {
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": base.get("target", len(rows)),
        "rows": list(rows.values()),
    }
    dest = a.out or str(root / "experiments_ledger.json")
    tmp = dest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, dest)
    print("ledger:", dict(Counter(r.get("status", "queued") for r in rows.values())), "->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
