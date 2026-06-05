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
from datetime import datetime, timezone
from pathlib import Path

RECONCILE_FILES = {"experiments_ledger.json", "queue.jsonl", "reconcile_report.json", "grid.json"}
RUNNER_KEYS = ("status", "sandbox", "run_root", "rollout_dir", "reward", "error", "partial",
               "timing_total_s", "tokens", "usage_source", "started_at", "updated_at", "enospc")


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
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
        for k in ("reward", "tokens", "timing_total_s", "effort_ok", "loaded_task_skills", "usage_source"):
            if rv.get(k) is not None:
                r[k] = rv[k]
        r["status"] = {"pass": "review_pass", "fail": "review_fail",
                       "quarantine": "quarantined"}.get(rv.get("verdict"), r.get("status"))
        if rv.get("updated_at"):
            r["updated_at"] = rv["updated_at"]

    # 3) publish records (terminal)
    for f in glob.glob(str(root / "published" / "*.json")):
        pb = _load(f)
        if not pb or pb.get("cell_id") not in rows:
            continue
        r = rows[pb["cell_id"]]
        r["status"] = "published"
        r["hf_path"] = pb.get("hf_path")
        if pb.get("updated_at"):
            r["updated_at"] = pb["updated_at"]

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": base.get("target", len(rows)),
        "rows": list(rows.values()),
    }
    dest = a.out or str(root / "experiments_ledger.json")
    tmp = dest + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, dest)
    print("ledger:", dict(Counter(r.get("status", "queued") for r in rows.values())), "->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
