#!/usr/bin/env python3
"""Requeue storm-damaged cells: delete state+review for every quarantine-verdict cell
(and any orphaned running-state cell) so the runner re-runs them with DNS fixed."""
import json, glob, os
REV = os.path.expanduser("~/sb-fill/review")
ST = os.path.expanduser("~/sb-fill/state")
SKIP = {"queue.jsonl", "experiments_ledger.json", "grid.json", "reconcile_report.json"}

q = 0
for f in glob.glob(REV + "/*.json"):
    try:
        rv = json.load(open(f))
    except Exception:
        continue
    if rv.get("verdict") != "quarantine":
        continue
    cell = rv.get("cell_id") or os.path.basename(f)[:-5]
    os.remove(f)                                  # drop review verdict
    sf = os.path.join(ST, cell + ".json")
    if os.path.exists(sf):
        os.remove(sf)                             # drop state -> runner re-runs
    q += 1

r = 0
for f in glob.glob(ST + "/*.json"):
    if os.path.basename(f) in SKIP:
        continue
    try:
        st = json.load(open(f))
    except Exception:
        continue
    if st.get("status") == "running":             # orphaned by reboot/kill
        os.remove(f); r += 1
print(f"requeued quarantine cells: {q} | reset orphaned running: {r}")
