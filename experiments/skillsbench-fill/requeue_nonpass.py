#!/usr/bin/env python3
"""Fresh-start requeue: reset EVERY non-pass cell (fail + quarantine) to queued so
the runners re-run them — after #914's skill fix is applied, the skill_posture fails
should now pass; gemini 429 / opus Bedrock / infra get another attempt."""
import json, glob, os
REV = os.path.expanduser("~/sb-fill/review")
ST = os.path.expanduser("~/sb-fill/state")
SKIP = {"experiments_ledger.json", "queue.jsonl", "reconcile_report.json", "grid.json"}
n = 0
for f in glob.glob(REV + "/*.json"):
    try:
        rv = json.load(open(f))
    except Exception:
        continue
    if rv.get("verdict") == "pass":
        continue
    cell = rv.get("cell_id") or os.path.basename(f)[:-5]
    os.remove(f)
    sf = os.path.join(ST, cell + ".json")
    if os.path.exists(sf):
        os.remove(sf)
    n += 1
r = 0
for f in glob.glob(ST + "/*.json"):
    if os.path.basename(f) in SKIP:
        continue
    try:
        st = json.load(open(f))
    except Exception:
        continue
    if st.get("status") == "running":
        os.remove(f); r += 1
print(f"requeued non-pass: {n} | reset orphaned running: {r}")
