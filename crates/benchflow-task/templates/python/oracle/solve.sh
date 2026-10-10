#!/bin/bash
python3 - <<'PY'
import csv
import json

rows = list(csv.DictReader(open("/app/scores.csv")))
scores = [int(r["score"]) for r in rows]
best = max(rows, key=lambda r: int(r["score"]))
summary = {
    "count": len(rows),
    "mean": sum(scores) / len(scores),
    "max": {"name": best["name"], "score": int(best["score"])},
}
json.dump(summary, open("/app/summary.json", "w"))
PY
