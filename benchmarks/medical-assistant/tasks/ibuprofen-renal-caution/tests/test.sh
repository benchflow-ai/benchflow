#!/usr/bin/env bash
# Deterministic verifier: score /app/answer.md against this task's keyword groups
# (OR within a group, AND across groups). reward = matched_groups / total_groups.
set -uo pipefail
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS_DIR"
ANSWER="${ANSWER_FILE:-/app/answer.md}"
GT="$(dirname "$0")/ground_truth.json"
python3 - "$ANSWER" "$GT" "$LOGS_DIR/reward.txt" <<'PYEOF'
import json, os, sys
answer_path, gt_path, reward_path = sys.argv[1], sys.argv[2], sys.argv[3]
text = ""
if os.path.exists(answer_path):
    text = open(answer_path, encoding="utf-8", errors="ignore").read().lower()
groups = json.load(open(gt_path))["keyword_groups"]
hit = sum(1 for grp in groups if any(kw.lower() in text for kw in grp))
reward = hit / len(groups) if groups else 0.0
open(reward_path, "w").write(f"{reward:.4f}\n")
print(f"matched {hit}/{len(groups)} keyword groups -> reward {reward:.4f}")
if not text:
    print("WARN: /app/answer.md missing or empty")
PYEOF
cat "$LOGS_DIR/reward.txt"
