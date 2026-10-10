#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${BENCHFLOW_WORKSPACE:-/workspace}"
REWARD_TEXT="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
REWARD_JSON="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
DETAILS_JSON="${BENCHFLOW_REWARD_DETAILS_JSON:-/logs/verifier/reward-details.json}"

mkdir -p "$(dirname "$REWARD_TEXT")" "$(dirname "$REWARD_JSON")" "$(dirname "$DETAILS_JSON")"

python - "$WORKSPACE" "$REWARD_TEXT" "$REWARD_JSON" "$DETAILS_JSON" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
reward_text = Path(sys.argv[2])
reward_json = Path(sys.argv[3])
details_json = Path(sys.argv[4])

checks = {
    "order_id_file_present": False,
    "order_id_matches_private_fact": False,
    "recovery_json_present": False,
    "recovery_json_valid": False,
    "recovery_order_id_matches": False,
    "recovery_has_failure_point": False,
    "recovery_has_next_action": False,
}

order_id_path = workspace / "order_id.txt"
if order_id_path.is_file():
    checks["order_id_file_present"] = True
    checks["order_id_matches_private_fact"] = (
        order_id_path.read_text().strip() == "BF-1042"
    )

recovery_path = workspace / "recovery.json"
if recovery_path.is_file():
    checks["recovery_json_present"] = True
    try:
        recovery = json.loads(recovery_path.read_text())
    except json.JSONDecodeError:
        recovery = {}
    else:
        checks["recovery_json_valid"] = isinstance(recovery, dict)
        checks["recovery_order_id_matches"] = recovery.get("order_id") == "BF-1042"
        checks["recovery_has_failure_point"] = bool(recovery.get("failure_point"))
        checks["recovery_has_next_action"] = bool(recovery.get("next_action"))

reward = 1.0 if all(checks.values()) else 0.0
reward_text.write_text(f"{reward:.1f}\n")
reward_json.write_text(json.dumps({"reward": reward}) + "\n")
details_json.write_text(
    json.dumps(
        {
            "reward": reward,
            "checks": checks,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
