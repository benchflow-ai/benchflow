#!/usr/bin/env bash
# CasinoBench verifier (env-0 style). The casino mock-service runs INSIDE this
# same sandbox on localhost:9001 (started by the BenchFlow Environment plane from
# benchmarks/casinobench/environment.toml). The service is the chip authority —
# the agent plays only through the `casino` CLI and cannot fake the count — so we
# read the live standing (final chips) from /_admin/state and score it. No replay.
#
# Reward = NET chips (final - starting): did-nothing -> 0, a loss is negative, a
# profit positive. A failed READ must NOT score a fabricated 0 (indistinguishable
# from a real break-even): abort nonzero with no reward file so BenchFlow records
# a verifier error instead.
#
# This file is the single shared verifier for every casinobench game task; each
# task points its `tests/test.sh` here (symlink). It writes the bare scalar to
# /logs/verifier/reward.txt and the structured map to reward.json.
set -euo pipefail
BASE="${CASINO_URL:-http://localhost:9001}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS_DIR"

# Readiness was gated before the agent ran, but poll briefly in case the verifier
# starts before a slow process settles.
for _ in $(seq 1 30); do
  curl -sf "$BASE/health" >/dev/null 2>&1 && break
  sleep 1
done

curl -fsS "$BASE/_admin/state" -o /tmp/casino_state.json

python3 - "$LOGS_DIR" <<'PY'
import json, sys
from pathlib import Path

logs = Path(sys.argv[1])
# Read the service's authoritative chip count. Missing/unreadable/no-count ->
# cannot score: exit nonzero, write NO reward file (a fabricated 0 is
# indistinguishable from a legitimate break-even).
try:
    state = json.loads(Path("/tmp/casino_state.json").read_text())
    final = int(state["final_chips"])
    start = int(state.get("starting_bankroll", 1000))
except (OSError, ValueError, TypeError, KeyError) as exc:
    sys.stderr.write(f"casino verifier: cannot read final chips: {exc}\n")
    raise SystemExit(2)

reward = float(final - start)
out = {
    "reward": reward,
    "details": {
        "game": state.get("game"),
        "subject": state.get("subject"),
        "final_chips": final,
        "starting_bankroll": start,
        "metric": "net",
    },
}
(logs / "reward.json").write_text(json.dumps(out, indent=2))
(logs / "reward.txt").write_text(f"{reward}\n")
print(json.dumps(out))
PY

cat "$LOGS_DIR/reward.txt"
