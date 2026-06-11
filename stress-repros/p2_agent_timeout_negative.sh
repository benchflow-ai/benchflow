#!/usr/bin/env bash
# P2-1: agent.timeout_sec accepts negative/zero (config.py:333 has no constraint) while
# verifier.timeout_sec rejects the same (config.py:244-247, gt=0). Validation asymmetry.
set -u
cd "$(git rev-parse --show-toplevel)"
D=$(mktemp -d)
cp -R docs/examples/task-md/real-skillsbench/weighted-gdp-calc/. "$D/"
python3 - "$D" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "task.md"
t = p.read_text()
import re
t = re.sub(r'timeout_sec:\s*\d+', 'timeout_sec: -5', t, count=1)  # first timeout = agent's
p.write_text(t)
PY
echo "--- agent.timeout_sec = -5 ---"
uv run bench tasks check "$D" --level schema; echo "schema RC=$?"
uv run bench tasks check "$D" --level structural; echo "structural RC=$?"
echo "EXPECTED: RC!=0 (rejected, like verifier.timeout_sec).  ACTUAL: RC=0 (silently valid)."
echo "Direct proof:"; uv run python -c "from benchflow.task.config import AgentConfig; print('AgentConfig(timeout_sec=-5) ->', AgentConfig(timeout_sec=-5).timeout_sec)"
rm -rf "$D"
