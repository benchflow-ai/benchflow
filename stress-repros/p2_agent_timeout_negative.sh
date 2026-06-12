#!/usr/bin/env bash
# P2-1: agent.timeout_sec accepts negative/zero (config.py:333 has no constraint) while
# verifier.timeout_sec rejects the same (config.py:244-247, gt=0). Validation asymmetry.
set -u
cd "$(git rev-parse --show-toplevel)"
D=$(mktemp -d)
cp -R docs/examples/task-md/real-skillsbench/weighted-gdp-calc/. "$D/"
python3 - "$D" <<'PY'
import pathlib, sys, re
p = pathlib.Path(sys.argv[1]) / "task.md"
t = p.read_text()
# Target the AGENT block specifically (NOT the verifier block, which is validated and would
# mask the defect). Matches `agent:\n  timeout_sec: <n>` and only that.
t2 = re.sub(r'(agent:\s*\n\s*timeout_sec:\s*)\d+', r'\g<1>-5', t, count=1)
assert t2 != t, "agent.timeout_sec block not found — task.md layout changed"
p.write_text(t2)
PY
echo "--- agent.timeout_sec = -5 ---"
uv run bench tasks check "$D" --level schema; echo "schema RC=$?"
uv run bench tasks check "$D" --level structural; echo "structural RC=$?"
echo "EXPECTED: RC!=0 (rejected, like verifier.timeout_sec).  ACTUAL: RC=0 (silently valid)."
echo "Direct proof:"; uv run python -c "from benchflow.task.config import AgentConfig; print('AgentConfig(timeout_sec=-5) ->', AgentConfig(timeout_sec=-5).timeout_sec)"
rm -rf "$D"
