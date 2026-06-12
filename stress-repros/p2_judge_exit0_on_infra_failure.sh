#!/usr/bin/env bash
# P2-6: the shipped example judge (generated-skill-eval/*/verifier/judge.py) exits 0 and writes
# reward 0.0 when NO LLM judge could run (SDK/key/API failure). A judge INFRASTRUCTURE failure is
# silently recorded as a legitimate 0.0 score, indistinguishable from "agent did the work but wrong."
# Here we scrub all provider keys so no judge can run, then show RC=0 + reward 0.0.
set -u
cd "$(git rev-parse --show-toplevel)"
SRC=docs/examples/task-md/generated-skill-eval/models-as-skills/regex-email-parser/verifier
WORK=$(mktemp -d); LOGS=$(mktemp -d); VDIR=$(mktemp -d)
cp -R "$SRC/." "$VDIR/"
cat > "$WORK/solution.py" <<'PY'
import re
def parse_email_addresses(text):
    out=[]
    for m in re.finditer(r'<?([\w.+-]+)@([\w.-]+\.\w+)>?', text):
        out.append({'local':m.group(1),'domain':m.group(2),'full':m.group(1)+'@'+m.group(2)})
    return out
PY
echo '[{"role":"assistant","content":"wrote parse_email_addresses"}]' > "$LOGS/acp_trajectory.jsonl"
echo "Running judge.py with ALL provider keys scrubbed (simulating judge infra failure)..."
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY -u GEMINI_API_KEY \
    JUDGE_MODEL=gemini-3.1-flash-lite \
    BENCHFLOW_VERIFIER_DIR="$VDIR" BENCHFLOW_WORKSPACE="$WORK" BENCHFLOW_AGENT_LOG_DIR="$LOGS" \
    uv run python "$VDIR/judge.py"
echo "judge.py EXIT CODE = $?   <-- EXPECTED: non-zero (verifier-infra error).  ACTUAL: 0"
echo "reward written:"; cat "$VDIR/reward.txt" 2>/dev/null || find "$VDIR" "$WORK" -name 'reward*.json' -exec cat {} \;
echo "=> a 0.0 from 'no judge ran' is indistinguishable from a real agent-failure 0.0."
rm -rf "$WORK" "$LOGS" "$VDIR"
