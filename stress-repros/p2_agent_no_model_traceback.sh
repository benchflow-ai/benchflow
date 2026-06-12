#!/usr/bin/env bash
# P2-4: `--agent codex` with no `--model` raises an uncaught ValueError from effective_model
# (evaluation.py:276), shown as a Rich traceback instead of "agent 'codex' requires --model".
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess
r = subprocess.run(['uv','run','bench','eval','create','--tasks-dir','/tmp/bf-stress/oracle_smoke_tasks',
                    '--agent','codex','--sandbox','docker','--concurrency','1'],
                   capture_output=True, text=True, timeout=60)
out = r.stdout + r.stderr
print("ACTUAL RC", r.returncode, "| traceback present:", "Traceback (most recent call last)" in out, "| ValueError:", "ValueError" in out)
print(out[-500:])
print("EXPECTED: clean 'agent codex has no default model; pass --model'.")
PY
