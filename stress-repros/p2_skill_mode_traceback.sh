#!/usr/bin/env bash
# P2-3: `--skill-mode bogus` prints a multi-frame Rich traceback instead of a clean
# "Invalid value for '--skill-mode'" (compare: `--level bogus` does it right with RC=2).
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess
r = subprocess.run(['uv','run','bench','eval','create','--tasks-dir','/tmp/bf-stress/oracle_smoke_tasks',
                    '--agent','oracle','--sandbox','docker','--skill-mode','bogus'],
                   capture_output=True, text=True, timeout=60)
out = r.stdout + r.stderr
print("ACTUAL RC", r.returncode, "| traceback present:", "Traceback (most recent call last)" in out)
print(out[-500:])
print("EXPECTED: clean 'Invalid value for --skill-mode: ...' with valid choices, no traceback")
PY
