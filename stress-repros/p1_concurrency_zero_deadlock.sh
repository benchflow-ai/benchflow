#!/usr/bin/env bash
# P1-1: `--concurrency 0` builds asyncio.Semaphore(0) (evaluation.py:1007) and deadlocks forever.
# EXPECTED: usage error rejecting concurrency<1.  ACTUAL: hangs (bounded to 20s here -> TIMEOUT).
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess
try:
    r = subprocess.run(
        ['uv','run','bench','eval','create','--tasks-dir','/tmp/bf-stress/oracle_smoke_tasks',
         '--agent','oracle','--sandbox','docker','--concurrency','0'],
        capture_output=True, text=True, timeout=20)
    print("ACTUAL RC", r.returncode); print((r.stdout + r.stderr)[-400:])
except subprocess.TimeoutExpired:
    print("ACTUAL: TIMEOUT after 20s -> DEADLOCK CONFIRMED (Semaphore(0) never acquires)")
print("EXPECTED: immediate non-zero exit, e.g. 'Invalid value for --concurrency: must be >= 1'")
PY
