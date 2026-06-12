#!/usr/bin/env bash
# P1-2: `--sandbox modal` (advertised in help) with the sandbox-modal extra absent dies with a raw
# ModuleNotFoundError at sandbox/setup.py:186 -> per-task [ERR], misleading "Job complete: 0/1".
# EXPECTED: actionable "install the 'sandbox-modal' extra" before any import.
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess
try:
    r = subprocess.run(
        ['uv','run','bench','eval','create','--tasks-dir','/tmp/bf-stress/oracle_smoke_tasks',
         '--agent','oracle','--sandbox','modal','--concurrency','1'],
        capture_output=True, text=True, timeout=40)
    out = r.stdout + r.stderr
    print("ACTUAL RC", r.returncode)
    print("  ModuleNotFoundError present:", "ModuleNotFoundError" in out)
    print("  actionable 'sandbox-modal' / 'extra' hint:", ("sandbox-modal" in out) or ("extra" in out.lower()))
    print(out[-400:])
except subprocess.TimeoutExpired:
    print("ACTUAL: TIMEOUT")
print("EXPECTED: clean error naming the missing 'sandbox-modal' extra, no raw traceback")
PY
