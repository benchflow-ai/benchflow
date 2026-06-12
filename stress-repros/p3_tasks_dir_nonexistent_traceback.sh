#!/usr/bin/env bash
# P3-1: `--tasks-dir /nonexistent/path` raises a raw FileNotFoundError traceback instead of a
# clean "tasks dir not found" error.
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess
r = subprocess.run(['uv','run','bench','eval','create','--tasks-dir','/nonexistent/path',
                    '--agent','oracle','--sandbox','docker','--concurrency','1'],
                   capture_output=True, text=True, timeout=60)
out = r.stdout + r.stderr
print("ACTUAL RC", r.returncode, "| FileNotFoundError traceback:", "FileNotFoundError" in out)
print(out[-500:])
print("EXPECTED: clean 'tasks dir /nonexistent/path does not exist'.")
PY
