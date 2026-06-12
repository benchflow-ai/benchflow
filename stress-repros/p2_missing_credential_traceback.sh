#!/usr/bin/env bash
# P2-5: an agent whose required credential is unset fast-fails (~1s, no build started -- good) but
# the failure surfaces as a raw in-rollout Python traceback + per-task [ERR] + a misleading
# "Score: 0/1" rather than a clean preflight "set ANTHROPIC_API_KEY / log in".
set -u
source /tmp/bf-stress-env.sh
python3 - <<'PY'
import subprocess, os
env = {k:v for k,v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
r = subprocess.run(['uv','run','bench','eval','create','--tasks-dir','/tmp/bf-stress/oracle_smoke_tasks',
                    '--agent','claude','--sandbox','docker','--concurrency','1'],
                   capture_output=True, text=True, timeout=60, env=env)
out = r.stdout + r.stderr
print("ACTUAL RC", r.returncode)
print("  raw traceback present:", "Traceback (most recent call last)" in out)
print("  started a docker build:", any(s in out for s in ("Starting environment","Installing","Building")))
print("  misleading 'Score:' line:", "Score:" in out)
print(out[-500:])
print("EXPECTED: clean preflight: 'agent claude needs ANTHROPIC_API_KEY (or login); none found'.")
PY
