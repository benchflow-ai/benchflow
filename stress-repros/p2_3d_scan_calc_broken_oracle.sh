#!/usr/bin/env bash
# P2 (example): docs/examples/task-md/real-skillsbench/3d-scan-calc oracle cannot self-pass in the
# default `--agent oracle` invocation. solve.sh does `from mesh_tool import MeshAnalyzer` (a module in
# environment/skills/mesh-analysis/scripts/) but the task does NOT declare environment.skills_dir, so
# under the default no-skill policy the skill is never injected -> ModuleNotFoundError -> no
# /root/mass_report.json -> verifier fails -> reward 0.0. Fails IDENTICALLY on docker and daytona
# (so it is NOT a sandbox-parity bug; the harness no-skill default is correct and test-guarded).
set -u
source /tmp/bf-stress-env.sh
cd "$(git rev-parse --show-toplevel)"
STAGE=$(mktemp -d); cp -R docs/examples/task-md/real-skillsbench/3d-scan-calc "$STAGE/"
JOBS=$(mktemp -d)
echo "Running oracle on docker (expect FAIL with ModuleNotFoundError)..."
uv run bench eval create --tasks-dir "$STAGE" --agent oracle --sandbox docker --concurrency 1 --jobs-dir "$JOBS" 2>&1 | tail -4
echo "--- oracle solve.sh stdout (look for ModuleNotFoundError: No module named 'mesh_tool') ---"
find "$JOBS" -name oracle.txt -exec tail -5 {} \;
echo "FIX: add 'environment: { skills_dir: environment/skills }' to its task.md, or make solve.sh self-contained."
rm -rf "$STAGE" "$JOBS"
