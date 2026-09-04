#!/bin/bash
# Oracle: install the reference IK controller as the agent's policy (reward 1.0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/policy.py" "${APP_DIR:-/app}/policy.py"
echo "installed oracle policy at ${APP_DIR:-/app}/policy.py"
