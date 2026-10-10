#!/bin/bash
# Synced from eval/verifier/shared/test.sh by eval/scripts/sync_tests.py. Do not edit here.
set -euo pipefail
cd /app
mkdir -p /logs/verifier
python3 /tests/consult_lib.py prepare /tests
python3 /tests/consult_lib.py bundle > /logs/verifier/judge-bundle.md
rewardkit /tests --workspace /app --output /logs/verifier/reward.json
