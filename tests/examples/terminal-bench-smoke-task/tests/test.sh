#!/bin/bash
set -euo pipefail

if python /tests/test_state.py; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
