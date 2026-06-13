#!/usr/bin/env bash
# Verifier entrypoint. Runs pytest against /tests/test_outputs.py and writes a
# reward to /logs/verifier/reward.txt: 1.0 if pytest exits 0, else 0.0.
#
# Always exits 0 so benchflow records the reward we wrote above rather than a
# verifier-crash state.
set -u

mkdir -p /logs/verifier

cd /tests

if python -m pytest test_outputs.py -q --tb=no; then
    echo "1.0" > /logs/verifier/reward.txt
else
    echo "0.0" > /logs/verifier/reward.txt
fi

exit 0
