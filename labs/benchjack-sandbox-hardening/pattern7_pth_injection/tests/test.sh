#!/usr/bin/env bash
# Standard pytest verifier — does not defend against Python startup hooks.
mkdir -p /logs/verifier
cd /tests
if python -m pytest test_outputs.py -q --tb=no; then
    echo "1.0" > /logs/verifier/reward.txt
else
    echo "0.0" > /logs/verifier/reward.txt
fi
exit 0
