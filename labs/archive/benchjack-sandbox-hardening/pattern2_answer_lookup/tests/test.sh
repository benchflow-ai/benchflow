#!/usr/bin/env bash
mkdir -p /logs/verifier
AGENT=$(cat /tmp/answer.txt 2>/dev/null | tr -d '[:space:]')
TRUTH=$(cat /tests/ground_truth.txt | tr -d '[:space:]')
if [ "$AGENT" = "$TRUTH" ]; then
    echo "1.0" > /logs/verifier/reward.txt
else
    echo "0.0" > /logs/verifier/reward.txt
fi
exit 0
