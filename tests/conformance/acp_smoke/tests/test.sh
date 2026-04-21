#!/bin/bash
set -e

REWARD=0

if [ -f conformance.txt ]; then
    content=$(cat conformance.txt | tr -d '\n')
    if [ "$content" = "ok" ]; then
        REWARD=1
    fi
fi

echo "$REWARD" > /logs/verifier/reward.txt
