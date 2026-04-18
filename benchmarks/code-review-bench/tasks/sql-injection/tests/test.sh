#!/bin/bash
set -e

REWARD=0

if [ -f /app/review.md ]; then
    content=$(cat /app/review.md | tr '[:upper:]' '[:lower:]')

    # Check if the review mentions SQL injection
    if echo "$content" | grep -qi "sql.inject"; then
        REWARD=1
    elif echo "$content" | grep -qi "f-string.*query\|string.*interpolat\|format.*sql\|unsanitized.*input"; then
        REWARD=1
    fi
fi

echo "$REWARD" > /logs/verifier/reward.txt
