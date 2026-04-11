#!/bin/bash
set -e

REWARD=0

if [ -f hello.txt ]; then
    content=$(cat hello.txt | tr -d '\n')
    if [ "$content" = "Hello, world!" ]; then
        REWARD=1
    fi
fi

echo "$REWARD" > /logs/verifier/reward.txt
