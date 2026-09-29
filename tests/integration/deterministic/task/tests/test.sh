#!/bin/bash
set -e

REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
