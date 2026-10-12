#!/bin/bash
REWARD=0
if [ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "Hello, world!" ]; then REWARD=1; fi
echo "$REWARD" > /logs/verifier/reward.txt
