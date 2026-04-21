#!/bin/bash
# Verify hello.txt exists and has the right content
set -e

if [ ! -f /app/hello.txt ]; then
    echo "FAIL: /app/hello.txt not found"
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

content=$(cat /app/hello.txt)
expected="Hello from benchflow!"

if [ "$content" = "$expected" ]; then
    echo "PASS: hello.txt matches"
    echo "1" > /logs/verifier/reward.txt
else
    echo "FAIL: expected '$expected', got '$content'"
    echo "0" > /logs/verifier/reward.txt
fi
