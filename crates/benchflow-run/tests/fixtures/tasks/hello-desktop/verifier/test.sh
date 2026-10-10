#!/bin/bash
# Passes when the agent's final answer holds the code the page shows for "BenchFlow".
expected="HELLO-3534"
answer=$(cat /logs/agent/final_answer.txt 2>/dev/null)
if printf '%s' "$answer" | grep -q "$expected"; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
echo "expected=$expected answer=$answer"
