#!/bin/bash
mkdir -p /logs/verifier
if [ "$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)" = "42" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
