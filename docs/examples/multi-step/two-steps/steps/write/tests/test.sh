#!/bin/bash
if [ "$(cat /app/a.txt 2>/dev/null)" = 1 ]; then echo "PASS: a.txt holds 1"; echo 1; else echo "FAIL: a.txt does not hold 1"; echo 0; fi | tail -1 > /logs/verifier/reward.txt
