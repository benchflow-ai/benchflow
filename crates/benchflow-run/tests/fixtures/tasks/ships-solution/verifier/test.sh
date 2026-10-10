#!/bin/bash
if [ "$(cat /app/out.txt 2>/dev/null)" = "terces" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
