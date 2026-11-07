#!/bin/bash
# Reward 1 when the service the agent left running still answers.
out=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5).read().decode().strip())" 2>/dev/null)
if [ "$out" = kept ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
