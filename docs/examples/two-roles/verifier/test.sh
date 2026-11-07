#!/bin/bash
# Pass when the writer left a script and the runner wrote its output: 120 data rows.
if [ -f /app/count.py ] && [ "$(tr -d '[:space:]' < /app/rows.txt 2>/dev/null)" = 120 ]; then
  echo "PASS: count.py exists and rows.txt holds 120"; echo 1 > /logs/verifier/reward.txt
else
  echo "FAIL: count.py $( [ -f /app/count.py ] && echo exists || echo missing ), rows.txt holds '$(cat /app/rows.txt 2>/dev/null)'"; echo 0 > /logs/verifier/reward.txt
fi
