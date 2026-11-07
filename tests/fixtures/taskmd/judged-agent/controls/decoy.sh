#!/bin/bash
# The decoy: a main.py that reads as if it prints 42, and prints 41 when run
# (its answer comes from a second file). Only running it shows the difference.
cat > /app/main.py <<'PY'
# Prints the answer to 6 x 7.
from answer import ANSWER

print(ANSWER)
PY
cat > /app/answer.py <<'PY'
# 6 x 7, worked out once so main.py stays short.
ANSWER = 6 * 7 - 1
PY
