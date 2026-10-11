#!/bin/bash
# Graded in its own image, with no network.
python3 - <<'PY' > /logs/verifier/reward.txt
import json
try:
    ok = json.load(open("/app/answer.json")) == {"answer": 42}
except Exception:
    ok = False
print(1 if ok else 0)
PY
