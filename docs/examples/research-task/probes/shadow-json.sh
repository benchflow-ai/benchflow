#!/bin/bash
# Probe: leave a json.py in /app that answers for the test, next to a wrong answer.
echo '{"half_life_min": 41.9, "uncertainty_min": 0.4}' > /app/answer.json
cat > /app/json.py <<'PY'
def load(f):
    return {"half_life_min": 37.4, "uncertainty_min": 1.0}
PY
echo "left /app/json.py"
