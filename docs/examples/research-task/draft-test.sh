#!/bin/bash
# A first draft of half-life/verifier/test.sh: it checks only that the estimate is within 15%.
python3 - <<'PY'
import json

TRUE = 37.4  # minutes: the half-life the data were generated with
reward = 0
try:
    answer = json.load(open("/app/answer.json"))
    est, unc = float(answer["half_life_min"]), float(answer["uncertainty_min"])
    err = abs(est - TRUE) / TRUE
    reward = int(err <= 0.15)
    print(("PASS" if reward else "FAIL") + f": {est:.2f} +- {unc:.2f} min is {100 * err:.1f}% from {TRUE}; allowed 15%")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"FAIL: no usable /app/answer.json ({exc})")
open("/logs/verifier/reward.txt", "w").write(f"{reward}\n")
PY
