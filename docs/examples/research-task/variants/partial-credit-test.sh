#!/bin/bash
# Partial credit: one third of the reward for each check the answer passes.
python3 -I - <<'PY'
import json

TRUE = 37.4
checks = {}
try:
    answer = json.load(open("/app/answer.json"))
    est, unc = float(answer["half_life_min"]), float(answer["uncertainty_min"])
    off = abs(est - TRUE)
    checks = {
        "close_enough": off <= 2.5,
        "plausible_uncertainty": 0.3 <= unc <= 3.0,
        "consistent": unc > 0 and off <= 3 * unc,
    }
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"no usable /app/answer.json ({exc})")
reward = round(sum(checks.values()) / 3, 4)
print(f"reward {reward}: " + ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in checks.items()))
json.dump({"reward": reward}, open("/logs/verifier/reward.json", "w"))
PY
