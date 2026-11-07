#!/bin/bash
# -I: import nothing from the working directory, which is the agent's workspace (/app).
python3 -I - <<'PY'
import json

TRUE = 37.4  # minutes: the half-life the data were generated with
reward = 0
try:
    answer = json.load(open("/app/answer.json"))
    est, unc = float(answer["half_life_min"]), float(answer["uncertainty_min"])
    off = abs(est - TRUE)
    # A fit that models the background gives 37.1 +- 1.1; one that ignores it gives 41.9 +- 0.4.
    checks = [
        (off <= 2.5, f"{off:.2f} min from the true value, more than 2.5"),
        (0.3 <= unc <= 3.0, f"uncertainty {unc:.2f} min, outside 0.3 to 3"),
        (unc > 0 and off <= 3 * unc,
         f"the true value is {off / unc:.1f} standard deviations away, more than 3" if unc > 0
         else "the uncertainty is not positive"),
    ]
    failed = [why for ok, why in checks if not ok]
    reward = 0 if failed else 1
    verdict = "FAIL: " + "; ".join(failed) if failed else "PASS"
    print(f"{verdict} (answer {est:.2f} +- {unc:.2f} min, true value {TRUE})")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"FAIL: no usable /app/answer.json ({exc})")
open("/logs/verifier/reward.txt", "w").write(f"{reward}\n")
PY
