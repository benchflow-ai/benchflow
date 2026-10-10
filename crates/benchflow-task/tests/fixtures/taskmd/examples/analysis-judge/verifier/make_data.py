"""Writes sandbox/events.csv for analysis-judge: 4,000 event sizes whose tail is an exact power law.

1,500 sizes at or above x_min = 10 come from a continuous power law with alpha = 2.5 (inverse-transform sampling), and
2,500 below it from a lognormal body (median 3, sigma 0.6) cut at 10; the two are shuffled together. The seed is fixed,
so the file is reproducible. The script lives in verifier/ because it states the generating values, which
verifier/answers.json also holds: nothing that names them may reach the solver's sandbox.

Stdlib only.

Usage: python3 verifier/make_data.py <package-dir>
"""

import csv
import math
import random
import sys
from pathlib import Path

ALPHA, XMIN, N_TAIL, N_BODY, SEED = 2.5, 10.0, 1500, 2500, 20260928


def sizes() -> list[float]:
    rng = random.Random(SEED)
    tail = [XMIN * (1.0 - rng.random()) ** (-1.0 / (ALPHA - 1.0)) for _ in range(N_TAIL)]
    body: list[float] = []
    while len(body) < N_BODY:
        x = rng.lognormvariate(math.log(3.0), 0.6)
        if x < XMIN:
            body.append(x)
    out = tail + body
    rng.shuffle(out)
    return [round(x, 4) for x in out]


def main(pkg: str) -> None:
    path = Path(pkg) / "sandbox" / "events.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["event_id", "size"])
        for i, x in enumerate(sizes(), 1):
            w.writerow([i, f"{x:.4f}"])
    xs = [x for x in sizes() if x >= XMIN]
    mle = 1.0 + len(xs) / sum(math.log(x / XMIN) for x in xs)
    print(f"wrote {path}: n_tail at x_min={XMIN}: {len(xs)}, MLE alpha there: {mle:.4f} +/- {(mle - 1) / math.sqrt(len(xs)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
