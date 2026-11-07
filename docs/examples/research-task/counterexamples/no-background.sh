#!/bin/bash
# Counterexample: fit a plain exponential and ignore the room's background.
cat > /app/fit.py <<'PY'
"""Weighted least squares of counts = A exp(-ln2 t / T), profiled over T. No background term."""
import json

import numpy as np

d = np.loadtxt("/app/data/counts.csv", delimiter=",", skiprows=1)
t, c = d[:, 0] + 0.5, d[:, 1]
w = 1 / np.maximum(c, 1)


def chi2(T):
    x = np.exp(-np.log(2) * t / T)
    a = np.sum(w * c * x) / np.sum(w * x * x)
    return float(np.sum(w * (c - a * x) ** 2))


grid = np.arange(20, 80, 0.005)
vals = np.array([chi2(T) for T in grid])
best = float(grid[vals.argmin()])
inside = grid[vals <= vals.min() + 1]
unc = float(inside.max() - inside.min()) / 2
json.dump({"half_life_min": round(best, 3), "uncertainty_min": round(unc, 3)}, open("/app/answer.json", "w"))
print(f"half-life {best:.3f} +- {unc:.3f} min; chi-square {vals.min():.1f} for {len(t) - 2} degrees of freedom")
PY
python3 /app/fit.py
cat > /app/report.md <<'MD'
# Half-life

- Model: counts in each minute = A exp(-ln2 t / T).
- Fit: weighted least squares with Poisson weights, T scanned from 20 to 80 minutes.
- Uncertainty: half the width of the range of T where chi-square is within 1 of its minimum.
- Reproduce: `python3 /app/fit.py`.
MD
