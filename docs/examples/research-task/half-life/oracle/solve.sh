#!/bin/bash
set -e
cat > /app/fit.py <<'PY'
"""Weighted least squares of counts = A exp(-ln2 t / T) + B, profiled over T."""
import json

import numpy as np

d = np.loadtxt("/app/data/counts.csv", delimiter=",", skiprows=1)
t, c = d[:, 0] + 0.5, d[:, 1]  # each minute's centre
w = 1 / np.maximum(c, 1)  # Poisson: the variance of a count is the count


def chi2(T):
    X = np.column_stack([np.exp(-np.log(2) * t / T), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], c * np.sqrt(w), rcond=None)
    return float(np.sum(w * (c - X @ coef) ** 2))


grid = np.arange(20, 80, 0.005)
vals = np.array([chi2(T) for T in grid])
best = float(grid[vals.argmin()])
inside = grid[vals <= vals.min() + 1]  # 1 sigma: chi-square within 1 of its minimum
unc = float(inside.max() - inside.min()) / 2
json.dump({"half_life_min": round(best, 3), "uncertainty_min": round(unc, 3)}, open("/app/answer.json", "w"))
print(f"half-life {best:.3f} +- {unc:.3f} min")
PY
python3 /app/fit.py
cat > /app/report.md <<'MD'
# Half-life

- Model: counts in each minute = A exp(-ln2 t / T) + B, with B the constant background, at each minute's centre.
- Fit: weighted least squares with Poisson weights (variance = counts); A and B are solved for each T, and T is scanned from 20 to 80 minutes.
- Uncertainty: half the width of the range of T where chi-square is within 1 of its minimum.
- Reproduce: `python3 /app/fit.py`.
MD
