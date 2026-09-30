#!/bin/bash
# A continuous maximum-likelihood fit above an x_min chosen by the minimum Kolmogorov-Smirnov
# distance (Clauset, Shalizi and Newman 2009), and a bootstrap that re-selects x_min in every resample.
set -euo pipefail
mkdir -p /work
cat > /work/fit_powerlaw.py << 'EOF'
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_TAIL = 50  # smallest tail a candidate x_min may leave


def fit_above(xs, xmin):
    """Continuous MLE of alpha for the sorted sample xs at or above xmin, and the KS distance of the fit."""
    tail = xs[xs >= xmin]
    n = len(tail)
    alpha = 1.0 + n / np.sum(np.log(tail / xmin))
    empirical = np.arange(n) / n
    model = 1.0 - (tail / xmin) ** (1.0 - alpha)
    return alpha, n, float(np.max(np.abs(empirical - model)))


def select(xs, candidates):
    """The candidate x_min with the smallest KS distance: (D, xmin, alpha, n)."""
    best = None
    for xmin in candidates:
        if np.count_nonzero(xs >= xmin) < MIN_TAIL:
            break
        alpha, n, d = fit_above(xs, xmin)
        if best is None or d < best[0]:
            best = (d, float(xmin), float(alpha), int(n))
    return best


x = np.sort(pd.read_csv("/data/events.csv")["size"].to_numpy(dtype=float))
d, xmin, alpha, n_tail = select(x, np.unique(x))
print(f"KS scan over {len(np.unique(x))} candidates: xmin={xmin:.4f} alpha={alpha:.4f} n_tail={n_tail} D={d:.4f}")

rng = np.random.default_rng(12345)
grid = np.quantile(x, np.linspace(0.0, 0.98, 400))
boot = []
for _ in range(300):
    xb = np.sort(rng.choice(x, size=len(x), replace=True))
    boot.append(select(xb, grid)[2])
boot = np.array(boot)
alpha_err = float(np.std(boot, ddof=1))
asymptotic = (alpha - 1.0) / np.sqrt(n_tail)
print(f"bootstrap (300 resamples, x_min re-selected on a 400-point quantile grid): alpha_err={alpha_err:.4f}; asymptotic (alpha-1)/sqrt(n)={asymptotic:.4f}")

sensitivity = {f"{m:g}": round(fit_above(x, m)[0], 4) for m in (8.0, 10.0, 12.0, 15.0, 20.0)}
print("alpha at fixed x_min:", sensitivity)

with open("/work/fit.json", "w") as f:
    json.dump({"alpha": round(alpha, 4), "alpha_err": round(alpha_err, 4), "xmin": round(xmin, 4), "n_tail": n_tail}, f, indent=2)

ccdf = 1.0 - np.arange(len(x)) / len(x)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(x, ccdf, ".", ms=2, color="0.4", label="data (CCDF)")
xt = np.logspace(np.log10(xmin), np.log10(x.max()), 100)
ax.loglog(xt, (n_tail / len(x)) * (xt / xmin) ** (1.0 - alpha), "-", color="C3", lw=2,
          label=f"power law, α = {alpha:.2f} ± {alpha_err:.2f}, x ≥ {xmin:.1f}")
ax.axvline(xmin, color="C0", ls="--", lw=1, label=f"x_min = {xmin:.1f}")
ax.set_xlabel("event size x")
ax.set_ylabel("P(X ≥ x)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("/work/fit.png", dpi=120)

lines = [
    "# Power-law tail of the detector event sizes",
    "",
    "## Method",
    "",
    f"I fit a continuous power law p(x) ∝ x^(−α) to the events at or above a threshold x_min by maximum likelihood, α = 1 + n / Σ ln(x_i / x_min). To choose x_min I scanned every distinct event size that leaves at least {MIN_TAIL} events in the tail and kept the one whose fit has the smallest Kolmogorov-Smirnov distance between the empirical and fitted tail distributions (Clauset, Shalizi and Newman 2009).",
    "",
    "## Uncertainty",
    "",
    f"I resampled the 4,000 events with replacement 300 times, re-selected x_min in each resample (on a grid of 400 quantiles, to keep it fast), and refit α. The standard deviation of the bootstrap estimates is the uncertainty, so it includes the uncertainty in x_min. For comparison, the asymptotic standard error at fixed x_min is {asymptotic:.3f}.",
    "",
    "## Result",
    "",
    f"α = {alpha:.3f} ± {alpha_err:.3f}, with x_min = {xmin:.3f} and n_tail = {n_tail} of 4,000 events (KS distance {d:.4f}).",
    "",
    "fit.png shows the empirical complementary CDF on log-log axes, with the fitted power law over x ≥ x_min.",
    "",
    "## Caveats",
    "",
    "- α depends on x_min: at fixed thresholds of " + ", ".join(f"{k} (α = {v})" for k, v in sensitivity.items()) + ". The KS choice is noisy, which is why the bootstrap re-selects it.",
    "- I did not test the power law against other heavy-tailed distributions, such as a lognormal or a truncated power law, with likelihood-ratio tests, so the data is consistent with a power law above x_min but not shown to require one.",
    "- The largest events are few, so the far tail carries little weight in the fit.",
]
with open("/work/report.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print(open("/work/fit.json").read())
EOF
python3 /work/fit_powerlaw.py
