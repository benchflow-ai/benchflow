#!/bin/bash
# Fits a straight line to the log-log histogram of all the event sizes and writes fit.json, report.md, and fit.png to /work.
set -euo pipefail
mkdir -p /work
cat > /work/fit_line.py << 'EOF'
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = pd.read_csv("/data/events.csv")["size"].to_numpy(dtype=float)
edges = np.logspace(np.log10(x.min()), np.log10(x.max()), 30)
density, _ = np.histogram(x, bins=edges, density=True)
centers = np.sqrt(edges[1:] * edges[:-1])
keep = density > 0
(slope, intercept), cov = np.polyfit(np.log10(centers[keep]), np.log10(density[keep]), 1, cov=True)
alpha, alpha_err = -slope, float(np.sqrt(cov[0, 0]))
print(f"slope={slope:.4f} alpha={alpha:.4f} err={alpha_err:.4f}")
with open("/work/fit.json", "w") as f:
    json.dump({"alpha": round(alpha, 4), "alpha_err": round(alpha_err, 4), "xmin": round(float(x.min()), 4), "n_tail": int(len(x))}, f, indent=2)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(centers[keep], density[keep], "o", label="histogram")
ax.loglog(centers[keep], 10 ** (intercept + slope * np.log10(centers[keep])), "-", label=f"fit, alpha = {alpha:.2f}")
ax.set_xlabel("size")
ax.set_ylabel("density")
ax.legend()
fig.savefig("/work/fit.png", dpi=100)
with open("/work/report.md", "w") as f:
    f.write(f"""# Power-law fit

## Method

I binned all 4,000 event sizes into 29 logarithmic bins, computed the density in each bin, and fit a straight line to log10(density) against log10(size) by least squares. The exponent is minus the slope. x_min is the smallest event, so the fit uses every event.

## Uncertainty

The uncertainty is the standard error of the fitted slope.

## Result

alpha = {alpha:.3f} ± {alpha_err:.3f}, with x_min = {x.min():.3f} and n_tail = {len(x)}.

## Caveats

The fit uses all the data.
""")
EOF
python3 /work/fit_line.py
