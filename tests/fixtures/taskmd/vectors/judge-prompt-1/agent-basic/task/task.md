<!-- task.md canary 3c9e5b1a-7d2f-4e8a-9b6c-0f1e2d3c4b5a -->
Estimate the power-law exponent of the event sizes in `/data/events.csv`. Fit it by maximum likelihood, choose the lower cutoff `x_min` by the Kolmogorov-Smirnov criterion, and give a 95% bootstrap interval. Put your code in `src/`, and write `fit.json` with `alpha`, `x_min`, and `ci`, a figure `fit.png`, and a short `report.md` that states the method and its caveats. Do not delete or edit `/data/events.csv`.

```toml task
name = "vectors/agent-basic"
version = "1.0.0"

[sandbox]
image = "python:3.12-slim"
workdir = "/work"
outputs = ["/work/fit.json", "/work/report.md", "/work/fit.png", { path = "/work/src" }]

[verifier.judges.agent]
model = "claude-opus-5-5"
samples = 3
budget = { tokens = "2M", tool_calls = 200 }
```
