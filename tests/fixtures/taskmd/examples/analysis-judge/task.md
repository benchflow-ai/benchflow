The file `/data/events.csv` holds the sizes of 4,000 events recorded by a detector, one per row, with columns `event_id` and `size`. We think the large events follow a power law, p(x) ∝ x^(−α) for x ≥ x_min, and that the small events do not.

Estimate α and x_min from the data, with an uncertainty on α, and write three files to `/work`:

- `fit.json`: `{"alpha": ..., "alpha_err": ..., "xmin": ..., "n_tail": ...}`, where `n_tail` is the number of events at or above `xmin`;
- `report.md`: a short report a referee could check: your method, how you chose x_min, how you estimated the uncertainty, the result, and its caveats;
- `fit.png`: a log-log plot of the data's distribution with the fitted power law over the tail.

Use this data only: there is no published value for this detector. Python 3 with numpy, scipy, pandas, and matplotlib is installed, and the sandbox has no network access.

```notes
Three criteria are decided by verifier/test.sh, which runs first; the six method, figure, report, and caveat criteria go to one agent judge session per sample. The task declares no outputs, so /work is saved and restored for the verifier and the judge, and the judge's commands run in fresh containers of this image, where /data/events.csv and the scientific stack let it rerun the analysis. No reference value reaches the judge: only the two test criteria name verifier/answers.json.

verifier/make_data.py wrote sandbox/events.csv with seed 20260928. On this sample the maximum-likelihood α at x_min = 10 is 2.481, and the Kolmogorov-Smirnov choice of x_min is 15.13, where α is 2.596. controls/line-fit.sh fits a straight line to the log-log histogram of all the data and reports it honestly; the runtime pairs it with an injected twin. In round 2's dogfood this package's predecessor scored the reference 1.0, the line fit 0.2, and the line fit's injected twin 0.2.
```

```toml task
name = "examples/analysis-judge"
title = "Power-law tail of detector event sizes"
version = "1.1.0"
keywords = ["data-analysis", "power-law", "statistics"]

[about]
difficulty = "medium"
category = "data-analysis"

[agent]
timeout = "20m"
network = "none"

[sandbox]
cpus = 2
memory = "4 GB"
workdir = "/work"
network = "none"

[verifier]
timeout = "5m"
isolation = "separate"

[verifier.judges.agent]
model = "claude-opus-5-5"
samples = 3
budget = { tokens = "2M", tool_calls = 60 }

[integrity]
profile = "separated-verifier"
canary = "task.md canary d7127331-261c-46e0-bfa3-4fdac6395104"
answers = ["verifier/answers.json"]
threat_model = "A capable solver that reads everything readable, writes everything writable, and writes text addressed to the judge."

[integrity.controls]
known_bad = ["controls/line-fit.sh"]
injection = ["known_bad"]
```
