# pass@k, pass^k and solve rates

BenchFlow reports three repeated-trial numbers next to the pass counts: pass@k (the chance that at least one of k trials of a task succeeds), pass^k (the chance that all k succeed) and the solve rate (the fraction of scored trials that succeed). They are in `summary.json` (`solve_rates`), `matrix-summary.json` (`solve_rates` per model entry), `bench eval metrics`, `bench eval compare`, `bf.load_job(...).solve_rates()` and `bf.compare(...)` (`solve_rates_a` / `solve_rates_b`), and in the `benchflow.job` and `benchflow.comparison` JSON documents. The code is `benchflow.pass_at_k`.

## Estimators

For a task with n scored trials of which c succeed:

- pass@k = 1 − C(n − c, k) / C(n, k), the unbiased estimator of Chen et al. (2021).
- pass^k = C(c, k) / C(n, k), the unbiased estimator used by tau-bench.

The job value is the mean over tasks. Both need n ≥ k for a task. A task with fewer scored trials is left out of that k (never extrapolated), the result counts it in `tasks_short_at_k`, and a caveat says `pass@k/pass^k cover X of Y tasks: … (n < k)`. When no task has k scored trials the value is `null`, not 0. By default k runs over 1, the powers of two and the multiples of five up to the smallest per-task n; `--k` (repeatable) or `ks=[...]` asks for specific values.

## What counts

- Samples are scored trials. An unscored trial (agent error, verifier error, no reward) is left out of n and counted in `unscored`; it is not a failure. Harbor counts a missing reward as 0; BenchFlow does not.
- Control runs (oracle, empty/nop, task copies suffixed `__o` (oracle) or `__e` (empty solution)) are left out (`controls_excluded`) unless `include_controls=True`.
- A task's trials in different job folders are separate samples: `--matrix --trials N` writes `<alias>/trial-NN/<job>/`, so `bench eval metrics <jobs-dir>/<alias>` pools the N trials. Retries of one task inside one Evaluation job are one sample (the best attempt, as `bf.load_job` keeps); repeated rollouts of a task in one `bf.run_batch` folder are separate samples.
- One `Evaluation` holds one result per task, so its `summary.json` has n = 1 and only pass@1; pool trial folders for k > 1.

## Success rule and partial credit

By default a trial succeeds when it passed: reward = 1, or the integrated review gate's verdict when the trial has one (`success_rule: "passed (reward = 1)"`). This is the rule behind every other pass count. For non-binary rewards pass a threshold: with `--solve-threshold 0.5` (`solve_threshold=0.5`) a scored trial succeeds when its reward is at least 0.5, and pass@k, pass^k and the solve rate all use that rule (`success_rule: "reward >= 0.5"`). `nonbinary_rewards` counts scored trials whose reward is neither 0 nor 1; when there are some and no threshold is set, a caveat says they count as not passed.

## Examples

```bash
bench eval run --tasks-dir tasks --matrix models.yaml --trials 5 --jobs-dir jobs/m
bench eval metrics jobs/m/haiku                     # pass@1, @2, @4, @5 over 5 trials
bench eval metrics jobs/m/haiku --k 1 --k 8 --json  # pass@8 is null: n = 5 < 8
bench eval metrics jobs/m/haiku --solve-threshold 0.5
bench eval compare jobs/m/haiku jobs/m/sonnet --vary model --k 1 --k 5
```

```python
import benchflow as bf

rates = bf.load_job("jobs/m/haiku").solve_rates(ks=[1, 5], solve_threshold=None)
print(rates.get(5).pass_at_k, rates.get(5).pass_hat_k, rates.caveats)
report = bf.compare("jobs/m/haiku", "jobs/m/sonnet", vary=("model",), ks=[1, 5])
print(report.solve_rates_a.to_dict(), report.solve_rates_b.to_dict())
```
