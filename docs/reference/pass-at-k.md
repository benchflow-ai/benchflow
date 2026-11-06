# pass@k, pass^k and solve rates

```bash
bench eval metrics jobs/m/haiku
```

When a task runs several times, three numbers say how reliably an agent solves it:

- **pass@k**: the chance that at least one of k trials succeeds.
- **pass^k**: the chance that all k trials succeed.
- **solve rate**: the fraction of trials that succeed.

`bench eval metrics <job>` prints them. `bf.load_job(path).solve_rates()` returns them. They are also in `summary.json`, `matrix-summary.json` (per model), `bf.compare(...)` (`solve_rates_a`, `solve_rates_b`) and the [JSON documents](./json-export.md).

## Estimators

For a task with n trials of which c succeed:

- pass@k = 1 − C(n − c, k) / C(n, k), the unbiased estimator of Chen et al. (2021).
- pass^k = C(c, k) / C(n, k), the unbiased estimator used by tau-bench.

Example: 5 trials, 2 successes. pass@2 = 1 − C(3,2)/C(5,2) = 1 − 3/10 = 0.7. With 3 successes, pass^2 = C(3,2)/C(5,2) = 0.3.

The job's value is the mean over tasks. A task needs at least k trials. A task with fewer is left out of that k and counted in `tasks_short_at_k`. A caveat names it: `pass@k/pass^k cover X of Y tasks`.

If no task has k trials, the value is `null`, not 0.

By default k is 1, the powers of two, and the multiples of five, up to the smallest trial count per task. Pass `--k 1 --k 8` (`ks=[1, 8]`) to choose.

### A worked job

Three tasks, five trials each (P passed, F failed, U unscored):

| Task | Trials | n | c | pass@1 | pass@2 | pass^2 |
|---|---|---|---|---|---|---|
| A | P P F F F | 5 | 2 | 0.4 | 0.7 | 0.1 |
| B | P P P P P | 5 | 5 | 1.0 | 1.0 | 1.0 |
| C | F F U U F | 5 | 0 | 0.0 | 0.0 | 0.0 |
| **Job** | | 15 | 7 | 0.47 | 0.57 | 0.37 |

The solve rate is 7/15 = 0.47. The scored-only solve rate leaves task C's two unscored trials out: 7/13 = 0.54.

## What counts

Since 0.8.0 every trial counts. The [JSON export changelog](./json-export.md#changelog) says how to tell new files from older ones.

- **Every trial is a sample.** n is `trials`. An unscored trial (agent error, verifier error, no reward) is a fail in the solve rate, pass@k and pass^k. It counts 0 in the mean reward. Otherwise an agent that errors could score above one that fails cleanly. (Rust `bench` 0.8.2 counts the same way, except that its `mean_reward` is over scored trials only.)
- **Negative rewards.** A task can allow rewards below 0 (`reward_range = [-1.0, 1.0]`). An unscored trial still counts 0, which beats a negative reward. When that happens, BenchFlow adds `mean_reward_caveat` to `summary.json`, `bench eval metrics`, the end-of-run line and `Job.denominators()`. Read `mean_reward_scored_only` beside it. The solve rate, pass@k and pass^k are unaffected.
- **Scored-only figures.** `solve_rate_scored_only`, `pass_at_k_scored_only`, `pass_hat_k_scored_only` and `mean_reward_scored_only` leave unscored trials out. Their n is `scored`, over `tasks_scored_only` tasks. Use them to compare with numbers published from older releases. `bench eval metrics` and `print(job)` show them only when some trial is unscored.
- **Interval.** The solve rate has a 95% interval over its own n (`solve_rate_interval` in JSON). With at most one trial per task, or with a single task, it is the Wilson score interval (`wilson`). With repeated trials over several tasks, a task's trials are correlated. BenchFlow then divides n by the design effect before applying Wilson (`wilson-clustered`). The interval is deterministic.
- **Controls.** Oracle and empty (nop) runs are left out and counted in `controls_excluded`, and a caveat says how many. So are task copies named `…__o` (oracle) or `…__e` (empty solution). Pass `include_controls=True` (`--include-controls` on `bench eval metrics` and `bench eval compare`) to count them. A folder or job of only control runs is counted: `bench eval metrics` on a folder of oracle trials gives the oracle's pass@k, and `bench eval compare` of two control jobs pairs their tasks. With every trial left out, each figure says `n/a: the N trial(s) are control runs`.
- **Pooling.** Trials of a task in different job folders are separate samples. `bench run --tasks-dir tasks/ --matrix m.yaml --trials 5` writes `<alias>/trial-NN/<job>/`, and `bench eval metrics <jobs-dir>/<alias>` pools them. Retries inside one job count once (the best attempt). Repeated rollouts in one `bf.run_batch` folder count separately.
- **One result per task.** One `Evaluation` holds one result per task, so its `summary.json` has n = 1 and only pass@1. Pool trial folders for k > 1.

## Repeated trials

pass@k needs k trials of a task. `--trials N` runs every task N times, each trial its own job, in `<jobs dir>/<name>/trial-NN/`. Without `--matrix`, the name is the agent's, and the run's own agent and model repeat:

```bash
bench run --tasks-dir tasks --agent oracle --trials 3 --jobs-dir jobs/o
bench run -d terminal-bench@2.0 --include fix-git --include regex-log --agent oracle --trials 3 --jobs-dir jobs/o
bench eval metrics jobs/o/oracle --k 1 --k 3          # pass@1 and pass@3 over the 3 trials
```

`--matrix models.yaml` repeats several models (or agents) in one command, one folder per alias. Its `models:` maps each alias to a model name, or to a mapping with `model` and, optionally, `agent` and `agent_env`. An agent that takes no model (`oracle`, `nop`) needs only `agent`. A model written as `none` or `null` is refused.

```yaml
models:
  haiku: claude-haiku-4-5
  sonnet:
    model: claude-sonnet-4-5
    agent: claude-agent-acp
    agent_env:
      MAX_THINKING_TOKENS: "8000"
  oracle:
    agent: oracle
```

`--matrix` and `--trials` take a task folder (`--tasks-dir`) or a dataset (`-d name@version`). A `--config` file or a `--source-env` run is refused rather than run once. The trials run one after another, each at `--concurrency`.

Every trial runs, even when one errors or scores low: a trial job applies no gate and never stops the matrix. `--fail-under`, `--fail-on` and `--summary-out` apply once, after the last trial. `--fail-under` follows the controls rule above: the trials of control entries (`{agent: oracle}`, `{agent: nop}`) are left out of its numerator and denominator, unless every entry is a control. A `Gate:` line says what it counted (`Gate: 0.75 over 8 trials of claude-code; 4 control trials (nop) left out`). Over the trials it counts, an errored or unscored trial is not passed. A trial job that fails to start counts as an error. The command exits 1 when any trial of any entry errored (controls included) or the gate failed.

`matrix-summary.json` in the jobs directory is always written, also after an interrupt (`complete: false`). It holds each alias's runs (`job_failed: true` for a trial job that did not run), what the gate counted (`gate`: `trials`, `agents`, `controls_left_out`, `control_kinds`, `controls_counted`) and its pooled solve rates.

## Success rule and partial credit

By default a trial succeeds when it passed: reward 1, or the rubric reviewer's pass verdict when there is one (`success_rule: "passed (reward = 1)"`).

For partial credit, pass `--solve-threshold 0.5` (`solve_threshold=0.5`). A scored trial then succeeds when its reward is at least 0.5 (`success_rule: "reward >= 0.5"`). An unscored trial never succeeds.

`nonbinary_rewards` counts scored rewards other than 0 and 1. Without a threshold they count as fails, and a caveat says so.

Only `reward` is counted here. The other keys of the `rewards` dict are not aggregated.

## Examples

```bash
# 5 trials per task per model
bench run --tasks-dir tasks/ --matrix models.yaml --trials 5 --jobs-dir jobs/m
bench eval metrics jobs/m/haiku                     # pass@1, @2, @4, @5
bench eval metrics jobs/m/haiku --k 1 --k 8 --json  # pass@8 is null: 5 < 8
bench eval metrics jobs/m/haiku --solve-threshold 0.5
# bench results has no --k or --vary
bench eval compare jobs/m/haiku jobs/m/sonnet --vary model --k 1 --k 5
```

```python
import benchflow as bf

rates = bf.load_job("jobs/m/haiku").solve_rates(ks=[1, 5], solve_threshold=None)
print(rates.get(5).pass_at_k, rates.get_scored_only(5).pass_at_k, rates.caveats)
report = bf.compare("jobs/m/haiku", "jobs/m/sonnet", vary=("model",), ks=[1, 5])
print(report.solve_rates_a.to_dict(), report.solve_rates_b.to_dict())
```

## Next steps

- [JSON export](./json-export.md)
- [Agent integration failures](./integration-failures.md): why a trial is unscored
- [Analysing runs](../results/analysing-runs.md)
- [Python API](./python-api.md#reading-finished-jobs)
