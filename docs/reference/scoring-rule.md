# The scoring rule

BenchFlow counts every headline score by one rule, `benchflow.score.v1`. `summary.json`, the end-of-run line, `report.md`, `bench eval metrics`, `bench results` and the run card's after-run lines all give the same figures. Any other reader, BenchFlow Hub included, can apply the same rule and check its arithmetic against the conformance fixture below.

`summary.json` names the rule in one sentence:

```json
"scoring_rule": {
  "id": "benchflow.score.v1",
  "text": "Every finished trial is in n and counts 1 only when it was scored and passed, so a wrong answer, an error, a timeout left with no score and a trial with no score at all each count 0.",
  "interval": "95% interval over tasks: ..."
}
```

## What counts

- **Every finished trial is in the denominator.** n is the number of finished trials. A task an Evaluation retried counts once, as its best attempt. A planned trial that never finished (the job stopped first) is not in n. `summary.json`'s `score_of_plan` counts it, and every reader says how many there are.
- **A trial counts 1 when it was scored and passed**: reward 1, or the rubric review's pass verdict. Every other trial counts 0: a wrong answer, a refusal, an agent or sandbox error, a model provider error, a verifier error, and a trial left with no score.
- **score = passed / n.**
- **The error-excluding figure is secondary**, and is always named as such: `score_excl_errors`, "scored only" on the end-of-run line and in `bench eval metrics`, "leaving out the trials with an error or no score" in `report.md`. It leaves out the trials with an error or no score. A refusal stays in it as a failure, because the agent can cause one: the plugin guard refused a pytest plugin the agent left, or the final verify found no usable reward file.

A reward decides when there is one: a trial that ran out of time and was still graded counts by its grade. A trial whose assessment is still pending has no score, whatever reward its folder holds.

Each trial gets one of six outcomes:

| Outcome | Meaning | In the score | In the error-excluding figure |
|---|---|---|---|
| `passed` | scored and passed | 1 | 1 |
| `failed` | scored below a pass | 0 | 0 |
| `refused` | no score, for a reason the agent can cause | 0 | 0 |
| `errored` | no score: the agent, its sandbox or its model provider errored, or BenchFlow's time limits stopped it | 0 | left out |
| `verifier_errored` | no score: the verifier failed | 0 | left out |
| `no_verdict` | no score and no recorded error, such as an assessment still pending | 0 | left out |

Every reader names the trials with no score in the same words, built from these outcomes: `U unscored (V verifier errors, R refused), E errored (N with no verdict)`. Unscored trials finished and the verifier gave them no score (a verifier error or a refusal); errored ones stopped before a verdict, a trial with no verdict among them, as the end-of-run line's `errors=` counts them. The run's `Outcomes:` line, `bench results` and `bench eval compare` all print it, for example `1 scored (1 passed, 0 failed), 1 unscored (1 verifier error), 2 errored`.

## Provider and harness endings

A trial the model provider ended (its API failed, it rate-limited the run, it refused the credential or the request, or the login's usage limit ran out), or that BenchFlow's own limits ended (the time limit, the idle watchdog), is retried under the run's retry policy. One that is still without a score after the retries counts 0, like any other error. It is also counted by who ended it, its `ending`:

| Ending | Error classes |
|---|---|
| `provider` | `api_error`, `invalid_provider_response`, `provider_auth`, `provider_rate_limit`, `provider_rejected`, `provider_unavailable`, `usage_limit` |
| `harness` | `timeout`, `idle_timeout` |

The error classes are the causes `bench explain` explains. The end-of-run line says, for example, `3 with no score, each counted 0: 2 ended by the model provider (api_error 1, provider_rate_limit 1); 1 ended by BenchFlow's time limits (idle_timeout 1)`.

## The interval

The 95% interval is over tasks, not trials, because a task's trials are not independent: an easy task passes on every try.

- `se` is the cluster-robust standard error of the score, with each task's trials one cluster and the small-sample factor T/(T−1) for T tasks: SE² = T/(T−1) × Σ over tasks of (passes − score × trials)² / n². This is the method of Miller, "Adding Error Bars to Evals" (2024). With one task there is no `se`.
- The interval is the one the solve rates have always used ([Choose the metric](../audit/compare-runs.md#choose-the-metric) covers pass@k and the solve rates). With one trial per task, or one task, it is Wilson's score interval over the trials (`ci_method: "wilson"`). Otherwise it is Wilson's interval on the effective sample size n / deff, where the design effect deff = max(1, SE² / (p(1−p)/n)) (`"wilson-clustered"`). When every trial passed, or none did, the effective sample size is the number of tasks.
- The interval is computed, not resampled, so it has no seed: the same trials always give the same interval.

The design effect is never below 1, so the interval is never narrower than one that treats every trial as independent.

## Warnings

The rule's figures carry warnings, written to `summary.json` (`scoring.warnings`) and printed by every reader:

- **k < 3**: some task has fewer than 3 trials. One or two tries per task say little about a task's pass rate, and pass@k needs k tries. Run `--trials 3` or more.
- **Error rate above 5%**: more than 5% of the trials ended with an error or no score.
- **Timeout rate above 5%**: more than 5% of the trials ran out of time, whether or not they were then graded.
- **Rate-limit rate above 5%**: more than 5% of the trials met a rate limit or a usage limit: their trial ended on one, or one of their calls got HTTP 429.

## One row per setting

A score means something for one setting only. Every reader splits the trials by what their records say ran, never by their labels:

| Key | Read from |
|---|---|
| agent and harness mode | `result.json`'s `agent`, `config.json`'s `harness_mode` |
| harness version | the installed adapter, CLI and engine packages in `harness_versions` |
| served model | the model that answered the agent's main calls (`settings_sent.models_served`); the run's model, marked `(requested)`, when no call recorded one |
| route | the route those calls took through BenchFlow's model proxy (`settings_sent.routes`) |
| effort | the effort those calls carried (`settings_sent.effort`); the effort the agent applied (`effort.applied`) when no call recorded one; else the effort asked for, marked as only that |
| task set | the dataset and its version, and which version of each task ran when one task ran at two versions (`task_digest`) |
| BenchFlow version | `config.json`'s `benchflow_version` |
| control run | oracle or empty |

A value carried by 1% of the calls or fewer does not count. A trial that made no model call of its own (it errored first) takes the one served model, route, effort and harness version the other trials of its launch show, so it stays in its launch's row and counts 0 there. A trial whose calls carried another effort than the one asked for is keyed by both, and its row says `effort high (requested max)`.

A job whose trials differ in any key reads as **mixed settings**: each setting is its own row with its own score, interval and warnings, no figure pools them, and a warning names what differs, for example `mixed settings: these trials ran 2 settings, which differ in route (anthropic/claude-opus-5, bedrock/us.anthropic.claude-opus-5-v1:0); ...`. A job restarted after a change (`restarted_after_change` in `summary.json` or `run-card.json`, written by `--restart-changed`) also reads as mixed settings: the warning says how many earlier trials ran with the old settings and are set aside, not counted.

## Comparing two runs

`bench results A B` and `bench eval compare` give each side's score by the rule on the tasks both sides ran, and the gap B − A:

- **The gap's interval is paired by task.** Its standard error is cluster-robust with each task one cluster: a task's residuals on both sides together, with the factor T/(T−1) for T paired tasks, so a task both sides find easy or hard cancels out. The 95% interval is the gap ± t × SE, with Student t's 97.5% quantile for T−1 degrees of freedom. With the same number of trials per task this is the t interval over tasks of the per-task differences, the one [`paired.py`](../examples/compare/paired.py) gives.
- **It warns** when that interval holds 0, when the gap is under 3 points (a published study of infrastructure noise found smaller gaps deserve skepticism), when a side has fewer than 3 trials per task, and when a side ran mixed settings. Under any of them, the comparison does not say which side is better.
- **It compares what ran.** Besides the recorded settings, each task's trials are checked for the effort their calls carried (`effort_sent`), the model that served them (`model_served`), their `route`, the `harness_version` and the `benchflow_version`. `--vary reasoning_effort` declares the effort sent too, `--vary model` the served model and route, `--vary harness` its version.

The `benchflow.comparison` document carries all of it under `scoring` (`gap`, `gap_se`, `gap_ci95`, `gap_inside_ci`, `gap_under_3_points`, `warnings`, each side's figures and `mixed`).

## Seeds

`summary.json`'s `scoring.seeding`, the `Seeds:` line of `report.md` and the run card's after-run lines say which seeds the numbers depend on:

| Key | Value |
|---|---|
| `task_order`, `task_order_seed` | `name` and `null`: a run starts its tasks in name order and nothing shuffles them, so there is no task-order seed. With concurrency they finish in any order |
| `task_sample` | `{"size", "seed"}` for `--n-tasks` with `--sample-seed` (a seeded random sample, kept in name order), the seed `null` for the first N in name order; `null` when every selected task ran |
| `rollout_seeds` | `--seeds`, for seeded rollouts; `null` without |
| `sampling_seed` | `requested`: the seed the run asked the model for (`BENCHFLOW_MODEL_SEED`, which the deepagents agent sends); `carried`: the main calls per seed value as they went upstream; `calls` and `without`; `status`: `carried`, `none` (none asked for or sent), `dropped` (asked for, carried by no call), `mismatch` or `not recorded` (no call went through BenchFlow's model proxy, which alone sees each request whole) |
| `warnings`, `text` | a warning for a dropped or mismatched seed, and the record in words |

A requested seed that no main call carried means the route dropped it: the provider sampled freely, and the warning says so. The run's checks list it too (`seed: requested 42; no main call carried it, so the provider's default applied`). It warns; it does not fail the run.

## Where it is written

`summary.json`:

| Key | Value |
|---|---|
| `scoring_rule` | `{"id", "text", "interval"}`: the rule this file was counted by |
| `score`, `score_ratio` | the score |
| `score_excl_errors`, `score_excl_errors_ratio` | the error-excluding figure |
| `scoring` | the rule's whole account, below |

`scoring`:

| Key | Value |
|---|---|
| `n`, `n_tasks`, `passed` | trials, tasks (the interval's clusters) and passes |
| `score`, `n_excl_errors`, `score_excl_errors` | the score, and the error-excluding figure with its n |
| `se`, `ci95`, `ci_method` | the standard error, the interval `[low, high]` and its method |
| `outcomes` | trials per outcome |
| `endings` | trials with no score per ending (`provider`, `harness`) |
| `error_classes` | trials with no score per error class |
| `timeouts`, `rate_limited`, `rates` | the trials that ran out of time and that met a rate limit, and the error, timeout and rate-limit rates |
| `tries` | the fewest and most trials of a task |
| `warnings` | the warnings above |
| `seeding` | the seeds, above |
| `settings` | `mixed`, `split`, `differ` (each key that differs, with its values), `warning`, `effort_warnings`, `restarted_after_change`, and `rows`: one per setting, with its keys, `label`, `trials` and its own `scoring` |

`passed`, `failed`, `errored`, `verifier_errored` and `unscored` at the top of `summary.json` keep their earlier meaning: an audit count that names a verifier error first, even when the trial was scored. `mean_reward` and `mean_reward_rule` are unchanged. With no finished trial, or none left once errors are left out, a ratio is 0.0 as before.

## What BenchFlow Hub must do

`bench hub push` uploads the job folder with one more file, `scoring.json`, made by the code `bench eval metrics` and `bench results` use:

| Key | Value |
|---|---|
| `schema` | `benchflow.scoring.v1` |
| `rule` | `{"id", "text", "interval"}`, as in `summary.json` |
| `trials` | one record per trial: `task`, `trial` (its folder), `outcome`, `ending`, `error_class`, `timed_out`, `rate_limited`, `agent_retries`, `upstream_retries`, `setting` (its row in `settings.rows`) and `control` |
| `settings` | the rows per setting, each with its figures, and `mixed`, `split`, `warning` and `effort_warnings` |
| `headline` | the rule's figures over the counted rows, or `null` when they are several settings |
| `seeding` | the seeds, as in `summary.json` |

To show the headline BenchFlow shows, the Hub must:

1. **Count every trial.** n is the number of trials in a row, errored ones included. A trial counts 1 only when its `outcome` is `passed`. Never leave out a trial because it errored or got no score: that gives a second, higher score for the same job.
2. **Give one score per setting.** Count each row of `settings.rows` apart. When `settings.split` is true, show no single number over the rows, only one per row; show `settings.warning` whenever `settings.mixed` is true, and each of `settings.effort_warnings`.
3. **Leave control runs out of an agent's score.** Trials with a `control` (oracle, empty) are their own rows; they check the task, not an agent.
4. **Give the same interval.** Use `ci95` from the row's figures, or compute it as this page says, from the trials' tasks and outcomes.
5. **Name a secondary figure as such.** If the Hub also shows the error-excluding figure, label it ("leaving out the trials with an error or no score"), and keep `refused` trials in it as failures.
6. **Check the arithmetic.** Reproduce every case of the conformance fixture below before showing any figure, and recompute each row's figures from `trials` to check the upload.

## Applying the rule elsewhere

```python
from benchflow.scoring import score, trial_record

records = [trial_record(result) for result in results]  # each trial's result.json
s = score(records)
s.score, s.score_excl_errors, s.n, s.n_tasks, s.se, s.ci95, s.ci_method
```

`trial_record(result)` classifies one trial from its `result.json`: its task, outcome, ending, error class, whether it ran out of time or met a rate limit, and the retries its calls show. `score(records)` takes records that each name their task and outcome, and refuses an unknown outcome or ending.

The conformance fixture is `tests/fixtures/scoring/benchflow.score.v1.json` in the BenchFlow repository. It lists trials, each with its task, outcome and ending, and most with the `result.json` fields BenchFlow reads them from. It gives the figures the rule yields for each case: every outcome, provider and harness endings, a reward that decides over a recorded error, one trial per task, one task, every trial passed, every trial errored, and no trials, with pass@k for each. A reader that reproduces every case counts the same way: exactly for the counts, within `tolerance` for the score, the error-excluding score, the standard error and the interval. The expected figures were computed apart from BenchFlow's code, with exact fractions for the score and SE², and the standard errors agree with a cluster-robust least-squares fit (statsmodels, tasks as clusters) to 1e-15. `tests/test_scoring_rule.py` checks that BenchFlow reproduces the fixture.

A change to what the rule counts gets a new id. `benchflow.score.v1` and its fixture do not change.
