# Job budget caps

A job can be given a budget. When it is reached, BenchFlow stops starting rollouts (a rollout is one run of a task: a trial's first attempt or a retry) and, for the spend caps, cancels the running ones. Cancelled and not-started trials are recorded as such, never as failures.

```bash
bench eval run --tasks-dir tasks --agent claude-agent-acp --sandbox daytona \
  --max-cost-usd 20 --max-sandbox-seconds 36000 --max-tokens 50000000 --max-rollouts 200
```

```python
import benchflow as bf

evaluation = bf.Evaluation(
    tasks_dir="tasks",
    jobs_dir="jobs/capped",
    config=bf.EvaluationConfig(environment="daytona"),
    budget=bf.Budget(max_cost_usd=20, max_sandbox_seconds=36_000, max_rollouts=200),
)  # same as EvaluationConfig(budget=...); YAML: budget: {max_cost_usd: 20}
result = await evaluation.run()
print(result.budget)  # the summary.json "budget" block, or None
```

## The four caps

| Cap | Counts | Known | Enforced |
|---|---|---|---|
| `--max-cost-usd` / `max_cost_usd` | USD over finished rollouts that reported a cost (usage tracking, or an estimate from the agent's own session log), retries included | only when the rollout reports USD; others are counted in `spent.usd_unknown_trials` and add nothing, with one warning | between rollout starts (below) |
| `--max-sandbox-seconds` / `max_sandbox_seconds` | trial wall-clock seconds, from the trial's start (sandbox creation included) to its end (retries included), for finished trials plus the elapsed time of running ones | always | while trials run, re-checked every 0.2 s |
| `--max-tokens` / `max_tokens` | total tokens over finished rollouts, retries included | whenever the agent's usage reaches BenchFlow; others are counted in `spent.tokens_unknown_trials` | between rollout starts (below) |
| `--max-rollouts` / `max_rollouts` | rollouts started: every trial's first attempt and every retry | always | at each start |

A spend cap is reached when spent ≥ cap. USD and tokens are only known when a rollout finishes, so they cannot stop a rollout that is running. They are enforced between starts instead: once some rollouts have finished, a trial waits to start while the running trials, at the job's mean USD (or tokens) per finished rollout, would reach the cap with it, and starts when one of them finishes and the estimate allows it. So a job passes a USD or token cap by about one rollout's spend. The exception is the first wave: until a rollout has finished there is no mean, so the first `concurrency` trials start without an estimate. With `concurrency` 1 nothing waits, and the last rollout can pass the cap by its own spend.

## What happens at the cap

- No new rollout starts: neither a new trial nor a retry of a running one (a trial that is not retried keeps its last attempt's result). Trials that were waiting are listed in `budget.not_started`.
- For the spend caps, running trials are cancelled. The cancellation reaches the rollout, whose cleanup tears down the sandbox, and no `result.json` is written. They are listed in `budget.cancelled`. At the rollout cap, running trials finish.
- Neither kind is in `total`, `passed`, `failed` or `errored`, so pass rates are over the trials that finished. The CLI prints a `Budget:` line with the reason and counts.
- A resume (`Evaluation.resume(job_dir)`, or re-running the same job) runs them, and counts what every earlier rollout of the job spent (tokens and USD from each attempt's `result.json`, retried and re-run attempts included; sandbox-seconds from `timing.total`; each attempt folder, even one that wrote no `result.json`, as a rollout), so the cap stays a per-job cap across resumes.

`summary.json` gets a `budget` block only when a budget was set:

```json
"budget": {
  "caps": {"max_cost_usd": null, "max_sandbox_seconds": null, "max_tokens": 150,
           "max_rollouts": null},
  "spent": {"cost_usd": 0.0, "sandbox_seconds": 12.4, "tokens": 200, "rollouts": 2,
            "usd_unknown_trials": 2, "tokens_unknown_trials": 0},
  "stopped": true,
  "reason": "budget reached: 200 tokens spent of a 150 cap",
  "cancelled": [],
  "not_started": ["task-2", "task-3"]
}
```

## Limits

- The cap is per job. With `--matrix`, each model × trial job has its own cap.
- Not supported with `--worker-concurrency` (each worker process would get the whole budget) or `--source-env` (vf-eval runs those rollouts); both are refused.
- Branch jobs (`bench eval branch`) do not take a budget.
