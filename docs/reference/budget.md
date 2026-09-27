# Job budget caps

A job can be given a hard budget. When it is reached, BenchFlow stops launching trials and cancels the running ones. Cancelled and not-started trials are recorded as such, never as failures.

```bash
bench eval run --tasks-dir tasks --agent claude-agent-acp --sandbox daytona \
  --max-cost-usd 20 --max-sandbox-seconds 36000 --max-tokens 50000000
```

```python
import benchflow as bf

evaluation = bf.Evaluation(
    tasks_dir="tasks",
    jobs_dir="jobs/capped",
    config=bf.EvaluationConfig(environment="daytona"),
    budget=bf.Budget(max_cost_usd=20, max_sandbox_seconds=36_000),
)  # same as EvaluationConfig(budget=...); YAML: budget: {max_cost_usd: 20}
result = await evaluation.run()
print(result.budget)  # the summary.json "budget" block, or None
```

## The three caps

| Cap | Counts | Known |
|---|---|---|
| `--max-cost-usd` / `max_cost_usd` | USD over finished trials that reported a cost (usage tracking) | only when the trial reports USD; others are counted in `spent.usd_unknown_trials` and add nothing, with one warning |
| `--max-sandbox-seconds` / `max_sandbox_seconds` | trial wall-clock seconds, from the trial's start (sandbox creation included) to its end (retries included), for finished trials plus the elapsed time of running ones, re-checked every 0.2 s | always |
| `--max-tokens` / `max_tokens` | total tokens over finished trials | whenever the agent's usage reaches BenchFlow; others are counted in `spent.tokens_unknown_trials` |

A cap is reached when spent ≥ cap. USD and tokens arrive when a trial finishes, so a job can overshoot those two caps by what the trials in flight spend; the sandbox-seconds cap also counts running trials.

## What happens at the cap

- No new trial starts. Trials that were waiting are listed in `budget.not_started`.
- Running trials are cancelled. The cancellation reaches the rollout, whose cleanup tears down the sandbox, and no `result.json` is written. They are listed in `budget.cancelled`.
- Neither kind is in `total`, `passed`, `failed` or `errored`, so pass rates are over the trials that finished. The CLI prints a `Budget:` line with the reason and counts.
- A resume (`Evaluation.resume(job_dir)`, or re-running the same job) runs them, and counts what the finished trials already spent (tokens and USD from their `result.json`, sandbox-seconds from `timing.total`), so the cap stays a per-job cap across resumes.

`summary.json` gets a `budget` block only when a budget was set:

```json
"budget": {
  "caps": {"max_cost_usd": null, "max_sandbox_seconds": null, "max_tokens": 150},
  "spent": {"cost_usd": 0.0, "sandbox_seconds": 12.4, "tokens": 200,
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
