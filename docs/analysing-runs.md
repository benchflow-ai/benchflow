# Analysing runs in Python or a notebook

`bf.load_job` reads finished runs back from disk; nothing here starts a sandbox. It counts trials the way the viewer does, so numbers you compute in a notebook match the viewer and `bench eval inspect`.

## Load a job

```python
import benchflow as bf

job = bf.load_job("jobs/run-b")        # a job dir, a folder of jobs, one trial, or a list
job.trials
# [Trial(task='fetch-note', agent='oracle', model=None, reward=1.0, execution='completed', assessment='scored'),
#  Trial(task='hello-world-task', agent='claude-agent-acp', model='claude-haiku-4-5', reward=1.0, execution='completed', assessment='scored'),
#  Trial(task='hello-world-task', agent='oracle', model=None, reward=1.0, execution='completed', assessment='scored')]
```

Each trial has two separate verdicts. `execution` says whether the run finished: `completed`, `errored` or `timed_out`. `assessment` says whether it got a score: `scored`, `error` (the verifier failed) or `unscored`. A trial with no verdict has `reward` None — never 0 — so an infrastructure failure does not count as a failed attempt.

## Count

```python
d = job.denominators()
# Denominators(attempted=1, scored=1, assessment_errors=0, unscored=0, execution_errors=0,
#              passed=1, mean_reward=1.0, clean_scored=1, clean_passed=1, controls_excluded=2)
d.pass_rate_scored, d.pass_rate_attempted     # (1.0, 1.0)
```

Control runs (the oracle, and the empty `nop` agent) check the task, not an agent, so they are left out: `controls_excluded=2` above. Pass `include_controls=True` to count them. `pass_rate_scored` divides by the trials that got a score; `pass_rate_attempted` divides by every attempt, so errors count against the agent. `clean_*` also leaves out scored runs whose execution failed.

Per agent and model, the facet table:

```python
for g in job.denominators_by(("agent", "model"), include_controls=True):
    print(g.key, g.denominators.attempted, g.denominators.passed)
# {'agent': 'claude-agent-acp', 'model': 'claude-haiku-4-5'} 1 1
# {'agent': 'oracle', 'model': None} 2 2
```

The group keys are listed in `bf.jobs.GROUP_KEYS` (agent, model, control, and the run settings such as reasoning_effort, environment and task_digest). `bench eval inspect JOB --by agent,model` prints the same table.

## Into pandas

```python
import pandas as pd

df = pd.DataFrame(job.to_records())       # one row per trial; job.to_csv(path) writes the same
df[["task_name", "agent", "model", "reward", "execution", "assessment", "control"]]
```

Rows carry usage (`n_input_tokens`, `n_output_tokens`, `n_cache_read_tokens`, `n_cache_creation_tokens`, `total_tokens`, `cost_usd`), timing, errors with categories, and the run settings. `cost_usd` is None for agents on a subscription login (`usage_source` `agent_native_acp`), which report tokens but no price. When you load several jobs at once, the `job` column tells them apart.

## Compare two runs

```python
cmp = bf.compare("jobs/run-a", "jobs/run-b", labels=("a", "b"))
print(cmp.to_markdown())
# On the 1 tasks both sides ran: a 1/1 scored passed (100%), …; b 1/1 scored passed (100%), …
# | Task | a | b | Delta | Status |
# | hello-world-task | 1 | 1 | 0 | paired |
```

Trials are paired by task name. The headline uses `cmp.a_paired` / `cmp.b_paired`, the denominators over the tasks both sides ran; `cmp.a` / `cmp.b` cover every task each side ran. `compare` also checks the settings each pair ran with (task digest, model, harness, reasoning effort, timeout, prompts, …) and warns when they differ. Name the settings you meant to change with `vary=("model",)`. To compare several agents or models in one pair of jobs, pair on them too with `by=("agent", "model")`; without it a side that mixes models is flagged in the caveats.

## Pitfalls

- A task retried several times keeps one trial per agent and model: the scored one, then the newest (`attempts="best"`, the default). Pass `attempts="all"` to see every attempt.
- `bf.load_job` wants a job folder or a `results.jsonl` file. It refuses `summary.json` or `result.json` paths and names the folder to pass instead.
- With n = 1 per task, agent variance alone can flip a pass to a fail; treat a difference as something to investigate, not as an effect size.
- Folders where an attempt started but never wrote `result.json` (a crash or a kill) are in `job.interrupted`, not in `job.trials`.

## Export

`job.to_json_dict()`, `trial.to_json_dict()` and `cmp.to_json_dict()` produce versioned documents (`benchflow.job`, `benchflow.trial`, `benchflow.comparison`) that the viewer and other tools read; see [JSON export](./reference/json-export.md). The full API is in the [Python API reference](./reference/python-api.md#reading-finished-jobs).
