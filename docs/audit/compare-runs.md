# Compare runs

Two scores differ. Before you say one agent, model or setting is better, check that the difference is bigger than the noise of the run. That means repeating the tasks, choosing the metric, counting the right trials, comparing the two runs task by task, and sizing the noise. The examples compare Claude Code on three SkillsBench tasks with and without the tasks' skills, three tries each ([Skills](../evaluate/skills.md)).

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`) with an API key: export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)); any model with an API key works the same way.

Words in the output: a **trial** is one rollout, one agent's attempt at one task; `--trials N` and the `trial-01` to `trial-NN` folders count **repeats**, passes over the whole task set; and `empty` in the Control column of `bench results` is the `nop` agent ([How BenchFlow works](../start/how-it-works.md#the-words)).

## Repeat each task

An agent doesn't do the same thing twice, so one try per task gives a number you can't tell from luck. `--trials N` runs the whole task set N times, each repeat a job of its own:

```bash
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -i offer-letter-generator \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 -j 3 \
  --jobs-dir jobs/no-skills
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -i offer-letter-generator \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 -j 3 \
  --task-skills --jobs-dir jobs/skills
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each)).

The two arms differ only in `--task-skills`, and each has a jobs folder of its own, so neither resumes the other.

Repeats are what pass@k and every interval below are computed from. More tasks usually help more than more tries per task: here two of the three tasks passed in every try, with or without skills, and told us nothing about the difference.

## Choose the metric

`bench eval metrics` reads every repeat under a folder:

```bash
bench eval metrics jobs/skills/claude-agent-acp --k 1 --k 3
```

```text
│ Score          │ 77.8% (7 of 9 trials)                          │
│ Mean reward    │ 0.778 (over 9 trials)                          │
│ Success rule   │ passed (reward = 1)                            │
│ Solve rate     │ 77.8% (7 of 9 trials, 3 tasks, n 3-3 per task) │
│ pass@1         │ 77.8% · pass^1 77.8% (3 tasks)                 │
│ pass@3         │ 100.0% · pass^3 66.7% (3 tasks)                │
```

| Metric | Means | Here |
|---|---|---|
| solve rate, pass@1 | the share of tries that pass, averaged over tasks | 77.8% |
| pass@k | the chance that at least one of k tries passes | pass@3 is 100%: every task passed at least once |
| pass^k | the chance that all k tries pass | pass^3 is 66.7%: two tasks passed every time |
| mean reward | the average reward, partial credit included | 0.778 |

pass@k and pass^k are the unbiased estimates from each task's n tries, c of them passing, averaged over the tasks with at least k tries: pass@k = 1 − C(n − c, k) / C(n, k) and pass^k = C(c, k) / C(n, k).

Headline the metric the benchmark itself reports: pass@1 for a pass-or-fail benchmark, mean reward for one that gives partial credit, as the SkillsBench leaderboard in [Reproduce a published score](./reproduce.md) does. pass@k says what an agent can do with retries and a way to pick the right answer; pass^k says how reliably it does it. Name the k and the number of tries behind each.

For an interval around one run's score, resample tasks, not trials: one task's tries aren't independent of each other. 0.9.2 prints no interval (0.10.0rc2 prints one over tasks), so compute it from the trials:

```python
import glob
import random
import statistics
from collections import defaultdict

import benchflow as bf

rewards = defaultdict(list)
for path in sorted(glob.glob("jobs/skills/claude-agent-acp/trial-*/*/")):
    for row in bf.load_job(path).to_records():
        rewards[row["task_name"]].append(row["reward"] or 0.0)  # unscored counts as 0
task_means = [statistics.fmean(r) for r in rewards.values()]
score = statistics.fmean(task_means)

random.seed(0)
boot = sorted(
    statistics.fmean(random.choices(task_means, k=len(task_means))) for _ in range(10_000)
)
print(f"score {score:.3f} over {len(task_means)} tasks, {sum(map(len, rewards.values()))} trials")
print(f"95% interval, resampling tasks: {boot[249]:.3f} to {boot[9749]:.3f}")
```

```text
score 0.778 over 3 tasks, 9 trials
95% interval, resampling tasks: 0.333 to 1.000
```

The score is the mean over tasks of each task's mean reward, so every task weighs the same however many tries it had; with the same number of tries per task, as here, it equals the mean over trials. It counts an unscored trial as 0, as the rate over attempted trials does; for the scored-only rate, skip the rows whose reward is `None`. Three tasks give a wide, rough interval; more tasks narrow it. To compare two runs, see [How big a difference is real](#how-big-a-difference-is-real).

## What counts in the denominator

A trial can pass, fail, end unscored (the verifier couldn't run) or error (the trial didn't finish). BenchFlow counts two ways: over every trial it attempted, with unscored and errored trials as 0, and over scored trials only. In this run they agree:

```text
skills: jobs/skills/claude-agent-acp — 7/9 attempted passed (78%), 7/9 scored passed (78%), 0 verifier errors, 0 unscored, 0 control runs left out
```

When they don't, the summary prints both, as in this run of a multi-step task that 0.9.2 can't execute:

```text
✗ Score: 0/1 (0.0%), mean reward 0.00 over all 1 trials (unscored as 0), errors=1; scored only: none scored
```

On 0.9.2, a scored 0 can also come from a broken test: a test script that crashed, timed out or wrote no usable reward scores the trial 0 and counts it as a failure ([When the test itself fails](../evaluate/failed-trials.md#when-the-test-itself-fails)). Before you compare, list those trials; each has a `Verifier scored 0:` warning in its `result.json`:

```bash
grep -rl --include=result.json "Verifier scored 0:" jobs/
```

```text
jobs/bad/2026-11-04__20-09-57/half-life__ce5a7fe9/result.json
jobs/crash/2026-11-04__19-57-52/half-life__71615bb5/result.json
jobs/range/2026-11-04__20-09-18/half-life__a93d947d/result.json
jobs/timeout/2026-11-04__19-58-09/half-life__00ddd411/result.json
```

Those four are the broken tests of that page's table: one wrote text instead of a number, one crashed, one wrote 2.0, one timed out.

Fix the test and score those trials again, which needs runs that kept their workspaces (`--save-outputs`, as in [Run a benchmark](../evaluate/run-a-benchmark.md); then [Fix a verifier and rescore](../tasks/verifiers.md#fix-a-verifier-and-rescore)), or leave their tasks out on both sides and say so. A rescore leaves each `result.json` as it was and writes the new reward to the trial's `regrade.json`, so compute the comparison again from those.

Report both, with the counts. To compare two runs, use the tasks both scored, so a failure on one side's infrastructure doesn't count against its agent, and give the rate over all attempted trials beside it; `bench results A B` already restricts itself to the tasks both sides ran. If the trials that went unscored are not random, say the hardest tasks, the scored-only comparison favours whichever side lost them: list the tasks you left out, and why. In Python, `bf.load_job(path).denominators()` returns them:

```text
Denominators(attempted=3, scored=3, assessment_errors=0, unscored=0, execution_errors=0, passed=3, mean_reward=1.0, ...)
```

## Compare two runs task by task

Give `bench results` two folders. It pairs the tasks by name and compares their rewards:

```bash
bench results jobs/no-skills/claude-agent-acp jobs/skills/claude-agent-acp
```

```text
3 tasks paired (3 scored on both sides): skills higher on 1, lower on 0, same on 2; mean delta (skills - no-skills) 0.111111. 0 only in no-skills, 0 only in skills.

| Task | no-skills | skills | Delta | Status |
|---|---|---|---|---|
| crystallographic-wyckoff-position-analysis | 0 | 0.333333 | 0.333333 | paired |
| econ-detrending-correlation | 1 | 1 | 0 | paired |
| offer-letter-generator | 1 | 1 | 0 | paired |

- Tasks are paired by recorded task name.
- Some tasks have several runs per side; per-side rewards are means over scored runs.
- No statistical significance is computed or implied.
- Control runs (oracle, empty/nop) are left out.
```

The table is the place to start: a gain on one task and none elsewhere reads differently from a small gain everywhere. The output above is 0.10.0rc1's, which tests no significance; the next section does that by hand. 0.10.0rc2 also prints both sides' scores by the [scoring rule](../reference/scoring-rule.md) on the tasks both ran, and the gap with its 95% interval paired by task: the t interval over tasks that `paired.py`, below, gives when every task has the same number of trials. They warn when that interval holds 0 or the gap is under 3 points, since then the comparison does not say which side is better, and when a side has fewer than 3 trials per task.

### Check that the runs differ only where you meant

`bench eval compare A B` pairs the same way and also checks, task by task, that both sides ran with the same settings: the task's digest, model, harness, dataset, effort, sandbox, sandbox user, timeout, the agent's variables and the prompts. 0.10.0rc2 also checks what each trial's records say ran: the effort its calls carried (`effort_sent`), the model that served them (`model_served`), their `route`, the `harness_version` and the `benchflow_version`. Two runs that asked for the same effort and sent different ones differ; declaring `--vary reasoning_effort` covers the effort sent, `--vary model` the served model and route, `--vary harness` its version. Name the settings your comparison is about with `--vary`; with `--on-mismatch raise`, any other difference stops it with exit code 1. Claude Code on MiMo against OpenCode on DeepSeek, with nothing declared:

```bash
bench eval compare jobs/skills/claude-agent-acp/trial-01 jobs/opencode/2026-11-04__17-49-57 --on-mismatch raise
```

```text
Settings differ between the sides on 1 paired task(s): agent_variable_names, harness, model (agent_variable_names on econ-detrending-correlation: ...
  next: `bench eval compare --help` describes --vary, --by and --on-mismatch
```

With `--vary model --vary harness` it pairs the task, prints the comparison and exits 0: the agent's variables follow from the harness and the model, so declaring either covers them. `--vary` and `--on-mismatch` work only on `bench eval compare`, not on `bench results A B`. The check doesn't look at skills: the run above with `--task-skills` and the one without compare as the same settings, so state that difference yourself.

## How big a difference is real

First say what you're estimating. Here it's the difference in pass rate between the arms, averaged over these three tasks: +0.111. Then ask one of two questions, and use the unit that matches it:

- **Did these tasks score differently?** The unit is the trial. Shuffle trials between the arms within each task, and see how often the shuffle gives a difference as large.
- **Would the difference hold on other tasks like these?** The unit is the task. Flip the sign of each task's difference, and see how often that gives one as large. This assumes the tasks are a random sample from the tasks you want to speak for, and that without a real effect each task's difference is as likely to fall either way. With k tasks, this p can't go below 2/2^k.

[`paired.py`](../examples/compare/paired.py) does both, exactly, from the two folders, and gives the difference a 95% interval over tasks. It needs only Python:

```bash
curl -fsSLO https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/compare/paired.py
python3 paired.py jobs/no-skills/claude-agent-acp jobs/skills/claude-agent-acp
```

```text
task                                                A       B  B-A
crystallographic-wyckoff-position-analysis    0/3     1/3     +0.333
econ-detrending-correlation                   3/3     3/3     +0.000
offer-letter-generator                        3/3     3/3     +0.000

mean difference in pass rate over 3 tasks (B - A): +0.111
95% t interval over tasks: -0.367 to +0.589
within-task permutation test, two-sided: p = 1.000 (8000 splits)
sign-flip test over tasks, two-sided: p = 1.000 (the smallest possible with 3 tasks: 0.250)
```

Neither test can tell this difference from chance, and the interval runs from a 37-point loss to a 59-point gain. The within-task test can't, because one task moved by one trial; the across-task test never could, with three tasks.

Intervals on the pooled trials say the same. With 9 trials a side, 95% intervals (Wilson's) are wide:

| Arm | Passed | 95% interval |
|---|---|---|
| no skills | 6 of 9 | 35% to 88% |
| skills | 7 of 9 | 45% to 94% |

They treat the 9 trials as independent draws, so they describe the noise of these three tasks, not of the benchmark. By the same assumption, telling 67% from 78% apart, at the usual 5% false-positive rate with an 80% chance of seeing a real difference, would take about 250 trials per arm. For a claim about the benchmark, add tasks rather than tries: the across-task test needs at least 6 tasks to reach p below 0.05 at all, and a difference that holds on most of them.

So write the result as what it is: 7 of 9 against 6 of 9, a difference of one trial on one task, not significant. The [report](./report.md) page lists what else a result needs. 0.10's `report.md` prints each task's interval for you, counting trials as independent, and 0.10.0rc2's gives the score's interval over tasks too.
