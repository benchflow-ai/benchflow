# Write a task

This tutorial builds a small research task the way a physics research benchmark does: the agent gets real data, must produce a number with an uncertainty, and a test checks the answer against a tolerance. You'll write the task, prove it with its reference solution and an agent that does nothing, then run Claude Code on it. Building and checking it takes about twenty minutes; the three Claude Code tries at the end run up to 15 minutes each. You need BenchFlow [installed](../start/install.md) and Docker.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

The finished task is in the repository at [`docs/examples/research-task/half-life`](../examples/research-task/half-life), with the files used to test it. Other guides run it as `tasks/half-life`; to follow them without building it, copy it:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/research-task/half-life tasks/
rm tasks/half-life/verifier/rubric.json   # Rubric review adds it; the other guides run without it
```

## The task

A radioactive sample was counted every minute for two hours, in a room with a constant background that wasn't measured. The agent must estimate the half-life, with a one-standard-deviation uncertainty. The data were generated with a half-life of 37.4 minutes, so the test knows the answer; the agent doesn't.

A good test for this has to fail the tempting wrong answer. A fit that ignores the background gets 41.9 ± 0.4 minutes; one that models it gets 37.1 ± 1.1. The test below fails the first and passes the second.

## Scaffold it

```bash
bench new half-life
```

```text
Created: tasks/half-life/
  environment/Dockerfile
  oracle/solve.sh
  task.md
  verifier/rubrics/verifier.md
  verifier/rubrics/verifier.toml
  verifier/test.sh
  verifier/test_outputs.py
  verifier/verifier.md
Next: replace every [REPLACE: ...] placeholder (bench check lists them), then prove the task with its reference solution:
bench check tasks/half-life
bench run tasks/half-life --sandbox docker --jobs-dir jobs/half-life-oracle --fresh
```

Every file starts with a placeholder that fails on purpose, so an unfinished task can't pass. `task.md` holds the settings and the instruction, `environment/Dockerfile` the sandbox and `oracle/solve.sh` the reference solution. The verifier folder holds more than one task needs: `test.sh` is the test, which writes the reward; `test_outputs.py` is a pytest file, though the scaffold's `test.sh` never runs it (to keep it, call it from `test.sh` as [Verifiers](./verifiers.md#run-pytest-in-a-test) shows; `bench new --no-pytest` leaves it out); and `verifier.md` with `rubrics/` is a verifier document, which combines several checks into one weighted reward. This task grades with `test.sh` alone, so delete the rest:

```bash
rm -r tasks/half-life/verifier/rubrics tasks/half-life/verifier/test_outputs.py tasks/half-life/verifier/verifier.md
```

## Build the environment and add the data

The sandbox is built from `environment/Dockerfile`. Copy the data in and install what a solution needs:

```dockerfile
FROM ubuntu:24.04
RUN apt-get update -qq && apt-get install -y -qq python3 python3-numpy && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data/ /app/data/
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
```

The data are 120 rows of `minute,counts`, generated with a half-life of 37.4 minutes. Download them into the task:

```bash
mkdir -p tasks/half-life/environment/data
curl -fsSL -o tasks/half-life/environment/data/counts.csv \
  https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/research-task/half-life/environment/data/counts.csv
```

Everything in the image is visible to the agent, so nothing that gives away the answer goes here.

## Write the instruction

`task.md` holds the settings between the `---` lines and the instruction below them:

```markdown
---
schema_version: "1.3"
metadata:
  author_name: BenchFlow docs
  difficulty: medium
  category: capability
  tags: [physics, data-analysis]
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 120
sandbox:
  cpus: 1
  memory_mb: 2048
---
# half-life

## prompt

The file `/app/data/counts.csv` holds the counts recorded in each minute of a two-hour measurement of a radioactive sample (columns `minute` and `counts`). The room adds a constant background, which was not measured separately.

Estimate the sample's half-life in minutes, with a one-standard-deviation uncertainty. Write `/app/answer.json` as `{"half_life_min": <number>, "uncertainty_min": <number>}`. Write `/app/report.md`: how you estimated both numbers, how you handled the background and the counting noise, and how to reproduce your result. Leave the script you used in `/app`.
```

`agent.timeout_sec` is the agent's time limit, 15 minutes here. [Task format](./task-format.md) lists every setting.

The instruction states the standard to meet, an estimate with an honest uncertainty, and not the grader's numbers: it doesn't give the 2.5-minute tolerance or the accepted range of uncertainties, which come from running a right and a wrong method against the test (below). If your benchmark wants agents to see them, put them in the instruction; either way, write the choice down.

## Write the test

`verifier/test.sh` runs after the agent stops. It reads the answer and writes the reward, 1 or 0, to `/logs/verifier/reward.txt`:

```bash
#!/bin/bash
# -I: import nothing from the working directory, which is the agent's workspace (/app).
python3 -I - <<'PY'
import json

TRUE = 37.4  # minutes: the half-life the data were generated with
reward = 0
try:
    answer = json.load(open("/app/answer.json"))
    est, unc = float(answer["half_life_min"]), float(answer["uncertainty_min"])
    off = abs(est - TRUE)
    # A fit that models the background gives 37.1 +- 1.1; one that ignores it gives 41.9 +- 0.4.
    checks = [
        (off <= 2.5, f"{off:.2f} min from the true value, more than 2.5"),
        (0.3 <= unc <= 3.0, f"uncertainty {unc:.2f} min, outside 0.3 to 3"),
        (unc > 0 and off <= 3 * unc,
         f"the true value is {off / unc:.1f} standard deviations away, more than 3" if unc > 0
         else "the uncertainty is not positive"),
    ]
    failed = [why for ok, why in checks if not ok]
    reward = 0 if failed else 1
    verdict = "FAIL: " + "; ".join(failed) if failed else "PASS"
    print(f"{verdict} (answer {est:.2f} +- {unc:.2f} min, true value {TRUE})")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"FAIL: no usable /app/answer.json ({exc})")
open("/logs/verifier/reward.txt", "w").write(f"{reward}\n")
PY
```

Three choices in it matter:

- **The tolerances are stated and calibrated.** 2.5 minutes passes the right method's 37.1 and fails the background-free 41.9. The uncertainty must be plausible (0.3 to 3 minutes) and must cover the true value within three standard deviations, so a wide guess and an overconfident one both fail.
- **It prints the verdict first.** `test-stdout.txt` then says why a trial failed in one line.
- **`python3 -I`.** The test runs in the agent's workspace. Without `-I`, a `json.py` the agent left in `/app` would replace Python's own `json` module and could answer for the test. [Stop reward hacking](./reward-hacking.md) shows this attack failing against this test. A test that runs pytest uses `$BENCHFLOW_PYTEST` instead ([Verifiers](./verifiers.md#run-pytest-in-a-test)).

## Write the reference solution

`oracle/solve.sh` solves the task the way an expert would; the `oracle` agent runs it. This one writes and runs a weighted least-squares fit of `A·exp(−ln2·t/T) + B`, with Poisson weights, takes the uncertainty from the chi-square profile, and writes the answer and a report:

```bash
#!/bin/bash
set -e
cat > /app/fit.py <<'PY'
"""Weighted least squares of counts = A exp(-ln2 t / T) + B, profiled over T."""
import json

import numpy as np

d = np.loadtxt("/app/data/counts.csv", delimiter=",", skiprows=1)
t, c = d[:, 0] + 0.5, d[:, 1]  # each minute's centre
w = 1 / np.maximum(c, 1)  # Poisson: the variance of a count is the count


def chi2(T):
    X = np.column_stack([np.exp(-np.log(2) * t / T), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], c * np.sqrt(w), rcond=None)
    return float(np.sum(w * (c - X @ coef) ** 2))


grid = np.arange(20, 80, 0.005)
vals = np.array([chi2(T) for T in grid])
best = float(grid[vals.argmin()])
inside = grid[vals <= vals.min() + 1]  # 1 sigma: chi-square within 1 of its minimum
unc = float(inside.max() - inside.min()) / 2
json.dump({"half_life_min": round(best, 3), "uncertainty_min": round(unc, 3)}, open("/app/answer.json", "w"))
print(f"half-life {best:.3f} +- {unc:.3f} min")
PY
python3 /app/fit.py
cat > /app/report.md <<'MD'
# Half-life

- Model: counts in each minute = A exp(-ln2 t / T) + B, with B the constant background, at each minute's centre.
- Fit: weighted least squares with Poisson weights (variance = counts); A and B are solved for each T, and T is scanned from 20 to 80 minutes.
- Uncertainty: half the width of the range of T where chi-square is within 1 of its minimum.
- Reproduce: `python3 /app/fit.py`.
MD
```

## Check it and run the controls

`bench check` reads the task without starting a sandbox:

```bash
bench check tasks/half-life
```

```text
✓ half-life — valid (structural)
```

`valid (structural)` means the files and settings are well formed; it doesn't say the test is right. The controls do that. Then the two controls. The reference solution must score 1, and doing nothing must score 0:

```bash
bench run tasks/half-life -a oracle
bench run tasks/half-life -a nop --fresh
```

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
...
• Score: 0/1 (0.0%), mean reward 0.00, errors=0
nop control: 0/1, as expected: doing nothing passes none of the tasks
```

A task that fails either control can't tell you anything about an agent. The test also has to fail wrong answers; we ran three wrong solutions against it as scripted agents, and it failed each one:

| Wrong answer | What the test printed |
|---|---|
| A fit that ignores the background | `FAIL: 4.50 min from the true value, more than 2.5; the true value is 12.2 standard deviations away, more than 3 (answer 41.90 +- 0.37 min, true value 37.4)` |
| The right value with a wide error | `FAIL: uncertainty 8.00 min, outside 0.3 to 3 (answer 38.50 +- 8.00 min, true value 37.4)` |
| A close value claimed too precisely | `FAIL: the true value is 5.3 standard deviations away, more than 3 (answer 39.00 +- 0.30 min, true value 37.4)` |

A fourth one passes: the right number with no method behind it (`PASS (answer 37.40 +- 1.00 min, true value 37.4)`). A test of the number can't see how it was found. [Rubric review](./rubric-review.md) adds a check of the method to this task. [Check a benchmark](../audit/check-a-benchmark.md) shows how to run wrong solutions like these as agents.

## Run an agent on it

Three tries of Claude Code on MiMo V2.6 Flash, with each reward's integrity audited ([Stop reward hacking](./reward-hacking.md#audit-how-each-reward-was-earned)):

```bash
bench run tasks/half-life -a claude -m xiaomi/mimo-v2.6-flash --trials 3 --integrity audit
```

```text
[half-life] Agent prompt exceeded wall-clock budget 900s
[half-life] Running verifier...
[half-life] Rewards: {'reward': 1.0}
[PASS] half-life (reward=1.00, tools=16) (Agent prompt exceeded wall-clock budget 900s)
...
Matrix: 3 trial job(s), 3/3 trials passed, errors=0 verifier-errors=0
```

| Try | What the test printed | Tokens |
|---|---|---|
| 1 | `PASS (answer 37.30 +- 1.15 min, true value 37.4)` | 397,000 |
| 2 | `PASS (answer 37.30 +- 1.17 min, true value 37.4)` | 579,794 |
| 3 | `PASS (answer 37.30 +- 1.14 min, true value 37.4)` | 659,155 |

All three passed, and all three ran into the 15-minute limit: the agent wrote its answer early and went on checking it until time ran out. The test scores only the files the agent left, so a trial that hits the limit after writing a right answer still passes. Each such trial shows `timed_out` in the Exec column of `bench results`, and its `result.json` records the timeout. If finishing in time matters to you, count those trials separately, or make a run fail on them with `--fail-on timeout`.

The audit found nothing the agent did that reached the reward in any try. [Stop reward hacking](./reward-hacking.md#audit-how-each-reward-was-earned) shows how to read its verdicts.

## Next

- Grade the report and the method as well as the number: [Rubric review](./rubric-review.md).
- Prove an agent can't cheat the test: [Stop reward hacking](./reward-hacking.md).
- Share a folder of tasks as a benchmark: [Publish a benchmark](./publish.md).
