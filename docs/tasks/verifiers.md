# Verifiers

The verifier scores a trial. After the agent stops, BenchFlow copies the task's `verifier/` folder into the sandbox, runs `verifier/test.sh`, and reads the reward the script writes.

## Choose how a task is graded

| Grader | Use it when | It needs | Guide |
|---|---|---|---|
| `verifier/test.sh` | a script can check the answer: a file, a value, a command's output | no model | [How a verifier scores](#how-a-verifier-scores) |
| pytest, as `$BENCHFLOW_PYTEST` | the checks are pytest tests | no model | [Run pytest in a test](#run-pytest-in-a-test) |
| A rubric review, `verifier/rubric.json` | a right-looking answer can hide a wrong method, so someone has to read the work | a reviewer agent and model, on every run | [Rubric review](./rubric-review.md) |
| A model alone, an `llm-judge` strategy in `verifier/verifier.md` | no script can check the answer, such as a poem, and the files the agent leaves are all there is to grade | a model, with its key in the task's `verifier.env` | [Grade with a model, no test script](./llm-judge.md) |
| Human preferences (0.10) | quality is a judgement call (a design, a piece of writing), made in blind pairs | raters; 0.10.0rc2 | [Human preferences](./human-preferences.md) |
| The verifier in its own sandbox | the test shouldn't share a machine with what the agent left running | the task's artifacts declared | [Run the verifier in its own sandbox](#run-the-verifier-in-its-own-sandbox) |

`bench new` also writes a verifier document (`verifier/verifier.md` and `verifier/rubrics/`) that combines several checks into one weighted reward. Combining checks isn't covered in these guides, and [Write a task](./write-a-task.md#scaffold-it) removes the document. A verifier document can also hand the grading to a model, with no test ([Grade with a model, no test script](./llm-judge.md)); 0.9.2 refuses the `verifier.type: llm-judge` setting in `task.md`.

## How a verifier scores

`test.sh` writes the reward to `/logs/verifier/reward.txt` (a number) or `/logs/verifier/reward.json`. Whatever it prints goes to the trial's `verifier/test-stdout.txt`, so print the verdict first:

```text
FAIL: uncertainty 8.00 min, outside 0.3 to 3 (answer 38.50 +- 8.00 min, true value 37.4)
```

These rules decide the reward:

- Before `test.sh` runs, BenchFlow deletes any `reward.txt` or `reward.json` already there, so an agent can't leave a reward behind.
- A reward the script wrote counts, whatever its exit code. A script that exits with an error and writes nothing scores 0, with a warning: `Verifier scored 0: verifier exited with rc=1; no reward file found at ...`. A script that runs past `verifier.timeout_sec` is stopped and scores 0 the same way (`Verifier scored 0: verifier timed out after 10.0s`). Either way the trial counts in the score, as a failure, and what the script printed is in the trial's `verifier/test-stdout.txt`.
- `reward.json` holds the reward and any other scores: `{"reward": 1.0, "precision": 0.8}`. Every number in it is a score held to the reward range, so a detail like `"seed": 3` scores the trial 0; write details that aren't scores to `reward-details.json`. A `reward.txt` beside a `reward.json` gives the reward, and the JSON's details that aren't numbers are kept, not scored.
- The reward is 0.0 to 1.0. A task can widen that with `verifier.reward_range`, for example `[-1.0, 1.0]` to score a deliberate violation below doing nothing; it can't narrow it. A trial passes at 1.0; with a [rubric review](./rubric-review.md#write-the-rubric), it passes when its test and every blocker pass, and its reward is then the rubric's score, which can be below 1.

## Pass or fail

Write 1 when every check passes and 0 otherwise. The half-life task's test checks the estimate, the size of the uncertainty, and that the two agree; [Write a task](./write-a-task.md#write-the-test) has the whole script. Calibrate each tolerance against a right and a wrong solution: this one passes a fit that models the room's background (37.1 minutes) and fails one that ignores it (41.9).

Run a Python script in the test with `python3 -I`. The test runs in the agent's workspace, and without `-I` Python imports modules from there first: a `json.py` the agent left behind would replace the real one. [Stop reward hacking](./reward-hacking.md) shows the attack, and the test holding.

### Run pytest in a test

Don't run pytest under `python3 -I`: BenchFlow loads a plugin guard into pytest, `-I` keeps it out, and the trial ends unscored (`pytest could not load the plugin guard`). Run tests with `$BENCHFLOW_PYTEST`, BenchFlow's protected pytest. BenchFlow brings it, so the task's image needs no pytest of its own, and it prepares it only when a file in `verifier/` (or a value in the verifier's `env`) contains the text `$BENCHFLOW_PYTEST`, as the `test.sh` below does; a comment line is enough. Anywhere else, `$BENCHFLOW_PYTEST` names a stand-in that prints why and exits 4, and the test scores 0. It runs with only BenchFlow's checked plugins and the tests folder on its path. A test file, `verifier/test_answer.py`, for a task that asks for 42 in `/app/answer.txt`:

```python
from pathlib import Path


def test_answer_is_42():
    assert Path("/app/answer.txt").read_text().strip() == "42"
```

and `verifier/test.sh`:

```bash
#!/bin/bash
# The tests run in BenchFlow's protected pytest; its exit code decides the reward.
if "$BENCHFLOW_PYTEST" /verifier/test_answer.py -q; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

On 0.9.1 the oracle scores 1 and `nop` 0, both scored; the oracle's `verifier/test-stdout.txt` ends with `1 passed in 0.00s`.

### A warning about python3 in the image

Before the verifier runs, BenchFlow freezes the workspace and removes files the agent could use to change how pytest behaves, with root's own `python3`. An image with no root-owned `python3` in `/usr/bin`, `/bin` or `/usr/local/bin` still gets its score, and `result.json` lists a warning:

```text
the task's image has no root-owned python3 in /usr/bin, /bin or /usr/local/bin, so the Python steps of the verifier hardening (workspace freeze, hook clean-up, cache reclaim) were skipped; install python3 there to have them run
```

The Terminal-Bench 2.0 task `merge-diff-arc-agi-task` gave it with its reference solution, which scored 1. For a task of your own, install `python3` in the image, so the agent's files can't reach the test.

## Partial credit and tolerances

A reward between 0 and 1 gives credit for part of the work. Use it when the parts are worth something on their own, or for training, where a reward that moves a little with each improvement is easier to learn from than one that jumps from 0 to 1. This version of the half-life test gives a third for each check the answer passes ([`partial-credit-test.sh`](../examples/research-task/variants/partial-credit-test.sh)):

```python
checks = {
    "close_enough": off <= 2.5,                      # minutes from the true value
    "plausible_uncertainty": 0.3 <= unc <= 3.0,
    "consistent": unc > 0 and off <= 3 * unc,        # within 3 standard deviations
}
reward = round(sum(checks.values()) / 3, 4)
json.dump({"reward": reward}, open("/logs/verifier/reward.json", "w"))
```

The oracle and three wrong answers, run against it:

| Answer | The test's last line |
|---|---|
| the oracle's fit | `reward 1.0: close_enough pass, plausible_uncertainty pass, consistent pass` |
| the right value with a wide error | `reward 0.6667: close_enough pass, plausible_uncertainty fail, consistent pass` |
| a close value claimed too precisely | `reward 0.6667: close_enough pass, plausible_uncertainty pass, consistent fail` |
| a fit that ignores the background | `reward 0.3333: close_enough fail, plausible_uncertainty pass, consistent fail` |

A trial still passes only at 1.0, so a partial answer is a fail with a reward:

```text
[FAIL] half-life (reward=0.67, tools=0)
Job complete: 0/1 (0.0%), mean_reward=0.67, errors=0, idle_timeouts=0, time=0.3min
```

Report the pass rate and the mean reward separately: they answer different questions. Each tolerance here (2.5 minutes, 0.3 to 3 minutes, 3 standard deviations) was set by running a right and a wrong solution against it, as in [Write a task](./write-a-task.md#check-it-and-run-the-controls).

## Run the verifier in its own sandbox

By default the verifier runs in the agent's sandbox, after the agent's processes are stopped. For long tasks, or when you don't want the test to share a machine with whatever the agent left running, run it in a fresh sandbox that receives only the workspace:

```bash
bench run tasks/half-life -a oracle --verifier-sandbox separate
```

```text
[half-life] Running verifier...
[half-life] Rewards: {'reward': 1.0}
[PASS] half-life (reward=1.00, tools=0)
Verifier: 1 separate
--verifier-sandbox separate moved 1 task(s) that declare no artifacts: half-life. Their verifier gets only the workspace and /logs/artifacts, so an answer kept elsewhere (a service, a database, a file outside the workspace) scores 0 like an agent failure; check each task's oracle under the switch before comparing these zeros with task-mode runs.
```

The separate sandbox gets the workspace and the files the task declares under `artifacts:`. Anything else the agent changed (a running service, a file outside the workspace) stays behind, so run the oracle this way before you compare scores across the two modes. A task can ask for it on its own with `verifier.sandbox_mode: separate`.

## When the verifier can't run

On 0.9.2, a test that crashes before writing a reward, runs past its time limit, or writes a reward that isn't a number in range scores 0 and counts as a failure, with a `Verifier scored 0:` warning; 0.10.0rc2 records a missing or unusable reward as a verifier error instead (`verifier wrote no usable reward`), still a failure in every score, and a number out of range still scores 0; [When a trial fails](../evaluate/failed-trials.md#when-the-test-itself-fails) has each case's warning. A reward the test did write, such as the 0.67 above, counts as written. A trial ends **unscored** only when nothing was assessed: BenchFlow couldn't run the verifier, or its sandbox failed. Such a trial has no reward, and the summary counts it apart from failures; the reason is in `result.json` under `verifier_error`, with a category in `verifier_error_category`. A Terminal-Bench 4.1 task whose artifacts couldn't be copied to its separate verifier, on 0.9.1 and 0.9.2, ends like this:

```text
[ERR] interleaved-vigenere (tools=0) (separate verifier transfer failed: artifact…)
...
Outcomes: 0 scored (0 passed, 0 failed), 1 unscored, 0 errored
  1 unscored: the verifier's sandbox failed (infrastructure problem): interleaved-vigenere
      next: `bench eval resume jobs-0.9.1/2026-11-04__16-55-38` scores it again
```

An unscored trial is BenchFlow's or the task's problem, not the agent's; [Compare runs](../audit/compare-runs.md#what-counts-in-the-denominator) explains how unscored trials enter a score. `bench explain <category>` says what a category means.

## Fix a verifier and rescore

When a test turns out wrong, fix it and score the saved work again, without running the agent again. Runs started with `--save-outputs` keep each trial's final workspace for this. A first draft of the half-life test accepted anything within 15% of the true value, so a fit that ignored the background passed:

```text
PASS: 41.90 +- 0.37 min is 12.0% from 37.4; allowed 15%
```

After replacing the draft with the final test, `bench regrade` reruns it on the saved workspace:

```bash
bench regrade jobs/draft-cx/2026-11-04__17-58-34
```

```text
Running verifier...
Rewards: {'reward': 0.0}
           Regrade 20261104T175847Z-866168
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━━┓
┃ Trial               ┃ Original ┃ New ┃ Verdict    ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━━┩
│ half-life__c8819b13 │ 1        │ 0   │ pass->fail │
└─────────────────────┴──────────┴─────┴────────────┘
1/1 regraded, 1 changed (0 fail->pass, 1 pass->fail), 0 failed, 0 not regradable. Summary:
.../jobs/draft-cx/2026-11-04__17-58-34/regrade-summary.json
```

The original reward stays in each trial's `result.json`. The new one is in the trial's `regrade.json`, and the job's `regrade-summary.json` lists every change.
