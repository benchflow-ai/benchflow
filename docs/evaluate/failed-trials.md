# When a trial fails

A trial is one agent's attempt at one task. The end of every run names each trial that didn't pass, with a code and the next step. Do what the `next:` line says, or find the code below:

```text
Outcomes: 0 scored (0 passed, 0 failed), 0 unscored, 1 errored
  1 errored: the agent integration broke: it does not offer the model (setup problem): econ-detrending-correlation
      next: pick a model the agent lists in the error
  `bench explain agent_integration:agent_model` says more
```

Here Claude Code was asked for `-m claude-sonnet-5-5`, a name 0.9.1's Claude Code doesn't offer, and the trial's error lists the ones it does ([Agents and models](./agents.md#choose-the-model-and-effort)).

A trial that scored 0 with a warning starting `Verifier scored 0:` in its `result.json` failed because of the task's test, not the agent: see [When the test itself fails](#when-the-test-itself-fails). A trial that scored 0 with no warning means the test ran and didn't pass the agent's work; [read its logs](#read-the-logs) for why. Some causes lie outside the agent, such as an output cap that cut a long answer short ([Known pitfalls](../audit/pitfalls.md#scores-that-count-the-wrong-thing)). If the run stopped before any trial started, `bench doctor` checks this machine's setup.

## Codes and what to do

A trial ends **scored** (the test reached a verdict: its reward, or 0 when the test itself broke), **unscored** (nothing was assessed, or a full disk lost the reward) or **errored** (it stopped before the test). The codes you'll meet most:

| Code | Outcome | What happened | What to do |
|---|---|---|---|
| `timeout` | errored | the agent ran out of time and its workspace couldn't be scored | raise the task's `agent.timeout_sec`, or pass `--timeout-multiplier` |
| `R203` | errored | the agent went silent for 10 minutes, the idle limit | raise `--idle-timeout` if the model thinks for long |
| `suspected_api_error` | errored | the agent did nothing: no tokens, no tool calls; usually a wrong model id or an expired login | check the model id, and the key with one direct call to the provider; then `bench eval resume <job>` |
| `R212` | errored | the model provider refused the credential | `bench doctor` shows which logins and keys it found; one direct call to the provider shows whether the key works |
| `provider_rate_limit` | errored | the provider rate-limited the run | lower `-j` (trials at a time), then `bench eval resume <job>` |
| `image_build` | errored | the task's image didn't build | fix that line in the task's `environment/Dockerfile`; `docker build <task>/environment` prints the whole log |
| `R401` | errored | the task asks for something this sandbox can't do, refused before it starts | the error names what; [Sandboxes](../scale/sandboxes.md#choose-a-sandbox) says which sandbox has it |
| `verifier_infra` | unscored | the verifier's sandbox failed | `bench eval resume <job>` runs the whole trial again, agent included |
| `disk_full` | errored or unscored | the disk holding the job folder filled; the reward was lost | free space, or pass another `--jobs-dir`; then `bench eval resume <job>` |

`<job>` is the job folder the run printed. Some codes also have a number, such as `R203`; `bench explain <code>` takes the name or the number and says more. An errored or unscored trial counts as 0 in the pass rate over all trials and drops out of the scored-only rate ([Compare runs](../audit/compare-runs.md#what-counts-in-the-denominator)).

Running out of time usually leaves a scored trial: the test grades what the agent left, so it counts as a pass or a fail, shown as `timed_out` and `scored` in [`bench results`](./read-results.md#a-job-trial-by-trial). `--fail-on timeout` makes the run exit 1 when any trial ran out of time.

## When the test itself fails

A broken test scores the trial 0 and counts it as a failure, with a `Verifier scored 0:` warning. We ran each case with the oracle (the task's reference solution) on 0.9.1:

| What the test did | What BenchFlow recorded | The warning |
|---|---|---|
| exited with an error and wrote no reward | scored, reward 0 | `Verifier scored 0: verifier exited with rc=1; no reward file found at ...` |
| ran past `verifier.timeout_sec`, set in the task's `task.md` | scored, reward 0 | `Verifier scored 0: verifier timed out after 10.0s` |
| wrote a reward outside 0 to 1, or outside the task's `verifier.reward_range` | scored, reward 0; `result.json` keeps the value in `reward_out_of_range_info` | `Verifier scored 0: reward out of range: verifier/reward.txt: reward = 2.0 is outside [0.0, 1.0]` |
| wrote something that isn't a number | scored, reward 0 | `Verifier scored 0: Failed to parse rewards from text file ...` |

What the test printed is in the trial's `verifier/test-stdout.txt`. Fix the test before you count such a trial against the agent.

A trial whose [rubric reviewer](../tasks/rubric-review.md) failed is unscored instead: `bench eval score <trial folder> --tasks-root <tasks folder> --force` runs the reviewer again on the saved work, without the agent.

One mismatch to know: `bench explain verifier_failure` and `bench explain R303` (a verifier timeout) describe a crashed or timed-out test as unscored, but 0.9.1 and 0.10.0rc1 score both 0, as the table shows.

## Read the logs

| File in the trial's folder | Holds |
|---|---|
| `result.json` | the outcome, the reward, warnings, and the error with its category |
| `verifier/test-stdout.txt` | what the tests printed |
| `agent/` | the agent's own log, such as `claude_agent_acp.txt` |
| `trajectory/acp_trajectory.jsonl` | every message and tool call, as the agent reported them over ACP (the Agent Client Protocol) |
| `error-traceback.txt` | the Python traceback, when BenchFlow itself raised |

`bench -v run ...` prints debug logs and each failed trial's traceback as it happens.

## Common failures

- **The agent did nothing.** A wrong model id or a dead key often looks like an agent error. Test the key with one direct call to the provider before a large run: `bench doctor` lists the keys it found but doesn't try them.
- **`Permission denied` in the agent's log.** The agent runs as the non-root `agent` user, which owns the working directory and its own home folder, nothing else. Keep the agent's work there, and install in the task's Dockerfile what it would need root for ([Task format](../tasks/task-format.md#the-environment)).
- **Out of disk.** Task images and agent installs take space: six Claude Code trials starting at once added about 14 GB on a 16-CPU build machine. A full disk ends trials with `disk_full`.
- **`Run failed: UnsupportedTaskFeatureError`.** The sandbox can't run what the task asks (`R401`). A multi-step task on 0.9.1, which no sandbox runs until [0.10](../tasks/multi-step.md):

  ```text
  [two-steps] Run failed: UnsupportedTaskFeatureError: Task uses parsed runtime features that BenchFlow cannot execute on the selected sandbox: - steps: multi-step tasks are not supported (one prompt per task) (sandbox=docker)
  ...
    1 errored: the task asks for something this sandbox backend cannot do (setup problem): two-steps
  ```

## Keep the sandbox and look inside (0.10)

> Needs BenchFlow `0.10.0rc1`, a release candidate that isn't published yet, so you can't install it from PyPI ([what you can do](../start/install.md#install-the-010-prerelease)).

`tasks/half-life` is the task [Write a task](../tasks/write-a-task.md) builds and shows how to copy.

`--keep-sandbox on-failure` keeps the sandbox of every trial that didn't pass, so you can look at what the agent left:

```bash
bench run tasks/half-life -a nop --keep-sandbox on-failure --jobs-dir jobs/keep
```

```text
[half-life] Kept the sandbox for inspection: bench sandbox attach jobs/keep/2026-11-04__18-11-41/half-life__14651333 (delete it with bench sandbox release jobs/keep/2026-11-04__18-11-41/half-life__14651333)
[FAIL] half-life (reward=0.00, tools=0)
```

`bench sandbox attach <trial>` opens a root shell in it; with a command after `--`, it runs that and exits:

```bash
bench sandbox attach jobs/keep/2026-11-04__18-11-41/half-life__14651333 -- ls -la /app
```

```text
total 20
drwxr-xr-x 1 root root 4096 Nov  4 18:09 .
drwxr-xr-x 1 root root 4096 Nov  4 18:11 ..
drwxr-xr-x 1 root root 4096 Nov  4 18:09 data
```

The `nop` agent, which does nothing, left only the task's data. A kept sandbox runs until you release it, on Docker or [Daytona](../scale/daytona.md):

```bash
bench sandbox release jobs/keep/2026-11-04__18-11-41/half-life__14651333
```

```text
Released the kept sandbox (Docker project half-life__14651333-d623edf8).
```
