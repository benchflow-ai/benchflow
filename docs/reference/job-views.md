# Job views: Outcomes, Pareto and Training

`bench eval view <job>` opens a job's run list with three more tabs. They answer the questions a run list cannot: which tasks each model or step solves, what a point of reward costs, and whether training moved held-out tasks. Give several job folders to compare them (`bench eval view jobs/a jobs/b`); a `job` dimension tells their trials apart.

Everything stays on your machine. The server reads the jobs once, in a background thread, through `bf.load_job` and serves the result at `/api/outcomes` on localhost; the page only regroups and draws. `--export out.html` writes the same views as one file you can share (see [Sharing](#sharing)).

## Outcomes

One square per trial. Rows are tasks, datasets, or all trials; columns group by agent, model, harness (ACP or native), seed or repeat, step or checkpoint, split, job, or outcome. **transpose** swaps rows and columns, rows sort by name, pass@1, mean reward or trial count, and a square opens its trial.

- **Reward** shades each square on one blue scale from 0 to 1, so partial credit shows as partial. **color: passed or not** switches to two shades.
- **Unscored** trials (the verifier failed, the sandbox never started, a usage limit, an agent crash) are hatched gray, never a reward color: they have no reward and are not counted as 0. Hovering one shows its cause, whose problem it is (task, agent, infrastructure or setup) and the next step, from `benchflow.failures.cause_of`, the table behind the end-of-run summary.
- A trial whose agent **errored or timed out and was still scored** keeps its reward and gets an amber border: the solution caused the failure, so it counts.
- **Retried** trials (an Evaluation job retries a task in its own folder) show a bar; hovering lists every attempt, oldest first. The square is the trial's result (`attempts="best"`), and its cost is the sum of its attempts.
- A **BenchShield** verdict (`--integrity audit|strict`, read with `Trial.integrity`) marks the square: a red ring and dot for an exploit (`AgentViolation`), a dashed ring for `VectorExposed`. A verdict never changes a reward. Trials run without `--integrity` have no mark, and the legend says so.
- Each row shows **pass@1 with its 95% interval**, pass@k at the largest k every task in the row reached, the mean reward when it differs, and how many trials were scored and unscored (from `benchflow.pass_at_k`, the numbers `bench eval metrics` prints). Control runs (oracle, nop) and a hill-climb's optimizer runs are left out of every statistic and hidden unless **show control and optimizer runs** is on.

## Pareto

Each point is one model (or agent, harness, step, job, seed or split), per dataset or split. x is the mean per trial of cost in USD, tokens, sandbox seconds or wall time, retries included; y is the mean reward, the solve rate or any other key of the trials' `rewards`. Both come with 95% bootstrap intervals that resample tasks (200 resamples, a fixed seed per point, so a page shows the same bars every time). The dashed line per dataset or split is its Pareto frontier: the points no other point beats on both cost and reward.

USD comes from the provider's price, or from the Claude Code session log when a subscription run was priced at the end of the run (`price_source: agent_session_log`, marked "estimated" on hover). Points with no USD recorded are listed under the chart instead of being drawn at zero.

## Training

When trials record a step, the Training tab plots reward per step for each split (or dataset), with 95% intervals, and compares held-out tasks between two steps you pick. Steps are read, in order, from:

1. `policy_version`, `policy_step`, `step`, `global_step` or `checkpoint` in the trial's `result.json` or `config.json`;
2. `step` in a `rollouts.jsonl` next to the trials (the RL adapters' audit log), matched by rollout folder;
3. a hill-climb run's rounds (`hillclimb.json` and its `evals/<version>/` folders), in the run's order.

Numeric steps sort as numbers. The rollout stream (`benchflow.rollout-stream.v1`) has no step field yet; a trainer that records its policy version in the trial's `config.json` gets this view with no other change.

## Where each dimension comes from

| Dimension | Read from |
| --- | --- |
| dataset | `dataset_name` in the result or config; else the folder holding the task (`task_path`), or the job's `tasks_dir` in `evaluation.json`, skipping folders named `tasks`, `train`, `test` and the like |
| split | a hill-climb's `split`; a recorded `split`; a `train`/`test`/`val`/`heldout` folder in the trial's path or its task's path |
| seed | a recorded `seed`; else a `trial-NN` repeat folder |
| harness | `harness_mode` in `config.json` (older trials: acp) |
| job | the trial's folder relative to the job you passed |

The page names the sources it used under each view.

## Sharing

`bench eval view <job> [<job> ...] --export outcomes.html` writes one self-contained HTML file (no network requests) and prints what it masked. Before writing, every string in the data goes through the same redaction `bench traj upload` applies (secret-shaped values such as API keys, bearer tokens and URL credentials become `<XXX-benchflow-key-values-XXX>`); the served folders' absolute paths become `<job>` and your home folder `~`. The file holds no trajectories and no trial links, only what the three views draw: task, model and folder names, rewards, costs, causes and the first line of each error.

## Size

The server builds the views once per start (`?refresh=1` on `/api/outcomes` rebuilds). On a 5,000-trial job it takes about 6 seconds, most of it reading the trial folders, and sends about 850 KB; the page draws the 5,000-square grid in about half a second.

## The document

`/api/outcomes` returns `benchflow.outcomes/1`, built by `benchflow.trajectories.viewer.outcomes.build_outcomes`: per-trial columns (`reward`, `outcome`, `execution`, `role`, `attempts`, `integrity`, `usd`, `tokens`, `sandbox_sec`, `wall_sec`, `cause`, `link`, `name`, `detail`), one `values`/`codes` pair per dimension, `stats` per dimension value, `pareto` points and frontiers, `training`, `sources` and `notes`.
