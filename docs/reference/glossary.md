# Glossary

Each term these docs use, in a line, with the page that covers it.

| Term | Meaning | More |
|---|---|---|
| ACP | Agent Client Protocol: how BenchFlow talks to most agents. | [Agents and models](../evaluate/agents.md) |
| Advantage | How much better a rollout did than the others in its group; trainers learn from it. | [BenchFlow as an RL environment](../train/environment.md#rewards-and-advantages) |
| Agent | The program that does a task, such as Claude Code or Codex. | [Agents and models](../evaluate/agents.md) |
| Artifacts | Files a task declares, saved in each trial's folder and given to a separate verifier. | [Task format](../tasks/task-format.md) |
| ATIF | A JSON format for agent trajectories; each trial writes one in `trainer/atif.json`. | [Trajectory and rollout formats](../train/formats.md) |
| Attempt | One try at a trial. A trial that fails for a reason another try could fix, such as a dropped connection, runs again in a new sandbox, up to 2 more times by default; the trial counts its best attempt. | [Resume, retries and regrade](../scale/resume.md#retries) |
| Benchmark | A set of tasks. Also called a dataset. | [Find a benchmark](../evaluate/find-a-benchmark.md) |
| bfx | A small Linux program BenchFlow copies into each sandbox to run its own steps there; `bench doctor` checks it. | [Install](../start/install.md#check-your-machine) |
| Blocker | A rubric criterion that must pass; if it fails, the trial's reward is 0. | [Rubric review](../tasks/rubric-review.md#write-the-rubric) |
| Control | A run whose result you know in advance: the oracle should score 1, `nop` 0. | [Check a benchmark](../audit/check-a-benchmark.md) |
| Digest | A hash over a task's files; the same digest means the same task. | [Publish a benchmark](../tasks/publish.md#version-the-tasks) |
| Effort | How hard a reasoning model thinks, set with `-e`. | [Agents and models](../evaluate/agents.md#choose-the-model-and-effort) |
| Errored | A trial that stopped before it could be scored. | [When a trial fails](../evaluate/failed-trials.md) |
| Environment | What a task's sandbox is built from: its `environment/Dockerfile` and files, or a named image. | [Task format](../tasks/task-format.md#the-environment) |
| Group | Rollouts of the same task, agent and model, compared with each other for an advantage. | [BenchFlow as an RL environment](../train/environment.md#rewards-and-advantages) |
| Harness | How an agent runs: through ACP (the default) or its own CLI (`--harness native`). | [Agents and models](../evaluate/agents.md) |
| Integrity audit | `--integrity audit`: a verdict next to each reward on whether anything the agent did reached it. | [Stop reward hacking](../tasks/reward-hacking.md#audit-how-each-reward-was-earned) |
| Job | The rollouts one command starts, one per task, with their folder under `jobs/`; Python calls it an `Evaluation`. | [How BenchFlow works](../start/how-it-works.md) |
| LLM judge | A model that grades work against a rubric. | [Rubric review](../tasks/rubric-review.md) |
| Matrix | The jobs one command starts: the repeats of `--trials`, or one job per model listed in a `--matrix` file. Its totals print on the `Matrix:` line and in `matrix-summary.json`. | [Run a benchmark](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each) |
| Model proxy | The proxy API-key model calls go through, which counts tokens and keeps your key from the agent; it runs on your machine for local Docker and inside each sandbox on Daytona and Modal. | [Usage and telemetry](../scale/telemetry.md) |
| Multi-step task | A task in steps, each checked before the next starts (0.10). | [Multi-step tasks](../tasks/multi-step.md) |
| nop | The agent that does nothing; a control. | [Check a benchmark](../audit/check-a-benchmark.md) |
| Oracle | The agent that runs a task's reference solution (`oracle/solve.sh`); a control. | [Quick start](../start/quick-start.md) |
| pass@k | The chance that at least one of k tries at a task passes. | [Compare runs](../audit/compare-runs.md) |
| pass^k | The chance that all k tries at a task pass. | [Compare runs](../audit/compare-runs.md#choose-the-metric) |
| Registry | The list of named, versioned datasets `bench run -d` resolves. | [Find a benchmark](../evaluate/find-a-benchmark.md) |
| Repeat | One pass of a job over its whole task set. `--trials 3` makes three repeats, each a job of its own in a `trial-01` to `trial-03` folder with one trial per task; the `Matrix:` line counts them as trial jobs. | [Run a benchmark](../evaluate/run-a-benchmark.md#repeat-each-task) |
| Reward | The verifier's score for a trial, 0.0 to 1.0 unless the task widens it. | [Verifiers](../tasks/verifiers.md) |
| Rollout | One agent's attempt at one task, in its own sandbox: BenchFlow's unit, which every score, vote and training row is read from. The command line and the result folders call it a trial; Python calls it `Rollout`. | [How BenchFlow works](../start/how-it-works.md) |
| Round | One exchange with a user: the agent works, a soft check runs, the user writes the next prompt or stops. | [Multi-turn tasks](../tasks/multi-turn.md#turns-and-rounds) |
| Rubric review | A reviewer agent grading a trial's work against weighted criteria. | [Rubric review](../tasks/rubric-review.md) |
| Run card | The settings summary 0.10.0rc1 prints before a run. | [Before you run an eval](../evaluate/before-you-run.md#the-run-card-010) |
| Run config | A YAML file with a whole run's settings, given to `bench run --config`. | [Configs and variables](../scale/configs.md) |
| Sandbox | The isolated machine a trial runs in. | [Sandboxes](../scale/sandboxes.md) |
| Scene | A list of turns by named roles in one sandbox; a task can run several in order. | [Multi-turn tasks](../tasks/multi-turn.md#several-agents-in-one-sandbox) |
| Scored | A trial whose test reached a verdict: its reward, or 0 when the test itself broke. | [When a trial fails](../evaluate/failed-trials.md) |
| Script agent | A shell script run as the agent (`-a script:./probe.sh`), as probes of a verifier are. | [Stop reward hacking](../tasks/reward-hacking.md#probe-your-verifier) |
| Scripted user | Python code that writes the agent's next prompt from the last round's result. | [Multi-turn tasks](../tasks/multi-turn.md) |
| Separate verifier | The verifier run in a fresh sandbox that gets only the workspace and artifacts. | [Verifiers](../tasks/verifiers.md#run-the-verifier-in-its-own-sandbox) |
| Skill | Instructions and files an agent can load for a kind of task. | [Skills](../evaluate/skills.md) |
| Soft check | The task's test run between rounds or steps, on the workspace as the agent left it; the final check is the one that counts. | [Multi-turn tasks](../tasks/multi-turn.md#turns-and-rounds) |
| Task | A folder with an instruction, an environment, a verifier and usually a reference solution. | [Task format](../tasks/task-format.md) |
| Trajectory | Everything the agent did in a trial: messages, thoughts, tool calls. | [Read the results](../evaluate/read-results.md) |
| Trial | The command line's and the result folders' name for a rollout. The `--trials` flag and the `trial-NN` folders count repeats instead (see Repeat). | [How BenchFlow works](../start/how-it-works.md) |
| Turn | One prompt to one agent. | [Multi-turn tasks](../tasks/multi-turn.md#turns-and-rounds) |
| Unscored | A trial whose agent finished but whose verifier couldn't score it. | [When a trial fails](../evaluate/failed-trials.md) |
| Verifier | The task's check, usually `verifier/test.sh`, which writes the reward. | [Verifiers](../tasks/verifiers.md) |
| Warm pool | Sandboxes prepared ahead of their trial's turn on Docker: 1 by default, 0 with `-j 1`; `--warm-pool` sets the number. | [Concurrency and budgets](../scale/cost.md#concurrency) |
| Workspace | The folder the agent works in (the task's `WORKDIR`), which the verifier grades. | [How BenchFlow works](../start/how-it-works.md) |
