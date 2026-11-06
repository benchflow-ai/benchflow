# BenchFlow as an RL environment

For reinforcement learning, a task is an environment and a trial is a rollout: BenchFlow starts the sandbox, runs the policy as an agent, scores the result with the task's verifier, and writes everything the trainer needs.

## The loop

1. Your policy is served as a model the agent calls ([A model you serve](../evaluate/agents.md#a-model-you-serve)).
2. `bench run` runs a group of rollouts per task and scores each one.
3. A trainer reads the rollouts with their rewards and group advantages, from the stream while the job runs or as rows afterwards ([Connect a trainer](./trainer.md)).
4. The trainer updates the policy, and the next group runs against the new one.

BenchFlow does steps 2 and 3. For training on tokens rather than text, turn token capture on from the first run: [Token ids and logprobs](./trainer.md#token-ids-and-logprobs).

## Run a group of rollouts

`--trials N` gives each task N rollouts. The repeats run one after another, each a job of its own; `-j` sets how many trials run at once within a job, so it speeds up a run of many tasks, not a group of one. Four rollouts of OpenCode on DeepSeek V4 Flash, on the half-life task with the [partial-credit test](../tasks/verifiers.md#partial-credit-and-tolerances), so a reward can land between 0 and 1:

```bash
bench run tasks/half-life \
  -a opencode -m deepseek/deepseek-v4-flash --trials 4 -j 2 \
  --jobs-dir jobs/group
```

The last line each rollout's test printed:

```text
trial-01  reward 1.0: close_enough pass, plausible_uncertainty pass, consistent pass
trial-02  reward 1.0: close_enough pass, plausible_uncertainty pass, consistent pass
trial-03  reward 1.0: close_enough pass, plausible_uncertainty pass, consistent pass
trial-04  reward 1.0: close_enough pass, plausible_uncertainty pass, consistent pass
```

Each rollout cost about $0.01 (190,000 to 270,000 tokens) and took 2 to 5 minutes.

## Rewards and advantages

`bench train convert` writes one row per rollout, with its group's advantage when you ask for one. Leave `job` out of `--group-by`, since each of the four tries is a job of its own:

```bash
bench train convert jobs/group/opencode --out train/group.jsonl --group-advantage grpo --group-by task,agent,model
```

Here each group holds one policy and one version of the task, because the folder holds one run of one task. The key `task,agent,model` doesn't tell versions apart: add `task_digest` to it when a folder may hold more than one version of a task, and keep each policy version's rollouts in jobs of their own, since the model name can stay the same while the policy changes.

```text
Converted 4 row(s) from 4 rollout(s) -> train/group.jsonl
Groups (grpo, by task,agent,model): 1; 0 unscored rollout(s) left out of every baseline.
```

Each row's `group` says how its advantage was made:

```json
{"id": "task=half-life|agent=opencode|model=deepseek/deepseek-v4-flash", "normalisation": "grpo",
 "formula": "advantage = (reward - mean) / (std + eps); mean and sample std (ddof 1) over the scored rollouts of the group",
 "rollouts": 4, "scored": 4, "mean": 1.0, "std": 0.0, "eps": 0.0001}
```

Every advantage here is 0: all four rollouts scored 1.0, so the group says nothing about what to change. This task is too easy for this policy to train on. A group helps only when its rewards differ: some rollouts pass and others fail, or partial credit spreads the rewards out. The crystallographic SkillsBench task did that for Claude Code with skills ([Skills](../evaluate/skills.md)):

```bash
bench train stream jobs/skills/claude-agent-acp --no-follow --group-size 3 --group-by task,agent,model
```

The `rollout_path`, `reward` and `advantage` of that task's three lines:

```text
trial-01/.../crystallographic-wyckoff-position-analysis__c6049d70   reward 0.0  advantage -0.577
trial-02/.../crystallographic-wyckoff-position-analysis__2168c849   reward 1.0  advantage  1.155
trial-03/.../crystallographic-wyckoff-position-analysis__6b313a34   reward 0.0  advantage -0.577
```

`--group-advantage loo` gives each rollout its reward minus the mean of the others instead.

## Check the rows before you train

Two things in 0.9.1 and 0.10.0rc1 can leave the rows thinner than the rollouts:

- **Rollouts that ended with an error are left out.** Converting the crystallographic group above writes the passing rollout, not the two that hit their time limit, though both were scored 0. `--manifest stats.json` says so: `"skipped_terminal_error": 2`. The advantages still count them.
- **A Claude Code rollout loses its tool calls.** `bench train validate` on the OpenCode group says `{"ok": true, "rows": 4, "rows_with_tool_calls": 4}`; on Claude Code rollouts it says `rows_with_tool_calls: 0`. Add `--require-tool-calls` to make that a failure.

[Export training data](./export.md) shows a whole row; [Known pitfalls](../audit/pitfalls.md#infrastructure-that-fails-quietly) lists both problems.

## From Python

`bf.Evaluation` runs a job from Python, with the flags as arguments ([Run a benchmark](../evaluate/run-a-benchmark.md#from-python)), and `bf.load_job` reads each rollout's reward back ([Read the results](../evaluate/read-results.md#in-python)).
