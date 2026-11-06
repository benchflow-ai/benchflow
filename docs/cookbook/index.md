# Cookbook

Recipes built on BenchFlow. Each says what has been measured; check its limits before you spend on it. Each one lives in the repository under `docs/examples/`, so clone BenchFlow and run it from the repository root.

```bash
git clone https://github.com/benchflow-ai/benchflow
cd benchflow
```

| Recipe | What it does | Start with |
|---|---|---|
| [Hill-climb a skill](../examples/hillclimb/) | An optimizer agent edits a skill, round by round; a change is kept only if a held-out test split improves | `cd docs/examples/hillclimb && ./run.sh smoke` |
| [Train with TRL GRPO](../training/grpo-recipe.md) | Train `Qwen/Qwen3-4B` on BenchFlow tasks, then measure the lift on held-out tasks | `bench tasks snapshot-hf ...` (step 2) |
| [Multi-role scenes: four patterns](./scene-patterns.md) | One task run as one agent, self-review, two specialists, or client and advisor | `bench check docs/examples/task-md/multi-scene --level schema` |
| [Simulated user](https://github.com/benchflow-ai/benchflow/blob/main/docs/examples/task-md/user-runtime/private-facts-nudges/task.md) | A task with scenes, a scripted user with private facts, and a verifier | `bench run docs/examples/task-md/user-runtime/private-facts-nudges` |
| [Python SDK scripts](../examples/python-sdk/) | One rollout, a batch, a branch, a comparison of two jobs, from Python | `uv run python docs/examples/python-sdk/run-oracle.py` |
| [A small task in the older layout](https://github.com/benchflow-ai/benchflow/blob/main/docs/examples/nanofirm-task/instruction.md) | A contract-review task in the `task.toml` layout, the input for `bench import` | `cp -r docs/examples/nanofirm-task /tmp/nanofirm && bench import /tmp/nanofirm` |

With no `--agent`, `bench run` runs the task's oracle, so the simulated-user command needs no login. `bench import` writes `task.md` next to the old files, so import a copy and leave the example as it is.

## Beyond evals: optimizing agents

Tasks that grade an agent can also train or tune one. Two recipes here do that:

- [Hill-climb a skill](../examples/hillclimb/) tunes a prompt-level artifact, a skills folder, and guards against overfitting with a held-out split.
- [Train with TRL GRPO](../training/grpo-recipe.md) updates model weights, with BenchFlow's sandboxes and verifiers as the environment and the reward.

## Next steps

- [Create a task](../tasks/create-a-task.md): write your own task to run these recipes on.
- [Run a job](../jobs/run.md): every way to start a run.
