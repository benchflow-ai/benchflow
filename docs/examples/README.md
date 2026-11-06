# Examples

Runnable examples for the BenchFlow docs.

```bash
bench run benchflow-ai/skillsbench/tasks/edit-pdf \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox daytona
```

Sandboxes are `docker` (the default), `remote-docker`, `daytona`, `modal`, `agentcore` and `apple-container`.

## Featured

- **Hill-climbing a skill.** An optimizer agent edits a skills folder each round. An edit stays only if the train score rises and the held-out test score does too. The optimizer only ever sees the train split. Based on [Automating eval design and hillclimbing with Claude](https://claude.dev/blog/automating-eval-design-and-hillclimbing/), with a SkillsBench recipe. See [`hillclimb/`](./hillclimb/README.md).
- **Progressive disclosure on SWE-bench Pro.** A simulated user gives the agent a short prompt first, then failing-test hints, then the full spec. Notebook: [`swebench_pro_progressive_disclosure.ipynb`](./swebench_pro_progressive_disclosure.ipynb). Guide: [Progressive disclosure](../tasks/scripted-users.md).

## Skills

To give the agent a task's own skills (`environment/skills`), add `--task-skills`:

```bash
bench run tasks/my-task \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox daytona --task-skills
```

For a skills folder from elsewhere, pass it with `--skills-dir` and `--skill-mode with-skill`. See [Architecture: skill loading](../architecture.md#skill-loading).

## Python SDK

[`python-sdk/`](./python-sdk/README.md) holds small scripts for the Python API. Start with `python-sdk/quickstart.py`, a tour in `# %%` cells.

| Script | Shows |
|---|---|
| `run-oracle.py` | one task with the oracle, and the typed result |
| `run-agent.py` | one task with a real agent, its trajectory and token usage |
| `run-many.py` | several agents and models on one task with `bf.run_batch` |
| `run-batch.py` | a folder of tasks as one `Evaluation`, with resume |
| `run-branch.py` | a run forked into two scored children with `bf.branch` |
| `run-with-manifest.py` | a task inside an environment manifest whose service BenchFlow starts |
| `compare-jobs.py` | two finished jobs compared with `bf.compare`; reads files only |

## Tasks

| Folder | What it is |
|---|---|
| [`task-md/`](./task-md/) | `task.md` packages: runnable tasks and schema-only fixtures |
| `nanofirm-task/` | a tiny task in the older `task.toml` layout, the input for `bench import`; keep it in that layout |

## Scenes and simulated users

| File | Shows |
|---|---|
| `coder-reviewer-demo.py` | a single-agent baseline, then a coder-reviewer scene, on one task folder |
| `scene-patterns.md`, `scene-patterns.py`, `scene-patterns.ipynb` | single-agent, self-review, specialist-review and client-advisor scenes |
| `user_dogfood.py` | a rule-based `FunctionUser` progressive-disclosure loop |
| `swebench_pro_user_dogfood.py` | the same loop on SWE-bench Pro-style tasks |
| `swebench_pro_progressive_disclosure.ipynb` | the SWE-bench Pro walk-through as a notebook |

## Branching and training

| File | Shows |
|---|---|
| `branch-agent-run.py` | an agent run on the hello-world task branched into two verifier-scored children, then the parent finished; see [Composed checkpoints](../jobs/branching-reference.md) |
| [GRPO recipe](../training/grpo-recipe.md) | a GRPO training workflow with TRL: task snapshots, a baseline eval, training, a final eval and the lift |

## Next steps

- Run your first task: [Quick start](../getting-started/quick-start.md).
- Several agents in one task: [Scenes and simulated users](../tasks/scenes.md).
