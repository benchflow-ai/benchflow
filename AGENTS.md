# benchflow

Multi-turn agent benchmarking with ACP. Docs live in [`docs/`](./docs/).

## Setup + test

```bash
uv sync --extra dev --locked
uv run python -m pytest tests/
uv run ty check src/
uv run ruff check .
```

## Conventions

- **Don't rewrite passing tests** to match new behavior. Update for shape changes, not for semantic changes you don't understand. No tautological tests.
- **Regression tests must name the PR/commit they guard** in the docstring (e.g. `Guards the fix from PR #198 against the regression introduced by PR #193`).
- **Human review before `main`.** PRs only. No force-pushes to `main`. Self-approval doesn't count.
- **Trunk-based:** branch off `main`, PR back to `main`. No long-lived release branches.
- **Releases:** bump `pyproject.toml` to the stable version, tag `v<version>` on main, push tag (CI publishes to PyPI), then bump main to the next `.dev0`.

## Experiment guidance (when using benchflow to run batch tasks experiments)

- When running tasks experiments on Daytona, keep in mind that Daytona supports up to 100 parallel VMs.
- When running tasks experiments by default use the highest inference level for all the model calls (For example, xhigh for GPT5.5, or MAX for Opus4.8).
- When running tasks experiments, for each API key that available to you, verify 1. whether the key is still working by sending basic LLM calls; 2. ask the user how many parallel tasks can each key support / figure it out by yourself. Thus you can use highest available concurrency when running experiments.
- Use Huggingface as the ground truth of tasks traj database. The target is making sure number_of_tasks * 2 (with/without skills) * number_of_trials (usually 3 / 5) for each model + each harness is complete. Each traj on Huggingface must be "healthy" - "healthy" means 1. the trial has complete traj information (agent pass / fail / timeout the task); 2. the trial has complete meta information (for example: token usage; timing; etc.). For each traj, we should use subagents to audit 1. whether the fail / timeout is due to agent capability instead of task config or env setting; 2. there is no reward hacking behavior; 3. in "no-skill" trials the agents do not have access to any skill files or accessed any skill files in traj.