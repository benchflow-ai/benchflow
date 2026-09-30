# What's new in 0.8

For users upgrading from 0.7. It picks out what changes how you work; the [CHANGELOG](../CHANGELOG.md) has every entry, with the reasons and the evidence.

```bash
uv tool install --python 3.12 --upgrade benchflow
bench --version
bench doctor
```

## What to do differently

- **Use current Claude model ids.** The sandbox now installs `claude-agent-acp` 0.81.2 with its own Claude Code 2.1.280 (#1137). Its model option accepts `claude-haiku-4-5-20251001` (or `claude-haiku-4-5`), `claude-sonnet-5` and `claude-opus-5-5`, and the aliases `haiku`, `sonnet` and `opus`, and refuses older ids such as `claude-sonnet-4-6`. Update the `--model` of Claude Code runs and the `model:` of `claude-agent-acp` roles.
- **Check the agent names you type.** A one-word agent name that neither BenchFlow nor the agents catalog knows now fails before any sandbox starts; it used to run as a command. OpenClaw now loads from the agents repository, and the catalog is pinned to a commit per release: set `BENCHFLOW_AGENTS_SOURCE=benchflow-ai/agents@main` to follow `main`, or `BENCHFLOW_AGENTS_DIR` to use a local checkout.
- **Expect the task's own agent timeout from every `bf.run` form.** `bf.run(bf.Agent(...), bf.Environment...)` applied 900 s over the task's `timeout_sec`; it now uses the task's value and warns when they differ. Pass `RuntimeConfig(timeout=900)` to keep the old limit.
- **Check `RolloutResult`, not `RuntimeResult`.** The `Agent + Environment` form of `bf.run` returns `RolloutResult` like every other form. `RuntimeResult` is deprecated, so code that checks `isinstance(result, RuntimeResult)` must check `RolloutResult`.
- **Read null as unscored.** An unscored trial's `results.jsonl` row now has reward `null`, not `0.0`, and results awaiting assessment stay unscored everywhere.
- **Pass `--codex-apps-policy inherit` if scored Codex tasks need your account's Apps.** They are now off by default for scored `codex-acp` tasks, and scored Codex runs as root or with `--sandbox-user null` are refused.
- **Install verifier pytest plugins where the plugin guard trusts them.** Hardened verifiers refuse pytest plugins backed by code the agent could write. Install them with `uvx` into uv's default cache, or into the image; `bench tasks check` warns when a `test.sh` installs plugins into the workspace or `/tmp`.
- **Fix typos in `task.toml`.** A legacy or Harbor `task.toml` with an unknown key now runs with a warning, but a likely typo of a known key, and any unknown key in `[sandbox]`, `[agent]`, `[verifier]` or `[steps]`, is refused before the run. `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` restores the old leniency for corpora you do not control.
- **Replace tasks that rely on symlinks.** Sandbox uploads skip symlinks, so `bench tasks check` now reports such a task and a run skips it with a warning that names it.
- **Import from defining modules.** 129 unused re-exports were removed from `benchflow.task.verifier`, `benchflow._utils.task_authoring`, `benchflow.task.acceptance_live` and `benchflow.rollout`.
- **Expect smaller downloads from `--source-repo`.** With `--source-path`, only that folder is downloaded (a sparse clone), so one SkillsBench task is a few megabytes instead of the 1.1 GB repository plus a 644 MB snapshot. Caches from 0.7 are reused as they are; delete `.cache/datasets/` to reclaim their space.

## New in 0.8

- **First run:** `bench doctor` checks the machine and each agent's login and prints a fix per problem; `bench eval smoke` runs a bundled hello-world task once per logged-in agent; `bench --help` and `bench tasks init` end by naming the command that comes next. See [Getting started](./getting-started.md).
- **Controls:** `--agent nop` runs nothing, so a sound verifier scores the untouched workspace 0; with `--agent oracle` it brackets a task before any model sees it.
- **CI:** `--fresh`, `--job-name`, `--fail-under`, `--fail-on` and `--summary-out` on `bench eval run`, and `bench eval resume JOB_DIR`. SIGTERM cancels a run, deletes its sandboxes and exits 143.
- **Budgets:** `--max-cost-usd`, `--max-sandbox-seconds` and `--max-tokens` (`bf.Budget` in Python) stop a job at a cap. See [Budgets](./reference/budget.md).
- **Reading results:** `bf.load_job` and `bf.load_trial`, `bench eval inspect` and `bench eval compare`, pass@k and pass^k, and versioned JSON exports. See [Analysing runs](./analysing-runs.md).
- **Python:** `bf.run_sync`, `bf.arun`, `bf.run_batch`, `Evaluation.stream()` and `Evaluation.resume(job_dir)`. See the [examples gallery](./examples/python-sdk/README.md).
- **Branching:** `bench eval branch` and `bf.branch` fork a run at a checkpoint into verifier-scored children, in parallel or nested, with cost per child. See [Branching](./branching.md).
- **Sandboxes:** `--sandbox remote-docker` runs tasks on a Docker host you control ([Remote Docker](./remote-docker.md)), and tasks can score in a separate verifier sandbox that shares nothing with the agent's ([Separate verifier sandboxes](./separate-verifier.md)).
- **Training:** `bench train stream` emits rollouts to a trainer while a job runs, `bench train token-coverage` checks gateway token capture, and `sglang/...` models capture token ids and logprobs.
- **Agents:** the Google Antigravity CLI (`antigravity`, alias `agy`).
- **Authoring:** `bench tasks init` writes a canary GUID into each new task, and `bench tasks check` warns when a task has none.
- **Concurrent jobs** on one Docker daemon no longer delete each other's containers and networks.

## Deprecated

Each of these still works and warns when used:

- `benchflow.SDK`: use `bf.run(bf.RolloutConfig(...))`, which takes the same keyword arguments.
- `bf.snapshot`, `bf.restore` and `bf.list_snapshots`: use `bf.workspace_snapshot`, `bf.workspace_restore` and `bf.list_workspace_snapshots`.
- `RuntimeResult`: use `RolloutResult`.
- The unused `RuntimeConfig.max_rounds`, `snapshot_policy` and `reward_stream` fields.
- `bench eval create`: use `bench eval run`.
