# Configs

A run config file holds a whole run, so it can be repeated and version-controlled.

```bash
bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml --model gemini/gemini-3.5-flash --dry-run
```

```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
  ref: main
agent: claude-agent-acp
model: claude-sonnet-5
environment: daytona
concurrency: 8
max_retries: 1
```

## How a config combines with flags

- A flag on the command line wins over the file's key, and the key wins over the default.
- A key the loader does not read stops the run before anything starts. The error names the file, the line and the closest key: `run.yaml:4: concurrenzy: unknown key (did you mean 'concurrency'?)`.
- `sandbox:` is accepted for `environment:`, as the flag is `--sandbox`. Writing both is an error.
- A config with no `agent:` runs `claude-agent-acp` and prints a warning. `bench run` without a config runs the oracle.
- `--dry-run` prints the plan the config makes, without running it.

From Python, `Evaluation.from_yaml(path)` loads a config and `evaluation.to_yaml(path)` writes one. No CLI command writes a config yet.

## Keys

The file needs `source:` or `tasks_dir:`. Every other key is optional.

### Tasks

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` | mapping | none | A GitHub folder: `repo` (`org/repo`), `path` (optional) and `ref` (optional branch, tag or commit) |
| `tasks_dir` | path | none | A local task folder, or a folder of tasks |
| `include` | list | all | Task names or globs to run (`hard-*`) |
| `exclude` | list | none | Task names or globs to skip. Exclude wins over include |
| `n_tasks` | int | all | Run at most N tasks: the first N by name, or a seeded sample |
| `sample_seed` | int | none | The seed for the `n_tasks` sample |
| `seeds` | list or `0-4` | none | Run every task once per seed (tasks that write seeded variants, such as embodied tasks) |

### Agent

| Key | Type | Default | Meaning |
|---|---|---|---|
| `agent` | str | `claude-agent-acp` | An agent `bench agents` lists, `oracle` or `nop` |
| `model` | str | the agent's default | The model, in the agent's own naming |
| `reasoning_effort` | str | none | `low`, `medium`, `high`, `max`, ... when the agent has one |
| `harness` | str | `acp` | `acp`, or `native` (the agent's own CLI in headless JSON mode; `claude-agent-acp` and `codex-acp`) |
| `agent_env` | mapping | none | Variables for the agent process. See [Environment variables](../jobs/environment-variables.md) |
| `agent_env_keys` | list | none | Variable names the run needs without their values. A warning names any you did not set |
| `prompts` | list | the task's prompt | Prompts sent in order; `null` is the task's own prompt |
| `extra_instruction` | str | none | Text appended to every task's prompt |
| `loop_strategy` | str | `single-shot` | `verify-retry:k=3` or `self-review:k=3` |
| `codex_apps_policy` | str | `disabled` (`inherit` when verification is skipped) | `disabled` or `inherit`: whether Codex may use your account's Apps |
| `skill_mode` | str | `no-skill` | `no-skill`, `with-skill` or `self-gen` |
| `skills_dir` | path | none | Skills to install for the agent |
| `skill_creator_dir` | path | none | The skill-creator skill, for `self-gen` |
| `self_gen_no_internet` | bool | `false` | No web tools in a `self-gen` run |
| `skip_install` | bool | `false` | Do not install the agent: the image already has it |
| `usage_tracking` | str | `auto` | `auto`, `required` (a trial with no token usage fails) or `off` |

### Sandbox

| Key | Type | Default | Meaning |
|---|---|---|---|
| `environment` (or `sandbox`) | str | `docker` | `docker`, `remote-docker`, `daytona`, `modal`, `apple-container` or `agentcore` |
| `sandbox_user` | str | `agent` | The user the agent runs as; `null` for root |
| `sandbox_locked_paths` | list | the defaults | Paths locked away from the agent, added to `/oracle`, `/solution`, `/verifier`, `/tests` and `/testbed_verify`. `[]` locks nothing |
| `sandbox_setup_timeout` | int | `120` | Seconds to set up the sandbox user |
| `environment_manifest` | path, `name@version` or mapping | none | Services the agent works with. See [Environment manifests](../environment-plane.md) |
| `config_override` | mapping | none | Merged into every task's config for this run, for example `{agent: {timeout_sec: 120}}` |
| `base_image_override` | str | none | Rewrites the task Dockerfile's `FROM`. Only for reproducing old runs whose base image moved |
| `context_root` | path | none | Where Dockerfile `COPY` sources live, for tasks inside a monorepo |
| `fresh_install` | bool | `false` | Install the proxy and agent in every trial, with no [cached layer](../jobs/cache.md) |
| `build_concurrency` | int | `concurrency` | Docker image builds at a time |

### Verifier and integrity

| Key | Type | Default | Meaning |
|---|---|---|---|
| `verifier_sandbox` | str | `task` | `task` (each task's own choice) or `separate` (every verifier in a fresh sandbox) |
| `separate_verifier` | bool | `false` | An older spelling of `verifier_sandbox: separate` |
| `separate_verifier_include` | list | none | Paths outside the workspace that a separate verifier sandbox also gets |
| `integrity` | str | `off` | `off`, `audit` or `strict`. See [Integrity](../integrity.md) |
| `freeze_workspace` | bool | `false` | Save each workspace so [`bench regrade`](../jobs/regrade.md) can score it again (the flag is `--save-outputs`) |
| `reviewer` | mapping | none | The rubric reviewer: `agent` (`opencode`), `model`, `reasoning_effort`, `environment` (`docker`), `timeout_sec`, `idle_timeout_sec`, `concurrency` (4), `image`, `agent_env`, `open_network`. See [Rubric review](../judging/rubric-review.md) |

### Concurrency, time and retries

| Key | Type | Default | Meaning |
|---|---|---|---|
| `concurrency` | int | `4` | Trials at a time |
| `agent_idle_timeout` (or `agent_idle_timeout_sec`) | int | `600` | Seconds of silence before a prompt is stopped; `0` or `none` turns it off |
| `timeout_multiplier` | float | `1` | Scales every task's agent timeout |
| `max_retries` | int | `2` | Retries per trial after a retryable error. See [When a run fails](../jobs/when-a-run-fails.md#does-it-count) |
| `retry` | mapping | none | The full retry policy (below). It wins over `max_retries` |
| `budget` | mapping | none | Job caps: `max_cost_usd`, `max_sandbox_seconds`, `max_tokens`, `max_rollouts`. See [Budgets](../reference/budget.md) |

`retry:` takes these keys:

| Key | Default | Meaning |
|---|---|---|
| `max_retries` | `2` | Retries per trial |
| `exclude_categories` | `timeout`, `provider_auth`, `provider_rate_limit`, `provider_rejected` | Categories never retried |
| `retry_on_install`, `retry_on_pipe`, `retry_on_acp`, `retry_on_idle_timeout`, `retry_on_infra`, `retry_on_api_error`, `retry_on_verifier_infra` | `true` | Turn one retried category off |
| `wait_multiplier`, `min_wait_sec`, `max_wait_sec` | `2.0`, `1`, `30` | The backoff between retries |

### Checkpoints

| Key | Type | Default | Meaning |
|---|---|---|---|
| `checkpoints` | str | none | `every-prompt` or `prompt:N[,M]`: keep a sandbox snapshot after those prompts. See [Branching](../jobs/branching.md) |
| `checkpoint_keep` | int | `3` | Snapshots kept per trial; the oldest is deleted |
| `retry_from_checkpoint` | str | none | `on-failure`, `on-timeout` or both: retry a failed trial from its last snapshot |
| `retry_prompt` | str | the remaining prompts | The retry's prompt; `@instruction` and `@verifier_feedback` are filled in |
| `retry_resume_session` | bool | `false` | The retry continues the failed conversation |

### Job

| Key | Type | Default | Meaning |
|---|---|---|---|
| `jobs_dir` | path | `jobs` | Where job folders go |
| `job_mode` | str | `parallel-independent` | Or `sequential-shared` (below) |

### Written by BenchFlow

These keys appear in the configs BenchFlow records (`evaluation.json`, `to_yaml`). Leave them as they are:

| Key | Meaning |
|---|---|
| `dataset_name`, `dataset_version`, `dataset_task_digests` | The dataset version a `-d` run resolved |
| `source_provenance` | Where `source:` was fetched from |
| `usage_proxy`, `usage_proxy_url`, `usage_proxy_port`, `usage_proxy_bind_host`, `usage_proxy_advertised_base_url` | Refused: the run stops with `<key> is no longer supported; use usage_tracking=auto\|required\|off instead`. Remove the key, or set `usage_tracking` |

## Learning across trials (`job_mode: sequential-shared`)

By default, trials run at the same time and share nothing. `job_mode: sequential-shared` runs them one at a time over one shared store of memory and skills:

```yaml
source:
  repo: benchflow-ai/skillsbench
  path: tasks
agent: claude-agent-acp
model: claude-haiku-4-5
job_mode: sequential-shared
```

After each scored trial, an improvement saves a new version of the store. A regression reverts it to the best version so far. `concurrency` is ignored in this mode. See [Architecture](../architecture.md), capability 5.

## YAML job configs

A YAML job config also runs: a file with `agents:` and `datasets:` lists, or a `tasks:` list, including job `config.json` files. It runs all of it, or nothing: a setting BenchFlow cannot honour exactly is refused before anything starts, by file, line and key, with what to do instead. Default-valued blocks are accepted, so fully expanded configs load too.

```bash
bench run --config <job>/config.json
```

**What runs.** Every agent with every dataset, each `n_attempts` times:

- one agent, one dataset and `n_attempts: 1` is one job, in `<jobs_dir>/<job>/` as for any run config;
- anything more is a matrix, as `--matrix --trials` runs one ([pass@k](../reference/pass-at-k.md)). Each agent and dataset pair is a cell, named `<agent>__<model>__<dataset>` (`<agent>__<dataset>` for an agent without a model, `adhoc` for a `tasks:` list); a name two cells would share gets the next free `-2`, `-3`, so every cell has its own folder. Each attempt is a job in `<jobs_dir>/<cell>/trial-NN/`. `<jobs_dir>/matrix-summary.json` holds every trial job, the agent and model each cell ran, and pass@k per cell, controls (oracle, nop) included;
- `n_attempts: 3` is three trials of every task, scored one by one and as pass@k at k = 1, 2 and 3 (the usual ks up to `n_attempts`, and `n_attempts` itself). It is never one trial with retries: a YAML job config never retries a trial;
- the cells run one after another, each with up to `n_concurrent_trials` trials at once;
- a cell that selects no task (after `--include`/`--exclude` or `task_names`) stops the run before anything starts, naming the cell;
- a stopped matrix (Ctrl-C, SIGTERM) says to run the same command again: finished trial jobs are kept, and the rest run;
- `bench run --config <file> --dry-run` prints every cell, its tasks and its trials.

From Python, `Evaluation.from_yaml` loads one job: a file of several is refused unless `job_cell=<n>` picks one of `benchflow.job_config.job_plan(path).cells`.

| Job key | What BenchFlow does |
|---|---|
| `agents` | One cell per entry. An entry takes `name`, `model_name`, `env` (a mapping, for the agent only, with `${VAR}` and `${VAR:-default}` filled from your shell) `n_concurrent` (a lower cap for that cell, at most `n_concurrent_trials`) and `concurrency_group` (with `n_concurrent`, the same in every agent of the group; two entries of the same agent with no group must use the same `n_concurrent` because they share a concurrency pool), plus BenchFlow's `reasoning_effort`, `harness` and `skip_install`. No `agents:` (or `[]`) runs the oracle, with a note. With several agents, `--agent` and `--model` are refused |
| `datasets` | One cell per entry: `path`, with `task_names` and `exclude_task_names` (names or globs) and `n_tasks` (the first N after them). An entry that names a remote dataset (`name`, `version`, `repo`, a registry) is refused: run it with `--dataset <name>@<version>`, or clone it and give its folder as `path` |
| `tasks` | One cell: one task folder, or tasks in one folder, matched by exact name. Refused with `datasets`, with `include`/`exclude`, or when an entry is pinned to a git commit |
| `n_attempts` | Trials per task (above) |
| `environment` | `type` is the sandbox |
| `n_concurrent_trials` | Concurrency. It wins over `orchestrator.n_concurrent_trials` |
| `agent_timeout_multiplier`, `verifier_timeout_multiplier`, `environment_build_timeout_multiplier` | Scale the task's agent, verifier and environment build timeouts. The scaled build timeout (at any multiplier, 1 included) bounds the whole environment start: building or pulling the image and starting the sandbox. A start past it stops the trial with a sandbox startup error |
| `timeout_multiplier` | Scales each of those three when its own multiplier is not set. BenchFlow installs agents with a separate time budget, so a value other than 1 is refused unless `agent_setup_timeout_multiplier: 1` is set |
| `extra_instruction_paths`, `extra_instructions` | Appended to every prompt: each file, then each line, separated by blank lines. A missing file is refused |
| `metrics` | `[{type: mean}]`, the default. BenchFlow scores the mean reward and pass@k only, so any other metric is refused |
| `quiet` | Hides the live progress, as `--quiet` |
| `job_name` | Each job's folder name, unless a folder of that name exists and is not a BenchFlow job |
| `jobs_dir` | Where the jobs go; `--jobs-dir` wins |

These are refused when set to anything but their accepted default: `install_only`, `debug`, `retry` (and `orchestrator.retry`), `verifier`, `user_agent`, `artifacts`, `source_jobs`, `agent_setup_timeout_multiplier` (other than 1), `environment.env` (this would affect every sandbox command, including the verifier; move it to each agent's `env`, or pass `--agent-env`), every other `environment` key, and the agent keys `import_path`, `skills`, `override_timeout_sec`, `override_setup_timeout_sec`, `max_timeout_sec`, `resume_trajectory`, `load_trajectory`, `extra_allowed_hosts`, `include_logs`, `exclude_logs`, `kwargs` and `mcp_servers`.

A YAML job config also takes these BenchFlow keys, for every cell: `jobs_dir`, `include`, `exclude`, `reasoning_effort`, `harness`, `codex_apps_policy`, `agent_idle_timeout`, `sandbox_user`, `sandbox_locked_paths`, `sandbox_setup_timeout`, `base_image_override`, `context_root`, `fresh_install`, `skill_mode`, `skills_dir`, `skill_creator_dir`, `self_gen_no_internet`, `separate_verifier`, `separate_verifier_include`, `verifier_sandbox`, `integrity`, `reviewer` and `usage_tracking`. Any other key is refused with a did-you-mean.

`--sandbox-setup-timeout` and `--base-image-override` each override the file only when supplied. The setup timeout defaults to 120 seconds for a folder run. JSON files are parsed as JSON, including scientific notation in positive, finite timeout multipliers. A CLI `--config-override` can supply timeout multipliers for phases the file did not explicitly set; different phases merge, and setting the same phase in both is refused. The implicit build multiplier of 1 still bounds the whole sandbox start when only a verifier multiplier is supplied on the command line.

## Next steps

- Run it: [Run a job](../jobs/run.md).
- Variables the config passes: [Environment variables](../jobs/environment-variables.md).
- Every flag: `bench run --help-all`, or the [CLI reference](../reference/cli.md).
