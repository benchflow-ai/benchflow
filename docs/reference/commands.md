# Commands

Each `bench` command and flag has its type, default and help here; `benchflow` is the same command as `bench`. A command or flag new since 0.9.2 has `(0.10)` after its name when 0.10.0rc3 has it, or `(unreleased)` when no release has it yet; the options `--help` leaves out have a table of their own under their command, and [the commands `bench --help` hides](#commands-hidden-from-bench---help) come last. Which options `--help` leaves out follows 0.10.0rc3: 0.9.2's `bench run --help` also leaves out `--trials`, `--max-cost-usd`, `--max-tokens` and `--integrity`, which its `--help-all` lists. To see the commands at work, start with the [Quick start](../start/quick-start.md) and [Run a benchmark](../evaluate/run-a-benchmark.md); a run config's keys are in [Files BenchFlow reads and writes](./files.md), and [CLI and Python](./cli-python.md) gives each flag's Python argument.

<!-- This page is generated from the code by tools/docs_reference.py. To change it, change the command's help text, then run `uv run python tools/docs_reference.py --write`. -->

| Command | What it does |
|---|---|
| [`bench new`](#bench-new) | Scaffold a new benchmark task. |
| [`bench check`](#bench-check) | Validate task structure and the run's plan without starting a sandbox. |
| [`bench import`](#bench-import) | Convert task.toml + instruction.md into the unified task.md format. |
| [`bench run`](#bench-run) | Run an agent on tasks and score them; without -a, the oracle. |
| [`bench branch`](#bench-branch) | Fork an agent run at a checkpoint into labelled children. |
| [`bench results`](#bench-results) | Summarize jobs; with two, compare them. |
| [`bench view`](#bench-view) | View a trial or session trajectory in the browser. |
| [`bench regrade`](#bench-regrade) | Re-run the task's current verifier on each trial's frozen workspace. |
| [`bench export`](#bench-export) | Export a task to a compatibility split layout with a loss report. |
| [`bench doctor`](#bench-doctor) | Check this machine can run evals: Python/uv, sandbox, agent credentials, network. |
| [`bench agents`](#bench-agents) | List the agents, or show one in detail. |
| [`bench explain`](#bench-explain) | Explain why a trial has no score: what happened, whose problem, what next. |
| [`bench cache`](#bench-cache) | List or clean what BenchFlow keeps: cached layers, build records, snapshots. |
| [`bench cache list`](#bench-cache-list) | List what BenchFlow keeps, with sizes and creation times (the default). |
| [`bench cache clean`](#bench-cache-clean) | Print what would be removed; remove it with --remove. |
| [`bench eval`](#bench-eval) | Run agents on tasks (smoke, run) and read the results (metrics, view, inspect, compare). |
| [`bench eval smoke`](#bench-eval-smoke) | Run a bundled hello-world task once per credentialed agent, one at a time. |
| [`bench eval run`](#bench-eval-run) | Run an evaluation; single task or batch. |
| [`bench eval adopt`](#bench-eval-adopt) | Adopt an external benchmark into benchflow. |
| [`bench eval compare-lift`](#bench-eval-compare-lift) | Compare paired baseline vs trained eval job directories. |
| [`bench eval score`](#bench-eval-score) | Retry a failed reviewer using the saved workspace and trajectory. |
| [`bench eval regrade`](#bench-eval-regrade) | Re-run the task's current verifier on each trial's frozen workspace. |
| [`bench eval list`](#bench-eval-list) | List the jobs in a jobs folder, one row each, with their scores. |
| [`bench eval metrics`](#bench-eval-metrics) | Collect and display metrics from a jobs directory. |
| [`bench eval view`](#bench-eval-view) | View a trial or session trajectory in the browser. |
| [`bench eval continue`](#bench-eval-continue) | Resume a previous unfinished (timed-out) OpenHands run to completion. |
| [`bench eval branch`](#bench-eval-branch) | Fork an agent run at a checkpoint into labelled children. |
| [`bench eval inspect`](#bench-eval-inspect) | Read a finished job or trial: denominators, one row per trial, or --json. |
| [`bench eval compare`](#bench-eval-compare) | Pair two jobs by task: rewards, deltas, fair denominators, setting checks. |
| [`bench eval resume`](#bench-eval-resume) | Finish an interrupted job from its folder: rerun only the unfinished tasks. |
| [`bench eval branches`](#bench-eval-branches) | Show every fork and child of a job's (or one trial's) branches: request, status, reward, advantage, tokens, USD and sandbox-seconds. |
| [`bench tasks`](#bench-tasks) | Task authoring commands. |
| [`bench tasks generate`](#bench-tasks-generate) | Generate benchmark tasks from agent traces. |
| [`bench tasks list-sources`](#bench-tasks-list-sources) | List known HuggingFace trace datasets. |
| [`bench tasks init`](#bench-tasks-init) | Scaffold a new benchmark task. |
| [`bench tasks check`](#bench-tasks-check) | Validate task structure and the run's plan without starting a sandbox. |
| [`bench tasks overlap`](#bench-tasks-overlap) | Compare two task manifests for exact task-id and digest overlap. |
| [`bench tasks migrate`](#bench-tasks-migrate) | Convert task.toml + instruction.md into the unified task.md format. |
| [`bench tasks normalize`](#bench-tasks-normalize) | Expand minimal task.md authoring profiles into canonical task.md. |
| [`bench tasks export`](#bench-tasks-export) | Export a task to a compatibility split layout with a loss report. |
| [`bench tasks snapshot-hf`](#bench-tasks-snapshot-hf) | Materialize a Hugging Face dataset task tree with provenance. |
| [`bench tasks digest`](#bench-tasks-digest) | Compute the content digest that pins a task's files, independent of git. |
| [`bench agent`](#bench-agent) | Agent management commands. |
| [`bench agent list`](#bench-agent-list) | List all registered agents. |
| [`bench agent show`](#bench-agent-show) | Show details for a registered agent. |
| [`bench review`](#bench-review) | Grade finished rollouts against a rubric with a reviewer agent. |
| [`bench assess`](#bench-assess-010) (0.10) | Blind pairwise human assessments of saved outputs, kept in an opt-in sidecar. |
| [`bench assess attempt-ref`](#bench-assess-attempt-ref-010) (0.10) | Hash a completed rollout into a saved-attempt reference (JSON). |
| [`bench assess create`](#bench-assess-create-010) (0.10) | Start an assessment of two saved attempts of one task. |
| [`bench assess release`](#bench-assess-release-010) (0.10) | Record that every runtime of both attempts was observed released. |
| [`bench assess assign`](#bench-assess-assign-010) (0.10) | Bind the collector's persisted blind assignment (after the release). |
| [`bench assess import-vote`](#bench-assess-import-vote-010) (0.10) | Bind the collector's vote on the assignment (a skip becomes abstain). |
| [`bench assess correct`](#bench-assess-correct-010) (0.10) | Replace the effective answer with a corrected one. |
| [`bench assess retract`](#bench-assess-retract-010) (0.10) | Retract the effective answer; its history stays in the log. |
| [`bench assess reduce`](#bench-assess-reduce-010) (0.10) | Record the deterministic label the log calls for, and refresh the label file. |
| [`bench assess consume`](#bench-assess-consume-010) (0.10) | Record that a consumer used one exact ready label revision. |
| [`bench assess withdraw`](#bench-assess-withdraw-010) (0.10) | Bar a corrected or retracted label revision from future use. |
| [`bench assess show`](#bench-assess-show-010) (0.10) | Verify the sidecar's log and print its state. |
| [`bench assess verify`](#bench-assess-verify-010) (0.10) | Verify the hash chain, the label file and, optionally, the evidence. |
| [`bench assess label`](#bench-assess-label-010) (0.10) | Print the label rebuilt from the sidecar's log alone (JSON). |
| [`bench preferences`](#bench-preferences-010) (0.10) | Human preferences and rubric scores on finished trials. |
| [`bench preferences pairs`](#bench-preferences-pairs-010) (0.10) | Make the store and the blind assignment. |
| [`bench preferences import`](#bench-preferences-import-010) (0.10) | Check every vote in the file, then add the new ones to the store. |
| [`bench preferences rank`](#bench-preferences-rank-010) (0.10) | Ratings from the store's votes (also written to ratings.v1.json). |
| [`bench bundle`](#bench-bundle-010) (0.10) | Write one zip of a run's evidence to share, redacted. |
| [`bench skills`](#bench-skills) | Skill discovery and evaluation. |
| [`bench skills list`](#bench-skills-list) | List discovered skills. |
| [`bench skills eval`](#bench-skills-eval) | Evaluate a skill using its evals/evals.json test cases. |
| [`bench skills validate`](#bench-skills-validate) | Run static checks on skills: NVIDIA SkillEvaluator Tier 1 and SkillSpector. |
| [`bench traj`](#bench-traj) | Trajectory commands. |
| [`bench traj setup`](#bench-traj-setup) | Install the submit skill, or print the prompt to send to an agent. |
| [`bench traj upload`](#bench-traj-upload) | Inspect, redact, confirm, and upload trajectory JSONL. |
| [`bench traj status`](#bench-traj-status) | Check whether an uploaded trajectory is verified in storage. |
| [`bench train`](#bench-train) | Training data commands. |
| [`bench train convert`](#bench-train-convert) | Convert BenchFlow rollout artifacts into trainer-ready data. |
| [`bench train token-coverage`](#bench-train-token-coverage) | Report whether each rollout's model calls carry token ids and logprobs. |
| [`bench train stream`](#bench-train-stream) | Print each finished rollout of a (running) job as one JSON line. |
| [`bench train validate`](#bench-train-validate) | Validate trainer-ready data. |
| [`bench train run`](#bench-train-run) | Launch training jobs. |
| [`bench train run sft`](#bench-train-run-sft) | Run a Prime-RL SFT job and record a BenchFlow manifest. |
| [`bench hub`](#bench-hub) | Environment hubs: browse a hub's environments (list/show/inspect), check registry tasks (check), and publish datasets and push runs to a BenchFlow Hub (publish/push). |
| [`bench hub list`](#bench-hub-list) | List a hub's datasets or environments (benchflow or primeintellect). |
| [`bench hub show`](#bench-hub-show) | Show a dataset's tasks, or a hosted environment's metadata. |
| [`bench hub inspect`](#bench-hub-inspect) | Inspect a file from a hub environment package. |
| [`bench hub publish`](#bench-hub-publish-010) (0.10) | Publish a folder of tasks as an immutable dataset version on a BenchFlow Hub. |
| [`bench hub push`](#bench-hub-push-010) (0.10) | Push a finished run (traces, results, an embodied trial's record and video) to a BenchFlow Hub. |
| [`bench hub check`](#bench-hub-check) | Inventory or structurally check representative registry tasks. |
| [`bench daemon`](#bench-daemon) | Manage the opt-in warm local runtime. |
| [`bench daemon start`](#bench-daemon-start) | Start a private runtime; attach runs with BENCHFLOW_DAEMON=auto. |
| [`bench daemon status`](#bench-daemon-status) | Show whether this user's local runtime is running. |
| [`bench daemon stop`](#bench-daemon-stop) | Stop an idle local runtime; never interrupt an active call. |
| [`bench sandbox`](#bench-sandbox) | Daytona sandboxes: create, list and clean up (cleanup also removes leftover Docker containers). |
| [`bench sandbox create`](#bench-sandbox-create) | Create an environment from a task directory (does not start it). |
| [`bench sandbox list`](#bench-sandbox-list) | List active Daytona sandboxes (kept Docker sandboxes: their trial folders hold kept-sandbox.json). |
| [`bench sandbox attach`](#bench-sandbox-attach-010) (0.10) | Open a root shell in a kept sandbox; with a command after --, run it there and exit with its status. |
| [`bench sandbox release`](#bench-sandbox-release-010) (0.10) | Delete a sandbox kept with --keep-sandbox. |
| [`bench sandbox cleanup`](#bench-sandbox-cleanup) | Clean up orphaned sandboxes and stale shared runtimes. |

## bench

The universal environment framework; run, author, and adopt agent benchmarks across any environment.

```text
bench [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`new`](#bench-new) | Scaffold a new benchmark task. |
| [`check`](#bench-check) | Validate task structure and the run's plan without starting a sandbox. |
| [`import`](#bench-import) | Convert task.toml + instruction.md into the unified task.md format. |
| [`run`](#bench-run) | Run an agent on tasks and score them; without -a, the oracle. |
| [`branch`](#bench-branch) | Fork an agent run at a checkpoint into labelled children. |
| [`results`](#bench-results) | Summarize jobs; with two, compare them. |
| [`view`](#bench-view) | View a trial or session trajectory in the browser. |
| [`regrade`](#bench-regrade) | Re-run the task's current verifier on each trial's frozen workspace. |
| [`export`](#bench-export) | Export a task to a compatibility split layout with a loss report. |
| [`doctor`](#bench-doctor) | Check this machine can run evals: Python/uv, sandbox, agent credentials, network. |
| [`agents`](#bench-agents) | List the agents, or show one in detail. |
| [`explain`](#bench-explain) | Explain why a trial has no score: what happened, whose problem, what next. |
| [`cache`](#bench-cache) | List or clean what BenchFlow keeps: cached layers, build records, snapshots. |
| [`eval`](#bench-eval) | Run agents on tasks (smoke, run) and read the results (metrics, view, inspect, compare). |
| [`tasks`](#bench-tasks) | Task authoring commands. |
| [`agent`](#bench-agent) | Agent management commands. |
| [`review`](#bench-review) | Grade finished rollouts against a rubric with a reviewer agent. |
| [`assess`](#bench-assess-010) (0.10) | Blind pairwise human assessments of saved outputs, kept in an opt-in sidecar. |
| [`preferences`](#bench-preferences-010) (0.10) | Human preferences and rubric scores on finished trials. |
| [`bundle`](#bench-bundle-010) (0.10) | Write one zip of a run's evidence to share, redacted. |
| [`skills`](#bench-skills) | Skill discovery and evaluation. |
| [`traj`](#bench-traj) | Trajectory commands. |
| [`train`](#bench-train) | Training data commands. |
| [`hub`](#bench-hub) | Environment hubs: browse a hub's environments (list/show/inspect), check registry tasks (check), and publish datasets and push runs to a BenchFlow Hub (publish/push). |
| [`daemon`](#bench-daemon) | Manage the opt-in warm local runtime. |
| [`sandbox`](#bench-sandbox) | Daytona sandboxes: create, list and clean up (cleanup also removes leftover Docker containers). |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--version` | flag |  | Show the version and exit. |
| `--verbose`, `-v` | flag |  | Print debug logs, with the traceback of each failed trial (the same as BENCHFLOW_LOG_LEVEL=DEBUG). Goes before the command: bench -v run … |
| `--install-completion` | flag |  | Install completion for the current shell. |
| `--show-completion` | flag |  | Show completion for the current shell, to copy it or customize the installation. |

## bench new

Scaffold a new benchmark task.

```text
bench new [OPTIONS] NAME
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `NAME` | `text` | yes | Task name |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--dir` | `path` | `tasks` | Parent directory (default: tasks/) |
| `--no-pytest` | flag |  | Skip pytest template |
| `--no-oracle`, `--no-solution` | flag |  | Skip oracle template |
| `--format` | `text` | `task-md` | Task format. New tasks use task-md; legacy scaffolding is retired. |

## bench check

Validate task structure and the run's plan without starting a sandbox.

Warnings (e.g. task.toml keys BenchFlow does not know) are printed but do not fail the check.

```text
bench check [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory or folder of tasks |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--level` | `schema` \| `structural` \| `runtime-capability` \| `publication-grade` \| `acceptance` \| `acceptance-live` \| `equivalence` | `structural` | Validation level: schema, structural, runtime-capability, publication-grade, acceptance, acceptance-live, or equivalence (runs the oracle in --sandbox, then grades wrong and value-equivalent variants of its output) |
| `--sandbox` | `text` |  | Validate runtime semantics for docker, remote-docker, daytona, modal, apple-container, or agentcore (default: a sandbox that can run the task: docker, or modal for a task that asks for GPUs) |
| `--report-output` | `path` |  | Write the acceptance-live report to this host path instead of the task-declared report path; with --level equivalence, write the battery's JSON report here |
| `--no-report-write` | flag |  | Validate acceptance-live without writing the declared report or its .sha256 sidecar (report-only dogfood; leaves the task package unmodified). Takes precedence over --report-output. |
| `--allow-task-env` | `text (repeatable)` |  | Let tasks read this variable of your environment (a $&#123;NAME&#125; env value, or their docker-compose.yaml); repeatable. A task reading an unnamed one is refused; $&#123;NAME:-default&#125; gets the default |
| `--verifier-sandbox` | `text` |  | With separate: also check what `bench run --verifier-sandbox separate` checks before it starts (on --sandbox, default docker), with the same refusals, and name where the verifier image comes from |
| `--agent` | `text` | `claude-agent-acp` | Agent to validate, as for eval run |
| `--model` | `text` |  | Model to validate, as for eval run |
| `--sandbox-user` | `text` | `agent` | Agent user to validate, as for eval run |

## bench import

Convert task.toml + instruction.md into the unified task.md format.

```text
bench import [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Legacy task directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--overwrite` | flag |  | Replace an existing task.md |
| `--remove-legacy` | flag |  | Delete split files and promote tests/solution aliases after task.md is verified |

## bench run

Run an agent on tasks and score them; without -a, the oracle.

The oracle is the task's reference solution: no model and no key. Rerunning the same command resumes its job. A run config (--config) is the whole run: its agent runs unless -a names another, and flags win over its keys.

```text
bench run [OPTIONS] [TARGETS]...
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TARGETS` | `text` | no | A task folder, a folder of tasks, or a GitHub repository (org/repo, org/repo/path or org/repo/path@ref); one per run |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--agent`, `-a` | `text` |  | The agent: a name `bench agents` lists, or nop (does nothing) (default: oracle, the task's reference solution) |
| `--model`, `-m` | `text` |  | The model, in the agent's own naming |
| `--effort`, `--reasoning-effort`, `-e` | `text` |  | Reasoning effort, when the agent exposes one (low, medium, high, max, …) |
| `--harness` | `text` |  | How the agent runs: acp (default, its ACP adapter) or native (its own CLI in headless JSON mode; claude-agent-acp and codex-acp) |
| `--agent-env`, `--env` | `text (repeatable)` |  | Environment variable for the agent, KEY=VALUE (repeatable) |
| `--task-skills`, `--no-task-skills` | flag |  | Install the task's own skills (environment/skills) for the agent; left out by default, as in BenchFlow 0.8 |
| `--dataset`, `-d` | `text` |  | A dataset instead of a target: name@version from a registry, or org/name@version from a BenchFlow Hub |
| `--registry` | `text` |  | The registry for --dataset: a URL, a file or hub:&lt;url&gt; (default: BenchFlow's registry, then $BENCHFLOW_HUB_URL for an org/name dataset) |
| `--include`, `-i` | `text (repeatable)` |  | Only tasks whose names match: a name, or a glob pattern such as 'hard-*' (comma-separated or repeated) |
| `--exclude`, `-x` | `text (repeatable)` |  | Skip tasks whose names match (same patterns as --include) |
| `--limit`, `--n-tasks`, `-n` | `integer` |  | Run at most N tasks: the first N by name, or a seeded random sample with --sample-seed |
| `--sandbox`, `-s` | `text` |  | Where trials run: docker, remote-docker, daytona, modal, apple-container, or agentcore |
| `--concurrency`, `-j` | `integer` |  | Trials running at a time (on Docker, up to --warm-pool more are prepared ahead) |
| `--warm-pool` | `integer` |  | Docker: sandboxes prepared ahead of their trial's turn, beyond the running ones (0 to --concurrency; default 1, or 0 at -j 1) |
| `--jobs-dir` | `text` |  | Where job folders go (default: jobs); a plain rerun resumes its latest job |
| `--timeout-multiplier` | `number` |  | Scale every task's agent timeout, e.g. 2.0 |
| `--agent-idle-timeout`, `--idle-timeout` | `text` |  | Abort a prompt when the agent is silent this long: seconds or 90s, 10m, 1h; 0 or none turns it off (default: 10m, 30m at effort xhigh or max) |
| `--fail-under` | `number` |  | Exit 1 when the pass rate over the planned tasks is below this (0 to 1) |
| `--freeze-workspace`, `--save-outputs` | flag |  | Save each trial's final workspace before the verifier runs, so `bench regrade` can score it again |
| `--extra-instruction` | `text` |  | Text appended to every task's instruction |
| `--verifier-sandbox` | `text` |  | task or separate. separate runs every task's verifier in a fresh sandbox that gets only the workspace and declared artifacts; a task it cannot run there refuses the whole run before anything starts (default: task, each task's own choice) |
| `--job`, `--job-name` | `text` |  | Job name under --jobs-dir (an existing job is resumed) |
| `--fresh` | flag |  | Start a new job instead of resuming the latest one in --jobs-dir |
| `--dry-run` | flag |  | Print the plan as JSON and the run card on stderr, and run nothing; for a local folder or run config, exits 1 when the run card would refuse the run. In 0.9.2 it prints the plan only, with no run card. |
| `--quiet`, `-q` | flag |  | Print only warnings, errors and the final score |
| `--config`, `--run-config` | `path` |  | YAML run-config file (the whole run spec) |
| `--source-repo` | `text` |  | GitHub repo of the tasks as org/repo (e.g. benchflow-ai/skillsbench) |
| `--source-path` | `text` |  | Folder in the repo: one task (tasks/econ-detrending-correlation) or a folder of tasks (tasks). Only this folder is downloaded |
| `--keep-sandbox` (0.10) | `text` |  | Keep a trial's sandbox: on-failure, always or never (default). `bench sandbox attach <trial>` opens a shell in it; `bench sandbox release <trial>` deletes it. |
| `--accept-deviations` (0.10) | `text` |  | Why this run differs from its task set's run-reference.toml (a strict one refuses the run without it); kept in run-card.json. |
| `--integrity` | `text` |  | off, audit (record what the agent did and check it against the task) or strict (audit, and the verifier in a separate sandbox). The verdict is in integrity/claim_verdict.json; rewards never change. |
| `--max-cost-usd` | `number` |  | Job budget in USD, checked between starts, so a job can pass it by about one rollout's cost; at the cap running rollouts are cancelled. With --trials, each repeat has its own cap |
| `--max-tokens` | `integer` |  | Job budget in tokens, checked between starts like --max-cost-usd. With --trials, each repeat has its own cap |
| `--trials` | `integer` | `1` | Run every task N times (pass@k); each repeat is a job of its own, in &lt;jobs dir&gt;/&lt;agent&gt;/trial-NN |
| `--help-all` | flag |  | Show every flag, the run settings --help leaves out included |

**More options (shown by `--help-all`)**

| Option | Type | Default | Help |
|---|---|---|---|
| `--hub` (0.10) | `text` |  | Read --dataset from this BenchFlow Hub, as the bench hub commands' --hub does (the same as --registry hub:&lt;url&gt;) |
| `--tasks-dir` | `path` |  | Local tasks directory |
| `--source-ref` | `text` |  | Branch, tag or full commit SHA to fetch (e.g. main) |
| `--source-env` | `text` |  | Hosted environment source (e.g. primeintellect/general-agent) |
| `--source-env-version` | `text` |  | Hosted environment version |
| `--source-env-arg` | `text (repeatable)` |  | Hosted environment arg as KEY=VALUE; repeatable |
| `--source-env-num-examples` | `integer` | `1` | Number of env examples |
| `--source-env-rollouts-per-example` | `integer` | `1` | Rollouts per hosted env example |
| `--source-env-max-tokens` | `integer` | `1024` | Max tokens for hosted env run |
| `--source-env-temperature` | `number` |  | Temperature for hosted env run (default: none sent, so the provider's own) |
| `--source-env-sampling-arg` | `text (repeatable)` |  | Hosted env sampling arg as KEY=VALUE; repeatable (e.g. reasoning_effort=minimal) |
| `--source-env-verifiers-version` | `text` |  | Pin the verifiers release installed with the hosted env (default: the env package's own requirement) |
| `--source-env-base-url` | `text` |  | Model endpoint for the hosted env run (default: verifiers' own, Prime inference with PRIME_API_KEY) |
| `--source-env-api-key-var` | `text` |  | Name of the environment variable holding that endpoint's API key |
| `--usage-tracking` | `text` |  | Telemetry-enforcement policy: auto, required, or off. The LiteLLM proxy is always used for routable agents (usage, cost, and llm_trajectory.jsonl are always captured); this flag only controls whether trusted telemetry is required. |
| `--environment-manifest` | `path` |  | Environment-plane manifest applied to every rollout: a path to an environment.toml, OR a 'name@version' registry spec resolved via $BENCHFLOW_ENV_REGISTRY when set, else the built-in registry shipped with benchflow (env0@prod, env0@outage). The manifest-declared stateful environment is provisioned, gated on readiness, and torn down. |
| `--state` | `text` |  | The stateful environment to bind, apart from the task. Inline JSON with an optional tool subset (e.g. &#123;"name": "env0", "tools": ["gmail", "gcal"]&#125;), OR a name@version spec, OR a manifest path. Takes precedence over --environment-manifest. |
| `--prompt` | `text (repeatable)` |  | Prompt(s) to send (default: the task prompt: the task.md body, or instruction.md for older tasks) |
| `--config-override` | `text` |  | Deep-merge a config patch into each task's resolved config for this run. Inline JSON/YAML/TOML or an @file ref, e.g. --config-override '&#123;"agent":&#123;"timeout_sec":120&#125;&#125;'. Varies one knob (budget, skills, stopping rules) while the tasks, agent, model and environment stay fixed; recorded by content hash for replay. |
| `--build-concurrency` | `integer` |  | Max concurrent docker image builds. Defaults to --concurrency. Set lower (e.g. 8) when --concurrency is high to avoid overwhelming the docker daemon with parallel builds. |
| `--worker-concurrency` | `integer` |  | Run batch eval through isolated worker subprocesses, each with at most this many concurrent tasks; --concurrency remains the aggregate target. |
| `--worker-retries` | `integer` | `1` | Retry a crashed worker shard this many times, resuming its jobs_dir. |
| `--worker-start-stagger-sec` | `number` | `1.0` | Seconds to stagger worker starts to avoid Daytona connection storms. |
| `--checkpoints` | `text` |  | Keep a sandbox snapshot after these prompts: every-prompt or prompt:N[,M]. Later `bench branch --from-checkpoint <trial>` forks from one. Docker, or Daytona direct; each is a full filesystem image. See docs/train/environment.md |
| `--checkpoint-keep` | `integer range` |  | Keep at most this many checkpoints per trial (default 3, or a run config's checkpoint_keep); the oldest is deleted. |
| `--fresh-install` | flag |  | Install the model proxy and the agent in every trial instead of starting from their cached image layers (same as BENCHFLOW_FRESH_INSTALL=1) |
| `--allow-cloud-metadata` | flag |  | Let the agent reach the cloud metadata endpoints under an open network (no block; each trial records the opt-out). Without it, or the task's allow_cloud_metadata = true, the agent starts only once the block is proven on |
| `--retry-from-checkpoint` | `text` |  | on-failure, on-timeout, or both: when a trial fails or times out, fork one retry from its last --checkpoints snapshot, verified the same way. The trial's reward is kept; the retry's is reported next to it (result.json retry block, summary checkpoint_retries). |
| `--retry-prompt` | `text` |  | Prompt for the retry, replacing the prompts after the checkpoint (the default). '@instruction' is the task instruction, '@verifier_feedback' the failed trial's reward and verifier output tail. |
| `--max-sandbox-seconds` | `number` |  | Job budget in trial wall-clock seconds, running trials included and re-checked as they run; same stop/cancel behaviour as --max-cost-usd |
| `--max-rollouts` | `integer` |  | Job budget in rollouts started, retries included: at the cap no rollout starts and running ones finish |
| `--token-budget` (0.10) | `integer` |  | Per-trial budget in tokens (input, cached included, plus output), subagents' calls included: BenchFlow's model proxy refuses the next model call once it is spent, the agent is stopped and its work graded. Replaces a task's [agent] token_budget |
| `--turn-budget` (0.10) | `integer` |  | Per-trial budget in turns, one turn being one model call through BenchFlow's model proxy (subagents' and side calls included); enforced like --token-budget. Replaces a task's [agent] turn_budget |
| `--retry-resume-session` | flag |  | The retry resumes the failed trial's conversation at the checkpoint (ACP session/load; Claude Code) instead of a fresh session. |
| `--restart-changed` (0.10) | flag |  | When the named job ran with other settings or task files, set those trials aside and run them again (default: refuse) |
| `--fail-on` | `text (repeatable)` |  | Exit 1 when any trial ended this way: timeout, error, verifier-error (comma-separated or repeated) |
| `--summary-out` | `path` |  | Write the run's benchflow.run-summary JSON (job dir, counts, gate) here |
| `--codex-apps-policy` | `text` |  | Codex account Apps: disabled (scored-task default) or inherit (explicit opt-in). |
| `--sandbox-user` | `text` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `integer` |  | Timeout (seconds) for sandbox user setup inside the environment (default: 120; keeps the file's value with --config). |
| `--context-root` | `path` |  | Repo/build-context root used to stage Dockerfile COPY sources for monorepo-authored local tasks. |
| `--base-image-override` | `text` |  | Rewrite task Dockerfile FROM images on the runtime task copy. Use only for reproducing runs whose base image moved namespaces. |
| `--separate-verifier` | flag |  | Run each verifier in a fresh sandbox built from the verifier image that receives only the frozen workspace, declared artifacts and /logs/artifacts, as if every task set \[verifier].sandbox_mode = 'separate'. Docker, remote-docker, Daytona and Modal. |
| `--separate-verifier-include` | `text (repeatable)` |  | A path outside the workspace that a separate verifier sandbox receives too, at the same path, for tasks whose agent writes graded files there (e.g. /root when the workspace is /app). Repeatable. |
| `--skills-dir` | `path` |  | Skills directory to deploy |
| `--skill-mode` | `text` | `no-skill` | Skill mode: no-skill, with-skill, or self-gen |
| `--skill-creator-dir` | `path` |  | Path to skill-creator or a skills root containing it |
| `--self-gen-no-internet` | flag |  | Disable web tools for the self-generated run |
| `--loop-strategy` | `text` |  | Harness loop strategy, e.g. 'verify-retry:k=3,feedback=names', 'self-review:k=3', or 'single-shot' (default). verify-retry re-prompts with filtered soft-verifier feedback; self-review re-prompts the agent to critique its OWN work (no verifier signal); both loop until the soft verifier passes or k retries are spent. |
| `--reviewer-agent` | `text` |  | Rubric reviewer harness |
| `--reviewer-model` | `text` |  | Rubric reviewer model |
| `--reviewer-reasoning-effort` | `text` |  | Rubric reviewer reasoning effort |
| `--reviewer-sandbox` | `text` |  | Rubric reviewer sandbox backend |
| `--reviewer-timeout-sec` | `integer range` |  | Reviewer execution budget in seconds |
| `--reviewer-idle-timeout` | `text` |  | Abort a reviewer prompt after this many idle seconds (default: 600). Pass 0 or 'none' to disable the idle watchdog and leave --reviewer-timeout-sec in charge. |
| `--reviewer-storage-mb` (0.10) | `integer range` |  | Reviewer sandbox disk in MB (default: the evidence it holds plus 2048, at least 4096, at most the sandbox backend's limit) |
| `--reviewer-memory-mb` (0.10) | `integer range` |  | Reviewer sandbox memory in MB (default: 2048) |
| `--reviewer-setup-timeout` (0.10) | `integer range` |  | Seconds for the reviewer sandbox's setup, its model proxy install among it (default: the run's --sandbox-setup-timeout) |
| `--reviewer-evidence-timeout` (0.10) | `integer range` |  | Seconds to unpack, and again to check, the workspace evidence in the reviewer sandbox (default: 1 per MiB, at least 600) |
| `--reviewer-concurrency` | `integer range` |  | Max concurrent rubric reviewers |
| `--reviewer-image` | `text` |  | Rubric reviewer container image |
| `--reviewer-agent-env` | `text (repeatable)` |  | Reviewer environment variable (KEY=VALUE) |
| `--reviewer-open-network` | flag |  | Allow unrestricted reviewer network access |
| `--reviewer-pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to read and render a submitted PDF (default: off) |
| `--reviewer-submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources (default: off) |
| `--reviewer-criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description (default: off) |
| `--allow-task-env` | `text (repeatable)` |  | Let tasks read this variable of your environment (a $&#123;NAME&#125; env value, or their docker-compose.yaml); repeatable. A task reading an unnamed one is refused; $&#123;NAME:-default&#125; gets the default |
| `--ignore-bench-version` | flag |  | Run a dataset even when this bench version is outside the range it was validated against. Only valid with --dataset. |
| `--task-manifest-out` | `path` |  | Write selected task-set manifest JSON |
| `--run-config-out` | `path` |  | Write redacted normalized run config JSON |
| `--health-summary-out` | `path` |  | Write trajectory health summary JSON |
| `--expected-tasks` | `integer` |  | Fail unless the selected task count matches |
| `--canonicalize` | `text` | `none` | Canonicalization policy: none or one-healthy-per-task |
| `--canonical-selection-out` | `path` |  | Write canonical rollout-selection JSON |
| `--canonical-jobs-dir` | `path` |  | Materialize selected rollout dirs for trainer conversion |
| `--retry-policy` | `text` | `default` | Retry policy: default or unscored-only |
| `--retry-attempts` | `integer` |  | Retries per task after a retryable failure (overrides the default of 2) |
| `--retry-concurrency` | `integer` |  | Reserved retry concurrency |
| `--publish-hf` | `text` |  | Upload final eval artifacts to a HF dataset repo |
| `--hf-prefix` | `text` |  | Path prefix in the HF repo |
| `--hf-public-read-check` | flag |  | Verify public HF reads after upload |
| `--publish-bucket` | `text` |  | Upload final eval artifacts to a HF bucket |
| `--eval-results-model` | `text` |  | HF model repo to open a community eval-results PR on |
| `--eval-results-dataset` | `text` |  | HF benchmark dataset id, e.g. org/benchmark |
| `--eval-results-task` | `text` |  | Benchmark task_id from the dataset's eval.yaml |
| `--matrix` | `path` |  | YAML file whose models: maps each alias to a model, or to &#123;model, agent, agent_env&#125;; &#123;agent: oracle&#125; for a control (docs/audit/compare-runs.md#choose-the-metric) |
| `--sample-seed` | `integer` |  | Seed for the --n-tasks random sample |
| `--seeds` | `text` |  | Seeded rollouts: run every task once per seed, e.g. 0-4 or 0,3,7 (tasks in a task format that writes seeded variants, such as embodied tasks). summary.json gets pass@k and variance per task. |

## bench branch

Fork an agent run at a checkpoint into labelled children.

Each task runs once up to the checkpoint; the sandbox (and, when asked, declared environment state) is snapshotted; every --child starts from that snapshot with a fresh agent session (child prompts must stand alone unless --resume-session reloads the parent's conversation) and is scored by the task's verifier. The job folder has the usual layout plus, per trial, tree.json and branches/&lt;fork&gt;/children/&lt;node&gt;/observation.json.

Example:

bench branch --tasks-dir tasks/my-task --agent claude-agent-acp --sandbox daytona --prompt "Write draft.txt containing: Hello world" --prompt @instruction --checkpoint-after-prompt 1 --child "label=baseline" --child "label=hint,prompt=Rename draft.txt."

```text
bench branch [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-dir` | `path` |  | A task directory, or a directory of tasks (run one after another). |
| `--agent` | `text` |  | Agent name, or 'oracle' (children run the task's solve.sh). Default: the checkpoint's agent with --from-checkpoint, else claude-agent-acp. |
| `--model`, `-m` | `text` |  | Model id for the agent, e.g. claude-haiku-5-5 (none for oracle and nop) |
| `--reasoning-effort` | `text` |  | Agent reasoning effort (e.g. max) |
| `--harness` | `text` |  | How the agent runs: acp (default, its ACP adapter) or native (its own CLI in headless JSON mode; claude-agent-acp and codex-acp) |
| `--sandbox` | `text` |  | docker or daytona (Daytona direct mode snapshots). Default: docker, or the checkpoint's provider. |
| `--prompt` | `text (repeatable)` |  | Parent prompt; repeatable, sent in order. '@instruction' is the task's instruction. Default: the task's prompts. |
| `--checkpoint-after-prompt` | `integer` |  | Branch after this many parent prompts (0: before the first). Default 1; 0 for the oracle and with --from-checkpoint. |
| `--child` | `text (repeatable)` |  | One child, as label=NAME[,parent=LABEL][,prompt=TEXT\|,prompt-file=PATH]; give at least two. prompt= takes the rest of the option (commas allowed). parent=LABEL forks this child from child LABEL's state after LABEL has run its prompts (a nested fork of at least two children). Without a prompt the child runs the parent's remaining prompts, or the task's prompts when the checkpoint is after the last one. |
| `--snapshot-layers` | `text` | `sandbox` | What the checkpoint captures: sandbox (container filesystem), environment (declared database state, needs a manifest), or both. |
| `--parent` | `text` |  | continue: restore the parent world after the children, send the remaining prompts and verify. discard: skip that restore (one fewer), leave the parent unverified. Default: continue, or discard with --from-checkpoint. |
| `--retain-snapshots` | flag |  | Keep the checkpoint's sandbox snapshot so a later --from-checkpoint can branch from it again (otherwise deleted). `bench sandbox cleanup` removes kept Daytona snapshots once stale. |
| `--from-checkpoint` | `path` |  | A trial folder from an earlier `bench branch --retain-snapshots` run: branch again from its kept sandbox snapshot. Needs --tasks-dir holding the same task. |
| `--checkpoint`, `--fork` | `text` |  | With --from-checkpoint: a fork id from tree.json or an automatic checkpoint as prompt:N (default: the last kept fork, else the last kept checkpoint). |
| `--checkpoints` | `text` |  | Also keep a sandbox snapshot after these parent prompts (every-prompt or prompt:N[,M]) for a later --from-checkpoint. |
| `--checkpoint-keep` | `integer range` | `3` | Keep at most this many --checkpoints per trial. |
| `--resume-session` | flag |  | Children resume the parent's agent conversation (ACP session/load) instead of starting fresh, so a child prompt can say 'continue'. Needs an agent that advertises loadSession and keeps its session on disk (Claude Code), and a checkpoint after prompt 1+. |
| `--child-retries` | `integer range` | `1` | Retry a child this many times when it failed before its agent did anything (a provider hiccup such as a connect timeout). |
| `--stop-on-child-failure` | flag |  | Stop a fork at the first child that fails for good (default: the other children still run and the fork is partial). |
| `--concurrency` | `integer range` | `1` | Run up to this many children of a fork at once, each in its own sandbox created from the snapshot (implies --isolate-children). |
| `--isolate-children` | flag |  | Run each child in its own sandbox created from the snapshot instead of restoring the parent's sandbox between children. Implied by --concurrency &gt; 1 and by nested children. |
| `--agent-env` | `text (repeatable)` |  | Agent env var (KEY=VALUE); repeatable |
| `--include` | `text (repeatable)` |  | Only these task names; repeatable |
| `--jobs-dir` | `path` | `jobs` | Output directory |
| `--job-name` | `text` |  | Job folder name (default: branch-&lt;time&gt;) |

## bench results

Summarize jobs; with two, compare them.

With no job, or a folder of jobs, it lists the jobs (`bench eval list`); with one, it reads the job (`bench eval inspect`); with two, it pairs them by task (`bench eval compare`).

```text
bench results [OPTIONS] [JOBS]...
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOBS` | `text` | no | Job folders, or a folder of jobs (default: ./jobs); one job is read trial by trial, two are compared task by task |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the benchflow.job document (one job) or the benchflow.comparison document (two) |
| `--attempts` | `text` | `best` | best (one attempt per task, as resume picks) or all |
| `--include-controls` | flag |  | Count control runs (oracle, empty) in the denominators |
| `--by` | `text` |  | One job: also count per group, e.g. agent,model; two: pair on these keys too |

## bench view

View a trial or session trajectory in the browser.

```text
bench view [OPTIONS] ROLLOUT_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `ROLLOUT_DIR` | `text` | yes | Rollout directory, job directory, trajectory JSONL file, or an hf://&lt;org&gt;/&lt;dataset&gt; trajectory dataset (optionally with /subpath) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--port` | `integer` | `8888` | Server port |
| `--confirm` | flag |  | Show an Approve & submit / Not this one bar on the page; print DECISION: approved\|rejected and exit 0 on approve, 3 on reject. |
| `--redaction-summary` | `text` |  | Masked-secret breakdown shown in the --confirm bar, e.g. '2 API keys, 1 bearer token' (lift it from `bench traj upload PATH --dry-run`). Display-only; no effect without --confirm. |

## bench regrade

Re-run the task's current verifier on each trial's frozen workspace.

The new score is written beside the original (regrade.json in each trial, regrade-summary.json in PATH); result.json is never changed. Trials whose workspace was not frozen (--freeze-workspace) are listed as not regradable.

```text
bench regrade [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A job folder, or one trial folder inside it |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-dir` | `path` |  | Folder of task folders (or one task folder) holding the changed verifiers. Default: the task path each trial recorded. |
| `--sandbox` | `text` |  | Backend for the fresh sandboxes (default: as run) |
| `--concurrency` | `integer range` | `4` | Trials regraded at once |
| `--reason` | `text` |  | Why the verifier changed; kept in each block |
| `--json` | flag |  | Print the summary as JSON |
| `--review` (0.10) | flag |  | Also rerun each trial's rubric review on its saved outputs, gated by the new test, and compare the combined score. The reviewer is the one the trial recorded, changed by --reviewer-* |
| `--reviewer-agent` (0.10) | `text` |  | Rubric reviewer harness |
| `--reviewer-model` (0.10) | `text` |  | Rubric reviewer model |
| `--reviewer-reasoning-effort` (0.10) | `text` |  | Rubric reviewer reasoning effort |
| `--reviewer-sandbox` (0.10) | `text` |  | Rubric reviewer sandbox backend |
| `--reviewer-timeout-sec` (0.10) | `integer range` |  | Reviewer execution budget in seconds |
| `--reviewer-idle-timeout` (0.10) | `text` |  | Abort a reviewer prompt after this many idle seconds (default: 600). Pass 0 or 'none' to disable the idle watchdog and leave --reviewer-timeout-sec in charge. |
| `--reviewer-storage-mb` (0.10) | `integer range` |  | Reviewer sandbox disk in MB (default: the evidence it holds plus 2048, at least 4096, at most the sandbox backend's limit) |
| `--reviewer-memory-mb` (0.10) | `integer range` |  | Reviewer sandbox memory in MB (default: 2048) |
| `--reviewer-setup-timeout` (0.10) | `integer range` |  | Seconds for the reviewer sandbox's setup, its model proxy install among it (default: the run's --sandbox-setup-timeout) |
| `--reviewer-evidence-timeout` (0.10) | `integer range` |  | Seconds to unpack, and again to check, the workspace evidence in the reviewer sandbox (default: 1 per MiB, at least 600) |
| `--reviewer-concurrency` (0.10) | `integer range` |  | Max concurrent rubric reviewers |
| `--reviewer-image` (0.10) | `text` |  | Rubric reviewer container image |
| `--reviewer-agent-env` (0.10) | `text (repeatable)` |  | Reviewer environment variable (KEY=VALUE) |
| `--reviewer-open-network` (0.10) | flag |  | Allow unrestricted reviewer network access |
| `--reviewer-pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to read and render a submitted PDF (default: off) |
| `--reviewer-submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources (default: off) |
| `--reviewer-criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description (default: off) |

## bench export

Export a task to a compatibility split layout with a loss report.

```text
bench export [OPTIONS] TASK_DIR [OUTPUT_DIR]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory to export |
| `OUTPUT_DIR` | `path` | no | Destination split-layout directory (omit with --report-only) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--target` | `text` | `task-toml` | Layout to write: task-toml |
| `--overwrite` | flag |  | Replace an existing export directory |
| `--report-only` | flag |  | Print the compatibility loss report without writing files |

## bench doctor

Check this machine can run evals: Python/uv, sandbox, agent credentials, network.

Credentials are reported by name, source and expiry only; values are never printed. Exits 1 when a required check fails.

```text
bench doctor [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--sandbox` | `text` | `docker` | Sandbox the checks treat as required: docker (default), remote-docker or daytona. Others are listed but not checked yet. |
| `--offline` | flag |  | Skip network probes and the live Daytona credential check. |
| `--json` | flag |  | Print the report as JSON. |
| `--agent-start` | `text (repeatable)` |  | Also install this agent in a fresh --sandbox and open its ACP connection without a prompt (no model call), to catch a broken install or login before a batch (repeatable). |

## bench agents

List the agents, or show one in detail.

```text
bench agents [OPTIONS] [NAME]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `NAME` | `text` | no | Show this agent in detail |

## bench explain

Explain why a trial has no score: what happened, whose problem, what next.

The code is the one the run summary names; without one, every code is listed.

Examples:

```bash
bench explain                    # every code
bench explain R212               # the model provider refused the credential
bench explain verifier_failure --json
```

```text
bench explain [OPTIONS] [CODE]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `[CODE]` | `text` | no | The code the run summary names, such as R212 or verifier_failure (omit to list every code). |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the page, or the list, as JSON. |

## bench cache

List or clean what BenchFlow keeps: cached layers, build records, snapshots.

Without a command, lists (as `bench cache list`).

Examples:

```bash
bench cache list --only images -s docker   # cached layers on Docker
bench cache clean --older-than 30d         # what would go
```

```text
bench cache [OPTIONS] [COMMAND] [ARGS]...
```

| Command | What it does |
|---|---|
| [`list`](#bench-cache-list) | List what BenchFlow keeps, with sizes and creation times (the default). |
| [`clean`](#bench-cache-clean) | Print what would be removed; remove it with --remove. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--only` | `text (repeatable)` |  | Only these kinds: images, build-cache, pulled-images (comma-separated or repeated; default: images and build-cache) |
| `--sandbox`, `-s` | `text (repeatable)` |  | docker or daytona (default: docker when installed, daytona when DAYTONA_API_KEY is set); daytona lists this BENCHFLOW_DAYTONA_OWNER's content snapshots |
| `--older-than` | `text` |  | Only what was created longer ago than this (e.g. 30d, 12h); the age is the creation time, not the last use |
| `--json` | flag |  | Print JSON on stdout; notes go to stderr |

## bench cache list

List what BenchFlow keeps, with sizes and creation times (the default).

Nothing is removed or tagged.

Examples:

```bash
bench cache list --only images -s docker   # cached layers on Docker
bench cache list --older-than 30d --json   # made 30+ days ago, as JSON
```

```text
bench cache list [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--only` | `text (repeatable)` |  | Only these kinds: images, build-cache, pulled-images (comma-separated or repeated; default: images and build-cache) |
| `--sandbox`, `-s` | `text (repeatable)` |  | docker or daytona (default: docker when installed, daytona when DAYTONA_API_KEY is set); daytona lists this BENCHFLOW_DAYTONA_OWNER's content snapshots |
| `--older-than` | `text` |  | Only what was created longer ago than this (e.g. 30d, 12h); the age is the creation time, not the last use |
| `--json` | flag |  | Print JSON on stdout; notes go to stderr |

## bench cache clean

Print what would be removed; remove it with --remove.

What something still uses stays, with the reason.

Examples:

```bash
bench cache clean                            # what would be removed
bench cache clean --only images --remove     # remove idle cached layers
bench cache clean --only pulled-images --remove
```

```text
bench cache clean [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--only` | `text (repeatable)` |  | Only these kinds: images, build-cache, pulled-images (comma-separated or repeated; default: images and build-cache) |
| `--sandbox`, `-s` | `text (repeatable)` |  | docker or daytona (default: docker when installed, daytona when DAYTONA_API_KEY is set); daytona lists this BENCHFLOW_DAYTONA_OWNER's content snapshots |
| `--older-than` | `text` |  | Only what was created longer ago than this (e.g. 30d, 12h); the age is the creation time, not the last use |
| `--json` | flag |  | Print JSON on stdout; notes go to stderr |
| `--remove` | flag |  | Remove what is listed; without it, nothing is removed |

## bench eval

Run agents on tasks (smoke, run) and read the results (metrics, view, inspect, compare).

```text
bench eval [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`smoke`](#bench-eval-smoke) | Run a bundled hello-world task once per credentialed agent, one at a time. |
| [`run`](#bench-eval-run) | Run an evaluation; single task or batch. |
| [`adopt`](#bench-eval-adopt) | Adopt an external benchmark into benchflow. |
| [`compare-lift`](#bench-eval-compare-lift) | Compare paired baseline vs trained eval job directories. |
| [`score`](#bench-eval-score) | Retry a failed reviewer using the saved workspace and trajectory. |
| [`regrade`](#bench-eval-regrade) | Re-run the task's current verifier on each trial's frozen workspace. |
| [`list`](#bench-eval-list) | List the jobs in a jobs folder, one row each, with their scores. |
| [`metrics`](#bench-eval-metrics) | Collect and display metrics from a jobs directory. |
| [`view`](#bench-eval-view) | View a trial or session trajectory in the browser. |
| [`continue`](#bench-eval-continue) | Resume a previous unfinished (timed-out) OpenHands run to completion. |
| [`branch`](#bench-eval-branch) | Fork an agent run at a checkpoint into labelled children. |
| [`inspect`](#bench-eval-inspect) | Read a finished job or trial: denominators, one row per trial, or --json. |
| [`compare`](#bench-eval-compare) | Pair two jobs by task: rewards, deltas, fair denominators, setting checks. |
| [`resume`](#bench-eval-resume) | Finish an interrupted job from its folder: rerun only the unfinished tasks. |
| [`branches`](#bench-eval-branches) | Show every fork and child of a job's (or one trial's) branches: request, status, reward, advantage, tokens, USD and sandbox-seconds. |

## bench eval smoke

Run a bundled hello-world task once per credentialed agent, one at a time.

Checks the machine first (as `bench doctor`), then runs each agent in its own `bench eval run`, prints agent, reward, time and trajectory path, and exits 1 unless every run scores 1.0.

```text
bench eval smoke [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--agent` | `text (repeatable)` |  | Agent to smoke, as NAME or NAME=MODEL; repeatable. Default: every agent `bench doctor` found a working credential for (claude, codex, gemini). |
| `--sandbox` | `text` | `docker` | Sandbox: docker (default) or daytona. |
| `--jobs-dir` | `path` | `jobs/smoke` | Parent directory; each smoke writes a timestamped folder here. |
| `--timeout-sec` | `integer range` | `900` | Kill a run that takes longer than this many seconds. |

## bench eval run

Run an evaluation; single task or batch.

```text
bench eval run [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--config`, `--run-config` | `path` |  | YAML run-config file (the whole run spec) |
| `--tasks-dir` | `path` |  | Local tasks directory |
| `--source-repo` | `text` |  | GitHub repo of the tasks as org/repo (e.g. benchflow-ai/skillsbench) |
| `--source-path` | `text` |  | Folder in the repo: one task (tasks/econ-detrending-correlation) or a folder of tasks (tasks). Only this folder is downloaded |
| `--source-ref` | `text` |  | Branch, tag or full commit SHA to fetch (e.g. main) |
| `--source-env` | `text` |  | Hosted environment source (e.g. primeintellect/general-agent) |
| `--source-env-version` | `text` |  | Hosted environment version |
| `--source-env-arg` | `text (repeatable)` |  | Hosted environment arg as KEY=VALUE; repeatable |
| `--source-env-num-examples` | `integer` | `1` | Number of env examples |
| `--source-env-rollouts-per-example` | `integer` | `1` | Rollouts per hosted env example |
| `--source-env-max-tokens` | `integer` | `1024` | Max tokens for hosted env run |
| `--source-env-temperature` | `number` |  | Temperature for hosted env run (default: none sent, so the provider's own) |
| `--source-env-sampling-arg` | `text (repeatable)` |  | Hosted env sampling arg as KEY=VALUE; repeatable (e.g. reasoning_effort=minimal) |
| `--source-env-verifiers-version` | `text` |  | Pin the verifiers release installed with the hosted env (default: the env package's own requirement) |
| `--source-env-base-url` | `text` |  | Model endpoint for the hosted env run (default: verifiers' own, Prime inference with PRIME_API_KEY) |
| `--source-env-api-key-var` | `text` |  | Name of the environment variable holding that endpoint's API key |
| `--agent`, `-a` | `text` |  | Agent: claude, codex, gemini, … (bench agents), oracle (the task's reference solution) or nop (does nothing) |
| `--model`, `-m` | `text` |  | Model id for the agent, e.g. claude-haiku-5-5 (none for oracle and nop) |
| `--reasoning-effort` | `text` |  | Agent reasoning/thinking effort when the agent exposes one (e.g. max) |
| `--harness` | `text` |  | How the agent runs: acp (default, its ACP adapter) or native (its own CLI in headless JSON mode; claude-agent-acp and codex-acp) |
| `--sandbox`, `-s` | `text` |  | Sandbox: docker, remote-docker, daytona, modal, apple-container, or agentcore |
| `--usage-tracking` | `text` |  | Telemetry-enforcement policy: auto, required, or off. The LiteLLM proxy is always used for routable agents (usage, cost, and llm_trajectory.jsonl are always captured); this flag only controls whether trusted telemetry is required. |
| `--environment-manifest` | `path` |  | Environment-plane manifest applied to every rollout: a path to an environment.toml, OR a 'name@version' registry spec resolved via $BENCHFLOW_ENV_REGISTRY when set, else the built-in registry shipped with benchflow (env0@prod, env0@outage). The manifest-declared stateful environment is provisioned, gated on readiness, and torn down. |
| `--state` | `text` |  | The stateful environment to bind, apart from the task. Inline JSON with an optional tool subset (e.g. &#123;"name": "env0", "tools": ["gmail", "gcal"]&#125;), OR a name@version spec, OR a manifest path. Takes precedence over --environment-manifest. |
| `--prompt` | `text (repeatable)` |  | Prompt(s) to send (default: the task prompt: the task.md body, or instruction.md for older tasks) |
| `--config-override` | `text` |  | Deep-merge a config patch into each task's resolved config for this run. Inline JSON/YAML/TOML or an @file ref, e.g. --config-override '&#123;"agent":&#123;"timeout_sec":120&#125;&#125;'. Varies one knob (budget, skills, stopping rules) while the tasks, agent, model and environment stay fixed; recorded by content hash for replay. |
| `--concurrency`, `-j` | `integer` |  | Trials running at a time. On Docker a job also prepares up to --warm-pool sandboxes ahead (default 1 when this is 2 or more) |
| `--build-concurrency` | `integer` |  | Max concurrent docker image builds. Defaults to --concurrency. Set lower (e.g. 8) when --concurrency is high to avoid overwhelming the docker daemon with parallel builds. |
| `--warm-pool` | `integer` |  | Docker: how many sandboxes to prepare ahead of their trial's turn, beyond the --concurrency running ones (0 to --concurrency; default 1, or 0 when --concurrency is 1) |
| `--worker-concurrency` | `integer` |  | Run batch eval through isolated worker subprocesses, each with at most this many concurrent tasks; --concurrency remains the aggregate target. |
| `--worker-retries` | `integer` | `1` | Retry a crashed worker shard this many times, resuming its jobs_dir. |
| `--worker-start-stagger-sec` | `number` | `1.0` | Seconds to stagger worker starts to avoid Daytona connection storms. |
| `--agent-idle-timeout`, `--idle-timeout` | `text` |  | Abort a prompt when the agent is silent this long: seconds, or 90s, 10m, 1h; 0, none or null turns it off and leaves the task's wall-clock timeout (default: 10m, 30m at effort xhigh or max) |
| `--checkpoints` | `text` |  | Keep a sandbox snapshot after these prompts: every-prompt or prompt:N[,M]. Later `bench branch --from-checkpoint <trial>` forks from one. Docker, or Daytona direct; each is a full filesystem image. See docs/train/environment.md |
| `--checkpoint-keep` | `integer range` |  | Keep at most this many checkpoints per trial (default 3, or a run config's checkpoint_keep); the oldest is deleted. |
| `--freeze-workspace` | flag |  | Save each trial's final workspace and declared artifacts (evidence/, with a manifest) before the verifier runs, so `bench regrade` can re-score it with a changed verifier. |
| `--keep-sandbox` (0.10) | `text` |  | Keep a trial's sandbox: on-failure, always or never (default). `bench sandbox attach <trial>` opens a shell in it; `bench sandbox release <trial>` deletes it. |
| `--accept-deviations` (0.10) | `text` |  | Why this run differs from its task set's run-reference.toml (a strict one refuses the run without it); kept in run-card.json. |
| `--fresh-install` | flag |  | Install the model proxy and the agent in every trial instead of starting from their cached image layers (same as BENCHFLOW_FRESH_INSTALL=1) |
| `--allow-cloud-metadata` | flag |  | Let the agent reach the cloud metadata endpoints under an open network (no block; each trial records the opt-out). Without it, or the task's allow_cloud_metadata = true, the agent starts only once the block is proven on |
| `--integrity` | `text` |  | off, audit (record what the agent did and check it against the task) or strict (audit, and the verifier in a separate sandbox). The verdict is in integrity/claim_verdict.json; rewards never change. |
| `--verifier-sandbox` | `text` |  | task (default: each task's own choice) or separate: every verifier in a fresh sandbox with only the workspace and the declared artifacts; a task that cannot run there refuses the run. |
| `--retry-from-checkpoint` | `text` |  | on-failure, on-timeout, or both: when a trial fails or times out, fork one retry from its last --checkpoints snapshot, verified the same way. The trial's reward is kept; the retry's is reported next to it (result.json retry block, summary checkpoint_retries). |
| `--retry-prompt` | `text` |  | Prompt for the retry, replacing the prompts after the checkpoint (the default). '@instruction' is the task instruction, '@verifier_feedback' the failed trial's reward and verifier output tail. |
| `--max-cost-usd` | `number` |  | Job budget in USD, checked between starts, so a job can pass it by about one rollout's cost; at the cap running rollouts are cancelled. With --trials, each repeat has its own cap |
| `--max-sandbox-seconds` | `number` |  | Job budget in trial wall-clock seconds, running trials included and re-checked as they run; same stop/cancel behaviour as --max-cost-usd |
| `--max-tokens` | `integer` |  | Job budget in tokens, checked between starts like --max-cost-usd. With --trials, each repeat has its own cap |
| `--max-rollouts` | `integer` |  | Job budget in rollouts started, retries included: at the cap no rollout starts and running ones finish |
| `--token-budget` (0.10) | `integer` |  | Per-trial budget in tokens (input, cached included, plus output), subagents' calls included: BenchFlow's model proxy refuses the next model call once it is spent, the agent is stopped and its work graded. Replaces a task's [agent] token_budget |
| `--turn-budget` (0.10) | `integer` |  | Per-trial budget in turns, one turn being one model call through BenchFlow's model proxy (subagents' and side calls included); enforced like --token-budget. Replaces a task's [agent] turn_budget |
| `--retry-resume-session` | flag |  | The retry resumes the failed trial's conversation at the checkpoint (ACP session/load; Claude Code) instead of a fresh session. |
| `--quiet` | flag |  | Print only warnings, errors and the final score: no Rich dashboard on a TTY, no progress heartbeat (the '… Nmin, K tool calls' lines) and no per-trial log lines |
| `--jobs-dir` | `text` |  | Output directory (default jobs); a plain run resumes its latest job |
| `--fresh` | flag |  | Start a new timestamped job instead of resuming the latest job in --jobs-dir (the default reuses its finished tasks) |
| `--job-name` | `text` |  | Job folder name under --jobs-dir (resumed if it exists) |
| `--restart-changed` (0.10) | flag |  | When the named job ran with other settings or task files, set those trials aside and run them again (default: refuse) |
| `--fail-under` | `number` |  | Exit 1 when the pass rate (passed / planned tasks; one a budget kept from running counts as not passed) is below this, e.g. 0.8 |
| `--fail-on` | `text (repeatable)` |  | Exit 1 when any trial ended this way: timeout, error, verifier-error (comma-separated or repeated) |
| `--summary-out` | `path` |  | Write the run's benchflow.run-summary JSON (job dir, counts, gate) here |
| `--codex-apps-policy` | `text` |  | Codex account Apps: disabled (scored-task default) or inherit (explicit opt-in). |
| `--sandbox-user` | `text` | `agent` | Sandbox user (null for root) |
| `--sandbox-setup-timeout` | `integer` |  | Timeout (seconds) for sandbox user setup inside the environment (default: 120; keeps the file's value with --config). |
| `--context-root` | `path` |  | Repo/build-context root used to stage Dockerfile COPY sources for monorepo-authored local tasks. |
| `--base-image-override` | `text` |  | Rewrite task Dockerfile FROM images on the runtime task copy. Use only for reproducing runs whose base image moved namespaces. |
| `--separate-verifier` | flag |  | Run each verifier in a fresh sandbox built from the verifier image that receives only the frozen workspace, declared artifacts and /logs/artifacts, as if every task set \[verifier].sandbox_mode = 'separate'. Docker, remote-docker, Daytona and Modal. |
| `--separate-verifier-include` | `text (repeatable)` |  | A path outside the workspace that a separate verifier sandbox receives too, at the same path, for tasks whose agent writes graded files there (e.g. /root when the workspace is /app). Repeatable. |
| `--skills-dir` | `path` |  | Skills directory to deploy |
| `--skill-mode` | `text` | `no-skill` | Skill mode: no-skill, with-skill, or self-gen |
| `--skill-creator-dir` | `path` |  | Path to skill-creator or a skills root containing it |
| `--self-gen-no-internet` | flag |  | Disable web tools for the self-generated run |
| `--loop-strategy` | `text` |  | Harness loop strategy, e.g. 'verify-retry:k=3,feedback=names', 'self-review:k=3', or 'single-shot' (default). verify-retry re-prompts with filtered soft-verifier feedback; self-review re-prompts the agent to critique its OWN work (no verifier signal); both loop until the soft verifier passes or k retries are spent. |
| `--reviewer-agent` | `text` |  | Rubric reviewer harness |
| `--reviewer-model` | `text` |  | Rubric reviewer model |
| `--reviewer-reasoning-effort` | `text` |  | Rubric reviewer reasoning effort |
| `--reviewer-sandbox` | `text` |  | Rubric reviewer sandbox backend |
| `--reviewer-timeout-sec` | `integer range` |  | Reviewer execution budget in seconds |
| `--reviewer-idle-timeout` | `text` |  | Abort a reviewer prompt after this many idle seconds (default: 600). Pass 0 or 'none' to disable the idle watchdog and leave --reviewer-timeout-sec in charge. |
| `--reviewer-storage-mb` (0.10) | `integer range` |  | Reviewer sandbox disk in MB (default: the evidence it holds plus 2048, at least 4096, at most the sandbox backend's limit) |
| `--reviewer-memory-mb` (0.10) | `integer range` |  | Reviewer sandbox memory in MB (default: 2048) |
| `--reviewer-setup-timeout` (0.10) | `integer range` |  | Seconds for the reviewer sandbox's setup, its model proxy install among it (default: the run's --sandbox-setup-timeout) |
| `--reviewer-evidence-timeout` (0.10) | `integer range` |  | Seconds to unpack, and again to check, the workspace evidence in the reviewer sandbox (default: 1 per MiB, at least 600) |
| `--reviewer-concurrency` | `integer range` |  | Max concurrent rubric reviewers |
| `--reviewer-image` | `text` |  | Rubric reviewer container image |
| `--reviewer-agent-env` | `text (repeatable)` |  | Reviewer environment variable (KEY=VALUE) |
| `--reviewer-open-network` | flag |  | Allow unrestricted reviewer network access |
| `--reviewer-pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to read and render a submitted PDF (default: off) |
| `--reviewer-submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources (default: off) |
| `--reviewer-criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description (default: off) |
| `--agent-env` | `text (repeatable)` |  | Agent env var (KEY=VALUE) |
| `--allow-task-env` | `text (repeatable)` |  | Let tasks read this variable of your environment (a $&#123;NAME&#125; env value, or their docker-compose.yaml); repeatable. A task reading an unnamed one is refused; $&#123;NAME:-default&#125; gets the default |
| `--include`, `-i` | `text (repeatable)` |  | Only run these task names; repeatable (e.g. --include jax-computing-basics --include data-to-d3) |
| `--exclude`, `-x` | `text (repeatable)` |  | Skip these task names; repeatable (e.g. --exclude quantum-numerical-simulation) |
| `--dataset`, `-d` | `text` |  | Registry dataset to run, as &lt;name&gt;@&lt;version&gt; (e.g. skillsbench@1.1). Resolves the pinned snapshot and verifies task content digests. |
| `--registry` | `text` |  | Dataset registry JSON URL or local file, or a BenchFlow Hub as hub:&lt;url&gt; (default: BenchFlow's registry, then $BENCHFLOW_HUB_URL). Only valid with --dataset. |
| `--ignore-bench-version` | flag |  | Run a dataset even when this bench version is outside the range it was validated against. Only valid with --dataset. |
| `--task-manifest-out` | `path` |  | Write selected task-set manifest JSON |
| `--run-config-out` | `path` |  | Write redacted normalized run config JSON |
| `--health-summary-out` | `path` |  | Write trajectory health summary JSON |
| `--expected-tasks` | `integer` |  | Fail unless the selected task count matches |
| `--canonicalize` | `text` | `none` | Canonicalization policy: none or one-healthy-per-task |
| `--canonical-selection-out` | `path` |  | Write canonical rollout-selection JSON |
| `--canonical-jobs-dir` | `path` |  | Materialize selected rollout dirs for trainer conversion |
| `--retry-policy` | `text` | `default` | Retry policy: default or unscored-only |
| `--retry-attempts` | `integer` |  | Retries per task after a retryable failure (overrides the default of 2) |
| `--retry-concurrency` | `integer` |  | Reserved retry concurrency |
| `--publish-hf` | `text` |  | Upload final eval artifacts to a HF dataset repo |
| `--hf-prefix` | `text` |  | Path prefix in the HF repo |
| `--hf-public-read-check` | flag |  | Verify public HF reads after upload |
| `--publish-bucket` | `text` |  | Upload final eval artifacts to a HF bucket |
| `--eval-results-model` | `text` |  | HF model repo to open a community eval-results PR on |
| `--eval-results-dataset` | `text` |  | HF benchmark dataset id, e.g. org/benchmark |
| `--eval-results-task` | `text` |  | Benchmark task_id from the dataset's eval.yaml |
| `--matrix` | `path` |  | YAML file whose models: maps each alias to a model, or to &#123;model, agent, agent_env&#125;; &#123;agent: oracle&#125; for a control (docs/audit/compare-runs.md#choose-the-metric) |
| `--trials` | `integer` | `1` | Run every task N times (pass@k); each repeat is a job of its own, in &lt;jobs dir&gt;/&lt;agent&gt;/trial-NN |
| `--limit`, `--n-tasks`, `-n` | `integer` |  | Run at most N tasks after --include/--exclude: the first N, or a seeded random sample with --sample-seed |
| `--sample-seed` | `integer` |  | Seed for the --n-tasks random sample |
| `--timeout-multiplier` | `number` |  | Scale every task's agent time budget (task.md agent.timeout_sec), e.g. 2.0 |
| `--extra-instruction` | `text` |  | Text appended to every task prompt (prompt ablations; recorded per rollout) |
| `--dry-run` | flag |  | Resolve the plan (agent, model, selected tasks, seeds, budgets) and print it without running; exits 1 when the run card would refuse the run. In 0.9.2 it prints the plan only, with no run card. |
| `--seeds` | `text` |  | Seeded rollouts: run every task once per seed, e.g. 0-4 or 0,3,7 (tasks in a task format that writes seeded variants, such as embodied tasks). summary.json gets pass@k and variance per task. |

## bench eval adopt

Adopt an external benchmark into benchflow.

Default (convert): `bench eval adopt <source>` scaffolds benchmarks/&lt;name&gt;/ if missing, then drives the codex conversion. `--scaffold-only` only writes the package; `--verify` runs the parity gate for the named benchmark.

```text
bench eval adopt [OPTIONS] [TARGET]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TARGET` | `text` | no | In convert mode (default) the SOURCE repo/path to adopt; in --verify / --scaffold-only mode the benchmark SLUG. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--name` | `text` |  | Benchmark slug for convert mode (default: derived from source) |
| `--verify` | flag |  | Run the parity gate for the named benchmark |
| `--scaffold-only` | flag |  | Only scaffold the package, do not convert |
| `--model` | `text` |  | Model for the codex driver |
| `--dry-run` | flag |  | Print the launch command, do not run |
| `--codex-bin` | `text` | `codex` | Host codex binary |
| `--codex-config`, `-c` | `text (repeatable)` |  | Codex config override as key=value, passed to codex as `-c key=value`; repeatable (e.g. -c service_tier=flex to work around host ~/.codex/config.toml drift) |
| `--benchmarks-dir` | `path` |  | Target benchmarks/ directory |
| `--tolerance` | `number` | `0.02` | Max abs reward delta (statistical) |
| `--issue-out` | `path` |  | Write the divergence issue draft here |
| `--roundtrip-task` | `path` |  | Also run the structural round-trip check on this task dir |
| `--rerun` | flag |  | Independently re-execute parity_test.py --mode side-by-side and score its fresh output, instead of the recorded parity_experiment.json |

## bench eval compare-lift

Compare paired baseline vs trained eval job directories.

```text
bench eval compare-lift [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--baseline` | `directory` |  | Baseline BenchFlow job directory. Required. |
| `--trained` | `directory` |  | Trained BenchFlow job directory. Required. |
| `--out` | `path` |  | Markdown report output path. Required. |
| `--json-out` | `path` |  | JSON report output path. Required. |
| `--bootstrap-samples` | `integer range` | `1000` | Bootstrap samples for paired delta confidence intervals. |
| `--bootstrap-seed` | `integer` | `0` | Deterministic seed for bootstrap confidence intervals. |
| `--allow-duplicate-first-by-path` | flag |  | Allow duplicate healthy task rollouts and keep the first by deterministic path order. |

## bench eval score

Retry a failed reviewer using the saved workspace and trajectory.

```text
bench eval score [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A saved task rollout directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-root` | `path` |  | Trusted original task directory or collection Required. |
| `--force` | flag |  | Create a new revision even if scoring already completed |
| `--reviewer-agent` | `text` |  | Rubric reviewer harness |
| `--reviewer-model` | `text` |  | Rubric reviewer model |
| `--reviewer-reasoning-effort` | `text` |  | Rubric reviewer reasoning effort |
| `--reviewer-sandbox` | `text` |  | Rubric reviewer sandbox backend |
| `--reviewer-timeout-sec` | `integer range` |  | Reviewer execution budget in seconds |
| `--reviewer-idle-timeout` | `text` |  | Abort a reviewer prompt after this many idle seconds (default: 600). Pass 0 or 'none' to disable the idle watchdog and leave --reviewer-timeout-sec in charge. |
| `--reviewer-storage-mb` (0.10) | `integer range` |  | Reviewer sandbox disk in MB (default: the evidence it holds plus 2048, at least 4096, at most the sandbox backend's limit) |
| `--reviewer-memory-mb` (0.10) | `integer range` |  | Reviewer sandbox memory in MB (default: 2048) |
| `--reviewer-setup-timeout` (0.10) | `integer range` |  | Seconds for the reviewer sandbox's setup, its model proxy install among it (default: the run's --sandbox-setup-timeout) |
| `--reviewer-evidence-timeout` (0.10) | `integer range` |  | Seconds to unpack, and again to check, the workspace evidence in the reviewer sandbox (default: 1 per MiB, at least 600) |
| `--reviewer-concurrency` | `integer range` |  | Max concurrent rubric reviewers |
| `--reviewer-image` | `text` |  | Rubric reviewer container image |
| `--reviewer-agent-env` | `text (repeatable)` |  | Reviewer environment variable (KEY=VALUE) |
| `--reviewer-open-network` | flag |  | Allow unrestricted reviewer network access |
| `--reviewer-pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to read and render a submitted PDF (default: off) |
| `--reviewer-submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources (default: off) |
| `--reviewer-criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description (default: off) |

## bench eval regrade

Re-run the task's current verifier on each trial's frozen workspace.

The new score is written beside the original (regrade.json in each trial, regrade-summary.json in PATH); result.json is never changed. Trials whose workspace was not frozen (--freeze-workspace) are listed as not regradable.

```text
bench eval regrade [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A job folder, or one trial folder inside it |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-dir` | `path` |  | Folder of task folders (or one task folder) holding the changed verifiers. Default: the task path each trial recorded. |
| `--sandbox` | `text` |  | Backend for the fresh sandboxes (default: as run) |
| `--concurrency` | `integer range` | `4` | Trials regraded at once |
| `--reason` | `text` |  | Why the verifier changed; kept in each block |
| `--json` | flag |  | Print the summary as JSON |
| `--review` (0.10) | flag |  | Also rerun each trial's rubric review on its saved outputs, gated by the new test, and compare the combined score. The reviewer is the one the trial recorded, changed by --reviewer-* |
| `--reviewer-agent` (0.10) | `text` |  | Rubric reviewer harness |
| `--reviewer-model` (0.10) | `text` |  | Rubric reviewer model |
| `--reviewer-reasoning-effort` (0.10) | `text` |  | Rubric reviewer reasoning effort |
| `--reviewer-sandbox` (0.10) | `text` |  | Rubric reviewer sandbox backend |
| `--reviewer-timeout-sec` (0.10) | `integer range` |  | Reviewer execution budget in seconds |
| `--reviewer-idle-timeout` (0.10) | `text` |  | Abort a reviewer prompt after this many idle seconds (default: 600). Pass 0 or 'none' to disable the idle watchdog and leave --reviewer-timeout-sec in charge. |
| `--reviewer-storage-mb` (0.10) | `integer range` |  | Reviewer sandbox disk in MB (default: the evidence it holds plus 2048, at least 4096, at most the sandbox backend's limit) |
| `--reviewer-memory-mb` (0.10) | `integer range` |  | Reviewer sandbox memory in MB (default: 2048) |
| `--reviewer-setup-timeout` (0.10) | `integer range` |  | Seconds for the reviewer sandbox's setup, its model proxy install among it (default: the run's --sandbox-setup-timeout) |
| `--reviewer-evidence-timeout` (0.10) | `integer range` |  | Seconds to unpack, and again to check, the workspace evidence in the reviewer sandbox (default: 1 per MiB, at least 600) |
| `--reviewer-concurrency` (0.10) | `integer range` |  | Max concurrent rubric reviewers |
| `--reviewer-image` (0.10) | `text` |  | Rubric reviewer container image |
| `--reviewer-agent-env` (0.10) | `text (repeatable)` |  | Reviewer environment variable (KEY=VALUE) |
| `--reviewer-open-network` (0.10) | flag |  | Allow unrestricted reviewer network access |
| `--reviewer-pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to read and render a submitted PDF (default: off) |
| `--reviewer-submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources (default: off) |
| `--reviewer-criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description (default: off) |

## bench eval list

List the jobs in a jobs folder, one row each, with their scores.

```text
bench eval list [OPTIONS] [JOBS_DIR]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOBS_DIR` | `path` | no | Jobs directory to list (default: ./jobs) |

## bench eval metrics

Collect and display metrics from a jobs directory.

One row per trial, as bf.load_job counts them: a retried task of an Evaluation job counts once, and repeated rollouts (a bf.run_batch folder, the trial-NN folders of --matrix --trials) count as separate trials. pass@k / pass^k pool a task's trials across every job folder under the directory. Every figure counts every trial, an unscored one as a fail; with unscored trials, the scored-only figures (unscored trials left out, as BenchFlow 0.7's mean_reward was) follow with their own n. Control runs are left out of the solve rates unless --include-controls, or unless the folder holds nothing else. The score follows the scoring rule benchflow.score.v1, with a 95% interval over tasks. The trials are split by what ran: agent, harness version, served model, route, effort sent, task set and BenchFlow version. A folder of several settings gets one line per setting and no pooled score, since one number would mix them. When the pooled trials differ in agent, model or a setting bench eval compare checks, a warning names it and a table splits the trials by agent.

```text
bench eval metrics [OPTIONS] JOBS_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOBS_DIR` | `path` | yes | Jobs directory to analyze |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--benchmark` | `text` |  | Benchmark name |
| `--agent` | `text` |  | Agent name |
| `--model` | `text` |  | Model name |
| `--json` | flag |  | Output as JSON |
| `--k` | `integer (repeatable)` |  | k for pass@k and pass^k (repeatable; default 1, powers of two and multiples of five, using both trial and scored-only counts) |
| `--solve-threshold` | `number` |  | Count a scored trial as solved when reward &gt;= this value (partial credit); default: solved = passed (reward 1) |
| `--include-controls` | flag |  | Count control runs (oracle, empty/nop) in the solve rates and pass@k; a folder of only control runs counts them anyway |

## bench eval view

View a trial or session trajectory in the browser.

```text
bench eval view [OPTIONS] ROLLOUT_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `ROLLOUT_DIR` | `text` | yes | Rollout directory, job directory, trajectory JSONL file, or an hf://&lt;org&gt;/&lt;dataset&gt; trajectory dataset (optionally with /subpath) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--port` | `integer` | `8888` | Server port |
| `--confirm` | flag |  | Show an Approve & submit / Not this one bar on the page; print DECISION: approved\|rejected and exit 0 on approve, 3 on reject. |
| `--redaction-summary` | `text` |  | Masked-secret breakdown shown in the --confirm bar, e.g. '2 API keys, 1 bearer token' (lift it from `bench traj upload PATH --dry-run`). Display-only; no effect without --confirm. |

## bench eval continue

Resume a previous unfinished (timed-out) OpenHands run to completion.

```text
bench eval continue [OPTIONS] FOLDER
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `FOLDER` | `path` | yes | Original run output folder (contains config.json + trajectory/llm_trajectory.jsonl). |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-dir` | `path` |  | Directory holding the task source (instruction + verifier). Required unless the recorded task_path still exists on disk. |
| `--model` | `text` |  | Override the live-continuation model (default: the original run's model). |
| `--timeout` | `integer` |  | Wall-clock budget for the continuation, in seconds (default: the original run's timeout). |
| `--output` | `path` |  | Output jobs dir for the new run (default: &lt;orig-parent&gt;/continued). |
| `--require-timeout`, `--no-require-timeout` | flag |  | Refuse runs whose recorded status is not a timeout. |
| `--strict-divergence`, `--no-strict-divergence` | flag |  | Abort if replay leaves the original rails (message-count mismatch) instead of warning. |
| `--replay-only`, `--no-replay-only` | flag |  | Rebuild the env via replay and stop at the cut-point (no live model needed); useful for testing. |
| `--proxy-mode` | `text` | `auto` | Replay proxy placement: auto, host, or sandbox. Auto uses sandbox-local replay for Daytona/Modal and host replay for Docker. |

## bench eval branch

Fork an agent run at a checkpoint into labelled children.

Each task runs once up to the checkpoint; the sandbox (and, when asked, declared environment state) is snapshotted; every --child starts from that snapshot with a fresh agent session (child prompts must stand alone unless --resume-session reloads the parent's conversation) and is scored by the task's verifier. The job folder has the usual layout plus, per trial, tree.json and branches/&lt;fork&gt;/children/&lt;node&gt;/observation.json.

Example:

bench branch --tasks-dir tasks/my-task --agent claude-agent-acp --sandbox daytona --prompt "Write draft.txt containing: Hello world" --prompt @instruction --checkpoint-after-prompt 1 --child "label=baseline" --child "label=hint,prompt=Rename draft.txt."

```text
bench eval branch [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--tasks-dir` | `path` |  | A task directory, or a directory of tasks (run one after another). |
| `--agent` | `text` |  | Agent name, or 'oracle' (children run the task's solve.sh). Default: the checkpoint's agent with --from-checkpoint, else claude-agent-acp. |
| `--model`, `-m` | `text` |  | Model id for the agent, e.g. claude-haiku-5-5 (none for oracle and nop) |
| `--reasoning-effort` | `text` |  | Agent reasoning effort (e.g. max) |
| `--harness` | `text` |  | How the agent runs: acp (default, its ACP adapter) or native (its own CLI in headless JSON mode; claude-agent-acp and codex-acp) |
| `--sandbox` | `text` |  | docker or daytona (Daytona direct mode snapshots). Default: docker, or the checkpoint's provider. |
| `--prompt` | `text (repeatable)` |  | Parent prompt; repeatable, sent in order. '@instruction' is the task's instruction. Default: the task's prompts. |
| `--checkpoint-after-prompt` | `integer` |  | Branch after this many parent prompts (0: before the first). Default 1; 0 for the oracle and with --from-checkpoint. |
| `--child` | `text (repeatable)` |  | One child, as label=NAME[,parent=LABEL][,prompt=TEXT\|,prompt-file=PATH]; give at least two. prompt= takes the rest of the option (commas allowed). parent=LABEL forks this child from child LABEL's state after LABEL has run its prompts (a nested fork of at least two children). Without a prompt the child runs the parent's remaining prompts, or the task's prompts when the checkpoint is after the last one. |
| `--snapshot-layers` | `text` | `sandbox` | What the checkpoint captures: sandbox (container filesystem), environment (declared database state, needs a manifest), or both. |
| `--parent` | `text` |  | continue: restore the parent world after the children, send the remaining prompts and verify. discard: skip that restore (one fewer), leave the parent unverified. Default: continue, or discard with --from-checkpoint. |
| `--retain-snapshots` | flag |  | Keep the checkpoint's sandbox snapshot so a later --from-checkpoint can branch from it again (otherwise deleted). `bench sandbox cleanup` removes kept Daytona snapshots once stale. |
| `--from-checkpoint` | `path` |  | A trial folder from an earlier `bench branch --retain-snapshots` run: branch again from its kept sandbox snapshot. Needs --tasks-dir holding the same task. |
| `--checkpoint`, `--fork` | `text` |  | With --from-checkpoint: a fork id from tree.json or an automatic checkpoint as prompt:N (default: the last kept fork, else the last kept checkpoint). |
| `--checkpoints` | `text` |  | Also keep a sandbox snapshot after these parent prompts (every-prompt or prompt:N[,M]) for a later --from-checkpoint. |
| `--checkpoint-keep` | `integer range` | `3` | Keep at most this many --checkpoints per trial. |
| `--resume-session` | flag |  | Children resume the parent's agent conversation (ACP session/load) instead of starting fresh, so a child prompt can say 'continue'. Needs an agent that advertises loadSession and keeps its session on disk (Claude Code), and a checkpoint after prompt 1+. |
| `--child-retries` | `integer range` | `1` | Retry a child this many times when it failed before its agent did anything (a provider hiccup such as a connect timeout). |
| `--stop-on-child-failure` | flag |  | Stop a fork at the first child that fails for good (default: the other children still run and the fork is partial). |
| `--concurrency` | `integer range` | `1` | Run up to this many children of a fork at once, each in its own sandbox created from the snapshot (implies --isolate-children). |
| `--isolate-children` | flag |  | Run each child in its own sandbox created from the snapshot instead of restoring the parent's sandbox between children. Implied by --concurrency &gt; 1 and by nested children. |
| `--agent-env` | `text (repeatable)` |  | Agent env var (KEY=VALUE); repeatable |
| `--include` | `text (repeatable)` |  | Only these task names; repeatable |
| `--jobs-dir` | `path` | `jobs` | Output directory |
| `--job-name` | `text` |  | Job folder name (default: branch-&lt;time&gt;) |

## bench eval inspect

Read a finished job or trial: denominators, one row per trial, or --json.

The same data as bf.load_job / bf.load_trial; control runs (oracle, empty) are counted apart. Reads files only.

```text
bench eval inspect [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `text` | yes | A job folder, a trial folder, a results.jsonl file, or a glob of folders |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the benchflow.job (or benchflow.trial) JSON document |
| `--trajectories` | flag |  | Include trajectories in the JSON |
| `--verifier`, `--no-verifier` | flag | `True` | Include verifier output (test logs) in the JSON |
| `--attempts` | `text` | `best` | best (one per task, as resume picks) or all |
| `--out` | `path` |  | Write the output to this file |
| `--by` | `text` |  | Also print denominators per group, e.g. agent,model |
| `--include-controls` | flag |  | Count control runs (oracle, empty) in the denominators |
| `--csv` | `path` |  | Also write one CSV row per trial to this file |

## bench eval compare

Pair two jobs by task: rewards, deltas, fair denominators, setting checks.

The same code as bf.compare. Each paired task is checked for comparable settings (task digest, model, harness, dataset, reasoning effort, sandbox, sandbox user, timeout, agent variables, prompts); declare the settings the comparison is about with --vary. Reads files only.

```text
bench eval compare [OPTIONS] JOB_A JOB_B
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOB_A` | `text` | yes | Side A: a job folder, a trial, or a glob of folders |
| `JOB_B` | `text` | yes | Side B |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the benchflow.comparison JSON document |
| `--labels` | `<text text>` |  | Names for the two sides |
| `--vary` | `text (repeatable)` |  | A setting the comparison is about (repeatable), e.g. model |
| `--on-mismatch` | `text` | `warn` | warn (default), raise (exit 1) or ignore an undeclared setting difference |
| `--include-controls` | flag |  | Keep control runs (oracle, empty) |
| `--attempts` | `text` | `best` | best or all |
| `--out` | `path` |  | Write the output to this file |
| `--by` | `text` |  | Pair on these keys as well as the task, e.g. agent,model |
| `--k` | `integer (repeatable)` |  | k for pass@k and pass^k (repeatable) |
| `--solve-threshold` | `number` |  | Solved = reward &gt;= this value (partial credit); default: passed |

## bench eval resume

Finish an interrupted job from its folder: rerun only the unfinished tasks.

Reads the tasks directory and config the job recorded in evaluation.json (no agent_env value that looks secret, so pass those again with --agent-env), then runs the tasks with no finished result. A job that is still running is refused. Exit codes as for bench eval run; 2 when the folder is not a job, or when its trials ran with other settings or task files than this resume has (an --agent-env value, an edited task): --restart-changed runs them again.

```text
bench eval resume [OPTIONS] JOB_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOB_DIR` | `path` | yes | The job folder to finish |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--agent-env` | `text (repeatable)` |  | KEY=VALUE for the agent (repeatable); values that look secret are never stored in the job |
| `--concurrency` | `integer` |  | Override the job's concurrency |
| `--restart-changed` (0.10) | flag |  | When the job's trials ran with other settings or task files than this resume, set them aside in &lt;job&gt;/.set-aside/ and run them again (default: refuse) |

## bench eval branches

Show every fork and child of a job's (or one trial's) branches: request, status, reward, advantage, tokens, USD and sandbox-seconds.

```text
bench eval branches [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A trial folder or a job folder. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the benchflow.branch-view/1 documents (a JSON list, one per branched trial); see docs/train/environment.md. |

## bench tasks

Task authoring commands.

```text
bench tasks [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`generate`](#bench-tasks-generate) | Generate benchmark tasks from agent traces. |
| [`list-sources`](#bench-tasks-list-sources) | List known HuggingFace trace datasets. |
| [`init`](#bench-tasks-init) | Scaffold a new benchmark task. |
| [`check`](#bench-tasks-check) | Validate task structure and the run's plan without starting a sandbox. |
| [`overlap`](#bench-tasks-overlap) | Compare two task manifests for exact task-id and digest overlap. |
| [`migrate`](#bench-tasks-migrate) | Convert task.toml + instruction.md into the unified task.md format. |
| [`normalize`](#bench-tasks-normalize) | Expand minimal task.md authoring profiles into canonical task.md. |
| [`export`](#bench-tasks-export) | Export a task to a compatibility split layout with a loss report. |
| [`snapshot-hf`](#bench-tasks-snapshot-hf) | Materialize a Hugging Face dataset task tree with provenance. |
| [`digest`](#bench-tasks-digest) | Compute the content digest that pins a task's files, independent of git. |

## bench tasks generate

Generate benchmark tasks from agent traces.

Exactly one source flag (--from-local, --from-file or --from-hf) is required.

Examples:

```bash
bench tasks generate --from-local
bench tasks generate --from-local --project my-repo --limit 5
bench tasks generate --from-file session.jsonl --dry-run
bench tasks generate --from-hf opentraces-test --limit 50 --outcome success
```

```text
bench tasks generate [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--from-local` | flag |  | Generate from local Claude Code sessions (~/.claude/projects/) |
| `--from-file` | `path` |  | Generate from a JSONL trace file (Claude Code or opentraces) |
| `--from-hf` | `text` |  | Generate from a HuggingFace dataset (ID or alias; run `bench tasks list-sources` for aliases) |
| `--output` | `path` | `tasks` | Output directory for generated tasks |
| `--projects-dir` | `path` |  | Claude Code projects directory (default: ~/.claude/projects/) |
| `--project` | `text` |  | Filter local sessions by project path substring |
| `--format` | `text` | `auto` | Trace format override: auto, claude-code, opentraces, claude-messages |
| `--split` | `text` | `train` | HuggingFace dataset split |
| `--max-rows` | `integer` | `100` | Max rows to download from HuggingFace |
| `--limit` | `integer` | `20` | Max traces to process |
| `--min-steps` | `integer` | `2` | Minimum steps per trace |
| `--outcome` | `text` |  | Filter by outcome: success, failure, unknown |
| `--author` | `text` | `benchflow-traces` | Author name written into each generated task |
| `--task-format` | `text` | `task-md` | Generated task package format: task-md or legacy |
| `--dry-run` | flag |  | Preview traces without generating tasks |

## bench tasks list-sources

List known HuggingFace trace datasets.

These aliases can be passed to ``bench tasks generate --from-hf``.

```text
bench tasks list-sources [OPTIONS]
```

## bench tasks init

Scaffold a new benchmark task.

```text
bench tasks init [OPTIONS] NAME
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `NAME` | `text` | yes | Task name |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--dir` | `path` | `tasks` | Parent directory (default: tasks/) |
| `--no-pytest` | flag |  | Skip pytest template |
| `--no-oracle`, `--no-solution` | flag |  | Skip oracle template |
| `--format` | `text` | `task-md` | Task format. New tasks use task-md; legacy scaffolding is retired. |

## bench tasks check

Validate task structure and the run's plan without starting a sandbox.

Warnings (e.g. task.toml keys BenchFlow does not know) are printed but do not fail the check.

```text
bench tasks check [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory or folder of tasks |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--level` | `schema` \| `structural` \| `runtime-capability` \| `publication-grade` \| `acceptance` \| `acceptance-live` \| `equivalence` | `structural` | Validation level: schema, structural, runtime-capability, publication-grade, acceptance, acceptance-live, or equivalence (runs the oracle in --sandbox, then grades wrong and value-equivalent variants of its output) |
| `--sandbox` | `text` |  | Validate runtime semantics for docker, remote-docker, daytona, modal, apple-container, or agentcore (default: a sandbox that can run the task: docker, or modal for a task that asks for GPUs) |
| `--report-output` | `path` |  | Write the acceptance-live report to this host path instead of the task-declared report path; with --level equivalence, write the battery's JSON report here |
| `--no-report-write` | flag |  | Validate acceptance-live without writing the declared report or its .sha256 sidecar (report-only dogfood; leaves the task package unmodified). Takes precedence over --report-output. |
| `--allow-task-env` | `text (repeatable)` |  | Let tasks read this variable of your environment (a $&#123;NAME&#125; env value, or their docker-compose.yaml); repeatable. A task reading an unnamed one is refused; $&#123;NAME:-default&#125; gets the default |
| `--verifier-sandbox` | `text` |  | With separate: also check what `bench run --verifier-sandbox separate` checks before it starts (on --sandbox, default docker), with the same refusals, and name where the verifier image comes from |
| `--agent` | `text` | `claude-agent-acp` | Agent to validate, as for eval run |
| `--model` | `text` |  | Model to validate, as for eval run |
| `--sandbox-user` | `text` | `agent` | Agent user to validate, as for eval run |

## bench tasks overlap

Compare two task manifests for exact task-id and digest overlap.

```text
bench tasks overlap [OPTIONS] LEFT_MANIFEST RIGHT_MANIFEST
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `LEFT_MANIFEST` | `path` | yes | Left task manifest JSON |
| `RIGHT_MANIFEST` | `path` | yes | Right task manifest JSON |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--out`, `-o` | `path` |  | Optional JSON output path |

## bench tasks migrate

Convert task.toml + instruction.md into the unified task.md format.

```text
bench tasks migrate [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Legacy task directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--overwrite` | flag |  | Replace an existing task.md |
| `--remove-legacy` | flag |  | Delete split files and promote tests/solution aliases after task.md is verified |

## bench tasks normalize

Expand minimal task.md authoring profiles into canonical task.md.

```text
bench tasks normalize [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory with task.md |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--output`, `-o` | `path` |  | Write normalized task.md to this path instead of stdout |
| `--write` | flag |  | Replace task.md in place with the normalized canonical form |

## bench tasks export

Export a task to a compatibility split layout with a loss report.

```text
bench tasks export [OPTIONS] TASK_DIR [OUTPUT_DIR]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory to export |
| `OUTPUT_DIR` | `path` | no | Destination split-layout directory (omit with --report-only) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--target` | `text` | `task-toml` | Layout to write: task-toml |
| `--overwrite` | flag |  | Replace an existing export directory |
| `--report-only` | flag |  | Print the compatibility loss report without writing files |

## bench tasks snapshot-hf

Materialize a Hugging Face dataset task tree with provenance.

```text
bench tasks snapshot-hf [OPTIONS] REPO_ID OUTPUT_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `REPO_ID` | `text` | yes | Hugging Face dataset repo ID, e.g. org/name |
| `OUTPUT_DIR` | `path` | yes | Local output directory for the materialized snapshot |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--ref`, `--revision` | `text` |  | Dataset revision, branch, tag, or commit |
| `--path` | `text` |  | Optional subpath inside the dataset repo, e.g. tasks |
| `--cache-dir` | `path` |  | Optional Hugging Face cache directory |
| `--overwrite` | flag |  | Replace an existing output directory |
| `--include-task` | `text (repeatable)` |  | Download only this task directory; repeat for multiple tasks |

## bench tasks digest

Compute the content digest that pins a task's files, independent of git.

Matches the digests in the skillsbench dataset registry (registry.json). Given a single task directory (a legacy ``task.toml`` or a native ``task.md`` task), prints the digest; given a directory of tasks, prints one "&lt;name&gt; &lt;digest&gt;" line per task.

```text
bench tasks digest [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | Path to a task directory, or a directory of task directories |

## bench agent

Agent management commands.

```text
bench agent [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`list`](#bench-agent-list) | List all registered agents. |
| [`show`](#bench-agent-show) | Show details for a registered agent. |

## bench agent list

List all registered agents.

```text
bench agent list [OPTIONS]
```

## bench agent show

Show details for a registered agent.

```text
bench agent show [OPTIONS] NAME
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `NAME` | `text` | yes | Agent name |

## bench review

Grade finished rollouts against a rubric with a reviewer agent.

Reviews run detached from the rollouts they grade: each review is an ordinary sandboxed rollout of a throwaway wrapper task, evidence is a read-only copy, and results land in ``review_report.json``. The reviewed rollouts' rewards and ``result.json`` are never modified.

```text
bench review [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A finished rollout directory, or a job directory of rollouts |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--rubric`, `-r` | `path` |  | Rubric JSON file. Default: an admitted task copy's verifier/rubric.json (requires --tasks-root and verified digest), else the built-in default rubric. |
| `--prompt`, `-p` | `path` |  | Custom reviewer instruction template (see benchflow.review.prompts) |
| `--agent`, `-a` | `text` | `opencode` | Reviewer agent harness |
| `--model`, `-m` | `text` |  | Reviewer model (default: gemini/gemini-3.5-flash-lite with opencode, else the agent's registry default) |
| `--sandbox` | `text` | `docker` | Sandbox: docker, remote-docker, daytona, modal, apple-container, or agentcore |
| `--concurrency`, `-n` | `integer` | `4` | Max concurrent reviews |
| `--passing` | flag |  | Only review passing rollouts (reward 1.0) |
| `--failing` | flag |  | Only review failing rollouts |
| `--timeout-sec` | `integer` | `1800` | Reviewer agent timeout per rollout |
| `--agent-env` | `text (repeatable)` |  | KEY=VALUE for the reviewer (repeatable) |
| `--image` | `text` |  | Prebuilt sandbox image for reviewer rollouts |
| `--tasks-root` | `path` |  | Trusted directory holding the reviewed tasks. Required to include task definitions in evidence: a task path recorded inside a rollout is untrusted input and is never read directly. Without this, reviews use run records only. |
| `--allow-open-network` | flag |  | Run reviewers WITHOUT the no-internet declaration. Required on backends that cannot enforce network isolation (e.g. agentcore); reviews there fail closed by default. The report records the relaxed posture. |
| `--out-dir`, `-o` | `path` |  | Review output directory (default: jobs/review-&lt;ts&gt;) |
| `--pdf-tools` (unreleased) | flag |  | Install PyMuPDF beside numpy and pypdf in each reviewer sandbox, to read and render a submitted PDF |
| `--submitted-file-rule` (unreleased) | flag |  | Tell the reviewer to judge a written deliverable by the file the agent submitted: read it first, cite pages, and take a compiled PDF over its sources |
| `--criterion-evidence` (unreleased) | flag |  | Show the reviewer each criterion's Evidence line, the Evidence: sentence of its rubric description |

## bench assess (0.10)

Blind pairwise human assessments of saved outputs, kept in an opt-in sidecar.

Two completed attempts of one task, a frozen rubric and judge protocol, a recorded release of every runtime, the review collector's blind assignment and vote, then a deterministic label: 1 (A preferred), 0 (B preferred), 0.5 (tie) or withheld (abstention). Rewards, result.json, regrade.json and rollout streams are never read or changed. See docs/jobs/assessment.md.

```text
bench assess [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`attempt-ref`](#bench-assess-attempt-ref-010) (0.10) | Hash a completed rollout into a saved-attempt reference (JSON). |
| [`create`](#bench-assess-create-010) (0.10) | Start an assessment of two saved attempts of one task. |
| [`release`](#bench-assess-release-010) (0.10) | Record that every runtime of both attempts was observed released. |
| [`assign`](#bench-assess-assign-010) (0.10) | Bind the collector's persisted blind assignment (after the release). |
| [`import-vote`](#bench-assess-import-vote-010) (0.10) | Bind the collector's vote on the assignment (a skip becomes abstain). |
| [`correct`](#bench-assess-correct-010) (0.10) | Replace the effective answer with a corrected one. |
| [`retract`](#bench-assess-retract-010) (0.10) | Retract the effective answer; its history stays in the log. |
| [`reduce`](#bench-assess-reduce-010) (0.10) | Record the deterministic label the log calls for, and refresh the label file. |
| [`consume`](#bench-assess-consume-010) (0.10) | Record that a consumer used one exact ready label revision. |
| [`withdraw`](#bench-assess-withdraw-010) (0.10) | Bar a corrected or retracted label revision from future use. |
| [`show`](#bench-assess-show-010) (0.10) | Verify the sidecar's log and print its state. |
| [`verify`](#bench-assess-verify-010) (0.10) | Verify the hash chain, the label file and, optionally, the evidence. |
| [`label`](#bench-assess-label-010) (0.10) | Print the label rebuilt from the sidecar's log alone (JSON). |

## bench assess attempt-ref (0.10)

Hash a completed rollout into a saved-attempt reference (JSON).

```text
bench assess attempt-ref [OPTIONS] ROLLOUT
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `ROLLOUT` | `path` | yes | A completed rollout folder (holds result.json). |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--output` (0.10) | `path` |  | The output to compare: a folder or one file. Required. |
| `--review-artifact` (0.10) | `path` |  | The file a reviewer sees (e.g. a preview image). Required. |
| `--sandbox-provider` (0.10) | `text` |  | Provider of result.json's sandbox_id (daytona, docker, modal, …). |
| `--sandbox-role` (0.10) | `text` | `solver` | solver, verifier or shared (one sandbox for both). |
| `--runtime` (0.10) | `text (repeatable)` |  | Another allocation of this attempt: PROVIDER:RUNTIME_ID[:ROLE]. Repeatable. |
| `--media-type` (0.10) | `text` |  | The review artifact's media type. |
| `--out` (0.10) | `path` |  | Write the reference here (a new file) instead of stdout. |

## bench assess create (0.10)

Start an assessment of two saved attempts of one task.

```text
bench assess create [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--attempt` (0.10) | `path (repeatable)` |  | A saved-attempt reference file. Give exactly two. Required. |
| `--rubric` (0.10) | `path` |  | The frozen rubric file. Required. |
| `--rubric-id` (0.10) | `text` |  | The rubric's id. Required. |
| `--rubric-version` (0.10) | `text` |  | The rubric's frozen version. Required. |
| `--protocol` (0.10) | `path` |  | The frozen judge protocol (benchflow.assessment-judge-protocol.v1). Required. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess release (0.10)

Record that every runtime of both attempts was observed released.

```text
bench assess release [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--proof` (0.10) | `path` |  | Observations that every runtime is gone (benchflow.assessment-release-proof.v1). Required. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess assign (0.10)

Bind the collector's persisted blind assignment (after the release).

```text
bench assess assign [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--collector-db` (0.10) | `path` |  | The collector's workspace.sqlite3 (a `cli.py backup` copy), read-only. Required. |
| `--collector-code` (0.10) | `path` |  | The collector's code folder (design-bench/prototype). Required. |
| `--assignment` (0.10) | `text` |  | The collector's assignment id. Required. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess import-vote (0.10)

Bind the collector's vote on the assignment (a skip becomes abstain).

```text
bench assess import-vote [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--collector-db` (0.10) | `path` |  | The collector's workspace.sqlite3, read-only. Required. |
| `--collector-code` (0.10) | `path` |  | The collector's code folder. Required. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess correct (0.10)

Replace the effective answer with a corrected one.

```text
bench assess correct [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--supersedes` (0.10) | `text` |  | The current effective answer's event id. Required. |
| `--choice` (0.10) | `text` |  | left (A), right (B), tie or abstain. Required. |
| `--reason` (0.10) | `text` |  | The new answer's rationale. Required. |
| `--correction-reason` (0.10) | `text` |  | Why the answer changed. Required. |
| `--actor` (0.10) | `text` |  | Who records this (the reviewer's collector name, or an operator). Required. |
| `--actor-role` (0.10) | `text` | `reviewer` | reviewer (the answer's own) or operator. |
| `--tag` (0.10) | `text (repeatable)` |  | A feedback tag. Repeatable. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess retract (0.10)

Retract the effective answer; its history stays in the log.

```text
bench assess retract [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--target` (0.10) | `text` |  | The current effective answer's event id. Required. |
| `--reason` (0.10) | `text` |  | Why it is retracted. Required. |
| `--actor` (0.10) | `text` |  | Who records this (the reviewer's collector name, or an operator). Required. |
| `--actor-role` (0.10) | `text` | `reviewer` | reviewer (the answer's own) or operator. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess reduce (0.10)

Record the deterministic label the log calls for, and refresh the label file.

```text
bench assess reduce [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess consume (0.10)

Record that a consumer used one exact ready label revision.

```text
bench assess consume [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--revision` (0.10) | `integer` |  | The label revision used. Required. |
| `--consumer` (0.10) | `text` |  | The consumer's id. Required. |
| `--receipt` (0.10) | `text` |  | The consumer's immutable receipt (e.g. a ledger row). Required. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess withdraw (0.10)

Bar a corrected or retracted label revision from future use.

```text
bench assess withdraw [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--revision` (0.10) | `integer` |  | The superseded label revision. Required. |
| `--reason` (0.10) | `text` |  | Why it is withdrawn. Required. |
| `--disposition` (0.10) | `text` |  | not_applicable, queued_compensation or compensated: what its consumers do. Required. |
| `--actor` (0.10) | `text` |  | Who records this (the reviewer's collector name, or an operator). Required. |
| `--actor-role` (0.10) | `text` | `operator` | operator or reducer. |
| `--idempotency-key` (0.10) | `text` |  | Key for this write (default: derived from its inputs). A retry with the same key and payload returns the original event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess show (0.10)

Verify the sidecar's log and print its state.

```text
bench assess show [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--events` (0.10) | flag |  | List every event. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess verify (0.10)

Verify the hash chain, the label file and, optionally, the evidence.

```text
bench assess verify [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--evidence` (0.10) | flag |  | Also hash both attempts, the rubric and the protocol again. |
| `--expect-event` (0.10) | `text` |  | An event SHA-256 a receipt holds; the log must still contain it. |
| `--json` (0.10) | flag |  | Print JSON instead of one line. |

## bench assess label (0.10)

Print the label rebuilt from the sidecar's log alone (JSON).

```text
bench assess label [OPTIONS] SIDECAR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SIDECAR` | `path` | yes | The assessment's sidecar directory, e.g. &lt;completed-rollout&gt;/assessment. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--write` (0.10) | flag |  | Rewrite the label file from the log. |
| `--check` (0.10) | flag |  | Exit 1 unless the label file equals the log's label. |

## bench preferences (0.10)

Human preferences and rubric scores on finished trials.

`pairs` makes blind pairs of trial outputs (same task version, different agent or model) for a rater's tool, `import` takes the votes and rubric scores it collected, and `rank` gives Bradley-Terry ratings on the Elo scale with rater checks. Votes are never rewards: no trial file is changed. See docs/tasks/human-preferences.md.

```text
bench preferences [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`pairs`](#bench-preferences-pairs-010) (0.10) | Make the store and the blind assignment. |
| [`import`](#bench-preferences-import-010) (0.10) | Check every vote in the file, then add the new ones to the store. |
| [`rank`](#bench-preferences-rank-010) (0.10) | Ratings from the store's votes (also written to ratings.v1.json). |

## bench preferences pairs (0.10)

Make the store and the blind assignment.

```text
bench preferences pairs [OPTIONS] [JOBS]...
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOBS` | `path` | no | Job or trial folders whose outputs to pair (none with only --items). |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--store` (0.10) | `path` |  | A new folder for the preference store. Required. |
| `--out` (0.10) | `path` |  | The file to write the blind pairs to, for the raters' tool. Required. |
| `--output` (0.10) | `text` | `artifacts` | Which folder of each trial the raters compare, relative to the trial (artifacts, or evidence/workspace with --freeze-workspace). |
| `--per-task` (0.10) | `integer` |  | At most this many pairs per task. |
| `--seed` (0.10) | `integer` |  | Seed for sampling and sides (default: random, recorded). |
| `--rubric` (0.10) | `path` |  | A rubric JSON file to freeze for aspects and scores. |
| `--bundle` (0.10) | `path` |  | Where to copy the outputs under their opaque ids for the rating tool (default: &lt;store&gt;-outputs beside the store). |
| `--keep-sides` (0.10) | flag |  | Do not shuffle which output is shown on the left. |
| `--items` (0.10) | `path` |  | A JSON-lines file of outputs made outside BenchFlow (a golden, a remix), each with its task, output folder and arm labels; they pair with the trials of their task's version. |
| `--condition` (0.10) | `text` |  | A condition to label every trial's output with (one-shot, say); --items rows carry their own. |
| `--entry` (0.10) | `text` |  | The page to open in each output, such as index.html; an output without it is left out. |
| `--golden` (0.10) | `text` | `side` | Whether a golden from --items is a side in pairs (side) or only the bar a rater's tool shows (bar). |

## bench preferences import (0.10)

Check every vote in the file, then add the new ones to the store.

```text
bench preferences import [OPTIONS] STORE
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `STORE` | `path` | yes | The preference store `bench preferences pairs` made. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--votes` (0.10) | `path` |  | Votes as JSON lines or CSV, or the FDB vote site's export (see the docs for the fields). Required. |
| `--source` (0.10) | `text` |  | The tool that collected them, recorded on each vote. |
| `--skip-unknown` (0.10) | flag |  | Leave out votes on outputs the store doesn't have, instead of refusing the file. |
| `--dry-run` (0.10) | flag |  | Check every vote and say what would change; write nothing. |

## bench preferences rank (0.10)

Ratings from the store's votes (also written to ratings.v1.json).

```text
bench preferences rank [OPTIONS] STORE
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `STORE` | `path` | yes | The preference store `bench preferences pairs` made. |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--by` (0.10) | `text` | `agent+model` | Rate each agent, model, agent+model or condition. |
| `--bootstrap` (0.10) | `integer` | `1000` | Resamples for the 95% intervals (0: none). |
| `--seed` (0.10) | `integer` | `0` | Seed for the resampling. |
| `--min-agreement` (0.10) | `number` |  | Raw agreement (not kappa) to check the raters against (0 to 1); rank says whether it is met. |
| `--raters-per-pair` (0.10) | `integer` |  | Raters each pair should have. |
| `--raters` (0.10) | `text` |  | Whose votes to rate together: person, guest, model, or several joined by commas (default: people; the others get tables of their own). |
| `--cluster` (0.10) | `text` | `auto` | What the intervals resample: vote, rater, task or rater+task (auto: raters and tasks, when there are several of each). |
| `--provisional` (0.10) | flag |  | Show the ratings before every pair has a vote. |
| `--json` (0.10) | flag |  | Print JSON. |

## bench bundle (0.10)

Write one zip of a run's evidence to share, redacted.

Per trial the result, trajectory, verifier output, reviews, regrades, artifacts and saved workspace; for a job its run card, summary and report. Every text file is redacted as a trajectory upload is. bundle.json lists each file with its SHA-256 and what was masked, and what was left out.

```text
bench bundle [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | A job folder, or one trial folder inside it |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--out`, `-o` (0.10) | `path` |  | Where to write the zip (default: &lt;PATH&gt;.bundle.zip) |
| `--workspace`, `--no-workspace` (0.10) | flag | `True` | Include each trial's saved workspace (evidence/) when it has one |

## bench skills

Skill discovery and evaluation.

```text
bench skills [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`list`](#bench-skills-list) | List discovered skills. |
| [`eval`](#bench-skills-eval) | Evaluate a skill using its evals/evals.json test cases. |
| [`validate`](#bench-skills-validate) | Run static checks on skills: NVIDIA SkillEvaluator Tier 1 and SkillSpector. |

## bench skills list

List discovered skills.

```text
bench skills list [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--dir` | `path` |  | Skills directory to scan |

## bench skills eval

Evaluate a skill using its evals/evals.json test cases.

Generates ephemeral tasks from the skill's eval dataset, runs each agent with and without the skill installed, and reports the lift.

Examples:

```bash
bench skills eval ./my-skill/ --agent claude-agent-acp
bench skills eval ./my-skill/ --agent claude-agent-acp --agent codex-acp --sandbox daytona --concurrency 4
bench skills eval ./my-skill/ --agent claude-agent-acp --no-baseline --export-gepa
```

```text
bench skills eval [OPTIONS] SKILL_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SKILL_DIR` | `path` | yes | Path to skill directory containing evals/evals.json |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--agent` | `text (repeatable)` |  | Agent(s) to evaluate (repeatable) |
| `--model` | `text (repeatable)` |  | Model(s) (matched 1:1 with agents) |
| `--sandbox` | `text` | `docker` | Sandbox: docker, remote-docker, daytona, modal, apple-container, or agentcore |
| `--concurrency` | `integer` | `1` | Max concurrent tasks |
| `--jobs-dir` | `text` | `jobs` | Output directory for results |
| `--no-baseline` | flag |  | Skip baseline (without-skill) runs |
| `--export-gepa` | flag |  | Export GEPA-compatible traces |

## bench skills validate

Run static checks on skills: NVIDIA SkillEvaluator Tier 1 and SkillSpector.

SkillEvaluator Tier 1 checks schema, version, PII, license, code integrity, dependencies, Unicode smuggling, quality score and script lint. SkillSpector scans for security issues. Both run without LLM calls; install them separately (see docs/judging/skill-evals.md). Exit 1 when a schema, security or dependency finding reaches --fail-on, or a checker fails to run.

Examples:

```bash
bench skills validate ./my-skill
bench skills validate tasks/ --json > skill-checks.json
bench skills validate tasks/citation-check --fail-on critical -o reports
```

```text
bench skills validate [OPTIONS] PATHS...
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATHS` | `path` | yes | Skill directories, or directories searched recursively for SKILL.md (e.g. a SkillsBench tasks/ folder) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--no-security` | flag |  | Skip the NVIDIA SkillSpector security scan |
| `--fail-on` | `text` | `high` | Exit 1 when a schema, security or dependency finding is at or above this severity: critical, high, medium, low, or never |
| `--checks` | `text` |  | Comma-separated SkillEvaluator Tier 1 checks (default: schema,version,pii,license,code-integrity,dependency,unicode,quality,lint) |
| `--min-score` | `integer range` |  | Quality score SkillEvaluator's quality check needs (default 70); advisory: changes SkillEvaluator's Tier 1 status, not the exit code |
| `--output-dir`, `-o` | `path` |  | Keep each skill's raw JSON reports under DIR/&lt;skill&gt;/ |
| `--workers`, `-j` | `integer range` | `4` | Skills checked in parallel |
| `--max-findings` | `integer range` | `5` | Blocking findings listed per skill (all are in --json) |
| `--json` | flag |  | Print machine-readable JSON |

## bench traj

Trajectory commands.

```text
bench traj [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`setup`](#bench-traj-setup) | Install the submit skill, or print the prompt to send to an agent. |
| [`upload`](#bench-traj-upload) | Inspect, redact, confirm, and upload trajectory JSONL. |
| [`status`](#bench-traj-status) | Check whether an uploaded trajectory is verified in storage. |

## bench traj setup

Install the submit skill, or print the prompt to send to an agent.

```text
bench traj setup [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--yes`, `-y` | flag |  | Install the skill without prompts |
| `--prompt` | flag |  | Print the copy-paste agent prompt and exit |
| `--list` | flag |  | List recent local sessions and exit |

## bench traj upload

Inspect, redact, confirm, and upload trajectory JSONL.

```text
bench traj upload [OPTIONS] [PATH]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | no | Trajectory JSONL file, directory, or trial directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--github-id` | `text` |  | Contributor GitHub username (inferred from gh/git when omitted) |
| `--email` | `text` |  | Contributor email stored in the manifest (inferred from git when omitted) |
| `--source-id` | `text` |  | Stable contributor source identifier |
| `--repo`, `--no-repo` | flag | `True` | Tag the upload with the session's repository (owner/name from its git remote) as the source id |
| `--workspace`, `--no-workspace` | flag | `True` | Attach an explicitly selected workspace folder as a zip (use --workspace-dir or select a folder at the prompt; skipped over 1 GiB) |
| `--workspace-dir` | `path` |  | Workspace folder to attach instead of auto-detection |
| `--direct` | flag |  | Upload with local Azure credentials |
| `--container-url` | `text` |  | Azure container URL for --direct |
| `--dry-run` | flag |  | Validate and stage without uploading |
| `--preview-steps` | `integer range` | `5` | Number of redacted trajectory steps to preview |
| `--wait`, `--no-wait` | flag | `True` | After uploading, wait until the capture is verified in storage (BENCHFLOW_TRAJ_WAIT_SECONDS overrides the budget) |

## bench traj status

Check whether an uploaded trajectory is verified in storage.

```text
bench traj status [OPTIONS] [DIGEST]
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `DIGEST` | `text` | no | Capture digest printed by bench traj upload (sha256:…) |

## bench train

Training data commands.

```text
bench train [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`convert`](#bench-train-convert) | Convert BenchFlow rollout artifacts into trainer-ready data. |
| [`token-coverage`](#bench-train-token-coverage) | Report whether each rollout's model calls carry token ids and logprobs. |
| [`stream`](#bench-train-stream) | Print each finished rollout of a (running) job as one JSON line. |
| [`validate`](#bench-train-validate) | Validate trainer-ready data. |
| [`run`](#bench-train-run) | Launch training jobs. |

## bench train convert

Convert BenchFlow rollout artifacts into trainer-ready data.

```text
bench train convert [OPTIONS] JOBS_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOBS_DIR` | `path` | yes | BenchFlow rollout or jobs directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--out`, `-o` | `path` |  | Output path: JSONL, or parquet for verl-sft Required. |
| `--format` | `text` | `prime-sft` | Trainer format: prime-sft, trl-sft, branch-tree (one row per branch child: shared prefix, continuation, reward, advantage), verl-sft (parquet for verl's multi-turn SFT trainer), skyrl-sft (rows for SkyRL's SFT trainer) or openenv-episodes (OpenEnv episode records). 0.9.2 writes prime-sft, trl-sft and branch-tree only. |
| `--pairs` | `path` |  | branch-tree only: also write one row per sibling pair whose rewards differ (chosen/rejected, margin) to this JSONL path |
| `--pairs-any-request` | flag |  | branch-tree only: also pair siblings that were asked different things (default: only siblings with the same first user message) |
| `--include-oracle` | flag |  | branch-tree only: keep oracle children as rows (skipped by default; their only message is the solution script's exit code) |
| `--min-reward` | `number` |  | Only include rows with reward &gt;= value |
| `--row-mode` | `rollout` \| `exchange` | `rollout` | rollout writes one row per rollout; exchange writes one row per LLM exchange |
| `--manifest` | `path` |  | Optional conversion stats JSON path |
| `--expected-rows` | `integer` |  | Fail (before writing the output file) unless exactly this many rows would be exported |
| `--canonical-selection` | `path` |  | Restrict conversion to rows selected by canonical-selection.json |
| `--redact`, `--no-redact` | flag | `True` | Redact secrets while exporting trainer JSONL. Use --no-redact only for private local SFT data when preserving exact tool-call argument tokens is required. |
| `--context-policy` | `full` \| `message-window` | `full` | TRL context handling: full or tokenizer-aware message-window |
| `--tokenizer` | `text` |  | Tokenizer/model ID for TRL message-window conversion |
| `--tokenizer-revision` | `text` |  | Immutable tokenizer revision for TRL conversion |
| `--max-length` | `integer` |  | Maximum rendered length for TRL message-window conversion |
| `--subagent-rows` | flag |  | Also write each Claude subagent conversation as separate rows tagged with the spawning tool call. By default rows contain the parent agent's conversation only. |
| `--reward-vector` | flag |  | prime-sft/trl-sft: add reward_vector to every row: the test gate and each rubric criterion (names, kinds, weights, 0-1 values) for rubric-scored trials, else every numeric key the verifier wrote. Unscored rollouts get null. |
| `--group-advantage` | `text` |  | prime-sft/trl-sft: add advantage and group to every row, normalised over the scored rollouts of the same group: grpo = (r - mean) / (std + 1e-4), sample std; loo = r - mean of the others. Unscored rollouts get null, never 0. |
| `--group-by` | `text` | `task,agent,model,job` | Grouping key for --group-advantage, comma-separated from task, agent, model, task_digest, job. A group stays inside one job unless job is left out (jobs may have run different policy checkpoints) |

## bench train token-coverage

Report whether each rollout's model calls carry token ids and logprobs.

Reads trajectory/llm_trajectory.jsonl token_capture blocks (see docs/train/trainer.md#token-ids-and-logprobs). A rollout is training-grade when every call has prompt token ids, sampled token ids and logprobs, and each prompt extends the previous call's prompt and sampled tokens.

```text
bench train token-coverage [OPTIONS] PATH
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `PATH` | `path` | yes | BenchFlow rollout or jobs directory |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--json` | flag |  | Print the report as JSON |

## bench train stream

Print each finished rollout of a (running) job as one JSON line.

For a trainer that consumes a job while it runs: every line is a benchflow.rollout-stream.v1 record with the reward, group id and, when the gateway captured them, per-call token ids and logprobs (docs/train/trainer.md#stream-rollouts-while-a-job-runs). Status goes to stderr. Exit 0 when the job finished, 1 when it was stopped or its process died first, 2 on bad input, 3 on --timeout.

```text
bench train stream [OPTIONS] JOB_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOB_DIR` | `path` | yes | Job folder (or a --jobs-dir holding one job); may not exist yet |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--format` | `text` | `jsonl` | Output format: jsonl |
| `--follow`, `--no-follow` | flag | `True` | Keep polling until the job finishes (default), or scan once |
| `--poll-interval` | `number` | `1.0` | Seconds between scans |
| `--timeout` | `number` |  | Stop after this many seconds (exit 3) |
| `--group-size` | `integer` |  | Hold rollouts until N of a group finished, then emit them with a GRPO advantage |
| `--group-by` | `text` |  | Grouping key, comma-separated from task, agent, model, task_digest, job (default task,agent,model,job) |
| `--withheld` (0.10) | `path` |  | --group-size: write the records of groups withheld from the trainer (a member unscored or without the others' token sequence, or the group still short at the end) to this JSONL file, each with its reason |

## bench train validate

Validate trainer-ready data.

```text
bench train validate [OPTIONS] JSONL
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JSONL` | `path` | yes | Trainer JSONL path to validate |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--format` | `text` | `prime-sft` | Trainer format. 0.9.2 and 0.10.0rc2 check prime-sft, trl-sft and branch-tree rows only. |
| `--expected-rows` | `integer` |  | Fail unless this many rows are present |
| `--source-jobs` | `path` |  | Source BenchFlow jobs dir to audit |
| `--source-canonical-selection` | `path` |  | Canonical selection JSON used for this trainer data |
| `--task-manifest` | `path` |  | Task manifest for source rows |
| `--require-llm-trajectory` | flag |  | Fail unless source selected rows have valid llm_trajectory.jsonl |
| `--require-tool-calls` | flag |  | Fail unless trainer rows and source rows include tool calls |
| `--tokenizer` | `text` |  | Tokenizer/model ID for TRL render and assistant-mask validation |
| `--tokenizer-revision` | `text` |  | Immutable tokenizer revision for TRL validation |
| `--max-length` | `integer` |  | Fail TRL validation when a rendered row exceeds this length |

## bench train run

Launch training jobs.

```text
bench train run [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`sft`](#bench-train-run-sft) | Run a Prime-RL SFT job and record a BenchFlow manifest. |

## bench train run sft

Run a Prime-RL SFT job and record a BenchFlow manifest.

```text
bench train run sft [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--config` | `path` |  | Prime-RL SFT TOML config Required. |
| `--backend` | `prime-rl` | `prime-rl` | Training backend |
| `--data` | `text` |  | Optional dataset override passed to Prime-RL as --data.name |
| `--output-dir` | `path` |  | Prime-RL trainer output dir. Defaults to &lt;work-dir&gt;/prime-rl-output. |
| `--compat-profile` | `text` |  | Named BenchFlow Prime-RL SFT compatibility profile. Currently supports env0-mobile300-pr828. |
| `--work-dir` | `path` | `train-runs/sft` | BenchFlow training run directory |
| `--prime-rl-dir` | `path` |  | Prime-RL checkout to run uv from. Defaults to the current directory. |
| `--dry-run` | flag |  | Pass --dry-run through to Prime-RL |
| `--follow` | flag |  | Stream trainer stdout while writing logs |
| `--uv-no-sync` | flag |  | Run Prime-RL with `uv run --no-sync`, useful after backend post-install steps such as flash-attn. |
| `--override` | `text (repeatable)` |  | Prime-RL config override as KEY=VALUE; repeatable |
| `--target-examples` | `integer` |  | Derive Prime-RL max_steps from a target number of training examples using data.batch_size, rounding up |
| `--target-micro-steps` | `integer` |  | Derive Prime-RL max_steps from custom-trainer batch-size-1 microsteps, dropping the final partial accumulation |
| `--sync-scheduler-to-max-steps`, `--no-sync-scheduler-to-max-steps` | flag | `True` | When --target-examples or --target-micro-steps is set, also derive scheduler.decay_steps from the computed max_steps |
| `--sync-ckpt-to-max-steps`, `--no-sync-ckpt-to-max-steps` | flag |  | When deriving max_steps, also derive ckpt.interval and ckpt.keep_interval from the computed max_steps |
| `--pack-function` | `text` |  | Optional first-class Prime-RL data.pack_function override: cat or stack |
| `--loss-mask` | `text` |  | Optional first-class Prime-RL data.loss_mask override: 'assistant', 'all', or comma-separated roles |
| `--loss-normalization` | `text` |  | Prime-RL SFT loss normalization: token_mean for native Prime-RL behavior, or sample_mean to match the historical custom trainer's per-row mean loss |
| `--model-attn` | `text` |  | Optional first-class Prime-RL model.attn override, e.g. sdpa |
| `--renderer-mode` | `text` |  | Optional Prime-RL renderer mode override. Use 'none' to fall back to tokenizer.apply_chat_template tokenization. |
| `--tool-defs-mode` | `text` | `preserve` | How local training JSONL exposes tool schemas to Prime-RL: preserve or omit |
| `--chat-template-kwarg` | `text (repeatable)` |  | Apply KEY=VALUE to every local Prime-SFT row's chat_template_kwargs before Prime-RL loads it; repeatable. Values are parsed as JSON literals when possible. |
| `--message-tail-truncation` | `text` | `off` | Local Prime-SFT row truncation before Prime-RL tokenizes it: off, keep-first-user, or custom-trainer-pretokenized. The keep-first-user mode keeps the initial user instruction plus the longest final message suffix that fits data.seq_len * data.micro_batch_size. The custom-trainer-pretokenized mode renders rows like the historical custom trainer, keeps the exact token tail, and stages shifted input_ids/target_ids/loss_mask tensors. |
| `--allow-unsafe-stack-flash-attn` | flag |  | Allow Qwen3.5 stack packing with flash attention despite known Prime-RL varlen-kernel risk |
| `--force` | flag |  | Overwrite an existing &lt;work-dir&gt;/train-run.json manifest |
| `--publish-model` | `text` |  | Upload trainer output to this HF model repo |
| `--model-tag` | `text` |  | Path prefix/tag for --publish-model upload |
| `--model-card` | `text` |  | Model card mode; currently accepts 'auto' |
| `--publish-artifacts` | `text` |  | Upload BenchFlow train run artifacts to this HF dataset repo |
| `--hf-prefix` | `text` |  | Path prefix for --publish-artifacts |
| `--hf-public-read-check` | flag |  | Verify public HF reads after upload |

## bench hub

Environment hubs: browse a hub's environments (list/show/inspect), check registry tasks (check), and publish datasets and push runs to a BenchFlow Hub (publish/push).

```text
bench hub [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`list`](#bench-hub-list) | List a hub's datasets or environments (benchflow or primeintellect). |
| [`show`](#bench-hub-show) | Show a dataset's tasks, or a hosted environment's metadata. |
| [`inspect`](#bench-hub-inspect) | Inspect a file from a hub environment package. |
| [`publish`](#bench-hub-publish-010) (0.10) | Publish a folder of tasks as an immutable dataset version on a BenchFlow Hub. |
| [`push`](#bench-hub-push-010) (0.10) | Push a finished run (traces, results, an embodied trial's record and video) to a BenchFlow Hub. |
| [`check`](#bench-hub-check) | Inventory or structurally check representative registry tasks. |

## bench hub list

List a hub's datasets or environments (benchflow or primeintellect).

```text
bench hub list [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--provider` | `text` | `benchflow` | Hub to browse: 'benchflow' (BenchFlow's dataset registry) or 'primeintellect' (hosted envs; needs the `prime` CLI) |
| `--owner` | `text` |  | Owner/namespace filter (primeintellect) |
| `--search` | `text` |  | Search query |
| `--limit` | `integer` |  | Maximum results |
| `--json` | flag |  | Emit raw JSON |

## bench hub show

Show a dataset's tasks, or a hosted environment's metadata.

```text
bench hub show [OPTIONS] SOURCE_ENV
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SOURCE_ENV` | `text` | yes | A dataset on a BenchFlow Hub as org/name@version (e.g. robouse/metaworld-sample@0.6.1; read from --hub, or from $BENCHFLOW_HUB_URL when no --provider is given), a dataset of BenchFlow's registry as name@version (e.g. terminal-bench@2.0), or a hosted environment (e.g. primeintellect/general-agent) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--version` | `text` |  | Version (of the dataset or hosted environment) |
| `--provider` | `text` |  | Where to read it without a Hub: 'benchflow' (BenchFlow's registry; lists the dataset's tasks) or 'primeintellect' (default: benchflow for name@version, primeintellect for owner/name) |
| `--json` | flag |  | Print the dataset's entry as JSON (for a Hub version, the version and its controls) |
| `--hub` (0.10) | `text` |  | A BenchFlow Hub to read org/name@version from, with its control results (default: $BENCHFLOW_HUB_URL when set) |

## bench hub inspect

Inspect a file from a hub environment package.

```text
bench hub inspect [OPTIONS] SOURCE_ENV
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `SOURCE_ENV` | `text` | yes | Hosted environment (e.g. primeintellect/general-agent) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--version` | `text` |  | Hosted environment version |
| `--path` | `text` | `README.md` | File inside the hosted environment package |

## bench hub publish (0.10)

Publish a folder of tasks as an immutable dataset version on a BenchFlow Hub.

```text
bench hub publish [OPTIONS] TASKS_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASKS_DIR` | `path` | yes | Folder holding one folder per task |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--name` (0.10) | `text` |  | The version to publish, as org/name@version (e.g. robouse/metaworld-sample@0.6.1) Required. |
| `--hub` (0.10) | `text` |  | Hub address (default: $BENCHFLOW_HUB_URL) |
| `--description` (0.10) | `text` |  | One line on what it is |
| `--license` (0.10) | `text` |  | The tasks' license, as an SPDX expression (e.g. Apache-2.0 AND MIT) |
| `--kind` (0.10) | `text` |  | What its tasks are, e.g. embodied |
| `--source-repo` (0.10) | `text` |  | Where the tasks came from (a repository URL) |
| `--source-commit` (0.10) | `text` |  | The commit they came from |
| `--exporter` (0.10) | `text` |  | What wrote the packages, e.g. 'robouse-export@d05027fb' (the tool and its commit) |

## bench hub push (0.10)

Push a finished run (traces, results, an embodied trial's record and video) to a BenchFlow Hub.

```text
bench hub push [OPTIONS] JOB_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `JOB_DIR` | `path` | yes | A job folder (jobs/&lt;job&gt;) holding one finished run, or the run (jobs/&lt;job&gt;/&lt;time&gt;) |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--hub` (0.10) | `text` |  | Hub address (default: $BENCHFLOW_HUB_URL) |
| `--dataset` (0.10) | `text` |  | The dataset version the job ran, org/name@version (default: what the job recorded) |
| `--label` (0.10) | `text` |  | A label for the run on the Hub |

## bench hub check

Inventory or structurally check representative registry tasks.

```text
bench hub check [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--registry` | `text` |  | Registry JSON URL or local file (default: BenchFlow's registry). |
| `--tasks-per-dataset` | `integer range` | `2` | Number of representative tasks to select per registry dataset. |
| `--level` | `text` | `inventory` | Compatibility level to run: inventory or check. |
| `--out` | `path` |  | Optional JSONL output path. |
| `--cache-dir` | `path` | `.cache/hub/registry` | Cache directory for sparse clones. |
| `--limit` | `integer range` |  | Optional cap on selected task refs. |

## bench daemon

Manage the opt-in warm local runtime.

```text
bench daemon [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`start`](#bench-daemon-start) | Start a private runtime; attach runs with BENCHFLOW_DAEMON=auto. |
| [`status`](#bench-daemon-status) | Show whether this user's local runtime is running. |
| [`stop`](#bench-daemon-stop) | Stop an idle local runtime; never interrupt an active call. |

## bench daemon start

Start a private runtime; attach runs with BENCHFLOW_DAEMON=auto.

```text
bench daemon start [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--prepare` | `integer range` | `0` | Prepare up to N sandboxes ahead: after a one-task Docker run ends, the same command is made ready for its next identical call (0: off). |
| `--prepare-ttl` | `float range` | `300.0` | Seconds a prepared sandbox waits for its call before it is removed. |

## bench daemon status

Show whether this user's local runtime is running.

```text
bench daemon status [OPTIONS]
```

## bench daemon stop

Stop an idle local runtime; never interrupt an active call.

```text
bench daemon stop [OPTIONS]
```

## bench sandbox

Daytona sandboxes: create, list and clean up (cleanup also removes leftover Docker containers).

```text
bench sandbox [OPTIONS] COMMAND [ARGS]...
```

| Command | What it does |
|---|---|
| [`create`](#bench-sandbox-create) | Create an environment from a task directory (does not start it). |
| [`list`](#bench-sandbox-list) | List active Daytona sandboxes (kept Docker sandboxes: their trial folders hold kept-sandbox.json). |
| [`attach`](#bench-sandbox-attach-010) (0.10) | Open a root shell in a kept sandbox; with a command after --, run it there and exit with its status. |
| [`release`](#bench-sandbox-release-010) (0.10) | Delete a sandbox kept with --keep-sandbox. |
| [`cleanup`](#bench-sandbox-cleanup) | Clean up orphaned sandboxes and stale shared runtimes. |

## bench sandbox create

Create an environment from a task directory (does not start it).

```text
bench sandbox create [OPTIONS] TASK_DIR
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TASK_DIR` | `path` | yes | Task directory with task.md or task.toml + Dockerfile |

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--sandbox` | `text` | `daytona` | Sandbox: docker, remote-docker, daytona, modal, apple-container, or agentcore |

## bench sandbox list

List active Daytona sandboxes (kept Docker sandboxes: their trial folders hold kept-sandbox.json).

```text
bench sandbox list [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--all` | flag |  | Every sandbox on the account, not only BENCHFLOW_DAYTONA_OWNER's |
| `--json` | flag |  | Print the sandboxes as JSON |

## bench sandbox attach (0.10)

Open a root shell in a kept sandbox; with a command after --, run it there and exit with its status.

```text
bench sandbox attach [OPTIONS] TRIAL
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TRIAL` | `path` | yes | The trial folder of a run with --keep-sandbox, or its kept-sandbox.json |

## bench sandbox release (0.10)

Delete a sandbox kept with --keep-sandbox.

```text
bench sandbox release [OPTIONS] TRIAL
```

| Argument | Type | Required | Help |
|---|---|---|---|
| `TRIAL` | `path` | yes | The trial folder of a run with --keep-sandbox, or its kept-sandbox.json |

## bench sandbox cleanup

Clean up orphaned sandboxes and stale shared runtimes.

```text
bench sandbox cleanup [OPTIONS]
```

**Options**

| Option | Type | Default | Help |
|---|---|---|---|
| `--dry-run` | flag |  | List sandboxes without deleting |
| `--max-age` | `integer range` | `1440` | Delete sandboxes older than N minutes |
| `--all` | flag |  | Delete every sandbox of BENCHFLOW_DAYTONA_OWNER, whatever its age or activity, except a STARTED one or one a live run still holds (a CI teardown step); others are never touched |
| `--id` | `text (repeatable)` |  | Delete this Daytona sandbox (repeatable), even when a live run lease holds it; only sandboxes of your owner scope |

## Commands hidden from `bench --help`

`bench --help` doesn't list these. They are internal, or older spellings kept so old scripts still run, and may change or go away in any release; each prints its own `--help`.

| Command | What it does |
|---|---|
| `bench eval create` | The same as `bench eval run`. Run an evaluation; single task or batch. |
| `bench eval continue-batch` | Continue all timed-out OpenHands runs under a directory tree. |
| `bench agent create` | Scaffold benchmarks/&lt;name&gt;/ for a new benchmark adoption (deprecated alias). |
| `bench agent run` | Drive the conversion workflow by launching the host codex CLI (deprecated alias). |
| `bench agent verify` | Run the parity gate for an adopted benchmark; emit a verdict (deprecated alias). |
| `bench hub env` | Alias of `bench hub`: use `bench hub list/show/inspect`. |
| `bench hub env list` | List a hub's datasets or environments (benchflow or primeintellect). |
| `bench hub env show` | Show a dataset's tasks, or a hosted environment's metadata. |
| `bench hub env inspect` | Inspect a file from a hub environment package. |
| `bench embodied` | Embodied rollouts (robots and simulators): reports, evidence checks, export, specs. |
| `bench embodied report` | Per task: rewards by seed, mean, std, pass@k and reset reproducibility. |
| `bench embodied export` | Write every trial's per-step transitions and episode index (steps.jsonl, episodes.jsonl). |
| `bench embodied verify` (0.10) | Check each trial's files against its verdict now: record hashes, video copies, reward vs the record. |
| `bench embodied check-spec` | Validate an embodiment spec. |
| `bench embodied robo-path` | Print the path of the agent-side `robo` script (standard library only). |
| `bench continue` | Resume a previous unfinished (timed-out) OpenHands run to completion. (DEPRECATED) |
| `bench continue-batch` | Continue all timed-out OpenHands runs under a directory tree. (DEPRECATED) |
| `bench adopt` | deprecated; use `bench eval adopt`. |
| `bench adopt init` | Scaffold benchmarks/&lt;name&gt;/ for a new benchmark adoption (deprecated alias). |
| `bench adopt convert` | Drive the conversion workflow by launching the host codex CLI (deprecated alias). |
| `bench adopt verify` | Run the parity gate for an adopted benchmark; emit a verdict (deprecated alias). |
| `bench environment` | Deprecated; use `bench sandbox` / `bench hub`. |
| `bench environment create` | Deprecated; use `bench sandbox create`. |
| `bench environment list` | Deprecated; use `bench sandbox list` (local) or `bench hub list` (hosted). |
| `bench environment show` | Deprecated; use `bench hub show`. |
| `bench environment inspect` | Deprecated; use `bench hub inspect`. |
| `bench environment cleanup` | Deprecated; use `bench sandbox cleanup`. |
| `bench monitor` | Monitor mode; score a rollout in production (#386). |
| `bench monitor run` | Score one trajectory under monitor semantics. |
| `bench monitor replay` | Re-score a persisted rollout under monitor semantics. |
| `bench monitor watch` | Stream-score live production events. |
