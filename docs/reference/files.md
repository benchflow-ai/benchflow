# Files BenchFlow reads and writes

You write two kinds of file for BenchFlow, a task's `task.md` header and a run config; every key they take is here, with its type and default. BenchFlow writes JSON documents for each job and trial; the ones with a schema file are here, with their top-level fields. Every file in a trial's folder, and every key of its `result.json`, is in [Read the results](../evaluate/read-results.md#everything-in-a-trials-folder).

<!-- This page is generated from the code by tools/docs_reference.py. To change it, change the field, flag help or schema file it comes from, then run `uv run python tools/docs_reference.py --write`. -->

## Files you write

### task.md

The header is YAML between two `---` lines at the top of `task.md`; the instruction is below it. Each table below is one key's settings, with the type, default and description the code gives them. A key is optional unless the table says required, and a key the header doesn't know is refused. A key new since 0.9.1 has `(0.10)` after it when 0.10.0rc1 has it, or `(unreleased)` when no release has it yet.

| Key | Type | Default | Description |
|---|---|---|---|
| `schema_version` | text | `1.3` |  |
| `task` | table ([`task`](#task)) |  | Package information from the [task] section. |
| `metadata` | mapping | `{}` |  |
| `verifier` | table ([`verifier`](#verifier)) |  |  |
| `agent` | table ([`agent`](#agent)) |  |  |
| `sandbox` | table ([`sandbox`](#sandbox)) |  |  |
| `oracle` | table ([`oracle`](#oracle)) |  | Also read as `solution`. |
| `source` | text |  |  |
| `reward` | mapping | `{}` | Legacy task-level reward expression metadata. |
| `multi_step_reward_strategy` | `mean` \| `final` |  | How to derive one rollout reward from per-step verifier results. Used only by a task with `steps`, which runs on 0.10.0rc1. |
| `steps` | list of tables ([`steps[]`](#steps)) |  | 0.9.1 reads this key but refuses to run a task that has it (`multi-step tasks are not supported`); multi-step tasks run on 0.10.0rc1. |
| `artifacts` | list of text or tables ([`artifacts[]`](#artifacts)) | `[]` |  |
| `preference` (0.10) | mapping | `{}` | How `bench preferences` pairs and ranks this task's outputs (protocol, randomize_sides, raters_per_pair, min_agreement; other keys are recorded). The run itself is unchanged. See docs/tasks/human-preferences.md. |
| `agents` | table: `roles`, each role by name ([`agents.roles.*`](#agentsroles)) |  | A multi-turn task's roles: which agent plays each. |
| `scenes` | list of tables ([`scenes[]`](#scenes)) |  | A multi-turn task's scenes, run in order; each has its turns. |
| `user` | table ([`user`](#user)) |  | A simulated user that answers the agent; its persona is the body's `## user-persona` section. |
| `benchflow` | table ([`benchflow`](#benchflow)) |  | BenchFlow's own settings, read by the features that use them. |

#### `task`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | required | Package name in org/name format (e.g., 'benchflow/hello-world') |
| `version` | text | `""` | Package version (task.toml schema 1.3 [task] field). Informational only; recorded, never interpreted. Absent from this model it was extra_forbidden, which made schema-1.3 tasks that set it unloadable. |
| `description` | text | `""` | Human-readable description of the task |
| `authors` | list of tables ([`task.authors[]`](#taskauthors)) | `[]` | List of package authors |
| `keywords` | list of text | `[]` | Keywords for search and categorization |

#### `task.authors[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | required | Author name |
| `email` | text |  | Author email address |

#### `verifier`

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `test-script` \| `llm-judge` | `test-script` | Verification method. In `task.md`, 0.9.1's `bench check` and `bench run` refuse `llm-judge` (no runnable verifier entrypoint); a verifier document grades with a model instead ([LLM judge](../tasks/llm-judge.md)). |
| `submission_files` | list of text | `[]` | Explicit verifier-recovery contract (submission-files-v1): the absolute paths of the task's submission files, at most 8, literal (no globs), outside /dev, /logs, /proc, /solution, /sys and /tests. Only these regular files are captured and restored; never a whole directory. |
| `workspace_recovery` | boolean | `false` | Allow verifier-only recovery from captured workspace after transport failure. The task author guarantees all solver changes needed for scoring are inside the workspace and a fresh task image/setup reproduces all other dependencies; no mutable external state. |
| `timeout_sec` | number (> 0) | `600.0` | Wall-clock budget in seconds for verifier execution. Omitting the field inherits this default; zero, negative, and non-finite budgets are rejected at validation time. |
| `env` | mapping of names to text | `{}` |  |
| `collect` (unreleased) | list of tables ([`verifier.collect[]`](#verifiercollect)) | `[]` | Commands run in compose services after the agent's turn and before artifact collection (task.toml [[verifier.collect]]). |
| `required_tools` | list of text |  | Complete list of PATH program names whose availability or lookup can affect grading, including subprocesses and later-created helpers. Required when any image PATH entry is omitted by hardening. Every name must already resolve in the trusted verifier PATH; use absolute paths for uploaded/generated helpers. [] declares no required PATH tools. |
| `optional_tools` | list of text | `[]` | PATH program names used only for optional probes. The task author guarantees that every use tolerates the program being unavailable without changing grading. Refusals of any other program invalidate every reward. |
| `reward_range` | [number, number] |  | Declared [lo, hi] reward contract for this task. Omitted (None) keeps BenchFlow's strict canonical [0.0, 1.0]. A declared range may only WIDEN the canonical range (lo &lt;= 0.0, hi &gt;= 1.0, lo &lt; hi, both finite) so 0 ('did nothing') and 1 ('solved') stay meaningful for every task; e.g. safety-floor benchmarks declare reward_range = [-1.0, 1.0] to score deliberate violations below doing nothing. Applies to the test-script reward contract (reward.txt / reward.json scalar reward, numeric metrics, aggregate results) and the final reward canonicalization gate; LLM-judge rubric scores stay [0, 1]. |
| `user` | text or integer |  | Username or UID to run the verifier as. |
| `network_mode` | `no-network` \| `public` \| `allowlist` \| `denylist` |  | Optional verifier-specific network policy override. |
| `allowed_hosts` | list of text |  | Hostnames reachable when network_mode='allowlist'. |
| `sandbox_mode` | `shared` \| `separate` |  | Whether the verifier runs in the agent sandbox ('shared') or a dedicated verifier sandbox ('separate'). |
| `sandbox` | table, the keys of [`sandbox`](#sandbox) |  | Optional separate verifier sandbox configuration. |
| `service` | text | `main` | Compose service the test-script verifier runs in. Defaults to 'main' (the agent container). Multi-container (vulhub-style) tasks set this to a target/database service so test.sh can inspect target-side state (RCE markers, DB modifications) rather than only the agent's workspace. The agent's anti-tamper hardening only applies to 'main'; deliberately vulnerable target containers are intentionally not hardened. See #248. |
| `judge` | table ([`verifier.judge`](#verifierjudge)) |  | LLM-judge configuration (used when type == 'llm-judge'). Read only with `type: llm-judge`, which 0.9.1 refuses in `task.md` ([LLM judge](../tasks/llm-judge.md)). |
| `memory` | table ([`verifier.memory`](#verifiermemory)) |  | Memory-space scoring fixtures. |
| `hardening` | table ([`verifier.hardening`](#verifierhardening)) |  | Per-task verifier hardening opt-outs. |
| `pytest_plugins` | list of text | `[]` | pytest11 plugin names to allow under PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Container-side auto-discovery handles most cases; this is the explicit fallback for plugins that discovery cannot see (e.g. ctrf, playwright). |
| `pytest_plugin_releases` | list of text | `[]` | Pinned PyPI releases (name==version, e.g. pytest-asyncio==0.24.0) of the pytest plugins test.sh installs itself. The plugin guard loads such a plugin from the agent-writable trees or uvx's cache only when its installed files are that release's wheel, byte for byte, as PyPI publishes it; pytest-json-ctrf needs no entry. |

#### `verifier.collect[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `command` | text | required |  |
| `service` | text | `main` |  |
| `timeout_sec` | number (> 0) | `60.0` |  |
| `user` | text or integer |  |  |

#### `verifier.judge`

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | text | `claude-sonnet-5-5` | Judge model identifier. Provider is routed from the prefix (claude-* / gpt-* / gemini-*). 0.9.1 and 0.10.0rc1 default to `claude-sonnet-4-6`. |
| `rubric_path` | text | `tests/rubric.toml` | Path to the rubric file, relative to the task directory. Both rubric.toml (native) and rubric.json (Harvey LAB style) are supported. |
| `input_dir` | text | `/app` | Directory inside the sandbox holding the agent's deliverables. Its contents are downloaded and shown to the judge. |
| `input_type` | `deliverables` | `deliverables` | What the judge evaluates. Only 'deliverables' (agent output files) is supported; trajectory judging is not available at verify time. |
| `context` | text | `""` | Optional extra context passed to the judge prompt (defaults to the task instruction when empty). |

#### `verifier.memory`

| Key | Type | Default | Description |
|---|---|---|---|
| `expected_skills` | list of text |  | Skill pack names the task expects the agent to add/update/remove. None means no answer key was supplied; [] means no skill should change. |

#### `verifier.hardening`

| Key | Type | Default | Description |
|---|---|---|---|
| `cleanup_conftests` | boolean | `true` |  |

#### `agent`

| Key | Type | Default | Description |
|---|---|---|---|
| `prompt_prefix` | text |  | Optional run-specific policy text prepended to each resolved task prompt. Intended for generic harness constraints such as benchmark integrity rules; must not contain task-specific solution content. |
| `timeout_sec` | number (> 0) |  | Wall-clock budget in seconds for the agent harness. Omitting the field gives the agent no limit of its own: the rollout's whole deadline bounds it (as in 0.7.11) unless the caller passes a timeout; zero, negative, and non-finite budgets are rejected at validation time, matching verifier.timeout_sec. |
| `user` | text or integer |  | Username or UID to run the agent as. |
| `network_mode` | `no-network` \| `public` \| `allowlist` \| `denylist` |  | Optional agent-specific network policy override. |
| `allowed_hosts` | list of text |  | Hostnames reachable when network_mode='allowlist'. |

#### `sandbox`

| Key | Type | Default | Description |
|---|---|---|---|
| `network_mode` | `no-network` \| `public` \| `allowlist` \| `denylist` | `public` | Network access policy for this environment. |
| `allowed_hosts` | list of text |  | Hostnames reachable when network_mode='allowlist'. |
| `blocked_urls` | list of text |  | URL prefixes the agent cannot reach when network_mode='denylist'; scheme optional, query ignored |
| `blocked_hosts` | list of text |  | Hostnames (and their subdomains) the agent cannot reach when network_mode='denylist' |
| `build_timeout_sec` | number | `600.0` |  |
| `docker_image` | text |  | Prebuilt image name; legacy SkillsBench tasks may use image. Also read as `image`. |
| `os` | `linux` \| `windows` | `linux` | Target operating system for the task container. |
| `cpus` | integer | `1` |  |
| `memory_mb` | integer | `2048` |  |
| `storage_mb` | integer | `10240` |  |
| `gpus` | integer (>= 0) | `0` |  |
| `gpu_types` | list of text |  | Acceptable GPU types. Modal accepts one entry, e.g. ['H100']; omit for any type when gpus is positive. |
| `mcp_servers` | list of tables ([`sandbox.mcp_servers[]`](#sandboxmcp_servers)) | `[]` |  |
| `env` | mapping of names to text | `{}` | Environment variables resolved from host at runtime. Supports $&#123;VAR&#125; and $&#123;VAR:-default&#125; template syntax. |
| `tpu` | table ([`sandbox.tpu`](#sandboxtpu)) |  | Optional TPU accelerator request. |
| `skills_dir` | text |  | Path to skills directory in the sandbox. |
| `healthcheck` | table ([`sandbox.healthcheck`](#sandboxhealthcheck)) |  | Healthcheck to run after environment start. |
| `workdir` | text |  | Default working directory for command execution. |
| `setup_commands` | list of tables ([`sandbox.setup_commands[]`](#sandboxsetup_commands)) | `[]` | Commands run after sandbox services start and before the agent is installed. Useful for benchmark-native workspace/database seeding. |
| `bugswarm_image_tag` | text |  | BugSwarm image tag used by build-repair benchmark tasks. |
| `allow_cloud_metadata` | boolean | `false` | Let the agent reach the cloud metadata endpoints under an open network: no block, and the trial records the opt-out. Without it the agent starts only once the block is proven on. |
| `require_cloud_metadata_block` | boolean | `false` | Refuse an agent that runs as root on a sandbox provider without BenchFlow's metadata helper (Daytona, Modal, AgentCore, Apple), where it otherwise runs under the provider's own metadata policy. |
| `allow_internet` | boolean | `true` | Deprecated compatibility field; use network_mode instead. |
| `memory` | text |  |  |
| `storage` | text |  |  |

#### `sandbox.mcp_servers[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | required |  |
| `transport` | `stdio` \| `sse` \| `streamable-http` | `sse` |  |
| `url` | text |  |  |
| `command` | text |  |  |
| `args` | list of text | `[]` |  |
| `cwd` | text |  |  |
| `env` | mapping of names to text | `{}` |  |
| `headers` | mapping of names to text | `{}` |  |
| `tools` | list of text |  |  |
| `include_tags` | list of text |  |  |
| `exclude_tags` | list of text |  |  |

#### `sandbox.tpu`

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | text (at least 1 long) | required |  |
| `topology` | text | required |  |

#### `sandbox.healthcheck`

| Key | Type | Default | Description |
|---|---|---|---|
| `command` | text | required |  |
| `interval_sec` | number | `5.0` |  |
| `timeout_sec` | number | `30.0` |  |
| `start_period_sec` | number | `0.0` |  |
| `start_interval_sec` | number | `5.0` |  |
| `retries` | integer | `3` |  |

#### `sandbox.setup_commands[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `command` | text (at least 1 long) | required |  |
| `timeout_sec` | number (> 0) | `600.0` |  |
| `cwd` | text |  |  |
| `env` | mapping of names to text | `{}` |  |
| `user` | text or integer |  |  |
| `service` | text | `main` |  |
| `host_lock` | text |  |  |
| `capture_dir` | text |  | Optional absolute sandbox directory to archive after the command succeeds. The archive is written to capture_dir_b64_env. |
| `capture_dir_b64_env` | text |  | Host environment variable to update with a base64 tar.gz archive of capture_dir after the command succeeds. |
| `capture_dir_b64_env_file_var` | text |  | Optional host environment variable whose value is a dotenv file path. When set, capture_dir_b64_env is also upserted there. |

#### `oracle`

| Key | Type | Default | Description |
|---|---|---|---|
| `env` | mapping of names to text | `{}` |  |
| `timeout_sec` | number |  |  |

#### `steps[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | text | required |  |
| `agent` | table, the keys of [`agent`](#agent) |  |  |
| `verifier` | table, the keys of [`verifier`](#verifier) |  |  |
| `min_reward` | number or mapping of names to numbers |  |  |
| `healthcheck` | table, the keys of [`sandbox.healthcheck`](#sandboxhealthcheck) |  |  |
| `artifacts` | list of text or tables, the keys of [`artifacts[]`](#artifacts) | `[]` |  |

#### `artifacts[]`

| Key | Type | Default | Description |
|---|---|---|---|
| `source` | text | required |  |
| `destination` | text |  |  |
| `exclude` | list of text | `[]` |  |
| `service` (unreleased) | text |  | The compose service to collect this artifact from (the task.toml format's sidecar artifacts): None or 'main' is the agent's own container; any other names a service of the task's compose file, needs a sandbox that runs compose services and an absolute source, and is read from that service's own filesystem. |

#### `agents.roles.*`

Each role of `agents.roles`, under its name.

| Key | Type | Description |
|---|---|---|
| `agent` | text | Required. |
| `model` | text |  |
| `reasoning_effort` | text |  |
| `env` | mapping of names to text |  |
| `timeout_sec` | integer | None = inherit from task config |
| `idle_timeout_sec` | integer |  |
| `skills_dir` | text or path |  |
| `capabilities` | list of text | e.g. ["tool-use", "agent-as-tool", "loop"] |

#### `scenes[]`

| Key | Type | Description |
|---|---|---|
| `name` | text |  |
| `roles` | list of role names |  |
| `turns` | list of turns |  |
| `skills_dir` | text or path |  |

#### `scenes[].turns[]`

A turn is a table, or the name of the role that takes it.

| Key | Type | Description |
|---|---|---|
| `role` | text | Required. |
| `prompt` | text | None = expand from the task prompt |

#### `user`

The keys BenchFlow reads: `model`, `private_facts`, `stop_rule`.

#### `benchflow`

The keys BenchFlow reads: `compat`, `environment`, `evidence`, `nudges`, `prompt`, `teams`, `verifier`.

### Run config

`bench run --config FILE` reads a whole run from a YAML file of these keys. It needs `tasks_dir` or `source`; a flag given beside it wins over its key; `sandbox` is read as `environment`; and a key not in this table is refused, with the closest one. The flag column names the flag that sets the same thing, and the help is that flag's. A key new since 0.9.1 has `(0.10)` after it when 0.10.0rc1 has it, or `(unreleased)` when no release has it yet.

| Key | Flag | Default | Help |
|---|---|---|---|
| `accept_deviations` (0.10) | `--accept-deviations` (0.10) |  | Why this run differs from its task set's run-reference.toml (a strict one refuses the run without it); kept in run-card.json. |
| `agent` | `--agent` | `claude-agent-acp` | Agent: claude, codex, gemini, … (bench agents), oracle (the task's reference solution) or nop (does nothing) |
| `agent_env` | `--agent-env` | `{}` | Agent env var (KEY=VALUE) |
| `agent_env_keys` |  | `[]` |  |
| `agent_idle_timeout` | `--agent-idle-timeout` | `600` | Abort a prompt when the agent is silent this long: seconds, or 90s, 10m, 1h; 0, none or null turns it off and leaves the task's wall-clock timeout (default: 10m, 30m at effort xhigh or max) |
| `agent_idle_timeout_sec` | `--agent-idle-timeout` | `600` | Abort a prompt when the agent is silent this long: seconds, or 90s, 10m, 1h; 0, none or null turns it off and leaves the task's wall-clock timeout (default: 10m, 30m at effort xhigh or max) |
| `allow_cloud_metadata` | `--allow-cloud-metadata` | `false` | Let the agent reach the cloud metadata endpoints under an open network (no block; each trial records the opt-out). Without it, or the task's allow_cloud_metadata = true, the agent starts only once the block is proven on |
| `base_image_override` | `--base-image-override` |  | Rewrite task Dockerfile FROM images on the runtime task copy. Use only for reproducing runs whose base image moved namespaces. |
| `budget` | `--max-cost-usd`, `--max-sandbox-seconds`, `--max-tokens`, `--max-rollouts` |  | A table of `max_cost_usd`, `max_sandbox_seconds`, `max_tokens`, `max_rollouts`. |
| `build_concurrency` | `--build-concurrency` |  | Max concurrent docker image builds. Defaults to --concurrency. Set lower (e.g. 8) when --concurrency is high to avoid overwhelming the docker daemon with parallel builds. |
| `checkpoint_keep` | `--checkpoint-keep` | `3` | Keep at most this many checkpoints per trial (default 3, or a run config's checkpoint_keep); the oldest is deleted. |
| `checkpoints` | `--checkpoints` |  | Keep a sandbox snapshot after these prompts: every-prompt or prompt:N[,M]. Later `bench branch --from-checkpoint <trial>` forks from one. Docker, or Daytona direct; each is a full filesystem image. See docs/train/environment.md |
| `codex_apps_policy` | `--codex-apps-policy` |  | Codex account Apps: disabled (scored-task default) or inherit (explicit opt-in). |
| `concurrency` | `--concurrency` | `4` | Trials running at a time. On Docker a job also prepares up to --warm-pool sandboxes ahead (default 1 when this is 2 or more) |
| `config_override` | `--config-override` |  | Deep-merge a config patch into each task's resolved config for this run. Inline JSON/YAML/TOML or an @file ref, e.g. --config-override '&#123;"agent":&#123;"timeout_sec":120&#125;&#125;'. Varies one knob (budget, skills, stopping rules) while the tasks, agent, model and environment stay fixed; recorded by content hash for replay. |
| `context_root` | `--context-root` |  | Repo/build-context root used to stage Dockerfile COPY sources for monorepo-authored local tasks. |
| `dataset_name` |  |  |  |
| `dataset_task_digests` |  | `{}` |  |
| `dataset_version` |  |  |  |
| `environment` | `--sandbox` | `docker` | Sandbox: docker, remote-docker, daytona, modal, apple-container, or agentcore |
| `environment_manifest` | `--environment-manifest` |  | Environment-plane manifest applied to every rollout: a path to an environment.toml, OR a 'name@version' registry spec resolved via $BENCHFLOW_ENV_REGISTRY when set, else the built-in registry shipped with benchflow (env0@prod, env0@outage). The manifest-declared stateful environment is provisioned, gated on readiness, and torn down. |
| `exclude` | `--exclude` | `[]` | Skip these task names; repeatable (e.g. --exclude quantum-numerical-simulation) |
| `extra_instruction` | `--extra-instruction` |  | Text appended to every task prompt (prompt ablations; recorded per rollout) |
| `freeze_workspace` | `--freeze-workspace` | `false` | Save each trial's final workspace and declared artifacts (evidence/, with a manifest) before the verifier runs, so `bench regrade` can re-score it with a changed verifier. |
| `fresh_install` | `--fresh-install` | `false` | Install the model proxy and the agent in every trial instead of starting from their cached image layers (same as BENCHFLOW_FRESH_INSTALL=1) |
| `harness` | `--harness` | `acp` | How the agent runs: acp (default, its ACP adapter) or native (its own CLI in headless JSON mode; claude-agent-acp and codex-acp) |
| `include` | `--include` | `[]` | Only run these task names; repeatable (e.g. --include jax-computing-basics --include data-to-d3) |
| `integrity` | `--integrity` | `off` | off, audit (record what the agent did and check it against the task) or strict (audit, and the verifier in a separate sandbox). The verdict is in integrity/claim_verdict.json; rewards never change. |
| `job_mode` |  | `parallel-independent` |  |
| `jobs_dir` | `--jobs-dir` | `jobs` | Output directory (default jobs); a plain run resumes its latest job |
| `keep_sandbox` (0.10) | `--keep-sandbox` (0.10) | `never` | Keep a trial's sandbox: on-failure, always or never (default). `bench sandbox attach <trial>` opens a shell in it; `bench sandbox release <trial>` deletes it. |
| `loop_strategy` | `--loop-strategy` |  | Harness loop strategy, e.g. 'verify-retry:k=3,feedback=names', 'self-review:k=3', or 'single-shot' (default). verify-retry re-prompts with filtered soft-verifier feedback; self-review re-prompts the agent to critique its OWN work (no verifier signal); both loop until the soft verifier passes or k retries are spent. |
| `max_retries` | `--retry-attempts` | `2` | Retries per task after a retryable failure (overrides the default of 2) |
| `model` | `--model` |  | Model id for the agent, e.g. claude-haiku-5-5 (none for oracle and nop) |
| `n_tasks` | `--n-tasks` |  | Run at most N tasks after --include/--exclude: the first N, or a seeded random sample with --sample-seed |
| `prompts` | `--prompt` |  | Prompt(s) to send (default: the task prompt: the task.md body, or instruction.md for older tasks) |
| `reasoning_effort` | `--reasoning-effort` |  | Agent reasoning/thinking effort when the agent exposes one (e.g. max) |
| `retry` | `--retry-attempts` |  | A table of `max_retries`, `retry_on_install`, `retry_on_pipe`, `retry_on_acp`, `retry_on_idle_timeout`, `retry_on_infra`, `retry_on_verifier_infra`, `retry_on_api_error`, `wait_multiplier`, `min_wait_sec`, `max_wait_sec`, `exclude_categories`. |
| `retry_from_checkpoint` | `--retry-from-checkpoint` |  | on-failure, on-timeout, or both: when a trial fails or times out, fork one retry from its last --checkpoints snapshot, verified the same way. The trial's reward is kept; the retry's is reported next to it (result.json retry block, summary checkpoint_retries). |
| `retry_prompt` | `--retry-prompt` |  | Prompt for the retry, replacing the prompts after the checkpoint (the default). '@instruction' is the task instruction, '@verifier_feedback' the failed trial's reward and verifier output tail. |
| `retry_resume_session` | `--retry-resume-session` | `false` | The retry resumes the failed trial's conversation at the checkpoint (ACP session/load; Claude Code) instead of a fresh session. |
| `reviewer` | `--reviewer-agent`, `--reviewer-model`, `--reviewer-reasoning-effort`, `--reviewer-sandbox`, `--reviewer-timeout-sec`, `--reviewer-idle-timeout`, `--reviewer-storage-mb` (0.10), `--reviewer-memory-mb` (0.10), `--reviewer-setup-timeout` (0.10), `--reviewer-evidence-timeout` (0.10), `--reviewer-concurrency`, `--reviewer-image`, `--reviewer-agent-env`, `--reviewer-open-network` |  | A table of `agent`, `model`, `reasoning_effort`, `environment`, `timeout_sec`, `idle_timeout_sec`, `storage_mb`, `memory_mb`, `setup_timeout_sec`, `evidence_timeout_sec`, `concurrency`, `image`, `agent_env`, `open_network`. |
| `sample_seed` | `--sample-seed` |  | Seed for the --n-tasks random sample |
| `sandbox_locked_paths` |  |  |  |
| `sandbox_setup_timeout` | `--sandbox-setup-timeout` | `120` | Timeout (seconds) for sandbox user setup inside the environment (default: 120; keeps the file's value with --config). |
| `sandbox_user` | `--sandbox-user` | `agent` | Sandbox user (null for root) |
| `seeds` | `--seeds` |  | Seeded rollouts: run every task once per seed, e.g. 0-4 or 0,3,7 (tasks in a task format that writes seeded variants, such as embodied tasks). summary.json gets pass@k and variance per task. |
| `self_gen_no_internet` | `--self-gen-no-internet` | `false` | Disable web tools for the self-generated run |
| `separate_verifier` | `--separate-verifier` | `false` | Run each verifier in a fresh sandbox built from the verifier image that receives only the frozen workspace, declared artifacts and /logs/artifacts, as if every task set \[verifier].sandbox_mode = 'separate'. Docker, remote-docker, Daytona and Modal. |
| `separate_verifier_include` | `--separate-verifier-include` | `[]` | A path outside the workspace that a separate verifier sandbox receives too, at the same path, for tasks whose agent writes graded files there (e.g. /root when the workspace is /app). Repeatable. |
| `skill_creator_dir` | `--skill-creator-dir` |  | Path to skill-creator or a skills root containing it |
| `skill_mode` | `--skill-mode` | `no-skill` | Skill mode: no-skill, with-skill, or self-gen |
| `skills_dir` | `--skills-dir` |  | Skills directory to deploy |
| `skip_install` |  | `false` |  |
| `source` | `--source-path`, `--source-ref`, `--source-repo` |  | A GitHub folder, a table of `path`, `ref`, `repo`. |
| `source_provenance` |  |  |  |
| `tasks_dir` | `--tasks-dir` |  | Local tasks directory |
| `timeout_multiplier` | `--timeout-multiplier` |  | Scale every task's agent time budget (task.md agent.timeout_sec), e.g. 2.0 |
| `usage_proxy` |  |  | No longer supported: a config with it is refused. Use `usage_tracking`. |
| `usage_proxy_advertised_base_url` |  |  | No longer supported: a config with it is refused. Use `usage_tracking`. |
| `usage_proxy_bind_host` |  |  | No longer supported: a config with it is refused. Use `usage_tracking`. |
| `usage_proxy_port` |  |  | No longer supported: a config with it is refused. Use `usage_tracking`. |
| `usage_proxy_url` |  |  | No longer supported: a config with it is refused. Use `usage_tracking`. |
| `usage_tracking` | `--usage-tracking` |  | Telemetry-enforcement policy: auto, required, or off. The LiteLLM proxy is always used for routable agents (usage, cost, and llm_trajectory.jsonl are always captured); this flag only controls whether trusted telemetry is required. |
| `verifier_sandbox` | `--verifier-sandbox` | `task` | task (default: each task's own choice) or separate: every verifier in a fresh sandbox with only the workspace and the declared artifacts; a task that cannot run there refuses the run. |
| `warm_pool` | `--warm-pool` |  | Docker: how many sandboxes to prepare ahead of their trial's turn, beyond the --concurrency running ones (0 to --concurrency; default 1, or 0 when --concurrency is 1) |

### YAML job config

A run config with `agents:` and `datasets:` (or a `tasks:` list) is a YAML job config: it runs every agent on every dataset, `n_attempts` times each. Its top-level keys: `agent_idle_timeout`, `agent_idle_timeout_sec`, `agent_setup_timeout_multiplier`, `agent_timeout_multiplier`, `agents`, `allow_cloud_metadata`, `base_image_override`, `codex_apps_policy`, `context_root`, `datasets`, `environment`, `environment_build_timeout_multiplier`, `exclude`, `exclude_tasks`, `excludes`, `extra_instruction_paths`, `extra_instructions`, `fresh_install`, `harness`, `include`, `include_tasks`, `includes`, `integrity`, `job_name`, `jobs_dir`, `metrics`, `n_attempts`, `n_concurrent_trials`, `orchestrator`, `quiet`, `reasoning_effort`, `reviewer`, `sandbox_locked_paths`, `sandbox_setup_timeout`, `sandbox_user`, `self_gen_no_internet`, `separate_verifier`, `separate_verifier_include`, `skill_creator_dir`, `skill_mode`, `skills_dir`, `tasks`, `timeout_multiplier`, `usage_proxy`, `usage_proxy_advertised_base_url`, `usage_proxy_bind_host`, `usage_proxy_port`, `usage_proxy_url`, `usage_tracking`, `verifier_sandbox`, `verifier_timeout_multiplier`. An agent entry takes `concurrency_group`, `env`, `harness`, `model_name`, `n_concurrent`, `name`, `reasoning_effort`, `skip_install`; a dataset entry `exclude_task_names`, `n_tasks`, `path`, `task_names`. A setting BenchFlow can't honour exactly is refused when the file loads, unless it holds its default:

| Key | Instead |
|---|---|
| `install_only` | BenchFlow has no install-only run; drop it |
| `debug` | drop it; the job folder keeps each trial's logs |
| `retry` | BenchFlow never retries trials from a YAML job config; drop it, or run the folder with bench run --retry-attempts N |
| `verifier` | BenchFlow runs each task's verifier as task.md (or task.toml) sets it; drop it, or change the task |
| `user_agent` | BenchFlow has no simulated user for a YAML job config; drop it |
| `artifacts` | drop it; the job folder keeps the agent's logs and trajectory |
| `source_jobs` | re-grade a job with bench regrade &lt;job folder&gt;; drop it |

## Files BenchFlow writes

The JSON documents BenchFlow writes, one section per schema file in [`schemas/`](./schemas/), with each document's top-level fields. A schema or field new since 0.9.1 has `(0.10)` after it when 0.10.0rc1 has it, or `(unreleased)` when no release has it yet.

### benchflow-assessment-event.v1.schema.json (0.10)

**One append-only event for a hash-bound blinded pairwise assessment**

Identifier: `https://benchflow.ai/schemas/benchflow-assessment-event.v1.schema.json`. File: [`schemas/benchflow-assessment-event.v1.schema.json`](./schemas/benchflow-assessment-event.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | constant "benchflow.assessment-event.v1" | yes |  |
| `assessment_id` | string | yes |  |
| `event_id` | string | yes |  |
| `event_seq` | integer | yes |  |
| `created_at` | string | yes |  |
| `kind` | one of "assessment.created", "compute.released", "assignment.issued", "review.submitted", "review.corrected", "review.retracted", "label.reduced", "label.consumed", "label.withdrawn" | yes |  |
| `actor` | object | yes |  |
| `idempotency_key` | string | yes |  |
| `expected_previous_event_sha256` | object (optionalDigest) | yes |  |
| `previous_event_sha256` | object (optionalDigest) | yes |  |
| `payload` | object | yes |  |
| `event_sha256` | object (digest) | yes |  |

### benchflow-assessment-label.v1.schema.json (0.10)

**The deterministic label an assessment sidecar's events reduce to**

Identifier: `https://benchflow.ai/schemas/benchflow-assessment-label.v1.schema.json`. File: [`schemas/benchflow-assessment-label.v1.schema.json`](./schemas/benchflow-assessment-label.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | constant "benchflow.assessment-label.v1" | yes |  |
| `assessment_id` | string | yes |  |
| `label_id` | string | yes |  |
| `revision` | integer | yes |  |
| `status` | one of "ready", "withheld", "withdrawn", "stale" | yes |  |
| `state` | one of "answered", "retracted", "label_ready", "consumed", "withheld", "withdrawn_for_future_use" | yes |  |
| `consumable` | boolean | yes |  |
| `resolved` | boolean | yes |  |
| `unresolved_revisions` | array of object | yes |  |
| `task_digest` | string | yes |  |
| `presentation` | object | yes |  |
| `attempts` | object | yes |  |
| `rubric` | object | yes |  |
| `judge_protocol` | object | yes |  |
| `collector` | object | yes |  |
| `reviewer_subject` | string | yes |  |
| `source` | object | yes |  |
| `revisions` | array of object | yes |  |
| `event_chain` | object | yes |  |
| `reducer` | constant &#123;"id": "benchflow.assessment.reduce", "version": "1"&#125; | yes |  |
| `value` | one of 1, 0, 0.5 | no |  |
| `withheld_reason` | one of "human_abstention", "answer_retracted" | no |  |

### benchflow-branch-child-row.v2.schema.json

**BenchFlow branch-tree child row (children.jsonl)**

One child of one fork: the shared prefix, this child's continuation and its reward, value and advantage.

Identifier: `https://benchflow.ai/schemas/benchflow-branch-child-row.v2.schema.json`. File: [`schemas/benchflow-branch-child-row.v2.schema.json`](./schemas/benchflow-branch-child-row.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `advantage` | number or null | yes | reward - value. |
| `agent` | string or null | no |  |
| `child` | object | yes |  |
| `continuation` | object (messages) | yes |  |
| `depth` | integer | yes | 1 for a fork of the trial, 2 for a fork inside a child… |
| `fork_id` | string | yes |  |
| `fork_kind` | one of "fork", "retry" | no |  |
| `fork_status` | string or null | no |  |
| `kind` | constant "branch_child" | yes |  |
| `messages` | object (messages) | yes | prefix + continuation, ready for apply_chat_template. |
| `model` | string or null | no |  |
| `parent_child` | string or null | no | Node id of the child whose fork this is (nested), else null. |
| `parent_node` | string or null | no |  |
| `prefix` | object (messages) | yes |  |
| `prefix_complete` | boolean | yes | False when the trial started from a kept checkpoint whose source conversation could not be found, so the prefix lacks it. |
| `prefix_source` | object or null | no | For a trial started from a kept checkpoint: the source trial, checkpoint and number of events prepended. |
| `schema_version` | constant 2 | yes |  |
| `session` | object (session) | yes |  |
| `siblings` | integer | no |  |
| `task` | string or null | no |  |
| `tool_names` | one of "agent", "inferred", "mixed", null | no | Whether tool names came from the agent or were inferred from the calls. |
| `tools` | object (tools) | yes |  |
| `trial` | string | yes | Trial folder name. |
| `value` | number or null | yes | Mean reward of the fork's children. |
| `value_stderr` | number or null | no |  |

### benchflow-branch-pair-row.v2.schema.json

**BenchFlow branch-tree pair row (pairs.jsonl)**

Two siblings of one fork with different rewards, in prompt/chosen/rejected form for preference training.

Identifier: `https://benchflow.ai/schemas/benchflow-branch-pair-row.v2.schema.json`. File: [`schemas/benchflow-branch-pair-row.v2.schema.json`](./schemas/benchflow-branch-pair-row.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `agent` | string or null | no |  |
| `chosen` | object (messages) | yes |  |
| `chosen_label` | string or null | no |  |
| `chosen_reward` | number | yes |  |
| `depth` | integer | no | 1 for a fork of the trial, 2 for a fork inside a child… |
| `fork_id` | string | yes |  |
| `kind` | constant "branch_pair" | yes |  |
| `margin` | number | yes |  |
| `model` | string or null | no |  |
| `parent_node` | string or null | no |  |
| `prefix_complete` | boolean | no | False when the trial started from a kept checkpoint whose source conversation could not be found, so the prefix lacks it. |
| `prompt` | object (messages) | yes | The shared prefix plus, when same_request, the shared user turn. |
| `rejected` | object (messages) | yes |  |
| `rejected_label` | string or null | no |  |
| `rejected_reward` | number | yes |  |
| `same_request` | boolean | yes | Both siblings were asked the same thing; only these are exported unless --pairs-any-request is given. |
| `schema_version` | constant 2 | yes |  |
| `session` | object (session) | no |  |
| `task` | string or null | no |  |
| `tools` | object (tools) | yes |  |
| `trial` | string | yes | Trial folder name. |

### benchflow-comparison.v1.schema.json

**ComparisonExport**

Identifier: `https://benchflow.ai/schemas/benchflow-comparison.v1.schema.json`. File: [`schemas/benchflow-comparison.v1.schema.json`](./schemas/benchflow-comparison.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `a` | object (ComparisonSideExport) | yes |  |
| `a_paired` | object (DenominatorsExport) or null | no | side A over the tasks both sides ran (1.1) |
| `b` | object (ComparisonSideExport) | yes |  |
| `b_paired` | object (DenominatorsExport) or null | no |  |
| `by` | array of string | no | extra pairing keys besides task (1.1) |
| `caveats` | array of string | yes |  |
| `include_controls` | boolean | yes |  |
| `kind` | string | no |  |
| `mismatches` | array of object (SettingMismatchExport) | yes |  |
| `rows` | array of object (ComparisonRowExport) | yes |  |
| `schema_minor` | integer | no | optional fields added within schema_version; see the changelog |
| `schema_version` | integer | no |  |
| `summary` | object (ComparisonSummaryExport) | yes |  |
| `vary` | array of string | yes |  |

### benchflow-comparison.v2.schema.json

**ComparisonExport**

Identifier: `https://benchflow.ai/schemas/benchflow-comparison.v2.schema.json`. File: [`schemas/benchflow-comparison.v2.schema.json`](./schemas/benchflow-comparison.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `a` | object (ComparisonSideExport) | yes |  |
| `a_paired` | object (DenominatorsExport) or null | no | side A over the tasks both sides ran (1.1) |
| `b` | object (ComparisonSideExport) | yes |  |
| `b_paired` | object (DenominatorsExport) or null | no |  |
| `by` | array of string | no | extra pairing keys besides task (1.1) |
| `caveats` | array of string | yes |  |
| `include_controls` | boolean | yes |  |
| `kind` | string | no |  |
| `mismatches` | array of object (SettingMismatchExport) | yes |  |
| `rows` | array of object (ComparisonRowExport) | yes |  |
| `schema_minor` | integer | no | optional fields added within schema_version; see the changelog |
| `schema_version` | integer | no |  |
| `scoring` (unreleased) | object or null | no | the scoring rule's figures for each side on the tasks both ran, the gap b - a with its 95% interval paired by task, its warnings, and each side's mixed settings (2.1) |
| `summary` | object (ComparisonSummaryExport) | yes |  |
| `vary` | array of string | yes |  |

### benchflow-job.v1.schema.json

**JobExport**

Identifier: `https://benchflow.ai/schemas/benchflow-job.v1.schema.json`. File: [`schemas/benchflow-job.v1.schema.json`](./schemas/benchflow-job.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `cost_usd` | number or null | yes |  |
| `denominators` | object (DenominatorsExport) | yes | agent runs; controls left out |
| `denominators_with_controls` | object (DenominatorsExport) | yes |  |
| `error_categories` | object | no | agent runs that errored, by error_category (1.1) |
| `groups` | array of object (GroupExport) | no | agent-run denominators per agent and model (1.1) |
| `interrupted` | array of object (InterruptedExport) | no | attempt folders that never wrote result.json (1.1) |
| `job_kind` | string | yes |  |
| `kind` | string | no |  |
| `path` | string | yes |  |
| `schema_minor` | integer | no | optional fields added within schema_version; see the changelog |
| `schema_version` | integer | no |  |
| `solve_rates` | object (SolveRatesExport) or null | no | agent runs; default ks and success rule |
| `summary` | object or null | no | the job's summary.json |
| `timing_totals` | object | no | seconds per timing phase, summed over agent runs (1.1) |
| `trials` | array of object (TrialExport) | yes |  |

### benchflow-job.v2.schema.json

**JobExport**

Identifier: `https://benchflow.ai/schemas/benchflow-job.v2.schema.json`. File: [`schemas/benchflow-job.v2.schema.json`](./schemas/benchflow-job.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `cost_usd` | number or null | yes |  |
| `denominators` | object (DenominatorsExport) | yes | agent runs; controls left out, unless every trial is one |
| `denominators_with_controls` | object (DenominatorsExport) | yes |  |
| `error_categories` | object | no | agent runs that errored, by error_category (1.1) |
| `groups` | array of object (GroupExport) | no | agent-run denominators per agent and model (1.1) |
| `interrupted` | array of object (InterruptedExport) | no | attempt folders that never wrote result.json (1.1) |
| `job_kind` | string | yes |  |
| `kind` | string | no |  |
| `path` | string | yes |  |
| `schema_minor` | integer | no | optional fields added within schema_version; see the changelog |
| `schema_version` | integer | no |  |
| `scoring` (unreleased) | object or null | no | the scoring rule's figures over every trial and the trials split by setting, as summary.json's scoring block (2.1) |
| `solve_rates` | object (SolveRatesExport) or null | no | agent runs (controls left out, unless every trial is one); default ks and success rule |
| `summary` | object or null | no | the job's summary.json |
| `timing_totals` | object | no | seconds per timing phase, summed over agent runs (1.1) |
| `trials` | array of object (TrialExport) | yes |  |

### benchflow-model-call.v1.schema.json (unreleased)

**One model call (trajectory/model-calls.jsonl)**

Identifier: `https://benchflow.ai/schemas/benchflow-model-call.v1.schema.json`. File: [`schemas/benchflow-model-call.v1.schema.json`](./schemas/benchflow-model-call.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `cache_read_tokens` | integer or null | yes |  |
| `cache_write_tokens` | integer or null | yes |  |
| `cost_usd` | number or null | yes |  |
| `duration_ms` | number or null | yes |  |
| `effort` | string or null | yes |  |
| `finish_reason` | string or null | yes |  |
| `http_status` | integer or null | yes |  |
| `index` | integer | yes |  |
| `input_tokens` | integer or null | yes |  |
| `max_output_tokens` | integer or null | yes |  |
| `model_requested` | string or null | yes |  |
| `model_served` | string or null | yes |  |
| `output_tokens` | integer or null | yes |  |
| `purpose` | string or null | yes |  |
| `reasoning_tokens` | integer or null | yes |  |
| `refused` | boolean or null | yes |  |
| `route` | string or null | yes |  |
| `schema` | constant "benchflow.model-call.v1" | yes |  |
| `source` | one of "proxy", "claude-code-otel", "claude-code-session-log", "codex-session-log" | yes |  |
| `started_at` | string or null | yes |  |
| `subagent` | boolean or null | yes |  |
| `temperature` | number or null | yes |  |
| `thinking` | boolean or null | yes |  |
| `top_p` | number or null | yes |  |
| `truncated` | boolean or null | yes |  |

### benchflow-model-call.v2.schema.json (unreleased)

**One model call (trajectory/model-calls.jsonl)**

Identifier: `https://benchflow.ai/schemas/benchflow-model-call.v2.schema.json`. File: [`schemas/benchflow-model-call.v2.schema.json`](./schemas/benchflow-model-call.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `cache_read_tokens` | integer or null | yes |  |
| `cache_write_tokens` | integer or null | yes |  |
| `cost_usd` | number or null | yes |  |
| `duration_ms` | number or null | yes |  |
| `effort` | string or null | yes |  |
| `finish_reason` | string or null | yes |  |
| `http_status` | integer or null | yes |  |
| `index` | integer | yes |  |
| `input_tokens` | integer or null | yes |  |
| `max_output_tokens` | integer or null | yes |  |
| `model_requested` | string or null | yes |  |
| `model_served` | string or null | yes |  |
| `output_tokens` | integer or null | yes |  |
| `purpose` | string or null | yes |  |
| `reasoning_tokens` | integer or null | yes |  |
| `refused` | boolean or null | yes |  |
| `route` | string or null | yes |  |
| `schema` | constant "benchflow.model-call.v2" | yes |  |
| `seed` | integer or null | yes |  |
| `source` | one of "proxy", "claude-code-otel", "claude-code-session-log", "codex-session-log" | yes |  |
| `started_at` | string or null | yes |  |
| `subagent` | boolean or null | yes |  |
| `temperature` | number or null | yes |  |
| `thinking` | boolean or null | yes |  |
| `tools` | integer or null | yes |  |
| `top_p` | number or null | yes |  |
| `truncated` | boolean or null | yes |  |

### benchflow-model-calls.v1.schema.json (unreleased)

**A trial's model calls, totalled (model-calls.json)**

Identifier: `https://benchflow.ai/schemas/benchflow-model-calls.v1.schema.json`. File: [`schemas/benchflow-model-calls.v1.schema.json`](./schemas/benchflow-model-calls.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `agent_effort` | object | no |  |
| `agent_retries` | integer or null | no |  |
| `calls` | integer | yes |  |
| `cost_usd` | number or null | no |  |
| `effort` | object | no |  |
| `finish_reasons` | object | no |  |
| `first_agent_tools` | object or null | no |  |
| `first_call_at` | string or null | no |  |
| `http_errors` | integer or null | no |  |
| `http_status_counts` | object | no |  |
| `input_files` | object or null | no |  |
| `last_agent_call` | object or null | no |  |
| `max_output_tokens` | object | no |  |
| `model_time_ms` | number or null | no |  |
| `models_requested` | object | no |  |
| `models_served` | object | no |  |
| `purposes` | object | no |  |
| `refused_calls` | integer or null | no |  |
| `schema` | constant "benchflow.model-calls.v1" | yes |  |
| `source` | string or array or null | yes |  |
| `sources` | object | yes |  |
| `thinking` | object | no |  |
| `tokens` | object | yes |  |
| `truncated_calls` | integer or null | no |  |
| `truncation_rate` | number or null | no |  |
| `upstream_attempts` | integer or null | no |  |
| `upstream_errors` | integer or null | no |  |
| `upstream_retries` | integer or null | no |  |

### benchflow-rollout-stream.v1.schema.json

**One finished rollout from bench train stream / stream_rollouts**

One JSON line per finished rollout. reward is null for an unscored rollout, never 0. calls holds the gateway's token capture (benchflow.token_capture.v1) per model call; sequences (one per conversation) are present only when every call is complete and each prompt extends the previous prompt and sampled tokens of its conversation (token-in/token-out).

Identifier: `https://benchflow.ai/schemas/benchflow-rollout-stream.v1.schema.json`. File: [`schemas/benchflow-rollout-stream.v1.schema.json`](./schemas/benchflow-rollout-stream.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | constant "benchflow.rollout-stream.v1" | yes |  |
| `job` | string | yes | Job folder name. |
| `rollout` | string | yes | Rollout folder name. |
| `rollout_path` | string | yes | Rollout folder relative to the job folder. |
| `task` | string or null | yes |  |
| `agent` | string or null | yes |  |
| `model` | string or null | yes |  |
| `group_id` | string | yes | Stable id of the group, e.g. 'task=hello\|agent=claude-agent-acp\|model=vllm/policy\|job=2026-01-01__12-00-00'. |
| `group` | object | yes | The grouping key (task, agent, model by default). |
| `reward` | number or null | yes |  |
| `scored` | boolean | yes |  |
| `outcome` | one of "passed", "failed", "errored", "verifier_errored" | yes |  |
| `error` | string or null | yes |  |
| `finished_at` | string or null | yes |  |
| `token_capture` | object | yes |  |
| `calls` | array of object | yes |  |
| `sequences` | array of object | yes | One merged token stream per conversation of the rollout (agent loop, helper calls, subagents), only when the rollout is training-grade; empty otherwise. |
| `advantage` | number or null | yes | GRPO advantage over the group's scored rollouts (group_size only); null otherwise. |
| `group_complete` | boolean or null | yes | group_size only: false when the stream ended before the group filled. |
| `replaces` | string or null | no | rollout_path of the earlier attempt of the same trial that this rollout replaces (an Evaluation's retry or resume re-run); that attempt is not a sample. |
| `retried` | boolean | no | group_size only: true when a later attempt replaced this rollout before its group formed; it joins no group. |

### benchflow-run-summary.v1.schema.json

**RunSummaryExport**

The result of one ``bench eval run`` (``--summary-out``), for CI.

Identifier: `https://benchflow.ai/schemas/benchflow-run-summary.v1.schema.json`. File: [`schemas/benchflow-run-summary.v1.schema.json`](./schemas/benchflow-run-summary.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `budget` | object or null | no | summary.json's budget block when the job had a budget cap (caps, spent, stopped, reason, cancelled, not_started) |
| `errored` | integer | yes |  |
| `exit_code` | integer | yes | the process exit code of the run |
| `failed` | integer | yes |  |
| `gate` | object (GateExport) | yes |  |
| `job_dir` | string or null | yes |  |
| `job_name` | string | yes |  |
| `kind` | string | no |  |
| `mean_reward` | number or null | yes |  |
| `pass_rate` | number | yes | passed / total |
| `passed` | integer | yes |  |
| `ran` | integer | yes | tasks run by this invocation |
| `reused` | integer | yes | finished tasks reused from an earlier run (resume) |
| `schema_version` | integer | no |  |
| `timed_out` | integer | yes | trials whose agent timed out (scored or not) |
| `total` | integer | yes |  |
| `verifier_errored` | integer | yes |  |

### benchflow-run-summary.v2.schema.json

**RunSummaryExport**

The result of one ``bench eval run`` (``--summary-out``), for CI.

Identifier: `https://benchflow.ai/schemas/benchflow-run-summary.v2.schema.json`. File: [`schemas/benchflow-run-summary.v2.schema.json`](./schemas/benchflow-run-summary.v2.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `budget` | object or null | no | summary.json's budget block when the job had a budget cap (caps, spent, stopped, reason, cancelled, not_started) |
| `errored` | integer | yes |  |
| `exit_code` | integer | yes | the process exit code of the run |
| `failed` | integer | yes |  |
| `gate` | object (GateExport) | yes |  |
| `interrupted` | integer | no | of not_run, the tasks whose trial had started when it stopped |
| `job_dir` | string or null | yes |  |
| `job_name` | string | yes |  |
| `kind` | string | no |  |
| `mean_reward` | number or null | yes | over every trial, an unscored one counting 0 (2.0) |
| `mean_reward_scored_only` | number or null | no | over scored trials only, as mean_reward was in 1.x (2.0) |
| `not_run` | integer | no | planned tasks a budget, a usage limit or a stop kept from finishing; not in total, counted as not passed by --fail-under |
| `pass_rate` | number | yes | passed / total |
| `passed` | integer | yes |  |
| `ran` | integer | yes | tasks run by this invocation |
| `reused` | integer | yes | finished tasks reused from an earlier run (resume) |
| `schema_version` | integer | no |  |
| `stopped` | boolean | no | the run was stopped (Ctrl-C, SIGTERM): the counts are over the trials that finished first, and no gate was applied |
| `timed_out` | integer | yes | trials whose agent timed out (scored or not) |
| `total` | integer | yes |  |
| `verifier_errored` | integer | yes |  |

### benchflow-training-signal.v1.schema.json

**Training-signal fields of a bench train convert row**

Fields added by --reward-vector and --group-advantage to prime-sft and trl-sft rows. Other row fields are described by the trainer format. Unscored rollouts have a null reward_vector and a null advantage, never 0.

Identifier: `https://benchflow.ai/schemas/benchflow-training-signal.v1.schema.json`. File: [`schemas/benchflow-training-signal.v1.schema.json`](./schemas/benchflow-training-signal.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `reward_vector` | null or object | no |  |
| `advantage` | number or null | no |  |
| `advantage_vector` | null or array of number or null | no |  |
| `group` | object | no |  |

### benchflow-trial.v1.schema.json

**TrialExport**

Identifier: `https://benchflow.ai/schemas/benchflow-trial.v1.schema.json`. File: [`schemas/benchflow-trial.v1.schema.json`](./schemas/benchflow-trial.v1.schema.json).

| Field | Type | Required | Description |
|---|---|---|---|
| `agent` | string | yes |  |
| `agent_name` | string | yes |  |
| `assessment` | string | yes |  |
| `attempts` | integer | no | rollouts this trial took: 1, plus each retry (or resume re-run) of its task in an Evaluation job (1.3) |
| `checkpoints` | array of object | yes |  |
| `control` | string or null | yes | control runs check the task, not an agent |
| `duration_sec` | number or null | yes |  |
| `error` | string or null | yes |  |
| `error_category` | string or null | yes |  |
| `execution` | string | yes | integration_failed (1.2): the agent did nothing because its integration broke; see integration_failure |
| `finished_at` | string or null | yes |  |
| `forks` | array of object (ForkExport) | yes |  |
| `integration_failure` | object or null | no | why the agent did nothing useful when its integration broke: cause (agent_auth, agent_install, truncated_trajectory, empty_trajectory, immediate_exit, no_activity), evidence, evidence_source, activity counts, reward_withheld; detected: 'on read' for results written before 1.2 (1.2) |
| `kind` | string | no |  |
| `model` | string or null | yes |  |
| `n_prompts` | integer | yes |  |
| `n_tool_calls` | integer | yes |  |
| `passed` | boolean | yes |  |
| `path` | string | yes | trial folder, or the results.jsonl file |
| `review` | object (ReviewExport) or null | yes |  |
| `reward` | number or null | yes | rewards.reward; null when unscored |
| `rewards` | object or null | yes |  |
| `rollout_name` | string | yes |  |
| `row` | integer or null | no | row index for a results.jsonl trial |
| `rubric_reviews` | array of object (RubricReviewRecordExport) | no | every rubric review of the trial: scoring revisions oldest first, then detached audits (added in 1.1) |
| `sandbox` | object (SandboxExport) or null | no | sandbox.json, written when the sandbox was created (1.1) |
| `schema_minor` | integer | no | optional fields added within schema_version; see the changelog |
| `schema_version` | integer | no |  |
| `score_outcome` | string | yes |  |
| `scoring` | object or null | no | the automatic reviewer's gate verdict |
| `settings` | object | yes | what bf.compare checks; see SETTINGS |
| `source` | string | yes |  |
| `started_at` | string or null | yes | ISO 8601, local time as recorded |
| `task_name` | string | yes |  |
| `timing` | object | yes |  |
| `trajectory` | array of object or null | no | ACP events; null when not included in the export |
| `usage` | object (UsageExport) | yes |  |
| `verifier` | object (VerifierExport) or null | yes |  |
| `verifier_error` | string or null | yes |  |
| `verifier_error_category` | string or null | yes |  |
