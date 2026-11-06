# Configuration

`task.md` starts with YAML frontmatter, the task's settings, between two `---` lines. The body below it is the [instruction](./instruction.md).

```markdown
---
schema_version: "1.3"
task:
  name: my-org/hello-world
metadata:
  difficulty: easy
  category: file-ops
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
sandbox:
  cpus: 1
  memory_mb: 2048
---

Create a file `/app/hello.txt` containing exactly `Hello, world!`.
```

A `task.md` without frontmatter is read as a [task.md draft 2](./taskmd-draft-2.md) package. BenchFlow refuses a key it does not know, so a typo fails when the task loads instead of being ignored.

## Settings

| Key | What it sets | Default |
|---|---|---|
| `schema_version` (or `version`) | Settings schema version | `"1.3"` |
| `task` | `name` (`org/name`), `description`, `authors`, `keywords`, `version` (kept, not used) | none |
| `metadata` | Anything descriptive: difficulty, category, tags | none |
| `agent` | How the agent runs; see [below](#agent) | no timeout |
| `verifier` | How the verifier runs; see [below](#verifier) | `timeout_sec: 600` |
| `sandbox` | The sandbox; see [below](#sandbox) | 1 CPU, 2048 MB memory, 10240 MB disk |
| `oracle` | `env`, `timeout_sec` for `oracle/solve.sh`; see [Oracle](./oracle.md) | none |
| `artifacts` | Paths to save after the agent runs; see [Environment](./environment.md#artifacts) | none |
| `source`, `reward` | Provenance and reward metadata | none |
| `steps`, `multi_step_reward_strategy` | task.toml multi-step settings. They parse, and the run refuses them. | none |

The task.toml `[environment]` table is `sandbox` here. `bench import` renames it. Writing `environment:` in `task.md` fails with a hint.

Use `oracle`. `solution` is accepted for imported tasks; a task with both fails to load.

### agent

| Key | Meaning |
|---|---|
| `timeout_sec` | The agent's time budget in seconds |
| `user` | The OS user the agent runs as |
| `network_mode`, `allowed_hosts` | A network policy for the agent alone; see [Network policies](./network-policies.md) |
| `prompt_prefix` | Text put before the instruction, for run-wide rules. Never task-specific hints. |

Set `agent.timeout_sec` on every published task. A task that omits it gives the agent (and the oracle) no time limit of its own, as in 0.7.11: the rollout's whole deadline bounds it (a 900 s agent share plus the verifier, build and setup budgets; `bench check` prints the figure for the task). A per-run timeout (`RolloutConfig(timeout=...)`) sets one, and `--timeout-multiplier` scales that deadline-bounded budget.

### verifier

| Key | Meaning |
|---|---|
| `timeout_sec` | The verifier's time budget, default 600 |
| `env` | Variables for `test.sh`; see [Verifier](./verifier.md#environment-variables) |
| `user` | The OS user `test.sh` runs as, default root |
| `service` | The compose service `test.sh` runs in, default `main`; see [Environment](./environment.md#run-the-verifier-in-another-service) |
| `reward_range` | The allowed reward range, default `[0, 1]` |
| `network_mode` | `no-network` or `public` while the verifier runs |
| `sandbox_mode`, `sandbox` | Run the verifier in its own sandbox; see [Separate verifier](./separate-verifier.md) |
| `workspace_recovery`, `submission_files` | Grade again without rerunning the agent; see [Verifier recovery](./verifier-recovery.md) |

### sandbox

| Key | Meaning | Default |
|---|---|---|
| `docker_image` | A prebuilt image instead of `environment/Dockerfile` | none |
| `cpus`, `memory_mb`, `storage_mb` | Resources | 1, 2048, 10240 |
| `build_timeout_sec` | Time allowed to build the image | 600 |
| `workdir` | The working directory, an absolute path other than `/` | the image's `WORKDIR` |
| `env` | Variables in the sandbox | none |
| `setup_commands` | Commands run after the services start, before the agent is installed | none |
| `healthcheck` | A readiness check: `command`, `interval_sec`, `timeout_sec`, `start_period_sec`, `retries` | none |
| `mcp_servers` | MCP servers for the agent | none |
| `network_mode`, `allowed_hosts`, `blocked_urls`, `blocked_hosts` | The network policy; see [Network policies](./network-policies.md) | `public` |
| `gpus`, `gpu_types` | [Modal GPU request](../sandboxes/modal.md#gpus): non-negative integer count and at most one type; other backends refuse GPUs | 0, none |
| `os`, `tpu` | Parsed. `windows` and any TPU are refused before launch. | `linux`, none |

## Multi-agent keys

`agents` declares named roles, each with `agent`, `model` and `reasoning_effort`. `scenes` orders their turns, and `user` sets up a simulated user. A turn that names an undeclared role fails to parse. See [Instruction](./instruction.md#several-agents).

Keys under `benchflow:` are reserved for BenchFlow.

## Shorthands

BenchFlow expands these when it reads the file:

| Write | Means |
|---|---|
| `name: hello-world` | `task.name: benchflow/hello-world` (a `/` in the value keeps your org) |
| `image: ubuntu:24.04` | `sandbox.docker_image: ubuntu:24.04` |
| `verifier: verifier/` | The default verifier folder and entry point |
| `oracle: oracle/` | The default oracle folder |
| `profile: code-change` | A bundle of defaults; see below |

## Profiles

`profile: <name>` merges a bundle of defaults under your own keys. Your keys win.

| Profile | Sets |
|---|---|
| `code-change` | `ubuntu:24.04`, no network, 4 CPUs, 8192 MB, workdir `/repo`, agent 7200 s, verifier 1200 s as root, prompts appended in order |
| `task-toml-compatible` | 1 CPU, 2048 MB, and marks the task as exportable to task.toml |
| `criteria-runner` | A criteria-runner verifier: `verifier/verifier.md`, `verifier/criteria_runner/reward.py` |
| `acceptance-live` | Live acceptance cases: an oracle rerun and a reference verifier run, each at reward ≥ 0.99 |
| `multi-agent` | Three roles (architect, implementer, reviewer) in three scenes |
| `leaderboard-local` | Requires a leaderboard verdict in the live acceptance report |

An unknown name fails to parse. `profiles:` takes a list. To see the expanded result:

```bash
bench tasks normalize tasks/my-task          # print the expanded task.md
bench tasks normalize tasks/my-task --write  # replace task.md with it
```

## Versions

```yaml
schema_version: "1.3"        # task configuration schema version
benchflow:
  document_version: "0.6"    # BenchFlow-only task.md syntax under benchflow:
```

A `schema_version` with a major version other than 1 is refused.

## Next steps

- [Instruction](./instruction.md): the body of `task.md`.
- [Verifier](./verifier.md): the reward contract.
- [Task package reference](../reference/task-package.md): reward rules, compatibility and every check.
