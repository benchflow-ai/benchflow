# Third-Party Framework Integration Plan

Date: 2026-05-18

BenchFlow v0.4 should present third-party framework support as a compatibility
showcase, not a full migration of every task in every ecosystem. The practical
claim is:

> BenchFlow can ingest mainstream agent-eval frameworks, run representative
> tasks through the Rollout/Sandbox/Reward path, and emit unified trajectories,
> rewards, and artifacts.

This plan is grounded in current code inspection, small local compatibility
tests, Linear ENG-43/48/49/51, and current public docs for Harbor, Prime
Intellect Verifiers, Open Reward Standard, and Inspect AI.

## Current Reality

### What Already Works

- Harbor-style task packages are close to native BenchFlow task packages:
  `task.toml`, `instruction.md`, `environment/Dockerfile`, `solution/solve.sh`,
  and `tests/test.sh`.
- `TaskConfig` loads `[environment]` into internal `sandbox`, preserving
  file-level compatibility with Harbor-style task packages.
- The legacy `test.sh -> reward.txt` verifier pattern is wrapped by
  `TestRewardFunc` under the composable `Rubric` layer.
- `VerifyResult` and `RewardEvent` can be converted into an ORS-shaped reward
  dict via `ORSAdapter`.
- `InspectAdapter` can turn BenchFlow `Scene` + optional `Rubric` metadata into
  a thin Inspect-like dict.
- `bench eval create --source-repo/--source-path/--source-ref` can clone a GitHub
  task source and run local or batch task directories.
- The integration-suite manifest already has an `adapter-release-set` lane,
  but today it covers first-party BenchFlow adapters, not the wider Harbor
  registry or Verifiers/ORS/Inspect imports.

### What Is Mostly Aspirational Today

- There is no first-class `from_harbor_registry`, `from_verifiers`,
  `from_ors`, or `from_inspect` task import command.
- There are no optional extras for `verifiers`, `inspect-ai`, or `ors` SDKs.
- The current ORS adapter is a reward-format converter, not an ORS environment
  server or ORS client wrapper.
- The current Inspect adapter is a dict converter, not an importer for real
  `@task` definitions, datasets, scorers, or sandboxes.
- There is no Verifiers task/rubric adapter module yet.
- There is no compatibility board that tracks per-framework or per-benchmark
  pass level.

## Small Tests Run

- `uv run python -m pytest tests/test_adapters.py tests/test_rewards.py tests/test_tasks.py tests/test_integration_run_suite.py`
  - Result: `72 passed`.
- Parsed a Harbor-style `task.toml` with `[environment]`, deprecated
  `memory = "4G"`, `storage = "10G"`, and `allow_internet = false`.
  - Result: loaded successfully into `TaskConfig`; `memory_mb=4096`,
    `storage_mb=10240`, `allow_internet=false`.
- Converted a `VerifyResult` with a `RewardEvent` through `ORSAdapter`.
  - Result: returned ORS-shaped `{reward, is_valid, metadata}` dict.
- Sparse-cloned two QuixBugs tasks from Harbor registry source
  and ran `bench tasks check` one task at a time.
  - Result: `quixbugs-java-bitcount` and `quixbugs-java-breadth_first_search` valid.
  - Note: `bench tasks check` accepts one task path at a time, so the future
    compatibility runner should iterate task paths itself.

## External Framework Mapping

| Framework | External Primitive | BenchFlow Primitive | Near-Term Plan |
|---|---|---|---|
| Harbor | hosted environment/task registry, task directory, `test.sh`, `reward.txt` | hosted-environment source, task package, sandbox, verifier, rollout | Treat registry entries as hosted envs with `env_uid`; materialize two representative env/task refs per registry dataset and run package/build/oracle smoke where possible. |
| Verifiers / PrimeIntellect | hosted environment, parser, rubric, rollout output | hosted-environment source, harness, reward funcs, rollout result | Add optional `benchflow[verifiers]`; inventory selected hosted envs by `env_uid`, then wrap envs behind a task adapter. |
| ORS / OpenReward | hosted environment server, tasks/splits, tools, sessions, rewards | hosted-environment source, scene/turn loop, reward events | Add ORS client wrapper first, then ORS server mode later. Map session lifecycle to rollout lifecycle. |
| Inspect AI | `@task`, dataset, solver, scorer, sandbox | task package, agent/harness, reward func, sandbox | Start with scorer/task metadata import; defer full solver execution unless there is a clear use case. |

## Source Identity

Do not model PrimeIntellect, OpenReward, or Harbor registry entries as BenchFlow
benchmarks just because BenchFlow can eventually materialize tasks from them.
They are hosted environment catalogs first.

The canonical hub surfaces are:

- OpenReward environments: https://openreward.ai/environments
- Harbor Hub: https://hub.harborframework.com/
- PrimeIntellect Environments Hub: https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections

Use two separate identity fields:

```yaml
env_uid: primeintellect:<owner>/<name>@<version>
source:
  kind: hosted_env
  platform: primeintellect
  hub_url: https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections
  env: <owner>/<name>
  version: <version>
```

For ORS/OpenReward:

```yaml
env_uid: openreward:<owner>/<name>@<environment-id>
source:
  kind: ors
  platform: openreward
  hub_url: https://openreward.ai/environments
  env: <owner>/<name>
  environment_id: <environment-id>
  split: <split>
  updated_at: <catalog-updated-at>
```

OpenReward currently exposes stable environment IDs and `updated_at` catalog
metadata through `orwd list` / `orwd get`; it does not expose a SemVer-style
environment version in the catalog. Use the environment ID in `env_uid` and
record `updated_at` as source metadata.

For Harbor registry entries:

```yaml
env_uid: harbor:<dataset>/<env-or-task-slug>@<version-or-ref>
source:
  kind: harbor_registry
  hub_url: https://hub.harborframework.com/
  registry: https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json
  dataset: <dataset>
  env: <env-or-task-slug>
  source_repo: <org/repo>
  source_path: <path>
  source_ref: <commit-or-HEAD>
```

If a hosted environment is converted into checked-in BenchFlow task sources, add
`benchflow_uid` separately using the existing `{source.repo}:{source.path}@{source.ref}`
format. Do not replace `env_uid` with the converted task path.

## Compatibility Pass Levels

Use badges instead of binary support claims.

| Badge | Meaning |
|---|---|
| `parse` | BenchFlow can read or normalize the source metadata. |
| `package` | BenchFlow can materialize a task directory or use the source task directory directly. |
| `check` | `bench tasks check` passes. |
| `build` | Sandbox image or equivalent environment builds. |
| `oracle` | Oracle/reference path reaches expected reward. |
| `agent-smoke` | One cheap/mainstream agent can run end-to-end. |
| `reward-map` | Reward maps into `VerifyResult` / `RewardEvent`. |
| `trace` | BenchFlow captures trajectory/artifacts in its own format. |
| `parity` | Scores match the source framework under an agreed parity criterion. |

For GTM, `parse + package + check + reward-map` is enough to say
"compatibility smoke." For release-quality claims, require
`build + oracle + trace`. For benchmark-equivalence claims, require `parity`.

## Harbor Registry Showcase

Harbor is the most immediate ecosystem win because its hosted env/task entries
materialize into task packages that are already similar to BenchFlow tasks.
Current Harbor registry inspection found about 80 dataset entries and about
51,625 task records.

The right near-term target is:

> Two representative tasks per Harbor registry dataset, not all tasks.

This yields roughly 160 smoke tasks if every dataset has at least two tasks.
The sample should prefer:

- first task by registry order,
- one additional task from a different category/split when visible,
- CPU-only task when a benchmark has a separate GPU/Modal variant,
- parity subset when the adapter publishes one,
- skip or mark `blocked` for tasks requiring unavailable secrets, paid APIs,
  Windows containers, GPU, custom sidecars, or unsupported sandboxes.

Current Harbor env selections:

- `harbor:terminal-bench/adaptive-rejection-sampler@69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`: Terminal-Bench 2 task for shell/task packaging compatibility.
- `harbor:binary-audit/caddy-backdoor-detect@75f3e6e331776b80f77faa3d2ff80627b8b5d069`: security-style task for future hard-isolation and DinD adaptation checks.

The Firecracker/K8s DinD backlog lane uses
`harbor:termigen-environments/docker_escape_privileged_container_medium@dc329464161db64b0c670f46fa39b62e4719dddd`
as the concrete nested-container security smoke task. The lane remains blocked
and outside the current release gate until those sandbox backends are exposed by
the CLI and can run Docker-in-Docker workloads without special user
intervention.

### Harbor Runner Shape

Build a small compatibility runner:

```text
bench compat harbor-registry \
  --registry https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json \
  --tasks-per-dataset 2 \
  --level check \
  --out reports/compat/harbor-registry.jsonl
```

Initial implementation status: `bench compat harbor-registry` now supports
`inventory` and `check` levels, JSONL output, local registry files, remote
registry URLs, local task repos, and sparse-cloned remote Git task sources.

Initial levels:

1. `inventory`: fetch registry, list datasets, selected task refs, and blockers.
2. `check`: sparse-clone selected task paths and run `bench tasks check` one path at a time.
3. `oracle`: run selected tasks with oracle when `solution/solve.sh` exists and the sandbox is supported.
4. `agent-smoke`: run one cheap agent on a small strategic subset.

The output should be a machine-readable board:

```json
{
  "framework": "harbor",
  "env_uid": "harbor:quixbugs/quixbugs-java-bitcount@HEAD",
  "dataset": "quixbugs",
  "task": "quixbugs-java-bitcount",
  "source_repo": "laude-institute/harbor-datasets",
  "source_ref": "HEAD",
  "source_path": "datasets/quixbugs/quixbugs-java-bitcount",
  "badges": ["parse", "package", "check"],
  "blocked_reason": null,
  "notes": []
}
```

## Verifiers Integration Plan

Verifiers is strongest for RL-native hosted environment authoring: environment
rows become rollouts, rubrics provide reward functions, and rollout outputs include
trajectory/reward/timing/token data.

Phase 1 should not try to run all Verifiers environments inside BenchFlow. Start
with import and reward compatibility:

- Add optional extra: `benchflow[verifiers]`.
- Add `benchflow.adapters.verifiers`.
- Support two adapter modes:
  - `materialize`: convert a small Verifiers taskset/dataset sample into
    BenchFlow task directories.
  - `wrap`: run a Verifiers environment as an external reward/harness wrapper
    and normalize output into `RolloutResult`.
- Map Verifiers `Rubric` reward functions to BenchFlow `RewardFunc` where
  feasible; otherwise preserve them as external scorer calls.
- Preserve Verifiers rollout metadata as BenchFlow trajectory metadata rather
  than flattening it away.

Acceptance:

- two PrimeIntellect/Verifiers hosted envs inventoried or wrapped by `env_uid`,
- rewards normalize to `VerifyResult`,
- one generated or wrapped task produces BenchFlow-compatible result JSON.

Current PrimeIntellect env selections:

- `primeintellect:primeintellect/reverse-text@0.1.4`: lightweight single-turn text transformation with LCS reward.
- `primeintellect:primeintellect/math-python@0.1.10`: tool-using math environment that exercises Python tool calls through Prime sandboxes.

## ORS / OpenReward Integration Plan

ORS is the cleanest protocol bridge for RL environments. It owns sessions,
tools, task splits, finished signals, and rewards.

Phase 1 should be an ORS client wrapper:

- Add optional extra: `benchflow[ors]`.
- Add `benchflow.adapters.ors_client`.
- Import `list_splits`, `list_tasks`, `get_prompt`, `list_tools`.
- For a selected task, create a BenchFlow task wrapper whose harness calls ORS
  tools and records each tool result as a `RewardEvent`.
- Map ORS `finished=true` to rollout termination.
- Keep ORS server-side state external; BenchFlow owns trace/reward/artifact
  normalization.

Phase 2 can expose BenchFlow tasks as ORS servers:

- `POST /sessions`
- `GET /sessions/{id}/prompt`
- `GET /sessions/{id}/tools`
- `POST /sessions/{id}/call`
- `DELETE /sessions/{id}`

Acceptance:

- one representative ORS environment fixture modeled on a real hosted env runs through BenchFlow,
- one real OpenReward/KellyBench-style hosted env is inventoried or smoke-run by `env_uid`,
- tool outputs map to dense reward events,
- terminal reward maps to `VerifyResult.reward`.

Current OpenReward env selections:

- `openreward:GeneralReasoning/KellyBench@be14865a-3c70-422e-a2ba-f45c132cd29a`: long-horizon football betting environment with sandbox tools and deterministic reward.
- `openreward:GeneralReasoning/CTF@fcfcd0ef-1298-40e9-9492-83628fd98a1c`: sandboxed capture-the-flag environment with verifiable flag rewards.

## Inspect AI Integration Plan

Inspect is strong at task recipes: dataset + solver + scorer + sandbox. BenchFlow
should not initially try to become an Inspect solver runtime. The useful bridge
is task/scorer/sandbox compatibility.

Phase 1:

- Add optional extra: `benchflow[inspect]`.
- Add `benchflow.adapters.inspect_import`.
- Load an Inspect `@task` object and inspect its dataset, scorer, sandbox config,
  and limits.
- Materialize selected samples into BenchFlow task dirs when the scorer is
  simple enough.
- For scorer-only integration, wrap Inspect scorers behind `RewardFunc`.

Acceptance:

- two Inspect tasks imported or wrapped,
- one simple scorer maps to `RewardFunc`,
- task metadata and sandbox config are preserved in compatibility output.

## Product Surface

Add a new command group rather than hiding this under `bench eval create`:

```text
bench compat list-frameworks
bench compat harbor-registry --tasks-per-dataset 2 --level check
bench compat verifiers --env primeintellect/example --tasks 2 --level check
bench compat ors --url http://localhost:8080 --split test --tasks 2
bench compat inspect --task inspect_evals/gaia --samples 2
bench compat report reports/compat/*.jsonl
```

Keep `bench eval create` as the execution surface. `bench compat` is for
inventory, import, smoke validation, and reporting.

## Release/GTM Milestones

### Milestone 1: Truthful Compatibility Board

- Implement Harbor registry inventory + two-task selection.
- Implement `check` badge for Harbor tasks.
- Output JSONL + markdown report.
- Run against the full Harbor registry.

Public claim:

> BenchFlow can read and validate representative Harbor-format tasks across the
> Harbor registry.

### Milestone 2: Harbor Execution Smoke

- Add optional build/oracle levels.
- Run `build + oracle` on a curated subset:
  QuixBugs, CompileBench, Aider Polyglot, SpreadsheetBench
  if lightweight, one LLM-judge adapter if credentials are available.

Public claim:

> BenchFlow runs Harbor-style tasks through its native Rollout/Sandbox/Verifier
> path.

### Milestone 3: Framework Trio Demo

- One Verifiers env/taskset.
- One ORS environment.
- One Inspect task.
- One Harbor dataset sample.
- Unified report with rewards and trajectories.

Public claim:

> BenchFlow v0.4 interoperates with Harbor, Verifiers, ORS, and Inspect AI.

### Milestone 4: High-Fidelity Environment Differentiation

- Add Mockflow as the first-party high-fidelity environment pack layer.
- Show one Mock Gmail/Calendar/Drive/Slack workflow exposed through BenchFlow
  tasks and optionally through ORS.

Public claim:

> BenchFlow is not just an eval runner; it is the environment/runtime layer for
> high-fidelity agent benchmarks.

## Main Risks

- Calling thin converters "adapters" before they can import real external tasks
  will overclaim the v0.4 state.
- Harbor registry tasks may rely on secrets, GPUs, Windows, sidecars, custom
  agents, or provider-specific behavior; the board must record blockers instead
  of hiding them.
- ORS and Verifiers use live environment/session semantics; flattening them into
  static task dirs can lose the important part of the benchmark.
- Inspect tasks often include solvers; BenchFlow should import task/scorer
  semantics without pretending to be an Inspect runtime in phase 1.
- Full parity is expensive. Use compatibility badges so smoke support and parity
  support are not confused.

## Recommended Next Issue Split

1. `compat: Harbor registry inventory and two-task smoke board`
2. `compat: Harbor selected-task materializer and check runner`
3. `compat: ORS client wrapper for sessions/tools/rewards`
4. `compat: Verifiers taskset/rubric import spike`
5. `compat: Inspect task/scorer import spike`
6. `docs: mainstream framework compatibility page`
