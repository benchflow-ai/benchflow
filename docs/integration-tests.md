# Integration Tests

On-demand end-to-end tests that validate BenchFlow against real benchmark suites. Not part of CI — invoke manually before trial-ready releases and before large runtime refactors.

The core matrix runs 9 SkillsBench tasks across all 8 registered agents on Daytona. Release readiness also requires smoke coverage for the current adapter release set, the current feature release set, hosted environment compatibility, and Terminal-Bench-style tasks so BenchFlow keeps running existing suites even as public API names move to Rollout/Sandbox terminology.

## Prerequisites

Install Daytona sandbox support for local integration runs:

```bash
uv sync --extra dev --extra sandbox-daytona --locked
```

| Variable | Required for |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | gemini, pi-acp, openclaw, opencode, openhands |
| `DAYTONA_API_KEY` | all agents (sandbox) |
| `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` | claude-agent-acp |
| `OPENAI_API_KEY` | codex-acp |

This table covers the default integration suite models. Provider-specific
integration lanes may require different credentials; Azure Foundry lanes use
`AZURE_API_KEY` plus `AZURE_API_ENDPOINT`.

## Quick Start

```bash
# All 8 agents in parallel (each runs 9 tasks concurrently on Daytona)
export GEMINI_API_KEY=... DAYTONA_API_KEY=... CLAUDE_CODE_OAUTH_TOKEN=... OPENAI_API_KEY=...
tests/integration/run.sh

# Specific agents only
tests/integration/run.sh gemini pi-acp claude-agent-acp

# Review results from a previous run (no API calls)
tests/integration/run.sh --check-only

# Large validation override
BENCHFLOW_INTEGRATION_CONCURRENCY=100 tests/integration/run.sh gemini
```

## What It Does

1. **Resolves tasks** — downloads the full SkillsBench task set, then creates a symlinked subset of 9 selected tasks.
2. **Launches agents in parallel** — each agent is started as a background process running `bench eval create` with concurrency=64 by default. Set `BENCHFLOW_INTEGRATION_CONCURRENCY=100` for the large post-migration validation run.
3. **Waits and reports** — as each agent finishes, prints its score line. After all complete, runs `check_results.py` to validate output schema and print the results table.

## Release Readiness Coverage

Before cutting a trial-ready release, every current release blocker must pass. If any adapter, feature, hosted environment board entry, or first-party sandbox in the current release set fails validation, do not publish the release.

| Release blocker | Purpose | Minimum acceptance |
|---|---|---|
| SkillsBench agent matrix | Validates the full agent × task pipeline on Daytona | 9 selected tasks across all credentialed agents produce valid `result.json`, trajectory output, and summary schema |
| Adapter release set | Validates merged and open benchmark adapters preserve source-suite semantics | Each benchmark listed in [Running Adapted Benchmarks](./running-benchmarks.md#available-benchmarks), plus each open benchmark-adapter PR targeted at the release, has at least one representative smoke or parity run with verifier execution and reward schema validation |
| Terminal-Bench smoke | Validates Terminal-Bench-style task packaging and shell verifier behavior | At least one representative task runs through `bench eval create`, verifier executes from the task tests, and sandbox hardening does not break normal task execution |
| Trace-to-task | Validates `bench tasks generate` can turn real traces into runnable benchmark tasks | Generate from at least one local or JSONL trace and one HuggingFace/opentraces source, then run the generated task through `bench eval create` and validate the verifier outcome |
| Agent decoupling | Validates agents are pluggable runtime targets rather than core-coupled implementations | Core import and task validation work without optional agent packages; at least two non-oracle agents run the same representative task through the same Rollout path |
| Sandbox decoupling | Validates sandbox backends are pluggable and optional dependencies remain isolated | Core `import benchflow` works without optional sandbox extras; Docker and Daytona smoke runs use the same task contract; missing optional sandbox deps produce clear install guidance |
| Release-gated sandbox support | Validates the release-quality sandbox backends in the current release gate | Docker and Daytona can run a representative task via `bench eval create --sandbox docker` and `--sandbox daytona` without special user intervention beyond each backend's documented auth; Modal remains optional follow-up evidence, not a release blocker |

These blockers test benchmark-suite portability, not backward compatibility with old BenchFlow names. Public docs and new configs should use Rollout/Sandbox terminology.

For the current release-prep sweep, the adapter release set is keyed by benchmark UID. The UID is derived from the BenchFlow task source as `{source.repo}:{source.path}@{source.ref}` so display names and PR titles are never the primary identity.

| Benchmark UID | Name | Status |
|---|---|---|
| `benchflow-ai/benchmarks:datasets/harvey-lab/tasks@main` | Harvey LAB | Merged adapter |
| `benchflow-ai/benchmarks:datasets/programbench/tasks@main` | ProgramBench | Merged adapter |
| `benchflow-ai/skillsbench:tasks@main` | SkillsBench | Merged adapter |
| `benchflow-ai/benchmarks:datasets/hilbench/tasks@main` | HILBench | Open adapter PR [#279](https://github.com/benchflow-ai/benchflow/pull/279) |
| `benchflow-ai/benchmarks:datasets/opaquetoolsbench/tasks@main` | OpaqueToolsBench | Open adapter PR [#280](https://github.com/benchflow-ai/benchflow/pull/280) |
| `benchflow-ai/benchmarks:datasets/continuallearningbench/tasks@main` | ContinualLearningBench | Open adapter PR [#283](https://github.com/benchflow-ai/benchflow/pull/283) |

HILBench uses the Hugging Face dataset `ScaleAI/hil-bench` for task metadata,
but its SWE image tarballs live in the Hugging Face bucket
`ScaleAI/hil-bench-swe-images`. Adapter code should treat dataset
`repo_or_db_download_link` values such as
`hf://buckets/ScaleAI/hil-bench-swe-images/images/<uid>.tar.zst` as bucket
objects and fetch them through
`https://huggingface.co/buckets/ScaleAI/hil-bench-swe-images/resolve/images/<uid>.tar.zst`,
not through dataset `hf_hub_download`.

Hosted environment registries such as PrimeIntellect, OpenReward/ORS, and Harbor registry should not use benchmark UIDs unless their envs have been converted into checked-in BenchFlow task sources. Track those with `env_uid` plus the canonical `hub_url` in the compatibility board, and record a separate `benchflow_uid` only after conversion. The current hub URLs are `https://openreward.ai/environments`, `https://hub.harborframework.com/`, and `https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections`.

The current OpenReward compatibility-board selections are `openreward:GeneralReasoning/KellyBench@be14865a-3c70-422e-a2ba-f45c132cd29a` for long-horizon sandbox/tool use and `openreward:GeneralReasoning/CTF@fcfcd0ef-1298-40e9-9492-83628fd98a1c` for security sandbox coverage. The current PrimeIntellect selections are `primeintellect:primeintellect/reverse-text@0.1.4` for lightweight/no-secret inventory and `primeintellect:primeintellect/math-python@0.1.10` for tool-use plus Prime sandbox coverage. The current Harbor selections are `harbor:terminal-bench/adaptive-rejection-sampler@69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` for Terminal-Bench-style packaging and `harbor:binary-audit/caddy-backdoor-detect@75f3e6e331776b80f77faa3d2ff80627b8b5d069` for security-style adaptation.

OpenReward exposes stable environment IDs and `updated_at` timestamps in its catalog rather than semantic environment versions, so the compatibility board uses the environment ID in `env_uid` and records `updated_at` separately. OpenReward session-level smoke may still require account credits even when catalog inventory succeeds.

For the current release-prep sweep, the feature release set includes:

| Feature | Scope |
|---|---|
| Trace-to-task | `bench tasks generate` from local sessions, JSONL traces, and HuggingFace/opentraces datasets |
| Agent decoupling | Agents are selected and configured outside core benchmark/runtime logic |
| Sandbox decoupling | Sandbox backends are optional, selectable, and isolated behind the Sandbox contract |
| Release-gated sandboxes | Docker and Daytona are release-gated through `--sandbox docker` and `--sandbox daytona`; Modal may remain selectable but is not a current release blocker |
| Future hard-isolation backlog | Firecracker and Kubernetes are tracked in the backlog profile, not the current release gate. Promote them only after `--sandbox firecracker`, `--sandbox k8s`, and the `--sandbox kubernetes` alias are implemented with install guidance and smoke evidence |

## Large Test Suite Structure

The large suite should be built from explicit axes and named lanes, not a full Cartesian product. Agent × model × sandbox × task multiplies too quickly, so each lane must state what risk it covers.

The declarative source of truth for the release-ready suite is [`tests/integration/suites/release.yaml`](../tests/integration/suites/release.yaml). Runners should load named lanes from that manifest rather than hardcoding matrix logic in shell scripts.

Plan the release suite before running anything:

```bash
uv run python tests/integration/run_suite.py --list-lanes
uv run python tests/integration/run_suite.py --list-profiles
uv run python tests/integration/run_suite.py --profile near-term --dry-run
uv run python tests/integration/run_suite.py --profile release-gated-cli --dry-run --fail-on-todo
uv run python tests/integration/run_suite.py --profile hosted-envs --dry-run
uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo
# The backlog profile is expected non-zero until Firecracker/K8s are promoted.
uv run python tests/integration/run_suite.py --profile backlog --dry-run --fail-on-todo
uv run python tests/integration/run_suite.py --lane shared-sandbox-smoke --dry-run
```

`run_suite.py` starts dry-run-first and wires execution lane by lane. Adapter,
hosted-env, and trace-to-task evidence are currently executable:

```bash
# Regenerate ENG-93 trace-to-task evidence, including Docker oracle evals.
uv run python tests/integration/run_suite.py --lane trace-to-task-e2e --execute-trace-evidence --run-trace-eval

# Validate adapter evidence from checked-in parity artifacts, a SkillsBench result,
# and one worktree per open adapter PR.
uv run python tests/integration/run_suite.py --lane adapter-release-set --execute-adapter-evidence \
  --skillsbench-result dogfood/.../result.json \
  --open-pr-root HILBench=/path/to/pr-279-worktree \
  --open-pr-root OpaqueToolsBench=/path/to/pr-280-worktree \
  --open-pr-root ContinualLearningBench=/path/to/pr-283-worktree

# Validate hosted env hub metadata and regenerate Harbor inventory evidence.
uv run python tests/integration/run_suite.py --lane hosted-env-compatibility-board --execute-hosted-env-evidence
```

Trace-to-task evidence writes generated tasks, oracle job outputs, and `trace-evidence.json` under `dogfood/2026-05-19-trace-to-task-e2e/` by default. The evidence directory is declared in [`tests/integration/suites/release.yaml`](../tests/integration/suites/release.yaml) and can be overridden with `--trace-evidence-dir`.

Hosted-env evidence writes `hosted-env-evidence.json` plus Harbor registry JSONL
inventory under `dogfood/2026-05-19-release-gate/hosted-envs/` by default.
OpenReward and PrimeIntellect remain hub-metadata checks until account/credited
hosted eval support is available; Harbor has a public registry inventory path.

SkillsBench-vs-Harbor parity is an offline checker over existing artifacts. It
does not clone `benchflow-ai/skillsbench-trajectories` during the release run
because that baseline repository is large. Supply a local checkout or extracted
artifact root pinned to `2d86fe82f6a06f7c7b3a22a3ae90d554d0e9655c`:

```bash
uv run python tests/integration/run_suite.py --lane skillsbench-harbor-parity \
  --execute-skillsbench-harbor-parity \
  --skillsbench-harbor-benchflow-root jobs/integration-<run-id>/<agent> \
  --skillsbench-harbor-baseline-root /path/to/skillsbench-trajectories/shenghan/gemini-cli/gemini3flash/noskills \
  --skillsbench-harbor-task jax-computing-basics \
  --skillsbench-harbor-task python-scala-translation \
  --skillsbench-harbor-task jpg-ocr-stat \
  --skillsbench-harbor-task grid-dispatch-operator \
  --skillsbench-harbor-task threejs-to-obj \
  --skillsbench-harbor-task data-to-d3 \
  --skillsbench-harbor-task lake-warming-attribution \
  --skillsbench-harbor-task weighted-gdp-calc \
  --skillsbench-harbor-task shock-analysis-supply
```

The checker normalizes the expected schema differences explicitly: Harbor
rewards come from `verifier_result.rewards.reward` and ATIF trajectories from
`agent/trajectory.json`, while BenchFlow rewards come from `rewards.reward` and
ACP trajectories from `trajectory/acp_trajectory.jsonl`. It fails on missing
tasks, malformed artifacts, missing trajectories, unseen task outcomes, per-task
reward movement outside the Harbor observed range, and aggregate
outcome/reward-rate drift over the configured thresholds.

To refresh the Harbor pin, fetch the baseline repository, inspect the candidate
run root, run the parity checker against known BenchFlow evidence, and update
`tests/integration/suites/release.yaml` plus the command above only after the
new baseline is accepted:

```bash
git -C /path/to/skillsbench-trajectories fetch origin main
git -C /path/to/skillsbench-trajectories rev-parse origin/main
```

Use `--fail-on-todo` whenever a dry-run plan is being used as release evidence. Despite the name, this gate fails on unresolved TODOs and explicit `blocked_by` entries. The `near-term`, `release-gated-cli`, `hosted-envs`, and `full-release` profiles are expected to pass the gate today. The `backlog` profile is expected to fail this gate until the future Firecracker/Kubernetes security DinD lane is resolved; that failure tracks planned coverage, not a current release blocker.

### Run Tracking and Profiles

Future release-suite runs should be tracked in Linear. Until that is wired into automation, `tests/integration/suites/release.yaml` is the source of truth for run intent and the job output paths are the evidence artifact.

The `near-term` profile keeps release prep moving with a smaller plan: SkillsBench as the benchmark-suite focus, Daytona as the preferred cloud sandbox, and the adapter release set as the additional feature surface. This profile is not the full release gate; `full-release` includes every current release blocker.

The `release-gated-cli` profile is the TODO-clean release-planning profile for the current release-gated CLI surface. It adds the shared sandbox smoke, Terminal-Bench-style shell verifier smoke, and decoupling checks over the Docker/Daytona surface without treating optional Modal or future Firecracker/Kubernetes support as release blockers.

The `backlog` profile tracks planned validation lanes with concrete tasks and acceptance criteria. It currently holds the Firecracker/Kubernetes Docker-in-Docker smoke because those sandboxes are not part of the current release gate.

Adapter coverage should stay intentionally small in the near-term profile: one representative smoke or parity task per adapter unless the result is ambiguous or fails in a way that needs narrowing.

### Axes

| Axis | Examples | Notes |
|---|---|---|
| Agent | `gemini`, `claude-agent-acp`, `codex-acp`, `opencode`, `openhands` | Agents are selected independently from core runtime logic |
| Model | `gemini-3.1-flash-lite-preview`, `gpt-5.4-nano`, Claude model IDs | Model coverage should be representative, not every model on every lane |
| Sandbox | `docker`, `daytona`; optional `modal`; future `firecracker`, `k8s` / `kubernetes` | Docker/Daytona get the current release smoke lane; Modal is optional follow-up evidence; Firecracker/K8s stay in backlog hard-isolation lanes until implemented |
| Task set | SkillsBench subset, Terminal-Bench smoke, security DinD smoke, generated trace tasks | Task sets should be named and versioned |
| Benchmark adapter | Harvey LAB, ProgramBench, SkillsBench, open adapter PRs | Adapter lanes validate source-suite semantics and verifier behavior; benchmark identity is the source-derived UID, not the display name |
| Hosted env hub | OpenReward environments, Harbor Hub, PrimeIntellect Environments Hub | Hosted env lanes track `env_uid` patterns, selected envs, and canonical hub URLs |
| Feature | Trace-to-task, agent decoupling, sandbox decoupling | Feature lanes validate product/runtime behavior directly |

### Required Lanes

| Lane | Axis slice | Purpose |
|---|---|---|
| Shared sandbox smoke | 1 boring representative task × oracle or one cheap agent × Docker/Daytona | Proves the same task contract works across every current release-gated sandbox |
| SkillsBench agent matrix | 9 SkillsBench tasks × credentialed agents × default model × Daytona | Proves real agent execution across the registered-agent surface |
| Adapter release set | 1 representative smoke/parity run per merged adapter and open adapter PR | Proves every adapter in the release set preserves source-suite semantics |
| Hosted env compatibility board | OpenReward, Harbor Hub, PrimeIntellect hub entries × selected env UIDs or remaining selection TODOs | Proves hosted env catalogs are tracked as envs, not benchmark sources |
| Terminal-Bench smoke | 1 Terminal-Bench-style task × Docker and one cloud sandbox | Proves shell verifier and packaging compatibility |
| Trace-to-task e2e | local/JSONL trace + HuggingFace/opentraces source × generated tasks × Docker or Daytona | Proves generated tasks are runnable, not just emitted |
| Decoupling checks | core import/task validation without optional packages; representative task on at least two agents and two sandboxes | Proves agents and sandboxes are not hard-coupled to core |

### Backlog Lanes

| Lane | Axis slice | Promote when |
|---|---|---|
| Security DinD smoke | `harbor:termigen-environments/docker_escape_privileged_container_medium@dc329464161db64b0c670f46fa39b62e4719dddd` × Firecracker/K8s | `bench eval create --sandbox firecracker` and `bench eval create --sandbox k8s` run the task without questions, Docker-in-Docker-style service/container operations work inside both sandboxes, and the verifier executes after sandbox hardening |

### Coverage Policy

Release blockers are mandatory lanes inside the large suite. Backlog lanes must also name a real task and activation criteria, but they do not block the current release gate until promoted. Broader nightly or pre-merge runs can add pairwise coverage across agent/model/sandbox/task axes, but release decisions should stay tied to named lanes with explicit pass/fail evidence.

## Architecture

```
tests/integration/
├── run.sh              # Shell driver — parallel agent launch + wait
├── run_suite.py        # Manifest-aware planner and lane evidence dispatcher
├── check_results.py    # Result validator — schema checks + score table
├── check_adapter_evidence.py
├── check_trace_to_task_evidence.py
├── suites/
│   └── release.yaml    # Declarative release-blocker suite
└── configs/            # Per-agent YAML configs for standalone use
    ├── gemini.yaml
    ├── claude-agent-acp.yaml
    └── ...
```

Output lands in `jobs/integration/<agent>/`:
```
jobs/integration/
├── gemini/
│   ├── 2026-05-15__16-43-54/    # run directory
│   │   ├── jax-computing-basics__abc123/
│   │   │   ├── result.json
│   │   │   ├── trajectory/acp_trajectory.jsonl
│   │   │   └── ...
│   │   └── ...
│   └── summary.json
├── claude-agent-acp/
│   └── ...
└── .logs/                        # per-agent stdout/stderr logs
    ├── gemini.log
    └── ...
```

## Selected Tasks

The 9 tasks (3 low / 3 medium / 3 high complexity):

| Task | Complexity |
|---|---|
| jax-computing-basics | Low |
| python-scala-translation | Low |
| jpg-ocr-stat | Low |
| grid-dispatch-operator | Medium |
| threejs-to-obj | Medium |
| data-to-d3 | Medium |
| lake-warming-attribution | High |
| weighted-gdp-calc | High |
| shock-analysis-supply | High |

## Agents

All 8 registered agents run by default:

| Agent | Default Model | Notes |
|---|---|---|
| claude-agent-acp | claude-haiku-4-5-20251001 | Needs `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` |
| codex-acp | gpt-5.4-nano | Needs `OPENAI_API_KEY` |
| pi-acp | gemini-3.1-flash-lite-preview | |
| openclaw | gemini-3.1-flash-lite-preview | |
| gemini | gemini-3.1-flash-lite-preview | |
| opencode | gemini-3.1-flash-lite-preview | |
| harvey-lab-harness | gemini-3.1-flash-lite-preview | |
| openhands | gemini-3.1-flash-lite-preview | |

Agents missing credentials are automatically skipped.

The notes above describe the default integration configs. Provider-prefixed
model lanes can override the native agent auth requirements; Azure Foundry
models use `AZURE_API_KEY` plus `AZURE_API_ENDPOINT`.

## Standalone YAML Configs

Each agent has a YAML config in `configs/` with an `include` list restricting to the 9 selected tasks. These can be used directly with `bench eval create --config`:

```bash
uv run bench eval create --config tests/integration/configs/gemini.yaml
```

`run.sh` uses CLI arguments instead of these configs for parallel execution, but both approaches run the same 9 tasks at the active-dev concurrency of 64. The shell runner can be temporarily raised with `BENCHFLOW_INTEGRATION_CONCURRENCY=100` for the larger validation set.

## Result Validation

`check_results.py` checks:
- Every `result.json` has required fields (`task_name`, `agent`, `rewards`, `error`, `verifier_error`)
- No infrastructure errors (sandbox failures vs. task failures)
- `summary.json` exists with required keys (`total`, `passed`, `failed`, `errored`, `score`)

```bash
# Run validator standalone
uv run python tests/integration/check_results.py jobs/integration gemini pi-acp
```

## Agent-as-Judge Verification

`agent_judge.py` adds a second, model-based signal on top of the mechanical
schema checks. Given a completed rollout directory, it reuses BenchFlow's own
`call_judge` primitive (default model `gemini-3.1-flash-lite`) to grade whether
the run is a trustworthy measurement: the agent genuinely attempted the task,
the trajectory is coherent, and there is no obvious reward-hacking. The judge
reads `result.json` plus the recorded `trajectory/acp_trajectory.jsonl`, and
treats the trajectory as untrusted evidence rather than as instructions.

The gate combines two requirements:

1. **Realness** — the run is REAL only when `n_tool_calls > 0`, token usage
   `> 0`, and the reward is non-null. These mechanical invariants hold
   independently of the judge: a judge pass cannot rescue an unreal run.
2. **Agent judge** — the judge must return a `pass` verdict. It is
   fail-closed: a missing provider SDK, an API error, or an unparseable or
   fieldless verdict all read as FAIL, never a silent pass.

```bash
# Judge a rollout dir (or a jobs root to search for the latest rollout)
uv run python tests/integration/agent_judge.py jobs/integration-eval --json
```

The lightweight `.github/workflows/integration-eval.yml` workflow runs one
small task through `bench eval create --agent openhands --model
deepseek/deepseek-v4-flash --sandbox docker`, then runs the agent judge over
the rollout and fails the job if the run is not REAL or the judge fails. It
triggers on `workflow_dispatch` and nightly, and is capped at one task to keep
the check cheap. It references the `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and
`GEMINI_API_KEY` repository secrets; the judge SDKs come from the `judge`
extra (`uv sync --extra judge`).

## Robust integration suite (`tests/test_integration_suite.py`)

The agent-judge gate above is the seed; `tests/test_integration_suite.py` grows
it into a scenario suite whose checks are each grounded in a v0.6 dogfooding
finding. Reusable building blocks live in `tests/integration/scenarios.py`
(`run_eval`, `reward_of`, `synth_rollout`, the ATIF/ADP/secret-leak validators,
`reaper_dryrun_issues`). The suite has two tiers.

### Deterministic integrity gates (run in the normal suite, no credentials)

These exercise the gate over **synthetic rollout fixtures**, so they are fast,
deterministic, and run in regular CI — closing the gaps a single happy-path
live check leaves open:

| Gate | What it pins | Dogfooding origin |
|---|---|---|
| Realness gate | rejects a run that was scored but did no work (`n_tool_calls=0`), had no telemetry, was never scored, or errored | a "passed" run that is really resumed/empty results |
| Fail-closed judge | a missing SDK, API error, or unparseable verdict reads FAIL, never a silent pass | judge infra error silently recorded as reward 0.0 |
| Reward-hacking caught | a mechanically-REAL rollout (tool calls, tokens, reward 1.0) still fails when the judge flags verifier tampering | reward-hacking only the judge can see |
| Example `judge.py` fail-closed | the shipped generated-skill-eval judge exits non-zero (writes no reward) when no LLM judge can run | ENG-254 reward-integrity fix |
| Artifact integrity | ATIF is `ATIF-v1.x` with recognized step sources; ADP lines parse; no provider key leaks into any artifact | ATIF/ADP schema + secret-redaction findings |

```bash
# Runs with the normal test job; no keys, no sandbox.
uv run pytest tests/test_integration_suite.py -m "not integration"
```

### Live scenarios (`@pytest.mark.integration`, nightly / on demand)

Real sandbox + provider runs, each skipping cleanly when its prerequisites
(Docker daemon, `DAYTONA_API_KEY`, DeepSeek / Gemini keys) are absent:

| Scenario | Assertion | Dogfooding origin |
|---|---|---|
| `oracle_determinism_docker` | every oracle `solve.sh` self-scores reward 1.0 | the broken `3d-scan-calc` example oracle (ENG-256) |
| `sandbox_parity_docker_daytona` | the same oracle task scores identically on Docker and Daytona | the rollout/daytona package split |
| `agent_rollout_is_real_and_judged` | a real openhands + deepseek-v4-flash run is REAL and the agent judge runs over it; the verdict is recorded, not gated, since model success is stochastic | resumed/empty/idle-timeout shells |
| `reaper_dryrun_is_safe` | `environment cleanup --dry-run` deletes nothing; foreign sandboxes are never reaped | destructive-reaper scoping |

```bash
export DEEPSEEK_API_KEY=... DEEPSEEK_BASE_URL=https://api.deepseek.com \
       GEMINI_API_KEY=... DAYTONA_API_KEY=...
uv run pytest tests/test_integration_suite.py -m integration
```
