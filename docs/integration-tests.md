# Integration Tests

On-demand end-to-end tests that validate BenchFlow against real benchmark suites. Not part of CI — invoke manually before trial-ready releases and before large runtime refactors.

The core matrix runs 9 SkillsBench tasks across all 8 registered agents on Daytona. Release readiness also requires smoke coverage for the full adapter release set, the feature release set, and Terminal-Bench-style tasks so BenchFlow keeps running existing suites even as public API names move to Rollout/Sandbox terminology.

## Prerequisites

| Variable | Required for |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | gemini, pi-acp, openclaw, opencode, openhands |
| `DAYTONA_API_KEY` | all agents (sandbox) |
| `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` | claude-agent-acp |
| `OPENAI_API_KEY`, `CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, or `~/.codex/auth.json` | codex-acp |

## Quick Start

```bash
# All 8 agents in parallel (each runs 9 tasks concurrently on Daytona)
export GEMINI_API_KEY=... DAYTONA_API_KEY=... CLAUDE_CODE_OAUTH_TOKEN=... OPENAI_API_KEY=...
tests/integration/run.sh

# Specific agents only
tests/integration/run.sh gemini pi-acp claude-agent-acp

# Review results from a previous run (no API calls)
tests/integration/run.sh --check-only
```

## What It Does

1. **Resolves tasks** — downloads the full SkillsBench task set, then creates a symlinked subset of 9 selected tasks.
2. **Launches agents in parallel** — each agent is started as a background process running `bench eval create` with concurrency=30.
3. **Waits and reports** — as each agent finishes, prints its score line. After all complete, runs `check_results.py` to validate output schema and print the results table.

## Release Readiness Coverage

Before cutting a trial-ready release, every release blocker must pass. If any adapter, feature, or first-party sandbox in the release set fails validation, do not publish the release.

| Release blocker | Purpose | Minimum acceptance |
|---|---|---|
| SkillsBench agent matrix | Validates the full agent × task pipeline on Daytona | 9 selected tasks across all credentialed agents produce valid `result.json`, trajectory output, and summary schema |
| Adapter release set | Validates merged and open benchmark adapters preserve source-suite semantics | Each benchmark listed in [Running Adapted Benchmarks](./running-benchmarks.md#available-benchmarks), plus each open benchmark-adapter PR targeted at the release, has at least one representative smoke or parity run with verifier execution and reward schema validation |
| Terminal-Bench smoke | Validates Terminal-Bench-style task packaging and shell verifier behavior | At least one representative task runs through `bench eval create`, verifier executes from the task tests, and sandbox hardening does not break normal task execution |
| Trace-to-task | Validates `bench tasks generate` can turn real traces into runnable benchmark tasks | Generate from at least one local or JSONL trace and one HuggingFace/opentraces source, then run the generated task through `bench eval create` and validate the verifier outcome |
| Agent decoupling | Validates agents are pluggable runtime targets rather than core-coupled implementations | Core import and task validation work without optional agent packages; at least two non-oracle agents run the same representative task through the same Rollout path |
| Sandbox decoupling | Validates sandbox backends are pluggable and optional dependencies remain isolated | Core `import benchflow` works without optional sandbox extras; Docker and Daytona smoke runs use the same task contract; missing optional sandbox deps produce clear install guidance |
| First-party sandbox support | Validates release-quality sandbox backends needed for benchmark runs and security benchmarks | Docker, Daytona, Modal, Firecracker, and Kubernetes can run representative tasks via `bench eval create --sandbox docker`, `--sandbox daytona`, `--sandbox modal`, `--sandbox firecracker`, and `--sandbox k8s` / `--sandbox kubernetes` without special user intervention. Firecracker and Kubernetes also support Docker-in-Docker-style workloads suitable for adapting ExploitBench and CVEBench |

These blockers test benchmark-suite portability, not backward compatibility with old BenchFlow names. Public docs and new configs should use Rollout/Sandbox terminology.

For the current release-prep sweep, the adapter release set includes:

| Adapter source | Benchmarks |
|---|---|
| Merged adapters | Harvey LAB, ProgramBench, SkillsBench |
| Open adapter PRs | HILBench ([#279](https://github.com/benchflow-ai/benchflow/pull/279)), OpaqueToolsBench ([#280](https://github.com/benchflow-ai/benchflow/pull/280)), CLBench ([#283](https://github.com/benchflow-ai/benchflow/pull/283)) |

For the current release-prep sweep, the feature release set includes:

| Feature | Scope |
|---|---|
| Trace-to-task | `bench tasks generate` from local sessions, JSONL traces, and HuggingFace/opentraces datasets |
| Agent decoupling | Agents are selected and configured outside core benchmark/runtime logic |
| Sandbox decoupling | Sandbox backends are optional, selectable, and isolated behind the Sandbox contract |
| First-party sandboxes | Docker, Daytona, Modal, Firecracker, and Kubernetes support selectable through `--sandbox docker`, `--sandbox daytona`, `--sandbox modal`, `--sandbox firecracker`, and `--sandbox k8s` / `--sandbox kubernetes`. Firecracker and Kubernetes include Docker-in-Docker-style workload support for ExploitBench/CVEBench-style benchmark adaptation |

## Large Test Suite Structure

The large suite should be built from explicit axes and named lanes, not a full Cartesian product. Agent × model × sandbox × task multiplies too quickly, so each lane must state what risk it covers.

The declarative source of truth for the trial-ready release suite is [`tests/integration/suites/release.yaml`](../tests/integration/suites/release.yaml). Runners should load named lanes from that manifest rather than hardcoding matrix logic in shell scripts.

Plan the release suite before running anything:

```bash
uv run python tests/integration/run_suite.py --list-lanes
uv run python tests/integration/run_suite.py --list-profiles
uv run python tests/integration/run_suite.py --profile near-term --dry-run
uv run python tests/integration/run_suite.py --lane shared-sandbox-smoke --dry-run
```

`run_suite.py` is intentionally dry-run-only until execution semantics are wired lane by lane.

### Run Tracking and Profiles

Future release-suite runs should be tracked in Linear. Until that is wired into automation, `tests/integration/suites/release.yaml` is the source of truth for run intent and the job output paths are the evidence artifact.

The `near-term` profile keeps release prep moving with a smaller plan: SkillsBench as the benchmark-suite focus, Daytona as the preferred cloud sandbox, and the adapter release set as the additional feature surface. This profile is not the full release gate; `full-release` still includes every release blocker.

Adapter coverage should stay intentionally small in the near-term profile: one representative smoke or parity task per adapter unless the result is ambiguous or fails in a way that needs narrowing.

### Axes

| Axis | Examples | Notes |
|---|---|---|
| Agent | `gemini`, `claude-agent-acp`, `codex-acp`, `opencode`, `openhands` | Agents are selected independently from core runtime logic |
| Model | `gemini-3.1-flash-lite-preview`, `gpt-5.4-nano`, Claude model IDs | Model coverage should be representative, not every model on every lane |
| Sandbox | `docker`, `daytona`, `modal`, `firecracker`, `k8s` / `kubernetes` | All first-party sandboxes get a shared smoke lane |
| Task set | SkillsBench subset, Terminal-Bench smoke, security DinD smoke, generated trace tasks | Task sets should be named and versioned |
| Adapter | Harvey LAB, ProgramBench, SkillsBench, open adapter PRs | Adapter lanes validate source-suite semantics and verifier behavior |
| Feature | Trace-to-task, agent decoupling, sandbox decoupling | Feature lanes validate product/runtime behavior directly |

### Required Lanes

| Lane | Axis slice | Purpose |
|---|---|---|
| Shared sandbox smoke | 1 boring representative task × oracle or one cheap agent × Docker/Daytona/Modal/Firecracker/K8s | Proves the same task contract works across every first-party sandbox |
| Security DinD smoke | 1 security-style nested-container task × Firecracker/K8s | Proves Docker-in-Docker-style workloads work where ExploitBench/CVEBench-style tasks need them |
| SkillsBench agent matrix | 9 SkillsBench tasks × credentialed agents × default model × Daytona | Proves real agent execution across the registered-agent surface |
| Adapter release set | 1 representative smoke/parity run per merged adapter and open adapter PR | Proves every adapter in the release set preserves source-suite semantics |
| Terminal-Bench smoke | 1 Terminal-Bench-style task × Docker and one cloud sandbox | Proves shell verifier and packaging compatibility |
| Trace-to-task e2e | local/JSONL trace + HuggingFace/opentraces source × generated tasks × Docker or Daytona | Proves generated tasks are runnable, not just emitted |
| Decoupling checks | core import/task validation without optional packages; representative task on at least two agents and two sandboxes | Proves agents and sandboxes are not hard-coupled to core |

### Coverage Policy

Release blockers are mandatory lanes inside the large suite. Broader nightly or pre-merge runs can add pairwise coverage across agent/model/sandbox/task axes, but release decisions should stay tied to named lanes with explicit pass/fail evidence.

## Architecture

```
tests/integration/
├── run.sh              # Shell driver — parallel agent launch + wait
├── run_suite.py        # Manifest-aware dry-run planner
├── check_results.py    # Result validator — schema checks + score table
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
| codex-acp | gpt-5.4-nano | Needs `OPENAI_API_KEY`, `CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, or host Codex login |
| pi-acp | gemini-3.1-flash-lite-preview | |
| openclaw | gemini-3.1-flash-lite-preview | |
| gemini | gemini-3.1-flash-lite-preview | |
| opencode | gemini-3.1-flash-lite-preview | |
| harvey-lab-harness | gemini-3.1-flash-lite-preview | |
| openhands | gemini-3.1-flash-lite-preview | |

Agents missing credentials are automatically skipped.

## Standalone YAML Configs

Each agent has a YAML config in `configs/` with an `include` list restricting to the 9 selected tasks. These can be used directly with `bench eval create --config`:

```bash
uv run bench eval create --config tests/integration/configs/gemini.yaml
```

`run.sh` uses CLI arguments instead of these configs for parallel execution, but both approaches run the same 9 tasks.

## Result Validation

`check_results.py` checks:
- Every `result.json` has required fields (`task_name`, `agent`, `rewards`, `error`, `verifier_error`)
- No infrastructure errors (sandbox failures vs. task failures)
- `summary.json` exists with required keys (`total`, `passed`, `failed`, `errored`, `score`)

```bash
# Run validator standalone
uv run python tests/integration/check_results.py jobs/integration gemini pi-acp
```
