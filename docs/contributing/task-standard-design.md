# Task standard: design notes

These are the design goals and the roadmap behind the [task package reference](../reference/task-package.md). **Nothing on this page runs unless the reference says so.** It is for contributors deciding where a new field belongs.

## Goals

A BenchFlow task is one `task.md` that selects a mode on each of three planes
(see Planes below): how the environment is built, how the agent interacts, and
how the result is scored. The standard has three goals:

1. **Native authoring** — humans and generators write one `task.md` (plus sidecar
   dirs) as the primary surface.
2. **Interoperability** — existing split-layout formats (`task.toml` +
   `instruction.md` + `solution/` + `tests/`) import directly, and packages export
   back to that layout with an explicit, honest loss report when a native concept
   has no equivalent.
3. **Coverage** — the schema can express the full range of eval shapes (single-shot,
   multi-round, simulated-user, multi-agent, and live-arena interaction; workspace,
   trajectory, rubric, judge, and leaderboard reward), even where the runtime for a
   given mode lands in a later milestone (see Roadmap rows).

Native authoring is the priority surface; import is best-effort-faithful; export is
best-effort with a loss report. The standard does not require bidirectional-lossless
round-tripping with any one external format.

## Field discipline

Every normative standard field must map to a mode on one of the three planes
(below) or to a concrete BenchFlow runtime need. Fields without that mapping
belong under `metadata` or in an adapter, not in the standard.

Do not add root keys for every new idea. Put draft or BenchFlow-specific extensions under `benchflow:` until they have a stable interface.

## Three views

The standard is intentionally split into three views:

| View | Purpose | Owner |
|---|---|---|
| Authoring document | What humans and generators write: one `task.md` plus sidecar dirs | `TaskDocument` |
| Runtime task view | What rollout, verifier, hardening, provenance, and trajectories consume | `TaskRuntimeView` plus first `TaskPackage` boundary |
| Foreign adapter view | What external benchmark formats and hosted environments import/export | adapters |

Do not treat these as the same interface. A good authoring document can include
more information than a foreign format can export, and a foreign import can
preserve unknown data that native authoring would reject.

## Planes

The package has six planes. Keeping them separate is the core abstraction.

| Plane | Owns | Should not own |
|---|---|---|
| Package | identity, versions, provenance, source hashes, export policy | sandbox execution |
| Runtime | environment image, resources, network policy, setup/reset/readiness, mounts | prompt orchestration |
| Interaction | agents, roles, teams, scenes, turns, user/nudge loops, handoff policy | verifier checkpoints |
| Verifier | verifier package entrypoint, reward file contract, hidden fixtures, scorer type, rubrics, judge roles, separate verifier env | agent prompt text |
| Oracle | held-out reference implementation and oracle-only env | tests/verifier code |
| Evidence | validation runs, flake data, anti-cheat review, artifact hashes, leaderboard metadata | mutable task source |

Imported `steps` are verifier/runtime checkpoints. BenchFlow `scenes` are
interaction checkpoints. They are orthogonal. A task can have both, but a
runtime must define how they compose before executing them.

Of the six planes, three are the **authoring axes** an author selects a mode from
— Runtime (the environment), Interaction, and Verifier — while the remaining three
(Package, Oracle, Evidence) are supporting planes. A task is one mode-selection
per axis: `environment` × `interaction-mode` × `verifier-strategy`. New normative
fields must add or compose a mode on one of these axes; otherwise they belong in
`metadata` or an adapter.

## Presets and profiles

Two different reuse mechanisms were historically both called "profile," which is
why the mental model felt overloaded. v0.6 separates them:

- **`preset`** — *authoring sugar*. A named bundle of defaults that `bench tasks
  normalize` expands into the canonical contract and then **discards**; it has no
  runtime existence. Presets are the lightweight-authoring path: a tiny `task.md`
  becomes a full contract. Examples: `code-change`, `acceptance-live`,
  `task-toml-compatible`. (The current implementation still spells the preset key as
  `profile:`; M0 renames it to `preset:` and keeps `profile:` as a deprecated
  alias.)
- **`*-profile`** — *runtime-real shared config*. A benchmark-level object that
  many tasks **reference** and that **persists into `TaskRuntimeView`**, one per
  authoring axis:
  - `environment-profile` — a shared world reused across tasks (e.g. one
    clawsbench service catalog referenced by all 44 tasks)
  - `agent-profile` — shared harness / model / role wiring
  - `verifier-profile` — shared reward strategy and rubric

Rule of thumb: if removing it changes only how much you typed, it is a `preset`;
if removing it changes what runs, it is a `*-profile`. Presets normalize away;
profiles are inherited. `*-profile` resolution is M1 runtime work; `preset` exists
today.

The `preset:` key and the `*-profile` objects are not implemented. Today the key is `profile:`.

## Agent and runtime policy

Agent isolation is distinct from task environment networking. A no-search task
can still need dependency, LLM, or provider egress outside the sandbox; a
no-network task can still need the agent harness to call its model.

```yaml
benchflow:
  agent_policy:
    skill_access: none          # none | installed | declared
    allowed_skill_roots: []
    search: disabled            # allowed | disabled
    forbidden_tools: [web_search, browser_fetch]
    trajectory_audit:
      require_no_forbidden_tool_calls: true
  runtime_policy:
    backend: modal              # docker | daytona | modal | kubernetes | podman | hpc | queue
    required_capabilities: [gpu:B200, private_mounts, persistent_state]
    network:
      task_default: no-network
      agent_egress: model-and-dependencies
      allowed_hosts: []
      phase_overrides:
        verifier: no-network
    private_mounts:
      - source: modal-volume://org-models/qwen
        target: /mnt/models/qwen
        mode: ro
        visibility: agent
        secret_ref: org-models-readonly
    persistent_state:
      required: true
      scope: task-run
      cleanup: after-verifier
```

Unsupported `agent_policy`, `runtime_policy`, private mounts, registry secrets,
GPU types, phase-specific network overrides, or persistent state semantics must
fail closed before launch for the selected sandbox. No-search proof must come
from both launch policy and trajectory audit; it is not implied by
`network_mode: no-network`.

## Verifier inputs and transfer

The richer standard should model verifier inputs explicitly:

```yaml
verifier:
  type: test-script
  timeout_sec: 900
benchflow:
  verifier:
    entrypoint: verifier/test.sh
    visibility: hidden
    inputs:
      - path: verifier/test.patch
        kind: test_patch
        visibility: hidden_verifier
    outputs:
      reward_text: /logs/verifier/reward.txt
      reward_json: /logs/verifier/reward.json
    transfer:
      agent_to_verifier:
        - /logs/artifacts/**
      verifier_to_result:
        - /logs/verifier/reward.*
        - /logs/verifier/artifacts/**
```

This covers `tests/test.patch`, Windows entrypoints, artifact-only graders,
separate verifier images, and hidden fixtures without overloading the directory
name.

Separate verifier execution must define image resolution, hidden fixture
mounting, step-level verifier environments, and transfer rules. Hidden verifier
inputs such as `tests/test.patch` or `verifier/test.patch` are mounted only for
the verifier phase; agent logs and workspace files move into verifier scope
only through declared artifact transfer paths.

`verifier/task.toml` as a checked copy of `verifier/verifier.md` is planned, not implemented.

Validation evidence should include parser checks, legacy-vs-native migration parity, live rollout artifacts, negative-contract failures, and explicit fail-closed results for parsed fields that the selected runtime cannot honor.

## Assets, provenance and secrets

Native task packages need more than source code. The standard should track
assets as first-class objects:

```yaml
benchflow:
  provenance:
    images:
      - field: sandbox.docker_image
        reference: ghcr.io/org/task-image:2026-06
        digest: sha256:...
        registry: ghcr.io
        provider: modal
        build_source:
          path: environment/Dockerfile
          sha256: "<hash>"
        credential_secret_ref: task-image-pull
        never_persist_credentials: true
  assets:
    - path: environment/dataset.parquet
      visibility: agent
      sha256: "<hash>"
      source:
        url: https://huggingface.co/datasets/org/name
        revision: "<commit>"
      license: apache-2.0
      generated: false
      mount_phase: agent
    - path: verifier/hidden_cases.jsonl
      visibility: hidden_verifier
      sha256: "<hash>"
      mount_phase: verifier
  secrets:
    - name: task-image-pull
      scope: image-pull
      visibility: runtime_secret
      never_persist: true
```

Suggested visibility values:

- `agent`
- `hidden_verifier`
- `hidden_oracle`
- `runtime_secret`
- `external_dataset`
- `evidence_only`

Borrow the discipline, not the package shape, from METR-style task standards:

- keep BenchFlow document-first rather than Python `TaskFamily`-first
- make asset visibility explicit instead of relying on directory folklore
- declare required secrets/resources with scope and "never persist" semantics
- record oracle/no-op/partial/human baseline scores where available
- keep protected/intermediate scoring behind explicit visibility controls

## Interaction modes

An interaction declares one **mode**. The modes form the Interaction axis:

| Mode | Meaning | Runtime |
|---|---|---|
| `single-shot` | one agent, one prompt, scored once | executable |
| `multi-round` | oracle access / progressive disclosure across rounds (`BaseUser`) | executable |
| `simulated-user` | a declarative, model-backed user persona drives turns (not a live wire peer) | partial (linear; one active role per scene) |
| `multi-agent-sequential` | roles hand off in one shared workspace | partial (sequential handoff only) |
| `arena-concurrent` | an assessor agent and the agent-under-test run as live A2A+MCP peers, scored *during* the interaction | target (runtime-deferred; see Roadmap rows) |

`arena-concurrent` models a live interaction where the reward is produced *during*
the exchange (by an assessor agent) rather than by a post-hoc verifier. It is
declared in the schema so such tasks parse without loss; its runtime (an A2A bridge
for the agent-under-test plus a concurrently-running assessor) is deferred to a
later milestone (see Roadmap rows). Note ACP (BenchFlow ↔ agent)
is not A2A (agent ↔ agent); arena needs the A2A leg.

Team semantics beyond one sequential hand-off (role membership enforcement, summaries, hand-off artifacts, parallel teams, branch routing, full trajectory sharing) are parsed as draft surface and must fail closed until a runtime owns them. `branch_execution: forked-snapshot` waits on the user loop's integration with snapshot branching.

## Compatibility goals

```yaml
benchflow:
  compatibility:
    task-toml:
      export: full
      emits:
        config: task.toml
        prompt: instruction.md
        oracle: solution/
        verifier: tests/
```

- Mixed native and legacy files should be invalid when their settings, instruction, oracle or verifier drift. Today `task.md` wins without a check.
- Validation should compare `verifier/` and `tests/` file maps and fail closed on drift.
- ORS and AgentBeats imports and exports belong at the adapter boundary: ORS tool-output rewards become declared reward-event artifacts plus a terminal aggregate, and AgentBeats assessor agents become verifier strategies.

## Roadmap rows

| Feature | Parse | Runtime | Next |
|---|---|---|---|
| `arena-concurrent` interaction (G4) | no | no | Interaction-mode schema now; A2A bridge for the agent under test plus a concurrent assessor at M2 |
| Hybrid reward envelope (G1) | partial | no | A declared cross-surface product or sum of factors; M1 |
| GAIN aggregation (G2) | no | no | A live baseline and ceiling; M1 |
| Leaderboard submission (G5) | partial | no | A hosted or hidden external scorer with a durable result record; M1 |
| RL step reward (G6) | no | no | Per-action environment reaction; M1/M2 |
| `benchflow:` | raw | no | A typed schema once v0.3 stabilizes |

## Architecture slices

P0: Add `TaskPackage` / `TaskRuntimeView`.

The first runtime-facing slices exist in `src/benchflow/task/runtime_view.py`
and `src/benchflow/task/package.py`. They answer:

- which entrypoint is authoritative
- what prompt goes into `/instruction.md` compatibility materialization
- which verifier/oracle directories are native versus legacy
- which scenes were parsed for execution
- which source hashes and compatibility metadata apply
- which verifier document and selected strategy apply
- which sandbox runtime issues block launch
- which compatibility export report describes target-format loss
- which prompt plan composes base, role, scene, and turn prompts for rollout,
  with redacted document-declared user metadata

The remaining package-boundary work is richer adapter import/export state,
acceptance/calibration validation levels, and freezing selected verifier/user
runtime semantics through launch instead of reparsing at every edge.

P1: Add fail-closed capability checks.

Rollout, verifier, hardening, and adapters should stop parsing fields they do
not execute without surfacing a validation result.

This includes explicit gates for `task.md` plus split-file drift,
`verifier/` plus `tests/`, and `oracle/` plus `solution/`.

The first module is `src/benchflow/task/runtime_capabilities.py` with a pure
validator:

```python
validate_task_runtime_support(task, *, sandbox, task_dir) -> list[UnsupportedTaskFeature]
```

It reports stable config paths and reasons for unsupported `steps`, root/step
`artifacts`, allowlists on backends that cannot enforce them, verifier allowlists, separate verifier environments (off Docker and Daytona), Windows, TPU,
healthchecks, unsafe workdirs, document-only `user`/`benchflow` runtime
semantics, and non-`main` verifier services on backends that cannot run them.
It is wired into `bench check --sandbox <backend>` and the shared sandbox
factory used by rollouts and `Environment.from_task()`. Unsupported parsed
semantics now raise `UnsupportedTaskFeatureError` before Docker, Daytona, or
Modal construction. Safe absolute non-root `sandbox.workdir` values are
materialized before agent and verifier setup.

P2: Split native and adapter validation modes.

Native authoring should be strict. Foreign import should preserve and warn.

P3: Type the `benchflow:` namespace.

Start with `document_version`, `compatibility`, `provenance`, `assets`,
`secrets`, `evidence`, `teams`, `nudges`, `prompt`, `agent_policy`, and
`runtime_policy`.

P4: Extend exporters.

The first `bench export` path exports to a compatibility split layout
with explicit loss reports, backed by `export_task_to_split_layout()`. Extend
the same reporting discipline to external benchmark datasets and same-format
no-op exports.

## Next steps

- [Task package reference](../reference/task-package.md): what runs today.
- [Tasks](../tasks/overview.md): the user chapter.
