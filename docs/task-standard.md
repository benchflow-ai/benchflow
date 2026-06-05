# BenchFlow Task Package Standard

Status: draft v0.3, 2026-06-05

This document defines the direction for BenchFlow-native task packages. The
short version: `task.md` is the native authoring entrypoint; `oracle/` and
`verifier/` are the BenchFlow-native names for held-out reference behavior and
reward checks. Harbor/Pier split names such as `solution/` and `tests/` remain
compatibility names and export targets.

The standard is intentionally split into three views:

| View | Purpose | Owner |
|---|---|---|
| Authoring document | What humans and generators write: one `task.md` plus sidecar dirs | `TaskDocument` |
| Runtime task view | What rollout, verifier, hardening, provenance, and trajectories consume | `TaskPackage` / `TaskRuntimeView` |
| Foreign adapter view | What Harbor, Pier, Terminal-Bench, SWE-bench, Inspect, and hosted envs import/export | adapters |

Do not treat these as the same interface. A good authoring document can include
more information than a foreign format can export, and a foreign import can
preserve unknown data that native authoring would reject.

## Fork User Needs

The standard is not trying to be different from Harbor for its own sake. It is
trying to keep Harbor-compatible task authors, forks, and extensions from
solving the same problems in incompatible side channels.

| User / fork family | Need observed in the ecosystem | Current support | Target requirement |
|---|---|---|---|
| Harbor upstream task authors | Stable split layout, network policy, separate verifier envs, multi-step reward, artifacts | Harbor-shaped config parses; split layout still loads | Keep root config Harbor-compatible and add fail-closed runtime gates/export conformance |
| Harbor Reward Kit users | Directory-structured reward components, programmatic checks, LLM judges, agent judges, criterion isolation, reward details | Script verifier runs; Reward Kit can be used manually through `test.sh` | Native verifier package with `verifier/verifier.md`, reward directories, rubrics, judge roles, and reward-details preservation |
| Pier / DeepSWE | Harbor-format long-horizon tasks, installed agents, air-gapped runs, egress allowlists, critique trajectories | Harbor-compatible imports can load known fields | Future exporters emit Pier-compatible split layout; `benchflow:` carries agent/runtime/evidence extensions |
| FrontierSWE `harbor_ext` | Private images, Modal execution, GPU-heavy jobs, no-search agent semantics, persistent volumes, structured reward JSON | Resource fields parse; rich runtime policy is metadata-only | Runtime policy, assets, backend capability gates, and rich reward/evidence preservation |
| Alibaba / registry-oriented forks | Image-building, registry, ACK/provider integration, task image provenance | Image fields parse as config/metadata | Package and asset planes record image/source provenance and provider metadata |
| LLM360 / Kubernetes-style forks | Queue/Kubernetes providers, model-serving backends, SGLang-style infra | Provider-specific fields are not native runtime support | Runtime plane stays separate from authoring; unsupported backend capabilities fail closed |
| BitFun / agent-integration forks | New CLI agents, viewer/API additions, agent config examples | `agents.roles` declares harness choices | `benchflow:` carries viewer/trajectory metadata without root-key creep |
| Marina / OpenHands-oriented forks | OpenHands SDK runner and distribution changes near upstream | Adapters/runners coexist with task loading | Keep foreign adapter view separate from native authoring |
| TypeScript ports | Language-neutral task format pressure | `task.md` is static markdown/YAML | Define schema as language-neutral spec, not Python object semantics |
| Janelia HPC / Podman forks | Podman, HPC/LSF scripts, non-Docker resource scheduling | Resources parse but backend support varies | Runtime capability matrix distinguishes parse support from executable backend support |
| FormulaCode / Sentient benchmark forks | New benchmark adapters and publishing flows | Source metadata exists; export loss reports are target work | Package-level source provenance, compatibility reports, and degraded export reports |
| Refresh multimodal / CUA forks | Multimodal trajectory viewers, screenshots/video/browser evidence | Artifacts/log paths exist; multimodal semantics are target work | Evidence/artifact plane includes screenshots, HAR/video, and structured trajectories |
| Portage / provider-divergent forks | Large runtime/provider divergence without new task semantics | Provider concerns live outside root task config | Keep provider concerns in runtime/adapters rather than root task syntax |
| Watch-only forks | Visible forks with no substantive public task-format divergence | No standard fields added for them | No new standard fields without concrete user need |
| METR-style task developers | Asset visibility, protected scoring, secrets, QA evidence | `benchflow:` can carry raw metadata | Steal validity/evidence discipline while keeping BenchFlow document-first |
| AgentBeats / ORS evaluator authors | Assessor agents, episode rewards, tool-call trajectories, multi-agent participants, leaderboard-ready result records | Not native; can be represented as metadata today | Verifier package can declare assessor-agent and ORS-style reward roles without turning the whole task into a fixed test script |

This table is the acceptance test for new fields: if a field cannot be tied to
one of these user needs or a concrete BenchFlow architecture need, it belongs in
`metadata` or in an adapter, not in the standard.

## Fork User Stories And Acceptance Criteria

The fork table above is intentionally user-centered. These are the concrete
stories the standard has to satisfy.

| ID | User story | Standard surface | Acceptance proof |
|---|---|---|---|
| F1 | As a Harbor/Pier task author, I can mirror a split `task.toml` + `instruction.md` + `solution/` + `tests/` task as native `task.md` + `oracle/` + `verifier/` and export it back without semantic loss. | root Harbor config, `task.md`, `oracle/`, `verifier/`, `benchflow.compatibility` | canonical config diff passes, prompt text normalizes equal, oracle/verifier file maps hash equal, export report has no losses for Harbor-compatible fields |
| F2 | As a Pier/DeepSWE benchmark runner, I can run long-horizon tasks with installed agents, no-internet defaults, allowlisted egress, and critique/evidence traces without hidden side channels. | `agent`, `environment.network_mode`, allowlists, `agents.roles`, `benchflow.evidence` | no-network and allowlist negative tests fail closed; no-skill runs prove no skill files were read; trajectory metadata includes critique/evidence artifacts |
| F3 | As a FrontierSWE task owner, I can express private assets, no-search policy, GPU/backend requirements, separate verifier execution, persistent state, and rich scoring without forking root task syntax. | `verifier.environment`, resources, `benchflow.runtime_policy`, `benchflow.private_mounts`, `benchflow.assets`, `reward.json` | sandbox capability check refuses unsupported Modal/GPU/private-mount semantics; supported backends preserve structured reward JSON and verifier evidence |
| F4 | As a provider fork maintainer for Alibaba, LLM360, Janelia, or similar infrastructure, I can add registry, Kubernetes, Podman, HPC, and queue backends without turning backend settings into universal authoring keys. | runtime plane, backend capability matrix, image/source provenance, `benchflow.runtime_policy` | unsupported backend fields produce explicit parse-only or fail-closed diagnostics; supported backends record image hashes, registry/provider metadata, and resource decisions |
| F5 | As an agent/viewer fork maintainer for BitFun, Marina/OpenHands, Refresh, or CUA/browser tasks, I can declare agents, teams, simulated users, multimodal evidence, and viewer metadata without losing Harbor export compatibility. | `agents.roles`, `scenes`, `user`, `benchflow.teams`, `benchflow.nudges`, `benchflow.evidence` | scenes compile deterministically; user loops are executable or marked metadata-only; screenshots/HAR/video/trajectory artifacts appear in result metadata |
| F6 | As an adapter or dataset maintainer for FormulaCode, Sentient, SWE-bench, Terminal-Bench, Inspect, or TypeScript tooling, I can preserve source provenance and get an honest report when a target format cannot express native semantics. | `source`, `metadata`, `benchflow.provenance`, `benchflow.assets`, `benchflow.compatibility` | export loss reports list every dropped native concept; language-neutral schema validation does not depend on Python object names |
| F7 | As a maintainer watching inactive or shallow forks, I do not want fields added just because a fork exists. | no new surface | every proposed field cites F1-F8 or a BenchFlow runtime need |
| F8 | As a verifier/rubric author, I can package deterministic checks, Reward Kit criteria, LLM judges, agent judges, and ORS-style episode rewards under one verifier entrypoint. | `verifier/verifier.md`, `verifier/test.sh`, reward directories, judge/rubric metadata, `reward.json`, `reward-details.json` | verifier discovery finds the selected strategy; judge credentials stay verifier-scoped; deterministic and judge criteria can run in isolation; reward details and trajectories are preserved |

Field traceability rule: every normative standard field must cite one or more
story IDs from F1-F8 or an explicit BenchFlow runtime need. Fields without a
cited need remain adapter metadata or ordinary `metadata`.

## Native Layout

```text
task/
|-- task.md
|-- environment/
|   `-- Dockerfile
|-- verifier/
|   |-- verifier.md
|   `-- test.sh
|-- oracle/
|   `-- solve.sh
`-- evidence/
    `-- validation.json
```

Compatibility aliases:

| Native | Compatibility / foreign export | Meaning |
|---|---|---|
| `task.md` | `task.toml` + `instruction.md` | Task config plus prompt material |
| `oracle/` | `solution/` | Held-out reference implementation |
| `verifier/` | `tests/` | Verifier package, reward code, hidden checks, rubrics, and judges |

Target validation: native packages may carry both native and compatibility
directories only when hashes prove the duplicate content is equivalent. Current
runtime prefers the native spelling when both aliases exist; it does not yet
prove equivalence.

Within a BenchFlow-native package, selection is fail-closed:

- if `task.md` exists, it is the authoritative task definition
- if `verifier/` exists, it is the authoritative verifier directory; an empty
  or invalid `verifier/` does not fall back to `tests/`
- if `oracle/` exists, it is the authoritative oracle directory; an empty or
  invalid `oracle/` does not fall back to `solution/`
- duplicate alias trees must be byte-identical after normalized traversal or
  validation should report a collision

This does not deprecate split layouts. Foreign adapters can still load and emit
plain Harbor/Pier `task.toml` + `instruction.md` packages directly.

## Versioning

There are two versions:

```yaml
schema_version: "1.3"        # Harbor-compatible task config surface
benchflow:
  document_version: "0.3"    # BenchFlow task.md document syntax
```

`schema_version` is for the runtime config model shared with Harbor-style
`task.toml`. `benchflow.document_version` is for document-only concepts such as
teams, prompt composition, agent policy, runtime policy, private assets,
provenance, evidence, nudges, and export policy.

## Root Frontmatter

The root frontmatter has three classes of keys.

Harbor-compatible config keys are modeled by `TaskConfig` and must be rejected
when unknown in native authoring mode:

- `schema_version` / `version`
- `task`
- `metadata`
- `agent`
- `verifier`
- `environment`
- `oracle` (validation alias: `solution`)
- `source`
- `artifacts`
- `steps`
- `multi_step_reward_strategy`

Document orchestration keys are parsed by `TaskDocument`:

- `agents`
- `scenes`
- `user`

BenchFlow extension keys live under the reserved namespace:

- `benchflow`

Do not add new root keys for every new idea. Put draft or BenchFlow-specific
extensions under `benchflow:` until they have a stable interface.

Native authoring should use `oracle`. Importers may accept the compatibility
`solution` alias, but a config that contains both names is invalid.

## Prompt Body

Markdown sections are part of the document syntax:

| Section | Meaning |
|---|---|
| `## prompt` | Base agent-facing prompt |
| `## role:<name>` | Role prompt |
| `## scene:<name>` | Scene prompt |
| `## user-persona` | Simulated user persona |

Current runtime precedence is simple fallback:

1. inline turn prompt
2. scene prompt
3. role prompt
4. base prompt

The target standard should make composition explicit instead:

```yaml
benchflow:
  prompt:
    composition: append
    order: [base, role, scene, turn]
```

That avoids losing role guardrails when a scene prompt is present. `replace`
should be allowed only when the task author states the override explicitly.

## Agent And Runtime Policy

Fork stories: F2, F3, F4, F5.

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
      - source: modal-volume://frontierswe-models/qwen
        target: /mnt/models/qwen
        mode: ro
        visibility: agent
        secret_ref: frontier-models-readonly
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

Harbor `steps` are verifier/runtime checkpoints. BenchFlow `scenes` are
interaction checkpoints. They are orthogonal. A task can have both, but a
runtime must define how they compose before executing them.

## Verifier Package

Fork stories: F1, F2, F3, F5, F8.

The verifier is a peer package, not just a `tests/` directory. Native verifier
packages should have their own entry document:

```text
verifier/
|-- verifier.md
|-- task.toml
|-- test.sh
|-- rewards/
|   |-- correctness/
|   |   `-- reward.py
|   `-- quality/
|       `-- criteria.toml
|-- rubrics/
|   |-- verifier.md
|   `-- verifier.toml
|-- judges/
|   `-- reviewer.md
`-- fixtures/
    `-- hidden_cases.jsonl
```

`verifier/verifier.md` is analogous to `task.md` for the evaluation side. It
describes what evidence is read, which strategies can score the task, how
rubric dimensions compose, which judge agents are allowed, and what outputs the
runtime must preserve.

`verifier/task.toml` is an optional compatibility projection of
`verifier/verifier.md`; it is not a second native surface. If both files exist,
their canonical projection must match or validation fails closed.

```md
---
document_version: "0.3"
verifier:
  name: hidden-patch-and-quality
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
    rewardkit:
      type: reward-kit
      root: rewards/
      criteria: rewards/quality/criteria.toml
    judge:
      type: agent-judge
      role: verifier_judge
      model: gpt-5.5
      inputs: [trajectory/acp_trajectory.jsonl, /logs/artifacts/patch.diff]
      isolation: verifier-only
    ors:
      type: ors-episode
      reward_aggregation: last_non_null
      terminal_policy: finished_true
  rubric:
    combine: weighted_sum
    dimensions:
      correctness: {weight: 0.7, source: deterministic}
      maintainability: {weight: 0.2, source: judge}
      evidence_quality: {weight: 0.1, source: rewardkit}
    files:
      human: rubrics/verifier.md
      structured: rubrics/verifier.toml
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      field: reward
      fallback: weighted_mean
---

## role:verifier_judge

Grade only the submitted artifact and declared evidence. Do not infer intent
from private oracle files or hidden verifier fixtures.
```

`test.sh` is the minimum executable strategy and the Harbor/Pier export target.
Reward Kit-style criteria, ORS-style episode rewards, and AgentBeats-style
assessor agents are verifier strategies. They must run inside verifier scope,
not the agent scope, and they must emit either the canonical reward envelope or
a declared multi-metric map.

Verifier packages must be isolated:

- judge models and credentials are verifier-scoped, not agent-visible
- judge prompts live under `verifier/`, not inside the task prompt
- rubrics are separate from judge persona: `rubrics/verifier.md` is the human
  scoring contract, `rubrics/verifier.toml` or JSON is the structured scoring
  contract, and `## role:verifier_judge` is the assessor posture
- hidden fixtures and rubrics are mounted only during verifier execution
- agent-as-judge trajectories are recorded as evidence and audited for input
  leakage
- unsupported verifier strategies fail closed before scoring starts

## Verifier Contract

Native verifier code lives in `verifier/` and is mounted at `/verifier`.
Compatibility `tests/` remains supported and is mounted at `/tests`.

The minimum script verifier contract applies to the selected verifier
entrypoint: native packages with no `verifier/verifier.md` run
`verifier/test.sh`; imported split-layout packages may run `tests/test.sh`.

- write `/logs/verifier/reward.txt` with one float from `0.0` to `1.0`
- optionally write `/logs/verifier/reward.json` with structured rubric/evidence
  once rich JSON preservation is implemented
- prefer exit `0` after writing reward
- treat nonzero verifier exit without a fresh reward file as infrastructure
  failure; a nonzero exit with a fresh reward is a scored task result with
  verifier diagnostics

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

Target reward precedence:

- `reward.txt` remains the Harbor-compatible scalar minimum
- when `reward.json` exists, it should be the authoritative rich reward artifact
- `reward.json` may be an envelope with a numeric `reward`, or a Harbor Reward
  Kit-style map of named numeric rewards such as `{"correctness": 0.75,
  "quality": 0.9}`
- if both files exist and `reward.json` has a scalar aggregate, the scalar must
  match `float(reward.txt)` or validation should fail closed
- if `reward.json` is a multi-metric map, verifier metadata must declare the
  aggregate policy used for scalar exports and training records
- `reward-details.json` is preserved when present (copied through verifier
  rollout download without parsing)
- reward artifacts should preserve structured reserved keys such as `rubric`,
  `items`, `evidence`, `artifacts`, `metadata`, `reason`, `reasons`, `errors`,
  and task-specific payloads such as `metrics`, `regressions`, `participants`,
  `winner`, `raw`, and `debug`

Current runtime still does not fully match that target: `reward.json` is now
authoritative when present and scalar mismatches against `reward.txt` fail
closed, but structured top-level JSON remains limited and multi-metric maps
without a declared aggregate policy are not yet first-class.

Validation evidence should include parser checks, legacy-vs-native migration
parity, live rollout artifacts, negative-contract failures, and explicit
fail-closed results for parsed fields that the selected runtime cannot honor.

## Oracle Contract

Native oracle code lives in `oracle/` and is mounted at `/oracle`.
Compatibility `solution/` remains supported and is mounted at `/solution`.

The naming change is semantic:

- `oracle` means "held-out reference behavior used to prove solvability"
- `solution` is a compatibility name for Harbor/Pier/Terminal-Bench exports

Proposed outbound exporters should map native `oracle/` to foreign
`solution/`. Current runtime supports native and compatibility oracle paths, but
outbound export and equivalence enforcement are still target behavior.

## Assets, Provenance, And Evidence

Native task packages need more than source code. The standard should track
assets as first-class objects:

```yaml
benchflow:
  provenance:
    images:
      - field: environment.docker_image
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

Evidence should prove task validity, not just package shape:

```yaml
benchflow:
  evidence:
    oracle_runs:
      required_reward: 1.0
      last_job: jobs/task-standard/oracle/...
    verifier:
      reruns: 5
      flake_rate: 0.0
    review:
      anti_cheat: passed
      instruction_alignment: passed
      reviewer: benchflow
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      partial_solution_range: [0.3, 0.8]
      human_or_reference_examples:
        - name: gold
          expected_reward: 1.0
          artifact: evidence/calibration/gold-result.json
      judge_agreement:
        required: true
        sample_count: 5
        min_pairwise_agreement: 0.8
    trajectories:
      - path: trajectory/acp_trajectory.jsonl
        kind: acp
        visibility: evidence_only
        sha256: "<hash>"
      - path: trajectory/critique.jsonl
        kind: critique
        visibility: evidence_only
        sha256: "<hash>"
    artifacts:
      - path: artifacts/session.har
        kind: har
        mime_type: application/json
        produced_by: browser
        visibility: evidence_only
        sha256: "<hash>"
```

Borrow the discipline, not the package shape, from METR-style task standards:

- keep BenchFlow document-first rather than Python `TaskFamily`-first
- make asset visibility explicit instead of relying on directory folklore
- declare required secrets/resources with scope and "never persist" semantics
- record oracle/no-op/partial/human baseline scores where available
- keep protected/intermediate scoring behind explicit visibility controls

Valid evidence artifact kinds include `screenshot`, `video`, `har`,
`browser_trace`, `trajectory`, `critique`, `viewer_metadata`, and
`verifier_artifact`.

Oracle and calibration evidence is required for leaderboard-grade tasks and
recommended for all native tasks with hidden tests, subjective rubrics, or
LLM/agent-as-judge scoring.

## Interaction Model

The current executable slice is linear `scenes` that reference `agents.roles`.
The target model adds teams and handoff policy without changing the basic role
declaration:

```yaml
agents:
  roles:
    planner:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
    implementer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
benchflow:
  teams:
    default:
      roles: [planner, implementer]
      handoff:
        trajectory_visibility: summaries
        workspace_visibility: shared
        artifacts: [/app/plan.md]
```

Simulated users and nudges should be explicit about runtime type:

```yaml
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
benchflow:
  nudges:
    mode: simulated-user
    branchable: true
    confirmation_policy:
      destructive_actions: human
```

If a runtime cannot compile `user` into a concrete loop, it should mark the task
as parsed-only rather than silently ignoring the user.

## Compatibility

Compatibility must be explicit. A native package can guarantee export only for
the subset the target format supports.

```yaml
benchflow:
  compatibility:
    harbor:
      export: full
      emits:
        config: task.toml
        prompt: instruction.md
        oracle: solution/
        verifier: tests/
    pier:
      export: degraded
      losses:
        - benchflow.teams
        - benchflow.nudges
    terminal_bench:
      export: degraded
      losses:
        - scenes
        - separate_verifier_environment
```

Target compatibility rules, not all implemented today:

1. Native authoring rejects unknown root config keys.
2. Adapter import should preserve unknown foreign keys under
   `benchflow.compat.extra` and warn.
3. Export should emit a degraded-export report when the target format cannot
   express a native concept.
4. Mixed native/legacy files should be invalid unless hashes prove equivalence.
5. Harbor/Pier export should emit `task.toml`, `instruction.md`, `solution/`,
   and `tests/`.
6. BenchFlow import should prefer `task.md` only when compatibility metadata
   proves the legacy files are equivalent.
7. Export reports include selected definition, selected verifier/oracle dirs,
   ignored aliases, input/output file hashes, and any lost semantics.
8. `tests/` compatibility is path compatibility, not a separate native
   standard. For native packages, `verifier/` is authoritative. For split
   packages, `tests/` remains valid and may contain only `test.sh`.
9. Exporters map the selected native verifier tree to `tests/`; importers may
   preserve split-layout `tests/` without requiring `verifier.md` or rubric
   files. If both `verifier/` and `tests/` exist, validation should compare
   normalized file maps and fail closed on drift.
10. ORS/AgentBeats imports/exports live at the adapter boundary: ORS rewards
   become reward events plus a terminal aggregate, and AgentBeats assessor
   agents become verifier strategies rather than root task syntax.

Round-trip guarantees are semantic, not byte-exact. Config equivalence should
be checked through canonical `TaskConfig` dumps; prompt equivalence through
normalized prompt text; verifier/oracle equivalence through deterministic
SHA-256 file maps over regular files. Comments, TOML/YAML formatting, and blank
line trivia are not preserved unless a same-format no-op export asks for that.

## Runtime Capability Matrix

Current implementation status:

| Feature | Parse | Runtime | Next gate |
|---|---:|---:|---|
| `task.md` prompt | yes | yes | keep |
| native `verifier/` | yes | yes | keep |
| native `oracle/` | yes | yes | keep |
| `agents.roles` | yes | partial | move scene adoption into `TaskRuntimeView` |
| `scenes` | yes | partial | define prompt composition and handoff policy |
| `user` / `## user-persona` | yes | no | compile to a concrete simulated-user loop or mark metadata-only |
| `benchflow:` | raw | no | typed document schema after v0.3 stabilizes |
| Harbor `steps` | yes | no/partial | fail closed per sandbox until implemented |
| root/step artifacts | yes | no/partial | implement collection or fail closed |
| network allowlist | yes | no/partial | per-sandbox capability check |
| separate verifier env | yes | no/partial | materializer plus verifier runner support |
| Windows / TPU | yes | no | fail closed |
| healthcheck / workdir | yes | no/partial | materializer support |
| `reward.json` precedence | yes | yes | keep scalar agreement checks; add multi-metric aggregate policy |
| `reward-details.json` preservation | yes | yes | copy through verifier rollout download; no parsing yet |
| `TaskPackage` / `TaskRuntimeView` | yes | partial | entrypoint, dirs, scenes, alias collisions, `verifier_document`; prompt composition and sandbox gates still partial |
| runtime capability gates (docker/daytona) | yes | partial | `validate_task_runtime_support` wired before sandbox creation; more backends and `bench tasks check --sandbox` pending |
| Harbor/Pier export | yes | partial | native `task.md` export with loss reports; round-trip equivalence and alias parity pending |
| native `/task.md` upload | yes | yes | keep `/instruction.md` compatibility contract |
| alias dir collision checks | yes | partial | `bench tasks check` and `Task()` load fail closed; export parity pending |

Today `bench tasks check` is structural. A future sandbox-aware check, for
example `bench tasks check --sandbox <backend>`, should validate this matrix. A
parsed field that the selected runtime cannot honor is worse than a parse
error.

## Architecture Slices

P0: `TaskPackage` / `TaskRuntimeView` (partial).

This module should answer:

- which entrypoint is authoritative
- what prompt goes into `/instruction.md` compatibility materialization
- which verifier/oracle directories are native versus legacy
- which scenes are executable
- which Harbor fields are unsupported for the selected sandbox
- which source hashes and compatibility metadata apply
- whether mixed native/legacy definitions and alias directories are equivalent
  or collisions

P1: Fail-closed capability checks (partial on docker/daytona).

Rollout, verifier, hardening, and adapters should stop parsing fields they do
not execute without surfacing a validation result.

This includes explicit gates for `task.md` plus split-file drift,
`verifier/` plus `tests/`, and `oracle/` plus `solution/`.

The first module should be `src/benchflow/task/runtime_capabilities.py` with a
pure validator such as:

```python
validate_task_runtime_support(task, sandbox_type, task_path) -> list[UnsupportedTaskFeature]
```

It should report stable config paths and reasons for unsupported `steps`,
root/step `artifacts`, allowlists, separate verifier environments, Windows,
TPU, healthchecks, workdir, and non-`main` verifier services on backends that
cannot run them. Wire it before sandbox creation in rollout setup and in
`Environment.from_task()`. Keep ordinary `bench tasks check` structural, then
add a proposed `bench tasks check --sandbox <backend>` path for capability
validation.

P2: Split native and adapter validation modes.

Native authoring should be strict. Foreign import should preserve and warn.

P3: Type the `benchflow:` namespace.

Start with `document_version`, `compatibility`, `provenance`, `assets`,
`secrets`, `evidence`, `teams`, `nudges`, `prompt`, `agent_policy`, and
`runtime_policy`.

P4: Add exporters.

Export to Harbor/Pier, Terminal-Bench, Inspect, and SWE-bench-style datasets
with explicit loss reports.
