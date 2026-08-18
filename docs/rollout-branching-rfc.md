# RFC: Composed environment snapshotting + rollout branching

- **Status:** Draft for review
- **Tracking:** [FrontierPhysics #73](https://github.com/benchflow-ai/FrontierPhysics/issues/73) (the `[12pt - rollout]` infra ticket from the 2026-08-13 sync)
- **Author:** Jicheng Wang (@JeremyJC67)

## 1. Motivation

A failed rollout today is an undifferentiated zero. The FrontierPhysics failure cascade
names four stages a research-style agent run can die in — **(1) env/tool init, (2)
research, (3) execution, (4) self-judgment** — and reviewers need to attribute a failure
to a stage, and to ask counterfactuals: *would this run have passed with the skill pack?
with the tool available? with the oracle's plan?*

The mechanism for both is the same: **snapshot the rollout at stage boundaries, then
branch — resume from a snapshot with exactly one controlled change — and diff the
outcomes.** This RFC specifies that mechanism as a composition of subsystems benchflow
already ships, plus the missing glue.

Concretely, after this RFC a task author or reviewer can run:

- *skills ablation without a full re-run*: branch the env-init snapshot into a
  `with-skill` child and a `no-skill` child; every other bit of the world is identical.
- *tool-outage perturbation*: branch with an environment-manifest delta
  (the documented `env0@prod` vs `env0@outage` pattern) at any stage boundary.
- *plan injection*: branch the post-research boundary with the oracle's `PLAN.md`
  substituted, separating "researched wrong" from "executed wrong".
- *stage attribution for a real failure*: replay the recorded trajectory to a cut-point,
  then go live under a delta, localizing the first stage whose fix flips the outcome.

## 2. What already exists (this RFC composes; it does not invent)

| Substrate | Where | State |
|---|---|---|
| Branch engine: `Rollout.branch(n)` — quiesce → checkpoint → fork → restore → aggregate over a tree-native rollout | `src/benchflow/rollout_branch.py`, `src/benchflow/branch.py`, `docs/architecture.md` ("Branch lifecycle") | **Live**, but checkpoints the Environment plane only |
| Container snapshots: `Sandbox.snapshot()/restore()`, `SandboxImage`, `supports_snapshot` (fail-closed default) | `src/benchflow/sandbox/protocol.py` (from #384/#470); Docker = `docker commit`, Daytona-direct = provider snapshots | **Live**, but the branch engine gates on `supports_snapshot` and never calls it |
| Environment-state snapshots (declared sqlite files, `sqlite3 .backup`, fail-closed `EnvironmentSnapshotError`) | `src/benchflow/environment/manifest_env.py` (from #387/#486) | **Live**; sole restore point used by the branch engine |
| Record-replay of a finished run: replay `llm_trajectory.jsonl` responses by index through a proxy into a fresh sandbox | `src/benchflow/continue_run/` (`bench eval continue`) | **Live** (openhands-only), replays the full prefix only |
| Per-run variation axes: env-registry refs (S-axis), allowlisted `--config-override` patches (C-axis), `skill_mode` | `environment/_registry/`, `_utils/config_override.py`, `rollout/_config.py` | **Live**, but bound once at rollout setup — no per-child variation |
| Task-authoring surface: `branch_execution: forked-snapshot` | `docs/task-standard.md` | **Declared, fails closed** — waiting for exactly this machinery |

The gap is stated in `docs/architecture.md`: *"container and agent-session checkpoint
composition remain future work."* Branch children today also leave **no artifacts** —
no per-child `result.json`, no serialized tree, no record of what differed.

## 3. Design

### 3.1 Composed checkpoints (three layers, fixed order)

A **stage snapshot** is the ordered composition already sketched in `architecture.md`:

```
quiesce (agent disconnected / not yet connected)
  → environment.snapshot()          # declared state (sqlite), fail-closed integrity
  → sandbox.snapshot()              # container filesystem (docker commit / provider)
  → record StageSnapshot{stage, env_ref, sandbox_ref, meta}
```

Restore is symmetric and reversed: restore container, then environment state, then
(for service-topology deltas) stop/start services around the state restore, reusing
`ManifestEnvironment.reset()` semantics.

**Capability discipline (unchanged pattern):** each layer keeps its `supports_*` flag
and typed error (`SandboxSnapshotNotSupported`, `EnvironmentSnapshotError`). A branch
request declares which layers it requires; missing capability **fails closed with a
diagnostic**, never silently degrades. This generalizes today's behavior instead of
changing it: an env-state-only checkpoint (the current engine) remains expressible as
`require_layers={"environment"}`. It also *relaxes* today's hard constraint that any
branch requires declared sqlite state: a stateless env + snapshot-capable sandbox can
branch with `require_layers={"sandbox"}`.

Agent-session state is **explicitly layer three and out of scope for v1** (§7).

### 3.2 Stage boundaries map onto existing lifecycle phases

No new phase system. The four cascade stages pin to existing transitions in
`rollout/__init__.py`:

| Cascade stage | Lifecycle boundary | Snapshot point |
|---|---|---|
| env/tool init | end of `start()` (sandbox up, env plane provisioned, readiness gate passed) — **before `install_agent()`** | `env-ready` |
| research | cursor/step boundary inside `execute()` (e.g. the step that finalizes `PLAN.md`) | `post-research` |
| execution | agent finished / quiesced, **before `harden_before_verify`** | `pre-verify` |
| self-judgment | after verify, before review | `post-verify` |

`env-ready` deliberately precedes `install_agent()` so skills-on/off branches re-run
skill deployment from a skill-free world (skills are baked in at install time).
Mid-`execute()` boundaries are cursor positions in the existing tree — the branch
engine already forks at the cursor; this RFC adds *named* cut-points recorded as
stage-tagged exchange indices (§3.5).

Snapshot policy is opt-in per run (`--snapshot-stages env-ready,pre-verify`) or per
task (`sandbox:`-spelled frontmatter, post-#966 naming).

### 3.3 Per-child deltas (reuse the three run-level axes)

A branch child's delta is a recorded tuple; every member reuses an existing,
content-addressed mechanism:

```python
BranchDelta(
    environment_ref: str | None,      # S-axis: registry ref (env0@prod → env0@outage)
    config_override: dict | None,     # C-axis: allowlisted patch, hashed like #790
    skill_mode: SkillMode | None,     # no-skill | with-skill (re-runs install_agent)
    injected_prompt: str | None,      # e.g. oracle PLAN.md; recorded, never silent
)
```

Injection points dictate cost and validity:

- `skill_mode` / tool-set deltas ⇒ branch from `env-ready` (install re-runs).
- `config_override` ⇒ applied at child setup, same allowlist and hash trail as #790
  (never the scorer).
- `environment_ref` with `[[services]]` changes ⇒ service stop/start bracketing around
  state restore (§3.1).
- `injected_prompt` ⇒ delivered as an explicit user-visible message in the child's
  session and recorded in provenance. Precedent: 0.6.5 removed silent prompt-level
  skill injection (#908); we do not reintroduce it — injection is always a recorded,
  first-class delta.

### 3.4 Lineage: branched runs must be auditable and trainable

Today `RolloutTree` lives and dies in memory. This RFC makes branching leave the same
quality of evidence as a linear run:

- **`tree.json`** in the run folder: nodes, edges, stage tags, snapshot refs,
  per-child delta hashes, per-child rewards, aggregate `V(parent)`.
- **Per-child artifact directories** (`children/<child-id>/`) each with standard
  `result.json` / `config.json` / trajectory files — a child is a first-class rollout
  (implementation seam: `use_prebuilt_env` + the existing child-runner).
- **`source_provenance`** on every child, extending the existing seam (the same one
  `benchflow-continue` uses):

```json
{
  "kind": "benchflow-branch",
  "parent_rollout": "<path-or-id>",
  "parent_task_digest": "sha256:…",
  "branch_stage": "env-ready | post-research | pre-verify | post-verify | cursor:<n>",
  "snapshot_ref": {"sandbox": "<SandboxImage.ref>", "environment": "<StateSnapshot.id>"},
  "cut_point": {"n_replayed_exchanges": 41},
  "delta": {"skill_mode": "with-skill", "config_override_sha256": "…", "environment_ref": null, "injected_prompt_sha256": null}
}
```

- The `branched` phase joins the terminal-phase set so `Rollout.result` has clean
  semantics for branch-first workflows.

### 3.5 Replay cut-point API (the cheap bridge for mid-stage branching)

`ReplayRouter` already serves recorded LLM responses by index; continue-runs prove the
proxy seam end-to-end. Two additions:

1. **`max_exchanges: int`** — replay the first K exchanges, then switch the proxy to
   live passthrough. ("Replay research verbatim, go live at execution.")
2. **Stage-tagged indices** — when a run records stage boundaries (§3.2), the exchange
   index that closed each stage is stored, so cut-points can be named by stage instead
   of by number.

Divergence accounting: at the cut-point, record a content digest of the last replayed
request and the workspace (`tree.json.cut_point_digest`), so a silently-diverged replay
is detectable in artifacts. Fidelity caveats inherit from continue-runs and are
recorded, not hidden.

Generalizing replay beyond openhands (all agents already *record* through the same
LiteLLM gateway) is desirable but independent; it is a named follow-on, not v1.

### 3.6 Snapshot lifecycle

`docker commit` images (`bf-snap-*`) and in-sandbox state dirs currently die with the
rollout (or linger unmanaged). v1 adds: snapshot refs recorded in `tree.json`; optional
`--keep-snapshots` to export container images (`docker save`) into the run folder for
cross-run branching; a GC note in docs. A remote snapshot registry is out of scope.

## 4. Capability matrix (v1)

| Backend | container layer | env-state layer | notes |
|---|---|---|---|
| Docker | ✅ `docker commit` | ✅ sqlite | reference implementation |
| Daytona (direct) | ✅ provider snapshots | ✅ sqlite | immutable snapshots; restore = recreate |
| Daytona (DinD) | ❌ fail-closed | ✅ | |
| Apple Container | ❌ fail-closed | ✅ | |
| AgentCore / Modal | ❌ fail-closed | ✅ | |

Known scope limits carried over from the substrate (documented, unchanged): container
snapshots exclude host-mounted volumes and sibling compose services; env-state covers
declared sqlite only.

## 5. Validation plan

Three tiers, cheapest first; all deterministic and credential-free:

- **T1 — mechanical correctness (unit):** snapshot→restore reproduces workspace and
  env-DB digests; unsupported backends raise typed errors; per-child artifacts and
  `tree.json` validate against the schema; regression tests name this PR per house
  convention.
- **T2 — oracle invariants (integration, docker):** on a small task, (a) **zero-delta
  branch ⇒ child verifier reward equals parent's** at every stage boundary — an
  executable end-to-end proof that restore is lossless; (b) a known-breaking delta
  (removing a required tool) ⇒ the child fails at the expected stage with the expected
  diagnostic.
- **T3 — attribution demo (evidence for the FrontierPhysics paper):** re-run the
  documented no-skill failure of the `surface-ion-trap-shuttling` reference task and
  produce the stage-attribution table (branch at `env-ready` with skills ⇒ pass;
  branch at `post-research` without ⇒ still fails execution).

### Stage-level ablation (the T3 surface)

T3 is not a script anyone re-derives per experiment: it is a command,
`bench eval ablate` (flags in
[docs/reference/cli.md](reference/cli.md#bench-eval-ablate)). One invocation
runs the task once with `--at-stage` captured, forks that recorded world into
one child per `--arms` entry (`with-skill`, `no-skill`, `inject:<file>` — the
§3.3 deltas the engine executes), and writes `ablation.json` beside the
per-child lineage artifacts of §3.4:

```bash
bench eval ablate --tasks-dir tasks/surface-ion-trap-shuttling \
  --at-stage env-ready --arms with-skill,no-skill
```

Two properties keep the output publishable. **The arms are comparable:** they
restore the same snapshot rather than re-running the task, so the world they
differ in is exactly the recorded delta. **The verdict is an observation:**
each row states the two rewards it compares, the boundary they were forked
from, and that it rests on one run per arm — the cross-stage claim ("the
intervention matters at or before stage X") is only earned by a *second*
ablation at a second boundary, which is the T3 table, not one invocation of the
command.

Out of the command's reach in v1, by construction: `post-research` (only an
explicit `Rollout.mark_stage()` records a mid-`execute()` cut point, §3.2),
`environment_ref` / `config_override` arms (no execution path yet, §3.3), and
repeated arms for variance — one run per arm is one observation, not an
estimate.

## 6. Compatibility

- Targets the #470 `Sandbox` contract as-is — stable across the 0.7 line (#827).
- Any new task frontmatter uses the post-#966 `sandbox:` spelling.
- No prompt-content changes to existing modes (respects #908).
- Branch trees are designed to render in the trace-viewer work (benchflow#987).
- Makes `branch_execution: forked-snapshot` (task-standard) real instead of fail-closed.

## 7. Out of scope (v1), named follow-ons

1. **Agent-session snapshot** — documented as the unsolved hard part; v1 children get a
   fresh session with replayed-or-injected context. Follow-on: ACP `session/load`.
2. Replay for ACP-native agents (record side already agent-agnostic).
3. Remote snapshot registry / cross-host branching.
4. Verifier-stage re-judgment under alternative judges (needs verifier-isolation
   materializer, tracked in task-standard).

## 8. Workstreams

| WS | Content | Size |
|---|---|---|
| WS-1 | Composed checkpoint layer (§3.1) + capability matrix + T1 tests | S |
| WS-2 | Deltas (§3.3) + lineage artifacts (§3.4) + T1 tests | M |
| WS-3 | Replay cut-point (§3.5) + T2 oracle invariants + demo (T3) | S/M |

Matching the sync's "2–3 people" sizing; WS-2/WS-3 are parallelizable after WS-1.
