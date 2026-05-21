# BenchFlow — Architecture & Design

*The settled architecture: every understanding and adaptation, release-agnostic. The doc has **two altitudes** — the **core** (build now; small enough to hold in your head) and the **deferred / platform layer** (real, designed, consistent with the core, not yet built). Release scoping and milestones are tracked separately.*

---

## What BenchFlow is

BenchFlow is the **environment-and-rollout engine for agentic RL** — it turns a stateful environment into evaluated, training-ready trajectory data, for any model and any trainer. **It stops where the gradient starts.**

**One engine, three modes.** There is one thing — a *scored rollout*. **Eval** = score it and stop. **Train** = score it and hand the trajectory to a trainer. **Monitor** = score it in production. (Han Lee: "eval = monitoring = reward = one activity.")

**The bet.** A complete RL environment is **E = {T, H, V, S, C}** — Tasks, Harness, Verifier, **State**, Config. Harbor is "terminal-bench thinking" — *E minus S*, with V collapsed to a terminal pass/fail. BenchFlow is the complete E, with **S (state management) as the moat**: stateful, multi-service environments are the frontier of agentic RL — top models score 12–49% on them — and are exactly what Harbor's model cannot host.

**The boundary.** BenchFlow owns **environment + rollout + reward**. Trainers own **weights + gradients + optimizer**. The **trajectory is the seam.** This makes BenchFlow a *partner* to every RL trainer and a *competitor* only to Harbor.

**The business.** Three offerings — RL environments, the SDK, managed evals — are one platform packaged three ways. Lead with **RL environments = selling data**. PrimeIntellect is a partner; Harbor is the competitor.

## Design principles

1. **The kernel depends only on contracts** — the call graph is the source of truth; anything exported with no live caller is wired in or deleted.
2. **Four planes, each swappable, each managed + BYO.**
3. **Zero-modification adoption** — a benchmark brings a self-describing package + a manifest; it never subclasses BenchFlow or touches private APIs.
4. **Eval = monitoring = reward** — one activity, scored on the same trajectory.
5. **The environment is a stateful state machine** the framework provisions and tears down.
6. **BenchFlow is the ACP Client** — the "user" is a pluggable policy, not a special actor.
7. **The harness is not intelligent** — its only job is to extract the most from the model; self-improvement targets the model and skills, never the harness.
8. **Readiness and teardown are framework guarantees** — never the benchmark's burden.
9. **Verifiable-first rewards** — a graded signal on the trajectory.
10. **Ship beats design** — a better design that doesn't run loses to an adequate one that does.
11. **Two altitudes** — the core stays small enough to hold in your head; everything else is the explicitly-labeled platform layer.

## The conceptual model

```
        bench CLI · bf.run() · the environment manifest
                          │
        ┌─────────────────▼──────────────────┐
        │  KERNEL                              │
        │  Rollout lifecycle · reward · trajectory │
        │  depends ONLY on contracts/          │
        └──┬─────────┬───────────┬──────────┬──┘
           ▼         ▼           ▼          ▼
       Sandbox     Agent     Environment   Reward
       (where)    (who)      (the world)  (how scored)
```

The kernel is **three subsystems** — Rollout lifecycle, reward, trajectory — importing only `contracts/` (four `Protocol`s). Concrete providers (Docker, ACP, mockflow, RewardFuncs) join via a registry. The four planes map onto Han's **E**:

| Han's component | BenchFlow |
|---|---|
| T — Tasks | Task / `task.toml` (kernel) |
| H — Harness | Agent plane + kernel |
| V — Verifier | Reward plane |
| **S — State** | **Environment plane** |
| C — Config | `RolloutConfig` (kernel) |

## Core vs deferred — the two altitudes

The whole architecture, sorted by altitude. **Build the left column; the right column is the [deferred / platform layer](#the-deferred--platform-layer)** — designed, consistent, not yet built.

| Area | **Core (build now)** | **Deferred / platform layer** |
|---|---|---|
| Kernel | Rollout lifecycle, reward, trajectory | — |
| Planes | Sandbox, Agent, Environment, Reward (4 contracts) | — |
| Execution nouns | Job, Rollout, Step, Trajectory | Branch / rollout trees |
| Environment ops | `provision` / `readiness` / `query` / `teardown` | `snapshot` / `restore` (environment-state) |
| Evaluation | the **Output** space + the reward tag | Action / Reasoning / Memory / Latent spaces |
| Reward | `Rubric` / `VerifyResult` / `RewardEvent`, turn-granular | hindsight relabel passes |
| Interaction | ACP (`session/prompt`, `request_permission`) | ACPX telemetry profile |
| Jobs | `parallel-independent` | `sequential-shared` + continual learning |
| Topology | in-sandbox environment | shared-fleet / sidecar + `AccountBroker` |
| Trainer seam | trajectory → Verifiers/ORS JSONL export | outbound whole-environment adapter; policy-version stamping; partial/resumable rollouts |

The core is **3 kernel subsystems + 4 planes + 4 execution nouns**. That is the whole thing you must build to ship the first scored, trainable rollout.

## The execution model

**The Rollout = one RL episode** — the atomic unit.

```
Job        — a batch of episodes
└ Rollout  — one RL episode
   └ Step  — one interaction cycle: agent acts (tool call) → environment/responder reacts
```

- **Trajectory** = the root-to-leaf record of a Rollout. For a linear (unbranched) Rollout it is simply the Rollout's full Step sequence. The **trajectory** is the unit that crosses the seam to trainers and is the unit of the data product.
- **No "Scene."** The environment is a **state machine**; the framework's core state operations are `provision` and `teardown`. Progressive disclosure / multi-stage tasks = the environment advancing its own internal state.
- **No "Round" as a separate noun.** A Step *is* the interaction cycle — agent action plus the environment's reaction. (Earlier drafts split `Round` and `Step`; they encoded the same beat.)
- **Kernel invariant:** **token-in/token-out** — exact token-ids + logprobs, never re-tokenize. Guaranteed for trainer-served policy endpoints; best-effort for ACP agents that don't expose logprobs.

Branching (a Rollout becoming a *tree*), policy-version stamping, and partial/resumable rollouts are the [platform layer](#the-deferred--platform-layer) — they are not needed for the first scored, exportable rollout.

## The four planes

**Sandbox — where it runs.** Compute substrate. Built-in: `Local` (raw Linux) + `Docker`. Optional: `Daytona`, `Modal`, `Firecracker`, K8s. BYO via the `Sandbox` protocol. Hardening (`lockdown`) is a capability flag. Framework-guaranteed readiness gate + teardown. An environment is declared once and runs on any provider. (Container-level `snapshot`/`restore` is a platform-layer capability.)

**Agent — who acts.** The agent under test (eval) or the policy under training. Protocol: **ACP**. BYO via `--agent-import-path`. The registry stores agent *declarations* as data, not install code in core. A trainer-served policy endpoint (OpenAI-compatible, hot-swappable) is an agent provider type. (A telemetry profile, *ACPX*, is a platform-layer extension.)

**Environment — the world (Han's S).** The stateful world the agent acts in. A real module — it owns the world's lifecycle: **`provision / readiness / query / teardown`** (core), plus `reset` and `snapshot / restore` (platform layer). See "The Environment plane & the manifest."

**Reward — how it's scored.** `RewardFunc` / `Rubric` / verifier. Verifiable-first; turn-granular process rewards. See "Evaluation."

## The four contracts

The kernel imports only these. Sketch signatures:

```python
class Sandbox(Protocol):            # where it runs — container level
    async def exec(cmd, *, user, timeout) -> ExecResult: ...
    async def upload(local, remote) -> None: ...
    async def download(remote, local) -> None: ...
    async def expose_port(port) -> Endpoint: ...
    async def teardown() -> None: ...
    async def snapshot() -> SandboxImage: ...      # platform layer
    async def restore(image: SandboxImage) -> None: ...  # platform layer

class Agent(Protocol):              # who acts
    async def connect(sandbox, role) -> Session: ...
    def capabilities() -> AgentCapabilities: ...

class Environment(Protocol):        # the world — Han's S
    async def provision(ctx) -> EnvHandle: ...
    async def readiness() -> ReadinessProbe: ...
    async def query() -> EnvState: ...             # for the verifier
    async def teardown() -> None: ...
    async def reset() -> None: ...                 # platform layer
    async def snapshot() -> StateSnapshot: ...     # platform layer
    async def restore(snap: StateSnapshot) -> None: ...  # platform layer

class Reward(Protocol):             # how it's scored
    async def score(rollout: Rollout) -> VerifyResult: ...
```

The contract surface is **stable** — `snapshot`/`restore` stay in the Protocol so branching wires in later without a contract change. The first implementation (`ManifestEnvironment`) raises `NotImplementedError` for them: a stable contract, an unbuilt capability.

## Evaluation

eval = monitoring = reward. The **core reward signal is the Output space** — *did it finish the job?* (the terminal reward).

Every reward record is tagged **`(space, granularity, scope, value)`** — so the other evaluation lenses (Action, Reasoning, Memory, Latent) wire in later **without a schema change**. Granularity is **terminal / turn / step**; turn-level is the default (an episode-level scalar is inadequate beyond ~50 turns).

`reward.txt` / `reward.json` are sandbox-boundary wire formats parsed by `TestRewardFunc`; the in-kernel model is `VerifyResult` + `RewardEvent`. The other four spaces and post-rollout hindsight passes are the [platform layer](#the-deferred--platform-layer).

## The Environment plane & the manifest

The **Environment plane** is a real module defined by its Protocol (above). What a benchmark *author* writes is the **manifest** — the configuration the default adapter (`ManifestEnvironment`) reads, and the entire integration surface. *Write a manifest; your stateful environment runs anywhere and trains anything, with zero framework modification.*

A **BYO Environment Package** = an image (or Dockerfile + context) + the manifest + tasks + skills + verifier:

```toml
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
ports          = [8020, 8023, 8100, 8200]
owns_lifecycle = true              # the image's entrypoint starts the services
keep_alive     = true
isolation      = "per_task"        # OR "persistent" (cross-episode state)

[environment.task_selection]
mechanism   = "env_var"
key         = "CHI_BENCH_TASK_ID"
inject_into = "entrypoint"         # reaches PID 1, not just exec()

[environment.readiness]            # framework gates on this before the agent runs
http           = ["http://localhost:8023/health"]
mcp_initialize = [8020, 8100, 8200]
timeout_sec    = 120

[environment.forward_env]
keys = ["ANTHROPIC_API_KEY", "..."]

[environment.sandbox]
supports = ["docker", "modal"]     # declared once; Sandbox plane runs either

[verifier]
kind              = "agent"
hidden_from_agent = ["expectations.json", "tasks/*/fixtures"]
```

**State is a real database**; tools are read-write ops over the schema — which makes state snapshot-able, diffable, and verifiable. The **core topology is in-sandbox** (the environment runs in the rollout's own sandbox). The shared-fleet / sidecar topology — a `TaskDatabase` + `AccountBroker` for multi-tenant per-task accounts — is the [platform layer](#the-deferred--platform-layer) scale path, behind the same Protocol.

**The stateful multi-service pattern.** ClawsBench and chi-bench are structurally the same machine; the plane hosts both.
- **ClawsBench** — the internal dogfood (runs on BenchFlow, hard-coded today as a `SERVICES` registry); the manifest's design partner. Onboarding it = replacing that registry with a manifest.
- **chi-bench** — the external proof; a ~25k-LOC heavy simulator with a thin MCP transport, onboarded via a ~25-line manifest with its environment **untouched** — its ~920 LOC of Harbor coupling collapse into the manifest.

Conformance bar: *chi-bench's image runs with zero image edits — only the manifest is new.*

## The interaction model — ACP

Human interaction is modeled through ACP's role split: **BenchFlow is the ACP Client; the "user" is a pluggable User Model inside the Client role.** Two channels carry everything:
- `session/prompt` (Client → Agent) — the task instruction and every **nudge** (user-initiated).
- `request_permission` (Agent → Client, with options) — `ask_user` (agent-initiated).

The **User Model** modes: scripted / simulated (LLM persona) / real-human / auto. `ask_user` with enumerated options is the **branchable interaction primitive** — finite options ⇒ a finite, scoreable interaction tree (the tree itself is the platform layer).

## The deferred / platform layer

Everything below is **designed, consistent with the core, and not built in the first pass.** It lives in one section so the core stays small and the diagram stays honest. Each item names the benchmark or condition that forces it into existence.

- **Branching & rollout trees** — a `snapshot` + N×`restore` at a choice point makes a Rollout a tree; the trajectory becomes one root-to-leaf path. Serves human-feedback choice points *and* GRPO group rollouts. *Requires* Environment-state `snapshot`/`restore`. **Forced by:** NudgeBench.
- **The other four evaluation spaces** — Action (right actions / no reward-hacking), Reasoning (sound, connected chain-of-thought), Memory (skill/memory updates), Latent (SAEs, with interpretability access). The `(space, …)` reward tag already reserves room; only the scorers are deferred. **Forced by:** SkillsBench (Memory), interpretability work (Latent).
- **Continual learning** — a **`sequential-shared` Job mode** over a persistent **learner store** (memory + skills): the store is versioned (a generation counter stamped per rollout) and rollback-capable (revert a generation when a learning-curve metric regresses). Learning-curve metrics track improvement **and drift** + adoption rate, scored against a human-skill ceiling. Skills are useful long-term **only if continuously evolved**. **Forced by:** clbench.
- **Hindsight reward passes** — post-rollout relabeling over a whole trajectory/tree. Needed once episodes exceed ~50 turns.
- **Policy-version stamping & partial/resumable rollouts** — tagging every rollout with the checkpoint that produced it; resuming 100+-turn episodes across iterations. Needed for trainer-served policies, not for eval.
- **ACPX** — an ACP telemetry profile (logprobs, token-ids) for ACP agents that don't expose them natively. Trainer-served endpoints provide them already.
- **Shared-fleet / sidecar topology** — a `TaskDatabase` + `AccountBroker` for multi-tenant per-task accounts against a long-lived service fleet. The in-sandbox topology is the core; this is the scale path.
- **Outbound whole-environment adapter** — re-package a BenchFlow environment as a standalone Verifiers/ORS package. (Note: the *trajectory* → Verifiers/ORS JSONL export is **core** — it is the trainer seam. Re-packaging the entire environment outward is the deferred, larger thing.)
- **Container-level snapshot/restore** — `Sandbox.snapshot` of the whole container, a coarser layer beneath Environment-state snapshot. Two snapshot layers (container + environment-state), plus agent-session state, together make a Rollout fully checkpointable; the learner store is the one layer that deliberately does *not* roll back.

This section **resolves two of the doc's three open questions as a side effect**: the "five spaces" terminology question is moot (only Output is core), and the `Job` mode vs. `isolation` overlap is moot (`sequential-shared` is deferred, so `isolation` is the only live vocabulary).

## The edges — adapters & trainers

**Inbound env adapters** — Harbor / Inspect / ORS / PrimeIntellect → run foreign benchmarks; **Terminal-Bench backward-compatible** via the Harbor env-adapter. (Outbound whole-environment export is the platform layer.)

**Trainer seam.** BenchFlow is a rollout *service*; trainers are external (the boundary). The seam is the **trajectory exported as a Verifiers/ORS JSONL record** — one `{prompt, completion, reward, info}` object per scored rollout. The scope is **PI-compatibility**: being a Verifiers/ORS-compatible producer yields a trainer (prime-rl) with zero trainer code. A general trainer layer (Tinker, VeRL, NeMo-RL) is a later possibility, not a current commitment.

## The benchmarks — the forcing functions

The roadmap is benchmarks, not abstract tracks. Each benchmark is a forcing function; "done" for a capability = its benchmark runs clean.

| Benchmark | Forces into existence | Altitude |
|---|---|---|
| **ClawsBench** | the Environment plane (internal dogfood) | **core** |
| **chi-bench** | zero-modification external adoption | **core** |
| **SkillsBench** | skills + skills-eval (Memory space) | platform |
| **NudgeBench** (followupbench) | the interaction model + branching | platform |
| **clbench** | continual learning (the `sequential-shared` Job) | platform |
| **Terminal-Bench / SWE-bench** | env-adapter backward compatibility | core (inbound adapter) |

## Adaptations — what changed, and why

A record of the deliberate design decisions, so they are not re-litigated.

| Adaptation | From → To | Why |
|---|---|---|
| **Two altitudes — core vs deferred** | a flat spec where every concept read as build-now → a small **core** + a labeled **deferred / platform layer** | The spec was over-built *as a document* — ~1/3 of its nouns failed the deletion test for the core. This is re-leveling, not deleting; nothing is lost. |
| **Removed "Round" as a noun** | `Round` and `Step` as two execution nouns → **Step** is the interaction cycle | They encoded the same beat (agent acts → environment reacts). |
| **Removed "Scene"** | a runtime object (the codebase had two `Scene` classes) → the environment is a **state machine**; framework ops are `provision`/`teardown` | RL has no "scene" — a phase is just state. Removes a noun and a duplicate-class debt. |
| **Removed "Lineage"** | a new orchestration object for continual learning → a **`sequential-shared` Job mode** (platform layer) | The field models continual learning as ordered experience over a store; a new object was redundant. |
| **Kept "Environment plane"; disambiguated the word** | the word "environment" was overloaded → it now means *only the world* (the plane = the product = the package); Han's full `{T,H,V,S,C}` is always "the E-tuple", never bare "environment" | the plane *is* the environment (the mock-service world). "State" is the per-step value `sₜ` — wrong as a plane name. |
| **One reward path** | 3–4 disagreeing reward schemas, the documented `Rubric`/`RewardFunc` package dormant → `Rubric` becomes the one scoring path; `reward.txt`/`reward.json` demoted to wire formats | The documented design was not running; the live path was the least-capable one. A cutover with a regression gate. |
| **Trainer integration → PI-compat** | a general trainer-integration layer → **export the trajectory as a Verifiers/ORS record** | The field standardized rollout-as-a-service; PI already wired trainer ↔ environment. Building our own is premature. |
| **Branching → platform layer** | branching treated as a core execution feature → deferred; the contract reserves `snapshot`/`restore` so it wires in without a contract change | Branching is real (NudgeBench, GRPO) but not needed for the first scored rollout. |
| **Dead-architecture cleanup** | exported-but-dead modules (`_run.py`, `experimental/mcp/` genuinely dead; `sdk.py`/`runtime.py`/`scenes.py` live legacy) → delete the dead, **migrate** the live legacy onto the single `Rollout` path | The call graph, not `__all__`, is the source of truth. |
| **eval = monitoring = reward** | three framings (eval framework / RL framework / monitoring tool) → **one engine, three modes** | Han's insight — the single biggest complexity reducer. |
| **Manifest as the only seam** | benchmarks subclass framework internals (chi-bench: ~920 LOC of Harbor subclasses) → a declarative manifest; zero framework modification | A benchmark should never modify the framework. |

## Open questions

1. **Repo shape** — one monorepo, with mockflow folded in as `benchflow-env`; ClawsBench stays a separate repo (non-commercial license). Confirm. *(Operational, not architectural.)*
2. ~~"Five spaces" terminology~~ — **resolved** by the core/deferred split: only the Output space is core.
3. ~~`Job` mode vs. manifest `isolation`~~ — **resolved**: `sequential-shared` is deferred, so `isolation` is the only live vocabulary.

## Appendix — research validation

The architecture was checked against ~14 recent agentic-RL papers; the field has converged on this shape.
- Agentic RL = a temporally-extended POMDP; the capability taxonomy maps onto the five spaces. *(The Landscape of Agentic RL, arXiv:2509.02547)*
- Rollout-as-a-service decoupled from training; staged async; token-in/token-out; partial rollouts. *(ProRL Agent, arXiv:2603.18815)*
- Episode-scalar reward fails beyond ~50 turns; turn-level + hindsight credit required. *(Credit Assignment, arXiv:2604.09459)*
- Stateful, DB-backed environments; automated generation; verifier-reliability gating. *(EnterpriseOps-Gym, AutoEnv, Agent World Model)*
- Continual learning = base policy + a persistent, evolving skill library; version + rollback the store. *(MetaClaw, MemSkill, SkillLearnBench)* — validates a **platform-layer** feature.
- Training-time tree search (branching) improves RL. *(TSR, arXiv:2602.11767)* — validates a **platform-layer** feature.

**Verdict:** the architecture is consensus-correct, and the core is now small — 3 kernel subsystems, 4 planes, 4 execution nouns. The remaining risk is execution, not design. The first thread to build is the vertical slice: *ClawsBench → declared as a manifest on the Environment plane → produces scored trajectories → exported to prime-rl → a model trains on it.*
