# BenchFlow — Architecture

*The whole architecture, as one coherent picture — every concept we need, no build-order tiering. Release scoping and milestones are tracked separately (Linear). This document is derived from the sources that count — Han Lee's writing and conversation, our project notes, and the agentic-RL literature — not from the current doc or the current code, both of which are snapshots that follow this, not the other way round.*

---

## What BenchFlow is

BenchFlow is the **environment-and-rollout engine for agentic RL** — it turns a stateful environment into evaluated, training-ready trajectory data, for any model and any trainer. **It stops where the gradient starts.**

**One engine, three modes.** There is one thing — a *scored rollout*. **Eval** = score it and stop. **Train** = score it and hand the trajectory to a trainer. **Monitor** = score it in production. (Han Lee: *"evaluation, reward and monitoring … it's really all the same thing under different circumstances."*)

**The bet.** A complete RL environment is **E = {T, H, V, S, C}** — Tasks, Harness, Verifier, **State**, Config (Han Lee, *RL Environments for LLM Agents*). Harbor is "terminal-bench thinking" — *E minus S*, with V collapsed to a deterministic pass/fail script. BenchFlow is the complete E, with **S — state management — as the moat**: stateful, multi-service environments that can **roll out, roll back, and branch** are the frontier of agentic RL, and are exactly what Harbor's model cannot host.

**The boundary.** BenchFlow owns **environment + rollout + reward**. Trainers own **weights + gradients + optimizer**. The **trajectory is the seam.** This makes BenchFlow a *partner* to every RL trainer and a *competitor* only to Harbor.

**The business.** Three offerings — RL environments, the SDK, managed evals — are one platform packaged three ways. Lead with **RL environments = selling data**. PrimeIntellect is a partner; Harbor is the competitor.

## Grounding

This architecture rests on three sources, kept honest against each other.

**Han's `{T,H,V,S,C}`** (blog, *RL Environments for LLM Agents*) — the environment decomposition. Verbatim: T = "problems the agent tries to solve"; H = the agent harness, "scaffolding that … controls *how* the model interacts, but does not improve what it knows"; V = the verifier, "V: (task prompt, completion, info) → [0,1]"; S = state, "stateless (fresh starts) … or stateful (persistent across actions/episodes)"; C = configuration, "turn limits, context budgets, sampling temperature, curriculum scheduling."

**Han's conversation** (advisory call) — the dynamics the blog's static set does not capture:
- *"Environment 总是要 roll out, roll back"* — **roll-out and roll-back are definitional** for an environment. Roll-back = *"snapshot environment and go back to its different stage."*
- Branching: an `ask_user`-type interaction *"literally is a checkpoint … to different type of rollout"*, and from it — *"From reward function to a value function … of the current state."* Branching is *"very important for large horizon tasks."*
- *"eval = monitoring = reward"* — one activity, observed across **five spaces** (output, action, reasoning, memory, latent).
- *"The harness is not meant to be intelligent"* — self-improvement targets the **model and skills**, never the harness; *"skill 是属于 memory"* (skills are memory).
- ACP is the mechanism for modelling human interaction inside a rollout.

**The agentic-RL literature** — agentic RL is a *"temporally extended, partially observable MDP"* (*The Landscape of Agentic RL*, 2509.02547) — definitionally a branching structure. Tree-structured rollouts give *"step-wise process supervised signals even using only the outcome reward"* and *"more rollouts within a fixed budget of tokens or tool calls"* (*Tree Search for LLM Agent RL* / Tree-GRPO, ICLR 2026). A scan of 13 RL libraries (verifiers, prime-rl, SkyRL, verl, NeMo-RL, Tinker, OpenEnv, Harbor, Terminal-Bench, Inspect, ORS, Gymnasium, agent-lightning) found **all model rollouts linearly** — so a tree-native rollout with environment snapshot/restore is genuine, defensible novelty, and the load-bearing hard part is snapshot/restore of *heavy* environment state (see "The hard part").

## Design principles

1. **The kernel depends only on contracts.** The call graph is the source of truth; anything exported with no live caller is wired in or deleted.
2. **Four planes, each swappable, each managed + BYO** — Sandbox, Agent, Environment, Reward.
3. **The Rollout is a tree.** An RL episode is a tree of states; a linear rollout is the degenerate degree-1 case. Branch, snapshot, and restore are first-class — they are how a reward function becomes a value function.
4. **Roll-back is definitional.** An environment that cannot snapshot and restore its state is incomplete (Han). `snapshot`/`restore` are real methods, not stubs.
5. **Zero-modification adoption.** A benchmark brings a self-describing package + a manifest; it never subclasses BenchFlow or touches private APIs.
6. **Eval = monitoring = reward** — one activity, scored on the same trajectory, across five spaces.
7. **The environment is a stateful state machine** the framework provisions, snapshots, restores, and tears down.
8. **BenchFlow is the ACP Client** — the "user" is a pluggable policy, not a special actor.
9. **The harness is not intelligent** — its only job is to extract the most from the model; self-improvement targets the model and skills.
10. **Readiness and teardown are framework guarantees** — never the benchmark's burden.
11. **Ship beats design** — a better design that doesn't run loses to an adequate one that does.

## The conceptual model — the planes

```
        bench CLI · bf.run() · the environment manifest
                          │
        ┌─────────────────▼───────────────────┐
        │  KERNEL                               │
        │  Rollout lifecycle · reward · trajectory │
        │  depends ONLY on contracts/           │
        └──┬──────────┬───────────┬──────────┬──┘
           ▼          ▼           ▼          ▼
       Sandbox      Agent     Environment   Reward
       (where)     (who)      (the world)  (how scored)
```

The kernel is **three subsystems** — Rollout lifecycle, reward, trajectory — importing only `contracts/` (four `Protocol`s). Concrete providers (Docker, ACP, `ManifestEnvironment`, `RewardFunc`s) join via a registry. The four planes map onto Han's **E**:

| Han's component | BenchFlow | Plane |
|---|---|---|
| **T** — Tasks | Task / `task.toml` | kernel concept |
| **H** — Harness | the agent + the kernel scaffolding around it | **Agent plane** |
| **V** — Verifier | `RewardFunc` / `Rubric` / verifier | **Reward plane** |
| **S** — State | the stateful world | **Environment plane** |
| **C** — Config | `RolloutConfig` | kernel concept |

T and C are kernel concepts (the inputs); H, V, S are the planes that *do* the work; Sandbox is the substrate all three run on. Four planes, one kernel, two kernel-level inputs — that is the whole conceptual surface.

## The execution model — tree-native

**A Rollout is one RL episode, and it is a tree.** Han's trajectory is a chain of *state → action → next-state*; a `Branch` makes that chain a tree; classical RL is defined over exactly this tree (a POMDP), and the value function `V(s)` is *defined* as the expected return over the continuations from a state. Modelling the Rollout as a tree is therefore the RL-native choice.

The execution model has **three primitives, one derived view, and one authoring form** — all defined on the one tree, not a Russian-doll hierarchy:

```
Job         — a set of Rollouts run together (an eval sweep · a GRPO group · a CL sequence)
Rollout     — one RL episode = a TREE of states (sₜ)                          PRIMITIVE
  • Step    — one edge of the tree: (reason → act) → (tool-in → tool-out)      PRIMITIVE
  • Branch  — the snapshot-and-fork operation; a node with >1 child            PRIMITIVE
Trajectory  — one root-to-leaf path. Computed from the tree, never declared    DERIVED VIEW
Scene       — a declared span carrying a role/skill configuration             AUTHORING SUGAR
```

- **The primitives are irreducible.** The tree (`Rollout`), its edges (`Step` — Han's atomic unit; one `Step` is one "turn"), and the `Branch` operation (snapshot + fork). `Branch` is the credit-assignment engine: it evaluates *one state across N continuations* — averaging the children's returns estimates `V(s)`. That is Han's *"from reward function to a value function of the current state"* and Tree-GRPO's peer-reviewed result that a tree yields process supervision from a single outcome reward. A GRPO group run as a shared-prefix tree beats N independent rollouts (more rollouts per token/tool budget). Branches occur at `ask_user`-style interaction checkpoints (one child per option), at GRPO group points, and at value-estimation points.
- **`Trajectory` is a derived view** — a pure function of the tree (a path), never declared. It is what serialises out: a linear `prompt / completion / reward / metrics / info` record, the Verifiers/ORS training unit.
- **`Scene` is authoring sugar** — the *declaration* form for multi-phase / multi-agent rollouts (`RolloutConfig.scenes`). It desugars completely to per-`Step` role/skill attribution plus config that changes along the tree, and adds no expressive power. It has no runtime object and no lifecycle of its own — `RolloutConfig.scenes` is a desugaring pass that lowers to per-`Step` config. Kept only as a convenient authoring affordance. (The original RFC's instinct — "a phase is just state" — was correct.)

**Tree-native is free for the mental model, not for the engine.** A linear rollout genuinely *is* a degree-1 tree, so the *data model* costs nothing extra for the common case. But the *engine* — checkpoint/fork, three-layer snapshot composition, node-addressed scoring, child scheduling — is paid for on day one even by users who only run linear rollouts. That is an accepted cost, not a hidden one: the tree is the correct foundation, and the linear path inherits its machinery.

## Lifecycles

Every lifecycle the framework owns, as ordered phases.

**Job lifecycle.** `plan` (resolve tasks × agents × repeats) → `schedule` (parallel-independent, or sequential-shared for continual learning) → `run Rollouts` → `aggregate` → `report`.

**Rollout lifecycle.** `setup` (resolve config, build the environment object) → `start` (sandbox up) → `provision environment` (Environment plane starts services) → `readiness gate` (framework-guaranteed; the agent never runs before the world is healthy) → `connect agent` (ACP) → `execute` (the tree grows: Steps and Branches) → `verify` (Reward plane scores) → `teardown`.

**Branch lifecycle.** `quiesce` (pause the agent at a stable point) → `checkpoint` (snapshot environment state, then container, then agent-session state — in that order, see "The hard part") → `fork` (N children) → `run children` → `score / aggregate` (per-child return → `V(parent)`) → optionally `restore` the winning child's state to continue.

**Environment lifecycle** (Han's roll-out / roll-back). `provision` → `readiness` → `query` (expose state to the verifier) → `snapshot` → `restore` → `reset` → `teardown`. `snapshot`/`restore` are definitional — the substrate every `Branch` runs on.

**Sandbox lifecycle.** `start` → `exec` / `upload` / `download` / `expose_port` → `snapshot` / `restore` (container-level, coarser than environment-state) → `stop`.

A Rollout is checkpointable because three snapshot layers compose — container (Sandbox) ⊃ environment-state (Environment) ⊃ agent-session — but composing them correctly is a real consistency problem (see "The hard part"). The one store that deliberately does **not** roll back with a `Branch` is the continual-learning learner store (capability 5).

## The four planes

**Sandbox — where it runs.** Compute substrate. Built-in: `Local` (raw Linux) + `Docker`. Optional: `Daytona`, `Modal`, `Firecracker`, K8s. BYO via the `Sandbox` protocol. Hardening (`lockdown`) is a capability flag. Framework-guaranteed readiness gate + teardown. An environment is declared once and runs on any provider.

**Agent — who acts.** The agent under test (eval) or the policy under training — Han's harness, "not intelligent." Protocol: **ACP** (the official `agent-client-protocol`). BYO via `--agent-import-path`. The registry stores agent *declarations* as data, not install code in the kernel. A trainer-served policy endpoint (OpenAI-compatible, hot-swappable) is one agent provider type. The plane's real surface is the `Session` (below) — not just `connect`.

**Environment — the world (Han's S).** The stateful world the agent acts in. Owns the world's lifecycle: `provision / readiness / query / snapshot / restore / reset / teardown`. See "The Environment plane & the manifest."

**Reward — how it's scored (Han's V).** `RewardFunc` / `Rubric` / verifier. `V: (task, completion, info) → [0,1]`, generalised to a graded, multi-space, multi-granularity signal over the trajectory tree. See "Evaluation."

## The four contracts

The kernel imports only these.

```python
class Sandbox(Protocol):            # where it runs — container level
    async def exec(cmd, *, user, timeout) -> ExecResult: ...
    async def upload(local, remote) -> None: ...
    async def download(remote, local) -> None: ...
    async def expose_port(port) -> Endpoint: ...
    async def snapshot() -> SandboxImage: ...
    async def restore(image: SandboxImage) -> None: ...
    async def teardown() -> None: ...

class Agent(Protocol):              # who acts — Han's harness
    async def connect(sandbox, role) -> Session: ...
    def capabilities() -> AgentCapabilities: ...

class Session(Protocol):            # a LIVE agent session — the Agent plane's real surface
    async def prompt(text: str) -> StopReason: ...   # the task instruction, or a nudge
    async def cancel() -> None: ...
    def on_ask_user(handler: AskUserHandler) -> None: ...  # agent-initiated; the branch hook
    @property
    def steps(self) -> list[Step]: ...               # the session's contribution to the tree

class Environment(Protocol):        # the world — Han's S
    async def provision(ctx) -> EnvHandle: ...
    async def readiness() -> ReadinessProbe: ...
    async def query() -> EnvState: ...               # for the verifier
    async def snapshot() -> StateSnapshot: ...       # roll-back: definitional
    async def restore(snap: StateSnapshot) -> None: ...
    async def reset() -> None: ...
    async def teardown() -> None: ...

class Reward(Protocol):             # how it's scored — Han's V
    async def score(node: RolloutNode) -> VerifyResult: ...
```

`Session` is part of the contract, not an untyped return — the entire ACP interaction (prompt, nudge, the `ask_user` branch hook) is the Agent plane's seam, so it must be specified to BYO an agent.

`Reward.score` takes a `RolloutNode`, and a node **carries its tree context**: `node.path` (root → node), `node.subtree`, `node.state`. One `score` method therefore expresses both outcome reward (read the leaf) and process reward (walk `node.path` across the Action and Reasoning spaces) — there is no per-step-in-isolation scoring. `VerifyResult` = `{reward: float, items: dict[str, float], events: list[RewardEvent], space, granularity}`.

## The Environment plane & the manifest

What a benchmark *author* writes is the **manifest** — the entire integration surface. *Write a manifest; your stateful environment runs anywhere and trains anything, with zero framework modification.* The default adapter `ManifestEnvironment` reads it.

```toml
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"   # OR base_image + [[services]] (framework-started)
owns_lifecycle = true                 # the image's entrypoint starts the services
isolation      = "per_task"           # OR "persistent" (cross-episode state)

[environment.task_selection]
mechanism   = "env_var"               # OR "image" (per-task images, smolclaws-style)
key         = "CHI_BENCH_TASK_ID"
inject_into = "entrypoint"            # reaches PID 1, not just exec()

[environment.readiness]               # the framework gates on this before the agent runs
http        = ["http://localhost:8023/health"]
timeout_sec = 120

[verifier]
kind              = "agent"
hidden_from_agent = ["expectations.json", "tasks/*/fixtures"]
```

**State is a real database**; tools are read-write ops over the schema — which is what makes state snapshot-able, diffable, and verifiable. Two topologies behind one contract: **in-sandbox** (the environment runs in the rollout's own sandbox — the default) and **shared-fleet / sidecar** (a long-lived service fleet + a `TaskDatabase` + `AccountBroker` for multi-tenant per-task accounts — the scale path).

**The environment package — and the simulated user as part of it.** A benchmark ships as a self-describing **package** (folders + files + the manifest), not a Harbor task dir. Beyond `image` / `[[services]]` / `[environment.state]` / `[verifier]`, the package first-classes one thing Harbor's model cannot: a **simulated user** declared *with the environment*. In tool-agent-user benchmarks (the τ-bench family — τ-bench / τ²-bench), the user is a stateful participant with its own policy and goals, not a fixed prompt; BenchFlow packages that user as a **User Model** (scripted / LLM-persona / dual-agent) the kernel drives through the ACP Client role (see "The interaction model"). This is the BenchFlow-native counterpart to OpenReward-style reward packaging and tau-style user simulation — distinct from, and richer than, the Harbor task format. *The on-disk package spec (file layout, the user-policy file, how it desugars to `RolloutConfig.user` + per-`Step` config) is being finalised in the v0.5 env-package-format RFC and is summarised in [`v05-roadmap.md`](./v05-roadmap.md).*

**The Stateful Multi-Service Benchmark (SMSB).** ClawsBench and chi-bench are structurally the same machine; the plane hosts both. ClawsBench is the internal dogfood (the manifest's design partner); chi-bench is the external proof — a ~25k-LOC heavy simulator with a thin MCP transport, onboarded via a ~25-line manifest with its environment **untouched**, its ~920 LOC of Harbor coupling collapsing into the manifest.

## Evaluation — the five spaces

eval = monitoring = reward. The same scoring runs at train time, at eval time, and in production — only the context differs. A reward signal is read from the trajectory across **five spaces** (Han):

| Space | What it checks |
|---|---|
| **Output** | did it finish the job? (the terminal/verifiable reward) |
| **Action** | right actions, no reward-hacking, no out-of-distribution tool use; *did it ask when it should have?* |
| **Reasoning** | is the chain-of-thought sound and connected to the action and answer? (CoT monitoring) |
| **Memory** | did it update its memory / skills correctly? (diff the store) |
| **Latent** *(future)* | with interpretability access — SAEs over post-attention embeddings. No benchmark needs it yet; named so it isn't reinvented later, not built. |

Every reward record is tagged **`(space, granularity, value)`**. Granularity is **terminal** (the whole trajectory) or **step** (one edge) — an episode-level scalar alone is inadequate beyond ~50 steps; the tree's structure supplies finer credit. **Process reward** is read by walking a node's `path` across the Action and Reasoning spaces — *not* by scoring each step in isolation (process supervision "hard to judge" per-step — Han). The wire formats `reward.txt` / `reward.json` cross the sandbox boundary; the in-kernel model is `VerifyResult` + `RewardEvent`.

## The interaction model — ACP

Human interaction is modelled through ACP's role split: **BenchFlow is the ACP Client; the "user" is a pluggable User Model inside the Client role.** Two channels carry everything:
- `session/prompt` (Client → Agent) — the task instruction and every **nudge** (user-initiated follow-up).
- `request_permission` / `ask_user` (Agent → Client, with enumerated options) — agent-initiated, surfaced through `Session.on_ask_user`.

`ask_user` with enumerated options is the **branchable interaction primitive** — finite options ⇒ a finite, scoreable interaction tree (each option is one `Branch` child). The interaction tool is never hard-coded as "step one"; the agent chooses to use it, and the **Action space** scores *whether it asked* — an under-specified task makes "ask the user" the correct behaviour, and failing to ask is a negative reward (Han). User Model modes: scripted / simulated (LLM persona) / real-human / auto. (Branching is not a User Model mode — it is a property of the `Rollout` tree.)

## The edges — adapters & trainers

The manifest is BenchFlow's native format. The edge is **two-track** — translate a foreign format *to* the manifest, **or** run the foreign task *in place* through its own system — and a benchmark may be onboarded either way.

**Track 1 — inbound translation.** Adapters translate a foreign format to the manifest so it runs natively on the four planes: Harbor, Inspect, ORS, PrimeIntellect/Verifiers environments → run foreign benchmarks natively. **Terminal-Bench is backward-compatible** through the Harbor adapter (Harbor is itself terminal-bench-derived) or a direct Terminal-Bench adapter — old terminal tasks keep running.

**Track 2 — run via their own system (interop proof).** BenchFlow must also prove it *composes with* each ecosystem by invoking that ecosystem's **own runner** directly — not re-implementing it: **verifier tasks via `vf-eval`**, **Harbor tasks via `harbor run`**, **OpenReward tasks via the OpenReward runner**. BenchFlow provides the sandbox, rollout, trajectory capture, and trainer export around the foreign runner; the foreign system stays the source of truth for task semantics and scoring. The end-to-end claim is concrete and testable: *the same BenchFlow run loop drives a Verifiers task, a Harbor task, and an OpenReward task, each through its native system.* (Track 1 gives native-plane features like state snapshot/branching; Track 2 gives zero-divergence fidelity and a migration on-ramp. `hosted_env.py` already shells out to `vf-eval` — the seed of Track 2.)

**Outbound — the trainer seam.** A scored trajectory exports as a **Verifiers / ORS JSONL record** (`prompt / completion / reward / metrics / info`). Being a Verifiers/ORS-compatible producer yields a trainer — prime-rl — with zero trainer code. BenchFlow is a rollout *service*; trainers (Tinker, verl, NeMo-RL) stay external. The trajectory is the seam.

## How a Task flows through the architecture

A **Task** (Han's T) is the problem spec — `task.toml` + instruction + the environment package + the verifier. It is a kernel concept, and it is what wires the planes together for one run:

```
Task ─┬─→ selects the Environment package + manifest  ───→ Environment plane provisions S
      ├─→ carries the instruction / prompt            ───→ Agent plane (H) receives it
      ├─→ names the verifier + hidden fixtures         ───→ Reward plane (V) scores
      └─→ carries config (turn limits, budgets)        ───→ RolloutConfig (C)
                                   │
                                   ▼
              one Rollout (a tree) runs in a Sandbox
                                   │
                                   ▼
          Trajectory(s) + reward  ───→  export  ───→  trainer
```

One Task → one Rollout tree → one or more Trajectories. A Job is many Tasks (or one Task × many repeats). `{T,H,V,S,C}` is not an abstraction layered on top — it *is* the wiring diagram of a single run.

## The eight capabilities — how each fits

The architecture is one shape; these are the eight things it must carry. Capabilities 1–6 and 8 are benchmark-forced — "done" = that benchmark runs clean. Capability 7 is the substrate the others ride on, not a benchmark.

| # | Capability | How it fits the architecture |
|---|---|---|
| 1 | **SkillsBench** | An Environment-plane benchmark package (skills + skill-eval tasks). Skills are **memory** (Han); the Reward plane's **Memory space** scores skill use and skill updates. Skills are installed as per-`Step` config (the `Scene` desugaring) and deployed into the sandbox. |
| 2 | **ClawsBench** | The SMSB on the Environment plane — `base_image` + `[[services]]`, framework-started, `image` task-selection. The internal dogfood; the manifest's design partner. |
| 3 | **chi-bench** | The same SMSB archetype — `image` + `owns_lifecycle = true` + `env_var` task-selection. The external proof: onboarded by a ~25-line manifest, environment untouched. |
| 4 | **followupbench (NudgeBench)** | The **ACP interaction model** (`session/prompt` nudges + `ask_user` via `Session.on_ask_user`) + the **tree-native Rollout** — every interaction checkpoint is a `Branch` — + the **Action space** reward scoring *whether the agent followed up / asked*. |
| 5 | **Continual learning** | A **Job run in `sequential-shared` mode**: Rollouts run in order over a persistent **learner store** (memory + skills). The store is versioned (a generation stamped per rollout) and rollback-capable; the **Memory space** tracks improvement, drift, and adoption. Skills stay useful *only if continuously evolved* (Han). The learner store is the one snapshot layer that does not roll back with a `Branch`. |
| 6 | **RL-native** | The whole execution model: the Rollout is a tree, the Trajectory is a path, the Reward contract scores any node, and the trajectory exports as a trainer-ready record. Agentic RL is a temporally-extended POMDP — and the architecture is shaped like one. |
| 7 | **Branching, rollback, Han's framework** | *Not a benchmark — the RL-native substrate itself.* First-class `Branch`; `Environment.snapshot`/`restore` as definitional roll-back; the value-function purpose of the tree; the five spaces; eval = monitoring = reward; the non-intelligent harness. Capabilities 4–6 ride on it. |
| 8 | **Env adapters & interop — Harbor / PrimeIntellect / OpenReward** | The **two-track edge**: *(Track 1)* inbound adapters translate foreign formats to the manifest (Terminal-Bench backward compat rides the Harbor adapter); *(Track 2)* run foreign tasks via their **own systems directly** — `vf-eval` (verifiers), `harbor run`, the OpenReward runner — proving BenchFlow composes with each ecosystem. Outbound, the trajectory exports to Verifiers/ORS. "Done" = one run loop drives a verifier, a Harbor, and an OpenReward task through their native systems. |

All eight land on one architecture — four planes, a tree-native Rollout, an adapter edge. None requires a new top-level concept.

## The hard part — honest risk

The library scan is unambiguous: **no agentic-RL library ships environment snapshot/restore.** Tree-GRPO branches a *token prefix*; Inspect's `fork()` deep-copies *conversation* state, not the sandbox; its checkpoint system is resume-only — its design note says *"reality doesn't have a fork command."* BenchFlow's bet is to branch a **heavy stateful environment** — a mock-Gmail SQLite database, a healthcare simulator, eventually a K8s cluster. The `Branch` checkpoint is genuinely three unsolved problems, not one:

1. **Environment-state snapshot** — DB dump/restore, copy-on-write volumes, fork-able service processes. Designed deliberately, environment-class by environment-class — not one generic call.
2. **Agent-session snapshot** — freezing and restoring a *live ACP session* (and the running agent process and its context behind it) is the same class of hard problem. Inspect can only deep-copy conversation state precisely because it cannot snapshot the process. The architecture does not get this for free.
3. **Cross-layer consistency** — the three snapshot layers (container / environment-state / agent-session) have different consistency models; a naïve capture can produce a container snapshot and a DB snapshot that disagree about a write in flight. The `Branch` lifecycle therefore **quiesces the agent first**, then snapshots environment → container → session in order.

Tree-native rollout *structure* is proven and safe to commit to. The `Branch` *checkpoint* — all three layers — is the frontier: it is where the engineering risk concentrates and where the moat is, and it must be designed deliberately, not hand-waved as one call.

## Adaptations — the decision log

So decisions are not re-litigated.

| Adaptation | From → To | Why |
|---|---|---|
| **One picture, no build-order tiering** | a core/deferred two-altitude split → the whole architecture as one coherent overview | The overview describes *what we need*; sequencing is a roadmap concern (Linear), not a property of the concepts. |
| **The Rollout is a tree** | a linear Rollout with optional, deferred branching → a tree-native Rollout; linear = degree-1 | Agentic RL is a POMDP; `V(s)` is defined over a tree; Tree-GRPO shows the tree manufactures credit assignment. Branching is the engine, not a feature. |
| **`snapshot`/`restore` are real** | platform-layer `NotImplementedError` stubs → definitional methods on the Environment contract | Han: *"Environment 总是要 roll out, roll back."* |
| **Branching's purpose is the value function** | a user-feedback feature → credit assignment / `V(s)` estimation; `ask_user` and GRPO groups are two cases of it | Han: *"from reward function to a value function."* Tree-GRPO confirms it. |
| **`Session` is in the contract** | the Agent plane returned an untyped `Session` → `Session` is a specified `Protocol` | The whole ACP interaction is the Agent seam; a shallow `connect`-only contract can't carry BYO agents. |
| **Scene fully desugars** | a runtime object with its own lifecycle → pure authoring sugar; `RolloutConfig.scenes` lowers to per-`Step` config | Scene adds no expressive power; a phase is just state (the RFC's original instinct). |
| **Renamed `Batch` → `Job`** | `Batch` for a set of Rollouts → `Job` | "Batch" means the gradient minibatch in every trainer — a collision at the trainer seam. |
| **Manifest as the only seam** | benchmarks subclass framework internals → a declarative manifest, zero framework modification | A benchmark must never modify the framework. |
| **eval = monitoring = reward** | three framings → one engine, three modes | Han's single biggest complexity reducer. |

## Appendix — research validation

Checked against the recent agentic-RL literature; the field's shape matches.
- Agentic RL = a temporally-extended POMDP — definitionally a branching structure. *(The Landscape of Agentic RL, 2509.02547)*
- Tree-structured rollouts yield process supervision from a single outcome reward and more rollouts per token/tool budget. *(Tree Search for LLM Agent RL / Tree-GRPO, ICLR 2026, 2509.21240)*
- All 13 surveyed RL libraries model rollouts linearly with no environment snapshot/restore — tree-native + heavy-environment snapshot is real novelty, not a reinvention.
- Rollout-as-a-service decoupled from training; the trajectory is the seam. *(ProRL Agent; PrimeIntellect Environments Hub)*
- Continual learning = a base policy + a persistent, evolving skill library; version and roll back the store. *(MetaClaw, MemSkill, SkillLearnBench)*

**Verdict:** the architecture is consensus-correct on shape and deliberately ahead of the field on one primitive — the `Branch` checkpoint (environment + agent-session + container snapshot) for stateful branching. The risk is execution of that primitive, not the design.
