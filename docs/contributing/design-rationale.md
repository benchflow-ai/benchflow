# Design rationale

This page explains why BenchFlow is built the way [Architecture](../architecture.md) describes. It is for contributors. Users need only the architecture page.

## The bet

A complete RL environment has five parts: Tasks, Harness, Verifier, State and Config, written `{T, H, V, S, C}`. BenchFlow aims to cover all five.

State is the part we build around. Stateful, multi-service environments that can roll out, roll back and branch are where agentic RL is heading. Few tools support them today.

BenchFlow owns the environment, the rollout and the reward. Trainers own the weights, the gradients and the optimizer. The trajectory is the interface between them, so any RL trainer can read BenchFlow output.

## Grounding

The design rests on three sources: a published decomposition of RL environments, our own design discussions, and the agentic-RL literature.

### The `{T,H,V,S,C}` decomposition

The split comes from Han Lee's post *RL Environments for LLM Agents*. Its definitions:

| Part | Definition in the post |
|---|---|
| T, Tasks | "problems the agent tries to solve" |
| H, Harness | "scaffolding that … controls *how* the model interacts, but does not improve what it knows" |
| V, Verifier | "V: (task prompt, completion, info) → [0,1]" |
| S, State | "stateless (fresh starts) … or stateful (persistent across actions/episodes)" |
| C, Config | "turn limits, context budgets, sampling temperature, curriculum scheduling" |

### What the five parts leave out

A static set of five parts leaves out the dynamics. Our design discussions added these:

- An environment must both roll out and roll back. Rolling back means snapshotting the environment and returning it to an earlier stage, so both are part of what an environment is.
- A point where the agent asks the user is a checkpoint: other kinds of rollout can branch from it. Branching turns a reward function into a value function over the current state, which matters most for long-horizon tasks.
- Evaluation, monitoring and reward are one activity in different settings. It is observed across five spaces: output, action, reasoning, memory and latent.
- The harness is not where intelligence lives. Self-improvement targets the model and its skills, never the harness, and skills belong to memory.
- ACP is the way to model human interaction inside a rollout.

### The literature

- Agentic RL is a "temporally extended, partially observable MDP" ([The Landscape of Agentic RL](https://arxiv.org/abs/2509.02547)). Such a process branches by definition.
- Tree-structured rollouts give "step-wise process supervised signals even using only the outcome reward". They also give "more rollouts within a fixed budget of tokens or tool calls" ([Tree Search for LLM Agent RL](https://arxiv.org/abs/2509.21240), Tree-GRPO).
- We surveyed RL libraries: verifiers, prime-rl, SkyRL, verl, NeMo-RL, Tinker, OpenEnv, Terminal-Bench, Inspect, ORS, Gymnasium and agent-lightning. All of them model rollouts linearly.

So a tree-native rollout with environment snapshot and restore is new. The hard part is snapshotting heavy environment state (see [The hard part](#the-hard-part)).

## More design principles

[Architecture](../architecture.md#design-principles) lists the five principles users meet. These are the rest:

1. **Rolling back is part of an environment.** An environment that cannot snapshot and restore its state is incomplete.
2. **Eval, monitoring and reward are one activity,** scored on the same trajectory across five spaces. Monitor mode is not built yet.
3. **The environment is a state machine** that BenchFlow provisions, snapshots, restores and tears down.
4. **BenchFlow is the ACP client.** The "user" is a policy you plug in, not a special actor.
5. **The harness is not intelligent.** Its only job is to get the most from the model. Self-improvement targets the model and its skills.
6. **Every export has a live caller.** Code that nothing calls is wired in or deleted.
7. **Shipping beats design.** A better design that doesn't run loses to an adequate one that does.

## Why the rollout is a tree

An RL trajectory is a chain of state → action → next state. A branch turns that chain into a tree. Classical RL is defined over this tree, and the value function `V(s)` is the expected return over the continuations from a state. So a tree is the natural shape for a rollout.

A branch evaluates one state across N continuations. Averaging the children's returns estimates `V(s)`. That turns a reward function into a value function, and it matches Tree-GRPO's result that a tree gives process supervision from one outcome reward.

A GRPO group run as a shared-prefix tree gets more rollouts per token or tool budget than N independent rollouts.

A trajectory is a function of the tree, never declared on its own. A scene is authoring sugar: it compiles to per-step roles and config, and adds no expressive power.

### What the tree costs

A linear rollout is a tree with one child per node, so the data model costs nothing extra. The engine is different. Checkpoint and fork, layered snapshots, node-addressed scoring and child scheduling exist even for users who only run linear rollouts. We accept that cost: the linear path inherits the tree's machinery.

## Stateful benchmarks

ClawsBench and chi-bench are built the same way: services the agent reads and writes, backed by a database. The Environment plane hosts both.

chi-bench keeps its environment unchanged and joins BenchFlow through a short manifest (`benchmarks/chi-bench/environment.toml`, 15 lines of settings). ClawsBench uses `base_image` with services BenchFlow starts, and per-task images.

A shared, long-lived service fleet with per-task accounts is a possible next step for scale. It is not built.

## Capabilities

The architecture has to carry eight capabilities. Capabilities 1 to 6 and 8 are each tied to a benchmark: a capability is done when that benchmark runs cleanly. Capability 7 is the substrate the others ride on.

| # | Capability | How it fits the architecture |
|---|---|---|
| 1 | SkillsBench | An Environment-plane benchmark package with skills and skill-eval tasks. Skills are memory, so the memory space scores skill use and skill updates. Skills are mounted into the sandbox where the agent looks for them ([Skill loading](../architecture.md#skill-loading)). |
| 2 | ClawsBench | A stateful multi-service benchmark: `base_image` + `[[environment.services]]` started by BenchFlow, with `image` task selection. |
| 3 | chi-bench | The same kind of benchmark: `image` + `owns_lifecycle = true` + `env_var` task selection, with its environment unchanged. |
| 4 | followupbench (NudgeBench) | The ACP user model (`session/prompt` nudges, `ask_user` through `Session.on_ask_user`), explicit branches at chosen points, and action-space scoring of whether the agent asked. |
| 5 | Continual learning | A job with `job_mode: sequential-shared`. Rollouts run in order over a persistent learner store of memory and skills. The store stamps a generation per write and reverts a regression (`LearnerStore.commit_or_revert`). It never rolls back with a branch. |
| 6 | RL-native | The rollout is a tree, the trajectory is a path, a `Reward` scores any node, and each job writes trainer-ready records. |
| 7 | Branching and rollback | Not a benchmark: the substrate itself. It covers branches, environment snapshot and restore, the value-function purpose of the tree, and the five spaces. Capabilities 4 to 6 ride on it. |
| 8 | Adapters: task.toml, PrimeIntellect, OpenReward | `bench run` reads `task.toml` folders directly. It ignores unknown keys with a warning and refuses keys it cannot honour. The inbound adapter reads them into an `InboundTask`. Outbound, trajectories export to Verifiers and ORS. |

## The hard part

No agentic-RL library we surveyed ships environment snapshot and restore. Tree-GRPO branches a token prefix, not a world. BenchFlow branches a heavy stateful environment, such as a mock Gmail database or a healthcare simulator.

A branch checkpoint is three hard problems, not one:

1. **Environment-state snapshot.** Database dump and restore, copy-on-write volumes, services that can fork. Each environment class needs its own design. Today BenchFlow copies SQLite files.
2. **Agent-session snapshot.** Freezing a live ACP session, with the agent process and its context, is just as hard. It is not built. A child either starts a fresh session or replays the parent's conversation with ACP `session/load`.
3. **Consistency across layers.** The sandbox and the environment state have different consistency models. A naive capture can produce a container snapshot and a database snapshot that disagree about a write in flight. So a branch quiesces the agent first, then captures environment state, then the container.

The tree structure is proven and safe to build on. The branch checkpoint is where the engineering risk sits, and it needs deliberate design per layer.

## Decision log

These decisions are settled. Reopen one only with new evidence.

| Decision | From → to | Why |
|---|---|---|
| One picture, no build-order tiers | a core/deferred split → one overview of the whole architecture | The overview describes what we need; sequencing belongs to the roadmap. |
| The rollout is a tree | a linear rollout with optional branching → a tree; linear = one child per node | Agentic RL is a POMDP; `V(s)` is defined over a tree; Tree-GRPO shows the tree gives credit assignment. |
| `snapshot`/`restore` are real methods | `NotImplementedError` stubs → methods on the `Environment` contract | An environment must both roll out and roll back. They need `[environment.state]` in the manifest. |
| Branching serves the value function | a user-feedback feature → credit assignment and `V(s)` estimates | `ask_user` points and GRPO groups are two cases of it. |
| `Session` is in the contract | an untyped return → a `Session` protocol | The whole ACP exchange runs through it; a `connect`-only contract can't carry your own agent. |
| Scene compiles away | a runtime object with a lifecycle → authoring sugar; `RolloutConfig.scenes` compiles to steps | A scene adds no expressive power; a phase is just state. |
| `Batch` renamed to `Job` | `Batch` → `Job` | "Batch" means the gradient minibatch in every trainer. The rename is partial: `benchflow.batch` and `run_batch` still exist. |
| The manifest is the integration point | benchmarks subclass BenchFlow internals → a declarative manifest | A benchmark must never modify BenchFlow. |
| Eval, train and monitor are one engine | three framings → one scored rollout, three modes | It removes the most complexity. Monitor is still a stub. |

## Research checks

The design matches the recent agentic-RL literature:

- Agentic RL is a temporally extended POMDP, which branches by definition ([2509.02547](https://arxiv.org/abs/2509.02547)).
- Tree-structured rollouts give process supervision from one outcome reward, and more rollouts per budget ([2509.21240](https://arxiv.org/abs/2509.21240)).
- The 13 libraries we surveyed model rollouts linearly, with no environment snapshot or restore.
- Rollouts can run as a service, decoupled from training, with the trajectory as the interface (ProRL Agent; PrimeIntellect Environments Hub).
- Continual learning pairs a base policy with a persistent, evolving skill library that is versioned and can roll back (MetaClaw, MemSkill, SkillLearnBench).

## Branch guarantees

How branching keeps a fork faithful to its parent, for maintainers. Users need only [Branching](../jobs/branching.md) and the [Branching reference](../jobs/branching-reference.md).

Capture runs environment first, then sandbox; restore runs sandbox first, then environment, which preserves the in-container SQLite backup files. Collision-safe `StateSnapshot.files` mappings stay attached to the environment handle. If both a child's and the parent's restore fail, both errors are kept in an exception group.

Docker snapshots a committed image and recreates the main container, keeping the supported host configuration and mounts; it rejects unsupported reconstruction before removing the old container. Daytona direct advertises snapshot support; Daytona DinD and other backends fail the capability check.

SQLite restore after container recreation removes captured WAL, shared-memory and rollback-journal sidecars before installing the standalone online backups, so later writes cannot replay over them. This needs stopped database processes and does not apply to environment-only restore. Entrypoint-owned service lifecycles are rejected before capture. There are no cross-service transactional snapshots. Daytona snapshot names carry the owner scope (`bf-snap-<owner>-<owner hash>-…` when `BENCHFLOW_DAYTONA_OWNER` is set).

Credential scrubbing was chosen over refusing the sandbox layer: each child starts a fresh agent session that needs those files, so refusing would rule out sandbox branching for codex-acp and Claude host-login users. Native Claude OAuth through `CLAUDE_CODE_OAUTH_TOKEN` writes no file. On Daytona the files move through the file API, because the Daytona daemon keeps session command output on disk inside the sandbox.

Custody: in-place children hold the parent's mounted log entries aside under the fork directory. Parent entries and mutable result state (timing, native usage, solver recovery markers) are restored after a child or environment restore failure when the child was quiesced. If disconnect or provider cleanup cannot finish within its grace period, the entries stay held and world restore is deferred (`parent_restore: deferred`). The rollout keeps an unsafe-world marker and refuses further work; a failed parent restore does the same. Child artifact links appear only after observation writing and custody succeed. A `tree.json` publication failure keeps the previous valid file.

Already-active shared provider runtimes are rejected before branching: background callback writers and cumulative usage need a runtime fork contract. The bounded custody path supports native-subscription and provider-free continuations; it is not general proxy-runtime isolation.

## Next steps

- [Architecture](../architecture.md): the planes, contracts and lifecycles.
- [Composed checkpoints](../jobs/branching-reference.md): how branch layers are captured and restored.
- [Environment plane](../environment-plane.md): manifests and state.
- [Release](./release.md): how BenchFlow ships.
