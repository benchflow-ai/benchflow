# BenchFlow v0.5 — Roadmap

*The engineering-sequencing companion to [`architecture.md`](./architecture.md). The architecture doc is one coherent picture with deliberately **no build-order tiering**; this doc is where the order, the gates, and the status live. Canonical milestone tracking is the Linear project "BenchFlow v0.5 — architecture migration" (rendered in the dashboard via `dashboard/roadmap.py`); this file is the in-repo view engineers work from.*

Last updated: 2026-05-28.

## The thesis being built

A thin **kernel** depending only on `contracts/`, driving four swappable planes — **Sandbox · Agent · Environment · Reward** — over a **tree-native Rollout** whose `Branch` turns a reward function into a value function, with the scored **trajectory** as the seam to trainers. The readiness assessment (2026-05-28) put the substrate at ~47/100: the headline primitives existed as tested libraries with **no production call path**. The phases below wire them onto the live path, each **gated by a real `bench`/SDK run**, not a unit test.

## Phase ladder (each gated by a real run)

| # | Phase | Gate | Status |
|---|---|---|---|
| **0** | **Contracts seam + kernel-on-contracts** — `benchflow.contracts` aggregates the four Protocols; sandbox/environment provider registries; kernel typed against contracts, providers resolved via registry. | oracle 1/1 via CLI **and** `bf.run()`; kernel imports the four Protocols from `contracts/`; providers resolve through the registries. | ✅ **done** — [#577](https://github.com/benchflow-ai/benchflow/pull/577) |
| **1a** | **Canonical reward on the live path** — `Reward.score(node)→VerifyResult` built once during scoring (`verify_result_from_reward_map`), persisted to `verifier/verify_result.json`, **read** by the trainer export (no more downgrade-on-export). `result.json['rewards']` stays the legacy dict. | real oracle run: legacy dict intact, `verify_result.json` carries `(space, granularity)`, `verifiers.jsonl` sourced from the VerifyResult, all headline rewards byte-identical. | ✅ **done** — [#578](https://github.com/benchflow-ai/benchflow/pull/578) |
| **2** | **Prove snapshot/restore on a real stateful benchmark** — add `[environment.state]` (sqlite) to clawsbench; `ManifestEnvironment.snapshot/restore` does a real round-trip; readiness gate framework-owned + unconditional across providers. | Docker-gated test: seed a real SQLite DB → snapshot → mutate → restore → assert rollback; clawsbench runs provision→readiness→verify on Docker. | 🛠 **in progress** (investigation workflow → implement) |
| **3** | **Wire `Branch` into a real run (3-layer checkpoint)** — the moat. Real branch trigger (`ask_user` options / value branch); `checkpoint` composes container ⊃ env-state ⊃ session; aggregate `V(parent)` from node-scored child returns. **Bundles deferred Phase 1b** (route branch aggregation through `VerifyResult`). | real fork on clawsbench; e2e asserts a docker-commit **and** a sqlite `.backup`; container mutations roll back per child; `V(parent)` from child VerifyResults; visible in dashboard. | ⏳ planned (closes the last P0) |
| **4** | **Capability packs — NudgeBench + continual learning** — a real NudgeBench pack (ACP nudges + `on_ask_user` + Branch-per-interaction + Action-space reward for *whether the agent asked*); sequential-shared CL over the versioned LearnerStore. | NudgeBench yields non-null Action-space follow-up rewards + an `ask_user` Branch child; CL run produces versioned LearnerStore generations with Memory-space deltas — both in the dashboard. | ⏳ planned |
| **5** | **Tree-derived trajectory + real trainer handoff** — export derived from the tree (structured tool calls, not flattened); validated against the real Verifiers `RolloutOutput`; a documented prime-rl ingestion recipe. | `verifiers.jsonl` from a real run validates against the Verifiers type and loads in a prime-rl/Verifiers smoke test; tool-call structure survives round-trip. | ⏳ planned |
| **6** | **Adapter edge consolidation + first-class interop** — one `adapters/` package (inbound/outbound/identity), delete `compat/`, fold `hosted_env.py`; **two-track edge** (translate-to-manifest **and** run-via-their-own-system). | one run loop drives a **verifier** (`vf-eval`), a **Harbor** (`harbor run`), and an **OpenReward** task through their native systems → scored rollout + `verifiers.jsonl` artifact, all in the dashboard. | ⏳ planned |
| **7** | **Surface honesty** — dashboard sample-jobs fixture; flag aspirational docs (Monitor stub, latent space) as planned vs shipped. | clean checkout's dashboard shows ≥1 real scored task; docs distinguish shipped from planned. | ⏳ planned |

## Cross-cutting workstreams

These thread through the phases above rather than being a single phase.

### A. The native environment package format (design RFC — in progress)
A BenchFlow-native env **package** (folders + files + manifest), distinct from the Harbor task dir, that first-classes:
- **Simulated users** packaged *with* the environment — a **User Model** (scripted / LLM-persona / dual-agent) the kernel drives via the ACP Client role. Grounded in the **τ-bench family** (τ-bench / τ²-bench) tool-agent-user model, where the user is a stateful participant with its own policy, not a fixed prompt.
- **OpenReward-style** reward packaging alongside the native `[verifier]` / Rubric path.

This **extends** the existing Environment manifest + `[environment.state]` (Phase 2) — it does not replace it. The on-disk spec (file layout, the user-policy file, desugaring to `RolloutConfig.user` + per-`Step` config) is being produced by the **env-package-format research+design RFC** (primary-source research on τ-bench / OpenReward / Verifiers / Harbor). Lands into Phase 4 (NudgeBench needs the simulated-user package) and Phase 6 (interop). See [`architecture.md`](./architecture.md) → "The Environment plane & the manifest".

### B. End-to-end interop — "run their systems directly" (→ Phase 6)
Prove BenchFlow **composes with** each ecosystem by invoking its own runner, not re-implementing it:
1. **Verifier tasks** — via `vf-eval` (seed already exists in `hosted_env.py`).
2. **Harbor tasks** — via `harbor run`.
3. **OpenReward tasks** — via the OpenReward runner.
BenchFlow supplies sandbox + rollout + trajectory capture + trainer export around the foreign runner. This is **Track 2** of the two-track edge in `architecture.md`; Track 1 (translate-to-manifest) is the native-plane path that unlocks state/branching.

### C. Our differentiators (pushed on our side, not external-compat)
- **NudgeBench** (simulated-user follow-up evaluation) — Phase 4.
- **Rollout branching** with environment + (eventually) 3-layer snapshot — Phase 3, the architectural moat no surveyed RL library ships.

## Working process
Each phase: a read-only **investigation workflow** traces call paths + adversarially verifies assumptions + produces a concrete gated edit plan → sequential edits applied directly (parallel agents must not co-edit the hot `rollout.py`) → real gate → **thermo-nuclear structural-quality review** subagent → PR. PRs target `v0.5-integration` (not `main`) and stack.
