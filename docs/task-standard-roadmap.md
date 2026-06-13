# Task Standard Roadmap

Companion to [the task standard](./task-standard.md). It lists the open
primitives the standard will add, each tagged by plane and by whether it needs
new **runtime** or is **schema-only**, and the milestone sequencing for
landing them.

## Open Primitives

| # | Primitive | Axis / plane | Cost | Driving benchmarks |
|---|---|---|---|---|
| G1 | Hybrid trajectory×terminal reward envelope (declared product/sum of factors across surfaces) | Verifier | schema + light runtime | tau-bench (DB-state × action × NL), HILBench (Ask-F1 × success) |
| G2 | Dynamic-baseline / GAIN aggregation: `(system − live baseline) / live ceiling` | Verifier | schema + runtime | continual-learning-bench |
| G3 | Declarative model-backed simulated-user as a first-class concept | Interaction | schema + small runtime | tau / NudgeBench |
| G4 | Live assessor-agent / arena plane (A2A+MCP green vs purple, scored during) | Interaction | **heavy runtime** | AgentBeats/AAA |
| G5 | Leaderboard-submission reward surface (hosted / hidden external scorer) | Verifier | schema + integration | hosted competition platforms, AAA leaderboard |
| G6 | RL step-reward with environment reaction (per-action shaped reward) | Verifier + Runtime | runtime | OSWorld/BrowserGym, CLB |
| G7 | Language-neutral, runtime-separable environment interchange schema | Runtime / Package | schema (foundational) | Verifiers, hosted competition platforms |
| G8 | Per-phase / per-step runtime policy as executable behavior | Runtime | schema + runtime | Multi-step `steps` |

The standard already addresses **G3, G5, G7, G8** partially (see `user` /
`benchflow.nudges`, `acceptance_live.leaderboard`, the language-neutral schema
goal, and `runtime_policy.phase_overrides`). The genuinely missing primitives are
**G1 (hybrid reward), G2 (GAIN), G4 (arena), and G6 (RL step-reward)**; only G4
needs heavy new runtime.

## Sequencing

The schema covers all three axes now, but cross-standard interop
is narrower than the schema. ORS interop today is a reward-contract bridge —
`adapters/ors.py` exports `VerifyResult`/`RewardEvent` records in the ORS
reward-response format, and the `ors-episode` strategy normalizes declared
tool-output rewards — not a task or environment importer. There is no AAA
import adapter; the arena plane is schema-only until M2. Split-layout
conversion is the proven import/export path (see the conversion-parity
evidence). Runtime lands in tiers:

- **M0 (package + sequential runtime, today):** native authoring, `workspace-test`
  + `llm-judge`, environment interchange, per-phase policy, compat import/export.
  Fully unlocks tiers 1 and 3 plus split-format / Terminal-Bench / SWE-bench / METR /
  ORS-import.
- **M1:** trajectory/episode reward fed the structured trajectory (G1, G6),
  declarative simulated-user (G3), GAIN aggregation (G2), leaderboard-submission
  (G5). Unlocks tau / NudgeBench, CLB-GAIN, hosted leaderboard submission.
- **M2 (heaviest):** `arena-concurrent` runtime — A2A bridge + concurrent assessor
  (G4). Unlocks AgentBeats/AAA natively. The schema already represents it, so M2
  is additive, not a rewrite.
