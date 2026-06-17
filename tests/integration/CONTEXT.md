# CONTEXT — integration-test workflow (glossary)

Canonical vocabulary for `tests/integration/` + `.github/workflows/integration-eval.yml`
+ the `benchflow-experiment-review` skill. Glossary only — no implementation detail.

- **Integration test** — an end-to-end run that exercises the *real* eval path
  (adapters, sandboxes, agents, verifiers) via `bench eval run`, asserting the
  produced artifacts are trustworthy. NOT a unit test; NOT the agent's own
  task-pass/fail.

- **Rollout contract** — the artifact set every producer emits and every checker
  consumes: `result.json` + `config.json` + the ATIF/ADP trajectory. The shared
  interface that lets producers, checkers, and the judge stay decoupled.

- **Producer** — something that emits the rollout contract. `run.sh` (real agents
  on real tasks) and `deepagents_harness.py` (a *fixture factory* — a steerable
  agent that manufactures genuine and reward-hacking rollouts to harden the
  judge) are the two producers; they do not call each other.

- **Axis** — a named dimension of variation declared in `suites/release.yaml`
  (e.g. `agents`, `models`, `tasks`, `sandboxes`, and — new — `network_modes`).

- **Lane** — a named, executable unit of the release suite that selects specific
  axis values (deliberately not the full cartesian product). A lane resolves to a
  pytest marker in `tests/test_integration_suite.py`.
  - **Release-blocker lane** — a lane whose failure blocks release
    (`policy.on_failure: block_release`).

- **Scenario** — a reusable end-to-end assertion primitive in `scenarios.py`
  (the single home for assertions); lanes are composed of scenarios.

- **Realness gate** — the mechanical, judge-independent check that a rollout is a
  genuine run (tool calls > 0, tokens > 0, non-null reward, no infra/verifier
  error). ANDed with the LLM judge in `agent_judge.py`.

- **Verifier tamper** — an agent mutating the score-defining (verifier) files to
  fake a reward. Detected today by regex on shell input; hardened to a
  producer-side before/after file **hash** surfaced as a definitive boolean.

- **network_mode — identity vs conformance** (two distinct guarantees):
  - **Identity check** — the recorded mode in `result.json`/`config.json` matches
    the *requested* mode (cheap; catches "wrong posture requested"). Lives in
    `check_results.py`.
  - **Conformance check** — the run *actually* observed the right egress: an
    allowlisted host is reachable AND a non-allowlisted host is **blocked** under
    the enforced mode (the real security guarantee). A scenario ported from
    `net/live_lane_test.py`.

- **Evidence check** — a standalone checker that turns rollout artifacts into a
  release gate (`check_results`, `check_adapter_evidence`, `check_hosted_env_*`,
  `check_trace_to_task_*`, `check_skillsbench_harbor_parity`).

- **CI tiering** — *per-PR* gate: cheap, docker-only, hard-fail (one
  task×agent×model judge+checks + the network_mode conformance scenario).
  *Nightly/manual*: the full agent×task×sandbox matrix (advisory, promotable to
  release-blocker on tags).
