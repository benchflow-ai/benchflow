# 1. Integration-test lane execution home — matrix cells, not pytest markers

- Status: Accepted; partially implemented in #802 / remainder deferred
- Date: 2026-06-17 (revised for #802: 2026-06-18)

## Context

> **Ported from #799 and corrected to #802's reality.** The original ADR (written
> against the `tests/integration/suites/release.yaml` + `run_suite.py` planner)
> assumed lanes would resolve to **pytest markers** in a *missing*
> `tests/integration/test_integration_suite.py`. In #802 both premises are false,
> so the Context and Decision below are restated for #802. The Consequences /
> Alternatives are kept.

In #802 the integration CI is the **tiered L0–L3 system**, not the
`release.yaml`/`run_suite.py` planner that #799 hardened. The execution home is
already built and is **not** a per-lane pytest marker:

- **Lanes execute as matrix CELLS.** The planner
  `.github/scripts/integration_matrix.py` selects one of the seven named
  **task-sets** (`citation` / `low-smoke` / `low-3` / `medium-3` / `high-3` /
  `nine` / `expanded`, see [`../integration-tiers.md`](../integration-tiers.md)
  §2) from the diff and emits a matrix of cells. Each cell is run via
  `tests/integration/scenarios.run_eval(...)` (the real `bench eval run` path) by
  the L1/L2/L3 workflows (`.github/workflows/integration-light.yml`,
  `integration-scope.yml`, `integration-final-review.yml`), and graded by the
  trusted-main grader `.github/scripts/build_integration_review_pack.py` over
  `tests/integration/rubric_checks.py` (`RUBRIC_GATES`). There is **no
  lane↔marker convention** and **no `run_suite.py --run-lane`** in the gating
  path.

- **`tests/test_integration_suite.py` EXISTS** (note the path: `tests/`, not
  `tests/integration/`). It is the single home for the live oracle / parity /
  agent-judge scenarios — its `@pytest.mark.integration` tests call
  `scenarios.run_eval` and are invoked by the L2/L3 workflows (`uv run pytest
  tests/test_integration_suite.py -m integration ...`). Its **unmarked**
  deterministic integrity gates (realness rejects unscored/no-work rollouts, the
  agent judge fails closed, the judge catches reward-hacking the mechanical gate
  cannot, artifacts are well-formed and leak no secrets) run in the normal
  credential-free suite. So #802 already asserts **outcomes**, not just intent,
  for the cheap gates, and the live scenarios skip cleanly when prerequisites
  (Docker, Daytona, provider keys) are absent.

- **The orphan agent lanes are plan-only intent.**
  `tests/integration/suites/release.yaml` still ships in #802 as a *declarative
  backlog manifest* (release-blocker lanes + run-tracking evidence fields), and it
  still names three agent-running lanes that nothing in the L0–L3 matrix
  executes: `shared-sandbox-smoke`, `skillsbench-agent-matrix`, and
  `decoupling-checks`. They are not wired to `scenarios.run_eval`, so — exactly as
  #799 observed of `run_suite.py` — `block_release` over them asserts intent, not
  pass/fail.

## Decision

The **matrix cell** (`integration_matrix.py` → `scenarios.run_eval` →
`rubric_checks.py`) is the authoritative lane execution home in #802; the
`release.yaml`/`run_suite.py` planner is **not** resurrected as a gating path.
`tests/test_integration_suite.py` remains the single home for live end-to-end
scenarios (the role #799 wanted `scenarios.py`-via-markers to play), reachable
from CI.

The three orphan agent lanes **fold into the matrix task-sets** rather than
becoming pytest markers: `skillsbench-agent-matrix` is subsumed by the
`nine`/`expanded` agent×task fan-out; `shared-sandbox-smoke` and
`decoupling-checks` map onto the Docker/Daytona-parity and sandbox/root/path
scope rules of the Default-config-rules ([`../integration-tiers.md`](../integration-tiers.md)
§3). `release.yaml` is retained only as the declarative backlog/run-tracking
manifest, not as an execution surface, until those lanes are demonstrably covered
by matrix cells (then the orphan lane ids are removed from `release.yaml`).

## Consequences

- (+) One authoritative home for assertions (the matrix cell + `rubric_checks.py`,
  with `tests/test_integration_suite.py` for live scenarios), reachable from CI;
  the gating lanes resolve to real pass/fail.
- (+) `block_release` becomes meaningful (outcomes, not TODO-absence) for any lane
  that maps to a matrix cell.
- (−) `release.yaml`'s three orphan agent lanes are intent-only until their
  coverage is demonstrably folded into the matrix task-sets (the deferred
  remainder of this ADR).
- (−) A scope→task-set→cell convention must be enforced in the planner
  (`integration_matrix.py`) and the scope map
  (`.github/integration/scope_map.yml`) so the mapping cannot silently drift.

## Alternatives considered

- **Resurrect `run_suite.py` as a `--run-lane` pytest-marker executor** (the
  original #799 decision): adds a second, parallel execution path next to the
  already-built L0–L3 matrix. Rejected for #802 — it would duplicate the planner
  and split assertion logic between `run_suite.py` and the matrix grader.
- **`run_suite.py` executes lanes directly** (extend the `--execute-*` pattern):
  fewer files, but mixes planning with execution and splits assertion logic.
  Rejected.
- **Hybrid** (run_suite runs cheap lanes, the matrix for the rest): two execution
  paths to maintain. Rejected.
