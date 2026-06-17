# 1. Integration-test lane execution home — planner + pytest markers

- Status: Accepted (ENG-265)
- Date: 2026-06-17

## Context

The release suite (`tests/integration/suites/release.yaml`) declares lanes, and
`run_suite.py` plans them, but only 4 of 8 lanes actually execute. `block_release`
is enforceable today only for unresolved TODO/`blocked_by` strings — the
agent-running release-blocker lanes (shared-sandbox-smoke, skillsbench-agent-matrix,
decoupling-checks) are plan-only intent. The reusable assertion library
(`scenarios.py`) is consumed by `tests/test_integration_suite.py`, which is **absent
from the repo**, and is not imported by `run_suite.py`. So the suite asserts intent,
not outcomes, for most of what it claims to gate, and there is no single home for
end-to-end assertions.

## Decision

`run_suite.py` stays a **pure planner** over the named-axis / named-lane manifest.
Every lane id resolves to a **pytest marker** in `tests/test_integration_suite.py`
(committed as part of ENG-265), which is the sole importer of `scenarios.py`.
`run_suite.py` gains `--run-lane <lane-id>` that resolves the lane to its pytest
selector and shells out to `pytest`. Both CI's live-scenarios step and the suite
gate target the same markers. `load_suite` validates that every lane id has a
corresponding marker (and vice versa) so the mapping cannot silently drift.

## Consequences

- (+) One authoritative home for assertions (`scenarios.py`), reachable from both
  CI and the suite gate; release-blocker lanes resolve to real pass/fail.
- (+) `block_release` becomes meaningful (outcomes, not TODO-absence) for any lane
  with a marker.
- (−) `tests/test_integration_suite.py` must be authored/committed (explicit ENG-265
  deliverable).
- (−) A lane↔marker naming convention must be enforced in `load_suite`.

## Alternatives considered

- **`run_suite.py` executes lanes directly** (extend the `--execute-*` pattern):
  fewer files, but mixes planning with execution and splits assertion logic between
  `run_suite.py` and `scenarios.py`. Rejected.
- **Hybrid** (run_suite runs cheap lanes, pytest for the rest): two execution paths
  to maintain. Rejected.
