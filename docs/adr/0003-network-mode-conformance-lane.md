# 3. network_mode gated by a dedicated docker-only conformance lane

- Status: Accepted (ENG-265)
- Date: 2026-06-17

## Context

`network_mode` egress enforcement (the allowlist proxy + fail-closed lockdown shipped
in ENG-219/263/264) has **zero presence** in the release-gating integration suite —
no axis, lane, scenario, or checker field — so a network-posture regression ships
green. The conformance logic already exists and runs in `net/live_lane_test.py`
(allowlisted host reachable, non-allowlisted host BLOCKED via the real egress
sidecar). Daytona-credentialed lanes are availability-gated and flaky in CI; docker
needs no external creds and matches the proven `live_lane_test.py` setup.

## Decision

Add a **dedicated, docker-only, release-blocker lane** (`network-mode-enforcement`)
that runs **per-PR** and **hard-fails (never skipped)**, asserting via the egress
sidecar (logic ported from `net/live_lane_test.py`):

1. `no-network` blocks all egress;
2. `allowlist` permits only the listed hosts **and** the resolved model-provider host;
3. a disallowed-host egress attempt is denied with a clear signal.

A cheap **recorded-mode identity check** in `check_results.py` complements it (the
requested `network_mode`/`allowed_hosts` reconciles across config↔result↔summary),
mirroring how `agent_idle_timeout` is reconciled. `network_modes` becomes a
first-class axis. A **docker↔daytona parity** assertion is added to the nightly tier
only (contingent on reliable Daytona credentials), not the per-PR gate.

## Consequences

- (+) The security invariant is gated on every PR without Daytona credentials, as a
  hard-fail; a network-posture regression can no longer ship green.
- (+) Reuses already-proven machinery (`net/live_lane_test.py`) rather than
  re-deriving it.
- (−) The per-PR gate is docker-only — daytona enforcement parity is detected only in
  the nightly tier.
- (−) Requires threading `network_mode`/`allowed_hosts` through `scenarios.run_eval`,
  the per-agent configs, and the `check_results` expected-identity reconciliation.

## Alternatives considered

- **Layer network_mode onto an existing lane** (e.g. shared-sandbox-smoke): collides
  with those lanes being plan-only today and couples the security check to unrelated
  scope. Rejected in favor of a self-contained lane.
- **Identity check only** (recorded mode, no conformance): cheap but a true egress
  regression with the right mode recorded still passes. Kept as the complement, not
  the gate.
