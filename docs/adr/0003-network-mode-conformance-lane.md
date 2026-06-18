# 3. network_mode — static scope-gated assertion now, runtime conformance lane deferred

- Status: Accepted; partially implemented in #802 / remainder deferred
- Date: 2026-06-17 (revised for #802: 2026-06-18)

## Context

> **Ported from #799 and corrected to #802's reality.** The original ADR locked a
> **runtime egress conformance lane** (an always-on, docker-only, hard-fail lane
> asserting real allow/deny via the egress sidecar, ported from
> `net/live_lane_test.py`) as the per-PR gate. #802 ships only the cheap
> **static** half today and defers the runtime conformance lane — it is **blocked**
> on a dependency that does not exist in main. The Decision restates that split.

`network_mode` egress posture has presence in #802's integration system, but only
as a **static** check. There is **no `bench eval run --network` flag** and
`network_mode` is **never serialized into a rollout artifact** — it is a per-task
config field only (`src/benchflow/task/config.py`). So #802 gates network as a
**scope-gated static config assertion**:

- The grader's **`V-NETWORK`** check (`rubric_checks.network_hardening`, surfaced
  in [`../integration-review-rubric.md`](../integration-review-rubric.md)) asserts
  the *declared* policy is hardened — default `no-network`; `allowlist` passes
  only with a non-empty `allowed_hosts`; bare `public` is flagged (a hard `fail`
  on a verifier/sandbox PR). The cell carries the EXPECTED `network_mode` derived
  from the task config (`build_integration_review_pack._cell_network_config`), it
  is **not** passed to `bench`. The lane is scope-gated (the Q3 network lane,
  [`../integration-tiers.md`](../integration-tiers.md) §3.1), not always-on.
- The lane vehicle is the allowlist VARIANT
  `docs/examples/task-md/real-skillsbench/citation-check-network/`
  (`network_mode: allowlist`, `allowed_hosts: [eutils.ncbi.nlm.nih.gov,
  scholar.google.com, doi.org, api.crossref.org]`); validity is enforced upstream
  by `_validate_network_policy_fields` (`src/benchflow/task/config.py`).

What is **missing** vs the original ADR is the *runtime conformance* half: #802
does **not** observe that an allowlisted host is actually reachable and a
non-allowlisted host is actually **blocked** under the enforced mode. The original
ADR proposed porting that from `net/live_lane_test.py` via the real egress
sidecar — but `net/live_lane_test.py` and, critically, its dependency
`benchflow.sandbox._egress.build_egress_override` **do not exist in main** (the
`src/benchflow/sandbox/` tree has no `_egress` module). The runnable prober is
therefore **out of scope** for #802 and is **not** ported here.

This ADR aligns all network-posture naming to the **authoritative**
`src/benchflow/task/config.py` `NetworkMode` enum — **`no-network` / `allowlist`
/ `public`**. #799's `isolated` / `open` / wildcard naming is **not** introduced;
where the original text said `isolated` read `no-network`, and where it said
`open` read `public`.

## Decision

**Ships now in #802 (static, scope-gated).** Keep `V-NETWORK` as the deterministic
**static** assertion over the declared task config — default `no-network`,
`allowlist` requires non-empty `allowed_hosts`, bare `public` flagged/failed on a
verifier/sandbox PR — wired as a **scope-gated** lane (the Q3 network lane),
**not** always-on. This is the cheap identity-of-declaration guarantee.

**Deferred follow-up (runtime egress conformance lane), BLOCKED.** The runtime
conformance lane — assert via the egress sidecar that (1) `no-network` blocks all
egress, (2) `allowlist` permits only the listed hosts plus the resolved
model-provider host, (3) a disallowed-host attempt is denied — is deferred and
**blocked on `benchflow.sandbox._egress.build_egress_override` not existing in
main**. It cannot be built until that producer-side egress override surface lands.
The runnable prober from #799 (`net/live_lane_test.py`) is **deliberately not
ported**.

When unblocked, two things land together:

1. **`network_mode` result.json serialization** — serialize the *requested*
   `network_mode` (+ `allowed_hosts`) into the rollout contract so the lane can
   move from a static config assertion to an **observed** egress-policy check, and
   so `check_results` can reconcile recorded vs requested posture (config ↔
   result ↔ summary, mirroring `agent_idle_timeout`). This contract schema bump is
   **coordinated with ADR-0002's deferred `verifier_files_mutated` field as ONE
   schema bump** (both additive, defaulting; bump `result.json` + `GateResult`
   once).
2. **Trigger policy — attach to the existing scope rule, not always-on.** Run the
   conformance lane under the **existing verifier-rewards-judge scope rule**
   (the verifier / rewards / judge / anti-hack scope of the Default-config-rules),
   since egress enforcement is part of the same isolation boundary; do **not**
   make it an always-on per-PR lane. A **docker↔daytona parity** assertion stays
   in the nightly tier only, contingent on reliable Daytona credentials.

## Consequences

- (+) Today: a *declared* network-posture regression (empty allowlist, stray
  `allowed_hosts`, bare `public` on a verifier/sandbox PR) can no longer ship
  green — deterministically, with no Daytona credentials.
- (−) Today: `V-NETWORK` proves the **declaration** is hardened; it does **not**
  observe live egress. A run that records the right mode but leaks egress would
  still pass until the deferred conformance lane lands.
- (−) The conformance lane is **blocked** on `benchflow.sandbox._egress` landing
  in main; it reuses the proven `net/live_lane_test.py` logic but cannot run
  until that dependency exists.
- (−) When unblocked, it requires threading `network_mode`/`allowed_hosts`
  through `scenarios.run_eval` and the contract serialization (coordinated schema
  bump with ADR-0002).

## Alternatives considered

- **Port the runtime conformance prober now** (the original #799 always-on
  per-PR lane): impossible in #802 — `benchflow.sandbox._egress.build_egress_override`
  is absent from main, so the ported lane cannot import its dependency. Rejected
  as out of scope; captured as a deferred follow-up instead.
- **Make the deferred conformance lane always-on per-PR** (the original trigger):
  couples an expensive sidecar egress check to every PR. Rejected for #802 in favor
  of attaching it to the existing verifier-rewards-judge scope rule.
- **Identity check only** (recorded mode, no conformance — also deferred until
  serialization exists): cheap but a true egress regression with the right mode
  recorded still passes. Kept as the complement to the static check, not the gate.
