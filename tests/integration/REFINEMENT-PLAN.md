# Integration-test workflow hardening — implementation plan (ENG-265)

Implementation-ready companion to PRD **ENG-265**. The PRD owns the *what/why*; this
doc owns the *how* — concrete files, module interfaces (signatures encode the locked
decisions), acceptance criteria, and the test for each slice. Vocabulary is defined
in [`CONTEXT.md`](./CONTEXT.md); locked architectural decisions are in
[`../../docs/adr/`](../../docs/adr/) (ADR 0001 lane execution home, 0002 verifier-tamper
hash, 0003 network_mode conformance lane).

Paths are relative to the repo root. Slices are independently shippable and ordered so
each builds on the last; ship as separate PRs.

## Rollout-contract additions (shared across slices)

Two new fields enter the rollout contract (`result.json`) — additive, defaulting so
old artifacts still parse:

- `network_mode: str | None` and `allowed_hosts: list[str] | None` — the *requested*
  posture (identity), populated from the run config. (Slice 2)
- `verifier_files_mutated: bool | None` — producer-side tamper signal; `None` when the
  producer didn't record hashes. (Slice 4)

`GateResult` (the structure `agent_judge.py` emits via `--json`) gains
`verifier_files_mutated` and `network_mode` so the gate, the CI summary, and the skill
all speak the same shape.

## Lane ↔ marker convention (ADR 0001)

Each `release.yaml` lane id `L` maps to a pytest marker `lane_L` (hyphens → underscores)
on tests in `tests/integration/test_integration_suite.py`. `run_suite.py --run-lane L`
runs `pytest -m lane_L`. `load_suite` fails if any release-blocker lane has no marker or
any `lane_*` marker has no lane.

---

## Slice 1 — Foundations: glossary, missing linchpin files, lane resolver

**Goal:** make the suite runnable from the repo and give lanes a real execution home.

- **Create** `tests/integration/CONTEXT.md` (this bundle) — the area glossary.
- **Create** `tests/integration/test_integration_suite.py` — the pytest lane home that
  imports `scenarios.py`; one test (or class) per release-blocker lane, marked
  `@pytest.mark.lane_<id>`. Initially wraps the existing executable scenarios; new
  lanes are added by later slices.
- **Create** `tests/integration/select_integration_provider.py` — the CI provider
  selector. Interface:
  - `select_integration_provider(env: Mapping[str, str]) -> ProviderChoice` where
    `ProviderChoice = {agent, model, judge_model, sandbox}`.
  - **Contract:** raise `NoProviderAvailable` (non-zero exit, clear message listing
    which creds were probed) when zero providers are credentialed — never exit 0 with
    no provider. Add `assert_judge_available(choice, env)` preflight that fails if the
    judge model's provider key is absent.
- **Modify** `tests/integration/run_suite.py`:
  - add `--run-lane <lane-id>` → resolve to `pytest -m lane_<id>` and exec pytest;
  - in `load_suite`, add `validate_lane_markers(manifest, collected_markers)` →
    `list[str]` of mismatches; fail the dry-run gate on any mismatch.

**Acceptance:** `run_suite.py --run-lane <existing-lane>` runs the corresponding
pytest and returns its exit code; `--dry-run --fail-on-todo` additionally fails if a
release-blocker lane lacks a marker; `select_integration_provider` hard-fails on an
empty cred set.

**Tests** (`test_integration_suite.py` + a new `tests/test_run_suite.py` if absent):
`select_integration_provider` returns the expected choice for a representative env and
raises on empty env (deep module, pure given an injected env mapping);
`validate_lane_markers` reports a lane-without-marker and a marker-without-lane.

**Prior art:** existing `run_suite.py` manifest validation; `scenarios.py` import style.

---

## Slice 2 — network_mode: identity + conformance + lane + axis + CI per-PR (ADR 0003)

**Goal:** a network-posture regression fails CI on every PR.

- **Modify** `tests/integration/scenarios.py`:
  - `run_eval(..., network_mode: str | None = None, allowed_hosts: list[str] | None = None)`
    — thread the posture into the real `bench eval run` invocation.
  - add `assert_egress_conformance(rollout_dir: Path, *, allowed: list[str], blocked: list[str]) -> None`
    — ported from `net/live_lane_test.py`: asserts each `allowed` host reachable and
    each `blocked` host denied, observed via the egress sidecar. Raises with the
    offending host on failure.
- **Modify** `tests/integration/suites/release.yaml`:
  - add axis `network_modes: [isolated, allowlist, open]`;
  - add lane `network-mode-enforcement` (release_blocker, docker-only, per-PR) with
    acceptance items annotated `enforced:` (machine) vs `aspirational:`.
- **Add** test(s) in `test_integration_suite.py` marked `lane_network_mode_enforcement`
  that drive `run_eval(network_mode="allowlist", allowed_hosts=[...])` then call
  `assert_egress_conformance`, plus an `isolated`/`no-network` blocks-all case.
- **Modify** `tests/integration/check_results.py`: add `network_mode` (+ `allowed_hosts`)
  to the `EXPECTED` identity set and the config↔result↔summary reconciliation, mirroring
  `agent_idle_timeout`.
- **Modify** `tests/integration/configs/*.yaml`: add `network_mode`; (key standardization
  happens in Slice 3).
- **Modify** `.github/workflows/integration-eval.yml`: the per-PR job runs
  `run_suite.py --run-lane network-mode-enforcement` as a **hard-fail** step (never
  `continue-on-error`), docker-only (no Daytona secret needed).

**Acceptance:** with an allowlist task, a deliberately-blocked host causes the lane to
fail; the correct allow/deny passes; `check_results` flags a result whose recorded
`network_mode` ≠ requested. Per-PR CI fails on any of these.

**Tests:** `assert_egress_conformance` against a fixture rollout (allowed-reachable /
blocked-denied) — reuse `net/live_lane_test.py` / `net/test_network_modes.py` as the
model; `check_results` identity reconciliation unit test for a mismatched mode.

**Prior art:** `net/live_lane_test.py`, `net/test_network_modes.py`;
`check_results.py` `agent_idle_timeout` reconciliation.

---

## Slice 3 — Single source of truth for tasks/models/posture

**Goal:** configs and the manifest cannot drift.

- **Create** (in `run_suite.py` or a small `tests/integration/_consistency.py`):
  `validate_configs_against_manifest(manifest, configs) -> list[Drift]` (pure) — asserts
  every config's `agent ∈ axes.agents`, `model == axes.models.default`,
  `include == referenced task_set`, and `network_mode` matches. Wire into `load_suite`
  and a CI dry-run step.
- **Modify** `tests/integration/run.sh`: delete/derive the dead `SELECTED_TASKS` array
  (the YAML include drives execution); the task list comes from the manifest only.
- **Modify** `tests/integration/configs/*.yaml`: standardize the sandbox key on
  `sandbox` (rename the divergent `environment` key).

**Acceptance:** `load_suite` fails when a config's model/agent/task-set/network_mode
diverges from the manifest; `run.sh` no longer carries a second task list.

**Tests:** `validate_configs_against_manifest` returns the expected drift list for a
seeded-divergent config and `[]` for an aligned set (deep module, pure).

**Out of scope:** full config codegen-from-manifest (validation only this round).

---

## Slice 4 — Producer-side verifier-tamper hash (ADR 0002)

**Goal:** deterministic, language-agnostic reward-hacking detection.

- **Modify** the real benchflow producer (`src/benchflow/sandbox/` + the verifier path):
  record a hash of the score-defining file set before and after the agent phase; write
  `verifier_files_mutated: bool` into `result.json`. The score-defining set is declared
  per verifier/task.
  - Deep module: `verifier_files_mutated(before: Mapping[str, str], after: Mapping[str, str]) -> bool`
    (pure given two path→hash maps) — unit-test this in `src/`-side tests.
- **Modify** `tests/integration/agent_judge.py`: consume `verifier_files_mutated` — when
  `True`, the realness gate FAILs regardless of the judge; the judge prompt is told the
  files were mutated. Keep the trajectory regex as an advisory backstop only.
- **Modify** `GateResult`/`--json` to include `verifier_files_mutated`.

**Acceptance:** a rollout with `verifier_files_mutated=True` fails the gate even when
the judge would pass and the regex finds nothing; a clean rollout is unaffected.

**Tests:** `verifier_files_mutated` pure-function table (mutated / added / removed /
unchanged); `gate_rollout` fails-closed when the boolean is `True`; the regex remains
advisory (its absence no longer clears a mutated rollout).

**Prior art:** `check_results.py` file-hash recomputation (provenance audit) is the
model for hashing; `agent_judge.py` `_scan_verifier_tamper` is the regex it backs.

---

## Slice 5 — Power-aware parity + hermetic evidence gates

**Goal:** parity tolerance scales with sample size; evidence gates don't depend on live
network on the gate path.

- **Modify** `tests/integration/check_skillsbench_harbor_parity.py`:
  - `parity_verdict(baseline: TrialStats, candidate: TrialStats, *, min_trials: int) -> ParityVerdict`
    (pure) — require `min_trials`, compare via Wilson confidence-interval overlap for
    outcome rate, and a documented per-task delta band; replace the fixed `0.25` +
    observed-set membership.
- **Modify** `check_adapter_evidence.py` / `check_hosted_env_evidence.py` /
  `check_trace_to_task_evidence.py`: split generation from verification — a separate
  (nightly/opt-in) job produces attested evidence artifacts (with provenance/hashes per
  `check_results`' model); the release-gate checkers validate those artifacts offline.
  Any live regeneration is behind an opt-in flag with a subprocess timeout + retries.
  Move hardcoded remediation strings (e.g. the HILBench `hf://` string) to config.

**Acceptance:** parity passes/fails based on CI overlap and respects `min_trials`
(too-few-trials → inconclusive, not PASS); the three evidence gates run offline against
artifacts on the default gate path.

**Tests:** `parity_verdict` table (wide CI overlap → pass, disjoint → fail, < min_trials
→ inconclusive); an evidence checker validates a good artifact and rejects a
provenance-mismatched one (deep modules, pure given artifacts).

---

## Slice 6 — Reviewer skill cross-link + fixtures + fixture-factory CLI

**Goal:** the experiment-review skill is runnable, gate-first, and structurally scored.

- **Modify** `.agents/skills/benchflow-experiment-review/SKILL.md`: in "Completed Trial
  Review", run `agent_judge.py` FIRST and treat realness/tamper as authoritative;
  reserve prose for judgments the gate can't make (coverage matrix, capability
  attribution, publish verdict). Emit a **structured JSON verdict mirroring
  `GateResult`** so `evals.json` scores deterministically (not substring match).
- **Commit** the skill's referenced files that are absent on disk: `scripts/*`,
  `references/*.md`, and the `evals/files/*` fixtures — or convert any genuinely
  external reference to an explicit `canonical copy lives in benchflow repo at <path>`
  pointer. (Per ENG-265 "Full" decision: author them.)
- **Add** a **network-leak eval fixture** (run config declares `allowlist`/`off` but the
  trajectory shows egress to a non-allowlisted host) to `evals.json` so the SOP's evals
  catch a network_mode regression.
- **Modify** `tests/integration/deepagents_harness.py`: add a thin argparse CLI
  (`--instruction --verify-cmd --extra-system --rollout-dir`) so judge-hardening rounds
  are reproducible; add a network-leak fixture-generation mode. Keep `--network none` as
  a documented fixture, not the system under test.

**Acceptance:** `evals.json` runs against on-disk fixtures and scores via the structured
verdict; the new network-leak fixture is judged FAIL; the harness CLI reproduces a
named rollout.

**Tests:** the skill evals harness over the committed fixtures (incl. network-leak);
the harness CLI produces a rollout matching the contract.

---

## Slice 7 — CI observability, tiering, and budget

**Goal:** gate failures are legible and never mis-grade a stale rollout; cost is bounded.

- **Modify** `.github/workflows/integration-eval.yml`:
  - **Tiering:** per-PR job = cheap, docker-only, hard-fail (one task×agent×model
    judge+checks + the `network-mode-enforcement` lane). Nightly (`schedule`) + manual
    (`workflow_dispatch`) job = the full agent×task×sandbox matrix (advisory; promotable
    to release-blocker on release tags).
  - **Observability:** pipe `agent_judge.py --json` (realness issues, verdict reason,
    flagged verifier actions, `verifier_files_mutated`) into `$GITHUB_STEP_SUMMARY`; on
    PR failure, post a concise PR comment.
  - **Determinism:** `--jobs-dir jobs/integration-eval-${{ github.run_id }}`; the gate
    step FAILs (not warns) when `>1` `result.json` is found (also fix
    `agent_judge._find_rollout_dir`'s newest-only mtime selection to error on ambiguity).
  - **Budget:** add a token/$ ceiling assertion (fail if the run exceeds the budget),
    beyond timeouts.

**Acceptance:** a failing per-PR gate shows the verdict in the step summary + a PR
comment; two rollouts in the jobs dir fail the step; a run over budget fails.

**Tests:** `agent_judge.py --json` shape (used by the summary step); the
multiple-rollout case errors rather than picking newest.

---

## Suggested PR sequence

1 → 2 → 3 → 4 → 5 → 6 → 7. Slices 1–2 deliver the headline security gate; 3–7 are the
broader hardening. Each slice is green-on-its-own (full suite + ruff + ty) and updates
`CONTEXT.md`/ADRs if a new term or decision appears.
