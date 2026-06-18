# Integration Review Rubric

The success rubric for the Benchflow integration-review system. This is the
authoritative description of **what "mergeable" means** for a set of integration
rollouts produced by a Benchflow code change. It is written to be read alongside,
and to never drift from, the machine-readable single source of truth in
`tests/integration/rubric_checks.py` (`RUBRIC_GATES`).

> **Terminology rename.** The user-facing verdict is **`mergeable`** /
> **`mergeable with quarantines`** / **`not mergeable`** — renames of the internal
> grader labels `publishable` / `publishable-with-quarantines` /
> `not-publishable`. The labels below use the user-facing names; the grader source
> (`build_integration_review_pack.py`) still emits the internal strings.

Companion documents:

- [`integration-tiers.md`](./integration-tiers.md) — the L0–L3 level ladder, the
  seven-set taxonomy + Default-config-rules, the matrix cell schema, the Q3
  network lane, the Codex reviewer contract, the before/after baseline model, and
  the security / admin-setup model.
- `.agents/skills/benchflow-experiment-review/SKILL.md` — the human-procedure
  skill that this rubric mechanizes.
- `.agents/skills/benchflow-experiment-review/references/verifier-hardening-checklist.md`
  and `.../reward-hacking-patterns.md` — the prevention/detection background.

## 1. Purpose

A Benchflow code change is only safe to use for new production experiments if
the rollouts it produces are **real**, **healthy**, **uncontaminated**, and
**fully covered** against the planned scope. This rubric:

1. Defines the healthy-outcome taxonomy (what a "good" rollout looks like).
2. Enumerates the rubric gates (`RUBRIC_GATES`) — verbatim IDs so the docs
   cannot diverge from the code — and, for each, says exactly what artifact
   fields/markers it reads and **honestly** whether the gate is fully
   deterministic, a quarantine signal, a codex-judge call, or a residual risk
   we cannot mechanically prove from current artifacts.
3. Defines the verdict ladder that `build_integration_review_pack.py` computes,
   and the `worst(deterministic, codex)` final-verdict rule.
4. Defines the capability-attribution decision (model-capability vs
   experiment-fidelity).
5. Lists the known coverage gaps and how each is mitigated.
6. Gives the operational checklists (per-slot, skill-loading, reward-hacking),
   the success-rubric table, the merge rule, and the review-pack report ordering,
   each mapped to its enforcer (deterministic gate / codex / residual).

The rubric is consumed by two CLIs:

- `python tests/integration/rubric_checks.py ROLLOUT_DIR [--run-config F] [--json]`
  — runs the deterministic + quarantine gates over a single rollout, auto
  detecting flat-fixture vs production schema; exit `0` if no deterministic
  reject fires, else `1`.
- `python .github/scripts/build_integration_review_pack.py JOBS_ROOT --matrix matrix.json [--out review-pack/] [--json]`
  — classifies every planned slot, grades each rollout via `rubric_checks`,
  computes the verdict, and writes the `review-pack/` (layout in
  `integration-tiers.md` §7); exit `0` if the verdict is `mergeable` /
  `mergeable with quarantines`, non-zero for `not mergeable`. The matrix is
  produced by the planner `.github/scripts/integration_matrix.py`.

## 2. Healthy-Outcome Taxonomy

A rollout's *outcome* is one of three **healthy** states or it is **unhealthy**.
Only the three healthy states may be counted toward coverage; an unhealthy
rollout is a blocker unless it is explicitly quarantined as an infra/context
failure (see §5 Capability Attribution).

| Outcome | Predicate (over the normalized evidence shape) |
|---|---|
| `pass` | Agent ran to completion; verifier produced a valid score; `reward == 1.0` (or task-defined success threshold). Realness predicate holds. |
| `fail` | Agent ran to completion; verifier produced a valid score; `reward < 1.0`. Realness predicate holds. **AND** the failure is attributed to model capability, not experiment fidelity (see §5). |
| `normal_timeout` | Agent genuinely ran and timed out, but still emitted a complete trajectory plus reward/scoring metadata and token/timing telemetry. Attributed to model capability, not infra. |

**Realness predicate** (the spine of the taxonomy, enforced by `R-REAL`):
`n_tool_calls > 0` **AND** `total_tokens > 0` **AND** `reward != null`. A
rollout that fails the realness predicate is not a healthy `fail`/`timeout`; it
is an empty/never-launched/transport-failed rollout and is unhealthy.

**Unhealthy** (not a healthy outcome — blocker unless quarantined):
missing/truncated/unparsable trajectory; empty transcript or agent never
launched; missing token/timing/tool-usage telemetry for newly generated data;
agent lacked required task inputs / skills / keys / assets / runtime / compute;
verifier ran early or leaked into the agent phase; reward inconsistent with
visible task state; path/root mismatch; provider/Daytona/Docker/filesystem
error that prevented a fair attempt. These are the SKILL.md "reject or
quarantine" conditions, mechanized as the gates in §3.

## 3. RUBRIC_GATES

The table below is **verbatim** from the shared contract and mirrors
`tests/integration/rubric_checks.py:RUBRIC_GATES`. Each gate has a stable `id`,
a `title`, an `enforcement` class in
`{deterministic, quarantine, codex, residual}`, and a `skill_ref` pointing at
the code or skill artifact that implements it. **Do not edit an ID here without
editing it in `rubric_checks.py` in the same change** — the IDs are the contract
surface that the review pack, the docs, and the codex agent all key on.

| id | title | enforcement | skill reference |
|---|---|---|---|
| `R-REAL` | realness (`n_tool_calls>0` & `total_tokens>0` & `reward!=null`) | deterministic | `agent_judge.realness_issues` |
| `R-OUTCOME` | outcome in `{pass,fail,normal_timeout}` with a valid score | deterministic | `rubric_checks` (outcome predicate, §2) |
| `R-ARTIFACT` | trajectory present+parseable; ATIF/ADP well-formed; no secret leak | deterministic | `scenarios.atif_issues` / `scenarios.adp_issues` / `scenarios.secret_leak_issues` |
| `R-TELEMETRY` | token usage + timing present for newly generated data | deterministic | `rubric_checks` (telemetry presence over normalized evidence) |
| `S-WITHSKILL` | with-skill ⇒ `task_skills_loading == 1` | quarantine | `extract_harness_skills.py` (unknown / `catalog_not_serialized` ⇒ quarantine→codex) |
| `S-NOSKILL` | no-skill ⇒ `task_skills_loading == 0` **AND** no leakage markers | deterministic | leakage marker scan (`SKILL.md`, `.codex/skills`, `.agents/skills`, `.claude/skills`, `invoke_skill`, `activate_skill`, `ToolSearch`) |
| `V-TAMPER` | no agent write/delete/chmod of score-defining files | deterministic | `agent_judge._scan_verifier_tamper` (full trajectory) |
| `V-EMPTYPATCH` | empty/no-op patch FAILS (`<1.0`) while golden patch passes (`1.0`) | deterministic | null-agent lane + oracle determinism |
| `V-LIFECYCLE` | verifier starts only after agent exit/timeout | residual/codex | no lifecycle timestamps in artifacts → not mechanically observable |
| `V-ENVHARDEN` | git scrubbed past base; no answer/hidden-test/scorer files visible | residual/codex | sandbox-fs property; best-effort env-probe else residual |
| `V-REWARDHACK` | reward consistent with visible task state (no grader-gaming) | codex | `agent_judge` LLM judge |
| `P-REWARD` | same oracle task scores identically docker vs daytona | deterministic | `test_sandbox_parity` (oracle, reward-only) |
| `P-SCHEMA` | token/timing/schema parity on a REAL-MODEL paired cell | deterministic | `rubric_checks.parity_schema_diff` (requires a model-bearing paired docker+daytona cell, not oracle) |
| `C-ATTRIB` | fail/timeout = model-capability (not experiment-fidelity) | deterministic-first then codex residual | `check_results.INFRA_ERROR_CATEGORIES` + `verifier_error` + required-env presence, then codex residual |
| `P-PROV` | source repo/path/ref/hash provenance intact | deterministic | `check_results._source_hash_truth_issues` / `_source_git_truth_issues` |
| `P-PATHS` | config root / sandbox cwd / trajectory paths / result paths agree | deterministic-where-fields-exist else residual | `check_results` path-truth checks where fields exist, else residual |
| `X-SLOTS` | every planned cell present, unique, fresh (head sha) | deterministic | `build_integration_review_pack` slot classification: missing/duplicate/stale/healthy/unhealthy |
| `V-NETWORK` | default no-network; allowlist needs non-empty `allowed_hosts`; `public` flagged (fail on verifier/sandbox PR) | deterministic | `rubric_checks.network_hardening` (STATIC per-task config, Q3) |
| `V-NETWORK-CONFORM` *(deferred)* | run *observed* the enforced egress: allowlisted host reachable, non-allowlisted host **blocked** | residual *(not yet mechanical)* | egress-sidecar conformance lane — **deferred, blocked** on `benchflow.sandbox._egress` (ADR-0003) |

### 3.1 Per-gate detail (what it checks, what it reads, honest enforcement)

For each gate: the *check*, the *artifact fields/markers* it reads, and an
**honest** statement of how much of the gate is actually mechanical.

The artifacts come in two schemas, normalized by a schema adapter before any
gate runs (see §3.2):

- **Production rollout dir:** `result.json{task_name, agent, rewards{reward},
  n_tool_calls, agent_result{total_tokens}, error, verifier_error}`,
  `rewards.jsonl`, `trajectory/acp_trajectory.jsonl`, `trainer/atif.json`,
  `trainer/adp.jsonl`.
- **Flat skill-eval fixture:** `result.json{status, reward,
  verifier_started_after_agent, timing{started_at, ended_at,
  duration_seconds}, token_usage{input_tokens, output_tokens}, tool_usage}`
  with sibling `run_config.json{task_id, harness, model, skill_mode, trial_id,
  sandbox, source_ref, timeout_seconds, task_skills}`.

The reused production enforcers (`agent_judge`, `check_results`, `scenarios`)
read **only** the production schema; the adapter is what lets the same gates run
over a flat fixture.

---

**`R-REAL` — realness (deterministic).**
- Checks: the realness predicate (`n_tool_calls > 0` AND `total_tokens > 0` AND
  `reward != null`).
- Reads (production): `result.json.n_tool_calls`, `agent_result.total_tokens`,
  `rewards.reward` / `rewards.jsonl`. (Flat fixture: `tool_usage`,
  `token_usage.{input,output}_tokens`, `reward`.)
- Honesty: **fully deterministic.** Implemented by `agent_judge.realness_issues`
  (empty list == real). This is the load-bearing anti-fabrication gate.

**`R-OUTCOME` — valid outcome (deterministic).**
- Checks: the rollout resolves to exactly one of `{pass, fail, normal_timeout}`
  with a valid numeric score; an unparseable/absent status or a null reward on a
  non-timeout fails.
- Reads: production `rewards.reward` + `error`/`verifier_error`; flat `status` +
  `reward`. The §2 predicates are applied.
- Honesty: **fully deterministic** for the outcome *shape*. Note: deciding
  whether a `fail`/`timeout` is *legitimately* model-capability (not infra) is
  `C-ATTRIB`, not this gate.

**`R-ARTIFACT` — artifacts well-formed, no secret leak (deterministic).**
- Checks: a trajectory file exists and parses end-to-end; ATIF (`trainer/atif.json`)
  and ADP (`trainer/adp.jsonl`) are well-formed; no secret/credential leak in the
  artifact tree.
- Reads: `trajectory/acp_trajectory.jsonl` (parseability), `trainer/atif.json`
  via `scenarios.atif_issues`, `trainer/adp.jsonl` via `scenarios.adp_issues`,
  and the whole root via `scenarios.secret_leak_issues`.
- Honesty: **fully deterministic.**

**`R-TELEMETRY` — token+timing present (deterministic).**
- Checks: newly generated data carries token usage and timing.
- Reads (production): `agent_result.total_tokens`, plus timing fields where
  present; (flat) `token_usage.{input,output}_tokens` and
  `timing.{started_at, ended_at, duration_seconds}`.
- Honesty: **deterministic for presence.** It proves the fields exist and are
  populated; it does not (and cannot) prove the *values* are accurate to the
  provider's true accounting.

**`S-WITHSKILL` — with-skill skills loaded (quarantine).**
- Checks: for a `with-skill` cell, `task_skills_loading == 1` (every expected
  task-specific skill recovered in the startup catalog).
- Reads: `trajectory/llm_trajectory.jsonl` via `extract_harness_skills.py`
  (falls back to sibling `acp_trajectory.jsonl`); expected skill names from
  `run_config.json.task_skills`.
- Honesty: **quarantine, NOT a hard deterministic reject.** When the extractor
  returns `unknown`, `skill_count: 0`, `manual_review_required: true`, or
  `catalog_not_serialized`, that is **not** proof skills were absent — it means
  the harness did not serialize a recoverable catalog (notably `pi-acp`, see
  §6). The slot is **quarantined and routed to codex**, never auto-rejected. A
  clean `task_skills_loading == 1` is a deterministic pass.

**`S-NOSKILL` — no-skill clean (deterministic).**
- Checks: for a `no-skill` cell, `task_skills_loading == 0` AND no skill-leakage
  markers anywhere in the trajectory.
- Reads: the extractor's `task_skills_loading` plus a full-trajectory scan for
  the markers `SKILL.md`, `.codex/skills`, `.agents/skills`, `.claude/skills`,
  `invoke_skill`, `activate_skill`, `ToolSearch` (the marker tuple in
  `rubric_checks._NOSKILL_LEAK_MARKERS`). The `.claude/skills` marker was added
  by the prior audit — Claude Code loads skills from `.claude/skills`, so its
  appearance in a no-skill trajectory is leakage just like `.codex/skills` and
  `.agents/skills`.
- Honesty: **deterministic.** Marker presence in a no-skill run is a hard reject.
  (This direction is safe to enforce deterministically: a leakage marker is
  positive evidence of contamination, whereas an *absent* catalog in the
  with-skill direction is ambiguous — hence the asymmetry between `S-NOSKILL`
  and `S-WITHSKILL`.)

**`V-TAMPER` — no verifier/score-file tampering (deterministic).**
- Checks: the agent did not write, delete, or chmod any score-defining file
  (tests, verifier code, scoring code, result/reward files) during its phase.
- Reads: the full trajectory via `agent_judge._scan_verifier_tamper`.
- Honesty: **deterministic** for the *observable* write/delete/chmod vectors in
  the trajectory. It does **not** catch in-process monkey-patching or `__eq__`
  / operator overloading that never surfaces as a file op — those residual
  vectors are flagged by the codex judge under `V-REWARDHACK` (see §6 and the
  hardening checklist's note that test-reset/hidden-tests do not close them).

**`V-EMPTYPATCH` — empty patch fails, golden passes (deterministic).**
- Checks: the null/empty-patch lane scores `< 1.0` while the golden patch scores
  exactly `1.0`, deterministically across the oracle.
- Reads: the null-agent lane rollout reward and the oracle-determinism lane
  reward (`reward == 1.0`).
- Honesty: **deterministic.** This is the build-time leakage probe: a passing
  empty patch means the reward is reachable without solving (an answer leak or a
  grader bug). Requires the planner to have emitted a `null_patch` and an
  `oracle_determinism` cell.

**`V-LIFECYCLE` — verifier starts after agent exit (residual/codex).**
- Checks: the verifier process started only after the agent exited or timed out.
- Reads: ideally lifecycle timestamps — but **production artifacts do not carry a
  verifier-start timestamp relative to agent-exit.** The flat fixture has a
  best-effort boolean `verifier_started_after_agent`, which we surface when
  present.
- Honesty: **NOT deterministic from production artifacts.** When the flat-fixture
  boolean is present and `true`, we record it; otherwise this is a **residual**
  property carried by the harness's own lifecycle code and, where suspicious,
  raised to the **codex** judge. We do not claim mechanical coverage here.

**`V-ENVHARDEN` — git scrubbed, no answer files visible (residual/codex).**
- Checks: git history is scrubbed past the base commit; no answer / reference /
  hidden-test / scorer files are present in agent-readable paths.
- Reads: this is a **sandbox-filesystem property** that is not represented in the
  rollout artifacts. Where an env-probe rollout is available we read its probe
  output (best-effort); otherwise nothing.
- Honesty: **NOT deterministic from rollout artifacts.** Best-effort env-probe
  when one is run; otherwise a **residual** note, with the **codex** judge asked
  to flag any trajectory evidence of git-time-travel or answer-file reads. We do
  not claim mechanical coverage here.

**`V-REWARDHACK` — reward consistent with task state (codex).**
- Checks: the reward is consistent with the visible task state — no grader
  gaming, monkey-patching, operator-overload tricks, exit-code manipulation, or
  test-input hardcoding that `V-TAMPER`'s file-op scan cannot see.
- Reads: the full trajectory + reward, evaluated by the `agent_judge` LLM judge
  (model = `BENCHFLOW_JUDGE_MODEL`; do **not** hardcode a provider/key).
- Honesty: **codex/LLM-judge, not deterministic.** Fail-closed: if the judge
  cannot run (no judge model/key), the gate does not silently pass. This is the
  catch-all for the reward-hacking patterns the deterministic scans cannot
  mechanically prove.

**`P-REWARD` — oracle reward parity docker vs daytona (deterministic).**
- Checks: the same oracle task scores identically on docker and daytona.
- Reads: the reward of the paired oracle cells (reward-only), via
  `test_sandbox_parity`.
- Honesty: **deterministic** (oracle is reward-deterministic by construction).

**`P-SCHEMA` — token/timing/schema parity on a real-model paired cell (deterministic).**
- Checks: token usage, timing, and artifact schema match across a docker+daytona
  pair on a **model-bearing** cell (not oracle — oracle has no model telemetry to
  compare).
- Reads: both paired rollouts' schema + telemetry, diffed by
  `rubric_checks.parity_schema_diff`.
- Honesty: **deterministic on schema/field-shape**, but it **requires** the
  planner to have emitted a real-model paired docker+daytona cell. With only an
  oracle pair, `P-SCHEMA` is not satisfiable and the slot is reported missing.

**`C-ATTRIB` — failure attribution (deterministic-first, then codex residual).**
- Checks: a `fail`/`normal_timeout` is due to model capability, not
  experiment-fidelity (missing keys/skills/assets/deps/compute, provider outage,
  sandbox instability, path mismatch).
- Reads: `check_results.INFRA_ERROR_CATEGORIES` classification, `verifier_error`,
  and required-env presence (were the credentials/skills/assets the task needs
  actually present?).
- Honesty: **deterministic-first**, then **codex residual.** The infra-error
  categories and env-presence checks deterministically catch the *known* infra
  signatures and route them to quarantine; the residual judgment ("this looks
  like a genuine model failure, not a hidden environment gap") is handed to
  codex. See §5.

**`P-PROV` — provenance intact (deterministic).**
- Checks: source repo / path / ref / hash provenance is intact and truthful.
- Reads: via `check_results._source_hash_truth_issues` and
  `_source_git_truth_issues` (and `run_config.source_ref` on the flat fixture).
- Honesty: **deterministic.**

**`P-PATHS` — path/root agreement (deterministic-where-fields-exist else residual).**
- Checks: config root, sandbox cwd, trajectory-reported paths, and result paths
  all agree (the historical root-path regression class).
- Reads: the corresponding path fields via `check_results` path-truth checks.
- Honesty: **deterministic where the fields exist**; where a schema variant does
  not carry a given path field, that sub-check degrades to **residual** rather
  than a false pass.

**`X-SLOTS` — coverage completeness (deterministic).**
- Checks: every planned cell from the planner matrix is present exactly once and
  is fresh (built at the head SHA).
- Reads: the matrix `matrix[]` (from `integration_matrix.py`) vs the discovered
  rollouts under `JOBS_ROOT`, classified by `build_integration_review_pack` into
  `matrix_observed.json` as `missing / duplicate / stale / healthy / unhealthy`.
- Honesty: **deterministic.** A missing required slot or a stale (wrong-SHA) slot
  is a blocker.

**Identity vs conformance** (two distinct network-posture guarantees; both named
in the `no-network` / `allowlist` / `public` vocabulary of the authoritative
`NetworkMode` enum in `src/benchflow/task/config.py`):
- **Identity / declaration check** — the *declared* (and, once serialized, the
  *recorded*) posture is hardened and matches what was requested. Cheap; catches
  "wrong posture declared/requested". This is what `V-NETWORK` enforces today,
  statically over the task config.
- **Conformance check** — the run *actually* observed the right egress: an
  allowlisted host is reachable AND a non-allowlisted host is **blocked** under
  the enforced mode (the real security guarantee). This is `V-NETWORK-CONFORM`,
  **deferred** (below).

**`V-NETWORK` — network policy hardened (deterministic, STATIC per-task config —
the IDENTITY/declaration half).**
- Checks: the task's declared network policy is hardened. The safe default is
  `no-network`. Network access is acceptable **only** as `allowlist` with a
  **non-empty** `allowed_hosts`. A bare `public` mode is always flagged; on a PR
  that touches the verifier or the sandbox/lockdown surface, `public` is a hard
  `fail` (it controls the isolation boundary).
- Reads: the task config's `network_mode` + `allowed_hosts`, via
  `rubric_checks.network_hardening(task_config, verifier_or_sandbox_pr=...)`.
  **This is a STATIC check over the task config, not over a rollout** — there is
  **no `bench eval run --network` flag** and `network_mode` is **never serialized
  into a rollout artifact** (Q3). The grader maps the cell's EXPECTED
  `network_mode` (`default-off` | `allowlist`) onto the benchflow `NetworkMode`
  literal in `build_integration_review_pack._cell_network_config`, and reads
  `allowed_hosts` from the cell.
- Outcomes: `pass` for a hardened config (`no-network`, or `allowlist` with
  non-empty hosts); `fail` for an unsafe one (empty allowlist, `allowed_hosts`
  without `allowlist`, unknown mode, or `public` on a verifier/sandbox PR);
  `quarantine` for `public` on an unrelated PR (needs human sign-off); `na` when
  no policy is declared (runtime default `no-network` applies).
- Vehicle: the network lane's allowlist VARIANT,
  `docs/examples/task-md/real-skillsbench/citation-check-network/`
  (`network_mode: allowlist`, `allowed_hosts: [eutils.ncbi.nlm.nih.gov,
  scholar.google.com, doi.org, api.crossref.org]`). Validity is enforced upstream
  by `_validate_network_policy_fields` (`src/benchflow/task/config.py`).
- Honesty: **deterministic on the declared policy.** It proves the *declaration*
  is hardened; it does **not** observe live egress (that is the deferred
  conformance half, `V-NETWORK-CONFORM`, below; see `integration-tiers.md` §11).

**`V-NETWORK-CONFORM` — egress conformance observed (DEFERRED, blocked).**
- Checks (when built): the run *actually* observed the enforced egress via the
  egress sidecar — `no-network` blocks all egress; `allowlist` permits only the
  listed hosts **plus** the resolved model-provider host; a disallowed host is
  **denied** with a clear signal. This is the real security guarantee that
  `V-NETWORK`'s static declaration check cannot provide.
- Status: **deferred and blocked.** It is blocked on
  `benchflow.sandbox._egress.build_egress_override` not existing in main, so the
  egress sidecar the lane would assert against is unavailable; #799's runnable
  prober (`net/live_lane_test.py`) is **deliberately not ported**. It also depends
  on the deferred `network_mode` result.json serialization so the observed posture
  can be reconciled. See ADR-0003
  ([`adr/0003-network-mode-conformance-lane.md`](./adr/0003-network-mode-conformance-lane.md))
  and `integration-tiers.md` §11.
- Recommended trigger (when unblocked): attach to the **existing
  verifier-rewards-judge scope rule**, NOT always-on; docker↔daytona parity stays
  nightly-only.
- Honesty: **not mechanically observable today** — recorded as a **residual** until
  the `_egress` surface lands.

### 3.2 Schema adapter (why both fixture shapes grade identically)

There are **two** artifact schemas (production rollout dir vs flat skill-eval
fixture, fields enumerated in §3.1). A schema adapter normalizes **both** into a
single evidence shape before any gate runs, and `rubric_checks.py` auto-detects
which schema it is looking at via a field sniff (`--run-config` can supply the
sibling `run_config.json` for the flat fixture). The reused production enforcers
(`agent_judge`, `check_results`, `scenarios`) read only the production schema, so
the adapter is the single place that maps flat-fixture fields onto production
field names. This keeps one set of gate implementations valid over both shapes.

## 4. Verdict Ladder

`build_integration_review_pack.py` computes exactly one **deterministic** verdict
from the per-slot gate results. This deterministic verdict is the **REQUIRED,
fail-closed gate** (Q2):

- **not mergeable** — ANY of:
  - a deterministic reject fires on a slot counted as **healthy**, OR
  - a required planned slot is **missing**, OR
  - a **codex FAIL on a non-quarantinable gate** (e.g. `V-REWARDHACK` confirmed,
    or `C-ATTRIB` residual confirmed as experiment-fidelity).
- **mergeable with quarantines** — only **quarantine/residual** items remain
  unresolved (e.g. `S-WITHSKILL` `unknown`/`catalog_not_serialized`,
  `V-LIFECYCLE`, `V-ENVHARDEN`, or explicitly-quarantined infra-failed slots)
  **AND** all deterministic gates are green.
- **mergeable** — all gates green, full coverage, **zero** quarantines.

The exit code is `0` for `mergeable` and `mergeable with quarantines`, and
non-zero for `not mergeable`.

### 4.1 Final verdict = worst(deterministic, codex)

The deterministic verdict above is the **floor and the gate**. The **Codex**
before/after-equivalence reviewer (`integration-tiers.md` §5.2) is a
**strong-vetoing, advisory-stricter-only** signal: it can downgrade `mergeable` →
`not mergeable`, but it can **never upgrade** a deterministic `not mergeable`.

```
Final verdict = worst( deterministic_grader , codex_reviewer )
```

A dead/absent Codex key **at L3** ⇒ `not mergeable` (fail-closed). L1/L2 do **not**
require Codex.

## 5. Capability Attribution (model-capability vs experiment-fidelity)

A `fail` or `normal_timeout` may only be counted as a healthy outcome if it is a
**model-capability** failure. The decision (gate `C-ATTRIB`):

- **Model-capability failure (healthy):** required task resources were present;
  credentials and skills were available when needed; the sandbox had sufficient
  compute and runtime dependencies; the agent made a real attempt (realness
  predicate holds); the verifier scored the completed/timed-out state.
- **Experiment-fidelity failure (unhealthy → rerun or quarantine):** the agent
  was blocked by missing API keys, absent skill resources, hidden/mis-mounted
  assets, dependency-install failures, network/permission errors, insufficient
  compute/memory, broken rendering, path mismatches, provider outages, or
  sandbox instability.

Decision procedure: **deterministic-first** — `check_results.INFRA_ERROR_CATEGORIES`
plus `verifier_error` plus required-env presence catch the known infra
signatures and route them to quarantine. The **residual** judgment is handed to
the **codex** judge. Only the model-capability case counts as a healthy
`fail`/`normal_timeout`; experiment-fidelity failures are rerun after the
environment is fixed, or documented as quarantined infra/context failures.

## 6. Known Coverage Gaps & Residual Risk

This section states **honestly** what the current artifacts **cannot**
mechanically prove, and how each gap is mitigated. Do not let the rubric or the
review pack imply deterministic coverage for any of these.

| Gap | Why artifacts can't prove it | Gate | Mitigation |
|---|---|---|---|
| **Verifier-start timestamp** (verifier started only after agent exit) | Production rollout artifacts carry no verifier-start-vs-agent-exit timestamp. The flat fixture's `verifier_started_after_agent` boolean is best-effort and not present in production. | `V-LIFECYCLE` | Surface the flat-fixture boolean when present; otherwise **residual** note + **codex** judge asked to flag suspicious early-verifier evidence. Closed at source by the harness lifecycle (verifier as a separate process started post-exit), not by the grader. |
| **Git-scrub past base** (no future/golden commits readable) | Filesystem/git state inside the sandbox is not serialized into the rollout. | `V-ENVHARDEN` | Best-effort **env-probe** rollout when run; otherwise **residual** + **codex** judge scans the trajectory for `git log --all`/`git show`/reflog/hidden-ref reads. Prevention: the hardening checklist's git time-travel scrub. |
| **Answer-file absence** (no reference/hidden-test/scorer files in agent-readable paths) | Same — the agent-visible filesystem is not captured in artifacts. | `V-ENVHARDEN` | Best-effort env-probe; else **residual** + **codex** judge scans for reads of answer/scorer/hidden-test paths. Prevention: keep all answer artifacts outside agent-readable paths; build-time empty-patch probe (`V-EMPTYPATCH`) catches the *reachable-reward* symptom deterministically even when the file scan cannot. |
| **Reliable skill catalog on `pi-acp`** (and other non-serializing harnesses) | `pi-acp` does not serialize a startup skill catalog (`catalog_not_serialized`); `skill_count: 0` there means "not serialized", not "no skills". Some `claude-code` c50 rows also expose the `Skill` tool but no bullet catalog. | `S-WITHSKILL` | **Quarantine → codex**, never an auto-reject. The extractor's `unknown`/`catalog_not_serialized`/`manual_review_required` outputs route the slot to codex/manual review. For `no-skill` runs the deterministic **leakage-marker** scan (`S-NOSKILL`) still applies, because marker *presence* is unambiguous even when catalog *absence* is not. |
| **Monkey-patch / `__eq__` / operator-overload grader gaming** | These never surface as a file write/delete/chmod, so `V-TAMPER`'s file-op scan cannot see them (per the hardening checklist: test-reset and hidden-tests do not close these). | `V-REWARDHACK` | **Codex** LLM judge over the full trajectory, fail-closed. Kept active even on an otherwise hardened environment. |
| **Token/timing value accuracy** | Telemetry presence is observable; provider-true accounting is not. | `R-TELEMETRY` | Deterministic **presence** check only; value-accuracy is out of scope and noted as residual. |

Quarantine/residual items are surfaced in the review pack's Residual-risk
section (§7) with their gate IDs, so a maintainer can see exactly what was not
mechanically proven and why the verdict is `mergeable with quarantines` rather
than `mergeable`.

## 7. Review Checklists (per-slot, skill-loading, reward-hacking)

These are the operational checklists the human-procedure skill mechanizes. Each
item is mapped to its **enforcer**: a deterministic gate ID, the **codex**
reviewer, or a **residual** note. The Codex reviewer (`integration-tiers.md` §5.2)
applies these same checklists per rollout; the deterministic gates apply the
mechanizable subset.

### 7.1 Per-matrix-slot trajectory checklist

For **every** slot (`task_id × agent × model × skill_mode × trial × sandbox`):

| Check | Enforcer |
|---|---|
| Slot present exactly once, fresh (head SHA), not stale/duplicate/missing | `X-SLOTS` (deterministic) |
| Trajectory present + parseable end-to-end; ATIF/ADP well-formed; no secret leak | `R-ARTIFACT` (deterministic) |
| Realness: `n_tool_calls > 0` AND `total_tokens > 0` AND `reward != null` | `R-REAL` (deterministic) |
| Token usage + timing present for newly generated data | `R-TELEMETRY` (deterministic) |
| Outcome ∈ `{pass, fail, normal_timeout}` with a valid score | `R-OUTCOME` (deterministic) |
| Agent got the intended prompt and **no** verifier-only information | codex |
| Tool calls + observations form a plausible task attempt (not empty/setup-only) | codex |
| Final answer + verifier score refer to the same task workspace/trial | codex |
| `fail`/`timeout` is model-capability, not experiment-fidelity | `C-ATTRIB` (deterministic-first → codex residual) |
| Config root / sandbox cwd / trajectory paths / result paths agree | `P-PATHS` (deterministic where fields exist, else residual) |
| Source repo/path/ref/hash provenance intact | `P-PROV` (deterministic) |
| Docker↔Daytona reward + schema/telemetry parity | `P-REWARD` / `P-SCHEMA` (deterministic) |

### 7.2 Skill-loading checklist

| Check | Enforcer |
|---|---|
| `with-skill` ⇒ `task_skills_loading == 1` (every expected task skill recovered in the startup catalog) | `S-WITHSKILL` (quarantine → codex; `unknown`/`catalog_not_serialized` is **not** proof of absence) |
| `no-skill` ⇒ `task_skills_loading == 0` | `S-NOSKILL` (deterministic) |
| `no-skill` ⇒ **no** leakage markers anywhere in the trajectory: `SKILL.md`, `.codex/skills`, `.agents/skills`, `.claude/skills`, `invoke_skill`, `activate_skill`, `ToolSearch` | `S-NOSKILL` (deterministic; marker presence is a hard reject) |
| `with-skill` ⇒ native skill loading / injection evidence for **that** harness | codex (via `extract_harness_skills.py` + harness-skill-catalog SOP) |
| Skill catalog extraction recorded (source field, line index, names, count, SHA-256) | `skill_catalog_summary.json` + codex |

> **Asymmetry:** `S-NOSKILL` is deterministic because a leakage marker is
> *positive* evidence of contamination. `S-WITHSKILL` is a quarantine because an
> *absent* catalog (notably `pi-acp` → `catalog_not_serialized`) is ambiguous, so
> it routes to codex rather than auto-rejecting.

### 7.3 Reward-hacking checklist

| Check | Enforcer |
|---|---|
| Empty/no-op patch FAILS (`<1.0`) while golden patch passes (`1.0`) | `V-EMPTYPATCH` (deterministic) |
| No agent write/delete/chmod of score-defining files (tests, verifier, scorer, result/reward) | `V-TAMPER` (deterministic, file-op scan) |
| Reward consistent with visible task state — no grader gaming, monkey-patch, `__eq__`/operator overload, exit-code manipulation, test-input hardcoding | `V-REWARDHACK` (codex LLM judge, fail-closed; catches the non-file-op vectors `V-TAMPER` cannot see) |
| Verifier starts only after agent exit/timeout | `V-LIFECYCLE` (residual → codex; no lifecycle timestamps in artifacts) |
| Git scrubbed past base; no answer/hidden-test/scorer files in agent-readable paths | `V-ENVHARDEN` (residual → codex / best-effort env-probe) |
| Network off by default; egress only via a hardened allowlist | `V-NETWORK` (deterministic, STATIC config; default `no-network`, `allowlist` needs non-empty `allowed_hosts`) |

The reward-hacking pattern background is
`references/reward-hacking-patterns.md`; the prevention mapping is
`references/verifier-hardening-checklist.md`.

## 8. Success Rubric (Area / Pass / Fail)

| Area | Pass | Fail |
|---|---|---|
| **Realness** | `n_tool_calls > 0`, `total_tokens > 0`, `reward != null`; a genuine attempt is logged | empty/never-launched/transport-failed rollout; missing tool/token/timing telemetry |
| **Outcome** | exactly one of `pass` / `fail` / `normal_timeout` with a valid score | unparseable/absent status; null reward on a non-timeout; unscored timeout |
| **Artifacts** | trajectory parses end-to-end; ATIF/ADP well-formed; no secret leak | truncated/unparsable trajectory; malformed ATIF/ADP; secret/credential leak |
| **Skill loading** | `with-skill` ⇒ skills loaded; `no-skill` ⇒ none loaded and no leakage markers | with-skill catalog missing **and not** `catalog_not_serialized`; any no-skill leakage marker |
| **Verifier isolation** | verifier post-exit only; no score-file tampering; reward matches visible state | early verifier; tamper; reward-hack (incl. monkey-patch / `__eq__` / exit-code) |
| **Environment hardening** | network off (or hardened allowlist); git scrubbed; no answer files readable | `public` network without sign-off; answer/hidden-test/scorer files readable; reachable empty-patch reward |
| **Paths / provenance** | root/cwd/trajectory/result paths agree; source ref/hash truthful | path/root mismatch; provenance drift |
| **Parity** | docker↔daytona reward + schema/telemetry equal; pinned-baseline bands hold | reward/schema/telemetry divergence; baseline band exceeded |
| **Coverage** | every planned slot healthy, unique, fresh | a required slot missing / stale / duplicate / unhealthy |
| **Capability attribution** | `fail`/`timeout` is model-capability | experiment-fidelity failure (missing keys/skills/assets/deps/compute/network) |

"Same behavior" for the before/after baseline is **schema + lifecycle +
reward-BAND parity, NOT bit-identity** — *compare artifact schema and semantics,
not exact model wording*.

## 9. Merge Rule

The level at which a PR must be green (encoded as the workflow gate semantics; see
`integration-tiers.md` §1):

- **L0** green is required on **every** PR.
- **L1** green is required for PRs touching `src/`, `tests/integration/`, the
  integration workflows, or this review skill.
- **L2** green is required when a PR touches agent adapters, skill loading,
  verifier/reward, sandbox/root/path, network/dependency, or artifact schema.
- **L3** green **+ Codex verdict** is required before final review/merge for PRs
  changing rollout semantics, data validity, verifier isolation, sandbox
  behavior, or agent/task execution.
- **Docs-only:** no rollout unless the PR touches published eval evidence,
  citation metadata, or release notes.

`Final verdict = worst(deterministic, codex)`; the deterministic floor is the gate
and Codex can only make it stricter (§4.1).

## 10. Review-Pack Report Ordering

The review pack (`build_integration_review_pack.py` → `review-pack/verdict.md`) is
written in this fixed section order (matching the SKILL.md reporting order,
extended with explicit gate IDs and commands):

1. **Verdict** — `mergeable` | `mergeable with quarantines` | `not mergeable`.
2. **Blockers** — deterministic rejects and missing slots (the things that force
   `not mergeable`), each with its gate ID and the offending slot.
3. **Coverage** — the slot table: planned vs `healthy/missing/duplicate/stale/unhealthy`,
   grouped by task / skill-mode / sandbox / agent (gate `X-SLOTS`).
4. **Evidence** — run roots, refs, the gate IDs that fired, and the **exact**
   validation commands (`rubric_checks.py …`, `build_integration_review_pack.py …`)
   so a reviewer can reproduce every result.
5. **Residual risk** — the quarantine/codex/residual gates left unresolved
   (`S-WITHSKILL` unknowns, `V-LIFECYCLE`, `V-ENVHARDEN`, `V-REWARDHACK` notes,
   `C-ATTRIB` residuals), each tagged with its gate ID and §6 mitigation.
6. **Required reruns** — for every `fail`/`timeout` slot, the model-capability vs
   experiment-fidelity decision (§5) and the evidence behind it, plus any slot
   that must be rerun after an environment fix.
