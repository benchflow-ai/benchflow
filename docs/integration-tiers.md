# Integration Test Levels

The level ladder for Benchflow integration testing: which checks fire on which
trigger, how a change's scope maps to a required task set and `bench eval run`
axes, the deterministic **review-pack** verdict, the **Codex** equivalence
reviewer, the before/after baseline model, and the security model.

This is the operational companion to the success rubric in
[`integration-review-rubric.md`](./integration-review-rubric.md) — read that for
what each gate (`RUBRIC_GATES`) means, the per-slot / skill-loading /
reward-hacking checklists, the success-rubric table, and what a verdict means.

The locked architectural decisions behind this system live in
[`adr/`](./adr/): **ADR-0001** ([lane execution home](./adr/0001-integration-lane-execution-home.md))
— lanes execute as matrix cells, not pytest markers; **ADR-0002**
([verifier-tamper](./adr/0002-verifier-tamper-producer-side-hash.md)) — cheap
fail-closed `V-TAMPER` now, producer-side hash deferred; **ADR-0003**
([network_mode conformance](./adr/0003-network-mode-conformance-lane.md)) —
static `V-NETWORK` now, runtime egress conformance lane deferred (blocked on
`benchflow.sandbox._egress`). Vocabulary shared across the docs and the
`benchflow-experiment-review` skill is in the [Glossary](#glossary) (§12).

> **Terminology rename.** The deterministic verdict is user-facing as
> **`mergeable`** / **`mergeable with quarantines`** / **`not mergeable`**. These
> are renames of the internal grader labels `publishable` /
> `publishable-with-quarantines` / `not-publishable` emitted by
> `.github/scripts/build_integration_review_pack.py`.

## 1. The Four Levels (L0–L3)

The ladder runs from a cheap per-commit check to a human-gated final review.
**Every level workflow ALWAYS triggers** (no `on: paths` filter, Q4). A cheap
first job `detect-scope` (no secrets, ~seconds) computes from the diff whether
real work is needed; if not, it reports SUCCESS as a green no-op so the check can
be **unconditionally required** in branch protection.

| Level | Trigger / fires when | What runs | Cost | Merge-required for |
|---|---|---|---|---|
| **L0** | every PR / push | unit + static (`pytest` targeted, `ty`, `ruff`) via `astral-sh/setup-uv`. No sandbox, no agents, no keys. | seconds–minutes, CPU | **every PR** |
| **L1** | every PR; `detect-scope` decides real vs no-op | smallest representative lane: the planner emits a `citation` / `low-smoke` matrix; rollouts via `tests/integration/scenarios.run_eval`; graded by `rubric_checks.py`; review pack built. | low (1–few docker cells, no fan-out) | PRs touching `src/`, `tests/integration/`, the integration workflows, or the review skill |
| **L2** | every PR; `detect-scope` escalates on scope triggers | scope-gated set (low-3 / medium-3 / high-3 / nine + axes) per the Default-config-rules; docker **and** daytona where required; `with-skill`/`no-skill`; the **network lane**; the **cheat** lane. **All keys present** (provider incl. DEEPSEEK, DAYTONA, reviewer). | high (multi agent/task/sandbox, daytona fan-out, judge) | PRs touching agent adapters, skill loading, verifier/reward, sandbox/root/path, network/dependency, or artifact schema |
| **L3** | **manual** `workflow_dispatch`; final review before merge | the scope-selected matrix on the existing `pypi-internal-preview` env, band-compared against the **HuggingFace leaderboard `main`** deepseek-v4-flash run (golden truth) and the latest benchflow main, **plus** the **Codex** before/after-equivalence reviewer. Codex key **required** (dead/absent ⇒ `not mergeable`, fail-closed). | highest | PRs changing rollout semantics, data validity, verifier isolation, sandbox behavior, or agent/task execution |

**L0/L1/L2 are REQUIRED in branch protection. L3 is a manual `workflow_dispatch`
gate** whose golden truth is the HuggingFace leaderboard `main` runs compared
against the PR and the latest benchflow main. (A human-approval gate can be added
later by pointing the L3 jobs at a protected environment.)

**Docs-only:** `L0` only, **no rollout** — *unless* the PR touches published eval
evidence, citation/schema docs, or release notes (then the `citation` set fires).

## 2. The Seven Task Sets (authoritative)

The planner (`.github/scripts/integration_matrix.py`) selects one of these named
sets from the diff. Tasks live under `docs/examples/task-md/real-skillsbench/`.

| Set | Tasks |
|---|---|
| `citation` | `citation-check` |
| `low-smoke` | `jax-computing-basics` |
| `low-3` | `jax-computing-basics`, `python-scala-translation`, `jpg-ocr-stat` |
| `medium-3` | `grid-dispatch-operator`, `threejs-to-obj`, `data-to-d3` |
| `high-3` | `lake-warming-attribution`, `weighted-gdp-calc`, `shock-analysis-supply` |
| `nine` | `low-3` + `medium-3` + `high-3` |
| `expanded` | `nine` + `citation-check` + affected-task(s) + parity cases |

Nine agents: `claude-agent-acp`, `pi-acp`, `openclaw`, `codex-acp`, `gemini`,
`opencode`, `harvey-lab-harness`, `openhands`, `mimo`. The **baseline agent pair**
is `openhands` + `deepseek/deepseek-v4-flash`; the canonical "one high task" is
`weighted-gdp-calc`; the citation vehicle is `citation-check`.

## 3. Default-config-rules (PR scope → required set → required axes)

Authoritative mapping (mirrors `.github/integration/scope_map.yml`). The planner
derives the affected agent from a changed `src/benchflow/agents/<name>` path.

| PR scope | Required set | Required axes | Level |
|---|---|---|---|
| docs-only non-runtime | L0 only, **no rollout** | — | L0 |
| citation / evidence / schema docs | `citation` | Docker, no-skill, usage=required | L1 |
| `src/benchflow/eval*`, rollout lifecycle, artifact schema | `nine` | Docker, no-skill, usage=required, judge | L2 |
| `src/benchflow/agents`, ACP adapters, codex/openhands/pi/gemini | `low-3` + one high (`weighted-gdp-calc`) | affected agent + baseline agent (`openhands`+`deepseek`); no-skill AND with-skill when relevant | L2 |
| skill loading, `.agents/skills`, skill injection | `low-3` + `medium-3` | no-skill AND with-skill; run skill-catalog extraction | L2 |
| Docker / Daytona / sandbox / root / path | `low-3` + `medium-3` | Docker + Daytona parity; reaper dry-run | L2 |
| verifier, rewards, judge, anti-hack hardening | `citation` + `weighted-gdp-calc` + `shock-analysis-supply` | judge fail-closed, reward-hacking scan, verifier isolation | L3 |
| network / package install (Q3 triggers) | `jax-computing-basics` + `data-to-d3` + one high | default network-off + the `citation-check` allowlist variant | L2 |
| release-critical refactor | `expanded` | all affected axes, concurrency reduced | L3 |

### 3.1 The Q3 network lane (scope-gated)

There is **NO `bench eval run --network` flag** and **`network_mode` is NOT
serialized into artifacts** — it is a per-task config field only. So network is a
**scope-gated lane**, triggered by changes under: `src/benchflow/providers/**`,
`usage_tracking.py` (llm-proxy), network-installing agents / ACP shims,
`src/benchflow/sandbox/lockdown.py` + compose network files, and
`src/benchflow/task/runtime_capabilities.py`.

- **Default vehicle:** `citation-check` (network-off) **plus** a NEW minimal
  **allowlist VARIANT**:
  [`docs/examples/task-md/real-skillsbench/citation-check-network/`](./examples/task-md/real-skillsbench/citation-check-network/)
  — `network_mode: allowlist` with `allowed_hosts: [eutils.ncbi.nlm.nih.gov,
  scholar.google.com, doi.org, api.crossref.org]` (the real egress of the
  citation-management skill).
- **The cell carries `network_mode` as EXPECTED** (derived from the task config),
  **NOT passed to bench**. The planner emits one allowlist cell
  (`<task>-docker-no-skill-<agent>-allowlist`); the grader's static **`V-NETWORK`**
  check (`rubric_checks.network_hardening`) asserts the hardened posture:
  default `no-network`; `allowlist` passes **only** with a non-empty
  `allowed_hosts`; bare `public` is flagged (hard `fail` on a verifier/sandbox PR).
- Validity is enforced by `_validate_network_policy_fields`
  (`src/benchflow/task/config.py`): `allowlist` requires a non-empty
  `allowed_hosts`.

## 4. Matrix Cell Schema

The planner emits a matrix of cells (consumed by the workflow `fromJson` and the
grader). One cell:

```json
{
  "id": "<task>-<sandbox>-<skillmode>[-<agent>]",
  "level": "light|scope|final",
  "task": "<task>",
  "agent": "<agent>",
  "model": "<model>",
  "judge_model": "<judge-model>",
  "sandbox": "docker|daytona",
  "skill_mode": "no-skill|with-skill|self-gen",
  "network_mode": "default-off|allowlist",
  "timeout_minutes": 90,
  "agent_idle_timeout": 240,
  "audit_skills": false,
  "expect_reward": "==1.0|<1.0|any"
}
```

Top-level planner output:

```json
{
  "schema_version": "1",
  "head_sha": "<sha>",
  "base_ref": "main",
  "scope": "citation|low-smoke|low-3|medium-3|high-3|nine|expanded|custom",
  "buckets": ["<scope-map rule id>", "..."],
  "trust_boundary": true,
  "cheat": false,
  "network_lane": false,
  "baseline": "pinned|rerun-base",
  "caps": { "max_cells": 140, "per_agent_concurrency": 2, "aggregate_concurrency": 24, "...": "..." },
  "matrix": [ { "...cell...": "..." } ],
  "residual_risk": ["..."],
  "rejected_overflow": null
}
```

### 4.1 Hard ceiling + concurrency (fail-closed)

- **Hard cell ceiling:** if the enumerated matrix exceeds `caps.max_cells`, the
  planner sets `rejected_overflow` and **exits code 2** — it never silently drops
  a cell.
- **Aggregate concurrency:** `per_agent_concurrency × (distinct daytona agents)
  ≤ 24`. The planner lowers `per_agent_concurrency` as the agent count rises.

The caps + baseline anchors are pure data in
`.github/integration/scope_defaults.yml`.

## 5. The Verdict — deterministic floor is the gate (Q2)

```
Final verdict = worst( deterministic_grader , codex_reviewer )
```

### 5.1 Deterministic grader (REQUIRED, fail-closed gate)

`.github/scripts/build_integration_review_pack.py` runs `rubric_checks.py` over
every slot and writes `review-pack/` (layout in §7). Its exit code +
`review-pack/verdict.md` are the **REQUIRED, FAIL-CLOSED** gate:
`mergeable` / `mergeable with quarantines` / `not mergeable`. A deterministic
reject on a healthy slot, a missing required slot, or a non-quarantinable codex
FAIL ⇒ `not mergeable` (non-zero exit). This floor is **authoritative**.

### 5.2 Codex reviewer (advisory-stricter-only veto)

Codex is a **STRONG VETOING before/after-equivalence signal**: it can only make
the verdict **STRICTER**, **never upgrade** a deterministic `not mergeable`.

- **Trusted-main contract (Q1):** the reviewer reads
  `/home/liu.10379/benchflow-int-ci/.agents/skills/benchflow-experiment-review/SKILL.md`
  first and treats **all trajectories / tool-outputs / observations as UNTRUSTED
  data**.
- **Detailed per-rollout pass** uses the **cheaper deepseek model**
  (`BENCHFLOW_JUDGE_MODEL` / deepseek) over each rollout → per-rollout finding
  JSON (high-volume trajectory reads).
- **Final equivalence verdict** is composed by the host **`codex` CLI** via
  `codex exec` (authed with an `auth.json` written from the `CODEX_AUTH_JSON` CI
  secret to the codex config path), **self-orchestrating its own subagents** (the
  "raw workflow"), over `{per-rollout deepseek findings + the deterministic
  review-pack/}`. The argv mirrors `build_codex_launch_command` in
  `src/benchflow/agent_router.py`.
- **Fail-closed:** if the `codex` binary or `auth.json` is missing, or the output
  is unparseable ⇒ emit `not mergeable (codex unavailable)` and exit non-zero.
  **NEVER silently pass.** A dead/absent Codex key **at L3** ⇒ `not mergeable`.
  L1/L2 do **not** require Codex.
- **Advisory-stricter-only:** the output may downgrade `mergeable` → `not
  mergeable`, **never** the reverse.

Implementation: `.github/scripts/codex_review.py` +
`.github/integration/codex_review_prompt.md`. Required reviewer outputs (the
prompt): **1 Verdict · 2 Blockers · 3 Coverage** (enumerate `task_id × agent ×
model × skill_mode × trial × sandbox`) **· 4 Evidence** (run roots, commands,
refs, files) **· 5 Residual risk · 6 Required reruns**. It applies the
experiment-review rules: no aggregate-only; prove `with-skill` loaded; scan
`no-skill` leakage; infra failures are unhealthy; check verifier isolation,
reward hacking, root/path, network policy.

## 6. Before/After Baseline (Q2b)

"Same behavior" == **schema + lifecycle + reward-BAND parity**, **NOT
bit-identity** (per the skill: *"Compare artifact schema and semantics, not exact
model wording"*).

- **Default = `pinned`:** pinned-baseline reward-band parity via
  `tests/integration/check_skillsbench_harbor_parity.py`
  (`main(argv)`; flags `--benchflow-root --harbor-baseline-root --task
  --max-outcome-rate-delta --max-mean-reward-delta --max-task-reward-delta`),
  wrapped by `rubric_checks.parity_baseline_band`. The bands feed `P-REWARD` /
  `P-SCHEMA` in the review pack's `parity_summary.json`.
- **`rerun-base` only for `scope=expanded`:** a same-SHA base re-run, used only on
  the heaviest release-critical lane.

## 7. Review-Pack Layout

The grader writes `review-pack/`:

| file | content |
|---|---|
| `manifest.json` | PR, SHA, scope, matrix, source refs |
| `matrix_expected.json` | planned cells |
| `matrix_observed.json` | per slot: `healthy/missing/stale/duplicate/unhealthy` |
| `metrics.json` | per cell: task, reward, tokens, timing, `n_tool_calls` |
| `agent_judge_summary.json` | one row per rollout |
| `skill_catalog_summary.json` | with/no-skill `task_skills_loading` |
| `parity_summary.json` | docker/daytona within-PR + pinned-baseline band deltas |
| `hardening_summary.md` | verifier / network (`V-NETWORK`) / root / path |
| `red_flags.md` | reward-hacking / leakage / infra |
| `verdict.md` | `mergeable` \| `mergeable with quarantines` \| `not mergeable`, sections in skill order |
| `rollouts/` | trimmed / linked rollouts |

`verdict.md` section order (skill order): **Verdict · Blockers · Coverage ·
Evidence · Residual risk · Required reruns**.

## 8. Real `bench eval run` Axes

The real flags (verified): `--agent --model --sandbox (docker|daytona)
--concurrency --jobs-dir --include (repeatable) --skill-mode
(no-skill|with-skill|self-gen) --skills-dir --agent-idle-timeout
--usage-tracking (auto|required|off) --tasks-dir
--source-repo/--source-path/--source-ref --agent-env`. **There is no `--network`
flag.** Scoped/skill/sandbox cells go through
`tests/integration/scenarios.run_eval(jobs_dir, agent, sandbox, include=(),
model, concurrency, extra_args=[...], ...)`. `tests/integration/run.sh` is the
**FULL-9-on-daytona lane only** (positional agents +
`BENCHFLOW_INTEGRATION_JOBS_ROOT`). `select_integration_provider.py` exports
`BENCHFLOW_INTEGRATION_AGENT/_MODEL/_JUDGE_MODEL`.

## 9. Security Model (Q1)

The trust boundary: **only `src/benchflow` (the code under test) comes from the
PR head.** The planner, grader, harness, and review skill load from **`origin/main`
HEAD** (sparse-checkout ref `main`) — **NEVER** the PR head / base-commit. The
PR-head `bench` necessarily runs on the host (it is the orchestrator), but the
**VERDICT is computed only by trusted-main code**.

- **Plain `pull_request`** (not `pull_request_target`) on every secret-bearing
  job. `detect-scope` + planner + grader are sparse-checked-out from `origin/main`.
- **Keys in L2 (residual note).** Per the design, **L2 carries all keys** (provider
  incl. DEEPSEEK, DAYTONA, reviewer). This is a deliberate, documented residual:
  the L2 lane is secret-bearing on every triggering PR. L3 runs under the existing
  `pypi-internal-preview` environment (provider + DAYTONA secrets); **no separate
  protected env is created**, and the review job runs the **trusted-main grader**
  over the artifacts. The L3 golden truth is the HuggingFace leaderboard `main`
  deepseek-v4-flash baseline vs the PR vs the latest benchflow main.
- **Codex auth** is written from the `CODEX_AUTH_JSON` secret to the codex config
  path (`auth.json`) and invoked via `codex exec`, mirroring
  `build_codex_launch_command` — **fail-closed** if the binary / auth is absent.
- **`issue_comment` bodies** (if any path uses them) are read **via env only**,
  never inlined into `run:`.
- **SHA-pin actions** exactly as
  `/home/liu.10379/benchflow-int-ci/.github/workflows/test.yml`:
  `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683`,
  `astral-sh/setup-uv@caf0cab7a618c569241d31dcd442f54681755d39`,
  `actions/upload-artifact@65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08`. For
  `download-artifact` / `github-script` pin a real v4/v7 SHA and comment it as new
  surface.

## 10. Admin Setup

One-time setup before the heavy lanes are live:

1. **Secrets** live in the existing `pypi-internal-preview` environment
   (`DAYTONA_API_KEY`, `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`, provider keys —
   already present). Add `CODEX_AUTH_JSON` there for the L3 Codex reviewer
   (written to `auth.json` at runtime). **No protected environment is created** —
   L3 is a manual `workflow_dispatch` whose golden truth is the HuggingFace
   leaderboard `main` runs plus the latest benchflow main.
2. **Branch protection:** make **L0, L1, L2 required** status checks. L3 is run
   on demand before merge (manual dispatch); promote it to a protected env with
   required reviewers later if you want a hard human gate.

## 11. Deferred Follow-Ups

Documented, **NOT built** (see the ADRs in [`adr/`](./adr/) for the locked
decisions):

- **`--network-mode` CLI passthrough** on `bench eval run`. Today network is a
  per-task config field with no flag; the lane is a STATIC declaration check
  (ADR-0003).
- **`network_mode` result.json serialization** — serialize the requested
  `network_mode` (+ `allowed_hosts`) into the rollout artifact so the network
  lane can move from a STATIC config assertion to an **observed** egress-policy
  check, and so `check_results` can reconcile recorded vs requested posture
  (mirroring `agent_idle_timeout`). **Coordinate this as ONE rollout-contract
  schema bump with the deferred `verifier_files_mutated` field below** — both are
  additive defaulting fields on `result.json` + `GateResult` (ADR-0002, ADR-0003).
- **Runtime egress CONFORMANCE prober** — the lane that *observes* live egress via
  the egress sidecar (`no-network` blocks all egress; `allowlist` permits only the
  listed hosts plus the resolved model-provider host; a disallowed host is denied).
  **BLOCKED on `benchflow.sandbox._egress.build_egress_override` not existing in
  main**; #799's runnable prober (`net/live_lane_test.py`) is **deliberately not
  ported**. Recommended trigger: attach to the **existing verifier-rewards-judge
  scope rule** (§3), NOT always-on; docker↔daytona parity stays nightly-only
  (ADR-0003).
- **Producer-side verifier-tamper hash + `verifier_files_mutated` field** — the
  producer (sandbox/verifier) records a before/after hash of the score-defining
  file set and writes a definitive `verifier_files_mutated: bool` into the rollout
  contract, demoting the trajectory regex to advisory. #802 already ships the
  *cheap* fail-closed half (the regex signal feeds `realness_issues`, Task A1);
  this is the deferred producer-side authority (ADR-0002). Bundle its contract
  field with the `network_mode` serialization above as one schema bump.
- **Unbuilt REFINEMENT-PLAN slices (ported as backlog from #799):**
  - **Power-aware parity** — replace the fixed reward-band delta in
    `tests/integration/check_skillsbench_harbor_parity.py` with a
    sample-size-aware verdict: require a `min_trials` floor and compare outcome
    rate via **Wilson confidence-interval overlap** (too-few-trials →
    *inconclusive*, not PASS) plus a documented per-task delta band. Feeds
    `P-REWARD` / `P-SCHEMA`.
  - **Fixture-factory harness CLI + network-leak fixture** — add a thin argparse
    CLI to `tests/integration/deepagents_harness.py`
    (`--instruction --verify-cmd --extra-system --rollout-dir`) so judge-hardening
    rounds are reproducible, plus a **network-leak fixture** (run config declares
    `allowlist`/`no-network` but the trajectory shows egress to a non-allowlisted
    host) wired into the experiment-review skill's evals so a `network_mode`
    regression is caught by the SOP's own evals.

## 12. Glossary

Durable shared vocabulary for the integration system — used by these docs,
`tests/integration/`, the L0–L3 workflows, and the `benchflow-experiment-review`
skill. Network-posture terms use the **authoritative** `NetworkMode` enum from
`src/benchflow/task/config.py`: **`no-network` / `allowlist` / `public`**.

- **Integration test** — an end-to-end run that exercises the *real* eval path
  (adapters, sandboxes, agents, verifiers) via `bench eval run`
  (`scenarios.run_eval`), asserting the produced artifacts are trustworthy. NOT a
  unit test; NOT the agent's own task pass/fail.

- **Rollout contract** — the artifact set every producer emits and every checker
  consumes: `result.json` + the run config + the trajectory (ATIF/ADP). The
  shared interface that keeps producers, checkers, and the judge decoupled.

- **Producer** — something that emits the rollout contract. In #802 the matrix
  cell driven through `scenarios.run_eval` (real agents on real tasks) is the
  primary producer; `tests/integration/deepagents_harness.py` is a *fixture
  factory* — a steerable agent that manufactures genuine and reward-hacking
  rollouts to harden the judge.

- **Cell** — one unit of the planner's matrix (`integration_matrix.py`): a
  `task × agent × model × sandbox × skill_mode × network_mode` slot (schema in
  §4), run via `scenarios.run_eval` and graded by `rubric_checks.py`. The #802
  execution home for a lane (ADR-0001).

- **Task-set** — one of the seven named sets (§2) the planner selects from the
  diff; the #802 analogue of a #799 "lane axis value".

- **Realness gate** — the mechanical, judge-independent check that a rollout is a
  genuine measurement (`n_tool_calls > 0`, tokens `> 0`, non-null reward, no
  infra/verifier error, **and** no mechanically-flagged verifier tamper). It must
  hold even when the LLM judge passes. Implemented by
  `agent_judge.realness_issues`; surfaced as `R-REAL`.

- **Verifier tamper** — an agent mutating the score-defining (verifier) files to
  fake a reward. Detected today by the trajectory regex
  (`agent_judge._scan_verifier_tamper`), whose output (`flagged_actions`) is
  **fail-closed** into the realness gate (Task A1). The deferred producer-side
  before/after **hash** (`verifier_files_mutated`) is the future authoritative
  signal (ADR-0002). Surfaced as `V-TAMPER`.

- **network_mode — identity vs conformance** (two distinct guarantees over the
  `no-network` / `allowlist` / `public` posture):
  - **Identity / declaration check** — the *declared* (and, when serialized, the
    *recorded*) posture is hardened and matches what was requested (cheap; catches
    "wrong posture declared/requested"). #802 ships the **static** declaration
    half today via `V-NETWORK` (`rubric_checks.network_hardening`); the recorded
    reconciliation is deferred until `network_mode` is serialized.
  - **Conformance check** — the run *actually* observed the right egress: an
    allowlisted host is reachable AND a non-allowlisted host is **blocked** under
    the enforced mode (the real security guarantee). **Deferred** in #802 as
    `V-NETWORK-CONFORM`, blocked on `benchflow.sandbox._egress` (ADR-0003).

- **Evidence check** — a standalone checker that turns rollout artifacts into a
  release gate (`check_results`, `check_adapter_evidence`,
  `check_hosted_env_evidence`, `check_trace_to_task_evidence`,
  `check_skillsbench_harbor_parity`).

- **CI tiering** — the L0–L3 ladder (§1): the per-PR gate is cheap, docker-only,
  hard-fail; nightly/manual runs the full agent×task×sandbox matrix (advisory,
  promotable to release-blocker on tags).

---

See [`integration-review-rubric.md`](./integration-review-rubric.md) for the
`RUBRIC_GATES` table (incl. `V-NETWORK`), the per-matrix-slot / skill-loading /
reward-hacking checklists, the success-rubric table, the merge rule, and the
report ordering.
